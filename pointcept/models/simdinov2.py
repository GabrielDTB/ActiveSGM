import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.losses.ibot_patch_loss import iBOTPatchLoss
from pointcept.models.losses.dino_clstoken_loss import DINOLoss
from pointcept.models.losses.sim_dino_clstoken_loss import MCRLoss
from pointcept.models.losses.sim_ibot_patch_loss import CosinePatchLoss

from pointcept.models.utils.structure import Point, Point
from .builder import MODELS, build_model

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from pathlib import Path
from torch.nn.init import trunc_normal_
from pointcept.models.utils import offset2batch
from torch_geometric.nn.pool import voxel_grid


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        normalize=True,
        remove_last_layer=False
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        if not remove_last_layer:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        self.normalize = normalize
        self.remove_last_layer = remove_last_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-4)
        if not self.remove_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim,eps=1e-3))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


@MODELS.register_module()
class DefaultContrastiverSimDinoV2(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        backbone=None,
        local_crop_num=3,
        do_ema=True,
        do_ibot=True,
        enable_mae_loss=False,
        mask_ratio_min_max=(0.1, 0.5),
        mask_sample_probability=0.5,
        dino_weight=1.0,
        ibot_weight=1.0,
        mae_weight=1.0,
        mask_grid_size = 0.2, # 0.1 for indoor scene
        mask_type = 'patch', # 'splats' or 'patch'
    ):
        super().__init__()
        # sanity
        assert mask_type in ['patch', 'splats'], f"mask_type should be 'patch' or 'splats', but got {mask_type}"
        # following simdino 
        # by default we set do_ema and do_ibot to True for student-teacher model
        # enable_mae_loss is False but is optional
        self.do_ema = do_ema
        self.do_ibot = do_ibot
        self.enable_mae_loss = enable_mae_loss

        self.dino_weight = dino_weight
        self.ibot_weight = ibot_weight
        self.mae_weight = mae_weight
        self.num_layers = len(backbone['enc_depths'])
        self.local_crop_num = local_crop_num

        # self.enable_local_ibot = enable_local_ibot
        self.mask_ratio_min_max = mask_ratio_min_max
        self.mask_sample_probability = mask_sample_probability
        self.mask_grid_size = mask_grid_size
        self.mask_type = mask_type

        if do_ema:
            print("using ema, with teacher model and student model, supervise the encoder feature")
        
        if do_ibot:
            print("using ibot loss, with student model, supervise the decoder feature")
            assert do_ema, "do_ibot should be True, otherwise we only have student model"

        if enable_mae_loss:
            print("using mae loss, with student model, supervise the decoder feature")


        # start initialize
        backbone.do_mask = True # no matter 
        if do_ema:
            self.backbone_teacher = build_model(backbone)
            # set teacher model requires_grad to False
            for param in self.backbone_teacher.parameters():
                param.requires_grad = False
            # otherwise we only have student model
        
        self.backbone_student = build_model(backbone)
        
        if self.enable_mae_loss:
            self.mae_weight = mae_weight
            # assume non-normal use
            self.using_coord = True if backbone['in_channels'] == 14 else False
            self.mae_head = nn.Sequential(
                nn.Linear(backbone['dec_channels'][0], 32),
                nn.LayerNorm(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 11), # backbone['in_channels'] = 14 (with coord), 11 (without coord) as input
            )
            # initialize
            for m in self.mae_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


        # expansier
        # borrow from simdino
        self.dino_head = DINOHead(
            in_dim=backbone['enc_channels'][-1],
            out_dim = 256,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
            normalize=True,
            remove_last_layer=True
            )

        self.dino_loss = MCRLoss(
                    out_dim=256, 
                    expa_type = 1,
                    reduce_cov=0,
                    eps=0.05,
                    eps_end=-1,
                    coeff=0.1) 

        # we use seperate head here
        self.ibot_head = DINOHead(
            in_dim=backbone['dec_channels'][0],
            out_dim = 32,
            hidden_dim=256,
            bottleneck_dim=32,
            nlayers=3,
            normalize=True,
            remove_last_layer=True    
            )
        
        self.ibot_patch_loss = CosinePatchLoss(patch_out_dim=32)


    def update_teacher(self, m=0.999):

        student_param_list = []
        teacher_param_list = []
        # to confirm the teacher is updated
        with torch.no_grad():
            for name, param in self.backbone_student.named_parameters():
                if not param.requires_grad:
                    continue
                if 'mask_token' in name:
                    continue
                teacher_param = self.backbone_teacher.state_dict()[name]
                student_param_list.append(param)
                teacher_param_list.append(teacher_param)
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)


    def mask_generator(self, offset, view_origin_coord=None): # also inclue patch
        with torch.no_grad():
            B = len(offset) # totol samples
            offset_pad = nn.functional.pad(offset, (1, 0))
            n_samples_masked = int(np.ceil(B * self.mask_sample_probability))
            # probs = torch.linspace(*self.mask_ratio_min_max, n_samples_masked + 1)  # n_samples_masked have 0.1 - 0.5 probs
            upperbound = 0
            masks_list = []
            masks_weight_list = []
            # random sample indiceds 0 - to B with n_samples_masked
            mask_ge = torch.zeros(B, dtype=torch.long)
            mask_ge[:n_samples_masked] = 1
            mask_ge = mask_ge[torch.randperm(B)]

            if self.mask_type == 'splats':
                # 
                for i in range(0,B):
                    if mask_ge[i]:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask_rate = np.random.uniform(self.mask_ratio_min_max[0], self.mask_ratio_min_max[1])
                        masked_sample_num = token_num*mask_rate
                        # prighynt("masked_sample_num", masked_sample_num, "token_num", token_num)
                        mask = torch.zeros(token_num, dtype=torch.bool).to(offset.device)
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(offset.device)
                        mask[:int(masked_sample_num)] = 1
                        mask = mask[torch.randperm(token_num)]
                        upperbound += 1
                        masks_list.append(mask)
                        mask_weight = mask_weight* ( 1 / (masked_sample_num))
                        masks_weight_list.append(mask_weight)
                    else:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask = torch.zeros(token_num, dtype=torch.bool).to(offset.device)
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(offset.device)
                        masks_list.append(mask)
                        masks_weight_list.append(mask_weight)
                        
            elif self.mask_type == 'patch':
                # union origin coord
                for i in range(0,B):
                    if mask_ge[i]:
                        batch_i_coord = view_origin_coord[offset_pad[i]:offset_pad[i+1]] 
                        mask_patch_coord = batch_i_coord / (self.mask_grid_size)
                        mask_patch_grid_coord = torch.floor(mask_patch_coord).int()
                        min_coord = mask_patch_grid_coord.min(dim=0)[0]
                        mask_patch_grid_coord -= min_coord

                        mask_patch_cluster = voxel_grid(
                            pos=mask_patch_grid_coord, size=1, start=0,
                            )
                        unique, cluster, counts = torch.unique(
                            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
                        )
                        patch_num = unique.shape[0]
                        patch_max_point = counts.max().item()
                        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
                        patch2point_mask = torch.lt(
                            torch.arange(patch_max_point).cuda().unsqueeze(0), counts.unsqueeze(-1)
                        )
                        sorted_cluster_value, sorted_cluster_indices = torch.sort(cluster)
                        patch2point_map[patch2point_mask] = sorted_cluster_indices


                        patch_mask = torch.zeros(patch_num, dtype=torch.bool)
                        rand_perm = torch.randperm(patch_num)
                        mask_rate = np.random.uniform(self.mask_ratio_min_max[0], self.mask_ratio_min_max[1])
                        mask_patch_num = int(patch_num * mask_rate) 
                        # print("mask rate", mask_rate)
                        # print("mask_patch_num", mask_patch_num, "patch_num", patch_num)
                        patch_mask[rand_perm[0:mask_patch_num]] = True
                        point_mask = torch.zeros(
                            mask_patch_coord.shape[0]
                        ).bool()
                        point_mask[
                            patch2point_map[patch_mask][patch2point_mask[patch_mask]]
                        ] = True  

                        # print("point_mask", point_mask.sum(), point_mask.shape)

                        masks_list.append(point_mask.to(offset.device))
                        mask_weight = torch.ones(point_mask.shape[0], dtype=torch.float16).to(offset.device)
                        mask_weight = mask_weight * (1 / mask_patch_num)
                        masks_weight_list.append(mask_weight) 
                    else:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask = torch.zeros(token_num, dtype=torch.bool).to(offset.device)
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(offset.device)
                        masks_list.append(mask)
                        masks_weight_list.append(mask_weight)
                        
            masks_list = torch.cat(masks_list, dim=0)
            masks_weight_list = torch.cat(masks_weight_list, dim=0)
            masks_weight_list = masks_weight_list[masks_list] 
        return masks_list, masks_weight_list


    def calculate_sim_dino_loss_pool(self, teacher_dino_feat_global_pool, student_dino_feat_global_pool, student_dino_feat_local_pool_list):
        # teacher_dino_feat_global_pool: Batch x Crop x Feat_dim
        # student_dino_feat_global_pool: Batch x Crop x Feat_dim
        # student_dino_feat_local_pool_list [Batch x Crop x Feat_dim]
        # L2 normalize teacher_dino_feat_global_pool: 

        teacher_dino_feat_global_pool = self.dino_head(teacher_dino_feat_global_pool) # normalize in side
        teacher_dino_feat_global_pool = teacher_dino_feat_global_pool.permute(1, 0, 2) # make the global crop number at begining
        teacher_dino_feat_global_pool = teacher_dino_feat_global_pool.reshape(-1, teacher_dino_feat_global_pool.shape[-1]) # flatten the global crop number

        student_dino_feat_global_pool = self.dino_head(student_dino_feat_global_pool) # normalize in side
        student_dino_feat_global_pool = student_dino_feat_global_pool.permute(1, 0, 2) # make the global crop number at begining
        student_dino_feat_global_pool = student_dino_feat_global_pool.reshape(-1, student_dino_feat_global_pool.shape[-1]) # flatten the global crop number

        z_student_local = torch.stack(student_dino_feat_local_pool_list,dim=1) # we do not need z_teacher_local here
        z_student_local = self.dino_head(z_student_local)
        z_student_local = z_student_local.permute(1, 0, 2) # make the global crop number at begining
        z_student_local = z_student_local.reshape(-1, z_student_local.shape[-1]) # flatten the global crop number

        dino_crops_loss, dino_loss_dict = self.dino_loss(
            student_dino_feat_global_pool.chunk(2) + z_student_local.chunk(self.local_crop_num),
            teacher_dino_feat_global_pool.detach().chunk(2), no_diag=True
        )
        dino_loss_dict = {"dino_mcr_"+k: v for k, v in dino_loss_dict.items()}

        loss_dict = {}
        loss_dict.update(dino_loss_dict)
        loss_dict['sim_dino_crops_loss'] = dino_crops_loss
        return loss_dict


    def calculate_sim_dino_loss_patch(self, teacher_patch_feat_global_0_mask,  teacher_patch_feat_global_1_mask, student_patch_feat_global_0_mask,  student_patch_feat_global_1_mask, teacher_temp, global_mask_weight_0=None, global_mask_weight_1=None, view_num=1):   
        # teacher_patch_feat_global_0_mask: Masked_token_num  x Feat_dim
        # teacher_patch_feat_global_1_mask: Masked_token_num' x Feat_dim
        # student_patch_feat_global_0_mask: Masked_token_num  x Feat_dim
        # student_patch_feat_global_1_mask: Masked_token_num' x Feat_dim
        # global_mask_weight_0: Masked_token_num
        # global_mask_weight_1: Masked_token_num'

        # local if needed

        teacher_patch_feat_global_mask = torch.cat([teacher_patch_feat_global_0_mask, teacher_patch_feat_global_1_mask], dim=0) # Masked_token_num x Feat_dim
        student_patch_feat_global_mask = torch.cat([student_patch_feat_global_0_mask, student_patch_feat_global_1_mask], dim=0) # Masked_token_num x Feat_dim
        global_mask = torch.cat([global_mask_weight_0, global_mask_weight_1], dim=0) # Masked_token_num

        teacher_patch_feat_global_mask = self.ibot_head(teacher_patch_feat_global_mask) # Masked_token_num x prototype_dim
        student_patch_feat_global_mask = self.ibot_head(student_patch_feat_global_mask) # Masked_token_num x prototype_dim

 
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
                student_patch_feat_global_mask,
                teacher_patch_feat_global_mask.detach(),
                masks_weight=global_mask,
                view_nums=view_num
            )
        ibot_patch_loss, ibot_loss_dict = ibot_patch_loss
        ibot_loss_dict = {"ibot_"+k: v for k, v in ibot_loss_dict.items()}
        loss_dict = {}
        loss_dict.update(ibot_loss_dict)
        loss_dict['sim_ibot_patch_loss'] = ibot_patch_loss
        return loss_dict


    def forward(self, input_dict, teacher_temp, visualize=False):
        # prepare input
        global_crop0_data_dict = dict(
            coord=input_dict["global_crop0_coord"],
            feat=input_dict["global_crop0_feat"],
            offset=input_dict["global_crop0_offset"],
            grid_coord=input_dict["global_crop0_grid_coord"],
        )
        global_crop1_data_dict = dict(
            coord=input_dict["global_crop1_coord"],
            feat=input_dict["global_crop1_feat"],
            offset=input_dict["global_crop1_offset"],
            grid_coord=input_dict["global_crop1_grid_coord"],
        )
        local_crop_data_dict_list = []
        for i in range(self.local_crop_num):
            local_crop_data_dict = dict(
                coord=input_dict[f"local_crop{i}_coord"].half(),
                feat=input_dict[f"local_crop{i}_feat"].half(),
                offset=input_dict[f"local_crop{i}_offset"],
                grid_coord=input_dict[f"local_crop{i}_grid_coord"],
            )
            local_crop_data_dict_list.append(local_crop_data_dict)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # get teacher output funciton
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        def get_teacher_output(global_view_0, global_view_1, global_mask_0=None, global_mask_1=None, local_views=None, local_masks=None):

            batch_size = global_view_0['offset'].shape[0] 
            teacher_dino_feat_global_0, teacher_patch_feat_global_0 = self.backbone_teacher(Point(global_view_0), return_dec=True)
            teacher_dino_feat_global_1, teacher_patch_feat_global_1 = self.backbone_teacher(Point(global_view_1), return_dec=True)
            
            teacher_dino_feat_global_0_pool = torch_scatter.segment_csr(
                src=teacher_dino_feat_global_0.feat, 
                indptr=nn.functional.pad(teacher_dino_feat_global_0.offset, (1, 0)),
                reduce="mean"
            )

            teacher_dino_feat_global_1_pool = torch_scatter.segment_csr(
                src=teacher_dino_feat_global_1.feat, 
                indptr=nn.functional.pad(teacher_dino_feat_global_1.offset, (1, 0)),
                reduce="mean"
            )
            assert global_mask_0 is not None and global_mask_1 is not None
            teacher_patch_feat_global_0_mask = teacher_patch_feat_global_0.feat[global_mask_0] # M0xC
            teacher_patch_feat_global_1_mask = teacher_patch_feat_global_1.feat[global_mask_1] # M1xC
            

            teacher_dino_feat_global_pool = torch.cat([teacher_dino_feat_global_0_pool.unsqueeze(1), teacher_dino_feat_global_1_pool.unsqueeze(1)], dim=1) # Batch x Crop x Feat_dim
            return teacher_dino_feat_global_pool, teacher_patch_feat_global_0_mask,  teacher_patch_feat_global_1_mask


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # get student output funciton
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        def get_student_output(global_view_0, global_view_1, local_views, global_mask_0=None, global_mask_1=None, local_masks=None):
            batch_size = global_view_0['offset'].shape[0] 
            student_dino_feat_global_0, student_patch_feat_global_0 = self.backbone_student(Point(global_view_0), mask=global_mask_0, return_dec=True)
            student_dino_feat_global_1, student_patch_feat_global_1 = self.backbone_student(Point(global_view_1), mask=global_mask_1, return_dec=True)
    

            # print("student_dino_feat_global_0", student_dino_feat_global_0.offset)
            student_dino_feat_global_0_pool = torch_scatter.segment_csr(
                src=student_dino_feat_global_0.feat, 
                indptr=nn.functional.pad(student_dino_feat_global_0.offset, (1, 0)),
                reduce="mean"
            )

            # print("student_dino_feat_global_1", student_dino_feat_global_1.offset)
            student_dino_feat_global_1_pool = torch_scatter.segment_csr(
                src=student_dino_feat_global_1.feat, 
                indptr=nn.functional.pad(student_dino_feat_global_1.offset, (1, 0)),
                reduce="mean"
            )

            assert global_mask_0 is not None and global_mask_1 is not None
            student_patch_feat_global_0_mask = student_patch_feat_global_0.feat[global_mask_0]
            student_patch_feat_global_1_mask = student_patch_feat_global_1.feat[global_mask_1] 
    

            student_dino_feat_global_pool = torch.cat([student_dino_feat_global_0_pool.unsqueeze(1), student_dino_feat_global_1_pool.unsqueeze(1)], dim=1)



            # disable local ibot, here the local input is not masked, and return dec is False
            student_dino_feat_local_pool_list = []
            for i in range(self.local_crop_num):
                student_dino_feat_local_i, _ = self.backbone_student(Point(local_views[i]), return_dec=False)
                student_dino_feat_local_i_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_local_i.feat, 
                    indptr=nn.functional.pad(student_dino_feat_local_i.offset, (1, 0)),
                    reduce="mean"
                )

                student_dino_feat_local_pool_list.append(student_dino_feat_local_i_pool)
            

            return student_dino_feat_global_pool, student_patch_feat_global_0_mask,  student_patch_feat_global_1_mask, student_dino_feat_local_pool_list
          

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # main function contintuse
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # view from global
        global_mask_0, global_mask_0_weight = self.mask_generator(global_crop0_data_dict["offset"], global_crop0_data_dict["grid_coord"])
        global_mask_1, global_mask_1_weight = self.mask_generator(global_crop1_data_dict["offset"], global_crop1_data_dict["grid_coord"])
        # by default we use different mask for two global view


        # prepare mask feature
        if self.enable_mae_loss:
            # we can set local crop to 0 
            # assert self.do_mask # only enable mae loss when ibot is enabled
            with torch.no_grad():
                student_patch_feat_global_0_mask_gt = global_crop0_data_dict["feat"][global_mask_0].clone()
                student_patch_feat_global_1_mask_gt = global_crop1_data_dict["feat"][global_mask_1].clone()
                if self.using_coord:
                    student_patch_feat_global_0_mask_gt = student_patch_feat_global_0_mask_gt[:, 3:]
                    student_patch_feat_global_1_mask_gt = student_patch_feat_global_1_mask_gt[:, 3:]

            
        # get teacher output: 
        if self.do_ema:
            # only global ibot
            teacher_dino_feat_global_pool, teacher_patch_feat_global_0_mask,  teacher_patch_feat_global_1_mask = get_teacher_output(global_crop0_data_dict, global_crop1_data_dict, global_mask_0, global_mask_1)

            student_dino_feat_global_pool, student_patch_feat_global_0_mask,  student_patch_feat_global_1_mask, student_dino_feat_local_pool_list = get_student_output(global_crop0_data_dict, global_crop1_data_dict, local_crop_data_dict_list, global_mask_0, global_mask_1)
        else:
            # no teacher model, only student model
            student_dino_feat_global_pool, student_patch_feat_global_0_mask,  student_patch_feat_global_1_mask, student_dino_feat_local_pool_list = get_student_output(global_crop0_data_dict, global_crop1_data_dict, local_crop_data_dict_list, global_mask_0, global_mask_1)

 

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Calculate loss
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        loss_dict = {}
        output_dict = {}

        if self.do_ema:
            dino_loss_dict = self.calculate_sim_dino_loss_pool(teacher_dino_feat_global_pool, student_dino_feat_global_pool, student_dino_feat_local_pool_list)

            loss_dict.update(dino_loss_dict)


        if self.do_ibot:
            ibot_loss_dict = self.calculate_sim_dino_loss_patch(teacher_patch_feat_global_0_mask,  teacher_patch_feat_global_1_mask, student_patch_feat_global_0_mask, student_patch_feat_global_1_mask, teacher_temp=teacher_temp, global_mask_weight_0=global_mask_0_weight, global_mask_weight_1=global_mask_1_weight)
            teacher_ibot_feat_batch0_crop0_mask0 = teacher_patch_feat_global_0_mask[0,:] # 1x64 
            teacher_ibot_feat_batch1_crop0_mask0 = teacher_patch_feat_global_0_mask[-1,:] # use last masked point so from different batch # 1x64
            teacher_ibot_indicator = torch.nn.functional.cosine_similarity(teacher_ibot_feat_batch0_crop0_mask0, teacher_ibot_feat_batch1_crop0_mask0, dim=-1)
            teacher_ibot_indicator = teacher_ibot_indicator.mean()
            output_dict["teacher_ibot_global_indicator"] = teacher_ibot_indicator


            loss_dict.update(ibot_loss_dict)

        if self.enable_mae_loss:
            student_patch_feat_global_0_mask = self.mae_head(student_patch_feat_global_0_mask)
            # student_patch_feat_global_1_mask = self.mae_head(student_patch_feat_global_1_mask)

            mae_loss = nn.functional.mse_loss(student_patch_feat_global_0_mask, student_patch_feat_global_0_mask_gt.detach())  # only the global transform 0 is weak transformation
 
            loss_dict["global_mae_loss"] = mae_loss

        if self.training:
            # combine all loss
            loss = 0.
            if self.do_ema:
                loss += self.dino_weight * loss_dict["sim_dino_crops_loss"]

            if self.do_ibot:
                loss +=self.ibot_weight * loss_dict["sim_ibot_patch_loss"]
                output_dict["teacher_ibot_global_indicator"] = teacher_ibot_indicator

            if self.enable_mae_loss:
                loss += self.mae_weight * loss_dict["global_mae_loss"]

            # assert no NaN in loss
            if torch.isnan(loss):
                raise ValueError("loss is NaN")

            output_dict['loss'] = loss
            output_dict.update(loss_dict)

            return output_dict
        else:
            raise ValueError("Not implemented yet")


