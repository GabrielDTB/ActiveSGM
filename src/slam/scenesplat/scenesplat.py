"""
SceneSplatam: SLAM with SceneSplat 3D semantic features.

Extends SplatamOurs with per-Gaussian semantic features produced by SceneSplat
(PTv3 backbone + Autoencoder), applied after each map update rather than
per-frame like OneFormer in SemSplatam.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import mmengine
from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.slam.splatam.splatam import SplatamOurs
from src.slam.splatam.eval_helper import eval, report_progress
from src.utils.general_utils import InfoPrinter
from src.slam.splatam.exploration_map import ExplorationMap
from third_parties.splatam.utils.slam_external import calc_psnr, calc_ssim, build_rotation
from third_parties.splatam.scripts.splatam import (
    get_dataset, initialize_camera_pose, get_loss,
)
from third_parties.splatam.datasets.gradslam_datasets import load_dataset_config
from third_parties.splatam.utils.slam_helpers import (
    matrix_to_quaternion, transform_to_frame,
    transformed_params2rendervar, transformed_params2depthplussilhouette,
)
from third_parties.splatam.utils.keyframe_selection import keyframe_selection_overlap
from third_parties.splatam.utils.slam_external import prune_gaussians
from third_parties.splatam.utils.eval_helpers import report_loss
from third_parties.splatam.utils.common_utils import save_params_ckpt, save_params
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer_sparse

from src.slam.splatam.modified_ver.scripts.splatam import *
from src.slam.scenesplat.modified_ver.splatam.splatam import (
    setup_camera,
    initialize_first_timestep,
    initialize_optimizer,
    add_new_gaussians_with_seman,
    transformed_params2semrendervar_sparse,
    set_camera_sparse,
    prune_gaussians_w_semantic,
    densify,
)
from src.slam.scenesplat.scenesplat_model import (
    load_scenesplat_backbone,
    load_autoencoder,
    run_scenesplat,
)

PRINT_INFO = True


class SceneSplatam(SplatamOurs):
    def __init__(self,
                 main_cfg: mmengine.Config,
                 info_printer: InfoPrinter,
                 logger: SummaryWriter) -> None:
        SplatamOurs.__init__(self, main_cfg, info_printer, logger)

        # SceneSplat configuration
        self.n_cls = self.slam_cfg.get('num_semantic_classes', 16)
        self.topk = self.slam_cfg.get('num_topk_logits', 16)
        self.scenesplat_grid_size = self.slam_cfg.get('scenesplat_grid_size', 0.02)

        # Load SceneSplat backbone (PTv3)
        pointcept_path = self.slam_cfg['pointcept_path']
        scenesplat_ckpt = self.slam_cfg['scenesplat_checkpoint']
        print(f"Loading SceneSplat backbone from {scenesplat_ckpt}...")
        self.backbone = load_scenesplat_backbone(scenesplat_ckpt, pointcept_path, self.device)

        # Load Autoencoder
        autoencoder_ckpt = self.slam_cfg['autoencoder_checkpoint']
        print(f"Loading Autoencoder from {autoencoder_ckpt}...")
        self.autoencoder = load_autoencoder(autoencoder_ckpt, self.device)

    def init_exploration_map(self, sim2slam: torch.tensor):
        """Initialize exploration map (grid)."""
        self.explr_map = ExplorationMap(
            self.slam_cfg.bbox_bound,
            self.slam_cfg.bbox_voxel_size,
            self.device,
            sim2slam,
            use_xyz_filter=True,
            xy_sampling_step=self.main_cfg.planner.xy_sampling_step[0],
            gs_z_levels=self.main_cfg.planner.gs_z_levels[0],
        )

    def init_camera_parameters(self):
        """Initialize camera and Gaussian parameters (no semantic annotation needed)."""
        if self.seperate_densification_res:
            params, variables, intrinsics, first_frame_w2c, cam, \
                densify_intrinsics, densify_cam = initialize_first_timestep(
                    self.dataset_sample, self.num_frames,
                    self.config['scene_radius_depth_ratio'],
                    self.config['mean_sq_dist_method'],
                    densify_dataset=self.densify_dataset_sample,
                    gaussian_distribution=self.config['gaussian_distribution'],
                    n_cls=self.n_cls)
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        else:
            params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
                self.dataset_sample, self.num_frames,
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                n_cls=self.n_cls)
            self.densify_intrinsics = intrinsics
            self.densify_cam = cam

        if self.seperate_tracking_res:
            self.tracking_cam = setup_camera(
                self.tracking_color.shape[2], self.tracking_color.shape[1],
                self.tracking_intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
                num_channels=self.n_cls)

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam

        # Run SceneSplat on initial Gaussians to populate semantic features
        self.run_scenesplat_on_map()

    @torch.no_grad()
    def run_scenesplat_on_map(self):
        """Run SceneSplat inference on the current Gaussian map.

        Updates params['semantic_logits'] with 16-dim features and
        sets variables['seman_cls_ids'] to identity indices.
        """
        N = self.params['means3D'].shape[0]
        if N == 0:
            return

        feat_16 = run_scenesplat(
            self.backbone, self.autoencoder, self.params,
            grid_size=self.scenesplat_grid_size, device=self.device)

        # Update semantic logits (detached, overwrite)
        self.params['semantic_logits'] = torch.nn.Parameter(
            feat_16.clone().contiguous().requires_grad_(True))

        # Identity class IDs (all 16 dims used)
        self.variables['seman_cls_ids'] = torch.arange(
            self.n_cls, device=self.device).unsqueeze(0).expand(N, -1).contiguous()

    @torch.no_grad()
    def render(self, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render RGB, depth, and valid mask at the given pose.

        Args:
            c2w: [4,4] camera-to-world pose in SplaTAM system.

        Returns:
            im: (C,H,W) rendered image
            depth: (1,H,W) rendered depth
            valid_depth_mask: (1,H,W) valid rendering mask
            seen: (N,) which Gaussians are visible
        """
        cam = self.cam
        first_frame_w2c = self.first_frame_w2c
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)
        sil_thres = self.config['mapping']['sil_thres']

        with torch.no_grad():
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c,
                                                                     transformed_gaussians)
        im, radii, _, = Renderer(raster_settings=cam)(**rendervar)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (depth_sil[0:1] > 0)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        seen = (radii > 0)
        return im, rastered_depth, valid_depth_mask, seen

    @torch.no_grad()
    def render_semantic(self, c2w: torch.Tensor, seen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render semantic features at the given pose.

        Args:
            c2w: [4,4] camera-to-world pose in SplaTAM system.
            seen: (N,) visibility mask from render().

        Returns:
            class_id: (H,W) argmax of semantic features.
            logits: (C,H,W) rendered semantic feature map.
        """
        cam = self.cam
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)

        with torch.no_grad():
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        variables = self.variables
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)
        seman_rendervar = transformed_params2semrendervar_sparse(params, transformed_gaussians, seen)
        sparse_cam = set_camera_sparse(cam=cam, cls_ids=variables['seman_cls_ids'])
        logits, _, = SEMRenderer_sparse(raster_settings=sparse_cam)(**seman_rendervar)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        class_id = logits.argmax(dim=0)
        return class_id, logits

    def online_recon_step(self,
                          time_idx: int,
                          color: torch.Tensor,
                          depth: torch.Tensor,
                          c2w: torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          ) -> List:
        """Run one step of SLAM. No per-frame semantic annotation.

        Args:
            time_idx: Current frame step.
            color: [H,W,3] color image.
            depth: [H,W] depth map.
            c2w: [4,4] camera-to-world pose (RUB).
            force_map_update: Force map update if True.
            only_use_global_keyframe: Post-refinement stage.
        """
        if time_idx == 0:
            self.init_camera_parameters()

        self.update_gs_map(time_idx, color, depth, c2w,
                           force_map_update, dont_add_kf, only_use_global_keyframe)
        if self.slam_cfg.enable_active_planning:
            self.update_explr_map(time_idx, depth, c2w, force_map_update)

    @torch.no_grad()
    def update_explr_map(self,
                         time_idx: int,
                         depth: torch.Tensor,
                         c2w: torch.Tensor,
                         force_map_update: bool = False):
        """Update exploration map."""
        config = self.config
        depth = depth.to(self.device)
        c2w = c2w.to(self.device)
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0 or force_map_update:
            self.explr_map.update_from_depth_map(
                depth,
                self.intrinsics,
                torch.inverse(c2w),
                self.slam_cfg.surface_dist_thre,
                self.slam_cfg.get("find_free_indices_bs", 10000),
            )

    def update_gs_map(self,
                      time_idx: int,
                      color: torch.Tensor,
                      depth: torch.Tensor,
                      c2w: torch.Tensor,
                      force_map_update: bool = False,
                      dont_add_kf: bool = False,
                      only_use_global_keyframe: bool = False,
                      ) -> List:
        """Run one step of the splatam process. Update Gaussian map.

        Uses geometry-only loss (RGB + depth). After mapping, runs SceneSplat
        to enrich all Gaussians with semantic features.
        """
        params = self.params
        variables = self.variables
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        cam = self.cam
        seperate_densification_res = self.seperate_densification_res
        if seperate_densification_res:
            densify_intrinsics = self.densify_intrinsics
            densify_cam = self.densify_cam
        config = self.config
        gt_w2c_all_frames = self.gt_w2c_all_frames
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
            wandb_mapping_step = self.wandb_mapping_step
            wandb_time_step = self.wandb_time_step
        eval_dir = self.eval_dir
        seperate_tracking_res = self.seperate_tracking_res
        if seperate_tracking_res:
            tracking_cam = self.tracking_cam
            tracking_intrinsics = self.tracking_intrinsics
        keyframe_list = self.keyframe_list
        num_frames = self.num_frames
        keyframe_time_indices = self.keyframe_time_indices

        # Process poses
        gt_w2c = torch.linalg.inv(c2w)

        # Process RGB-D Data
        color = color.permute(2, 0, 1)
        color = color.to(self.device)
        depth = depth.unsqueeze(0)
        depth = depth.to(self.device)

        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        iter_time_idx = time_idx

        # Initialize Mapping Data for selected frame (no seman)
        curr_data = {
            'cam': cam, 'im': color, 'depth': depth,
            'id': iter_time_idx, 'intrinsics': intrinsics,
            'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
        }

        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_h = self.config['data']["tracking_image_height"]
            tracking_w = self.config['data']["tracking_image_width"]
            tracking_color = F.interpolate(color.unsqueeze(0), (tracking_h, tracking_w), mode='bilinear')[0]
            tracking_depth = F.interpolate(depth.unsqueeze(0), (tracking_h, tracking_w), mode='nearest')[0]
            tracking_curr_data = {
                'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                'id': iter_time_idx, 'intrinsics': tracking_intrinsics,
                'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
            }
        else:
            tracking_curr_data = curr_data

        num_iters_mapping = config['mapping']['num_iters']

        # Initialize camera pose for current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx,
                                            forward_prop=config['tracking']['forward_prop'])

        ##################################################
        # Tracking
        ##################################################
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                loss, variables, losses = get_loss(
                    params, tracking_curr_data, variables, iter_time_idx,
                    config['tracking']['loss_weights'],
                    config['tracking']['use_sil_for_loss'],
                    config['tracking']['sil_thres'],
                    config['tracking']['use_l1'],
                    config['tracking']['ignore_outlier_depth_loss'],
                    tracking=True,
                    plot_dir=eval_dir,
                    visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                    tracking_iteration=iter)
                if config['use_wandb']:
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter + 1, progress_bar,
                                            iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, wandb_run=wandb_run, wandb_step=wandb_tracking_step,
                                            wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter + 1, progress_bar,
                                            iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)

                iter_end_time = time.time()
                self.tracking_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_iter_time_count += 1

                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2 * num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                           "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran

        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_count += 1

        if (time_idx == 0 or (time_idx + 1) % config['report_global_progress_every'] == 0) and not config['tracking']['use_gt_poses']:
            try:
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar,
                                        iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, wandb_run=wandb_run, wandb_step=wandb_time_step,
                                        wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar,
                                        iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        ##################################################
        # Update global keyframe (completeness-based)
        ##################################################
        if self.slam_cfg.use_global_keyframe and not only_use_global_keyframe:
            self.update_global_keyframe_set_completeness(
                depth, c2w,
                self.slam_cfg.global_keyframe.completeness_thre,
                time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config)

        ##################################################
        # Densification & KeyFrame-based Mapping
        ##################################################
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0 or force_map_update:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                if seperate_densification_res:
                    densify_h = self.config['data']["densification_image_height"]
                    densify_w = self.config['data']["densification_image_width"]
                    densify_color = F.interpolate(color.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]
                    densify_depth = F.interpolate(depth.unsqueeze(0), (densify_h, densify_w), mode='nearest')[0]
                    densify_curr_data = {
                        'cam': densify_cam, 'im': densify_color, 'depth': densify_depth,
                        'id': time_idx, 'intrinsics': densify_intrinsics,
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
                    }
                else:
                    densify_curr_data = curr_data

                params, variables = add_new_gaussians_with_seman(
                    params, variables, densify_curr_data,
                    config['mapping']['sil_thres'], time_idx,
                    config['mean_sq_dist_method'],
                    config['gaussian_distribution'],
                    n_cls=self.n_cls)
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})

            with torch.no_grad():
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size'] - 2
                selected_keyframes = keyframe_selection_overlap(
                    depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list) - 1)
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)

                if PRINT_INFO:
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    if self.slam_cfg.use_global_keyframe:
                        global_kf_times = [t for t in self.global_keyframe_time_indices if t != time_idx]
                        print(f"\nGlobal Keyframes at Frame {time_idx}: {global_kf_times}")

            # Reset Optimizer
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                # Frame selection for map update
                if only_use_global_keyframe or (self.slam_cfg.use_global_keyframe and iter > num_iters_mapping // 2):
                    # Global Keyframe
                    if len(self.global_keyframe_indices) == 1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        selected_rand_keyframe_idx = np.random.choice(self.global_keyframe_indices[:-1])
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                else:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']

                iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx + 1]
                iter_data = {
                    'cam': cam, 'im': iter_color, 'depth': iter_depth,
                    'id': iter_time_idx, 'intrinsics': intrinsics,
                    'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c,
                }

                # Geometry-only loss (no semantic loss)
                loss, variables, losses = get_loss(
                    params, iter_data, variables, iter_time_idx,
                    config['mapping']['loss_weights'],
                    config['mapping']['use_sil_for_loss'],
                    config['mapping']['sil_thres'],
                    config['mapping']['use_l1'],
                    config['mapping']['ignore_outlier_depth_loss'],
                    mapping=True)
                if config['use_wandb']:
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)

                torch.nn.utils.clip_grad_norm_(
                    [v for v in self.params.values() if isinstance(v, torch.nn.Parameter)],
                    max_norm=100.0)
                loss.backward()

                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians_w_semantic(
                            params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(
                            params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter + 1, progress_bar,
                                            iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step,
                                            wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter + 1, progress_bar,
                                            iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)

                iter_end_time = time.time()
                self.mapping_iter_time_sum += iter_end_time - iter_start_time
                self.mapping_iter_time_count += 1

            if num_iters_mapping > 0:
                progress_bar.close()

            mapping_end_time = time.time()
            self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
            self.mapping_frame_time_count += 1

            # Run SceneSplat after mapping to enrich Gaussians with semantic features
            self.run_scenesplat_on_map()

            if time_idx == 0 or (time_idx + 1) % config['report_global_progress_every'] == 0:
                try:
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx,
                                            sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_time_step,
                                            wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True,
                                            eval_dir=self.eval_dir)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx,
                                            sil_thres=config['mapping']['sil_thres'],
                                            eval_dir=self.eval_dir,
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

        ##################################################
        # Update global keyframe (quality-based)
        ##################################################
        if self.slam_cfg.use_global_keyframe and not only_use_global_keyframe:
            quality_method = self.slam_cfg.global_keyframe.get("quality_method", "absolute")
            if quality_method == "absolute":
                self.update_global_keyframe_set_quality(
                    color, depth, c2w,
                    self.slam_cfg.global_keyframe.color_thre,
                    self.slam_cfg.global_keyframe.depth_thre,
                    time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config)
            elif quality_method == "relative":
                if time_idx > 0 and time_idx % self.slam_cfg.global_keyframe.quality_freq == 0:
                    self.update_global_keyframe_set_quality_rel()

        # Add frame to keyframe list
        if not dont_add_kf:
            if ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or
                (time_idx == num_frames - 2)) and \
                    (not torch.isinf(curr_gt_w2c[-1]).any()) and \
                    (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)

        # Checkpoint
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)

        if config['use_wandb']:
            self.wandb_time_step += 1

        ##################################################
        # Update self variables
        ##################################################
        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam
        self.seperate_densification_res = seperate_densification_res
        if self.seperate_densification_res:
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        self.config = config
        self.gt_w2c_all_frames = gt_w2c_all_frames
        if self.config['use_wandb']:
            self.wandb_run = wandb_run
            self.wandb_mapping_step = wandb_mapping_step
            self.wandb_time_step = wandb_time_step
        self.seperate_tracking_res = seperate_tracking_res
        if self.seperate_tracking_res:
            self.tracking_cam = tracking_cam
            self.tracking_intrinsics = tracking_intrinsics
        self.keyframe_list = keyframe_list
        self.num_frames = num_frames
        self.keyframe_time_indices = keyframe_time_indices

    def update_global_keyframe_set_completeness(self, depth, c2w, thre,
                                                 time_idx, curr_gt_w2c, dont_add_kf,
                                                 num_frames, force_map_update, config):
        """Add keyframes based on completeness (new pixel coverage)."""
        if not dont_add_kf:
            if ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or
                (time_idx == num_frames - 2)) and \
                    (not torch.isinf(curr_gt_w2c[-1]).any()) and \
                    (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    new_pixel_num = ((depth > 0) * (~self.render(c2w)[2])).sum()
                    _, h, w = depth.shape
                    new_pixel_ratio = new_pixel_num / (h * w)
                    is_global_kf = new_pixel_ratio > thre
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality(self, color, depth, c2w, color_thre, depth_thre,
                                            time_idx, curr_gt_w2c, dont_add_kf,
                                            num_frames, force_map_update, config):
        """Add keyframes based on quality (rendering error)."""
        if time_idx in self.global_keyframe_time_indices:
            return
        if not dont_add_kf:
            if ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or
                (time_idx == num_frames - 2)) and \
                    (not torch.isinf(curr_gt_w2c[-1]).any()) and \
                    (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    render_color, render_depth, _, _ = self.render(c2w)
                    valid_depth_mask = depth > 0
                    color_ig = calc_psnr(render_color * valid_depth_mask,
                                         color * valid_depth_mask).mean()
                    is_global_kf = color_ig < color_thre
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality_rel(self):
        """Add keyframes based on relative quality (percentile-based)."""
        color_igs = []
        for kf in self.keyframe_list[:-5]:
            c2w = torch.inverse(kf['est_w2c'])
            color, _, valid_mask, seen = self.render(c2w)
            valid_depth_mask = kf['depth'] > 0.2
            color_ig = calc_psnr(color * valid_depth_mask, kf['color'] * valid_depth_mask).mean()
            color_igs.append(color_ig)

        selected_kf_idxs = []
        if len(color_igs) > 0:
            color_igs = torch.stack(color_igs).float()
            color_thre = torch.quantile(color_igs, self.slam_cfg.global_keyframe.quality_perc_thre / 100.)
            color_kf_idxs = torch.where(color_igs <= color_thre)[0]
            selected_kf_idxs = color_kf_idxs

        if len(selected_kf_idxs) > 0:
            new_kf = [elem.item() for elem in selected_kf_idxs if elem not in self.global_keyframe_indices]
            self.global_keyframe_indices.extend(new_kf)
            new_kf_time_indices = [self.keyframe_time_indices[i] for i in new_kf]
            self.global_keyframe_time_indices.extend(new_kf_time_indices)

    def plot_render_depth(self, c2w: torch.Tensor):
        """Plot rendered depth at the given pose."""
        _, depth, _, _ = self.render(c2w)
        depth = depth[0].detach().cpu().numpy()
        plt.imshow(depth)
        plt.show()

    def plot_render_rgb(self, c2w: torch.Tensor):
        """Plot rendered RGB at the given pose."""
        im, _, _, _ = self.render(c2w)
        im = im.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(im)
        plt.show()

    def print_and_save_result(self, eval_dir_suffix="", is_prune_gaussians=False, ignore_first_frame=False):
        """Evaluate rendering results and save."""
        params = self.params.copy()
        variables = self.variables.copy()
        config = self.config
        dataset_config = self.config['data']
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        if is_prune_gaussians:
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
            params, variables = prune_gaussians_w_semantic(params, variables, optimizer, 0, config['mapping']['pruning_dict'])

        # Compute Average Runtimes
        if self.tracking_iter_time_count == 0:
            self.tracking_iter_time_count = 1
            self.tracking_frame_time_count = 1
        if self.mapping_iter_time_count == 0:
            self.mapping_iter_time_count = 1
            self.mapping_frame_time_count = 1
        tracking_iter_time_avg = self.tracking_iter_time_sum / self.tracking_iter_time_count
        tracking_frame_time_avg = self.tracking_frame_time_sum / self.tracking_frame_time_count
        mapping_iter_time_avg = self.mapping_iter_time_sum / self.mapping_iter_time_count
        mapping_frame_time_avg = self.mapping_frame_time_sum / self.mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg * 1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg * 1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")

        # Evaluate Final Parameters (RGB only, no semantic eval)
        dataset = self.dataset_sample
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params, len(dataset), eval_dir,
                     sil_thres=config['mapping']['sil_thres'],
                     wandb_run=self.wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'],
                     ignore_first_frame=ignore_first_frame)
            else:
                eval(dataset, params, len(dataset), eval_dir,
                     sil_thres=config['mapping']['sil_thres'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'],
                     ignore_first_frame=ignore_first_frame)

        # Save params
        params['timestep'] = variables['timestep']
        params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        params['org_width'] = dataset_config["desired_image_width"]
        params['org_height'] = dataset_config["desired_image_height"]
        params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in self.gt_w2c_all_frames:
            params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
        params['keyframe_time_indices'] = np.array(self.keyframe_time_indices)
        params['seman_cls_ids'] = variables['seman_cls_ids'].detach().cpu().contiguous().numpy()

        results_dir = os.path.join(self.results_dir, eval_dir_suffix) if eval_dir_suffix else self.results_dir
        os.makedirs(results_dir, exist_ok=True)
        save_params(params, results_dir)

    def load_params_by_step(self, step=1100, stage='final'):
        """Load checkpoint parameters."""
        config = self.config
        print(f"Loading Checkpoint for Frame {step}")
        if step == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"{stage}/params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{step}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        self.variables = {
            'seman_cls_ids': params.pop('seman_cls_ids').to(torch.long),
            'n_cls': self.n_cls,
        }
        self.params = params

    def override_config(self, config: Dict) -> Dict:
        """Override configs from main_cfg."""
        config["data"]["sequence"] = self.main_cfg.general.scene
        config["workdir"] = os.path.join(self.main_cfg.dirs.result_dir, "splatam")
        config["run_name"] = ""
        if self.main_cfg.general.dataset == 'Replica':
            config["data"]['gradslam_data_cfg'] = os.path.join(
                "third_parties/splatam", config["data"]['gradslam_data_cfg'])

        config = self.update_dict_recursive(config, self.main_cfg.slam.override)

        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking']:
            config['tracking']['use_depth_loss_thres'] = False
            config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking']:
            config['tracking']['visualize_tracking_loss'] = False
        if "gaussian_distribution" not in config:
            config['gaussian_distribution'] = "isotropic"
        print(f"{config}")

        return config

    def update_dict_recursive(self, dict1, dict2):
        """Recursively update dict1 with values from dict2."""
        for key, value in dict2.items():
            if key not in dict1:
                raise KeyError(f"Key '{key}' not found in dict1.")
            if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
                dict1[key] = self.update_dict_recursive(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    def update_prev_keyframes(self):
        """Update last stored keyframe indices."""
        self.prev_keyframe_idxs = self.keyframe_time_indices.copy()

    def get_new_keyframe_idxs(self) -> torch.Tensor:
        """Get mask for new keyframes since last check."""
        prev_kf_idxs = torch.tensor(self.prev_keyframe_idxs)
        new_kf_idxs = torch.tensor(self.keyframe_time_indices)
        prev_flat = prev_kf_idxs.view(-1, 1)
        new_flat = new_kf_idxs.view(1, -1)
        mask = (prev_flat == new_flat).any(dim=0)
        return ~mask
