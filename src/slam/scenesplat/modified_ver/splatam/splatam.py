"""
Adapted helper functions for SceneSplat SLAM.

Based on src/slam/semsplatam/modified_ver/splatam/splatam.py, but simplified:
- Semantic logits are initialized as zeros (filled later by SceneSplat inference)
- No semantic loss functions needed
- Keeps semantic rendering and pruning infrastructure for render_semantic
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor

from third_parties.splatam.utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    transform_to_frame,
)
from third_parties.splatam.utils.slam_external import build_rotation
from third_parties.splatam.utils.gs_external import (
    update_params_and_optimizer,
    inverse_sigmoid,
    cat_params_to_optimizer,
    accumulate_mean2d_gradient,
)
from third_parties.splatam.scripts.splatam import get_pointcloud
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer_sparse
from sparse_channel_rasterization import GaussianRasterizationSettings as Camera_sparse


def setup_camera(w, h, k, w2c, near=0.01, far=100, num_channels=16):
    from channel_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.zeros(num_channels, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
        num_channels=num_channels,
    )
    return cam


def initialize_params_with_seman(init_pt_cld, num_frames, mean3_sq_dist,
                                  gaussian_distribution, n_cls=16):
    """Initialize Gaussian parameters with zero semantic logits.

    Unlike SemSplatam, semantic logits start as zeros and are filled by SceneSplat
    inference after the first mapping step.
    """
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'semantic_logits': torch.zeros((num_pts, n_cls), dtype=torch.float, device="cuda"),
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    # seman_cls_ids: identity mapping [N, n_cls] = [0, 1, 2, ..., n_cls-1] per row
    cls_ids = torch.arange(n_cls, device="cuda").unsqueeze(0).expand(num_pts, -1)

    variables = {
        'max_2D_radius': torch.zeros(num_pts).cuda().float(),
        'means2D_gradient_accum': torch.zeros(num_pts).cuda().float(),
        'denom': torch.zeros(num_pts).cuda().float(),
        'timestep': torch.zeros(num_pts).cuda().float(),
        'seman_cls_ids': cls_ids,
    }
    return params, variables


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio,
                               mean_sq_dist_method, densify_dataset=None,
                               gaussian_distribution=None, n_cls=16):
    """Initialize first timestep with zero semantic logits.

    Unlike SemSplatam's version, this does not take a `seman` argument.
    """
    color, depth, intrinsics, pose = dataset[0]
    color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)

    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    cam = setup_camera(color.shape[2], color.shape[1],
                       intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(),
                       num_channels=n_cls)

    if densify_dataset is not None:
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1],
                                   densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(),
                                   num_channels=n_cls)
    else:
        densify_intrinsics = intrinsics

    # Get initial point cloud (no semantics — just xyz + rgb)
    mask = (depth > 0).reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(
        color, depth, densify_intrinsics, w2c,
        mask=mask, compute_mean_sq_dist=True,
        mean_sq_dist_method=mean_sq_dist_method)

    # Initialize parameters with zero semantic logits
    params, variables = initialize_params_with_seman(
        init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, n_cls)

    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio
    variables['n_cls'] = n_cls

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def add_new_gaussians_with_seman(params, variables, curr_data, sil_thres,
                                  time_idx, mean_sq_dist_method,
                                  gaussian_distribution, n_cls=16):
    """Add new Gaussians with zero semantic logits.

    New Gaussians are initialized with zero features; they will be enriched
    by the next SceneSplat inference pass.
    """
    transformed_gaussians = transform_to_frame(params, time_idx,
                                               gaussians_grad=False,
                                               camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data['w2c'], transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)

    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50 * depth_error.median())

    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    non_presence_mask = non_presence_mask.reshape(-1)

    if torch.sum(non_presence_mask) > 0:
        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(
            curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
            curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method)

        num_new_pts = new_pt_cld.shape[0]

        # Initialize new Gaussian params with zero semantics
        new_means3D = new_pt_cld[:, :3]
        new_rgb = new_pt_cld[:, 3:6]
        new_unnorm_rots = np.tile([1, 0, 0, 0], (num_new_pts, 1))
        new_logit_opacities = torch.zeros((num_new_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            new_log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            new_log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        new_params = {
            'means3D': new_means3D,
            'rgb_colors': new_rgb,
            'semantic_logits': torch.zeros((num_new_pts, n_cls), dtype=torch.float, device="cuda"),
            'unnorm_rotations': torch.tensor(new_unnorm_rots).cuda().float(),
            'logit_opacities': new_logit_opacities,
            'log_scales': new_log_scales,
        }
        for k, v in new_params.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            new_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

        # Update variables
        new_cls_ids = torch.arange(n_cls, device="cuda").unsqueeze(0).expand(num_new_pts, -1)
        variables['seman_cls_ids'] = torch.cat(
            (variables['seman_cls_ids'], new_cls_ids), dim=0)
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx * torch.ones(num_new_pts, device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep), dim=0)

    return params, variables


def transformed_params2semrendervar_sparse(params, transformed_gaussians, seen):
    """Prepare render variables for sparse semantic rasterization."""
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'][seen], (1, 3))
    else:
        log_scales = params['log_scales'][seen]

    rendervar = {
        'means3D': transformed_gaussians['means3D'][seen].detach(),
        'colors_precomp': params['semantic_logits'][seen],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations'][seen]).detach(),
        'opacities': torch.sigmoid(params['logit_opacities'][seen]).detach(),
        'scales': torch.exp(log_scales).detach(),
        'means2D': torch.zeros_like(params['means3D'][seen], requires_grad=True, device="cuda") + 0,
    }
    return rendervar


def set_camera_sparse(cam, cls_ids=None):
    """Create a sparse camera settings object for semantic rendering."""
    cam = Camera_sparse(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=cam.bg,
        scale_modifier=1.0,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=cam.sh_degree,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
        num_channels=cam.num_channels,
        cls_ids=cls_ids.to(torch.int32)
    )
    return cam


def prune_gaussians_w_semantic(params, variables, optimizer, iter, prune_dict):
    """Prune Gaussians while maintaining semantic logits consistency."""
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()

            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)

            params, variables = remove_points(to_remove, params, variables, optimizer)

        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def remove_points(to_remove, params, variables, optimizer):
    """Remove Gaussians and their semantic data."""
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'seman_cls_ids' in variables:
        variables['seman_cls_ids'] = variables['seman_cls_ids'][to_keep]
    if 'timestep' in variables:
        variables['timestep'] = variables['timestep'][to_keep]
    return params, variables


def densify(params, variables, optimizer, iter, densify_dict):
    """Gaussian-Splatting densification with semantic logits handling.

    Follows the same logic as semsplatam's densify, adapted for identity cls_ids.
    """
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(
                grads >= grad_thresh,
                torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius'])
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            # Extend seman_cls_ids for cloned points
            if 'seman_cls_ids' in variables:
                variables['seman_cls_ids'] = torch.cat(
                    (variables['seman_cls_ids'], variables['seman_cls_ids'][to_clone]), dim=0)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(
                padded_grad >= grad_thresh,
                torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables['scene_radius'])
            n = densify_dict['num_to_split_into']
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)

            # Update seman_cls_ids: match the original semsplatam pattern
            if 'seman_cls_ids' in variables:
                new_variables = {
                    'seman_cls_ids': variables['seman_cls_ids'][to_split].repeat(n, 1),
                }
                variables['seman_cls_ids'] = torch.cat(
                    (variables['seman_cls_ids'], new_variables['seman_cls_ids']), dim=0)

            num_pts = params['means3D'].shape[0]
            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    """Initialize optimizer for all parameters including semantic_logits."""
    param_groups = []
    for k, v in params.items():
        lr = lrs_dict.get(k, 0.0)
        param_groups.append({'params': [v], 'name': k, 'lr': lr})
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
