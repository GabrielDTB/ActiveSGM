"""
SceneSplat model loading and inference utilities.

Loads the PTv3 backbone and Autoencoder, and provides functions to run
SceneSplat inference on SplaTAM Gaussian parameters to produce per-Gaussian
16-dim semantic features.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Autoencoder(nn.Module):
    """Autoencoder for compressing 768-dim PTv3 features to 16-dim latent."""

    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, in_feature_dim=768):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(in_feature_dim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.BatchNorm1d(encoder_hidden_dims[i - 1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[i - 1], encoder_hidden_dims[i])
                )
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(
                    nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i])
                )
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(
                    nn.Linear(decoder_hidden_dims[i - 1], decoder_hidden_dims[i])
                )
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_hidden_dims[-1], in_feature_dim))
        self.decoder = nn.ModuleList(decoder_layers)

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


def load_scenesplat_backbone(checkpoint_path, pointcept_path, device):
    """Load PTv3 backbone from a SceneSplat checkpoint.

    Args:
        checkpoint_path: Path to the SceneSplat .pth checkpoint.
        pointcept_path: Path to the SceneSplat repo root (contains pointcept/).
        device: torch device.

    Returns:
        backbone: PointTransformerV3 model in eval mode.
    """
    if pointcept_path not in sys.path:
        sys.path.insert(0, pointcept_path)

    from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
        PointTransformerV3,
    )

    backbone = PointTransformerV3(
        in_channels=11,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(768, 512, 256),
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Strip 'module.backbone.' prefix from all keys
    prefix = "module.backbone."
    backbone_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            backbone_sd[k[len(prefix) :]] = v

    backbone.load_state_dict(backbone_sd)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def load_autoencoder(checkpoint_path, device):
    """Load the Autoencoder model.

    Args:
        checkpoint_path: Path to the autoencoder .pth checkpoint.
        device: torch device.

    Returns:
        autoencoder: Autoencoder model in eval mode.
    """
    autoencoder = Autoencoder(
        encoder_hidden_dims=[512, 256, 128, 64, 32, 16],
        decoder_hidden_dims=[32, 64, 128, 256, 256, 512],
        in_feature_dim=768,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def _voxel_downsample(coord, grid_size):
    """Voxel grid downsampling using FNV hashing (matching pointcept's GridSample).

    Args:
        coord: [N, 3] float tensor of 3D coordinates.
        grid_size: float, voxel size in meters.

    Returns:
        selected_indices: [M] long tensor, indices of selected representative points.
        inverse: [N] long tensor, maps each original point to its voxel representative.
    """
    scaled_coord = coord / grid_size
    grid_coord = torch.floor(scaled_coord).long()
    grid_coord = grid_coord - grid_coord.min(dim=0)[0]

    # FNV hash
    p0 = torch.tensor(73856093, dtype=torch.long, device=coord.device)
    p1 = torch.tensor(19349669, dtype=torch.long, device=coord.device)
    p2 = torch.tensor(83492791, dtype=torch.long, device=coord.device)
    keys = grid_coord[:, 0] * p0 ^ grid_coord[:, 1] * p1 ^ grid_coord[:, 2] * p2

    # Sort and find unique voxels
    sort_idx = torch.argsort(keys)
    keys_sorted = keys[sort_idx]

    # Find first occurrence of each unique key
    unique_mask = torch.cat(
        [torch.tensor([True], device=coord.device), keys_sorted[1:] != keys_sorted[:-1]]
    )
    # Selected indices (first point per voxel)
    selected_sorted = torch.where(unique_mask)[0]
    selected_indices = sort_idx[selected_sorted]

    # Build inverse mapping: for each original point, which voxel index it belongs to
    voxel_ids = torch.cumsum(unique_mask, dim=0) - 1  # [N_sorted] -> voxel id
    inverse = torch.empty_like(keys)
    inverse[sort_idx] = voxel_ids

    return selected_indices, inverse


def prepare_gaussian_input(params, grid_size=0.02):
    """Convert SplaTAM params to PTv3 input format.

    Args:
        params: dict of SplaTAM Gaussian parameters (means3D, rgb_colors, etc.)
        grid_size: voxel grid size for GridSample (meters).

    Returns:
        data_dict: dict with coord, feat, grid_coord, offset for PTv3.
        inverse: [N] long tensor mapping original points to voxel representatives.
    """
    device = params["means3D"].device

    # Extract and transform Gaussian attributes
    coord = params["means3D"].detach()  # [N, 3]
    color = torch.sigmoid(params["rgb_colors"].detach()) * 255.0  # [N, 3] in [0, 255]
    opacity = torch.sigmoid(params["logit_opacities"].detach())  # [N, 1]
    quat = F.normalize(params["unnorm_rotations"].detach(), dim=-1)  # [N, 4]
    # Enforce positive real part
    signs = torch.sign(quat[:, 0:1])
    signs[signs == 0] = 1
    quat = quat * signs
    scale = torch.exp(params["log_scales"].detach())  # [N, 1 or 3]
    if scale.shape[1] == 1:
        scale = scale.expand(-1, 3)

    # NormalizeColor: color / 127.5 - 1
    color_norm = color / 127.5 - 1.0  # [N, 3] in [-1, 1]

    # Voxel downsample
    selected_idx, inverse = _voxel_downsample(coord, grid_size)

    # Select representative points
    coord_ds = coord[selected_idx]  # [M, 3]
    color_ds = color_norm[selected_idx]  # [M, 3]
    opacity_ds = opacity[selected_idx]  # [M, 1]
    quat_ds = quat[selected_idx]  # [M, 4]
    scale_ds = scale[selected_idx]  # [M, 3]

    # feat = cat([color, opacity, quat, scale], dim=-1) -> [M, 11]
    feat = torch.cat([color_ds, opacity_ds, quat_ds, scale_ds], dim=-1)

    # Grid coordinates for the downsampled points
    grid_coord = torch.floor((coord_ds - coord_ds.min(dim=0)[0]) / grid_size).int()

    # Batch offset (single scene)
    offset = torch.tensor([coord_ds.shape[0]], dtype=torch.long, device=device)

    data_dict = {
        "coord": coord_ds.float(),
        "feat": feat.float(),
        "grid_coord": grid_coord,
        "offset": offset,
    }
    return data_dict, inverse


@torch.no_grad()
def run_scenesplat(backbone, autoencoder, params, grid_size=0.02, device=None):
    """Run SceneSplat inference on current Gaussian map.

    Args:
        backbone: PTv3 backbone model.
        autoencoder: Autoencoder model.
        params: dict of SplaTAM Gaussian parameters.
        grid_size: voxel grid size for input preparation.
        device: torch device (if None, uses params device).

    Returns:
        feat_16: [N, 16] float tensor of per-Gaussian 16-dim semantic features.
    """
    if device is None:
        device = params["means3D"].device

    N = params["means3D"].shape[0]
    if N == 0:
        return torch.zeros(0, 16, device=device)

    # Prepare input
    data_dict, inverse = prepare_gaussian_input(params, grid_size)

    # Move to device
    data_dict = {k: v.to(device) for k, v in data_dict.items()}

    # Run PTv3 backbone
    point_out = backbone(data_dict)
    feat_768 = point_out.feat  # [M, 768] (first decoder output dim)

    # Autoencoder encode: 768 -> 16
    feat_16_ds = autoencoder.encode(feat_768)  # [M, 16]

    # Map back to all original points via inverse mapping
    feat_16 = feat_16_ds[inverse]  # [N, 16]

    return feat_16
