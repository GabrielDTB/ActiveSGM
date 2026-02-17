import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from .GaussianLatentBKI import *
from autoencoder.model import Autoencoder
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt

def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_scenes(root: str):
    """
    List scene folder names under root which contain coord.npy / scale.npy / opacity.npy.
    """
    scenes = []
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if not os.path.isdir(scene_dir):
            continue
        if os.path.exists(os.path.join(scene_dir, "coord.npy")) and \
           os.path.exists(os.path.join(scene_dir, "scale.npy")) and \
           os.path.exists(os.path.join(scene_dir, "opacity.npy")):
            scenes.append(name)
    return scenes


def load_feat_pth(path: str) -> torch.Tensor:
    """
    Robust loader for .pth feature files.
    Supports:
      - a Tensor directly
      - a dict containing feature under common keys
      - numpy arrays inside dict
    Returns a torch.Tensor on CPU (caller moves to device).
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        feat = obj
    elif isinstance(obj, dict):
        for k in ["feat", "feats", "feature", "features", "semantic", "sem", "embedding", "emb"]:
            if k in obj:
                feat = obj[k]
                break
        else:
            tensor_like = [v for v in obj.values() if isinstance(v, (torch.Tensor, np.ndarray))]
            if len(tensor_like) == 1:
                feat = tensor_like[0]
            else:
                raise KeyError(f"Cannot find feature tensor in {path}. Keys={list(obj.keys())}")
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        feat = obj[0]
    else:
        raise TypeError(f"Unsupported feature file content type: {type(obj)}")

    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Feature is not Tensor after parsing. Got: {type(feat)}")

    return feat


def ensure_scale_shape(scale: torch.Tensor) -> torch.Tensor:
    """
    Ensure scale is (N,1). If (N,3), take max as an effective radius.
    """
    if scale.ndim == 1:
        return scale.view(-1, 1)
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale
    if scale.ndim == 2 and scale.shape[1] == 3:
        return scale.max(dim=1, keepdim=True).values
    raise ValueError(f"Unexpected scale shape: {tuple(scale.shape)}")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: (N,D) -> returns (N,)
    """
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1)


@torch.no_grad()
def build_edges_cdist_small(
    xyz_dst: torch.Tensor,        # (Q,3)
    r_dst: torch.Tensor,          # (Q,1)
    xyz_src: torch.Tensor,        # (M,3)
    src_global_idx: torch.Tensor, # (M,)
    Kmax: int,
):
    """
    Brute-force neighbor building for testing:
    - Compute full distance matrix (Q,M).
    - For each dst, keep neighbors within r_dst and truncate to Kmax closest.
    Returns flattened edges: dst_local, src_global, dist, ell
    """
    device = xyz_dst.device
    Q = xyz_dst.shape[0]

    dmat = torch.cdist(xyz_dst, xyz_src)  # (Q,M)

    dst_list, src_list, dist_list, ell_list = [], [], [], []

    for i in range(Q):
        d = dmat[i]
        mask = d < r_dst[i, 0]
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            continue

        d_sel = d[idx]
        if idx.numel() > Kmax:
            topk = torch.topk(d_sel, k=Kmax, largest=False)
            idx = idx[topk.indices]
            d_sel = topk.values

        dst_local = torch.full((idx.numel(),), i, device=device, dtype=torch.long)
        src_global = src_global_idx[idx]

        dst_list.append(dst_local)
        src_list.append(src_global)
        dist_list.append(d_sel.reshape(-1, 1))
        ell_list.append(r_dst[i].expand(idx.numel(), 1))

    if len(dst_list) == 0:
        return (None, None, None, None)

    return (
        torch.cat(dst_list, dim=0),
        torch.cat(src_list, dim=0),
        torch.cat(dist_list, dim=0),
        torch.cat(ell_list, dim=0),
    )

def scalar_to_rgb(val01: np.ndarray) -> np.ndarray:
    """
    val01: (N,) float in [0,1]
    returns rgb uint8 (N,3) using a heatmap colormap
    """
    val01 = np.clip(val01, 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("turbo")  # nice heatmap
        rgba = cmap(val01)           # (N,4)
        rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
        return rgb
    except Exception:
        # fallback: blue->red
        r = (val01 * 255.0).astype(np.uint8)
        g = np.zeros_like(r, dtype=np.uint8)
        b = ((1.0 - val01) * 255.0).astype(np.uint8)
        return np.stack([r, g, b], axis=1)

def save_ply_xyz_rgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    xyz: (N,3) float
    rgb: (N,3) uint8
    Writes ASCII PLY.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def save_ply_splats(
    path: str,
    xyz: np.ndarray,         # (N,3) float
    rgb_uint8: np.ndarray,   # (N,3) uint8 -> mapped to f_dc_0..2 in [0,1]
    opacity: np.ndarray,     # (N,1) or (N,) float
    scale: np.ndarray,       # (N,1) or (N,3) float
    quat: np.ndarray,        # (N,4) float  (written as rot_0..3)
):
    """
    Write a Gaussian-splat style PLY (ASCII) with fields:
      x y z
      f_dc_0 f_dc_1 f_dc_2   (color DC)
      opacity
      scale_0 scale_1 scale_2
      rot_0 rot_1 rot_2 rot_3

    Notes:
      - rgb is encoded as DC SH color in [0,1].
      - If scale is (N,1), it is broadcast to (N,3).
      - If opacity is (N,1), it's squeezed.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb_uint8.ndim == 2 and rgb_uint8.shape[1] == 3
    assert quat.ndim == 2 and quat.shape[1] == 4
    n = xyz.shape[0]
    assert rgb_uint8.shape[0] == n and quat.shape[0] == n

    # opacity -> (N,)
    if opacity.ndim == 2 and opacity.shape[1] == 1:
        opacity = opacity[:, 0]
    elif opacity.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected opacity shape: {opacity.shape}")

    # scale -> (N,3)
    if scale.ndim == 2 and scale.shape[1] == 1:
        scale = np.repeat(scale, 3, axis=1)
    elif scale.ndim == 1:
        scale = np.repeat(scale.reshape(-1, 1), 3, axis=1)
    elif scale.ndim == 2 and scale.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected scale shape: {scale.shape}")

    # rgb -> f_dc in [0,1]
    f_dc = (rgb_uint8.astype(np.float32) / 255.0)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float f_dc_0\n")
        f.write("property float f_dc_1\n")
        f.write("property float f_dc_2\n")
        f.write("property float opacity\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = xyz[i]
            r0, r1, r2 = f_dc[i]
            op = float(opacity[i])
            s0, s1, s2 = scale[i]
            q0, q1, q2, q3 = quat[i]
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{r0:.6f} {r1:.6f} {r2:.6f} "
                f"{op:.6f} "
                f"{s0:.6f} {s1:.6f} {s2:.6f} "
                f"{q0:.6f} {q1:.6f} {q2:.6f} {q3:.6f}\n"
            )

def cosine_to_rgb_bwr(cos: np.ndarray,
                      clamp: bool = True,
                      nan_color=(0.5, 0.5, 0.5)) -> np.ndarray:
    """
    Map cosine similarity in [-1, 1] to RGB in [0, 1]:
      -1 -> blue,  0 -> white,  +1 -> red
    """
    cos = np.asarray(cos, dtype=np.float64).reshape(-1)

    # handle NaN/Inf
    bad = ~np.isfinite(cos)
    cos_clean = cos.copy()
    cos_clean[bad] = 0.0

    if clamp:
        cos_clean = np.clip(cos_clean, -1.0, 1.0)

    # [-1,1] -> [0,1]
    t = (cos_clean + 1.0) * 0.5

    rgb = np.empty((t.shape[0], 3), dtype=np.float64)

    left = t <= 0.5
    right = ~left

    # blue -> white
    u = t[left] / 0.5
    rgb[left, 0] = u
    rgb[left, 1] = u
    rgb[left, 2] = 1.0

    # white -> red
    v = (t[right] - 0.5) / 0.5
    rgb[right, 0] = 1.0
    rgb[right, 1] = 1.0 - v
    rgb[right, 2] = 1.0 - v

    # color invalid values
    rgb[bad] = np.array(nan_color, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb255 = (rgb * 255.0 + 0.5).astype(np.uint8)

    # ensure [0,1]
    return rgb255


def nn_map_pred_to_gt_o3d(pred_pc: np.ndarray,
                          gt_pc: np.ndarray,
                          threshold: float,
                          return_dist: bool = False):
    """
    Map each point in pred_pc to its nearest neighbor in gt_pc within a distance threshold.

    Args:
        pred_pc: (N, 3) float array
        gt_pc:   (M, 3) float array
        threshold: max allowed Euclidean distance for a valid match
        return_dist: if True, also return nearest distances (np.inf if no match)

    Returns:
        mapping: (N,) int64, nearest gt index or -1
        mask:    (N,) bool, True if mapped (distance <= threshold)
        dists:   (N,) float (optional), nearest distance or np.inf
    """
    pred_pc = np.asarray(pred_pc, dtype=np.float64)
    gt_pc = np.asarray(gt_pc, dtype=np.float64)

    assert pred_pc.ndim == 2 and pred_pc.shape[1] == 3, "pred_pc must be (N,3)"
    assert gt_pc.ndim == 2 and gt_pc.shape[1] == 3, "gt_pc must be (M,3)"
    assert threshold >= 0, "threshold must be non-negative"

    n_pred = pred_pc.shape[0]
    mapping = np.full((n_pred,), -1, dtype=np.int64)
    mask = np.zeros((n_pred,), dtype=bool)
    dists = np.full((n_pred,), np.inf, dtype=np.float64)

    if gt_pc.shape[0] == 0 or n_pred == 0:
        return (mapping, mask, dists) if return_dist else (mapping, mask)

    # Build KDTree on gt
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pc)
    kdtree = o3d.geometry.KDTreeFlann(gt_pcd)

    thr2 = float(threshold) * float(threshold)

    # Query 1-NN for each pred point
    for i in range(n_pred):
        q = pred_pc[i]
        k, idx, dist2 = kdtree.search_knn_vector_3d(q, 1)
        if k > 0:
            # Open3D returns squared distances for KDTreeFlann
            if dist2[0] <= thr2:
                mapping[i] = int(idx[0])
                mask[i] = True
                dists[i] = float(np.sqrt(dist2[0]))

    return (mapping, mask, dists) if return_dist else (mapping, mask)

def compute_roc(opt,est,intervals = 20): # input torch.tensor
    ROC = []
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [torch.quantile(est, q / 100.0) for q in quants]
    subs = [est <= t for t in thres]
    ROC_points = [opt[s].mean().item() if s.any() else 0.0 for s in subs]
    ROC.extend(ROC_points)
    ROC_tensor = torch.tensor(ROC)
    AUC = torch.trapz(ROC_tensor, dx=1.0 / intervals).item()
    return ROC,AUC

def plot_roc(ROC_dict,fig_name, opt_label='opt',intervals = 20):
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    # plot opt
    ROC_opt = ROC_dict.pop(opt_label)
    plt.plot(quants, ROC_opt, marker="^", markersize=10, linewidth= 2,color='blue', label=opt_label)
    for est_label in ROC_dict.keys():
        if 'conf' in est_label:
            mark = "o"
            line = '--'
        else:
            mark = "P"
            line = '-.'
        plt.plot(quants, ROC_dict[est_label], marker=mark,linestyle=line, markersize=10,linewidth= 2, label=est_label)
    plt.xticks(quants)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative Cosine Similarity')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    fig.savefig(fig_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset root folder, e.g. .../Active_GenericGSDataset")
    parser.add_argument("--scene", type=str, default="",
                        help="Scene folder name. If empty, auto-pick the first valid scene.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[512, 256, 128, 64, 32, 16],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--radius_mult", type=float, default=3.0)
    parser.add_argument("--text_embeddings", type=str, default="")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=2000,
                        help="Number of GS to update per frame (simulate frustum subset).")
    parser.add_argument("--noise_std", type=float, default=0.05,
                        help="Gaussian noise added to sem_obs_subset for stress test.")

    parser.add_argument("--downsample", type=int, default=20000,
                        help="Downsample N for offline correctness test using cdist. Set 0 to disable.")
    parser.add_argument("--Kmax", type=int, default=64,
                        help="Max neighbors per dst (for test builder).")

    # parser.add_argument("--out_ply", type=str, default="conf_heatmap.ply",
    #                     help="Output PLY path for confidence heatmap point cloud.")

    args = parser.parse_args()
    seed_all(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float32

    root = args.data_dir
    assert os.path.isdir(root), f"Not a folder: {root}"

    scenes = list_scenes(root)
    assert len(scenes) > 0, f"No valid scenes found under: {root}"

    scene = args.scene.strip()
    if scene == "":
        scene = scenes[0]
    assert scene in scenes, f"Scene '{scene}' not found. Available: {scenes[:10]} ..."

    scene_dir = os.path.join(root, scene)
    feat_path = os.path.join(f"{root}_pred","result_GenericGSDataset", "feat", f"{scene}_feat.pth")
    assert os.path.exists(feat_path), f"Missing: {feat_path}"

    coord_path = os.path.join(scene_dir, "coord.npy")
    scale_path = os.path.join(scene_dir, "scale.npy")
    opacity_path = os.path.join(scene_dir, "opacity.npy")
    color_path = os.path.join(scene_dir, "color.npy")  # optional
    quat_path = os.path.join(scene_dir, "quat.npy")  # optional

    # Load arrays
    xyz_all = torch.from_numpy(np.load(coord_path)).float()         # (N,3)
    scale_all = torch.from_numpy(np.load(scale_path)).float()       # (N,1) or (N,3)
    opacity_all = torch.from_numpy(np.load(opacity_path)).float()   # (N,1)
    quat_all = torch.from_numpy(np.load(quat_path)).float()   # (N,1)
    if os.path.exists(color_path):
        color_all = np.load(color_path)

    feat_full = load_feat_pth(feat_path).float()                     # (N,D)

    N = xyz_all.shape[0]

    feat_full_dim = feat_full.shape[-1]
    model = Autoencoder(args.encoder_dims, args.decoder_dims, in_feature_dim=feat_full_dim).to(device)
    checkpoint = torch.load(args.embedding_model)
    model.load_state_dict(checkpoint)
    model.eval()

    feat_full = feat_full.to(device)
    feat_all = model.encode(feat_full)


    if feat_all.shape[0] != N:
        raise ValueError(f"Feature count mismatch: coord N={N}, feat N={feat_all.shape[0]}")
    if feat_all.ndim != 2:
        raise ValueError(f"Expected feat shape (N,D), got {tuple(feat_all.shape)}")
    if feat_all.shape[1] != args.latent_dim:
        raise ValueError(f"Expected feature dim {args.latent_dim}, got {feat_all.shape[1]}")

    scale_all = ensure_scale_shape(scale_all)
    if scale_all.shape[0] != N or opacity_all.shape[0] != N:
        raise ValueError("Scale/opacity N mismatch.")

    # Optional downsample for cdist-based test
    if args.downsample and args.downsample > 0 and args.downsample < N:
        idx = torch.randperm(N)[:args.downsample]
        xyz_all = xyz_all[idx]
        scale_all = scale_all[idx]
        opacity_all = opacity_all[idx]
        feat_all = feat_all[idx]
        N = xyz_all.shape[0]
        print(f"[Info] Downsampled to N={N} for cdist correctness test.")

    # Move to device
    xyz_all = xyz_all.to(device=device, dtype=dtype)
    scale_all = scale_all.to(device=device, dtype=dtype)
    opacity_all = opacity_all.to(device=device, dtype=dtype)
    feat_all = feat_all.to(device=device, dtype=dtype)

    # Import your BKI class (edit this line to your actual module name)

    bki = GSSubsetSemanticBKI(
        latent_dim=args.latent_dim,
        radius_mult=args.radius_mult,
        Kmax=args.Kmax,
        q_chunk=999999999,
        use_opacity_weight=True,
        device=device,
        dtype=dtype,
    ).to(device)

    # Monkey-patch: use cdist-based neighbor builder for offline correctness test
    def patched_builder(xyz_dst, r_dst, xyz_src, src_global_idx):
        return build_edges_cdist_small(xyz_dst, r_dst, xyz_src, src_global_idx, Kmax=args.Kmax)

    bki._build_edges_ball_query = patched_builder

    # Initialize state
    mean_all = torch.zeros((N, args.latent_dim), device=device, dtype=dtype)
    var_all  = torch.ones((N, args.latent_dim),  device=device, dtype=dtype)
    conf_all = torch.zeros((N, 1),               device=device, dtype=dtype)

    print(f"[Info] Scene={scene}, N={N}, frames={args.frames}, subset_size={args.subset_size}")
    batch = args.subset_size  # 2000
    for t in range(args.frames):
        t0 = time.time()

        perm = torch.randperm(N, device=device)

        # full update: cover all points once
        for s0 in range(0, N, batch):
            s1 = min(s0 + batch, N)
            subset_idx = perm[s0:s1]

            sem_obs_subset = feat_all[subset_idx].clone()
            if args.noise_std > 0:
                sem_obs_subset = sem_obs_subset + args.noise_std * torch.randn_like(sem_obs_subset)

            mean_all, var_all, conf_all = bki.update_subset(
                mean_all, var_all, conf_all,
                xyz_all, scale_all, opacity_all,
                subset_idx=subset_idx,
                sem_obs_subset=sem_obs_subset,
                use_all_as_sources=False,
            )

        # stats over ALL points after the full pass
        conf_stats = (conf_all.mean().item(), conf_all.min().item(), conf_all.max().item())
        cos = cosine_similarity(mean_all, feat_all)
        cos_stats = (cos.mean().item(), cos.min().item(), cos.max().item())

        dt = time.time() - t0
        print(f"[Epoch {t:03d}] time={dt:.3f}s | conf(mean/min/max)={conf_stats} | cos(mean/min/max)={cos_stats}")

    cos_all = cosine_similarity(mean_all, feat_all).mean().item()
    print(f"[Done] Mean cosine similarity over all GS: {cos_all:.4f}")

    xyz_np = xyz_all.detach().cpu().numpy()  # (N,3)
    conf_scalar = conf_all.detach().cpu().reshape(-1)  # (N,)
    var_scalar = var_all.detach().cpu().mean(-1)
    unc = var_scalar / (conf_scalar + 1e-6)

    # # Compute cosine similarity AUC
    torch.cuda.empty_cache()
    text_embeddings = torch.load(args.text_embeddings, weights_only=True)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1).to(device)
    text_encode = model.encode(text_embeddings).detach().cpu().numpy()

    # GT PCD
    gt_coord_path = os.path.join(scene_dir, "pc_coord.npy")  #
    gt_segment_path = os.path.join(scene_dir, "pc_segment.npy")  #
    gt_coord_all = np.load(gt_coord_path)  # (N,3)
    gt_segment_all = np.load(gt_segment_path)  # (N,1)
    thr = 0.05
    gen_points_kd_tree = KDTree(gt_coord_all)
    distances, idx = gen_points_kd_tree.query(xyz_np)
    gt_labels = gt_segment_all[idx]
    mask = (distances <= thr) & (gt_labels >= 0)
    filtered_gt = gt_labels[mask]
    gt_feat = text_encode[filtered_gt]
    pred_feat = feat_all.detach().cpu().numpy()[mask]
    cosine_score = cosine_similarity(torch.from_numpy(gt_feat), torch.from_numpy(pred_feat)).cpu().numpy()
    filtered_unc = unc[mask]
    cosine_score = torch.from_numpy(cosine_score)

    # opt,opt_auc = compute_roc(cosine_score,cosine_score)
    # est,est_auc = compute_roc(cosine_score,filtered_unc)
    #
    # ROC_dict = {'opt':opt,
    #             'unc':est}
    save_dir = os.path.join(f"{root}_pred", "unc", scene)
    os.makedirs(save_dir, exist_ok=True)
    # fig_name = os.path.join(save_dir, "roc_plot.png")
    # plot_roc(ROC_dict, fig_name, opt_label='opt', intervals=20)


    ###### save outputs #########################
    # Save cosine similarity to heatmap coloring
    breakpoint()
    filtered_xyz = xyz_np[mask]
    cosine_score = cosine_score.cpu().numpy()
    lo = np.quantile(cosine_score, 0.01)
    hi = np.quantile(cosine_score, 0.99)
    rgb = cosine_to_rgb_bwr(cosine_score)
    save_ply = os.path.join(save_dir, "cosine_map.ply")
    save_ply_xyz_rgb(save_ply, filtered_xyz, rgb)
    print(f"[Saved] {save_ply} | cosine similarity 1%/99% = {lo:.6f}/{hi:.6f}")

    # Normalize confidence to [0,1] for heatmap coloring
    lo = torch.quantile(conf_scalar, 0.01)
    hi = torch.quantile(conf_scalar, 0.99)
    conf = torch.clamp(conf_scalar, lo, hi)

    conf01 = (conf - lo) / (hi - lo + 1e-12)
    conf01 = conf01.numpy().reshape(-1)

    rgb = scalar_to_rgb(conf01)  # (N,3) uint8
    save_ply = os.path.join(save_dir, "conf_heatmap.ply")
    save_ply_xyz_rgb(save_ply, xyz_np, rgb)
    print(f"[Saved] {save_ply} | observation mass conf 1%/99% = {lo:.6f}/{hi:.6f}")




    # Normalize confidence to [0,1] for heatmap coloring
    lo = torch.quantile(conf_scalar, 0.01)
    hi = torch.quantile(conf_scalar, 0.99)
    conf = torch.clamp(conf_scalar, lo, hi)

    conf01 = (conf - lo) / (hi - lo + 1e-12)
    conf01 = conf01.numpy().reshape(-1)

    rgb = scalar_to_rgb(conf01)  # (N,3) uint8
    save_ply = os.path.join(save_dir, "conf_heatmap.ply")
    save_ply_xyz_rgb(save_ply, xyz_np, rgb)
    print(f"[Saved] {save_ply} | observation mass conf 1%/99% = {lo:.6f}/{hi:.6f}")

    # Normalize uncertainty to [0,1] for heatmap coloring

    lo = torch.quantile(unc, 0.01)
    hi = torch.quantile(unc, 0.99)
    unc = torch.clamp(unc, lo, hi)

    unc01 = (unc - lo) / (hi - lo + 1e-12)
    unc01 = unc01.numpy().reshape(-1)

    rgb = scalar_to_rgb(unc01)  # (N,3) uint8
    save_ply = os.path.join(save_dir, "unc_heatmap.ply")
    save_ply_xyz_rgb(save_ply, xyz_np, rgb)
    print(f"[Saved] {save_ply} | uncertainty 1%/99% = {lo:.6f}/{hi:.6f}")


if __name__ == "__main__":
    main()
