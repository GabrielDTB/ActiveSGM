'''
refer to https://github.com/UMich-CURLY/LatentBKI/blob/main/Models/LatentBKI.py
'''

import torch
from pytorch3d.ops import ball_query

class GSSubsetSemanticBKI(torch.nn.Module):
    def __init__(
        self,
        latent_dim=16,
        radius_mult=3.0,      # Use ell = radius_mult * scale (scale is GS radius)
        kernel="sparse",
        sigma=1.0,
        eps=1e-6,
        Kmax=64,              # Max neighbors returned per query (keep some headroom above 20–30)
        q_chunk=50000,        # Number of query (dst) points per chunk to control memory
        use_opacity_weight=True,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        self.D = latent_dim
        self.radius_mult = float(radius_mult)
        self.kernel = kernel
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.Kmax = int(Kmax)
        self.q_chunk = int(q_chunk)
        self.use_opacity_weight = use_opacity_weight
        self.device = device
        self.dtype = dtype
        self.pi = torch.acos(torch.zeros(1, device=device, dtype=dtype)).item() * 2

    # Sparse / compact-support kernel on distance d with support radius ell (per edge).
    def sparse_kernel(self, d, ell):
        # Force shapes to (E,1)
        if d.dim() == 1:
            d = d.view(-1, 1)
        if ell.dim() == 1:
            ell = ell.view(-1, 1)

        ell = torch.clamp(ell, min=self.eps)
        x = 2.0 * self.pi * d / ell

        kernel_val = self.sigma * (
                (1.0 / 3.0) * (2.0 + torch.cos(x)) * (1.0 - d / ell)
                + (1.0 / (2.0 * self.pi)) * torch.sin(x)
        )
        kernel_val = torch.where(d >= ell, torch.zeros_like(kernel_val), kernel_val)
        return torch.clamp(kernel_val, 0.0, 1.0)

    @torch.no_grad()
    def build_edges_ball_query(
            self,
            xyz_dst,  # (Q,3) query points (dst)
            r_dst,  # (Q,1) per-dst radius (ell)
            xyz_src,  # (M,3) source points to search in
            src_global_idx,  # (M,) mapping from src-local index -> global GS index
    ):
        """
        Build a flattened edge list from dst queries to nearby src points.

        Returns (all flattened):
          dst_local:  (E,)   indices in [0, Q)
          src_global: (E,)   global indices in [0, N)
          dist:       (E,1)  Euclidean distance
          ell:        (E,1)  per-edge support radius (here we store r_dst[dst_local] or fallback max)

        Guarantee:
          - Every dst point in [0, Q) appears at least once in dst_local.
          - If a dst point finds no neighbor within its threshold, we append ONE fallback edge for it,
            using "largest edge values" (dist=max_val, ell=max_val).
        """
        device = xyz_dst.device
        dtype_xyz = xyz_dst.dtype
        dtype_r = r_dst.dtype

        Q = int(xyz_dst.shape[0])
        M = int(xyz_src.shape[0])

        dst_local_out = torch.arange(Q, device=device, dtype=torch.long)

        # largest edge value (penalty)
        max_val = float(r_dst.max().item()) if Q > 0 else 0.0

        # If no src points exist, every dst gets a fallback edge
        if M == 0:
            src_global_out = torch.full((Q,), -1, device=device, dtype=torch.long)
            dist_out = torch.full((Q, 1), max_val, device=device, dtype=dtype_xyz)
            ell_out = torch.full((Q, 1), max_val, device=device, dtype=dtype_r)
            return dst_local_out, src_global_out, dist_out, ell_out

        p2 = xyz_src[None]  # (1,M,3)
        lengths2 = torch.tensor([M], device=device, dtype=torch.int64)

        # outputs to fill
        src_global_out = torch.empty((Q,), device=device, dtype=torch.long)
        dist_out = torch.empty((Q, 1), device=device, dtype=dtype_xyz)
        ell_out = torch.empty((Q, 1), device=device, dtype=dtype_r)

        fallback_src = int(src_global_idx[0].item())

        for s0 in range(0, Q, self.q_chunk):
            s1 = min(s0 + self.q_chunk, Q)
            q = s1 - s0

            p1 = xyz_dst[s0:s1][None]  # (1,q,3)
            lengths1 = torch.tensor([q], device=device, dtype=torch.int64)

            r_chunk = r_dst[s0:s1]  # (q,1)
            r_max = float(r_chunk.max().item())  # scalar radius for ball_query

            out = ball_query(
                p1, p2,
                lengths1=lengths1, lengths2=lengths2,
                K=self.Kmax, radius=r_max,
                return_nn=False,
                # skip_points_outside_cube=True,
            )

            idx = out.idx[0]  # (q,K) padded with -1
            d2 = out.dists[0]  # (q,K) squared distances

            valid = idx >= 0
            d = torch.sqrt(torch.clamp(d2, min=0.0))  # (q,K)

            # Per-point radius filtering (broadcast r_chunk (q,1) -> (q,K))
            within = valid & (d < r_chunk)

            # For each dst in this chunk, pick nearest "within" neighbor if exists
            d_inf = d.clone()
            d_inf[~within] = float("inf")  # only within candidates remain
            min_d, argmin_k = d_inf.min(dim=1)  # (q,), (q,)

            has = torch.isfinite(min_d)  # (q,)

            # default fallback
            src_global_chunk = torch.full((q,), fallback_src, device=device, dtype=torch.long)
            dist_chunk = torch.full((q, 1), max_val, device=device, dtype=dtype_xyz)
            ell_chunk = torch.full((q, 1), max_val, device=device, dtype=dtype_r)

            if has.any():
                rows = torch.nonzero(has, as_tuple=False).squeeze(1)  # (qh,)
                cols = argmin_k[rows]  # (qh,)

                src_local = idx[rows, cols].to(torch.long)  # (qh,)
                src_global_chunk[rows] = src_global_idx[src_local]  # (qh,)
                dist_chunk[rows, 0] = min_d[rows].to(dtype_xyz)  # (qh,)
                # if you prefer per-point ell, replace next line with r_chunk[rows,0]
                ell_chunk[rows, 0] = r_chunk[rows, 0].to(dtype_r)  # (qh,)

            src_global_out[s0:s1] = src_global_chunk
            dist_out[s0:s1] = dist_chunk
            ell_out[s0:s1] = ell_chunk

        return dst_local_out, src_global_out, dist_out, ell_out

    def update_subset(
        self,
        mean_all, var_all, conf_all,        # (N,16), (N,16), (N,1)
        xyz_all, scale_all, opacity_all,    # (N,3),  (N,1),  (N,1)
        subset_idx,                         # (Q,) long, indices of GS to update this frame
        sem_obs_subset,                     # (Q,16), per-frame semantic observation for subset
        use_all_as_sources=True,            # If True: neighbor search over all N; else only within subset
    ):
        """
        Online BKI update only for a subset of GS indices (e.g., within current view frustum).

        - dst nodes: subset_idx
        - src nodes: all GS (if use_all_as_sources=True) or subset only (faster)
        - per-dst support radius: ell_dst = radius_mult * scale_dst
        - weights: w = kernel(dist, ell_dst) * opacity_src (optional)
        """
        device = self.device
        dtype = self.dtype

        subset_idx = subset_idx.to(device=device, dtype=torch.long)
        sem_obs_subset = sem_obs_subset.to(device=device, dtype=dtype)

        xyz_all = xyz_all.to(device=device, dtype=dtype)
        scale_all = scale_all.to(device=device, dtype=dtype)
        opacity_all = opacity_all.to(device=device, dtype=dtype)

        mean_all = mean_all.to(device=device, dtype=dtype)
        var_all  = var_all.to(device=device, dtype=dtype)
        conf_all = conf_all.to(device=device, dtype=dtype)

        # Destination set (nodes to be updated this frame)
        xyz_dst = xyz_all[subset_idx]                        # (Q,3)
        #r_dst = self.radius_mult * scale_all[subset_idx]     # (Q,1)
        r_dst = torch.full((subset_idx.shape[0], 1), 0.05, device=device, dtype=dtype)  # (Q,1)

        # Source set (nodes providing semantic observations)
        if use_all_as_sources:
            xyz_src = xyz_all
            src_global_idx = torch.arange(xyz_all.shape[0], device=device, dtype=torch.long)

            # Observation features for src:
            # - for visible subset: use per-frame observation sem_obs_subset
            # - for non-visible: fall back to current mean (acts like a prior / smoother)
            obs_feat_all = mean_all.clone()
            obs_feat_all[subset_idx] = sem_obs_subset
        else:
            xyz_src = xyz_dst
            src_global_idx = subset_idx
            obs_feat_all = mean_all.clone()
            obs_feat_all[subset_idx] = sem_obs_subset

        # Build edge list (dst_local in [0,Q), src_global in [0,N))
        dst_local, src_g, dist, ell = self.build_edges_ball_query(
            xyz_dst=xyz_dst, r_dst=r_dst,
            xyz_src=xyz_src, src_global_idx=src_global_idx
        )
        if dst_local is None:
            return mean_all, var_all, conf_all

        # Compute weights
        w = self.sparse_kernel(dist, ell)  # (E,1)

        if self.use_opacity_weight:
            w = w * torch.clamp(opacity_all[src_g], 0.0, 1.0).view(-1,1)
        Q = subset_idx.shape[0]
        D = self.D

        # Accumulate k_bar and y_sum on the subset-local dst array
        k_bar = torch.zeros((Q, 1), device=device, dtype=dtype)
        y_sum = torch.zeros((Q, D), device=device, dtype=dtype)

        k_bar.index_add_(0, dst_local, w)
        y_sum.index_add_(0, dst_local, w * obs_feat_all[src_g])

        y_bar = y_sum / (k_bar + self.eps)  # (Q,D)

        # Select updated dst nodes (those that received any mass)
        upd_mask = (k_bar[:, 0] > 0)
        if not upd_mask.any():
            return mean_all, var_all, conf_all

        upd_local = torch.where(upd_mask)[0]
        upd_global = subset_idx[upd_local]

        # S_bar: within-frame scatter term accumulated on dst
        delta = obs_feat_all[src_g] - y_bar[dst_local]    # (E,D)
        S_sum = torch.zeros((Q, D), device=device, dtype=dtype)
        S_sum.index_add_(0, dst_local, w * (delta * delta))

        # E_bar: mean-shift term for updated nodes
        mean_old = mean_all[upd_global]                   # (K,D)
        conf_old = conf_all[upd_global]                   # (K,1)
        var_old  = var_all[upd_global]                    # (K,D)

        y_bar_u = y_bar[upd_local]                        # (K,D)
        k_u = k_bar[upd_local]                            # (K,1)
        S_u = S_sum[upd_local]                            # (K,D)

        delta_mean = y_bar_u - mean_old
        E_bar = delta_mean * delta_mean

        scaling = (conf_old * k_u) / (conf_old + k_u + self.eps)

        # Updated states for subset nodes
        var_new  = var_old + S_u + scaling * E_bar
        mean_new = (conf_old * mean_old + k_u * y_bar_u) / (conf_old + k_u + self.eps)
        conf_new = conf_old + k_u

        # Write back to global tensors
        mean_all = mean_all.clone()
        var_all  = var_all.clone()
        conf_all = conf_all.clone()

        mean_all[upd_global] = mean_new
        var_all[upd_global]  = var_new
        conf_all[upd_global] = conf_new

        return mean_all, var_all, conf_all

    def uncertainty(self, var, conf):
        # Normalized uncertainty proxy (consistent with the "accumulated second-moment" style update)
        return var / (conf + self.eps)

