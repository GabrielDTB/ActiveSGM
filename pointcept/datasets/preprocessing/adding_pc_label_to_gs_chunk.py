#!/usr/bin/env python
# ────────────────────────────────────────────────────────────
#  adding_pc_label_to_gs_chunk.py
#
#  1)  For    train / val / test      scene‑level 3D‑GS folders:
#        → copies   coord.npy  +  all segment*.npy
#          from the original (non‑GS) scene folder,
#          saving them as        pc_coord.npy  /  pc_segment*.npy
#
#  2)  For every  *chunk*  folder under train_grid*/val_grid*/test_grid*:
#        → builds / re‑uses a KD‑tree of the *full* scene point cloud,
#          finds the unique nearest‑neighbour indices (k = K_NEIGHBORS)
#          of the chunk’s 3D‑GS centroids,
#          slices coord + segment arrays,
#          writes             pc_coord.npy  /  pc_segment*.npy
#
#  One KD‑tree per scene is built **once** and re‑used for all its chunks.
#  Written with tens of thousands of chunks in mind (≈ minutes, not hours).
# ────────────────────────────────────────────────────────────

from __future__ import annotations
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# CONFIGURATION  (edit in one place, no CLI arguments needed)
# ────────────────────────────────────────────────────────────

K_NEIGHBORS = 16  # ≥1; tweak as you like
LOAD_INTO_MEMORY_GB = 10  # when a scene’s coord.npy is > this, fall back to mmap

DATASETS: Dict[str, Dict[str, str]] = {
    # dataset_key              gaussian_root                               original_root
    # "scannet_mcmc_3dgs":       "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannet_mcmc_3dgs",
    # "scannetpp_v2_mcmc_3dgs":  "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs",
    "matterport3d_region_mcmc_3dgs": "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs",
    # "holicity_mcmc_3dgs":      "/home/yli7/scratch2/datasets/gaussian_world/preprocessed/holicity_mcmc_3dgs",
}

ORIGINAL_ROOTS: Dict[str, str] = {
    # "scannet_mcmc_3dgs":       "/home/yli7/scratch2/datasets/ptv3_preprocessed/scannet_preprocessed",
    # "scannetpp_v2_mcmc_3dgs":  "/home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed",
    "matterport3d_region_mcmc_3dgs": "/home/yli7/scratch2/datasets/ptv3_preprocessed/matterport3d",
    # "holicity_mcmc_3dgs":      "/home/yli7/scratch/datasets/ptv3_preprocessed/holicity",
}

SCENE_SPLITS = ("train", "val", "test")  # folders at root level without “chunk”

# ────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────


def is_chunk_subdir(name: str) -> bool:
    """Return True if a directory name contains a chunked grid."""
    return "chunk" in name and "filtered" not in name


def split_from_subdir(subdir: str) -> str:
    """Infer split (train/val/test) from subdir name."""
    for s in SCENE_SPLITS:
        if subdir.startswith(s):
            return s
    raise ValueError(f"Cannot infer split from subdir: {subdir}")


def scene_and_chunk(dir_name: str) -> Tuple[str, str]:
    """
    Extract (scene_name, chunk_id) from   <scene_name>_<chunkId>.
    Scene names themselves may contain underscores, so rsplit once.
    """
    scene, chunk = dir_name.rsplit("_", 1)
    return scene, chunk


def load_coords(path: Path) -> np.ndarray:
    """
    Memory‑map large coord.npy files to save RAM.
    """
    byte_size = path.stat().st_size
    if byte_size > LOAD_INTO_MEMORY_GB * (1 << 30):
        return np.load(path, mmap_mode="r")
    return np.load(path)


# ────────────────────────────────────────────────────────────
# CORE PROCESSING
# ────────────────────────────────────────────────────────────


class SceneCache:
    """
    Keeps one scene’s data (coords, segments, KD‑tree) in memory,
    drops it automatically when we move to a different scene.
    """

    def __init__(self) -> None:
        self.current_scene: str | None = None
        self.split: str | None = None
        self.coords: np.ndarray | None = None
        self.segments: Dict[str, np.ndarray] = {}
        self.kdtree: cKDTree | None = None

    def load(self, scene_dir: Path) -> None:
        if self.current_scene == scene_dir.name:
            return  # already loaded
        # clear previous
        self.coords = None
        self.segments.clear()
        self.kdtree = None

        # load
        coord_path = scene_dir / "coord.npy"
        if not coord_path.exists():
            raise FileNotFoundError(coord_path)
        self.coords = load_coords(coord_path)
        self.kdtree = cKDTree(self.coords)

        for seg_path in scene_dir.glob("segment*.npy"):
            self.segments[seg_path.name] = np.load(seg_path, mmap_mode="r")

        self.current_scene = scene_dir.name

    # queried by chunk gs coord
    def slice(
        self,
        chunk_xyz: np.ndarray,
        dist_limit: float = 0.25,  # metres
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Return point‑cloud coords & segment labels that are *both*
        • among the k nearest neighbours of chunk_xyz  (k = K_NEIGHBORS)
        • within L2 distance ≤ dist_limit
        """
        dists, idx = self.kdtree.query(chunk_xyz, k=K_NEIGHBORS, workers=-1)

        # keep only neighbours whose distance ≤ dist_limit
        mask = dists <= dist_limit
        idx_valid = idx[mask]  # 2‑D array collapses to 1‑D
        if idx_valid.size == 0:  # no close neighbours at all
            return np.empty((0, 3), np.float32), {
                n: np.empty((0,), s.dtype) for n, s in self.segments.items()
            }

        idx_unique = np.unique(idx_valid)
        pc_coord = self.coords[idx_unique]
        pc_seg = {name: seg[idx_unique] for name, seg in self.segments.items()}
        return pc_coord, pc_seg

    # queried by chunk gs coord
    def semseg_label_slice(
        self,
        chunk_xyz: np.ndarray,
        dist_limit: float = 0.25,  # metres
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Return point‑cloud coords & segment labels that are *both*
        • among the k nearest neighbours of chunk_xyz  (k = K_NEIGHBORS)
        • within L2 distance ≤ dist_limit
        """
        dists, idx = self.kdtree.query(chunk_xyz, k=1, workers=-1)

        # return segment_nyu_160 from using the nn idx
        # if dists too large, set seg to -1
        valid_mask = dists <= dist_limit
        gs_seg_label = {name: seg[idx] for name, seg in self.segments.items()}
        for name, seg in gs_seg_label.items():
            seg[~valid_mask] = -1
        return gs_seg_label


def copy_scene_level(gs_root: Path, pc_root: Path) -> None:
    """
    Step 1  – simple copy of pc_* files for full‑scene 3D‑GS folders.
    """
    for split in ("val", "test"):
        gs_split_dir = gs_root / split
        pc_split_dir = pc_root / split
        if not gs_split_dir.exists():
            continue
        for scene_dir in tqdm(
            list(gs_split_dir.iterdir()), desc=f"[{gs_root.name}] {split} scenes"
        ):
            if not scene_dir.is_dir():
                continue
            src_scene_dir = pc_split_dir / scene_dir.name
            if not src_scene_dir.exists():
                print(f"⚠️  Original scene missing: {src_scene_dir}", file=sys.stderr)
                continue
            # copy coord
            dst_coord = scene_dir / "pc_coord.npy"
            if not dst_coord.exists():
                shutil.copy2(src_scene_dir / "coord.npy", dst_coord)
            # copy every segment*.npy
            for seg_path in src_scene_dir.glob("segment*.npy"):
                dst_seg = scene_dir / f"pc_{seg_path.name}"
                if not dst_seg.exists():
                    shutil.copy2(seg_path, dst_seg)
            # copy instance.npy if it exists
            src_instance = src_scene_dir / "instance.npy"
            if src_instance.exists():
                dst_instance = scene_dir / "pc_instance.npy"
                shutil.copy2(src_instance, dst_instance)


def build_chunks(gs_root: Path, pc_root: Path, write_semseg_label=False) -> None:
    """
    Step 2  – KD‑tree slicing for every chunked grid folder.
    """
    cache = SceneCache()

    # iterate over chunked subdirs (train_grid*, val_grid*, …)
    for subdir in sorted(
        d for d in gs_root.iterdir() if d.is_dir() and is_chunk_subdir(d.name)
    ):
        split = split_from_subdir(subdir.name)
        pc_split_dir = pc_root / split

        all_chunks: List[Path] = [p for p in subdir.iterdir() if p.is_dir()]
        desc = f"[{gs_root.name}] {subdir.name}"
        for chunk_dir in tqdm(all_chunks, desc=desc):
            scene_name, _ = scene_and_chunk(chunk_dir.name)
            src_scene_dir = pc_split_dir / scene_name
            if not src_scene_dir.exists():
                print(f"⚠️  Original scene missing: {src_scene_dir}", file=sys.stderr)
                continue

            # skip if already processed
            # if (chunk_dir / "pc_coord.npy").exists():
            #     continue

            # ensure scene cached
            cache.load(src_scene_dir)

            # load chunk gs coordinates
            chunk_xyz = np.load(chunk_dir / "coord.npy")

            # slice + save
            pc_coord, pc_segments = cache.slice(chunk_xyz)
            np.save(chunk_dir / "pc_coord.npy", pc_coord)
            for name, arr in pc_segments.items():
                np.save(chunk_dir / f"pc_{name}", arr)

            # semseg_label slice + save, used to update the gs segment labels
            if write_semseg_label:
                pc_seg_label = cache.semseg_label_slice(chunk_xyz)
                for name, arr in pc_seg_label.items():
                    if "nyu" in name:
                        np.save(chunk_dir / f"{name}", arr)
                        print(f"Saved {name} to {chunk_dir / f'{name}'}")


def main() -> None:
    for key, gs_root_str in DATASETS.items():
        gs_root = Path(gs_root_str)
        pc_root = Path(ORIGINAL_ROOTS[key])
        if not gs_root.exists() or not pc_root.exists():
            print(f"❌ Paths for “{key}” not found – skipping.", file=sys.stderr)
            continue
        print(f"════════════════════════════════════════\n🗂  DATASET: {key}")
        # copy_scene_level(gs_root, pc_root)
        build_chunks(gs_root, pc_root, write_semseg_label=True)

        print(f"✅  Finished processing {key}.\n")
    print("All done! 🎉\n")


if __name__ == "__main__":
    main()
