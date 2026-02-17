"""
Script to preprocess Holicity dataset: extract point cloud data and remapped segments per scene.

For each scene listed in train.txt, val.txt, or test.txt, this script:
 - Reads `points3d.ply` using Open3D (points, colors, normals)
 - Saves coords.npy, color.npy, normal.npy in output/<split>/<scene>/
 - Loads segment.npy, subtracts 1 from all labels, then remaps any label == 4 to -1, and saves.

Usage:
    python preprocess_holicity.py \
        --input_root /home/yli7/scratch/datasets/holicity/perspective/collected_by_region \
        --split_dir /home/yli7/scratch/datasets/holicity/splits/ours \
        --output_root /home/yli7/scratch/datasets/ptv3_preprocessed/holicity
"""

import os
import argparse
import logging
import numpy as np
import open3d as o3d
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Holicity scenes into numpy arrays"
    )
    parser.add_argument(
        "--input_root", required=True, help="Root folder containing raw Holicity scenes"
    )
    parser.add_argument(
        "--split_dir",
        default="/home/yli7/scratch/datasets/holicity/splits/ours",
        help="Folder containing train.txt, val.txt, test.txt",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root folder where preprocessed scenes will be saved",
    )
    parser.add_argument(
        "--train_file", default="train.txt", help="Filename for train split list"
    )
    parser.add_argument(
        "--val_file", default="val.txt", help="Filename for val split list"
    )
    parser.add_argument(
        "--test_file", default="test.txt", help="Filename for test split list"
    )
    return parser.parse_args()


def read_split_file(path):
    """
    Reads a split file listing one scene name per line.
    Skips blank lines and comments (#).
    """
    scenes = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            scenes.append(name)
    return scenes


def process_scene(scene, split, args):
    """
    Process a single scene: read PLY and segment, save numpy files.
    """
    src_dir = os.path.join(args.input_root, scene)
    if not os.path.isdir(src_dir):
        logging.warning(f"Scene folder not found: {src_dir}")
        return

    dst_dir = os.path.join(args.output_root, split, scene)
    os.makedirs(dst_dir, exist_ok=True)

    # Load PLY
    ply_path = os.path.join(src_dir, "points3d.ply")
    if not os.path.isfile(ply_path):
        logging.error(f"points3d.ply missing in {src_dir}")
        return

    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    norms = np.asarray(pcd.normals, dtype=np.float32)

    np.save(os.path.join(dst_dir, "coord.npy"), pts)
    np.save(os.path.join(dst_dir, "color.npy"), cols)
    np.save(os.path.join(dst_dir, "normal.npy"), norms)

    # Process segment labels
    seg_src = os.path.join(src_dir, "segment.npy")
    if os.path.isfile(seg_src):
        seg = np.load(seg_src).astype(np.int32)
        # Subtract 1 from all labels, we use -1 as ignore label
        seg -= 1
        # Remap any label value == 4 to -1, (others)
        seg[seg == 4] = -1
        np.save(os.path.join(dst_dir, "segment.npy"), seg)
    else:
        logging.warning(f"segment.npy missing in {src_dir}")

    # shape check
    assert pts.shape[0] == cols.shape[0] == norms.shape[0] == seg.shape[0], (
        f"Shape mismatch in {scene}: {pts.shape}, {cols.shape}, {norms.shape}, {seg.shape}"
    )

    # logging.info(f"Processed scene {scene} into {dst_dir}.")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Read splits
    train_list = os.path.join(args.split_dir, args.train_file)
    val_list = os.path.join(args.split_dir, args.val_file)
    test_list = os.path.join(args.split_dir, args.test_file)

    train_scenes = read_split_file(train_list)
    val_scenes = read_split_file(val_list)
    test_scenes = read_split_file(test_list)

    logging.info(
        f"Found {len(train_scenes)} train, {len(val_scenes)} val, {len(test_scenes)} test scenes."
    )

    # Process each split
    for scene in tqdm(sorted(val_scenes)):
        process_scene(scene, "val", args)
    for scene in tqdm(sorted(test_scenes)):
        process_scene(scene, "test", args)
    for scene in tqdm(sorted(train_scenes)):
        process_scene(scene, "train", args)


if __name__ == "__main__":
    main()
