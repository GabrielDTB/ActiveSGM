import argparse
import numpy as np
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from typing import Dict

################################################################################
# I/O Utilities
################################################################################


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".ply"]:
            return cls._read_ply(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)


################################################################################
# Gaussian Reading
################################################################################


def transform_points_numpy(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to a set of 3D points in NumPy.

    Args:
        points: (N, 3) numpy array of 3D points
        transform: (4, 4) transformation matrix

    Returns:
        Transformed points: (N, 3) numpy array
    """
    assert points.shape[-1] == 3, "Input points should have shape (N, 3)"
    assert transform.shape == (4, 4), "Transform must be of shape (4, 4)"

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_hom = np.concatenate([points, ones], axis=-1)  # (N, 4)

    # Apply transform
    transformed = (transform @ points_hom.T).T  # (N, 4)

    return transformed[:, :3]



def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attributes(vertex):
    """
    Reads a 'vertex' structure from a PlyData file and returns a dictionary
    with 'coord', 'opacity', 'scale', 'quat', 'color'.
    """
    data = {}

    # Coordinates (xyz)
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    data["coord"] = np.stack((x, y, z), axis=-1)  # [N, 3]

    # Opacity
    opacity = vertex["opacity"].astype(np.float32)
    opacity = np_sigmoid(opacity)  # range (0,1)
    data["opacity"] = opacity

    # Scale
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((data["coord"].shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = vertex[attr_name].astype(np.float32)
    scales = np.exp(scales)  # exponentiate to get actual scale
    data["scale"] = scales

    # Quaternion
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((data["coord"].shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = vertex[attr_name].astype(np.float32)

    rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
    # Enforce positive real part
    signs_vector = np.sign(rots[:, 0])
    rots = rots * signs_vector[:, None]
    data["quat"] = rots

    # Color (from spherical harmonics DC term)
    features_dc = np.zeros((data["coord"].shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
    features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
    features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

    feature_pc = features_dc.reshape(-1, 3)
    # Move from SH DC to approximate color
    C0 = 0.28209479177387814
    feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
    feature_pc = np.clip(feature_pc, 0, 1)
    data["color"] = (feature_pc * 255).astype(np.uint8)

    return data


################################################################################
# Main Processing Function
################################################################################


def process_ply_file(ply_path, output_dir, transform=None):
    """
    Process a single PLY file and save its parameters as separate NPY files.

    Args:
        ply_path: Path to the input PLY file
        output_dir: Directory where the NPY files will be saved
    """
    print(f"Processing: {ply_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the PLY file
    try:
        ply_data = IO.get(ply_path)
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        return False

    # Extract Gaussian attributes
    vertex = ply_data["vertex"]
    gs_data = read_gaussian_attributes(vertex)

    if transform is not None:
        gs_data["coord"] = transform_points_numpy(gs_data["coord"], transform)

    # Save individual parameters
    np.save(output_dir / "coord.npy", gs_data["coord"])
    np.save(output_dir / "color.npy", gs_data["color"])
    np.save(output_dir / "opacity.npy", gs_data["opacity"])
    np.save(output_dir / "scale.npy", gs_data["scale"])
    np.save(output_dir / "quat.npy", gs_data["quat"])

    print(f"Saved parameters to: {output_dir}")
    print(f"  - Number of Gaussians: {len(gs_data['coord'])}")

    return True

def load_scene_params(params_path: str) -> Dict:
    """ load SplaTAM checkpoint parameters
    """
    params = dict(np.load(params_path, allow_pickle=True))
    # params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(False) for k in params.keys()}
    return params
def write_ply_file(rgbs, opacities, params, ply_savepath):
    means = params['means3D']
    rotations = params['unnorm_rotations']
    scales = params['log_scales']
    normals = np.zeros_like(means)
    C0 = 0.28209479177387814
    colors = (rgbs - 0.5) / C0
    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))
    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', ]
    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_savepath)

def process_npz_file(npz_path, output_dir, transform=None):
    params = load_scene_params(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    ply_name = f"pretrained_gs.ply"
    ply_savepath = os.path.join(output_dir, ply_name)
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    if transform is not None:
        params["means3D"] = transform_points_numpy(params["means3D"], transform)

    write_ply_file(rgbs, opacities, params, ply_savepath)
    return ply_savepath






################################################################################
# Main script
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 3D Gaussian Splatting PLY files to NPY parameter files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input PLY/NPZ file or directory containing PLY/NPZ files",
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="if need to load in the transform matrix",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for NPY files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process PLY/NPZ files in subdirectories",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Collect Params files to process
    params_files = []

    if input_path.is_file() and input_path.suffix == ".ply":
        # Single file mode
        params_files = [input_path]
    elif input_path.is_file() and input_path.suffix == ".npz":
        params_files = [input_path]
    elif input_path.is_dir():
        if args.transform:
            T_slam2sim = np.loadtxt(os.path.join(input_path, "T_slam2sim.txt"))
        else:
            T_slam2sim = np.eye(4)
        # Directory mode
        if args.recursive:
            params_files = list(input_path.rglob("*.ply")) + list(input_path.rglob("*.npz"))
        else:
            params_files = list(input_path.glob("*.ply")) + list(input_path.glob("*.npz"))
    else:
        raise ValueError(
            f"Input path {input_path} is not a valid PLY/NPZ file or directory"
        )

    if not params_files:
        print("No Params files found!")
        exit(1)

    print(f"Found {len(params_files)} params file(s) to process")

    # Process each PLY file
    for params_file in tqdm(params_files):
        if input_path.is_dir():
            # Maintain relative directory structure
            relative_path = params_file.relative_to(input_path)
            output_dir = output_path / relative_path.parent / relative_path.stem
        else:
            # Single file - use output path directly
            output_dir = output_path

        if params_file.suffix == ".ply":
            success = process_ply_file(params_file, output_dir, T_slam2sim)
        elif params_file.suffix == ".npz":
            ply_file = process_npz_file(params_file, output_dir,T_slam2sim)
            success = process_ply_file(ply_file, output_dir)
        if not success:
            print(f"Failed to process: {params_file}")

    print("\nProcessing complete!")

################################################################################
# Example usage:
#
# Single file:
#   python preprocess_gs.py --input scene.ply --output output_dir/
#
# Directory of PLY files:
#   python preprocess_gs.py --input gaussians_dir/ --output output_dir/
#
################################################################################
