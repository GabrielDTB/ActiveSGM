"""
Script to preprocess KITTI-360 dataset: extract point cloud data and remapped segments per scene.

For each scene listed in train.txt, val.txt, or test.txt, this script:
 - Reads `points3d.ply` using Open3D (points, colors, normals)
 - Saves coords.npy, color.npy, normal.npy in output/<split>/<scene>/
 - Loads segment.npy, subtracts 1 from all labels, then remaps any label == 4 to -1, and saves.

"""

import os
import argparse
import logging
import numpy as np
# import open3d as o3d
from tqdm import tqdm
from labels import Label, label2kittiId
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from plyfile import PlyData
from sklearn.neighbors import KDTree
from tqdm import tqdm
import open3d as o3d


# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        elif file_extension in [".ply"]:
            return cls._read_ply(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)


################################################################################
# Gaussian Reading
################################################################################


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attribute(
    vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
):
    """
    Reads a 'vertex' structure from a PlyData file and returns a dictionary
    with 'coord', 'opacity', 'scale', 'quat', 'color' (all optional).
    """
    data = {}

    # Coordinates (xyz)
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    data["coord"] = np.stack((x, y, z), axis=-1)  # [N, 3]

    # Opacity
    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity)  # range (0,1)
        data["opacity"] = opacity

    # Scale & Quaternion
    if "scale" in attribute and ("quat" in attribute or "euler" in attribute):
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data["coord"].shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        scales = np.exp(scales)  # exponentiate to get actual scale
        data["scale"] = scales

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        # Normalize the quaternion
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # Enforce positive real part
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots

    # Color (from spherical harmonics DC term) or direct color
    if "sh" in attribute or "color" in attribute:
        # DC terms
        features_dc = np.zeros((data["coord"].shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        # Move from SH DC to approximate color
        C0 = 0.28209479177387814  # Spherical Harmonics Y00
        feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
        feature_pc = np.clip(feature_pc, 0, 1)
        # Store 0-255
        data["color"] = (feature_pc * 255).astype(np.uint8)

    return data


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties

def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Holicity scenes into numpy arrays"
    )
    parser.add_argument(
        "--gs_dir", default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_subset/2013_05_28_drive_0000_sync"
    )
    parser.add_argument(
        "--pc_dir", default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_preprocessed_pc/"
    )
    parser.add_argument(
        "--output_root",
        default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_preprocessed_gs/",
        help="Root folder where preprocessed scenes will be saved",
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
    src_dir = os.path.join(args.gs_dir, scene)
    pc_root = args.pc_dir 
    output_root = args.output_root
    scene_name = scene 
    gs_path = os.path.join(src_dir, "output_dynamic_mask", 'ckpts', 'point_cloud_10000.ply' )
    scene_pc_dir = os.path.join(pc_root, split, scene)
    if not os.path.isfile(gs_path):
        # raise FileNotFoundError(f"GS point cloud file not found: {gs_path}")
        print("GS point cloud file not found: ", gs_path)
        return
    
    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print(f"Error loading {gs_path}: {e}")
        return  # skip this scene

    # 3) Parse the Gaussians
    vertex = gs["vertex"]
    gs_data = read_gaussian_attribute(
        vertex, attribute=["coord", "opacity", "scale", "quat", "color"]
    )

    coord = gs_data["coord"]
    color = gs_data["color"]
    opacity = gs_data["opacity"]
    scale = gs_data["scale"]
    quat = gs_data["quat"]

    # 4) Load the PC from pc_root for nearest neighbor labeling
    #    We'll look for pc_root/ split / scene_name
    scene_pc_dir = Path(pc_root) / split / scene_name
    pc_coord_path = scene_pc_dir / "coord.npy"
    pc_segment_path = scene_pc_dir / "segment.npy"
    pc_normal_path = scene_pc_dir / "normal.npy"

    pc_coord = np.load(pc_coord_path)  # (N, 3)
    pc_segment = np.load(pc_segment_path)  # (N,) or (N,1)
    pc_normal = np.load(pc_normal_path) if pc_normal_path.exists() else None  # (N, 3)
    # handle shape
    if pc_segment.ndim == 2 and pc_segment.shape[1] == 1:
        pc_segment = pc_segment.squeeze(1)


    # 6) Nearest Neighbor from point clouds to get the semantic label, and normal for each Gaussian
    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    gs_segment = pc_segment[indices[:, 0]]

    # 7) Save results
    save_path = Path(output_root) / split / scene_name
    save_path.mkdir(parents=True, exist_ok=True)


    np.save(save_path / "coord.npy", coord)
    np.save(save_path / "color.npy", color)
    np.save(save_path / "opacity.npy", opacity)
    np.save(save_path / "scale.npy", scale)
    np.save(save_path / "quat.npy", quat)
    # Save nearest-neighbor semantic data
    np.save(save_path / "segment.npy", gs_segment)
    
    np.save(save_path /"pc_segment.npy", pc_segment)
    np.save(save_path / "pc_coord.npy", pc_coord)

    print(f"Processed scene {scene_name} into {save_path}.")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    pc_dir = args.pc_dir
    gs_dir = args.gs_dir
    output_root = args.output_root
    gs_list = os.listdir(gs_dir)
    gs_list = [d for d in gs_list if os.path.isdir(os.path.join(gs_dir, d))]
    gs_list = sorted(gs_list)

    for gs_scene_i in tqdm(gs_list):

        process_scene(gs_scene_i, "test", args)


if __name__ == "__main__":
    main()