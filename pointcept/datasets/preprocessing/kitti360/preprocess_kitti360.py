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
# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}



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
        "--pc_dir", default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_subset/2013_05_28_drive_0000_sync"
    )
    parser.add_argument(
        "--output_root",
        default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_preprocessed",
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
    src_dir = os.path.join(args.pc_dir, scene)
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

    groundTruthPcd = read_ply(ply_path)
    x = groundTruthPcd['x']
    y = groundTruthPcd['y']
    z = groundTruthPcd['z']
    pts = np.vstack((x, y, z)).T.astype(np.float32)
    cols = np.vstack((groundTruthPcd['red'], groundTruthPcd['green'], groundTruthPcd['blue'])).T.astype(np.float32)  
    cols = cols.astype(np.uint8)
    segment = groundTruthPcd['semantic'].astype(np.uint8)
    segment_copy = segment.copy()
    segment_new = np.zeros_like(segment_copy, dtype=np.int16)
    segment_new = segment_new - 1 # map to -1 
    # segment: map to kittiid
    for id, kitti_id in label2kittiId.items():
        segment_new[segment == id] = kitti_id
 

    # mapping something to -1 


    np.save(os.path.join(dst_dir, "coord.npy"), pts)
    np.save(os.path.join(dst_dir, "color.npy"), cols)
    np.save(os.path.join(dst_dir, "segment.npy"), segment_new)


    # logging.info(f"Processed scene {scene} into {dst_dir}.")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    pc_dir = args.pc_dir
    output_root = args.output_root
    pc_list = os.listdir(pc_dir)
    pc_list = [d for d in pc_list if os.path.isdir(os.path.join(pc_dir, d))]

    for pc_scene_i in tqdm(pc_list):

        process_scene(pc_scene_i, "test", args)


if __name__ == "__main__":
    main()