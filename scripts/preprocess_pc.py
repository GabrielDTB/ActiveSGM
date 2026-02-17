import argparse
import numpy as np
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from typing import Dict
import json
import pandas as pd
import re

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



################################################################################
# Main Processing Function
################################################################################

def _normalize_label(name: str) -> str:
    """
    Normalize label strings so variants like 'chest_of_drawers' and 'chest of drawers'
    become identical keys. Lowercase, remove punctuation, collapse whitespace/underscores,
    and remove spaces. Also add a simple plural fallback.
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s-]+", " ", s)  # keep letters/numbers/space/hyphen
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("-", " ")
    # final key without spaces
    key = s.replace(" ", "")
    return key


def build_index_to_seg_mapping_from_txt(mapping_df, label_txt_path, label_type):
    """
    Build {index -> seg_id} where seg_id is the 0-based position of the class
    name in the label text file (one class name per line).

    Args:
        mapping_df (pd.DataFrame): Must contain at least:
            - for mp3d* : columns ['index', 'mpcat40']
            - for nyu*  : columns ['index', 'nyuClass']
        label_txt_path (str): Path to label .txt (e.g., 'mp3d40.txt').
        label_type (str): One of {'mp3d40', 'nyu40', 'mp3d21', 'nyu160'}.

    Returns:
        dict: {index: seg_id} where seg_id in [0..len(label_list)-1], or 255 if not found.
    """
    # 1) Load ordered label list from the txt
    with open(label_txt_path, "r") as f:
        label_list = [line.strip() for line in f if line.strip()]
    name_to_seg = {name: i for i, name in enumerate(label_list)}

    # 2) Pick which column has the class names for the chosen label_type
    if label_type in ("mp3d40",):
        name_col = "mpcat40"
    elif label_type in ("nyu40", "mp3d21", "nyu160"):
        name_col = "nyuClass"
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")

    # 3) Build the mapping dict: index -> seg_id (by name lookup in the txt)
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        idx = int(row["index"])
        cls = row.get(name_col, None)
        if isinstance(cls, str) and cls in name_to_seg:
            mapping_dict[idx] = name_to_seg[cls]
        else:
            # not present or unknown → ignore code; change to -1 if you prefer
            mapping_dict[idx] = -1

    return mapping_dict


def build_index_to_nyu_mapping(mapping_df, num_classes=160):
    """
    Build a mapping dict from Matterport 'index' → NYU160 label ID.

    This follows the same logic used in your Matterport pre-processing:
      - Skip 'void' and 'unknown' categories.
      - Assign each unique nyuClass a new ID from 1..num_classes (up to 160).
      - Reuse IDs for repeated nyuClass names.
      - Everything else (unknown, void, overflow) maps to 0 (ignored).

    Args:
        mapping_df (pd.DataFrame): DataFrame with at least 'index' and 'nyuClass' columns.
        num_classes (int): Number of allowed NYU classes (default=160).

    Returns:
        dict: {index: nyu160_id (int)}, where 0 means ignored/unknown.
    """
    # Verify required columns
    if 'index' not in mapping_df.columns or 'nyuClass' not in mapping_df.columns:
        raise ValueError("mapping_df must contain columns ['index', 'nyuClass'].")

    eliminated_list = ['void', 'unknown']
    label_names = []  # track unique class names
    mapping_dict = {}
    counter = 1
    flag_stop = False

    for _, row in mapping_df.iterrows():
        idx = int(row['index'])
        cls_name = row['nyuClass'] if isinstance(row['nyuClass'], str) else None

        # skip void/unknown/missing
        if cls_name is None or cls_name in eliminated_list:
            mapping_dict[idx] = 0
            continue

        # reuse existing id
        if cls_name in label_names:
            mapping_dict[idx] = label_names.index(cls_name) + 1
        else:
            # assign new id
            if not flag_stop and counter <= num_classes:
                label_names.append(cls_name)
                mapping_dict[idx] = counter
                counter += 1
                if counter > num_classes:
                    flag_stop = True
            else:
                mapping_dict[idx] = 0  # overflow ignored

    return mapping_dict

def build_index_to_21label_mapping(mapping_df):
    """
    Build a mapping dictionary from 'index' to 21-class remapped label
    based on the Matterport-to-NYU mapping rules.

    Args:
        mapping_df: data_frame read in the category_mapping.tsv file.

    Returns:
        dict: {index: remapped_label_0_20_or_255}
    """
    mapping_dict = dict(zip(mapping_df['index'], mapping_df['nyu40id']))

    # Step 2: Define the Matterport → 21-label remap
    MATTERPORT_CLASS_REMAP = np.zeros(41)
    MATTERPORT_CLASS_REMAP[1] = 1
    MATTERPORT_CLASS_REMAP[2] = 2
    MATTERPORT_CLASS_REMAP[3] = 3
    MATTERPORT_CLASS_REMAP[4] = 4
    MATTERPORT_CLASS_REMAP[5] = 5
    MATTERPORT_CLASS_REMAP[6] = 6
    MATTERPORT_CLASS_REMAP[7] = 7
    MATTERPORT_CLASS_REMAP[8] = 8
    MATTERPORT_CLASS_REMAP[9] = 9
    MATTERPORT_CLASS_REMAP[10] = 10
    MATTERPORT_CLASS_REMAP[11] = 11
    MATTERPORT_CLASS_REMAP[12] = 12
    MATTERPORT_CLASS_REMAP[14] = 13
    MATTERPORT_CLASS_REMAP[16] = 14
    MATTERPORT_CLASS_REMAP[22] = 21  # ceiling
    MATTERPORT_CLASS_REMAP[24] = 15
    MATTERPORT_CLASS_REMAP[28] = 16
    MATTERPORT_CLASS_REMAP[33] = 17
    MATTERPORT_CLASS_REMAP[34] = 18
    MATTERPORT_CLASS_REMAP[36] = 19
    MATTERPORT_CLASS_REMAP[39] = 20

    MATTERPORT_ALLOWED_NYU_CLASSES = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        14, 16, 22, 24, 28, 33, 34, 36, 39
    ]

    # Step 3: Build remapper for 150-class → 21-class (unallowed → 255)
    remapper = np.ones(150, dtype=int) * 255
    for i, x in enumerate(MATTERPORT_ALLOWED_NYU_CLASSES):
        remapper[x] = i

    # Step 4: Compose index → mpcat40 → remapped_label
    index_to_21label = {}
    for idx, mpcat in mapping_dict.items():
        if mpcat < len(remapper):
            label = remapper[int(mpcat)]
            if label != 255 and mpcat < len(MATTERPORT_CLASS_REMAP):
                label = int(MATTERPORT_CLASS_REMAP[int(mpcat)])
            index_to_21label[int(idx)] = int(label)
        else:
            index_to_21label[int(idx)] = 255  # ignored / unknown

    return index_to_21label

def accumulate_face_labels_to_vertices(triangles, pc_coords, category_ids):
    """
    triangles: (F,) iterable of 3-vertex indices (e.g., PlyData['face'].data['vertex_indices'])
    pc_coords: (N, 3) points, only used for N (vertex count)
    category_ids: (F,) per-face integer labels in [0..K], 0 can be ignore
    """
    triangles = np.asarray(triangles, dtype=object)  # Ply stores as object; elements are length-3 arrays
    category_ids = np.asarray(category_ids, dtype=np.int64)

    # ensure non-negative; treat negatives as 0 (ignore)
    category_ids = np.where(category_ids >= 0, category_ids, 0)

    # pick class axis size = max label + 1
    C = int(category_ids.max()) + 1
    N = pc_coords.shape[0]
    vertex_labels_votes = np.zeros((N, C), dtype=np.int32)

    # vectorized scatter-add using np.add.at
    tri_np = np.stack(triangles)        # (F, 3)
    face_idx = np.repeat(np.arange(tri_np.shape[0]), 3)  # (F*3,)
    vert_idx = tri_np.reshape(-1)                        # (F*3,)
    face_labels = category_ids[face_idx]                 # (F*3,)

    # clip any accidental out-of-range to [0, C-1]
    face_labels = np.clip(face_labels, 0, C-1)

    np.add.at(vertex_labels_votes, (vert_idx, face_labels), 1)

    # majority vote per vertex
    vertex_labels = vertex_labels_votes.argmax(axis=1)

    # optional: map background/zero-vote to "ignore code"
    # if you want -1 as ignore:
    zero_vote_vertices = (vertex_labels_votes.sum(axis=1) == 0)
    vertex_labels = vertex_labels.astype(np.int32)
    vertex_labels[zero_vote_vertices] = 0   # ignore

    return vertex_labels

def process_ply_file(ply_path, output_dir,mapping_dict=None,label_type='mp3d40'):
    """
    Process a single PLY file and save its parameters as separate NPY files.
    xxx.ply = 3D mesh in ply format.   In addition to the usual fields, there are three additional fields for each face:
    face_material = unique id of segment containing this face
    face_segment = unique id of object instance containing this face
    face_category = unique id of the category label for the object instance containing this face
        (i.e., mapping to the "index" column of the category.tsv file)
    Args:
        ply_path: Path to the input PLY file
        output_dir: Directory where the NPY files will be saved
    """
    max_label_dict = {'mp3d40':40,
                  "nyu160":160,
                  "mp3d21":21,
                  "nyu40":40}

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
    vertex = np.array([list(x) for x in ply_data.elements[0]])
    pc_coords = np.ascontiguousarray(vertex[:, :3])

    object_ids = ply_data['face'].data['object_id']
    object_ids = np.asarray(object_ids)
    object_ids = object_ids.copy()
    object_ids[object_ids == -1] = 0

    if mapping_dict is not None:
        category_ids = np.array([mapping_dict.get(int(i), 0) for i in object_ids.flatten()])
        category_ids = category_ids.reshape(object_ids.shape)
        MAX_LABEL = max_label_dict[label_type]
        category_ids[category_ids>MAX_LABEL] = 0
    else:
        category_ids = object_ids

    triangles = ply_data['face'].data['vertex_indices']
    vertex_labels = accumulate_face_labels_to_vertices(triangles, pc_coords, category_ids)
    vertex_labels -= 1

    # Save individual parameters
    np.save(output_dir / "pc_coord.npy", pc_coords)
    np.save(output_dir / f"pc_segment_{label_type}.npy", vertex_labels)

    print(f"Saved GT segments to: {output_dir}")
    print(f" - Number of points: {len(pc_coords)}")

    return True


################################################################################
# Main script
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GT semantic mesh PLY files to NPY parameter files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input PLY file or directory containing PLY files",
    )

    parser.add_argument(
        "--category_mapping",
        default=None,
        help="Path to input mapping tsv file, mapping object id to semantic class",
    )

    parser.add_argument(
        "--scene_seg_info",
        default=None,
        help="Path to scene segementation json file, mapping segments to raw category",
    )

    parser.add_argument(
        "--label_type",
        type=str,
        default='mp3d40',
        choices=['mp3d40','nyu160','mp3d21'],
        help="map instance id to which kind of labels",
    )

    parser.add_argument(
        "--label_txt",
        default=None,
        help="Path to semantic label txt file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for NPY files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process PLY files in subdirectories",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Collect Params files to process
    params_files = []

    if input_path.is_file() and input_path.suffix == ".ply":
        # Single file mode
        params_files = [input_path]
    elif input_path.is_dir():
        # Directory mode
        if args.recursive:
            params_files = list(input_path.rglob("*semantic_clean.ply"))
        else:
            params_files = list(input_path.glob("*semantic_clean.ply"))
    else:
        raise ValueError(
            f"Input path {input_path} is not a valid PLY file or directory"
        )

    if not params_files:
        print("No Params files found!")
        exit(1)

    print(f"Found {len(params_files)} PLY file(s) to process")

    mapping_df = pd.read_csv(args.category_mapping, sep='\t')
    with open(args.scene_seg_info, "r") as f:
        semseg_data = json.load(f)

    if args.label_type == 'mp3d40':
        label_to_mpcat40 = mapping_df.set_index("index")["mpcat40index"].to_dict()
        mapping_dict = {
            int(group["id"]): int(label_to_mpcat40.get(group["label_index"], 0))
            for group in semseg_data["segGroups"]
        }
    elif args.label_type == 'mp3d21':
        label_to_mpcat21 = build_index_to_21label_mapping(mapping_df)
        mapping_dict = {
            int(group["id"]): int(label_to_mpcat21.get(group["label_index"], 0))
            for group in semseg_data["segGroups"]
        }
    elif args.label_type == 'nyu160':
        label_to_nyu160 = build_index_to_nyu_mapping(mapping_df,160)
        mapping_dict = {
            int(group["id"]): int(label_to_nyu160.get(group["label_index"], 0))
            for group in semseg_data["segGroups"]
        }
    else:
        mapping_dict = None


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
            success = process_ply_file(params_file, output_dir, mapping_dict=mapping_dict,label_type=args.label_type)
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
