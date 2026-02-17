# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
#
# class Autoencoder_dataset(Dataset):
#     def __init__(self, data_dir):
#         data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
#         self.data_dic = {}
#         for i in range(len(data_names)):
#             features = np.load(data_names[i])
#             name = data_names[i].split('/')[-1].split('.')[0]
#             self.data_dic[name] = features.shape[0]
#             if i == 0:
#                 data = features
#             else:
#                 data = np.concatenate([data, features], axis=0)
#         self.data = data
#
#     def __getitem__(self, index):
#         data = torch.tensor(self.data[index])
#         return data
#
#     def __len__(self):
#         return self.data.shape[0]

import os
import bisect
import random
import numpy as np
import torch
from torch.utils.data import Dataset,IterableDataset



class AutoencoderDataset(Dataset):
    def __init__(self, root_dir, file_name="lang_feat.npy", mmap=True):
        """
        Args:
            root_dir (str): Root folder that contains subfolders with lang_feat.npy.
            file_name (str): Target npy file name in each subfolder.
            mmap (bool): If True, use np.load(..., mmap_mode='r') to avoid loading
                         the full array into RAM.
        """
        self.root_dir = root_dir
        self.file_name = file_name

        # 1. Collect all lang_feat.npy paths recursively
        self.file_paths = []
        for cur_root, _, files in os.walk(root_dir):
            for f in files:
                if f == file_name:
                    self.file_paths.append(os.path.join(cur_root, f))

        if not self.file_paths:
            raise ValueError(f"No '{file_name}' files found under {root_dir}")

        # 2. Open each file (optionally as memmap) and record its length
        self.arrays = []          # list of np.ndarray / np.memmap, each (N_i, 768)
        self.lengths = []         # list of N_i
        self.scene_lengths = {}   # optional: map scene name -> N_i

        for path in self.file_paths:
            if mmap:
                arr = np.load(path, mmap_mode="r")
            else:
                arr = np.load(path)

            if arr.ndim != 2:
                raise ValueError(f"{path} has shape {arr.shape}, expected (N, 768)")
            # If you want to enforce 768-dim:
            # if arr.shape[1] != 768:
            #     raise ValueError(f"{path} has shape {arr.shape}, expected (N, 768)")

            self.arrays.append(arr)
            self.lengths.append(arr.shape[0])

            # use parent folder name as "scene" name (you can change this)
            scene_name = os.path.basename(os.path.dirname(path))
            self.scene_lengths[scene_name] = arr.shape[0]

        # 3. Build cumulative lengths to map global index -> (file_idx, local_idx)
        self.cum_lengths = np.cumsum(self.lengths).tolist()
        self.total_len = self.cum_lengths[-1]

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # support negative indices like a normal Python sequence
        if index < 0:
            index = self.total_len + index

        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} out of range [0, {self.total_len})")

        # Find which file this index falls into
        file_idx = bisect.bisect_right(self.cum_lengths, index)
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = index - prev_cum

        feat_np = np.array(self.arrays[file_idx][local_idx], copy=True)
        feat = torch.from_numpy(feat_np).float()

        return feat

class ChunkedAutoencoderDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        file_name="lang_feat.npy",
        files_per_chunk=5,
        shuffle_files=True,
        shuffle_items=True,
    ):
        """
        Args:
            root_dir: root folder that contains subfolders with lang_feat.npy
            file_name: npy file name to look for
            files_per_chunk: how many files to load into RAM at once
            shuffle_files: shuffle order of files each epoch
            shuffle_items: shuffle items inside each chunk
        """
        self.root_dir = root_dir
        self.file_name = file_name
        self.files_per_chunk = files_per_chunk
        self.shuffle_files = shuffle_files
        self.shuffle_items = shuffle_items

        # collect all lang_feat.npy
        self.file_paths = []
        for cur_root, _, files in os.walk(root_dir):
            for f in files:
                if f == file_name:
                    self.file_paths.append(os.path.join(cur_root, f))

        if not self.file_paths:
            raise ValueError(f"No '{file_name}' files found under {root_dir}")

        # cache for length
        self._length = None

    def _files_for_worker(self):
        # split files list between dataloader workers, if any
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = list(self.file_paths)
        else:
            num_workers = worker_info.num_workers
            per_worker = int(np.ceil(len(self.file_paths) / num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            files = self.file_paths[start:end]
        return files

    def __len__(self):
        """
        Total number of feature vectors across all files.
        This is computed once (using mmap, so only headers are read).
        """
        if self._length is None:
            total = 0
            for p in self.file_paths:
                arr = np.load(p, mmap_mode="r")
                if arr.ndim != 2:
                    raise ValueError(f"{p} has shape {arr.shape}, expected (N, 768)")
                total += arr.shape[0]
            self._length = total
        return self._length

    def __iter__(self):
        files = self._files_for_worker()

        if self.shuffle_files:
            random.shuffle(files)

        # process files in chunks
        for i in range(0, len(files), self.files_per_chunk):
            chunk_paths = files[i : i + self.files_per_chunk]

            arrays = []
            for p in chunk_paths:
                arr = np.load(p)  # shape (N, 768)
                if arr.ndim != 2:
                    raise ValueError(f"{p} has shape {arr.shape}, expected (N, 768)")
                arrays.append(arr)

            # build (file_idx, row_idx) pairs
            indices = []
            for f_idx, arr in enumerate(arrays):
                n = arr.shape[0]
                for r in range(n):
                    indices.append((f_idx, r))

            if self.shuffle_items:
                random.shuffle(indices)

            # yield features one by one
            for f_idx, r in indices:
                feat_np = arrays[f_idx][r]
                # feat_np is writable, so this avoids the previous warning
                feat = torch.from_numpy(feat_np.astype(np.float32, copy=False))
                yield feat