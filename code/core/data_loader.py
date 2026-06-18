#!/usr/bin/python
# A simple data loader that imports the train and test mat files
# from the `filename` and converts them to torch.tesnors()
# to be loaded for training and testing DLoc network
# `features_wo_offset`: targets for the consistency decoder
# `features_w_offset` : inputs for the network/encoder
# `labels_gaussian_2d`: targets for the location decoder
import torch
import h5py
import scipy.io
import numpy as np

def load_data(filename):
    print('Loading '+filename)
    arrays = {}
    f = h5py.File(filename,'r')
    features_wo_offset = torch.tensor(np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32)), dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32)), dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32)), dtype=torch.float32)
        
    return features_wo_offset,features_w_offset, labels_gaussian_2d


class LazyMultiHDF5Dataset(torch.utils.data.Dataset):
    """Reads samples on demand from multiple HDF5 .mat files.
    Uses ~0 RAM regardless of dataset size. Each __getitem__ does 3 small
    HDF5 slice reads (~700 KB total per sample).

    Returns the same tuple ordering as the original TensorDataset:
        (features_wo_offset, features_w_offset, labels_gaussian_2d)
    """

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self._handles = {}           # lazy-opened per worker/process
        self.cum_sizes = []
        self.var_names = {}          # per file: (w_name, wo_name, lbl_name)

        total = 0
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                keys = list(f.keys())
                w_name  = 'features_w_offset'  if 'features_w_offset'  in keys else 'features_with_offset'
                wo_name = 'features_wo_offset' if 'features_wo_offset' in keys else 'features_without_offset'
                lbl_name = 'labels_gaussian_2d'
                self.var_names[fp] = (w_name, wo_name, lbl_name)
                # HDF5 stores MATLAB dims reversed: last dim = N
                n = f[w_name].shape[-1]
                total += n
                self.cum_sizes.append(total)
                print(f'  Indexed {fp}: {n} samples')

        self.total_len = total
        print(f'  Total: {self.total_len} samples (lazy, ~0 MB RAM)')

    def __len__(self):
        return self.total_len

    def _get_handle(self, fp):
        if fp not in self._handles:
            self._handles[fp] = h5py.File(fp, 'r')
        return self._handles[fp]

    def __getitem__(self, idx):
        # Find which file this idx belongs to
        file_idx = 0
        for i, cs in enumerate(self.cum_sizes):
            if idx < cs:
                file_idx = i
                break

        offset = self.cum_sizes[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - offset

        fp = self.file_paths[file_idx]
        f = self._get_handle(fp)
        w_name, wo_name, lbl_name = self.var_names[fp]

        # HDF5 dims are reversed vs MATLAB:
        #   features: MATLAB [N,4,161,361] -> HDF5 [361,161,4,N]
        #   labels:   MATLAB [N,161,361]   -> HDF5 [361,161,N]
        feat_w  = np.array(f[w_name][:, :, :, local_idx],  dtype=np.float32).transpose()  # -> [4,161,361]
        feat_wo = np.array(f[wo_name][:, :, :, local_idx], dtype=np.float32).transpose()  # -> [4,161,361]
        label   = np.array(f[lbl_name][:, :, local_idx],   dtype=np.float32).transpose()  # -> [161,361]
        label   = label[np.newaxis, :, :]   # -> [1,161,361]

        return (
            torch.from_numpy(feat_wo),
            torch.from_numpy(feat_w),
            torch.from_numpy(label)
        )

    def __del__(self):
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass
