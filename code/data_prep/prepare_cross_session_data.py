"""
prepare_cross_session_data.py
Prepare Aug16 session data for cross-session evaluation (Phase 2, Step 1).

The WILD dataset has 5 Jacobs Hall sessions. Our models trained on sessions 1+2
(July28). To test cross-session generalization, we need sessions 3-5 (Aug16).

STEP 1 — Download raw feature files from WILD (password-protected):
  https://wcsng.ucsd.edu/wild/
  Place these in ./data/:
    features_jacobs_Aug16_1.mat
    features_jacobs_Aug16_3.mat
    features_jacobs_Aug16_4_ref.mat

STEP 2 — Run this script:
  python prepare_cross_session_data.py

OUTPUT — Creates in ./data/:
  dataset_train_jacobs_Aug16_1.mat     (all train samples from session 3)
  dataset_train_jacobs_Aug16_3.mat     (all train samples from session 4)
  dataset_train_jacobs_Aug16_4_ref.mat (all train samples from session 5)

These are the files that eval_cross_session.py and train_and_test.py expect.
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import sys
import h5py
import numpy as np

# Paths
DATA_DIR = './data'
SPLIT_DIR = '../data_split_idx'
CHUNK = 300


def load_indices(split_file, key):
    """Load MATLAB 1-based index array, return 0-based numpy int array."""
    import scipy.io
    if not os.path.exists(split_file):
        raise FileNotFoundError(f'Split file not found: {split_file}')
    try:
        with h5py.File(split_file, 'r') as f:
            idx = np.array(f[key]).flatten().astype(np.int64) - 1
    except Exception:
        mat = scipy.io.loadmat(split_file)
        idx = mat[key].flatten().astype(np.int64) - 1
    return idx


def get_var_names(h5_file):
    """Handle both naming conventions."""
    keys = list(h5_file.keys())
    w_key  = 'features_w_offset'  if 'features_w_offset'  in keys else 'features_with_offset'
    wo_key = 'features_wo_offset' if 'features_wo_offset' in keys else 'features_without_offset'
    return w_key, wo_key


def extract_and_save(src_path, dst_path, indices):
    """Read samples at indices from source HDF5, write to dest in dataset_* format."""
    sorted_idx = np.sort(indices)
    n = len(sorted_idx)
    print(f'  Writing {n} samples -> {os.path.basename(dst_path)}')

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        w_key, wo_key = get_var_names(src)

        src_shape_w   = src[w_key].shape
        src_shape_lbl = src['labels_gaussian_2d'].shape

        out_shape_w   = src_shape_w[:-1]   + (n,)
        out_shape_lbl = src_shape_lbl[:-1] + (n,)

        # Always write with standard names (features_w_offset, etc.)
        ds_w   = dst.create_dataset('features_w_offset',  shape=out_shape_w,   dtype='float32',
                                    compression='gzip', chunks=True)
        ds_wo  = dst.create_dataset('features_wo_offset', shape=out_shape_w,   dtype='float32',
                                    compression='gzip', chunks=True)
        ds_lbl = dst.create_dataset('labels_gaussian_2d', shape=out_shape_lbl, dtype='float32',
                                    compression='gzip', chunks=True)

        written = 0
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_idx = sorted_idx[start:end].tolist()

            ds_w  [..., written:written + (end - start)] = np.array(src[w_key ][..., chunk_idx], dtype=np.float32)
            ds_wo [..., written:written + (end - start)] = np.array(src[wo_key][..., chunk_idx], dtype=np.float32)
            ds_lbl[..., written:written + (end - start)] = np.array(src['labels_gaussian_2d'][..., chunk_idx], dtype=np.float32)

            written += (end - start)
            print(f'    {written}/{n}', end='\r', flush=True)
        print()
    print(f'  Saved: {dst_path}')


def prepare_aug16_session(session_name, features_filename, split_filename):
    """Prepare one Aug16 session's data."""
    src = os.path.join(DATA_DIR, features_filename)
    split_file = os.path.join(SPLIT_DIR, split_filename)

    if not os.path.exists(src):
        print(f'  MISSING: {src}')
        print(f'  -> Download {features_filename} from WILD website')
        return False

    if not os.path.exists(split_file):
        print(f'  MISSING: {split_file}')
        return False

    print(f'\n  Processing {session_name}...')
    print(f'  Source: {src}')
    print(f'  Split:  {split_file}')

    # Combine fov_train + non_fov_train for all training samples
    # (For cross-session eval, these are used as test data)
    fov_train = load_indices(split_file, 'fov_train_idx')
    non_fov_train = load_indices(split_file, 'non_fov_train_idx')
    all_train = np.concatenate([fov_train, non_fov_train])
    print(f'  Samples: {len(fov_train)} fov_train + {len(non_fov_train)} non_fov_train = {len(all_train)} total')

    dst = os.path.join(DATA_DIR, f'dataset_train_jacobs_{session_name}.mat')
    extract_and_save(src, dst, all_train)
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print('='*60)
    print('Preparing Aug16 cross-session data')
    print('='*60)

    sessions = [
        ('Aug16_1',     'features_jacobs_Aug16_1.mat',     'data_split_ids_jacobs_Aug16_1.mat'),
        ('Aug16_3',     'features_jacobs_Aug16_3.mat',     'data_split_ids_jacobs_Aug16_3.mat'),
        ('Aug16_4_ref', 'features_jacobs_Aug16_4_ref.mat', 'data_split_ids_jacobs_Aug16_4_ref.mat'),
    ]

    results = {}
    for session_name, feat_file, split_file in sessions:
        results[session_name] = prepare_aug16_session(session_name, feat_file, split_file)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for name, ok in results.items():
        status = 'READY' if ok else 'MISSING (download features file from WILD)'
        print(f'  {name}: {status}')

    all_ok = all(results.values())
    if all_ok:
        print('\nAll Aug16 data prepared. Run:')
        print('  python eval_cross_session.py')
    else:
        print('\nSome files missing. Download from: https://wcsng.ucsd.edu/wild/')
        print('Place features_jacobs_Aug16_*.mat files in ./data/')


if __name__ == '__main__':
    main()
