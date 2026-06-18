"""
eval_cross_session.py
Cross-session generalization evaluation (Phase 2, Step 1).

Evaluates trained student models on data from DIFFERENT collection sessions
to measure how well localization generalizes across time/environment changes.

WILD Jacobs Hall sessions:
  Session 1: jacobs_Jul28       — standard setup
  Session 2: jacobs_Jul28_2     — same setup, 1 hour later (our train/test data)
  Session 3: jacobs_Aug16_1     — extra furniture
  Session 4: jacobs_Aug16_3     — different furniture arrangement
  Session 5: jacobs_Aug16_4_ref — furniture + aluminum reflector

Models were trained on Session 1 + Session 2 (train split).
This script tests on:
  - Session 2 test (normal, in-distribution)
  - Session 3 (unseen session, different furniture)
  - Session 4 (unseen session, different furniture)
  - Session 5 (unseen session, furniture + reflector)

REQUIRED DATA:
  You MUST download the Aug16 feature files from WILD and process them.
  Option 1 (if pre-split files exist on server):
    ./data/dataset_train_jacobs_Aug16_1.mat
    ./data/dataset_train_jacobs_Aug16_3.mat
    ./data/dataset_train_jacobs_Aug16_4_ref.mat
  Option 2 (if you have raw WILD features):
    Run prepare_aug16_data() below to create them.

Usage:
  python eval_cross_session.py
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
import torch
import numpy as np

from utils import localization_error
from data_loader import load_data

# ============================================================
# CONFIG
# ============================================================
MODEL_TYPE = 'mobilenet'  # 'mobilenet', 'tinycnn', 'unet'
WEIGHTS_PATH = './weights/mobilenet_standalone_best.pth'
BATCH_SIZE = 32

# Session data paths
# Session 2 test (normal evaluation — in-distribution)
SESSION_2_TEST = [
    './data/dataset_fov_test_jacobs_July28_2.mat',
    './data/dataset_non_fov_test_jacobs_July28_2.mat',
]

# Sessions 3-5 (cross-session — out-of-distribution)
# These files contain ALL samples from each session (no train/test split needed
# since the model never trained on them)
CROSS_SESSION_FILES = {
    'Session 3 (Aug16_1, furniture)': './data/dataset_jacobs_Aug16_1.mat',
    'Session 4 (Aug16_3, diff furniture)': './data/dataset_jacobs_Aug16_3.mat',
    'Session 5 (Aug16_4_ref, reflector)': './data/dataset_jacobs_Aug16_4_ref.mat',
}


def load_model(model_type, weights_path, device):
    if model_type == 'mobilenet':
        from student_mobilenet import MobileNetStudent
        model = MobileNetStudent(input_nc=4)
    elif model_type == 'tinycnn':
        from student_tinycnn import TinyCNNStudent
        model = TinyCNNStudent(input_nc=4)
    elif model_type == 'unet':
        from student_unet import MobileNetUNetStudent
        model = MobileNetUNetStudent(input_nc=4)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_on_files(model, file_paths, device, batch_size=32):
    """Load data from file_paths and evaluate model. Returns error stats in meters."""
    w_list, lbl_list = [], []
    for fp in file_paths:
        if not os.path.exists(fp):
            return None  # Signal that file is missing
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        w_list.append(w)
        lbl_list.append(lbl)

    w = torch.cat(w_list, dim=0)
    lbl = torch.cat(lbl_list, dim=0)
    n_samples = w.shape[0]

    dataset = torch.utils.data.TensorDataset(w, lbl)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    all_errors = []
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            output = model(inputs)
            errors = localization_error(
                output.cpu().numpy(),
                labels.cpu().numpy(),
                scale=0.1,
            )
            all_errors.extend(errors)

    all_errors = np.array(all_errors)
    return {
        'n_samples': n_samples,
        'median': np.median(all_errors),
        'p90': np.percentile(all_errors, 90),
        'p99': np.percentile(all_errors, 99),
        'mean': np.mean(all_errors),
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model: {MODEL_TYPE}, Weights: {WEIGHTS_PATH}')

    # Load model
    model = load_model(MODEL_TYPE, WEIGHTS_PATH, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}\n')

    # Collect all sessions to evaluate
    sessions = {}
    sessions['Session 2 test (in-distribution)'] = SESSION_2_TEST
    for name, path in CROSS_SESSION_FILES.items():
        sessions[name] = [path]

    # Evaluate each session
    print(f'{"="*80}')
    print(f'  CROSS-SESSION EVALUATION -- {MODEL_TYPE.upper()}')
    print(f'{"="*80}')
    print(f'  Model trained on: Session 1 (Jul28) + Session 2 train (Jul28_2)')
    print(f'{"="*80}\n')

    print(f'{"Session":<42} {"N":>6} {"Median":>8} {"P90":>8} {"P99":>8} {"Mean":>8}')
    print(f'{"-"*80}')

    results = {}
    for name, file_paths in sessions.items():
        stats = evaluate_on_files(model, file_paths, device, BATCH_SIZE)
        if stats is None:
            print(f'{name:<42} {"MISSING DATA FILES":>40}')
            continue
        results[name] = stats
        print(f'{name:<42} {stats["n_samples"]:>6} '
              f'{stats["median"]:>7.3f}m {stats["p90"]:>7.3f}m '
              f'{stats["p99"]:>7.3f}m {stats["mean"]:>7.3f}m')

    print(f'{"="*80}')

    if not results:
        print('\nNo sessions could be evaluated. Check data file paths.')
        return

    # Show degradation from baseline
    if 'Session 2 test (in-distribution)' in results:
        baseline = results['Session 2 test (in-distribution)']
        print(f'\nDegradation vs in-distribution (Session 2 test):')
        print(f'{"Session":<42} {"Median delta":>14} {"P90 delta":>14}')
        print(f'{"-"*70}')
        for name, stats in results.items():
            if name == 'Session 2 test (in-distribution)':
                continue
            med_delta = stats['median'] - baseline['median']
            p90_delta = stats['p90'] - baseline['p90']
            print(f'{name:<42} {med_delta:>+13.3f}m {p90_delta:>+13.3f}m')

    # Save results
    save_path = f'cross_session_{MODEL_TYPE}.txt'
    with open(save_path, 'w') as f:
        f.write(f'Cross-Session Evaluation\n')
        f.write(f'Model: {MODEL_TYPE}\n')
        f.write(f'Weights: {WEIGHTS_PATH}\n')
        f.write(f'Trained on: Session 1 (Jul28) + Session 2 train (Jul28_2)\n\n')
        f.write(f'{"Session":<42} {"N":>6} {"Median":>8} {"P90":>8} {"P99":>8}\n')
        f.write(f'{"-"*72}\n')
        for name, stats in results.items():
            f.write(f'{name:<42} {stats["n_samples"]:>6} '
                    f'{stats["median"]:>7.3f}m {stats["p90"]:>7.3f}m {stats["p99"]:>7.3f}m\n')
    print(f'\nResults saved to {save_path}')


if __name__ == '__main__':
    main()
