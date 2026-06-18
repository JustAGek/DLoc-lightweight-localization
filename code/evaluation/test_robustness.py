"""
test_robustness.py
Robustness evaluation for student models (Phase 2, Steps 2 & 3).

Tests trained models under various input corruptions and reports
localization error in meters (median, P90, P99).

Step 2 — AP Dropout:
  - Remove each AP individually (zero out 1 of 4 channels)
  - Random 1-AP dropout (random channel zeroed per sample)
  - Random 2-AP dropout (2 random channels zeroed per sample)

Step 3 — Data Augmentation Robustness (optional):
  - Gaussian noise (sigma=0.1, 0.2)
  - AP attenuation (scale channel by 0.5)
  - Phase perturbation (random rotation in complex-like space)
  - Heatmap blur (Gaussian blur on input)

Usage:
  python test_robustness.py
  Edit CONFIG below to select model and weights path.
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
import torch.nn as nn
import torch.nn.functional as torchF
import numpy as np

from utils import localization_error
from data_loader import load_data

# ============================================================
# CONFIG
# ============================================================
MODEL_TYPE = 'mobilenet'  # 'mobilenet', 'tinycnn', 'unet'
WEIGHTS_PATH = './weights/mobilenet_standalone_best.pth'
BATCH_SIZE = 32

# Data paths (Fig 10b — Jacobs Hall)
testpath = [
    './data/dataset_fov_test_jacobs_July28_2.mat',
    './data/dataset_non_fov_test_jacobs_July28_2.mat',
]

# ============================================================
# CORRUPTION FUNCTIONS
# ============================================================
def no_corruption(inputs):
    """Baseline: no corruption."""
    return inputs

def drop_ap(inputs, ap_index):
    """Zero out a specific AP channel."""
    corrupted = inputs.clone()
    corrupted[:, ap_index, :, :] = 0.0
    return corrupted

def random_drop_1ap(inputs):
    """Randomly zero 1 AP per sample."""
    corrupted = inputs.clone()
    for i in range(corrupted.shape[0]):
        ap = torch.randint(0, 4, (1,)).item()
        corrupted[i, ap, :, :] = 0.0
    return corrupted

def random_drop_2ap(inputs):
    """Randomly zero 2 APs per sample."""
    corrupted = inputs.clone()
    for i in range(corrupted.shape[0]):
        aps = torch.randperm(4)[:2]
        for ap in aps:
            corrupted[i, ap, :, :] = 0.0
    return corrupted

def gaussian_noise(inputs, sigma=0.1):
    """Add Gaussian noise."""
    return inputs + torch.randn_like(inputs) * sigma

def ap_attenuation(inputs, scale=0.5):
    """Scale all channels by a factor (simulates weaker signal)."""
    return inputs * scale

def single_ap_attenuation(inputs, ap_index, scale=0.3):
    """Attenuate a single AP channel."""
    corrupted = inputs.clone()
    corrupted[:, ap_index, :, :] *= scale
    return corrupted

def gaussian_blur(inputs, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to each channel."""
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=inputs.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Make 2D kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(inputs.shape[1], 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return torchF.conv2d(inputs, kernel_2d, padding=pad, groups=inputs.shape[1])

# ============================================================
# MAIN
# ============================================================
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


def evaluate_condition(model, test_loader, corrupt_fn, device):
    """Run model on corrupted inputs, return error stats in meters."""
    all_errors = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            corrupted = corrupt_fn(inputs)
            output = model(corrupted)

            errors = localization_error(
                output.cpu().numpy(),
                labels.cpu().numpy(),
                scale=0.1,
            )
            all_errors.extend(errors)

    all_errors = np.array(all_errors)
    return {
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
    print(f'Parameters: {n_params:,}')

    # Load test data
    print('Loading test data...')
    test_w_list, test_lbl_list = [], []
    for fp in testpath:
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        test_w_list.append(w)
        test_lbl_list.append(lbl)
    test_w = torch.cat(test_w_list, dim=0)
    test_lbl = torch.cat(test_lbl_list, dim=0)
    print(f'Test samples: {test_w.shape[0]}')

    test_data = torch.utils.data.TensorDataset(test_w, test_lbl)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Define all test conditions
    conditions = {
        # Step 2: AP Dropout
        'Normal (no corruption)': no_corruption,
        'Drop AP 0': lambda x: drop_ap(x, 0),
        'Drop AP 1': lambda x: drop_ap(x, 1),
        'Drop AP 2': lambda x: drop_ap(x, 2),
        'Drop AP 3': lambda x: drop_ap(x, 3),
        'Random drop 1 AP': random_drop_1ap,
        'Random drop 2 APs': random_drop_2ap,
        # Step 3: Augmentation robustness
        'Gaussian noise (sigma=0.1)': lambda x: gaussian_noise(x, sigma=0.1),
        'Gaussian noise (sigma=0.2)': lambda x: gaussian_noise(x, sigma=0.2),
        'Attenuation (0.5x all)': lambda x: ap_attenuation(x, scale=0.5),
        'Attenuate AP 0 (0.3x)': lambda x: single_ap_attenuation(x, 0, 0.3),
        'Gaussian blur (k=5, s=1)': lambda x: gaussian_blur(x, kernel_size=5, sigma=1.0),
    }

    # Run all conditions
    print(f'\n{"="*75}')
    print(f'  ROBUSTNESS EVALUATION — {MODEL_TYPE.upper()}')
    print(f'{"="*75}')
    print(f'{"Condition":<30} {"Median":>8} {"P90":>8} {"P99":>8} {"Mean":>8}')
    print(f'{"-"*75}')

    results = {}
    for name, corrupt_fn in conditions.items():
        stats = evaluate_condition(model, test_loader, corrupt_fn, device)
        results[name] = stats
        print(f'{name:<30} {stats["median"]:>7.3f}m {stats["p90"]:>7.3f}m {stats["p99"]:>7.3f}m {stats["mean"]:>7.3f}m')

    print(f'{"="*75}')

    # Save results
    save_path = f'robustness_{MODEL_TYPE}.txt'
    with open(save_path, 'w') as f:
        f.write(f'Model: {MODEL_TYPE}\n')
        f.write(f'Weights: {WEIGHTS_PATH}\n')
        f.write(f'Test samples: {test_w.shape[0]}\n\n')
        f.write(f'{"Condition":<30} {"Median":>8} {"P90":>8} {"P99":>8} {"Mean":>8}\n')
        f.write(f'{"-"*62}\n')
        for name, stats in results.items():
            f.write(f'{name:<30} {stats["median"]:>7.3f}m {stats["p90"]:>7.3f}m {stats["p99"]:>7.3f}m {stats["mean"]:>7.3f}m\n')
    print(f'\nResults saved to {save_path}')


if __name__ == '__main__':
    main()
