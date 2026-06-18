"""
run_multiple_seeds.py
Multi-seed training for statistical significance (Phase 2, Step 4).

Trains a student model N times with different seeds, reports mean +/- std
for median, P90, P99 localization error in meters.

Supports: mobilenet, tinycnn, unet
Modes: standalone, distill

Usage:
  python run_multiple_seeds.py
  Edit CONFIG below to select model/mode/seeds.
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
import time
import random
import torch
import torch.nn as nn
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import localization_error, write_log
from data_loader import load_data

# ============================================================
# CONFIG
# ============================================================
STUDENT_TYPE = 'mobilenet'    # 'mobilenet', 'tinycnn', 'unet'
MODE = 'standalone'           # 'standalone' or 'distill'
SCHEDULER_TYPE = 'none'       # 'none', 'cosine', 'step' (only for unet)

SEEDS = [42, 123, 777]
EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4

ALPHA = 0.7
BETA = 0.3

TEACHER_TRAIN_PT = './data/teacher_train_outputs.pt'
TEACHER_TEST_PT = './data/teacher_test_outputs.pt'

# Data paths (Fig 10b -- Jacobs Hall)
trainpath = [
    './data/dataset_jacobs_July28.mat',
    './data/dataset_non_fov_train_jacobs_July28_2.mat',
    './data/dataset_fov_train_jacobs_July28_2.mat',
]
testpath = [
    './data/dataset_fov_test_jacobs_July28_2.mat',
    './data/dataset_non_fov_test_jacobs_July28_2.mat',
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_student(student_type, device):
    if student_type == 'mobilenet':
        from student_mobilenet import MobileNetStudent
        return MobileNetStudent(input_nc=4).to(device)
    elif student_type == 'tinycnn':
        from student_tinycnn import TinyCNNStudent
        return TinyCNNStudent(input_nc=4).to(device)
    elif student_type == 'unet':
        from student_unet import MobileNetUNetStudent
        return MobileNetUNetStudent(input_nc=4).to(device)
    else:
        raise ValueError(f'Unknown student type: {student_type}')


def train_one_seed(seed, train_loader, test_loader, device, save_dir):
    """Train a fresh model with the given seed. Returns best test metrics."""
    set_seed(seed)
    print(f'\n{"="*60}')
    print(f'  SEED {seed} — {STUDENT_TYPE} / {MODE}')
    print(f'{"="*60}')

    student = create_student(STUDENT_TYPE, device)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Optional LR scheduler
    scheduler = None
    if SCHEDULER_TYPE == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif SCHEDULER_TYPE == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_median = float('inf')
    best_p90 = float('inf')
    best_p99 = float('inf')
    seed_dir = os.path.join(save_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        student.train()
        train_loss_sum = 0
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            output = student(inputs)
            L_GT = criterion(output, labels)

            if MODE == 'distill':
                teacher_out = data[2].to(device)
                L_KD = criterion(output, teacher_out)
                loss = ALPHA * L_GT + BETA * L_KD
            else:
                loss = L_GT

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        # Test
        student.eval()
        test_errors = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                output = student(inputs)
                test_errors.extend(localization_error(
                    output.cpu().numpy(),
                    labels.cpu().numpy(),
                    scale=0.1,
                ))

        test_median = np.median(test_errors)
        test_p90 = np.percentile(test_errors, 90)
        test_p99 = np.percentile(test_errors, 99)

        if scheduler is not None:
            scheduler.step()

        # Log
        run_label = f'seed{seed}_{STUDENT_TYPE}'
        write_log([str(test_median)], run_label, log_dir=seed_dir, log_type='test_median_error')
        write_log([str(test_p90)], run_label, log_dir=seed_dir, log_type='test_90_error')
        write_log([str(test_p99)], run_label, log_dir=seed_dir, log_type='test_99_error')

        elapsed = time.time() - epoch_start
        marker = ''
        if test_median < best_median:
            best_median = test_median
            best_p90 = test_p90
            best_p99 = test_p99
            torch.save(student.state_dict(), os.path.join(seed_dir, f'best_{STUDENT_TYPE}.pth'))
            marker = ' *'

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f'  Epoch {epoch}/{EPOCHS} | {elapsed:.0f}s | '
                  f'med={test_median:.3f}m P90={test_p90:.3f}m P99={test_p99:.3f}m{marker}')

    print(f'  Seed {seed} done. Best: med={best_median:.4f}m P90={best_p90:.3f}m P99={best_p99:.3f}m')
    return {'median': best_median, 'p90': best_p90, 'p99': best_p99}


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Student: {STUDENT_TYPE} | Mode: {MODE} | Seeds: {SEEDS}')
    print(f'Epochs: {EPOCHS} | Scheduler: {SCHEDULER_TYPE}')

    # Load data once (shared across seeds)
    print('\nLoading training data...')
    train_w_list, train_lbl_list = [], []
    for fp in trainpath:
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        train_w_list.append(w)
        train_lbl_list.append(lbl)
    train_w = torch.cat(train_w_list, dim=0)
    train_lbl = torch.cat(train_lbl_list, dim=0)
    print(f'Training samples: {train_w.shape[0]}')

    print('Loading testing data...')
    test_w_list, test_lbl_list = [], []
    for fp in testpath:
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        test_w_list.append(w)
        test_lbl_list.append(lbl)
    test_w = torch.cat(test_w_list, dim=0)
    test_lbl = torch.cat(test_lbl_list, dim=0)
    print(f'Testing samples: {test_w.shape[0]}')

    # Load teacher if distilling
    if MODE == 'distill':
        print('Loading teacher outputs...')
        train_teacher = torch.load(TEACHER_TRAIN_PT, weights_only=True)
        test_teacher = torch.load(TEACHER_TEST_PT, weights_only=True)
        assert train_teacher.shape[0] == train_w.shape[0]
        train_data = torch.utils.data.TensorDataset(train_w, train_lbl, train_teacher)
        test_data = torch.utils.data.TensorDataset(test_w, test_lbl, test_teacher)
    else:
        train_data = torch.utils.data.TensorDataset(train_w, train_lbl)
        test_data = torch.utils.data.TensorDataset(test_w, test_lbl)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # Run all seeds
    save_dir = os.path.join('./runs', f'multiseed_{STUDENT_TYPE}_{MODE}_{time.strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        result = train_one_seed(seed, train_loader, test_loader, device, save_dir)
        all_results.append(result)

    # Statistical report
    medians = [r['median'] for r in all_results]
    p90s = [r['p90'] for r in all_results]
    p99s = [r['p99'] for r in all_results]

    print(f'\n\n{"="*65}')
    print(f'  MULTI-SEED RESULTS: {STUDENT_TYPE.upper()} / {MODE.upper()}')
    print(f'{"="*65}')
    for i, seed in enumerate(SEEDS):
        r = all_results[i]
        print(f'  Seed {seed:<5}: median={r["median"]:.4f}m  P90={r["p90"]:.3f}m  P99={r["p99"]:.3f}m')
    print(f'{"-"*65}')
    print(f'  Median error:  {np.mean(medians):.4f} +/- {np.std(medians):.4f} m')
    print(f'  P90 error:     {np.mean(p90s):.3f} +/- {np.std(p90s):.3f} m')
    print(f'  P99 error:     {np.mean(p99s):.3f} +/- {np.std(p99s):.3f} m')
    print(f'{"="*65}')

    # Save report
    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Student: {STUDENT_TYPE}\n')
        f.write(f'Mode: {MODE}\n')
        f.write(f'Scheduler: {SCHEDULER_TYPE}\n')
        f.write(f'Seeds: {SEEDS}\n')
        f.write(f'Epochs: {EPOCHS}\n\n')
        for i, seed in enumerate(SEEDS):
            r = all_results[i]
            f.write(f'Seed {seed}: median={r["median"]:.4f}m  P90={r["p90"]:.3f}m  P99={r["p99"]:.3f}m\n')
        f.write(f'\nMedian: {np.mean(medians):.4f} +/- {np.std(medians):.4f} m\n')
        f.write(f'P90:    {np.mean(p90s):.3f} +/- {np.std(p90s):.3f} m\n')
        f.write(f'P99:    {np.mean(p99s):.3f} +/- {np.std(p99s):.3f} m\n')
    print(f'\nReport saved to: {report_path}')


if __name__ == '__main__':
    main()
