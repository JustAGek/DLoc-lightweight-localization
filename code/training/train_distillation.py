#!/usr/bin/python
"""
train_distillation.py
Knowledge distillation: DLoc teacher -> lightweight student.

Loss: L = ALPHA * MSE(student, ground_truth) + BETA * MSE(student, teacher_output)
Default: ALPHA=0.7, BETA=0.3 (from professor's spec)

Requires pre-computed teacher outputs (run precompute_teacher.py first).
Set MODE='standalone' to train without distillation (ground truth only).

Usage:
  python precompute_teacher.py          # run once
  python train_distillation.py          # distillation (default: mobilenet)

  To change student or mode, edit CONFIG below.
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import localization_error, write_log
from data_loader import load_data
from student_mobilenet import MobileNetStudent
from student_tinycnn import TinyCNNStudent

# ============================================================
# CONFIG — edit these before running
# ============================================================
STUDENT_TYPE = 'mobilenet'          # 'mobilenet' or 'tinycnn'
MODE = 'distill'                    # 'distill' or 'standalone'

ALPHA = 0.7                         # ground truth loss weight
BETA = 0.3                          # distillation loss weight (ignored in standalone)
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4

TEACHER_TRAIN_PT = './data/teacher_train_outputs.pt'
TEACHER_TEST_PT = './data/teacher_test_outputs.pt'

# ============================================================
# DATA PATHS (Fig 10b — Jacobs Hall)
# ============================================================
trainpath = [
    './data/dataset_jacobs_July28.mat',
    './data/dataset_non_fov_train_jacobs_July28_2.mat',
    './data/dataset_fov_train_jacobs_July28_2.mat',
]
testpath = [
    './data/dataset_fov_test_jacobs_July28_2.mat',
    './data/dataset_non_fov_test_jacobs_July28_2.mat',
]

# ============================================================
# SETUP
# ============================================================
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0')
save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
run_label = f'{MODE}_{STUDENT_TYPE}'
save_dir = os.path.join('./runs', f'{run_label}_{save_name}')
os.makedirs(save_dir, exist_ok=True)

print(f'Mode: {MODE}')
print(f'Student: {STUDENT_TYPE}')
if MODE == 'distill':
    print(f'Loss: {ALPHA}*L_GT + {BETA}*L_KD')
else:
    print(f'Loss: L_GT (standalone, no distillation)')
print(f'Save dir: {save_dir}')

# ============================================================
# LOAD DATA INTO RAM
# ============================================================
print('Loading training data...')
train_wo, train_w, train_lbl = [], [], []
for fp in trainpath:
    wo, w, lbl = load_data(fp)
    if lbl.dim() == 3:
        lbl = lbl.unsqueeze(1)
    train_wo.append(wo)
    train_w.append(w)
    train_lbl.append(lbl)
train_w = torch.cat(train_w, dim=0)
train_lbl = torch.cat(train_lbl, dim=0)
print(f'Training samples: {train_w.shape[0]}')

print('Loading testing data...')
test_wo, test_w, test_lbl = [], [], []
for fp in testpath:
    wo, w, lbl = load_data(fp)
    if lbl.dim() == 3:
        lbl = lbl.unsqueeze(1)
    test_wo.append(wo)
    test_w.append(w)
    test_lbl.append(lbl)
test_w = torch.cat(test_w, dim=0)
test_lbl = torch.cat(test_lbl, dim=0)
print(f'Testing samples: {test_w.shape[0]}')

# Load pre-computed teacher outputs if distilling
if MODE == 'distill':
    print('Loading pre-computed teacher outputs...')
    train_teacher = torch.load(TEACHER_TRAIN_PT, weights_only=True)
    test_teacher = torch.load(TEACHER_TEST_PT, weights_only=True)
    assert train_teacher.shape[0] == train_w.shape[0], \
        f'Teacher/data mismatch: {train_teacher.shape[0]} vs {train_w.shape[0]}'
    print(f'Teacher outputs loaded: train={train_teacher.shape}, test={test_teacher.shape}')
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

# ============================================================
# CREATE STUDENT
# ============================================================
if STUDENT_TYPE == 'mobilenet':
    student = MobileNetStudent(input_nc=4).to(device)
elif STUDENT_TYPE == 'tinycnn':
    student = TinyCNNStudent(input_nc=4).to(device)
else:
    raise ValueError(f'Unknown student type: {STUDENT_TYPE}')

n_params = sum(p.numel() for p in student.parameters())
print(f'Student params: {n_params:,} ({n_params/1e6:.2f}M)')

optimizer = torch.optim.Adam(student.parameters(), lr=LR)
criterion = nn.MSELoss()

# ============================================================
# TRAINING LOOP
# ============================================================
print(f'Starting {MODE} training...')
best_median = float('inf')

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # --- TRAIN ---
    student.train()
    train_loss_sum = 0
    train_errors = []

    for data in train_loader:
        inputs = data[0].to(device)
        labels = data[1].to(device)

        student_output = student(inputs)
        L_GT = criterion(student_output, labels)

        if MODE == 'distill':
            teacher_out = data[2].to(device)
            L_KD = criterion(student_output, teacher_out)
            loss = ALPHA * L_GT + BETA * L_KD
        else:
            loss = L_GT

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_errors.extend(localization_error(
            student_output.detach().cpu().numpy(),
            labels.cpu().numpy(),
            scale=0.1,
        ))

    train_loss_avg = train_loss_sum / len(train_loader)
    train_median = np.median(train_errors)

    # --- TEST ---
    student.eval()
    test_loss_sum = 0
    test_errors = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            student_output = student(inputs)
            loss = criterion(student_output, labels)

            test_loss_sum += loss.item()
            test_errors.extend(localization_error(
                student_output.cpu().numpy(),
                labels.cpu().numpy(),
                scale=0.1,
            ))

    test_loss_avg = test_loss_sum / len(test_loader)
    test_median = np.median(test_errors)
    test_p90 = np.percentile(test_errors, 90)
    test_p99 = np.percentile(test_errors, 99)
    elapsed = time.time() - epoch_start

    # --- LOG ---
    write_log([str(train_loss_avg)], run_label, log_dir=save_dir, log_type='train_loss')
    write_log([str(train_median)], run_label, log_dir=save_dir, log_type='train_median_error')
    write_log([str(test_loss_avg)], run_label, log_dir=save_dir, log_type='test_loss')
    write_log([str(test_median)], run_label, log_dir=save_dir, log_type='test_median_error')
    write_log([str(test_p90)], run_label, log_dir=save_dir, log_type='test_90_error')
    write_log([str(test_p99)], run_label, log_dir=save_dir, log_type='test_99_error')

    print(f'Epoch {epoch}/{EPOCHS} | {elapsed:.0f}s | '
          f'train_loss={train_loss_avg:.6f} train_med={train_median:.3f}m | '
          f'test_med={test_median:.3f}m P90={test_p90:.3f}m P99={test_p99:.3f}m')

    # --- SAVE ---
    if test_median < best_median:
        best_median = test_median
        torch.save(student.state_dict(), os.path.join(save_dir, f'best_{STUDENT_TYPE}.pth'))
        print(f'  -> New best: {best_median:.4f}m')

    torch.save(student.state_dict(), os.path.join(save_dir, f'latest_{STUDENT_TYPE}.pth'))

print(f'\nDone. Best median error: {best_median:.4f}m')
print(f'Results saved to: {save_dir}')
