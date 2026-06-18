"""
train_unet_advanced.py
Train MobileNetV2-UNet student with LR scheduler comparison (Phase 2, Step 5).

Supports:
  - Option 1: MobileNetV2 + U-Net skip connections (student_unet.py)
  - Option 2: Cosine Annealing LR scheduler
  - Option 3: Step Decay LR scheduler (halve every 20 epochs)
  - Option 4: Knowledge distillation with best scheduler + U-Net
  - MODE: 'standalone' or 'distill'

Usage:
  # Compare schedulers in standalone mode:
  python train_unet_advanced.py   # edit SCHEDULER_TYPE below

  # After finding best scheduler, run distillation:
  # Set MODE='distill' and SCHEDULER_TYPE to the winner

Requires pre-computed teacher outputs for distillation mode
(run precompute_teacher.py first).
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
from student_unet import MobileNetUNetStudent

# ============================================================
# CONFIG
# ============================================================
MODE = 'standalone'           # 'standalone' or 'distill'
SCHEDULER_TYPE = 'cosine'     # 'cosine' or 'step'

EPOCHS = 120
LR = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4

ALPHA = 0.7                   # ground truth loss weight (distillation)
BETA = 0.3                    # teacher loss weight (distillation)

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

# ============================================================
# SETUP
# ============================================================
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
run_label = f'{MODE}_unet_{SCHEDULER_TYPE}'
save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
save_dir = os.path.join('./runs', f'{run_label}_{save_name}')
os.makedirs(save_dir, exist_ok=True)

print(f'Device: {device}')
print(f'Mode: {MODE} | Scheduler: {SCHEDULER_TYPE} | Epochs: {EPOCHS}')
print(f'Save dir: {save_dir}')

# ============================================================
# LOAD DATA
# ============================================================
print('Loading training data...')
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

# Load teacher outputs if distilling
if MODE == 'distill':
    print('Loading pre-computed teacher outputs...')
    train_teacher = torch.load(TEACHER_TRAIN_PT, weights_only=True)
    test_teacher = torch.load(TEACHER_TEST_PT, weights_only=True)
    assert train_teacher.shape[0] == train_w.shape[0], \
        f'Teacher/data mismatch: {train_teacher.shape[0]} vs {train_w.shape[0]}'
    print(f'Teacher outputs: train={train_teacher.shape}, test={test_teacher.shape}')
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
# MODEL + OPTIMIZER + SCHEDULER
# ============================================================
student = MobileNetUNetStudent(input_nc=4).to(device)
n_params = sum(p.numel() for p in student.parameters())
print(f'U-Net student params: {n_params:,} ({n_params/1e6:.2f}M)')

optimizer = torch.optim.Adam(student.parameters(), lr=LR)
criterion = nn.MSELoss()

if SCHEDULER_TYPE == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif SCHEDULER_TYPE == 'step':
    # Halve LR every 20 epochs (professor's spec)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
else:
    raise ValueError(f'Unknown scheduler: {SCHEDULER_TYPE}')

# ============================================================
# TRAINING LOOP
# ============================================================
print(f'Starting {MODE} training with {SCHEDULER_TYPE} scheduler...')
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
    current_lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - epoch_start

    # Step scheduler
    scheduler.step()

    # --- LOG ---
    write_log([str(train_loss_avg)], run_label, log_dir=save_dir, log_type='train_loss')
    write_log([str(train_median)], run_label, log_dir=save_dir, log_type='train_median_error')
    write_log([str(test_loss_avg)], run_label, log_dir=save_dir, log_type='test_loss')
    write_log([str(test_median)], run_label, log_dir=save_dir, log_type='test_median_error')
    write_log([str(test_p90)], run_label, log_dir=save_dir, log_type='test_90_error')
    write_log([str(test_p99)], run_label, log_dir=save_dir, log_type='test_99_error')

    marker = ' <-- NEW BEST' if test_median < best_median else ''
    print(f'Epoch {epoch}/{EPOCHS} | {elapsed:.0f}s | LR={current_lr:.6f} | '
          f'train_med={train_median:.3f}m | '
          f'test_med={test_median:.3f}m P90={test_p90:.3f}m P99={test_p99:.3f}m{marker}')

    # --- SAVE ---
    if test_median < best_median:
        best_median = test_median
        torch.save(student.state_dict(), os.path.join(save_dir, 'best_unet.pth'))

    torch.save(student.state_dict(), os.path.join(save_dir, 'latest_unet.pth'))

print(f'\nDone. Best median error: {best_median:.4f}m')
print(f'Scheduler: {SCHEDULER_TYPE} | Mode: {MODE}')
print(f'Results saved to: {save_dir}')
