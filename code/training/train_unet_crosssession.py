"""
train_unet_crosssession.py
Train the MobileNetV2-UNet student under the OFFICIAL DLoc Table 1
leave-one-session-out protocol, for a fair comparison against the DLoc baseline.

The fold file lists mirror the data tags in train_and_test.py exactly:
  env2 -> train {Jul28, Jul28_2, Aug16_3, Aug16_4_ref}, test Aug16_1
  env3 -> train {Jul28, Jul28_2, Aug16_1, Aug16_4_ref}, test Aug16_3
  env4 -> train {Jul28, Jul28_2, Aug16_1, Aug16_3},     test Aug16_4_ref

Standalone training (no distillation): U-Net distillation gave no median gain in
Phase 2, and a teacher trained on the Jul28 split would not match these folds.

Usage:
  python train_unet_crosssession.py --env rw_to_rw_env2 --seed 42
  python train_unet_crosssession.py --env rw_to_rw_env4 --seed 123 --epochs 120

Writes a per-run JSON to ./results_crosssession/unet_<env>_seed<seed>.json
and per-epoch logs to ./runs/unet_<env>_seed<seed>_<timestamp>/.
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
import json
import argparse
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import numpy as np

from utils import localization_error, write_log
from data_loader import load_data
from student_unet import MobileNetUNetStudent

# ------------------------------------------------------------
# Fold definitions (must match train_and_test.py env tags)
# ------------------------------------------------------------
BASE_TRAIN = [
    './data/dataset_jacobs_July28.mat',
    './data/dataset_non_fov_train_jacobs_July28_2.mat',
    './data/dataset_fov_train_jacobs_July28_2.mat',
]
AUG16 = {
    'env2_test': './data/dataset_train_jacobs_Aug16_1.mat',
    'env3_test': './data/dataset_train_jacobs_Aug16_3.mat',
    'env4_test': './data/dataset_train_jacobs_Aug16_4_ref.mat',
}
FOLDS = {
    # Fig 10b: in-distribution complex space (train/test on Jul28 + Jul28_2 splits)
    'rw_to_rw': {
        'train': BASE_TRAIN,
        'test':  ['./data/dataset_fov_test_jacobs_July28_2.mat',
                  './data/dataset_non_fov_test_jacobs_July28_2.mat'],
    },
    # Table 1: leave-one-furniture-setup-out (always train on Jul28 + 2 Aug16, test held-out)
    'rw_to_rw_env2': {
        'train': BASE_TRAIN + [AUG16['env3_test'], AUG16['env4_test']],
        'test':  [AUG16['env2_test']],
    },
    'rw_to_rw_env3': {
        'train': BASE_TRAIN + [AUG16['env2_test'], AUG16['env4_test']],
        'test':  [AUG16['env3_test']],
    },
    'rw_to_rw_env4': {
        'train': BASE_TRAIN + [AUG16['env2_test'], AUG16['env3_test']],
        'test':  [AUG16['env4_test']],
    },
    # Fig 13b: generalization across space (disjoint spatial train/test regions)
    'data_segment': {
        'train': ['./data/dataset_train_jacobs_July28.mat',
                  './data/dataset_train_jacobs_July28_2.mat'],
        'test':  ['./data/dataset_test_jacobs_July28.mat',
                  './data/dataset_test_jacobs_July28_2.mat'],
    },
}


def load_files(file_paths):
    w_list, lbl_list = [], []
    for fp in file_paths:
        if not os.path.exists(fp):
            sys.exit(f'ERROR: missing data file {fp}\n'
                     f'  (Table 1 folds need the dataset_train_jacobs_Aug16_*.mat symlinks; '
                     f'see CROSSSESSION_RUNBOOK.md)')
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        w_list.append(w)
        lbl_list.append(lbl)
    return torch.cat(w_list, dim=0), torch.cat(lbl_list, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, choices=list(FOLDS.keys()))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fold = FOLDS[args.env]
    tag = f'unet_{args.env}_seed{args.seed}'
    save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('./runs', f'{tag}_{save_name}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./results_crosssession', exist_ok=True)

    print(f'Device: {device} | Fold: {args.env} | Seed: {args.seed} | Epochs: {args.epochs}')
    print(f'Save dir: {save_dir}')

    print('Loading training data...')
    train_w, train_lbl = load_files(fold['train'])
    print(f'Training samples: {train_w.shape[0]}')
    print('Loading testing data...')
    test_w, test_lbl = load_files(fold['test'])
    print(f'Testing samples: {test_w.shape[0]}')

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_w, train_lbl),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_w, test_lbl),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
    )

    student = MobileNetUNetStudent(input_nc=4).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'U-Net student params: {n_params:,} ({n_params/1e6:.2f}M)')

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best = {'median': float('inf'), 'p90': float('inf'), 'p99': float('inf'), 'epoch': -1}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        student.train()
        train_loss_sum, train_errors = 0.0, []
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            out = student(inputs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_errors.extend(localization_error(
                out.detach().cpu().numpy(), labels.cpu().numpy(), scale=0.1))
        train_median = np.median(train_errors)

        student.eval()
        test_loss_sum, test_errors = 0.0, []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                out = student(inputs)
                test_loss_sum += criterion(out, labels).item()
                test_errors.extend(localization_error(
                    out.cpu().numpy(), labels.cpu().numpy(), scale=0.1))
        test_median = float(np.median(test_errors))
        test_p90 = float(np.percentile(test_errors, 90))
        test_p99 = float(np.percentile(test_errors, 99))
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        write_log([str(train_median)], tag, log_dir=save_dir, log_type='train_median_error')
        write_log([str(test_median)], tag, log_dir=save_dir, log_type='test_median_error')
        write_log([str(test_p90)], tag, log_dir=save_dir, log_type='test_90_error')
        write_log([str(test_p99)], tag, log_dir=save_dir, log_type='test_99_error')

        is_best = test_median < best['median']
        marker = ' <-- NEW BEST' if is_best else ''
        print(f'Epoch {epoch}/{args.epochs} | {time.time()-epoch_start:.0f}s | '
              f'LR={current_lr:.6f} | train_med={train_median:.3f}m | '
              f'test_med={test_median:.3f}m P90={test_p90:.3f}m P99={test_p99:.3f}m{marker}')

        if is_best:
            best = {'median': test_median, 'p90': test_p90, 'p99': test_p99, 'epoch': epoch}
            torch.save(student.state_dict(), os.path.join(save_dir, 'best_unet.pth'))
        torch.save(student.state_dict(), os.path.join(save_dir, 'latest_unet.pth'))

    result = {
        'model': 'unet', 'env': args.env, 'seed': args.seed,
        'epochs': args.epochs, 'n_params': n_params,
        'best_median': best['median'], 'best_p90': best['p90'],
        'best_p99': best['p99'], 'best_epoch': best['epoch'],
        'run_dir': save_dir,
    }
    out_json = f'./results_crosssession/{tag}.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nDone. Best: median={best["median"]:.4f}m P90={best["p90"]:.3f}m '
          f'P99={best["p99"]:.3f}m @epoch{best["epoch"]}')
    print(f'Result written to {out_json}')


if __name__ == '__main__':
    main()
