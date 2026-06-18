#!/usr/bin/python
"""
precompute_teacher.py
Pre-computes DLoc teacher outputs for all training samples and saves them.
Run this ONCE before distillation — eliminates teacher forward pass during training.

Usage:
  python precompute_teacher.py

Output:
  ./data/teacher_train_outputs.pt  (~2.3 GB for 17K samples at [1,161,361])
  ./data/teacher_test_outputs.pt
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
import time
import torch
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *
from data_loader import load_data
from modelADT import ModelADT
from joint_model import Enc_2Dec_Network

# ============================================================
# CONFIG
# ============================================================
TEACHER_RUN_DIR = './runs/2026-05-21-06-51-03'
TEACHER_EPOCH = 'best'
BATCH_SIZE = 64

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
# LOAD DATA
# ============================================================
device = torch.device('cuda:0')

print('Loading training data...')
train_wo, train_w, train_lbl = [], [], []
for fp in trainpath:
    wo, w, lbl = load_data(fp)
    if lbl.dim() == 3:
        lbl = lbl.unsqueeze(1)
    train_wo.append(wo)
    train_w.append(w)
    train_lbl.append(lbl)
train_wo = torch.cat(train_wo, dim=0)
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
test_wo = torch.cat(test_wo, dim=0)
test_w = torch.cat(test_w, dim=0)
test_lbl = torch.cat(test_lbl, dim=0)
print(f'Testing samples: {test_w.shape[0]}')

# ============================================================
# LOAD TEACHER
# ============================================================
print('Loading teacher model...')
from params_storage.params_fig10b_single_gpu import opt_exp, opt_encoder, opt_decoder, opt_offset_decoder

opt_encoder.checkpoints_load_dir = TEACHER_RUN_DIR
opt_decoder.checkpoints_load_dir = TEACHER_RUN_DIR
opt_offset_decoder.checkpoints_load_dir = TEACHER_RUN_DIR

enc_model = ModelADT()
enc_model.initialize(opt_encoder)
enc_model.setup(opt_encoder)

dec_model = ModelADT()
dec_model.initialize(opt_decoder)
dec_model.setup(opt_decoder)

offset_dec_model = ModelADT()
offset_dec_model.initialize(opt_offset_decoder)
offset_dec_model.setup(opt_offset_decoder)

teacher = Enc_2Dec_Network()
teacher.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

enc_model.load_networks(TEACHER_EPOCH, load_dir=TEACHER_RUN_DIR)
dec_model.load_networks(TEACHER_EPOCH, load_dir=TEACHER_RUN_DIR)
offset_dec_model.load_networks(TEACHER_EPOCH, load_dir=TEACHER_RUN_DIR)
teacher.eval()
print('Teacher loaded.')

# ============================================================
# PRE-COMPUTE OUTPUTS
# ============================================================
def precompute(name, w_data, lbl_data, wo_data):
    """Run teacher on all samples, return stacked outputs on CPU."""
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(wo_data, w_data, lbl_data),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    outputs = []
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            teacher.set_input(data[1], data[2], data[0], shuffle_channel=False)
            teacher.test()
            outputs.append(teacher.decoder.output.cpu())
            if (i + 1) % 50 == 0:
                print(f'  {name}: {(i+1)*BATCH_SIZE}/{len(w_data)} samples')
    result = torch.cat(outputs, dim=0)
    elapsed = time.time() - t0
    print(f'  {name}: {result.shape} in {elapsed:.1f}s')
    return result

print('\nPre-computing teacher outputs for training set...')
train_teacher_out = precompute('train', train_w, train_lbl, train_wo)

print('Pre-computing teacher outputs for test set...')
test_teacher_out = precompute('test', test_w, test_lbl, test_wo)

# ============================================================
# SAVE
# ============================================================
train_path = './data/teacher_train_outputs.pt'
test_path = './data/teacher_test_outputs.pt'

torch.save(train_teacher_out, train_path)
torch.save(test_teacher_out, test_path)

train_size = os.path.getsize(train_path) / 1e9
test_size = os.path.getsize(test_path) / 1e9
print(f'\nSaved:')
print(f'  {train_path} ({train_size:.2f} GB)')
print(f'  {test_path} ({test_size:.2f} GB)')
print('Done. Now run train_distillation.py')
