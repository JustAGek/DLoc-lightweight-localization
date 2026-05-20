# Mamba-DLoc + Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix data split, RNG persistence, and fix_split.py path bugs from Report_Grad.pdf, then implement a Mamba-based DLoc variant integrated into the existing training pipeline.

**Architecture:** The Mamba model replaces DLoc's 6 ResNet bottleneck blocks with 4 Mamba selective state-space blocks. A CNN stem handles local spatial extraction (4→32→64→128 channels with stride-2 downsampling), the feature map is tokenized into a 3731-length sequence for Mamba processing, then reshaped back and decoded by a CNN decoder to produce a [B,1,161,361] location heatmap. Integration uses the existing `train_and_test.py` pipeline via a `model_type="mamba"` config option.

**Tech Stack:** PyTorch, mamba-ssm (Linux, with pure PyTorch fallback for Windows), h5py, easydict

**Spec:** `docs/superpowers/specs/2026-05-20-mamba-dloc-and-bugfixes-design.md`

---

### Task 1: Fix data split path bug in fix_split.py

**Files:**
- Modify: `DLoc_pt_code/fix_split.py:19`

- [ ] **Step 1: Fix the split file path**

In `fix_split.py`, line 19 points to `./data/` but split indices are in `../data_split_idx/`. Change:

```python
# Old:
SPLIT_FILE = os.path.join(DATA_DIR, 'data_split_ids_jacobs_July28.mat')

# New:
SPLIT_FILE = os.path.join('..', 'data_split_idx', 'data_split_ids_jacobs_July28.mat')
```

- [ ] **Step 2: Verify the fix resolves the path**

Run from `DLoc_pt_code/`:
```bash
python -c "import os; print(os.path.exists(os.path.join('..', 'data_split_idx', 'data_split_ids_jacobs_July28.mat')))"
```
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add fix_split.py
git commit -m "fix: correct split index path in fix_split.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Fix data split — regenerate dataset_jacobs_July28.mat

**Files:**
- No code changes — run existing `prepare_data.py` (which already has the correct path)
- Affected data: `DLoc_pt_code/data/dataset_jacobs_July28.mat`

**Prereq:** The raw source file `features_jacobs_Jul28.mat` must exist in `DLoc_pt_code/data/`. If it doesn't exist (only the pre-split dataset files exist), use `fix_split.py` from Task 1 instead, which reads from the current (wrong) `dataset_jacobs_July28.mat` and extracts only the correct training indices.

- [ ] **Step 1: Check if raw source file exists**

```bash
cd DLoc_pt_code
python -c "import os; print('RAW EXISTS' if os.path.exists('data/features_jacobs_Jul28.mat') else 'RAW MISSING — use fix_split.py instead')"
```

- [ ] **Step 2a: If raw file EXISTS — run prepare_data.py**

```bash
cd DLoc_pt_code
python prepare_data.py
```

This regenerates `dataset_jacobs_July28.mat` with only 7,635 training samples.

- [ ] **Step 2b: If raw file MISSING — run fix_split.py**

```bash
cd DLoc_pt_code
python fix_split.py
```

This extracts the correct 7,635 training samples from the current (wrong) 11,440-sample file. The wrong file is backed up as `dataset_jacobs_July28_WRONG.mat`.

- [ ] **Step 3: Verify the corrected file**

```bash
cd DLoc_pt_code
python -c "
import h5py
with h5py.File('data/dataset_jacobs_July28.mat', 'r') as f:
    k = 'features_w_offset' if 'features_w_offset' in f else 'features_with_offset'
    n = f[k].shape[-1]
    print(f'Samples: {n}')
    assert n == 7635, f'WRONG: expected 7635, got {n}'
    print('OK: dataset_jacobs_July28.mat has correct 7,635 samples')
"
```

- [ ] **Step 4: Verify total train/test counts**

```bash
cd DLoc_pt_code
python check_data.py
```

Expected output should show:
- `dataset_jacobs_July28.mat`: 7,635 samples (TRAIN)
- Total train: ~17,574
- Total test: ~4,260

---

### Task 3: Add RNG state save/restore to modelADT.py

**Files:**
- Modify: `DLoc_pt_code/modelADT.py:147-154` (save_networks) and `DLoc_pt_code/modelADT.py:200-210` (load_training_state)

- [ ] **Step 1: Add RNG state to save_networks**

In `modelADT.py`, modify the `save_networks` method. The existing code at lines 147-154 saves optimizer and scheduler state. Add RNG state to the same dict:

```python
        # save optimizer and scheduler state
        if self.isTrain:
            optim_filename = '%s_optim_%s.pth' % (epoch, self.model_name)
            optim_path = os.path.join(self.save_dir, optim_filename)
            save_dict = {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.schedulers[0].state_dict() if self.schedulers else None,
                'rng_torch': torch.random.get_rng_state(),
                'rng_numpy': np.random.get_state(),
            }
            if torch.cuda.is_available():
                save_dict['rng_cuda'] = torch.cuda.get_rng_state_all()
            torch.save(save_dict, optim_path)
```

Also add `import numpy as np` at the top of the file if not already present.

- [ ] **Step 2: Add RNG state restore to load_training_state**

In `modelADT.py`, modify the `load_training_state` method (lines 200-210) to restore RNG states:

```python
    def load_training_state(self, epoch):
        optim_filename = f'{epoch}_optim_{self.model_name}.pth'
        load_path = os.path.join(self.load_dir, optim_filename)
        if os.path.exists(load_path):
            print(f'loading optimizer/scheduler from {load_path}')
            state = torch.load(load_path, weights_only=False)
            self.optimizer.load_state_dict(state['optimizer'])
            if state.get('scheduler') and self.schedulers:
                self.schedulers[0].load_state_dict(state['scheduler'])
            # Restore RNG states for session continuity
            if 'rng_torch' in state:
                torch.random.set_rng_state(state['rng_torch'])
            if 'rng_numpy' in state:
                np.random.set_state(state['rng_numpy'])
            if 'rng_cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['rng_cuda'])
        else:
            print(f'No optimizer state found at {load_path}, starting fresh')
```

- [ ] **Step 3: Verify import**

Ensure `import numpy as np` exists at the top of `modelADT.py`. Add it after the existing imports if missing.

- [ ] **Step 4: Commit**

```bash
git add modelADT.py
git commit -m "fix: save and restore RNG state across training sessions

Prevents random seed reset when resuming multi-session training,
ensuring consistent data shuffle order and reproducible results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Create mamba_model.py — the Mamba-DLoc architecture

**Files:**
- Create: `DLoc_pt_code/mamba_model.py`

This is the core new file. It contains the full end-to-end Mamba-DLoc network.

- [ ] **Step 1: Create mamba_model.py with all components**

Create `DLoc_pt_code/mamba_model.py` with this content:

```python
"""
mamba_model.py
--------------
Mamba-DLoc: replaces DLoc's 6 ResNet bottleneck blocks with 4 Mamba
selective state-space blocks for global spatial reasoning.

Architecture:
    CNN Stem (4->32->64->128) -> Tokenize (41*91=3731 tokens)
    -> 4x MambaBlock -> Reshape -> CNN Decoder -> Sigmoid heatmap

Uses mamba-ssm CUDA kernels on Linux, pure PyTorch fallback on Windows.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the fast CUDA Mamba; fall back to pure PyTorch
try:
    from mamba_ssm import Mamba as CudaMamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


# ---------------------------------------------------------------------------
# Pure PyTorch Mamba fallback (works on Windows / CPU / any platform)
# ---------------------------------------------------------------------------
class PurePytorchMamba(nn.Module):
    """Minimal Mamba (S6) block in pure PyTorch.

    Matches the mamba-ssm.Mamba interface:
        input:  [B, L, D]
        output: [B, L, D]

    Args:
        d_model:  model dimension D
        d_state:  SSM state dimension N
        d_conv:   local conv kernel size
        expand:   expansion factor for inner dimension E = expand * D
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection: D -> 2*E (split into x and gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local convolution on main path
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Learnable SSM parameters
        # A is initialized to negative log-spaced values (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection: E -> D
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: [B, L, D] -> [B, L, D]"""
        B, L, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)                        # [B, L, 2*E]
        x_main, z = xz.chunk(2, dim=-1)             # each [B, L, E]

        # Local conv (causal)
        x_main = x_main.transpose(1, 2)             # [B, E, L]
        x_main = self.conv1d(x_main)[:, :, :L]      # causal: trim to L
        x_main = x_main.transpose(1, 2)             # [B, L, E]
        x_main = F.silu(x_main)

        # SSM parameter computation
        ssm_params = self.x_proj(x_main)             # [B, L, dt_rank + 2*N]
        dt, B_param, C_param = ssm_params.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))             # [B, L, E]

        # Discretize continuous SSM
        A = -torch.exp(self.A_log)                    # [E, N]
        dA = torch.exp(dt.unsqueeze(-1) * A)          # [B, L, E, N]
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # [B, L, E, N]

        # Selective scan (sequential — correct but not optimized)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_main[:, t].unsqueeze(-1)  # [B, E, N]
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)         # [B, E]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                    # [B, L, E]

        # Skip connection with D
        y = y + x_main * self.D                       # [B, L, E]

        # Gate and output
        y = y * F.silu(z)                             # [B, L, E]
        return self.out_proj(y)                       # [B, L, D]


# ---------------------------------------------------------------------------
# Mamba block with residual + LayerNorm
# ---------------------------------------------------------------------------
class MambaResidualBlock(nn.Module):
    """LayerNorm -> Mamba -> residual add."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA_SSM:
            self.mamba = CudaMamba(d_model=d_model, d_state=d_state,
                                  d_conv=d_conv, expand=expand)
        else:
            self.mamba = PurePytorchMamba(d_model=d_model, d_state=d_state,
                                         d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Full Mamba-DLoc end-to-end network
# ---------------------------------------------------------------------------
class MambaDLocNet(nn.Module):
    """
    Mamba-DLoc: CNN stem encoder + Mamba global modeling + CNN decoder.

    Input:  [B, input_nc, 161, 361]  (AoA heatmaps from N APs)
    Output: [B, 1, 161, 361]         (location heatmap)

    Args:
        input_nc:   number of input channels (typically 4 for 4 APs)
        d_model:    Mamba model dimension (channels after stem = 128)
        n_mamba:    number of Mamba blocks (default 4)
        d_state:    SSM state dimension (default 16)
        d_conv:     Mamba local conv kernel size (default 4)
        expand:     Mamba expansion factor (default 2)
    """

    # Spatial sizes after each downsampling stage (for exact upsampling)
    SPATIAL_SIZES = [(161, 361), (81, 181), (41, 91)]

    def __init__(self, input_nc=4, d_model=128, n_mamba=4,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()

        # --- CNN Stem Encoder ---
        self.stem = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # --- Mamba blocks ---
        self.mamba_blocks = nn.Sequential(
            *[MambaResidualBlock(d_model, d_state, d_conv, expand)
              for _ in range(n_mamba)]
        )

        # --- CNN Decoder ---
        # Stage 1: [B, 128, 41, 91] -> interpolate to [81, 181] -> conv
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 2: [B, 64, 81, 181] -> interpolate to [161, 361] -> conv
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial sizes for the decoder upsampling
        self._up_size1 = self.SPATIAL_SIZES[1]   # (81, 181)
        self._up_size0 = self.SPATIAL_SIZES[0]   # (161, 361)

        # Store for reshape
        self._feat_h = self.SPATIAL_SIZES[2][0]  # 41
        self._feat_w = self.SPATIAL_SIZES[2][1]  # 91
        self.d_model = d_model

    def forward(self, x):
        B = x.shape[0]

        # CNN stem: [B, 4, 161, 361] -> [B, 128, 41, 91]
        feat = self.stem(x)

        # Tokenize: [B, 128, 41, 91] -> [B, 3731, 128]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Mamba global modeling: [B, 3731, 128] -> [B, 3731, 128]
        tokens = self.mamba_blocks(tokens)

        # Reshape back: [B, 3731, 128] -> [B, 128, 41, 91]
        feat = tokens.transpose(1, 2).view(B, self.d_model, self._feat_h, self._feat_w)

        # Decoder stage 1: upsample to (81, 181) + conv
        feat = F.interpolate(feat, size=self._up_size1, mode='bilinear', align_corners=False)
        feat = self.dec_conv1(feat)

        # Decoder stage 2: upsample to (161, 361) + conv
        feat = F.interpolate(feat, size=self._up_size0, mode='bilinear', align_corners=False)
        feat = self.dec_conv2(feat)

        # Heatmap head
        return self.head(feat)
```

- [ ] **Step 2: Verify module loads and shapes are correct**

Run from `DLoc_pt_code/`:

```bash
python -c "
import torch
from mamba_model import MambaDLocNet, HAS_MAMBA_SSM
print(f'Using mamba-ssm CUDA: {HAS_MAMBA_SSM}')

model = MambaDLocNet(input_nc=4)
x = torch.randn(2, 4, 161, 361)
with torch.no_grad():
    y = model(x)
print(f'Input:  {x.shape}')
print(f'Output: {y.shape}')
assert y.shape == (2, 1, 161, 361), f'Wrong output shape: {y.shape}'

n_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n_params:,} ({n_params/1e6:.2f}M)')
print('OK: MambaDLocNet forward pass works')
"
```

Expected output (approximately):
```
Using mamba-ssm CUDA: False
Input:  torch.Size([2, 4, 161, 361])
Output: torch.Size([2, 1, 161, 361])
Parameters: ~1,200,000 (1.20M)
OK: MambaDLocNet forward pass works
```

- [ ] **Step 3: Commit**

```bash
git add mamba_model.py
git commit -m "feat: add MambaDLocNet architecture with pure PyTorch fallback

CNN stem (4->32->64->128) + 4 Mamba blocks + CNN decoder.
Uses mamba-ssm CUDA kernels when available, falls back to pure
PyTorch selective scan on Windows/CPU.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Register MambaDLocNet in the model factory (utils.py)

**Files:**
- Modify: `DLoc_pt_code/utils.py:14` (imports) and `DLoc_pt_code/utils.py:39-46` (define_G)

- [ ] **Step 1: Add import for MambaDLocNet**

At the top of `utils.py`, after `from Generators import *` (line 14), add:

```python
from mamba_model import MambaDLocNet
```

- [ ] **Step 2: Add mamba_dloc branch to define_G**

In `utils.py`, in the `define_G` function, before the `else: raise NotImplementedError` (line 45-46), add:

```python
    elif net_type == 'mamba_dloc':
        net = MambaDLocNet(input_nc=input_nc)
```

The full `define_G` if/elif chain should now be:
```python
    if net_type == 'resnet_encoder':
        ...
    elif net_type == 'resnet_decoder':
        ...
    elif net_type == 'mamba_dloc':
        net = MambaDLocNet(input_nc=input_nc)
    else:
        raise NotImplementedError(...)
```

- [ ] **Step 3: Verify define_G can create the Mamba model**

```bash
cd DLoc_pt_code
python -c "
from utils import define_G
from easydict import EasyDict as edict
opt = edict(input_nc=4, output_nc=1, ngf=64, base_model='mamba_dloc',
            norm='batch', no_dropout=False, init_type='xavier', init_gain=0.02)
net = define_G(opt, gpu_ids=[])
print(f'Created via factory: {type(net).__name__}')
print('OK')
"
```

Expected: `Created via factory: MambaDLocNet` and `OK`

- [ ] **Step 4: Commit**

```bash
git add utils.py
git commit -m "feat: register MambaDLocNet in define_G model factory

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Add Mamba_Network joint model class (joint_model.py)

**Files:**
- Modify: `DLoc_pt_code/joint_model.py` — add new class at the end

- [ ] **Step 1: Add _EncoderProxy and Mamba_Network to joint_model.py**

Append the following classes at the end of `joint_model.py` (after the `Enc_2Dec_Network` class):

```python
class _EncoderProxy:
    """Lightweight proxy that shares opt/optimizer but no-ops on LR update.

    trainer.py calls update_learning_rate() on both model.encoder and
    model.decoder.  For the Mamba pipeline there is only one optimizer,
    so this proxy prevents double-stepping the LR scheduler.
    """

    def __init__(self, model_adt):
        self.opt = model_adt.opt
        self.optimizer = model_adt.optimizer

    def update_learning_rate(self):
        pass  # decoder handles the single scheduler

    def save_networks(self, epoch):
        pass  # decoder handles saving


class Mamba_Network():
    """Joint-model wrapper for the single end-to-end MambaDLocNet.

    Exposes the same interface that trainer.py expects
    (model.encoder, model.decoder, set_input, optimize_parameters, …)
    while internally using a single ModelADT with one optimizer.
    """

    def initialize(self, opt, mamba_model_adt, gpu_ids='0'):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.decoder = mamba_model_adt           # trainer reads .output, .loss, .model_name, .opt
        self.encoder = _EncoderProxy(mamba_model_adt)  # no-op LR update
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def set_input(self, input, target, shuffle_channel=False):
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.decoder.set_data(self.input, self.target, shuffle_channel=shuffle_channel)

    def save_networks(self, epoch):
        self.decoder.save_networks(epoch)

    def update_learning_rate(self):
        self.decoder.update_learning_rate()

    def forward(self):
        self.decoder.forward()

    def test(self):
        self.decoder.test()

    def backward(self):
        self.decoder.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward()
        self.decoder.optimizer.step()
        self.decoder.optimizer.zero_grad()

    def eval(self):
        self.decoder.eval()
```

- [ ] **Step 2: Commit**

```bash
git add joint_model.py
git commit -m "feat: add Mamba_Network joint model wrapper

Single-optimizer wrapper that exposes the same interface as
Enc_Dec_Network for trainer.py compatibility. Uses _EncoderProxy
to prevent double LR scheduler stepping.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Create Mamba params file

**Files:**
- Create: `DLoc_pt_code/params_storage/params_mamba_single_gpu.py`

- [ ] **Step 1: Create the params file**

Create `DLoc_pt_code/params_storage/params_mamba_single_gpu.py`:

```python
# params_mamba_single_gpu.py
# ===========================
# Mamba-DLoc: replaces ResNet bottleneck with Mamba SSM blocks.
# Single decoder (location only, no consistency decoder).
#
# To use: copy this file to ../params.py
#   cp params_storage/params_mamba_single_gpu.py params.py
# Then run: python train_and_test.py
#
# Required data files in ./data/ (same as Fig 10b):
#   dataset_jacobs_July28.mat          (must be the FIXED 7,635-sample version)
#   dataset_fov_train_jacobs_July28_2.mat
#   dataset_non_fov_train_jacobs_July28_2.mat
#   dataset_fov_test_jacobs_July28_2.mat
#   dataset_non_fov_test_jacobs_July28_2.mat

from easydict import EasyDict as edict
import time
from os.path import join

opt_exp = edict()

# ---------- Global Experiment params ----------
opt_exp.isTrain = True
opt_exp.continue_train = False
opt_exp.starting_epoch_count = 0
opt_exp.n_epochs = 50
opt_exp.gpu_ids = ['0']
opt_exp.data = "rw_to_rw"           # Same Jacobs Hall dataset as Fig 10b
opt_exp.n_decoders = 1               # Single decoder (location only)
opt_exp.model_type = "mamba"         # Selects Mamba pipeline in train_and_test.py

opt_exp.batch_size = 8
opt_exp.ds_step_trn = 1
opt_exp.ds_step_tst = 1
opt_exp.weight_decay = 1e-5
opt_exp.confidence = False

# ------ Experiment save paths ----------
opt_exp.save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
opt_exp.checkpoints_dir = join('./runs', opt_exp.save_name)
opt_exp.results_dir = opt_exp.checkpoints_dir
opt_exp.log_dir = opt_exp.checkpoints_dir
opt_exp.load_dir = opt_exp.checkpoints_dir

# ---------- Mamba model params (single model, acts as both encoder+decoder) ----------
# This is used by ModelADT to create and train the MambaDLocNet.
# The model is end-to-end: no separate encoder/decoder split.
opt_mamba = edict()
opt_mamba.parent_exp = opt_exp
opt_mamba.batch_size = opt_exp.batch_size
opt_mamba.ngf = 64                   # not used by MambaDLocNet but required by ModelADT
opt_mamba.base_model = 'mamba_dloc'  # selects MambaDLocNet in define_G
opt_mamba.net = 'G'
opt_mamba.no_dropout = False
opt_mamba.init_type = 'xavier'
opt_mamba.init_gain = 0.02
opt_mamba.norm = 'batch'
opt_mamba.beta1 = 0.5
opt_mamba.lr = 0.00001
opt_mamba.lr_policy = 'step'
opt_mamba.lr_decay_iters = 20
opt_mamba.lambda_L = 1
opt_mamba.lambda_cross = 0
opt_mamba.lambda_reg = 5e-4
opt_mamba.weight_decay = opt_exp.weight_decay

opt_mamba.input_nc = 4               # 4 APs
opt_mamba.output_nc = 1              # 1-channel location heatmap
opt_mamba.save_latest_freq = 5000
opt_mamba.save_epoch_freq = 1
opt_mamba.n_epochs = opt_exp.n_epochs
opt_mamba.isTrain = True
opt_mamba.continue_train = False
opt_mamba.starting_epoch_count = opt_exp.starting_epoch_count
opt_mamba.name = 'mamba_dloc'        # checkpoint subdirectory name
opt_mamba.loss_type = "L2_sumL1"     # MSE + L1 regularization (same as DLoc decoder)
opt_mamba.niter = 20
opt_mamba.niter_decay = 100

opt_mamba.gpu_ids = opt_exp.gpu_ids
opt_mamba.num_threads = 4
opt_mamba.checkpoints_load_dir = opt_exp.load_dir
opt_mamba.checkpoints_save_dir = opt_exp.checkpoints_dir
opt_mamba.results_dir = opt_exp.results_dir
opt_mamba.log_dir = opt_exp.log_dir
opt_mamba.max_dataset_size = float("inf")
opt_mamba.verbose = False
opt_mamba.suffix = ''

# ---------- Aliases for train_and_test.py compatibility ----------
# The Mamba pipeline uses a single ModelADT, but train_and_test.py
# still imports opt_encoder/opt_decoder. These are not used when
# model_type == "mamba", but must exist to avoid ImportError.
opt_encoder = opt_mamba
opt_decoder = opt_mamba
opt_offset_decoder = opt_mamba
```

- [ ] **Step 2: Commit**

```bash
git add params_storage/params_mamba_single_gpu.py
git commit -m "feat: add Mamba-DLoc params file for single GPU

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Integrate Mamba pipeline into train_and_test.py

**Files:**
- Modify: `DLoc_pt_code/train_and_test.py:16` (imports), `DLoc_pt_code/train_and_test.py:139-168` (model init), `DLoc_pt_code/train_and_test.py:186-204` (evaluation)

- [ ] **Step 1: Add Mamba_Network import**

At line 17 of `train_and_test.py`, after `from joint_model import Enc_Dec_Network`, add:

```python
from joint_model import Mamba_Network
```

The import block (lines 16-18) should look like:

```python
from data_loader import load_data, LazyMultiHDF5Dataset
from joint_model import Enc_2Dec_Network
from joint_model import Enc_Dec_Network
from joint_model import Mamba_Network
from params import *
```

- [ ] **Step 2: Add Mamba model initialization branch**

Replace the model initialization section (lines 139-168) with a version that handles the mamba case. The new code should be:

```python
'''
Initiate the Network and build the graph
'''

model_type = getattr(opt_exp, 'model_type', 'dloc')

if model_type == 'mamba':
    # Mamba: single end-to-end model wrapped in one ModelADT
    from params import opt_mamba
    mamba_model = ModelADT()
    mamba_model.initialize(opt_mamba)
    mamba_model.setup(opt_mamba)

    print('Making the Mamba joint model')
    joint_model = Mamba_Network()
    joint_model.initialize(opt_exp, mamba_model, gpu_ids=opt_exp.gpu_ids)

else:
    # Original DLoc pipeline
    # init encoder
    enc_model = ModelADT()
    enc_model.initialize(opt_encoder)
    enc_model.setup(opt_encoder)

    # init decoder1
    dec_model = ModelADT()
    dec_model.initialize(opt_decoder)
    dec_model.setup(opt_decoder)

    if opt_exp.n_decoders == 2:
        # init decoder2
        offset_dec_model = ModelADT()
        offset_dec_model.initialize(opt_offset_decoder)
        offset_dec_model.setup(opt_offset_decoder)

        # join all models
        print('Making the joint_model')
        joint_model = Enc_2Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

    elif opt_exp.n_decoders == 1:
        # join all models
        print('Making the joint_model')
        joint_model = Enc_Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

    else:
        print('Incorrect number of Decoders specified in the parameters')
        sys.exit(-1)

    if getattr(opt_exp, 'isFrozen', False):
        enc_model.load_networks(opt_encoder.starting_epoch_count)
        dec_model.load_networks(opt_decoder.starting_epoch_count)
        if opt_exp.n_decoders == 2:
            offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)
```

- [ ] **Step 3: Update evaluation section for Mamba**

Replace the evaluation section (lines 186-204) with:

```python
'''
Model Evaluation at the best epoch
'''

epoch = "best"  # int/"best"/"last"
eval_name = opt_exp.checkpoints_dir

if model_type == 'mamba':
    # Mamba: reload single model
    mamba_model.load_networks(epoch, load_dir=eval_name)
    joint_model.initialize(opt_exp, mamba_model, gpu_ids=opt_exp.gpu_ids)
else:
    # DLoc: reload encoder + decoder(s)
    enc_model.load_networks(epoch, load_dir=eval_name)
    dec_model.load_networks(epoch, load_dir=eval_name)
    if opt_exp.n_decoders == 2:
        offset_dec_model.load_networks(epoch, load_dir=eval_name)
        joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)
    elif opt_exp.n_decoders == 1:
        joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

# pass data through model
total_loss, median_error = trainer.test(joint_model,
    test_loader,
    save_output=True,
    save_dir=eval_name,
    save_name=f"decoder_test_result_epoch_{epoch}",
    log=False)
print(f"total_loss: {total_loss}, median_error: {median_error}")
```

- [ ] **Step 4: Commit**

```bash
git add train_and_test.py
git commit -m "feat: integrate Mamba pipeline into train_and_test.py

Adds model_type='mamba' branch that uses single ModelADT + Mamba_Network.
Falls back to original DLoc encoder/decoder pipeline when model_type is
not set or is 'dloc'.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Local smoke test

**Files:** No changes — verification only.

- [ ] **Step 1: Copy Mamba params to params.py**

```bash
cd DLoc_pt_code
cp params_storage/params_mamba_single_gpu.py params.py
```

- [ ] **Step 2: Run a quick forward-pass test without data**

```bash
cd DLoc_pt_code
python -c "
import torch
from mamba_model import MambaDLocNet, HAS_MAMBA_SSM
from modelADT import ModelADT
from joint_model import Mamba_Network
from params import *

print(f'mamba-ssm available: {HAS_MAMBA_SSM}')
print(f'model_type: {opt_exp.model_type}')
print(f'n_decoders: {opt_exp.n_decoders}')
print(f'base_model: {opt_mamba.base_model}')

# Create model through the full pipeline
model = ModelADT()
model.initialize(opt_mamba)
model.setup(opt_mamba)

joint = Mamba_Network()
joint.initialize(opt_exp, model, gpu_ids=opt_exp.gpu_ids)

# Simulate a training step
x = torch.randn(2, 4, 161, 361)
label = torch.randn(2, 1, 161, 361)
joint.set_input(x, label)
joint.optimize_parameters()

print(f'Output shape: {joint.decoder.output.shape}')
print(f'Loss: {joint.decoder.loss.item():.6f}')
print('OK: Full Mamba pipeline smoke test passed')
"
```

Expected: prints shapes, a loss value, and "OK" message.

- [ ] **Step 3: Verify original DLoc pipeline still works**

Restore the original params and verify nothing is broken:

```bash
cd DLoc_pt_code
cp params_storage/params_fig10b_single_gpu.py params.py
python -c "
from params import *
print(f'model_type: {getattr(opt_exp, \"model_type\", \"dloc\")}')
print(f'n_decoders: {opt_exp.n_decoders}')
print('OK: DLoc params load correctly, model_type defaults to dloc')
"
```

- [ ] **Step 4: Copy Mamba params back for future use**

```bash
cd DLoc_pt_code
cp params_storage/params_mamba_single_gpu.py params.py
```

---

### Task 10: Update project memory

**Files:**
- Modify: `~/.claude/projects/C--Users-JustAGeek-DLOC/memory/project_dloc.md`

- [ ] **Step 1: Update memory with Mamba model info and bug fix status**

Add to the project memory file:
- Mamba-DLoc architecture exists in `mamba_model.py`
- Data split bug status (fixed or pending regeneration)
- RNG state persistence added to modelADT.py
- How to switch between DLoc and Mamba: copy appropriate params file to `params.py`
- `model_type = "mamba"` in params.py selects the Mamba pipeline

- [ ] **Step 2: Commit all remaining changes**

```bash
git add -A
git commit -m "docs: update project memory with Mamba model and bug fix status

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
