# Mamba-DLoc Architecture + Bug Fixes — Design Spec

**Date:** 2026-05-20
**Goal:** Fix all Report_Grad.pdf bugs, then implement the MAMBA-based DLoc variant from Next Step.pdf integrated into the existing training pipeline.

## Part 1: Bug Fixes

### Fix 1 — Data Split (CRITICAL)

**Problem:** `dataset_jacobs_July28.mat` contains all 11,440 samples from time-instance-1 instead of the correct 7,635 training-only samples (fov_train + non_fov_train). This inflates the training set from 17,574 (80.5%) to 21,379 (83.4%), deviating from the paper's intended split.

**Fix:** Fix `fix_split.py` path bug (`SPLIT_FILE` points to `./data/` but split indices are in `../data_split_idx/`), then regenerate the file. Alternatively, re-run `prepare_data.py` which already has the correct path.

**Files:** `fix_split.py` — change `SPLIT_FILE` path from `os.path.join(DATA_DIR, 'data_split_ids_jacobs_July28.mat')` to `os.path.join('../data_split_idx', 'data_split_ids_jacobs_July28.mat')`.

**Verification:** After fix, `dataset_jacobs_July28.mat` must have exactly 7,635 samples. Total training = 17,574, test = 4,260, ratio = 80.5%.

### Fix 2 — Random Seed State Persistence

**Problem:** `train_and_test.py` sets `torch.manual_seed(0)` and `np.random.seed(0)` at startup. On multi-session training, the seed resets to 0, causing identical shuffle orders for the first epoch of each session.

**Fix:** Save RNG state alongside model checkpoints in `modelADT.py:save_networks()`. Restore in `modelADT.py:load_training_state()`.

**Details:**
- `save_networks(epoch)`: additionally save `torch.random.get_rng_state()`, `np.random.get_state()`, and `torch.cuda.get_rng_state_all()` into the optim checkpoint dict.
- `load_training_state(epoch)`: restore all three RNG states if present in the checkpoint.

### Fix 3 — fix_split.py Usability

**Problem:** `fix_split.py` also reads from `./data/data_split_ids_jacobs_July28.mat` which doesn't exist (split files are in `../data_split_idx/`).

**Fix:** Correct the path. Same file as Fix 1.

## Part 2: Mamba-DLoc Architecture

### Overview

Replace the 6 ResNet bottleneck blocks in DLoc's encoder with 4 Mamba (Selective State Space) blocks. The CNN stem handles local spatial feature extraction, Mamba handles global signal relationships, and a CNN decoder reconstructs the location heatmap.

- Single decoder (location only — no consistency decoder)
- Integrated into existing `train_and_test.py` via `opt_exp.model_type = "mamba"`
- Uses `mamba-ssm` library on Linux (vast.ai), falls back to pure PyTorch on Windows

### Architecture

```
Input: [B, 4, 161, 361]
    |
    v
CNN Stem Encoder:
    Conv2d(4->32, 3x3, pad=1) + BN + ReLU       -> [B, 32, 161, 361]
    Conv2d(32->64, 3x3, stride=2, pad=1) + BN + ReLU  -> [B, 64, 81, 181]
    Conv2d(64->128, 3x3, stride=2, pad=1) + BN + ReLU -> [B, 128, 41, 91]
    |
    v
Tokenize: reshape [B, 128, 41, 91] -> [B, 3731, 128]
    |
    v
4x MambaBlock:
    Each block: LayerNorm -> Mamba(d_model=128) -> residual
    Output: [B, 3731, 128]
    |
    v
Reshape back: [B, 3731, 128] -> [B, 128, 41, 91]
    |
    v
CNN Decoder:
    F.interpolate(size=(81,181)) + Conv2d(128->64, 3x3, pad=1) + BN + ReLU
    F.interpolate(size=(161,361)) + Conv2d(64->32, 3x3, pad=1) + BN + ReLU
    Conv2d(32->1, 1x1) + Sigmoid
    |
    v
Output: [B, 1, 161, 361]
```

### Mamba Block (Pure PyTorch Fallback)

When `mamba-ssm` is not available, a pure PyTorch implementation:

```
Input x: [B, L, D]
    |
    v
LayerNorm(D)
    |
    +--> Linear(D, E) -> SiLU -> Conv1d(E, E, kernel=4, groups=E) -> SiLU
    |       |
    |       v
    |   SSM Projection: Linear(E, dt_rank + 2*N)
    |       |
    |       v
    |   Selective Scan: discretize A,B with dt; sequential scan y = Cx
    |       |
    |       v
    |   Linear(E, D) -- output projection
    |
    +--> Linear(D, E) -> SiLU  (gate/skip branch)
    |
    v
Multiply gate * ssm_output
    |
    v
Linear(E, D) -> residual add with input
```

Where: D=128 (model dim), E=256 (expand=2), N=16 (state dim), dt_rank=16.

### Integration into Existing Pipeline

#### New file: `mamba_model.py`

Contains:
- `MambaDLocNet(nn.Module)` — full end-to-end model
- `MambaBlock(nn.Module)` — single Mamba block with residual
- `PurePytorchMamba(nn.Module)` — fallback selective scan implementation

#### Modified: `utils.py` (define_G factory)

Add `elif net_type == 'mamba_dloc':` branch that instantiates `MambaDLocNet`.

#### Modified: `joint_model.py`

Add `Mamba_Network` class:
- Wraps a single `ModelADT` as `self.decoder`
- Creates `_EncoderProxy` for `self.encoder` that shares `opt` but no-ops on `update_learning_rate()` (prevents double LR scheduler stepping since trainer.py calls update on both encoder and decoder)
- Implements same interface as `Enc_Dec_Network`: `set_input`, `forward`, `test`, `backward`, `optimize_parameters`, `save_networks`, `eval`, `update_learning_rate`

#### Modified: `train_and_test.py`

Add branch for `model_type == "mamba"`:
- Create single `ModelADT` with mamba params
- Create `Mamba_Network` joint model wrapper
- For evaluation/loading: only one model to load (not separate encoder + decoder)
- Data loading and trainer.py call remain unchanged

#### New file: `params_storage/params_mamba_single_gpu.py`

Key settings:
- `opt_exp.model_type = "mamba"` (new field)
- `opt_exp.n_decoders = 1`
- `opt_exp.data = "rw_to_rw"` (same dataset as Fig 10b)
- `opt_exp.batch_size = 8`
- Single model params: `base_model = 'mamba_dloc'`, `loss_type = "L2_sumL1"`, `lambda_reg = 5e-4`
- `lr = 1e-5`, `lr_decay_iters = 20`, `lr_policy = 'step'`

### Loss Function

Same as DLoc location decoder:
- MSE between predicted and ground-truth heatmap
- L1 regularization on output (lambda_reg = 5e-4)
- `loss_type = "L2_sumL1"`

### Parameter Count Estimate

| Component | Params (approx) |
|-----------|----------------|
| CNN Stem (3 conv+BN layers) | ~50K |
| 4 Mamba blocks (D=128, E=256, N=16) | ~1.1M |
| CNN Decoder (2 conv+BN + head) | ~45K |
| **Total** | **~1.2M** |

vs DLoc baseline: ~15M (12.5x reduction)

### Testing Strategy

1. **Local (Windows):** Pure PyTorch fallback, run 2-3 training steps to verify forward/backward pass, loss decreases, shapes are correct.
2. **vast.ai (Linux):** Full training with `mamba-ssm` CUDA kernels, 50 epochs, same data split.
3. **Validation:** Compare median/P90/P99 localization error against corrected DLoc baseline.

## Execution Plan Summary

1. Apply all 3 bug fixes
2. Verify data split is correct (7,635 + 9,939 train, 4,260 test)
3. Create `mamba_model.py` with MambaDLocNet + fallback
4. Integrate into pipeline (utils.py, joint_model.py, train_and_test.py)
5. Create mamba params file
6. Local smoke test (few batches on Windows)
7. User retrains baseline DLoc with correct data on vast.ai
8. User trains Mamba model on vast.ai
