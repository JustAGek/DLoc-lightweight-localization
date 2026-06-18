# DLoc replication handoff (Vast.ai + Hugging Face)

## Quick context
- Goal: replicate DLoc Fig 10b (Jacobs Hall) with corrected split, continuous run, and report best epoch.
- We created an isolated code package in `dloc_run/` and uploaded it to Hugging Face as `JustAGeek/dloc-code`.
- Dataset uploaded to Hugging Face as `JustAGeek/dloc-wild-fig10b`.

## Correct dataset composition (fixed split)
- Train: 17,574 samples (7,635 from July28 + 9,351 + 588 from July28_2)
- Test: 4,260 samples (4,008 + 252)
- Confirms intended 80/20 split. The wrong 11,440-sample file was corrected.

## Hugging Face
- Code repo (model): `JustAGeek/dloc-code`
- Dataset repo: `JustAGeek/dloc-wild-fig10b`
- Use `hf` CLI, not git, for uploads/downloads.
- `hf upload` and `hf download` are used; some `hf` versions do not support `--local-dir-use-symlinks` or `--path-in-repo`.

## Vast.ai notes
- We used H100 instances. Disk size matters (need 60+ GB). A volume is attached in some runs.
- Download data directly to `/workspace/dloc-code/data` to avoid symlink issues.
- tmux is used on the remote to keep sessions alive.

## Code changes made in `dloc_run/`
1) DataLoader speedups (in `train_and_test.py`):
   - `num_workers=4`, `pin_memory=True`, `persistent_workers=True`

2) RAM loading (instead of lazy HDF5):
   - `train_and_test.py` loads all .mat files into RAM via `load_data()` and builds a `TensorDataset`.
   - This makes first epoch load longer, but speeds training significantly.

3) Label shape fix for RAM loading:
   - `labels_gaussian_2d` loaded as `[N, H, W]` now unsqueezed to `[N, 1, H, W]` before concatenation.

4) Batch size updated in params:
   - `params_storage/params_fig10b_single_gpu.py`: `opt_exp.batch_size = 64`
   - `params_storage/params_mamba_single_gpu.py`: `opt_exp.batch_size = 64` (later reduced for Mamba due to OOM)

## Baseline run status
- With RAM loading + batch 64 on H100, epoch time ~3 minutes.
- Best baseline: `median_error = 0.7071` at `runs/2026-05-21-06-51-03`.

## Mamba run issues
1) CUDA kernel not installed
   - `HAS_MAMBA_SSM = False`, so Mamba uses slow pure-PyTorch SSM.
   - Installing `mamba-ssm` failed under build isolation because `torch` not found.
   - Use `pip install --no-build-isolation mamba-ssm` (may still be slow to build).

2) OOM with large batch
   - Mamba OOM at batch 64. Suggested to use 32 or 16.

3) Best checkpoint not saved
   - Training logic saves `best_*` only if median improves twice in a row (`stopping_count==2`).
   - For Mamba run (e.g. `runs/2026-05-21-09-39-03`), no `best_net_mamba_dloc.pth` exists.
   - Fix: determine best epoch from `mamba_dloc_test_median_error.txt` and manually copy that epoch to `best_*`.

## Common commands (remote)
- Download code:
  - `hf download JustAGeek/dloc-code --repo-type model --local-dir /workspace/dloc-code`
- Download dataset directly into correct path:
  - `hf download JustAGeek/dloc-wild-fig10b --repo-type dataset --local-dir /workspace/dloc-code/data --include "data/*.mat"`
  - Then move files out of nested `data/` folder if created:
    - `mv /workspace/dloc-code/data/data/* /workspace/dloc-code/data/ && rmdir /workspace/dloc-code/data/data`
- Run baseline:
  - `cp params_storage/params_fig10b_single_gpu.py params.py`
  - `python train_and_test.py`
- Run Mamba:
  - `cp params_storage/params_mamba_single_gpu.py params.py`
  - `python train_and_test.py`

## Notes on paper vs run
- Paper uses 50 epochs, batch 32. Current run is 120 epochs in some logs.
- Full uninterrupted run removes optimizer/scheduler/seed reset issues.
- Remaining differences: hardware, software version, batch size, reporting methodology.

## Known paths
- Repo: `/workspace/dloc-code`
- Runs: `/workspace/dloc-code/runs/<timestamp>/`
- Best results saved in `decoder_test_result_epoch_best.mat` for baseline.
