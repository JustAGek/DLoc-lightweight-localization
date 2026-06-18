# Cross-Session (Table 1) Run — Runbook

Official DLoc leave-one-session-out protocol, run on **our models** (MobileNetV2-UNet
standalone) + a **reproduced DLoc baseline** under identical splits, for a fair,
publication-grade comparison against paper Table 1.

**Matrix:** 5 setups × {U-Net, baseline} × N seeds (= 30 runs at 3 seeds).
All runnable paper setups with the data we have (4-AP Jacobs Hall):
  - `rw_to_rw`       → Fig 10b in-distribution (Jul28) — paper DLoc 64 cm
  - `rw_to_rw_env2`  → Table 1: test **Aug16_1** (furniture) — paper DLoc 71 cm
  - `rw_to_rw_env3`  → Table 1: test **Aug16_3** (diff furniture) — paper DLoc 82 cm
  - `rw_to_rw_env4`  → Table 1: test **Aug16_4_ref** (reflector) — paper DLoc 105 cm
  - `data_segment`   → Fig 13b: across space (disjoint spatial train/test regions)

NOT runnable without new data (separate track): Fig 10a Atkinson (no July18 data +
needs 3-AP models), Fig 13a bandwidth (no 20/40 MHz feature files).

## 1. Provision H100 + get code/data

```bash
# On the fresh Vast.ai instance:
PY=/opt/miniforge3/bin/python   # adjust if different
huggingface-cli download JustAGeek/dloc-code      --local-dir ./dloc-code            --repo-type model
huggingface-cli download JustAGeek/dloc-wild-fig10b --local-dir ./dloc-code/data     --repo-type dataset
cd dloc-code
```

⚠️ The `data_segment` (Fig 13b) fold needs the disjoint-region splits
`dataset_{train,test}_jacobs_July28.mat` and `..._July28_2.mat`. These exist locally in
`DLoc_pt_code/data/` — confirm they are also on the HF dataset repo (or upload them) so they
download to `data/` on the instance; otherwise the Fig 13b runs will fail with a missing file.

Copy in the new scripts if they are not yet on HF:
`train_unet_crosssession.py`, `aggregate_table1.py`, `run_crosssession_table1.sh`,
`params_storage/params_tab1_crosssession_single_gpu.py`, and the seed patch in `train_and_test.py`.

## 2. Create the Aug16 symlinks  (the env tags expect a `train_` prefix)

The dataset repo ships the Aug16 files WITHOUT the `train_` prefix, but
`train_and_test.py` / the U-Net fold lists reference `dataset_train_jacobs_Aug16_*`:

```bash
cd data
ln -sf dataset_jacobs_Aug16_1.mat     dataset_train_jacobs_Aug16_1.mat
ln -sf dataset_jacobs_Aug16_3.mat     dataset_train_jacobs_Aug16_3.mat
ln -sf dataset_jacobs_Aug16_4_ref.mat dataset_train_jacobs_Aug16_4_ref.mat
cd ..
```

## 3. ⚠️ Verify variable names BEFORE the baseline run

The **baseline** uses the consistency/offset decoder, so every training file must
contain `features_wo_offset` (the U-Net only needs `features_w_offset` + `labels_gaussian_2d`).
Confirm the Aug16 files carry all three:

```bash
$PY - <<'EOF'
import h5py
for s in ['1','3','4_ref']:
    f = h5py.File(f'data/dataset_jacobs_Aug16_{s}.mat','r')
    print(s, sorted(f.keys()))
EOF
```
Expect each to include `features_w_offset`, `features_wo_offset`, `labels_gaussian_2d`.
If `features_wo_offset` is missing, the baseline (n_decoders=2) cannot train on that
fold — regenerate the file with `prepare_cross_session_data.py`, or restrict the
baseline to folds that have it.

## 4. PILOT first (~$10, validates pipeline + calibrates epoch time)

Edit the top of `run_crosssession_table1.sh`:
```bash
SEEDS=(42)
ENVS=(rw_to_rw_env2)
```
Then:
```bash
tmux new -s table1 'bash run_crosssession_table1.sh 2>&1 | tee /tmp/table1_pilot.log'
```
Check: both runs finish, `results_crosssession/ALLSETUPS_SUMMARY.md` is produced, and the
per-epoch wall-clock is sane. Multiply by 5 setups × N seeds × 2 models to confirm the real
cost fits budget before scaling.

## 5. FULL run (paper-grade)

Restore:
```bash
SEEDS=(42 123 777)
ENVS=(rw_to_rw rw_to_rw_env2 rw_to_rw_env3 rw_to_rw_env4 data_segment)
```
Relaunch (the orchestrator SKIPs completed runs, so it resumes safely):
```bash
tmux new -s table1 'bash run_crosssession_table1.sh 2>&1 | tee /tmp/table1.log'
```

Estimated ~65–75 H100-hours for the full 30-run matrix (baseline 50 ep dominates).
Trim by reducing seeds, or set `RUN_BASELINE=0` once the baseline folds are done.

## 6. Collect + persist results

```bash
$PY aggregate_table1.py          # regenerate summary any time
# Upload artifacts back to HF:
huggingface-cli upload JustAGeek/dloc-code results_crosssession/ results_crosssession/
huggingface-cli upload JustAGeek/dloc-code runs/ runs/ --include "baseline_*/**" "unet_*/**"
```

Outputs:
- `results_crosssession/ALLSETUPS_SUMMARY.md` — mean ± std per fold, U-Net vs baseline vs paper
- `results_crosssession/unet_<env>_seed<seed>.json` — per-run U-Net metrics
- `runs/baseline_<env>_seed<seed>/` — per-epoch baseline logs + checkpoints
- `runs/unet_<env>_seed<seed>_<ts>/` — per-epoch U-Net logs + best/latest weights

## Notes
- Seed is wired through `DLOC_SEED` (baseline, read in `train_and_test.py`) and
  `--seed` (U-Net). Run dirs are deterministic via `RUN_TAG` so re-runs/aggregation are stable.
- U-Net is trained **standalone** (no distillation): distillation gave no median gain in
  Phase 2, and the Phase-2 teacher was trained on the Jul28 split, which does not match these folds.
- "Fair comparison" = identical data splits + identical meters metric; each model keeps its
  own optimal recipe (baseline: 50 ep encoder+2 decoders; U-Net: 120 ep cosine Adam).
