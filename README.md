# DLoc — Lightweight Deep Learning for WiFi Indoor Localization

A PyTorch replication and **efficiency-focused extension** of **DLoc**
([Deep Learning based Wireless Localization for Indoor Navigation](https://dl.acm.org/doi/10.1145/3372224.3380894),
MobiCom 2020, UC San Diego WCSNG). We compress DLoc's 16.5M-parameter encoder–decoder into
lightweight student models (MobileNetV2, TinyCNN, **MobileNetV2-UNet**) via architectural
redesign and knowledge distillation, and evaluate accuracy, robustness, and cross-session
generalization — including a **fair, protocol-matched comparison** under DLoc's own
leave-one-session-out scheme.

> Graduation project, Egypt–Japan University of Science and Technology (E-JUST).

---

## TL;DR results

| Model | Params | In-dist median | Cross-session (official, mean) | Notes |
|-------|-------:|:--------------:|:------------------------------:|-------|
| DLoc baseline (published) | 16.5M | 0.64 m | 0.86 m | the network we compress |
| DLoc baseline (our repro) | 16.5M | 0.707 m | — | implementation gap vs paper |
| MobileNetV2 | 2.3M | 0.707 m | — | matches repro at **7.2× smaller** |
| TinyCNN | 47K | 3.9 m | — | below the capacity floor |
| **MobileNetV2-UNet** | **2.41M** | **0.640 m** | **0.930 m** | **headline model, 6.8× smaller** |

**Honest summary:** the MobileNetV2-UNet is **competitive with DLoc at 6.8× fewer parameters** —
on par in-distribution, within ~0.1 m of DLoc on cross-session median (slightly behind on two of
three setups), and on par on P90. This is an **efficiency** result, not an accuracy win. The most
interesting findings are about **robustness** (see below).

### Robustness findings (the scientifically novel part)
- **Knowledge distillation = spatial label smoothing** → 3.4× Gaussian-noise robustness for MobileNetV2.
- **U-Net skip connections = architectural robustness** → 5.5× (routes shallow features past the noisy bottleneck).
- The two mechanisms are **complementary but not additive**.
- **Distillation has a capacity threshold**: it *helps* above ~2M params and *harms* below it (TinyCNN).

---

## Repository structure

```
.
├── README.md                       # this file
├── docs/                           # ★ defense documentation (LaTeX, compile yourself)
│   ├── 01_dloc_architecture.tex    #   the original DLoc network, in detail
│   ├── 02_dataset.tex              #   the WILD dataset: collection, sessions, splits
│   ├── 03_model_architectures.tex  #   every model, layer-by-layer
│   ├── 04_experiments.tex          #   all experiments + analysis + defense Q&A
│   └── README.md                   #   how to compile the docs
├── code/                           # ★ all training/eval code, organized by purpose
│   ├── core/        # DLoc engine: Generators, modelADT, joint_model, trainer, data_loader, utils
│   ├── models/      # student_mobilenet, student_tinycnn, student_unet, mamba_model
│   ├── configs/     # params + params_storage/ (experiment configs)
│   ├── training/    # train_*.py, run_crosssession_table1.sh, precompute_teacher.py
│   ├── evaluation/  # test_robustness, eval_cross_session, aggregate_table1, benchmark_models
│   ├── data_prep/          # prepare_cross_session_data.py
│   ├── feature_generation/ # MATLAB: raw CSI → AoA/ToF heatmaps (stage 0 of the pipeline)
│   └── README.md           # code layout + how to run
├── reports/  (report_*.tex)        # paper-style write-ups (full project + cross-session)
├── run.sh                          # convenience launcher (pilot / crosssession / docs / …)
├── DLoc_pt_code/                   # original DLoc code (reference clone)
└── dloc.pdf                        # the DLoc paper (reference)
```

Large artifacts (datasets, weights, environments) are **not** in git — see *Data & weights* below.

---

## Defense documentation (`docs/`)

Four standalone, compilable LaTeX documents written as a study/reference set for the thesis defense:

1. **`01_dloc_architecture.tex`** — the original DLoc: input representation (CSI→heatmaps),
   ResNet encoder, location decoder, the consistency/offset decoder, loss functions, training.
2. **`02_dataset.tex`** — the WILD dataset: MapFind collection, the two environments, the five
   Jacobs sessions, feature representation, and how the splits are used.
3. **`03_model_architectures.tex`** — every model layer-by-layer (baseline, MobileNetV2, TinyCNN,
   MobileNetV2-UNet, Mamba) + the knowledge-distillation setup.
4. **`04_experiments.tex`** — all six experiments (methodology, results, analysis) plus a
   **defense Q&A** section with anticipated questions and objective answers.

Compile any of them with `pdflatex <file>.tex` (twice, for the table of contents). Requires
`pgfplots` (standard in TeX Live / MiKTeX / Overleaf). See `docs/README.md`.

---

## Data & weights (HuggingFace)

| What | Where |
|------|-------|
| Processed feature `.mat` files (WILD Jacobs) | `JustAGeek/dloc-wild-fig10b` (dataset) |
| Code, trained weights, logs, results | `JustAGeek/dloc-code` (model) |
| Official cross-session results + weights + logs | `JustAGeek/dloc-code/crosssession_official/` |

```python
from huggingface_hub import snapshot_download
snapshot_download("JustAGeek/dloc-code", repo_type="model", local_dir="./dloc-code")
```

---

## Reproduce the official cross-session experiment

```bash
cd code
# 1) download the 6 feature files from HF JustAGeek/dloc-wild-fig10b into code/data/
# 2) create the Aug16 'train_' symlinks (see code/CROSSSESSION_RUNBOOK.md)
# 3) single fold/seed:
python training/train_unet_crosssession.py --env rw_to_rw_env2 --seed 42 --epochs 120
# 4) full sweep (3 folds × 3 seeds) + aggregation:
bash training/run_crosssession_table1.sh   # → results_crosssession/ALLSETUPS_SUMMARY.md
```

Dependencies: `torch`, `torchvision`, `h5py`, `easydict`, `hdf5storage`, `scipy`, `numpy`,
`huggingface_hub`.

---

## Limitations
- Comparisons use the *published* DLoc numbers; our DLoc reproduction (0.707 m) does not reach the
  paper's 0.64 m (an implementation gap).
- The median metric is pixel-quantized (`0.1·√n` m); sub-0.1 m differences are at the resolution limit.
- All data are from a single building (Jacobs Hall); cross-*building* generalization (Atkinson,
  3-AP) is out of scope and the data are unavailable.
- The Mamba variant diverged (1D flattening destroys spatial locality) and is kept for completeness.

## Citation
If referencing the original system, cite Ayyalasomayajula et al., *DLoc*, ACM MobiCom 2020.
