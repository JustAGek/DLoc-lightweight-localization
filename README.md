# DLoc — Lightweight Deep Learning for WiFi Indoor Localization

A PyTorch replication and efficiency-focused extension of DLoc (Deep Learning based Wireless
Localization for Indoor Navigation, MobiCom 2020, UC San Diego WCSNG). We reduce the parameter count
to study lightweight alternatives to DLoc and evaluate their performance. We find that MobileNetV2
matches DLoc in-distribution at 7.2× fewer parameters, while TinyCNN lacks the capacity to learn the
DLoc mapping; MobileNetV2-UNet, our proposed new model, achieves performance competitive with DLoc but
with 6.8× fewer parameters. An added benefit of the model is its robustness to various noise types and
intensities.

> Graduation project, Egypt–Japan University of Science and Technology (E-JUST).

## TL;DR

Results of the lightweight models (MobileNetV2, TinyCNN, MobileNetV2-UNet) and the DLoc baseline:

| Model | Params | In-dist median | Cross-session (official, mean) | Notes |
|---|---|---|---|---|
| DLoc baseline (published) | 16.5M | 0.64 m | 0.86 m | the network we compress |
| DLoc baseline (our repro) | 16.5M | 0.707 m | — | implementation gap vs paper |
| MobileNetV2 | 2.3M | 0.707 m | — | matches repro at 7.2× smaller |
| TinyCNN | 47K | 3.9 m | — | below the capacity floor |
| **MobileNetV2-UNet** | **2.41M** | **0.640 m** | **0.930 m** | **headline model, 6.8× smaller** |

### Honest summary

Our MobileNetV2-UNet architecture outperforms the DLoc algorithm in terms of model parameter count
while maintaining competitive performance. However, MobileNetV2 and TinyCNN struggle in the
cross-session evaluation metrics. The best of the lightweight models is the MobileNetV2-UNet, which
features robustness and achieves competitive performance at 6.8× fewer parameters than DLoc. The most
interesting findings use these metrics to study robustness (see below).

### Robustness findings (the scientifically novel part)

Knowledge distillation increases the model's robustness to spatial (Gaussian) noise by 3.4×. The use
of skip connections in the U-Net model increases the model's robustness by 5.5× (by allowing shallow
features to pass through the potentially-noisy bottleneck). These two methods exhibit complementary,
but not additive, robustness gains. Interestingly, knowledge distillation is only beneficial above a
certain capacity threshold for the student model; distillation harms the model's robustness if the
model is too small (the TinyCNN case).

## Repository structure

```
.
├── README.md
├── REPRODUCIBILITY.md              # how every thesis result was produced (start here)
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
│   ├── feature_generation/ # MATLAB: raw CSI → AoA/ToF heatmaps
│   └── README.md           # code layout + how to run
├── reports/  (report_*.tex)        # paper-style write-ups (full project + cross-session)
├── run.sh                          # convenience launcher (pilot / crosssession / docs / …)
├── DLoc_pt_code/                   # original DLoc code (reference clone)
└── dloc.pdf                        # the DLoc paper (reference)
```

Large artifacts (datasets, weights, environments) are not in git — see *Data & weights* below.

## Defense documentation (`docs/`)

Four standalone, compilable LaTeX documents written as a study/reference set for the thesis defense:

- **`01_dloc_architecture.tex`** describes the original DLoc network (input, encoder, decoder, loss).
- **`02_dataset.tex`** describes the WILD dataset: how it was collected, the two environments, the five
  Jacobs sessions, how the data are represented, and how the training/testing splits are defined.
- **`03_model_architectures.tex`** lists each model in this study (baseline, MobileNetV2, TinyCNN,
  MobileNetV2-UNet, Mamba) and describes their components and structure.
- **`04_experiments.tex`** contains descriptions of all six experiments (methods, results) plus a
  defense Q&A with anticipated questions and answers.

Compile with `pdflatex <file>.tex` twice. Requires `pgfplots`. See `docs/README.md`.

## Data & weights (HuggingFace)

The feature files are available on HuggingFace at **`JustAGeek/dloc-wild-fig10b`** (the `.mat` files for
the WILD Jacobs data). The code, trained models, and training logs are available at
**`JustAGeek/dloc-code`**. The results and weights for the cross-session experiment are under
`JustAGeek/dloc-code/crosssession_official/`.

Download the model code with:

```python
from huggingface_hub import snapshot_download
snapshot_download("JustAGeek/dloc-code", repo_type="model", local_dir="./dloc-code")
```

## Reproduce the official cross-session experiment

> **Full result→code map:** see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — every thesis table and
> figure mapped to the exact script, config, command, and where its output lives (repo vs. HuggingFace),
> plus the Vast.ai environment and dependencies.

```bash
cd code
# download the feature files
# create the symlinks for the train_ environments (see CROSSSESSION_RUNBOOK.md)

# run the experiment for a single fold and seed
python training/train_unet_crosssession.py --env rw_to_rw_env2 --seed 42 --epochs 120

# run a full sweep over all folds (3) and seeds (3); results go to results_crosssession/
bash training/run_crosssession_table1.sh
```

Dependencies: `torch`, `torchvision`, `h5py`, `easydict`, `hdf5storage`, `scipy`, `numpy`,
`huggingface_hub`.

## Limitations

- We used the published results of DLoc to compare our models. When we re-implemented DLoc, we obtained
  a slightly lower in-distribution performance of 0.707 m compared to the published 0.64 m.
- Both in-distribution and cross-session evaluations used the median distance between the predicted and
  true locations. Errors are quantized to multiples of 0.1·√n meters, so two results within about 0.1 m
  of each other are effectively equal.
- All data was collected in a single building (Jacobs), so cross-building localization is outside the
  scope of this project and the data is unavailable.
- The Mamba model was omitted from the results due to its poor performance (flattening the 2D heatmaps
  into a 1D sequence destroys spatial locality). It is included to provide a complete survey of the
  models and results.

## Citation

If you use this project in your research, please cite the original DLoc paper:

> Ayyalasomayajula, Roshan, et al. "Deep Learning based Wireless Localization for Indoor Navigation."
> *Proceedings of the 26th Annual International Conference on Mobile Computing and Networking
> (MobiCom)*. ACM, 2020.
