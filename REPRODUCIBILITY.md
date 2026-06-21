# Reproducibility — How Every Thesis Result Was Produced

Every result in the thesis (every table and figure in Chapter 5 and the Appendix) is indexed in this
document with the source file and command that produced each result, where the data lives, and the
compute environment and dependencies. This document is the index the source code should be read
through.

> **Layouts in one line.** The organized, readable code that was used to produce the results of the
> research can be found in the `code/` folder of this repository. Each of the scripts in the `code/`
> folder includes a `sys.path` bootstrap to ensure they can find the code's modules (see
> `code/README.md`). The experiments were originally performed from the flat package archived on
> Hugging Face (`JustAGeek/dloc-code`).

---

## Data & artifacts — what lives where

The data and artifacts from this research are split between two locations based on their size. Small
data and artifacts are included in this repository. Large data and artifacts are published to the
Hugging Face Hub.

### In this source repository (git)

All of the source code for this research is located in the `code/` folder. Additionally, all of the
results of the experiments are also published in this repository: the JSON files with the results
from the cross-session experiments (`code/results_crosssession/*.json`), the report that contains the
tables with the summary results of each experiment (`code/results_crosssession/ALLSETUPS_SUMMARY.md`),
and the TeX and Markdown report files that contain the tables of each experiment's results
(`reports/*.tex`, `*.md`). All of the data and figures from the experiments (each array of errors from
each sample in the dataset is published as `.npy` or `.npz` files in the `Figures/figure_data/` folder,
as well as each of the figures in the `Figures/generated/` and `thesis/figures/` folders) is also
located in this repository.

### On Hugging Face — `JustAGeek/dloc-wild-fig10b`

The processed WILD data that was used as the inputs to the DLoc deep learning models is published on
Hugging Face in this dataset repository (`JustAGeek/dloc-wild-fig10b`) as gzipped HDF5 files with the
extension `.mat`. The data includes the following:

- July's in-distribution train and test sessions (17,574 training samples, 4,260 test samples).
- August's cross-session data that was processed into three separate sessions
  (`dataset_jacobs_Aug16_1.mat` contains 15.46 GB of data from 11,201 samples;
  `dataset_jacobs_Aug16_3.mat` contains 12.08 GB of data from 8,754 samples;
  `dataset_jacobs_Aug16_4_ref.mat` contains 7.52 GB of data from 5,486 samples).

The `data_split_idx/` folder on the dataset repository contains files that contain the indices that
were used to split the dataset into its train and test sessions during each experiment.

### On Hugging Face — `JustAGeek/dloc-code`

In addition to the data used to train the models, the trained models and the files that contain their
run outputs are published on Hugging Face in this model repository (`JustAGeek/dloc-code`). The trained
models are published in the `weights/` folder (including 10 sets of checkpoints for the trained DLoc
baseline, MobileNetV2, TinyCNN, and U-Net models); the `logs/` folder contains per-epoch files that
record the models' training and test loss values (there are 131 such files for the 7 model training
runs); the `results/` folder contains files with the results of the experiments (including files with
the tables of the robustness of the models, the cross-session test results of the models, the training
logs for the U-Net model, and the `PHASE2_SUMMARY.md` file that includes the summary tables of all
experiments); and `multiseed/baseline/` contains the 3-seed DLoc baseline model trained with each of
the three seeds (42, 123, and 777) and the report of those seeds (`report.txt`). Additionally, the
`scripts/` folder and root of the model repository contain the flat, runnable copy of the source code
that was used to train the models.

### Not stored anywhere

The raw CSI data files used in the MATLAB-based feature-generation stage of the research are not
published anywhere; they are processed into the `.mat` feature files by the MATLAB code in
`code/feature_generation/`. The teacher model's precomputed outputs (used only in the distillation
training runs) are likewise not published — they are roughly 5 GB
(`data/teacher_{train,test}_outputs.pt`) and are regenerated on the training machine with
`precompute_teacher.py`.

### Where each kind of data lives (quick reference)

| Data / artifact | Repo (git) | HF `dloc-wild-fig10b` | HF `dloc-code` |
|---|---|---|---|
| Source code | ✅ | — | ✅ (flat copy) |
| Processed dataset (`.mat`) | — | ✅ | — |
| Trained weights | — | — | ✅ `weights/` |
| Per-epoch logs | — | — | ✅ `logs/` |
| Result tables / summaries | ✅ `reports/`, `results_crosssession/` | — | ✅ `results/` |
| Figure data + figures | ✅ `Figures/` | — | — |

### Which data each run needs

Each experiment and model that was trained and tested in this research used the data described below,
all published on the Hugging Face dataset repository (`JustAGeek/dloc-wild-fig10b`).

- **In-distribution accuracy (baseline, Mamba, MobileNetV2, TinyCNN, U-Net):** the data from the
  in-distribution July training and test sessions was used (17,574 training samples, 4,260 test
  samples). The baseline, Mamba, and U-Net models used only the in-distribution data for training. The
  MobileNetV2 and TinyCNN models additionally used the in-distribution test data during the
  distillation stage.
- **Robustness:** the same in-distribution July training and test data was used. Additionally, the
  test data from the in-distribution sessions was corrupted at test time to evaluate the robustness of
  each model (without retraining the models).
- **Cross-session LOSO (3 folds):** the in-distribution July dataset was used, as well as two of the
  three August cross-session datasets (for 3 trained models; each has three seeds for training).
- **Zero-shot cross-session:** the in-distribution July dataset was trained with the models, but
  tested with the August cross-session datasets.

### Notes on where the data is located

- The files containing the index data for each of the sessions on the Hugging Face dataset repository
  are located in the `dloc-wild-fig10b/data_split_idx/` folder: the `data_split_ids_jacobs_*.mat`
  files contain each session's train and test data indices.
- The cross-session data on the Hugging Face dataset repository does not ship with the `train_`
  prefixes that the cross-session fold lists expect. For example, `dataset_train_jacobs_Aug16_1.mat`
  is required but is published as `dataset_jacobs_Aug16_1.mat`. The train sessions with the proper
  prefixes can be created as symlinks to the original data using the procedure documented in
  `code/CROSSSESSION_RUNBOOK.md`.
- In addition to the dataset files, the training data is preprocessed differently per model. The
  baseline DLoc model requires the features from the WILD data that do **not** have the offset maps
  applied (`features_wo_offset`). The U-Net model requires the features that **do** have the offset
  maps applied (`features_w_offset`) and the labels for the 2D Gaussian distributions of the samples
  (`labels_gaussian_2d`).
- The in-distribution July 28 `fov` and `non_fov` split files are not published on Hugging Face. The
  original files are located in the [`DLoc_pt_code`](https://github.com/JustAGeek/DLoc_pt_code)
  repository in its `data/` folder, or can be regenerated using the `prepare_cross_session_data.py`
  script in the `data_prep/` folder of this repository.

---

## Environment & dependencies

### Compute used

The DLoc deep learning models were trained using rented access to an NVIDIA H100 GPU on the cloud
provider Vast.ai. The models were trained in Python 3.13 using PyTorch 2.12.0+cu130. The data was
staged on the Vast.ai instance at `/workspace/dloc-code/data/` (approximately 69 GB of data). The root
of the codebase was approximately 16 GB. Since Vast.ai instances are ephemeral, any NVIDIA H100 GPU
can be used to train the models. Because the cross-session models were trained on Vast.ai, the raw data
from the WILD dataset had to be preprocessed into HDF5 files, which took place on Kaggle using two T4
GPUs. Finally, the remaining deep learning and benchmarking models were trained and evaluated on a
local computer with an NVIDIA RTX 3060 6 GB GPU, an Intel i7-11800H 11th-generation processor, and
16 GB RAM under Windows 10, using the virtual environment (`.venv`) published with the code.

### Dependencies

The main Python dependencies are published in the `requirements.txt` file in the `code/` folder of
this research repository.

```bash
pip install -r code/requirements.txt
```

The main packages are `torch`, `torchvision`, `numpy`, `scipy`, `h5py`, `hdf5storage`, `easydict`,
`scikit-learn`, and `huggingface_hub`. Additionally, `matplotlib` is used to generate the figures.

The Mamba student models have an additional installation requirement, because they rely on CUDA-only
packages that will not work if installed with a regular `pip install` command:

```bash
pip install mamba-ssm --no-deps
pip install causal-conv1d
```

These packages are used in place of the standard convolutional layers in the training and testing of
the Mamba models. `mamba-ssm` must be installed with `--no-deps` because it will otherwise install a
version of torch that does not match the version installed on the training machine. Furthermore, the
batch size for the Mamba models must be set to 32 rather than the default of 64, since it will not fit
into GPU memory at a batch size of 64.

### One-time setup on a fresh GPU instance (e.g. Vast.ai H100)

```bash
huggingface-cli download JustAGeek/dloc-code        --local-dir ./dloc-code      --repo-type model
huggingface-cli download JustAGeek/dloc-wild-fig10b --local-dir ./dloc-code/data --repo-type dataset
cd dloc-code
pip install -r requirements.txt
cp params_storage/params_fig10b_single_gpu.py params.py
python precompute_teacher.py
```

The scripts in the `code/` folder are the same as those in the flat model repository (the archive of
the DLoc models on Hugging Face). The models in this experiment were trained using a config copied to
`params.py` from the `params_storage/` directory. The teacher model's outputs (which used the same
training data as the DLoc models) were precomputed with the `precompute_teacher.py` script.

---

## The pipeline at a glance

```
raw CSI ──► AoA/ToF heatmaps (.mat) ─► (teacher precompute) ─► train models ─► evaluate ─► figures
```

Each of the results published in this thesis was produced through the same general pipeline, at
different stages of the overall deep learning process.

---

## Result → code map (Chapter 5 + Appendix)

Each result published in the thesis is indexed below with the source-code file that produced it, the
config or change used to create it, and the location in which that result was published.

| Thesis result | Produced by | Config / how | Output published in |
|---|---|---|---|
| Table 5.1 — in-distribution accuracy (baseline) | `training/train_and_test.py` | `cp params_storage/params_fig10b_single_gpu.py params.py` | `logs/` and `weights/` on HF |
| Table 5.1 — in-distribution accuracy (Mamba student) | `training/train_and_test.py` | `cp params_storage/params_mamba_single_gpu.py params.py` | `logs/` and `weights/` on HF |
| Table 5.1 — in-distribution accuracy (MobileNetV2, TinyCNN) | `training/train_distillation.py` | change `STUDENT_TYPE` and `MODE` in the script | `logs/` and `weights/` on HF |
| Table 5.1, 5.2 — in-distribution accuracy (U-Net) | `training/train_unet_advanced.py` | change `SCHEDULER_TYPE` and `MODE` in the script | `logs/` and `weights/` on HF |
| Figure — in-distribution accuracy (CDF) | `figures/dump_per_sample_errors.py` → `figures/plot_cdfs.py` | — | `Figures/figure_data/` and `Figures/generated/` |
| Figure — qualitative heatmaps | `figures/plot_heatmaps.py` | — | `Figures/figure_data/` and `Figures/generated/` |
| Table 5.3, 5.4, 5.5, Appendix — robustness (each model) | `evaluation/test_robustness.py` | change `MODEL_TYPE` and `WEIGHTS_PATH` in the script | `results/` on HF |
| Table 5.6 — summary of Gaussian-noise robustness | aggregated from the robustness results | — | `reports/phase2_results.tex` |
| Cross-session median table + cross-session CDF figures | `training/train_unet_crosssession.py` → `evaluation/aggregate_table1.py` | `bash training/run_crosssession_table1.sh` (3 folds × 3 seeds, 120 epochs) | `code/results_crosssession/*.json` |
| Appendix — cross-session accuracy per seed | `training/train_unet_crosssession.py` | `--env rw_to_rw_env{2,3,4} --seed {42,123,777}` | `code/results_crosssession/unet_rw_to_rw_env*_seed*.json` |
| Table — zero-shot cross-session accuracy | `evaluation/eval_cross_session.py` | change `MODEL_TYPE` and `WEIGHTS_PATH` in the script | `results/` on HF |
| Table 5.7 — multi-seed baseline | `training/run_multiseed_baseline.py` | seeds 42, 123, 777; 120 epochs | `multiseed/baseline/` on HF |
| Table 5.8 — efficiency of the models | `evaluation/benchmark_models.py` | dummy data, batch size 1 | `reports/phase2_results.tex` |
| Figures — compression Pareto + parameter plots | `figures/plot_compression.py` | — | `Figures/generated/compression_*` |

---

## Exact commands to run each experiment

Note: all scripts are located in the `code/` folder. In the flat copy on Hugging Face the same scripts
have the same names without the `training/` or `evaluation/` prefixes.

```bash
cd code

# In-distribution accuracy
cp configs/params_storage/params_fig10b_single_gpu.py configs/params.py && python training/train_and_test.py
cp configs/params_storage/params_mamba_single_gpu.py  configs/params.py && python training/train_and_test.py
python training/train_distillation.py
python training/train_unet_advanced.py

# Robustness (no retraining): change MODEL_TYPE and WEIGHTS_PATH to select the model
python evaluation/test_robustness.py

# Cross-session, leave-one-session-out cross-validation
python training/train_unet_crosssession.py --env rw_to_rw_env2 --seed 42 --epochs 120
bash training/run_crosssession_table1.sh
python evaluation/aggregate_table1.py

# Zero-shot cross-session (ablation): train on in-distribution data, test on cross-session data
python evaluation/eval_cross_session.py

# Multi-seed baseline accuracy
python training/run_multiseed_baseline.py

# Efficiency comparison between the models
python evaluation/benchmark_models.py

# Figures
python figures/dump_per_sample_errors.py && python figures/plot_cdfs.py
python figures/plot_compression.py && python figures/plot_crosssession.py && python figures/plot_heatmaps.py
```

The top-level `run.sh` script will execute the common experiments (see the README files for details).

---

## Weights & logs index

Each training and testing run produced trained models (weights) and per-epoch logs of the training
loss values. These are published on the Hugging Face model repository (`JustAGeek/dloc-code`) in the
`weights/` and `logs/` folders.

| Result | Weights (HF `weights/`) | Logs (HF `logs/`) |
|---|---|---|
| DLoc baseline — 0.707 m | `baseline_{encoder,decoder,offset_decoder}_best.pth` | `baseline_50ep/` |
| MobileNetV2 — 0.707 m (std/distill) | `mobilenet_{standalone,distill}_best.pth` | `{standalone,distill}_mobilenet/` |
| TinyCNN — 3.9 m (std/distill) | `tinycnn_{standalone,distill}_best.pth` | `{standalone,distill}_tinycnn/` |
| U-Net — 0.640 m (cosine/step/distill) | `unet_{cosine_standalone,step_standalone,cosine_distill}_best.pth` | `{standalone,distill}_unet_{cosine,step}/` |
| Multi-seed baseline — 0.875 ± 0.036 m | — | `multiseed/baseline/` |
| Cross-session LOSO — 0.930 m mean | (retrained per fold) | `code/results_crosssession/*.json` |

Each trained model can be loaded into Python with `torch.load()` and `model.load_state_dict()`:

```python
model.load_state_dict(torch.load("weights/<file>.pth"))
```

---

> **Honest caveat.** The code published in this repository has been reorganized for human readability.
> The models were trained in their original flat layout, archived on Hugging Face
> (`JustAGeek/dloc-code`); therefore, if any import path needs adjusting when re-running from `code/`,
> the `sys.path` bootstrap at the top of each entry script is the place to look. Each of the numbers
> reported in this thesis is accounted for in one of the tables above, so every result is reproducible
> from the released code and weights.
