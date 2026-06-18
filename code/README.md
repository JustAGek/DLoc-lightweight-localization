# Code

The codebase, organized by purpose. (Previously a flat `dloc_run/` package; reorganized into
folders for clarity.)

```
code/
├── core/                # DLoc framework engine (imported by the scripts)
│   ├── Generators.py        # network builders (ResNet encoder/decoder)
│   ├── modelADT.py          # model wrapper / checkpoint I/O
│   ├── joint_model.py       # Enc_2Dec / Enc_Dec / Mamba joint models
│   ├── trainer.py           # train() / test() loops + metric logging
│   ├── data_loader.py       # HDF5 .mat loading
│   └── utils.py             # localization_error, write_log, network init, …
├── models/              # the model architectures
│   ├── student_mobilenet.py # MobileNetV2 student (2.3M)
│   ├── student_tinycnn.py   # TinyCNN student (47K)
│   ├── student_unet.py      # MobileNetV2-UNet (2.41M) — headline model
│   └── mamba_model.py       # Mamba SSM variant (653K)
├── configs/             # experiment parameters
│   ├── params.py            # the ACTIVE config (copied from params_storage/)
│   └── params_storage/      # fig10b / mamba / cross-session configs
├── training/            # training entry points
│   ├── train_and_test.py            # baseline DLoc (encoder + 2 decoders)
│   ├── train_distillation.py        # MobileNet/TinyCNN students (standalone/distill)
│   ├── train_unet_advanced.py       # U-Net (cosine/step schedulers)
│   ├── train_unet_crosssession.py   # U-Net under the official leave-one-out protocol
│   ├── run_crosssession_table1.sh   # orchestrator: 3 folds × N seeds + aggregation
│   ├── run_multiple_seeds.py        # multi-seed students
│   ├── run_multiseed_baseline.py    # multi-seed baseline
│   ├── run_all_multiseed.sh         # multi-seed orchestrator
│   └── precompute_teacher.py        # cache teacher outputs for distillation
├── evaluation/          # evaluation entry points
│   ├── test_robustness.py           # AP dropout / noise / attenuation / blur
│   ├── eval_cross_session.py        # cross-session generalization
│   ├── aggregate_table1.py          # aggregate the cross-session results
│   └── benchmark_models.py          # params / FLOPs / latency / memory
├── data_prep/
│   └── prepare_cross_session_data.py
├── results_crosssession/  # final cross-session results (JSONs + summary)
├── requirements.txt
└── CROSSSESSION_RUNBOOK.md
```

## How the imports work (important)

The scripts use flat imports (`from utils import …`, `from student_unet import …`). After the
reorganization, each **entry script** (in `training/`, `evaluation/`, `data_prep/`) begins with a
small **path bootstrap** that adds `core/`, `models/`, and `configs/` to `sys.path`, so the flat
imports resolve from any working directory. You do **not** need to set `PYTHONPATH`.

## How to run

Place the data in `code/data/` (download from HuggingFace `JustAGeek/dloc-wild-fig10b`), then run
from the `code/` directory so the relative `./data`, `./runs`, `./results_crosssession` paths
resolve:

```bash
cd code
# single fold/seed of the official cross-session experiment:
python training/train_unet_crosssession.py --env rw_to_rw_env2 --seed 42 --epochs 120
# full sweep (3 folds × 3 seeds) + aggregation:
bash training/run_crosssession_table1.sh
# robustness / benchmark:
python evaluation/test_robustness.py
python evaluation/benchmark_models.py
```

The baseline uses `configs/params.py` (copied from `configs/params_storage/…`); the orchestrator
handles this copy automatically. See `CROSSSESSION_RUNBOOK.md` for the data download + Aug16
symlink steps.

## Requirements
`torch`, `torchvision`, `h5py`, `easydict`, `hdf5storage`, `scipy`, `numpy`, `huggingface_hub`
(see `requirements.txt`).

> Note: the code was reorganized and byte-compiles cleanly, but has not been re-run end-to-end
> from this new layout (the experiments were completed in the original flat layout, with results
> archived on HuggingFace). If any import path needs adjustment, the bootstrap blocks at the top
> of the entry scripts are the place to look.
