# DLoc: Deep Learning WiFi Localization

PyTorch implementation of **DLoc** ([Deep Learning based Wireless Localization for Indoor Navigation](https://dl.acm.org/doi/10.1145/3372224.3380894), MobiCom 2020) with extensions: **Mamba SSM** variant, **MobileNetV2** and **TinyCNN** lightweight students via **knowledge distillation**.

> Graduation project — replicating and extending the UCSD WCSNG lab's DLoc system for WiFi-based indoor localization using AoA (Angle of Arrival) heatmaps.

---

## Results

All models trained for **50 epochs** on the WILD dataset (Jacobs Hall, Fig 10b configuration, 4 APs).

| Model | Parameters | Median Error | P90 | P99 | Compression |
|-------|-----------|-------------|-----|-----|-------------|
| **DLoc Baseline** (teacher) | 16.5M | **0.707 m** | 1.60 m | 4.00 m | 1x |
| **MobileNetV2** (standalone) | 2.3M | **0.707 m** | 1.60 m | 4.00 m | 7x smaller |
| **MobileNetV2** (distilled) | 2.3M | **0.707 m** | 1.53 m | 3.41 m | 7x smaller |
| **TinyCNN** (standalone) | 47K | 4.08 m | 14.0 m | 23.6 m | 350x smaller |
| **TinyCNN** (distilled) | 47K | 3.81 m | 13.5 m | 23.5 m | 350x smaller |
| **Mamba SSM** | 653K | 10.56 m | 20.0 m | 28.3 m | 25x smaller |

### Key Findings

- **MobileNetV2 matches the full DLoc baseline** with 7x fewer parameters — the original architecture is over-parameterized for this task
- **Knowledge distillation improves tail performance** — MobileNetV2 distilled achieves better P90 (1.53m vs 1.60m) and P99 (3.41m vs 4.00m)
- **TinyCNN benefits most from distillation** — 7% improvement in median error (3.81m vs 4.08m) with only 47K parameters
- **Mamba struggles with 2D spatial data** — flattening heatmaps to 1D sequences destroys spatial locality; needs learning rate warmup and gradient clipping to converge

### Paper Target (Fig 10b)

| Metric | Paper | Our Baseline | Our Best Student |
|--------|-------|-------------|-----------------|
| Median | ~0.64 m | 0.707 m | 0.707 m |
| 90th pct | ~1.60 m | 1.60 m | 1.53 m |

### Computational Benchmarks (H100, batch=1)

| Model | Params | FLOPs | CPU (ms) | GPU (ms) | GPU Mem (MB) | Weight Size |
|-------|--------|-------|----------|----------|-------------|-------------|
| **DLoc Baseline** | 10.2M | 81.06G | 66.5 | 1.6 | 92.9 | 65.9 MB |
| **MobileNetV2** | 2.27M | 1.28G | 10.5 | 2.7 | 22.4 | 9.3 MB |
| **TinyCNN** | 46.7K | 660.7M | 4.6 | 0.5 | 12.3 | 205 KB |
| **Mamba SSM** | 653K | 8.86G | N/A* | 2.0 | 84.4 | 2.6 MB |

*Mamba requires CUDA-only selective scan kernels.

---

## Architecture

### DLoc Baseline (16.5M params)
The original DLoc architecture with dual decoders:
- **Encoder**: ResNet-based (6 residual blocks, 2 downsampling layers) — processes N-channel AoA heatmaps
- **Location Decoder**: ResNet decoder — outputs 2D location heatmap (MSE + L1 regularization)
- **Consistency Decoder**: ResNet decoder — reconstructs offset-free features (MSE + L2 regularization)

### Mamba SSM (653K params)
Replaces the ResNet bottleneck with Mamba selective state-space blocks:
```
Input [B,4,161,361]
  -> CNN Stem (4->128 channels, 2x downsample)
  -> Flatten spatial -> 1D sequence
  -> 4x Mamba Residual Blocks (LayerNorm -> Mamba S6 -> residual)
  -> Reshape back to 2D spatial
  -> CNN Decoder (upsample to original resolution)
Output [B,1,161,361]
```

### MobileNetV2 Student (2.3M params)
Lightweight encoder-decoder using pretrained MobileNetV2:
```
Input [B,4,161,361]
  -> MobileNetV2 encoder (modified first conv: 3ch->4ch)
  -> Feature maps [B,320,H/32,W/32]
  -> Decoder: Upsample+Conv (320->128->64->32->1) + Sigmoid
Output [B,1,161,361]
```

### TinyCNN Student (47K params)
Ultra-lightweight depthwise separable CNN:
```
Input [B,4,161,361]
  -> Block1: Conv(4->16) + DepthwiseSep(16->16) + MaxPool
  -> Block2: DepthwiseSep(16->32) + DepthwiseSep(32->32) + MaxPool
  -> Block3: DepthwiseSep(32->64)
  -> Decoder1: ConvTranspose(64->32, stride=2)
  -> Decoder2: ConvTranspose(32->16, stride=2)
  -> Head: Conv(16->1, 1x1) + Sigmoid
Output [B,1,161,361]
```

---

## Knowledge Distillation

We use **offline distillation** — the teacher outputs are pre-computed once and stored, eliminating the teacher forward pass during student training.

**Loss function:**
```
L = alpha * MSE(student, ground_truth) + beta * MSE(student, teacher_output)
```
where alpha = 0.7, beta = 0.3.

**Pipeline:**
1. Train the DLoc baseline (teacher)
2. Run `precompute_teacher.py` — saves teacher outputs as `.pt` files (~5 GB)
3. Run `train_distillation.py` — trains student using pre-computed teacher outputs

---

## Project Structure

```
DLoc/
├── train_and_test.py          # Main training/eval script (baseline + Mamba)
├── train_distillation.py      # Student training (standalone + distillation)
├── precompute_teacher.py      # Pre-compute teacher outputs for offline distillation
├── trainer.py                 # Training loop logic
├── joint_model.py             # Joint encoder-decoder models (Enc_2Dec, Mamba_Network)
├── mamba_model.py             # Mamba-DLoc architecture (MambaDLocNet)
├── student_mobilenet.py       # MobileNetV2 student model
├── student_tinycnn.py         # TinyCNN student model
├── Generators.py              # ResNet encoder/decoder generators
├── modelADT.py                # Model wrapper (save/load/optimize)
├── data_loader.py             # Data loading utilities
├── utils.py                   # Localization error metrics, logging
├── params.py                  # Active config (copy from params_storage/)
├── params_storage/
│   ├── params_fig10b_single_gpu.py   # Baseline config (Fig 10b)
│   └── params_mamba_single_gpu.py    # Mamba config
├── weights/                   # Pre-trained model weights
│   ├── baseline_encoder_best.pth
│   ├── baseline_decoder_best.pth
│   ├── baseline_offset_decoder_best.pth
│   ├── mamba_best.pth
│   ├── mobilenet_standalone_best.pth
│   ├── mobilenet_distill_best.pth
│   ├── tinycnn_standalone_best.pth
│   └── tinycnn_distill_best.pth
└── requirements.txt
```

---

## Quick Start

### Requirements

```bash
pip install torch torchvision
pip install easydict h5py hdf5storage numpy scipy scikit-learn
# For Mamba model only:
pip install mamba-ssm causal-conv1d
```

### Dataset

Download the WILD dataset from [HuggingFace](https://huggingface.co/datasets/JustAGeek/dloc-wild-fig10b) and place in `./data/`.

Required files:
```
data/dataset_jacobs_July28.mat
data/dataset_fov_train_jacobs_July28_2.mat
data/dataset_non_fov_train_jacobs_July28_2.mat
data/dataset_fov_test_jacobs_July28_2.mat
data/dataset_non_fov_test_jacobs_July28_2.mat
```

### Train Baseline (DLoc Teacher)

```bash
cp params_storage/params_fig10b_single_gpu.py params.py
python train_and_test.py
```

### Train Mamba

```bash
cp params_storage/params_mamba_single_gpu.py params.py
python train_and_test.py
```

### Train Students (Knowledge Distillation)

```bash
# Step 1: Pre-compute teacher outputs (run once)
# Edit TEACHER_RUN_DIR in precompute_teacher.py to point to your baseline run
python precompute_teacher.py

# Step 2: Train student
# Edit STUDENT_TYPE and MODE in train_distillation.py
python train_distillation.py
```

### Configuration

Edit `train_distillation.py` CONFIG section:

```python
STUDENT_TYPE = 'mobilenet'   # 'mobilenet' or 'tinycnn'
MODE = 'distill'             # 'distill' or 'standalone'
ALPHA = 0.7                  # ground truth loss weight
BETA = 0.3                   # distillation loss weight
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
```

---

## Pre-trained Weights

All best model weights are available in the `weights/` directory and on [HuggingFace](https://huggingface.co/JustAGeek/dloc-code).

| File | Model | Size |
|------|-------|------|
| `baseline_encoder_best.pth` | DLoc encoder | 29 MB |
| `baseline_decoder_best.pth` | DLoc location decoder | 11 MB |
| `baseline_offset_decoder_best.pth` | DLoc consistency decoder | 24 MB |
| `mamba_best.pth` | Mamba-DLoc | 2.6 MB |
| `mobilenet_standalone_best.pth` | MobileNetV2 (GT only) | 9 MB |
| `mobilenet_distill_best.pth` | MobileNetV2 (distilled) | 9 MB |
| `tinycnn_standalone_best.pth` | TinyCNN (GT only) | 201 KB |
| `tinycnn_distill_best.pth` | TinyCNN (distilled) | 201 KB |

---

## Training Logs

Per-epoch metrics for all models are available on [HuggingFace](https://huggingface.co/JustAGeek/dloc-code) under `logs/`:
- Test median / P90 / P99 error
- Train median error
- Train / test loss

---

## Hardware

- **Training**: NVIDIA H100 80GB (Vast.ai)
- **Local dev**: RTX 3060 6GB, i7 11th gen, 16GB RAM, Windows 10
- **Previous runs**: Kaggle (2x T4)

---

## References

- **DLoc Paper**: Ayyalasomayajula et al., "Deep Learning based Wireless Localization for Indoor Navigation," MobiCom 2020. [ACM DL](https://dl.acm.org/doi/10.1145/3372224.3380894)
- **WILD Dataset**: [WCSNG Lab](https://wcsng.ucsd.edu/wild/)
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.
- **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018.

---

## License

This project is for academic/educational purposes. The original DLoc code and WILD dataset are from UCSD WCSNG lab.
