# Part 1 — Recreating and Improving the DLoc Model

Detailed plan for the first half of your graduation project.

---

## Phase 1: Deep Understanding (Pre-Requisite)

Before touching any code, you must understand:

### 1.1 ML/DL Foundations
See [understanding_dloc.md](understanding_dloc.md) for the full concept guide. Key topics:
- What a neural network does (forward pass, loss, backprop, gradient descent)
- Convolutional neural networks (Conv2d, stride, padding, feature maps)
- ResNet blocks (skip connections, why they help)
- Encoder-decoder architectures (downsampling, bottleneck, upsampling)
- Instance normalization vs batch normalization
- Loss functions (MSE, L1, regularization terms)
- Training mechanics (epochs, batches, learning rate, Adam optimizer, schedulers)

### 1.2 WiFi Signal Processing Foundations
- What is CSI (Channel State Information)
- Angle of Arrival (AoA) and Time of Flight (ToF)
- How AoA+ToF are turned into a 2D heatmap image
- Why timing offsets exist and how the consistency decoder removes them
- The polar-to-Cartesian transform that creates the XY heatmap

### 1.3 Codebase Understanding
Walk through every file and understand what it does:

| File | What it Does |
|------|-------------|
| `params.py` | Defines ALL hyperparameters for encoder, decoder, offset decoder |
| `train_and_test.py` | Entry point — loads data, builds models, trains, evaluates |
| `data_loader.py` | `load_data()` reads .mat files; `LazyMultiHDF5Dataset` reads on-demand |
| `Generators.py` | Defines `ResnetEncoder`, `ResnetDecoder`, `ResnetBlock` (the actual neural network layers) |
| `modelADT.py` | `ModelADT` — wrapper class that creates a network, sets up optimizer, loss, save/load |
| `joint_model.py` | `Enc_2Dec_Network` — joins encoder + 2 decoders, defines forward/backward pass |
| `trainer.py` | `train()` loop over epochs/batches; `test()` evaluation with error metrics |
| `utils.py` | Helper functions: `define_G`, `localization_error`, `write_log`, weight init, LR schedulers |
| `prepare_data.py` | Converts raw WILD features_*.mat → split dataset_*.mat files using index files |

### 1.4 Data Understanding
- Open a dataset .mat file in Python, inspect shapes and values
- Understand the 3 arrays: `features_w_offset` [N,4,161,361], `features_wo_offset` [N,4,161,361], `labels_gaussian_2d` [N,161,361]
- Visualize a few samples: plot the 4 AP heatmaps, the label Gaussian, overlay them
- Understand what the 161x361 grid represents (physical space at 0.1m/pixel = 16.1m x 36.1m)
- Understand the train/test splits: FOV (field of view = same spatial area) vs non-FOV

**Checkpoint**: You should be able to explain, from memory, how a WiFi signal at a phone turns into a location prediction, through every step of the pipeline.

---

## Phase 2: Exact Replication

### 2.1 Environment Setup
```bash
# Already done — .venv exists, Python 3.13
cd DLoc_pt_code
pip install -r requirements.txt
```

Verify GPU is accessible:
```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # RTX 3060
```

### 2.2 Replicate Fig 10b (Complex Environment — Jacobs Hall)

This is what your current `params.py` is set up for (`opt_exp.data = "rw_to_rw"`, `input_nc = 4`).

**Data needed** (already present in `DLoc_pt_code/data/`):
- `dataset_jacobs_July28.mat`
- `dataset_fov_train_jacobs_July28_2.mat`, `dataset_non_fov_train_jacobs_July28_2.mat`
- `dataset_fov_test_jacobs_July28_2.mat`, `dataset_non_fov_test_jacobs_July28_2.mat`

**Steps**:
1. Ensure params.py matches `params_fig10b_single_gpu.py` (batch=8, 1 GPU, n_epochs=50, data="rw_to_rw", input_nc=4)
2. Run: `python train_and_test.py`
3. Training will take ~50 epochs. Monitor:
   - `runs/<timestamp>/decoder_test_median_error.txt` — should converge toward ~0.64m
   - `runs/<timestamp>/decoder_test_90_error.txt` — should converge toward ~1.60m
4. After training, the script automatically evaluates the best model on the test set

**Target results** (from paper):
| Metric | Paper Value |
|--------|------------|
| Median error | ~0.64m |
| 90th percentile | ~1.60m |
| 99th percentile | ~3.2m |

**Your current best** (4 epochs): median=0.82m, P90=1.86m — needs more epochs.

### 2.3 Replicate Fig 10a (Simple Environment — Atkinson Hall)

**Data needed** (NOT yet present — must download):
- `features_July18.mat` from the WILD website

**Steps**:
1. Download `features_July18.mat` and place in `DLoc_pt_code/data/`
2. Run `python prepare_data.py` to create the split files
3. Copy params: `copy params_storage\params_fig10a_single_gpu.py params.py`
4. Run: `python train_and_test.py`

**Target results**:
| Metric | Paper Value |
|--------|------------|
| Median error | ~0.36m |
| 90th percentile | ~0.70m |
| 99th percentile | ~1.0m |

### 2.4 Replicate Ablation Studies (Optional but Valuable)

- **Without consistency decoder** (Fig 11): Set `n_decoders = 1` in params.py, retrain. Expect worse results (~0.48m median for simple, ~0.80m for complex)
- **Bandwidth generalization** (Fig 13a): Train with 80MHz data, test with 40/20MHz subsets
- **Spatial generalization** (Fig 13b): Use `data="data_segment"` for disjoint train/test splits
- **Cross-scenario generalization** (Table 1): Train on setups 1/3/4, test on 2, etc.

### 2.5 Produce CDF Plots

Write a script to:
1. Load the saved `decoder_test_result_epoch_best.mat` from the run
2. Extract the `error` array
3. Plot CDF (cumulative distribution function) of localization errors
4. Compare against SpotFi baseline numbers from the paper
5. Save publication-quality figures

---

## Phase 3: Model Improvement

After exact replication, try these improvement directions. **Always benchmark against your replicated baseline.**

### 3.1 Training Improvements (Easiest — Try First)

| Technique | What to Change | Expected Impact |
|-----------|---------------|----------------|
| **More epochs** | Increase `n_epochs` to 100-200 | Often the single biggest improvement |
| **Learning rate warmup** | Add linear warmup for first 5 epochs | Helps avoid bad early minima |
| **Cosine annealing LR** | Change `lr_policy='cosine'` in params | Smoother convergence |
| **Larger batch size** | If VRAM allows, try batch=16 with gradient accumulation | Stabler gradients |
| **Mixed precision (AMP)** | Wrap training in `torch.cuda.amp` | 2x speed, allows bigger batch |
| **Data augmentation** | Random horizontal flips, small rotations of heatmaps | More training variety |
| **Weight averaging (SWA)** | Average weights from last N epochs | Better generalization |

### 3.2 Loss Function Experiments

| Technique | How | Why It Might Help |
|-----------|-----|-------------------|
| **Weighted MSE** | Weight center pixels higher than edges | Focus on accuracy near the peak |
| **Focal loss** | Down-weight easy (background) pixels | Focus on the localization peak |
| **KL divergence** | Treat output as probability distribution | More natural for heatmaps |
| **Perceptual loss** | Compare feature maps, not raw pixels | Learn structural similarity |
| **Smooth L1 (Huber)** | Combines MSE + L1 | Less sensitive to outliers |

### 3.3 Architecture Improvements

| Technique | How | Why It Might Help |
|-----------|-----|-------------------|
| **U-Net skip connections** | Add skip connections from encoder to decoder | Preserve spatial detail |
| **Attention mechanism** | Add channel/spatial attention after encoder | Focus on relevant APs and regions |
| **SE blocks** | Squeeze-and-Excitation in ResNet blocks | Better channel weighting |
| **Deeper encoder** | More ResNet blocks (8-9 instead of 6) | More capacity for complex environments |
| **Separate AP encoders** | Process each AP independently before merging | Better per-AP feature extraction |
| **Transformer encoder** | Replace ResNet blocks with ViT-style attention | Global context awareness |

### 3.4 Knowledge Distillation (Already Started)

You already have `distill_train.py` and `student_model.py`. This trains a lightweight student from the full DLoc teacher.

**Steps**:
1. First complete a full teacher training run
2. Set `TEACHER_DIR` in `distill_train.py` to that run's path
3. Run `python distill_train.py`
4. Compare student vs teacher accuracy and parameter count

### 3.5 Experiment Tracking

For each experiment:
1. Record: technique name, params changed, epoch count, final median/P90/P99
2. Keep a table like:

| Experiment | Change | Epochs | Median | P90 | P99 | vs Baseline |
|-----------|--------|--------|--------|-----|-----|-------------|
| Baseline (paper replication) | — | 50 | 0.64m | 1.60m | 3.2m | — |
| Cosine LR | lr_policy=cosine | 50 | ? | ? | ? | ? |
| U-Net skips | Added skip connections | 50 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

---

## Deliverables for Part 1

1. **Replicated results** matching the paper (Fig 10a, Fig 10b, with CDF plots)
2. **Improvement results table** showing which techniques helped and by how much
3. **Best improved model** with saved weights
4. **Written report** documenting methodology, results, and analysis
5. **Clean codebase** with all experiments reproducible

---

## Common Pitfalls to Avoid

| Pitfall | Solution |
|---------|---------|
| Training too few epochs | Paper used 50 epochs; your 4-epoch run isn't converged yet |
| Comparing unfairly | Always compare at the same epoch count, same data split |
| VRAM OOM with larger batch | Use gradient accumulation or mixed precision |
| Forgetting to set `isTrain=False` for eval | The code handles this, but verify if modifying |
| Changing multiple things at once | Change ONE thing per experiment, compare, then combine winners |
| Not saving intermediate checkpoints | The code saves every epoch — keep them until you're sure |
