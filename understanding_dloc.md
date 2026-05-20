# Understanding DLoc — All Concepts from the Ground Up

This document covers every concept you need to fully understand the DLoc project. It goes from basic ML foundations all the way to the specific DLoc architecture and training pipeline.

---

## Table of Contents

1. [The Big Picture: What DLoc Does](#1-the-big-picture-what-dloc-does)
2. [WiFi Signal Processing Foundations](#2-wifi-signal-processing-foundations)
3. [Machine Learning Foundations](#3-machine-learning-foundations)
4. [Deep Learning Foundations](#4-deep-learning-foundations)
5. [Convolutional Neural Networks (CNNs)](#5-convolutional-neural-networks-cnns)
6. [ResNet and Skip Connections](#6-resnet-and-skip-connections)
7. [Encoder-Decoder Architecture](#7-encoder-decoder-architecture)
8. [Image-to-Image Translation](#8-image-to-image-translation)
9. [DLoc's Specific Architecture](#9-dlocs-specific-architecture)
10. [The Consistency Decoder (DLoc's Key Innovation)](#10-the-consistency-decoder-dlocs-key-innovation)
11. [Loss Functions in DLoc](#11-loss-functions-in-dloc)
12. [Training Mechanics](#12-training-mechanics)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [The Data Pipeline](#14-the-data-pipeline)
15. [MapFind (Data Collection Platform)](#15-mapfind-data-collection-platform)
16. [Code-to-Concept Mapping](#16-code-to-concept-mapping)

---

## 1. The Big Picture: What DLoc Does

### The Problem
GPS doesn't work indoors. We want to locate a smartphone inside a building using WiFi signals.

### Traditional Approach (and Why It Fails)
1. A phone sends WiFi packets
2. Multiple access points (APs) with antenna arrays receive the packets
3. Each AP estimates the **Angle of Arrival (AoA)** — the direction the signal came from
4. Each AP estimates the **Time of Flight (ToF)** — how long the signal took (= distance)
5. Triangulate the phone's position

**Why it fails:**
- **Multipath**: The signal bounces off walls, furniture, screens. The AP sees multiple copies of the signal from different directions
- **Non-Line-of-Sight (NLOS)**: If the direct path is blocked, the AP only sees reflected paths, giving completely wrong angles
- **ToF offsets**: The AP and phone aren't time-synchronized, so the distance measurement has an unknown random offset
- **Low bandwidth**: WiFi at 40MHz can only resolve objects 7.5m apart in distance

### DLoc's Solution
Instead of hand-crafted heuristics to handle these problems, **use a deep neural network to learn how the environment affects WiFi signals**. The network:
1. Takes AoA/ToF heatmaps from each AP as input images
2. Learns to ignore multipath reflections, handle blocked paths, and correct ToF offsets
3. Outputs a 2D heatmap with a Gaussian peak at the predicted phone location

This frames WiFi localization as an **image-to-image translation** problem, leveraging decades of computer vision research.

### Why It's Called "Image Translation"
- **Input**: N images (one per access point), each showing where the AP "thinks" the phone might be
- **Output**: 1 image showing where the phone actually is
- This is exactly like translating a sketch to a photo, or a satellite image to a map

---

## 2. WiFi Signal Processing Foundations

### Channel State Information (CSI)
When a WiFi device sends a packet, the signal travels through the air and interacts with the environment. The **channel** describes how the environment transforms the signal. CSI is a complex-valued matrix measured at the receiver that captures:
- **Amplitude**: How strong the signal is on each frequency and antenna
- **Phase**: How much the signal was delayed on each frequency and antenna

The CSI matrix has dimensions: `[frequency subcarriers] × [receiver antennas]`

For example, 802.11ac at 80MHz has ~256 subcarriers, and an AP might have 4 antennas, giving a 256×4 complex matrix per AP per measurement.

### Angle of Arrival (AoA)
With multiple antennas spaced apart, the same signal arrives at each antenna at slightly different times. This time difference depends on the **angle** the signal comes from. Using the MUSIC algorithm or beamforming, we can estimate the AoA.

Think of it like: if sound hits your left ear before your right ear, the sound source is to your left.

### Time of Flight (ToF)
The signal's travel time is proportional to the distance it traveled (at the speed of light). By analyzing how the signal's phase changes across different frequencies, we can estimate the ToF.

**Critical problem**: The AP and phone don't share a clock, so there's an unknown constant offset in all ToF measurements. This offset is different for each AP and changes randomly.

### The AoA-ToF Heatmap
For each AP, we create a 2D image where:
- X-axis: Angle of Arrival (θ)
- Y-axis: Time of Flight / distance (r)
- Pixel intensity: How likely a signal source is at that (angle, distance)

This is computed using the MUSIC algorithm or 2D beamforming on the CSI matrix. The MATLAB code in `CSI_to_DLocFeatures/` does exactly this.

### Polar to Cartesian Transform
The AoA-ToF heatmap is in polar coordinates (angle, distance) relative to each AP. To combine heatmaps from multiple APs, we convert each to a common X-Y Cartesian coordinate system. This uses the known positions of the APs.

After this transform, each AP's heatmap shows likelihood of the phone being at each (x, y) position in the room. The heatmap dimensions 161×361 represent the physical space at 0.1m resolution (16.1m × 36.1m).

### The Three Data Arrays

| Array | Shape (per sample) | What It Is |
|-------|-------------------|-----------|
| `features_w_offset` | [N_AP, 161, 361] | Input heatmaps WITH random ToF offsets (this is what the network sees) |
| `features_wo_offset` | [N_AP, 161, 361] | Target heatmaps WITHOUT offsets (used to train the consistency decoder) |
| `labels_gaussian_2d` | [161, 361] | Ground truth: a 2D Gaussian centered on the true phone location |

The "w_offset" (with offset) images are what you'd get from a real phone — they have the timing errors. The "wo_offset" images are computed after-the-fact using the ground truth location to estimate and remove the offset. They're only available during training.

---

## 3. Machine Learning Foundations

### What is Machine Learning?
Instead of writing explicit rules (if signal strength > X and angle = Y, then location is Z), we show the computer many examples of (input, correct output) pairs and let it learn the pattern.

### Supervised Learning
DLoc is a supervised learning problem:
- **Input (X)**: WiFi heatmap images from APs
- **Output (Y)**: Location heatmap (a Gaussian peak at the right spot)
- **Training data**: Thousands of (X, Y) pairs collected by the MapFind robot

### The Learning Process
1. Show the network an input → it produces a prediction
2. Compare the prediction to the correct answer using a **loss function** (a number measuring how wrong it is)
3. Compute how to adjust the network's internal parameters to reduce the loss (**backpropagation**)
4. Adjust the parameters slightly in the direction that reduces the loss (**gradient descent**)
5. Repeat thousands of times

### Key Terminology
- **Model/Network**: The function (with learnable parameters) that maps input → output
- **Parameters/Weights**: The millions of numbers inside the network that get adjusted during training
- **Hyperparameters**: Settings YOU choose (learning rate, batch size, number of layers). Not learned
- **Loss function**: Measures how wrong the prediction is. Lower = better
- **Epoch**: One pass through the entire training dataset
- **Batch**: A group of samples processed together (e.g., 8 images at once)
- **Mini-batch gradient descent**: Update parameters after each batch (not after each individual sample)
- **Overfitting**: Network memorizes training data but fails on new data
- **Generalization**: Network works well on data it hasn't seen before

---

## 4. Deep Learning Foundations

### Neurons and Layers
A neural network is made of layers of "neurons". Each neuron:
1. Takes a weighted sum of its inputs: `z = w₁x₁ + w₂x₂ + ... + b`
2. Applies a non-linear activation function: `output = f(z)`

The weights (w) and bias (b) are the learnable parameters.

### Activation Functions
Without activation functions, stacking layers would be pointless (a stack of linear functions is still linear). Common activations:
- **ReLU**: `f(x) = max(0, x)` — simple, effective, used in most of DLoc
- **Sigmoid**: `f(x) = 1/(1+e^(-x))` — squashes output to [0,1], used as DLoc's final output layer
- **Tanh**: `f(x) = tanh(x)` — squashes to [-1,1], used in DLoc's first encoder layer

### Backpropagation
The algorithm that computes how each parameter should change to reduce the loss. It uses the chain rule of calculus to propagate gradients backward from the loss through every layer.

### Gradient Descent and Adam Optimizer
**Gradient descent**: `new_weight = old_weight - learning_rate × gradient`

DLoc uses **Adam** (Adaptive Moment Estimation), which is smarter:
- Maintains separate learning rates for each parameter
- Uses momentum (remembers past gradients) to smooth updates
- `beta1=0.5` controls momentum, `weight_decay=1e-5` adds L2 regularization

### Learning Rate
The step size for weight updates. Too high → training oscillates and diverges. Too low → training is painfully slow.
- DLoc uses `lr = 1e-5` (0.00001) — quite conservative
- A **scheduler** can adjust LR during training (DLoc uses "step" decay: multiply LR by 0.9 every N epochs)

### GPU and CUDA
Neural networks involve massive matrix multiplications. GPUs are 10-100x faster than CPUs for this. PyTorch's `.to(device)` moves data/models to GPU. `DataParallel` splits work across multiple GPUs.

---

## 5. Convolutional Neural Networks (CNNs)

### Why CNNs for Images?
Regular neural networks treat each pixel independently. CNNs use **filters** (small sliding windows) that:
1. Process local neighborhoods of pixels
2. Share the same weights across the entire image (translation invariance)
3. Build hierarchical features: edges → textures → shapes → objects

### Conv2d (2D Convolution)
The core CNN operation. A filter of size `kernel_size × kernel_size` slides across the image:

```
Input image:     Filter (3×3):     Output:
1 2 3 4          1 0 1             For each position, compute
5 6 7 8          0 1 0             the dot product of the filter
9 0 1 2          1 0 1             with the overlapping input region
3 4 5 6
```

Key parameters (as used in DLoc):
- **kernel_size**: Filter size. DLoc uses 7×7 for first/last layers, 3×3 for everything else
- **stride**: How far the filter moves each step. stride=1 keeps size, stride=2 halves the image
- **padding**: Pixels added around the border. padding=3 with kernel=7 preserves spatial size
- **in_channels / out_channels**: Number of input and output feature maps

### ConvTranspose2d (Transposed Convolution / "Deconvolution")
The opposite of convolution — it **upsamples** the image. Used in the decoder to go from small feature maps back to full resolution. With stride=2, it approximately doubles the spatial dimensions.

### Feature Maps (Channels)
A CNN layer doesn't just produce one output image — it produces many (e.g., 64). Each is called a **feature map** or **channel**. Each captures a different pattern (one might detect horizontal edges, another diagonal lines, etc.).

In DLoc:
- Input: 4 channels (one per AP)
- After first conv: 64 channels
- After downsampling: 128, then 256 channels
- After decoder upsampling: back to 64, then 1 (the output heatmap)

### Pooling and Stride
**Stride=2 convolution** is DLoc's way of reducing spatial dimensions (instead of max pooling). Each stride=2 layer halves the height and width, so the network goes:
- 161×361 → 81×181 → 41×91 (2 downsampling layers)

The decoder uses ConvTranspose2d with stride=2 to upsample back (approximately):
- 41×91 → 81×181 → 161×361 (but the output sizes depend on kernel/padding)

---

## 6. ResNet and Skip Connections

### The Problem with Deep Networks
Very deep networks (many layers) are hard to train — gradients become very small (vanishing) or very large (exploding) as they propagate through many layers.

### The ResNet Solution
Instead of learning a mapping `H(x)`, learn the **residual** `F(x) = H(x) - x`. The layer output is:

```
output = x + F(x)
```

where `F(x)` is the output of the convolutional layers. This "skip connection" (or "shortcut") means:
- If the layer can't learn anything useful, it can just set F(x)≈0 and pass through x unchanged
- Gradients flow directly through the skip connection, making training much easier

### ResnetBlock in DLoc's Code

```python
class ResnetBlock:
    def build_conv_block(self, dim):
        # Conv2d(3×3) → InstanceNorm → ReLU → Conv2d(3×3) → InstanceNorm

    def forward(self, x):
        return x + self.conv_block(x)   # ← This is the skip connection
```

Each ResNet block has 2 convolutions with normalization and ReLU, and adds the input back to the output. DLoc's encoder has 6 such blocks, and the decoders add more.

---

## 7. Encoder-Decoder Architecture

### The Concept
An **encoder** compresses the input into a compact representation (smaller spatial size, more channels). A **decoder** expands it back to the desired output size.

```
Input [4, 161, 361]
    ↓ encoder (compress)
Bottleneck [256, 41, 91]    ← compact representation
    ↓ decoder (expand)
Output [1, 161, 361]
```

### Why Compress?
- Forces the network to learn the most important features (can't memorize everything)
- Larger receptive field — each pixel in the bottleneck "sees" a large region of the input
- Fewer parameters than a fully-connected network operating at full resolution

### DLoc's Encoder
```
Input [4, 161, 361]
  → Conv2d 7×7 (4→4ch), InstanceNorm, Tanh
  → Conv2d 7×7 (4→64ch), InstanceNorm, ReLU
  → Conv2d 3×3 stride=2 (64→128ch), InstanceNorm, ReLU    ← downsample ×2
  → Conv2d 3×3 stride=2 (128→256ch), InstanceNorm, ReLU   ← downsample ×2
  → 6 × ResnetBlock (256ch)                                ← process at low resolution
Output: [256, 41, 91]
```

### DLoc's Location Decoder
```
Input: [256, 41, 91] (from encoder)
  → 3 × ResnetBlock (256ch)                                     ← blocks 7-9 (skips 1-6)
  → ConvTranspose2d stride=2 (256→128ch), InstanceNorm, ReLU    ← upsample ×2
  → ConvTranspose2d stride=2 (128→64ch), InstanceNorm, ReLU     ← upsample ×2
  → Conv2d 7×7 (64→1ch)                                         ← to single channel
  → Sigmoid                                                      ← output in [0, 1]
Output: [1, 161, 361]
```

### DLoc's Consistency Decoder
Same structure as location decoder but with **6 extra ResNet blocks** (blocks 7-12 instead of 7-9), and output channels = N_AP (4 for Jacobs) instead of 1.

---

## 8. Image-to-Image Translation

### The Concept
A family of deep learning tasks where both input and output are images with the same spatial structure. Examples:
- Sketch → Photo
- Satellite image → Map
- Blurry image → Sharp image
- **WiFi heatmaps → Location heatmap** (DLoc!)

### pix2pix
DLoc's architecture is inspired by **pix2pix** (Phillip Isola et al., 2017), a famous image-to-image translation framework. The key ideas borrowed:
- ResNet-based encoder-decoder
- Instance normalization (better than batch norm for image translation)
- Xavier weight initialization
- Adam optimizer with beta1=0.5

### Why This Framing Helps DLoc
By converting WiFi signals to 2D images, DLoc can:
1. Use proven CNN architectures designed for images
2. Combine data from multiple APs by stacking images as channels
3. Output the location as an image (soft prediction over the whole space, not just a point)
4. The 2D spatial structure of the heatmaps matches the physical layout of the room

---

## 9. DLoc's Specific Architecture

### The Full Picture

```
                    features_w_offset [N, 4, 161, 361]
                              │
                              ▼
                    ┌──────────────────┐
                    │  Shared Encoder   │   input_nc=4, 6 ResNet blocks
                    │  (ResnetEncoder)  │   Conv → Conv → ↓2 → ↓2 → 6×ResBlock
                    └────────┬─────────┘
                             │
                 encoded features [N, 256, 41, 91]
                             │
              ┌──────────────┴──────────────┐
              │                              │
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────────┐
    │ Location Decoder │            │ Consistency Decoder   │
    │ (ResnetDecoder)  │            │ (ResnetDecoder)       │
    │ 9 total blocks   │            │ 12 total blocks       │
    │ (runs 3 new)     │            │ (runs 6 new)          │
    │ output_nc=1      │            │ output_nc=4           │
    └────────┬────────┘            └──────────┬────────────┘
             │                                 │
             ▼                                 ▼
    Location heatmap                 Corrected heatmaps
    [N, 1, 161, 361]               [N, 4, 161, 361]
             │                                 │
             ▼                                 ▼
    Compare with                     Compare with
    labels_gaussian_2d              features_wo_offset
    (Llocation loss)                (Lconsistency loss)
```

### Why Two Decoders?
The **Location Decoder** alone could learn to predict locations, but it struggles because:
- The random ToF offsets make inputs inconsistent across training samples
- The same physical location looks different each time due to different offsets

The **Consistency Decoder** forces the shared encoder to first learn to remove ToF offsets. Since both decoders share the same encoder, the location decoder benefits from the encoder's learned ability to produce offset-corrected features.

### The "Shared Encoder" Is Key
Both decoders receive the **same** encoded representation. Their losses both backpropagate through the shared encoder. This means:
- The encoder must produce features useful for BOTH tasks
- Removing offsets (consistency task) and predicting location both improve together
- This is a form of **multi-task learning**

---

## 10. The Consistency Decoder (DLoc's Key Innovation)

### The ToF Offset Problem
Each AP has an unknown, random clock offset relative to the phone. This shifts the AP's heatmap radially (closer or farther from the AP). The offset is:
- Different for each AP
- Different for each measurement
- Completely unpredictable

### How the Consistency Decoder Solves It
**Key insight**: While each AP's offset is random, ALL APs observe the same phone at the same physical location. After removing offsets, all AP heatmaps should show a peak at the same (x, y) position — they should be **consistent**.

The consistency decoder is trained to output **offset-free versions** of the input heatmaps. Its target images (`features_wo_offset`) are pre-computed during data preparation:
1. Take the input heatmap with offset
2. Use the ground truth location to identify the direct path
3. Compute the expected ToF for the direct path
4. Shift the heatmap to remove the offset

### Training vs Test Time
- **Training**: Both decoders are used. Consistency decoder loss helps train the encoder.
- **Test time**: Only the encoder + location decoder are needed. The consistency decoder is discarded. No ground truth needed for offsets.

### The Ablation Study (Fig 11 in Paper)
Without the consistency decoder:
- Simple environment: median error increases from 0.36m → 0.48m
- Complex environment: median error increases from 0.64m → 0.80m

This proves the consistency decoder significantly helps the encoder learn better features.

---

## 11. Loss Functions in DLoc

### Location Decoder Loss (L2_sumL1)
```python
L_location = MSE(predicted_heatmap, target_heatmap) + λ_reg × L1_norm(predicted_heatmap) / num_pixels
```
- **MSE (Mean Squared Error)**: Penalizes the squared difference between predicted and target heatmaps. Pushes the predicted Gaussian peak to match the target.
- **L1 regularization**: Encourages the output to be sparse (mostly zeros with a sharp peak). This prevents the network from producing a "blurry" heatmap spread across many locations.
- `λ_reg = 5e-4` controls the strength of regularization

### Consistency Decoder Loss (L2_offset_loss)
```python
L_consistency = MSE(predicted_corrected, target_corrected) + λ_reg × L2_norm(predicted_corrected) / num_pixels
```
- Same MSE loss but comparing corrected heatmaps (per-AP) against the offset-free targets
- L2 regularization (not L1) here

### Total Loss
The total loss that backpropagates through the encoder is:
```
L_total = L_location + L_consistency
```
Both losses backpropagate through the shared encoder, so the encoder learns from both tasks simultaneously.

### Why MSE for Heatmaps?
The target is a 2D Gaussian centered on the true location. MSE is natural for regression tasks — it pushes the predicted heatmap to match the target Gaussian pixel-by-pixel. The peak location in the predicted heatmap is the model's location estimate.

---

## 12. Training Mechanics

### The Training Loop (what `trainer.py:train()` does)

```
For each epoch (1 to 50):
    For each mini-batch of 8 samples:
        1. Load batch: features_w_offset, features_wo_offset, labels
        2. Forward pass:
           - Encoder processes features_w_offset → encoded features
           - Location decoder processes encoded → predicted location heatmap
           - Consistency decoder processes encoded → predicted corrected heatmaps
        3. Compute losses:
           - L_location = MSE(pred_location, labels) + regularization
           - L_consistency = MSE(pred_corrected, features_wo_offset) + regularization
        4. Backward pass: compute gradients of (L_location + L_consistency) w.r.t. all parameters
        5. Optimizer step: update encoder, location decoder, and consistency decoder weights
        6. Log losses and errors

    End of epoch:
        - Evaluate on test set (no gradients, no weight updates)
        - Log median error, 90th percentile, 99th percentile
        - Save model checkpoint
        - If median error improved twice consecutively, save as "best"
        - Update learning rate (scheduler step)
```

### Batch Size
How many samples are processed together before updating weights.
- Paper used 32 (on 4 GPUs)
- Your setup uses 8 (single RTX 3060 with 6GB VRAM)
- Smaller batch = noisier gradients but less memory needed

### Epochs
One epoch = one pass through the entire training set.
- Paper trained for 50 epochs
- Your best local run did 4 epochs (not yet converged)
- Training needs to run for the full 50 (or more) to match paper results

### Learning Rate Schedule
DLoc uses **StepLR**: multiply the learning rate by 0.9 every `lr_decay_iters` epochs.
- Encoder and consistency decoder: decay every 50 epochs (effectively no decay in 50 epochs)
- Location decoder: decay every 20 epochs (2-3 decays during training)

### Saving Strategy
- **Every epoch**: Save numbered checkpoint (e.g., `3_net_encoder.pth`)
- **On improvement**: After median error improves twice consecutively, save as `best_net_*.pth`
- **Latest**: Updated frequently during training as `latest_net_*.pth`

### model.eval() vs model.train()
- **train() mode**: Dropout is active, batch/instance norm uses batch statistics
- **eval() mode**: Dropout disabled, norm uses running statistics
- DLoc calls `model.eval()` during test/evaluation, and forward passes use `torch.no_grad()` to save memory

---

## 13. Evaluation Metrics

### Localization Error
For each test sample:
1. Find the peak of the predicted heatmap: `argmax(predicted)` → (row_pred, col_pred)
2. Find the peak of the ground truth heatmap: `argmax(label)` → (row_true, col_true)
3. Compute Euclidean distance in pixels: `sqrt((row_pred - row_true)² + (col_pred - col_true)²)`
4. Convert to meters: multiply by 0.1 (each pixel = 0.1m = 10cm)

This is implemented in `utils.py:localization_error()`.

### Summary Statistics
- **Median error**: The middle value when all errors are sorted. Robust to outliers.
- **90th percentile**: 90% of errors are below this value. Measures worst-case performance.
- **99th percentile**: 99% of errors are below this. Catches rare catastrophic failures.

### CDF Plot
The Cumulative Distribution Function plot shows:
- X-axis: localization error (meters)
- Y-axis: fraction of test points with error ≤ X
- A curve more to the left = better performance
- The paper uses this to compare DLoc vs SpotFi vs Baseline DL model

### Paper's Target Results
| Environment | Median | P90 | P99 |
|------------|--------|-----|-----|
| Simple (Atkinson, 500 sq ft, 3 APs) | 0.36m | 0.70m | 1.0m |
| Complex (Jacobs, 1500 sq ft, 4 APs) | 0.64m | 1.60m | 3.2m |

DLoc outperforms SpotFi by ~80% at both median and 90th percentile.

---

## 14. The Data Pipeline

### End-to-End Flow

```
Step 1: Physical Measurement
   MapFind robot walks through building with WiFi device
   APs record Channel State Information (CSI) for each received packet
   SLAM provides ground truth (x,y) location for each measurement

Step 2: Signal Processing (MATLAB — CSI_to_DLocFeatures/)
   Raw CSI [n_freq × n_ant × n_ap] per data point
     → 2D beamforming → AoA-ToF heatmap per AP
     → Polar-to-Cartesian transform → XY heatmap per AP [161 × 361]
     → Compute offset-free version using ground truth
     → Create ground truth Gaussian label centered on (x,y)
   Output: features_*.mat with 3 arrays

Step 3: Data Splitting (prepare_data.py)
   features_*.mat + data_split_idx/*.mat
     → Extract samples by index → dataset_*.mat files
   Splits: FOV train, non-FOV train, FOV test, non-FOV test

Step 4: Training (train_and_test.py)
   dataset_*.mat → LazyMultiHDF5Dataset → DataLoader → training loop
   Output: model weights (.pth), error logs (.txt)
```

### FOV vs Non-FOV Splits
- **FOV (Field of View)**: Data points where the phone is within the AP's line-of-sight region. "Easier" localization.
- **Non-FOV**: Data points where the phone is in a non-line-of-sight or edge region. "Harder" localization.
- Training uses BOTH FOV and non-FOV data
- Testing uses BOTH FOV and non-FOV data
- This ensures the model is tested on the full difficulty range

### HDF5 Format
The .mat files are stored in HDF5 format (MATLAB v7.3). In HDF5:
- Dimensions are stored reversed compared to MATLAB
- MATLAB `[N, 4, 161, 361]` → HDF5 `[361, 161, 4, N]`
- The `LazyMultiHDF5Dataset` in `data_loader.py` handles the transpose

### Lazy Loading
The original code loaded ALL data into RAM (could be 10+ GB). The edited version uses `LazyMultiHDF5Dataset` which:
- Only reads sample indices at init time (instant, ~0 RAM)
- Reads individual samples from disk on demand during training
- Each sample read is ~700KB (3 small HDF5 slices)
- Essential for your 16GB RAM system

---

## 15. MapFind (Data Collection Platform)

### What It Is
An autonomous robot that:
1. Navigates through a building using SLAM (Simultaneous Localization and Mapping)
2. Creates a physical map of the environment
3. Collects WiFi CSI data at every position it visits
4. Labels each CSI measurement with the ground truth location from SLAM

### Components
- **Clearpath Jackal**: Mobile robot platform
- **LiDAR**: For SLAM (simultaneous localization and mapping)
- **WiFi device**: Collects CSI from all APs in the environment
- **RTAB-Map SLAM**: Algorithm that builds the map and tracks position (5.7cm median accuracy)

### Path Planning
MapFind uses a smart path-planning algorithm to maximize spatial coverage while minimizing travel distance. It uses Probabilistic Road Maps (PRM) with a greedy coverage-maximizing heuristic. The random element mimics real human walking patterns.

### Why This Matters for Your Project
You won't have a Clearpath Jackal robot. For your university dataset, you'll need to build an alternative:
- Walk manually with a phone/laptop that can capture CSI
- Use LiDAR SLAM (iPhone/iPad LiDAR, or a cheap RPLiDAR)
- Or use architectural floor plans + manual distance measurements for ground truth

---

## 16. Code-to-Concept Mapping

### How Each Concept Maps to Code

| Concept | Code Location |
|---------|-------------|
| AoA-ToF heatmap creation | `CSI_to_DLocFeatures/compute_2D_multipath_profile.m` |
| Polar → Cartesian transform | `CSI_to_DLocFeatures/convert_multipathProfile_to_xy.m` |
| Data splitting | `prepare_data.py` + `data_split_idx/*.mat` |
| Lazy data loading | `data_loader.py:LazyMultiHDF5Dataset` |
| ResNet encoder (6 blocks) | `Generators.py:ResnetEncoder` |
| ResNet decoder | `Generators.py:ResnetDecoder` |
| Residual skip connection | `Generators.py:ResnetBlock.forward()` — `return x + self.conv_block(x)` |
| Network creation + optimizer setup | `modelADT.py:ModelADT.initialize()` |
| Loss function selection | `modelADT.py:ModelADT.initialize()` (lines 51-69) |
| Forward pass + loss computation | `modelADT.py:ModelADT.forward()` |
| Joining encoder + 2 decoders | `joint_model.py:Enc_2Dec_Network` |
| Joint forward pass | `joint_model.py:Enc_2Dec_Network.forward()` |
| Training loop | `trainer.py:train()` |
| Evaluation loop | `trainer.py:test()` |
| Localization error metric | `utils.py:localization_error()` |
| Learning rate scheduling | `utils.py:get_scheduler()` |
| Weight initialization | `utils.py:init_weights()` |
| Model save/load | `modelADT.py:save_networks()` / `load_networks()` |
| All hyperparameters | `params.py` |
| Instance normalization | `utils.py:get_norm_layer()` returns `InstanceNorm2d` |
| DataParallel (multi-GPU) | `utils.py:init_net()` wraps model in `nn.DataParallel` |

### Data Flow Through the Code During Training

```
train_and_test.py
  ├─ Selects data paths based on opt_exp.data ("rw_to_rw", etc.)
  ├─ Creates LazyMultiHDF5Dataset → DataLoader for train and test
  ├─ Creates 3 ModelADT instances (encoder, decoder, offset_decoder)
  │   └─ Each calls define_G() → creates ResnetEncoder or ResnetDecoder
  │   └─ Each sets up Adam optimizer and loss function
  ├─ Creates Enc_2Dec_Network (joins them)
  └─ Calls trainer.train(joint_model, train_loader, test_loader)
       └─ For each epoch, for each batch:
            ├─ joint_model.set_input(feat_w, labels, feat_wo)
            ├─ joint_model.optimize_parameters()
            │   ├─ encoder.forward(feat_w) → encoded
            │   ├─ decoder.forward(encoded) → location prediction
            │   ├─ offset_decoder.forward(encoded) → corrected heatmaps
            │   ├─ decoder.backward() → gradients for L_location
            │   ├─ offset_decoder.backward() → gradients for L_consistency
            │   └─ All 3 optimizers step (update weights)
            └─ Log losses, compute localization_error()
```

### The Key Insight to Remember
DLoc is, at its core, an **image translation network** (like pix2pix) with a clever second decoder that forces the encoder to learn physics-aware features. The consistency decoder is only needed during training — at test time, you just need the encoder + location decoder, and the model predicts location from raw WiFi heatmaps without needing ground truth for offset correction.
