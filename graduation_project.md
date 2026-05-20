# DLoc Graduation Project — Master Roadmap

## Project Vision

Build a deep-learning-based indoor WiFi localization system for your university: replicate the DLoc paper, improve the model, collect your own campus dataset, and create a multifloor navigation application.

---

## Part 1 — Model Replication and Improvement

> **Goal**: Fully understand, replicate, and then improve the DLoc model using the existing WILD dataset.

### 1.1 Deep Understanding Phase
- Learn all prerequisite ML/DL concepts (CNNs, ResNets, loss functions, training loops, etc.)
- Understand the DLoc paper end-to-end (WiFi signal processing, AoA/ToF, network architecture, consistency decoder logic)
- Understand the full codebase (both original and edited versions)
- Understand the data pipeline: raw CSI → MATLAB features → split datasets → PyTorch training

### 1.2 Exact Replication Phase
- Reproduce the paper's Fig 10b results on the Jacobs Hall (complex env) dataset
  - Target: median ~0.64m, P90 ~1.60m
- Reproduce Fig 10a results on the Atkinson Hall (simple env) dataset
  - Target: median ~0.36m, P90 ~0.70m
- Validate against all paper benchmarks (ablation study, generalization tests)
- Document all results with CDF plots matching the paper

### 1.3 Model Improvement Phase
- Research and apply improvement techniques:
  - Architecture changes (attention mechanisms, U-Net skip connections, transformer-based encoder)
  - Training improvements (learning rate scheduling, data augmentation, mixed precision)
  - Loss function experiments (focal loss, perceptual loss, weighted MSE)
  - Knowledge distillation (already have student_model.py started)
  - Hyperparameter optimization (batch size, learning rate, regularization)
- Benchmark every improvement against the replicated baseline
- Document which techniques help and by how much

---

## Part 2 — University Dataset Collection and Application

> **Goal**: Collect WiFi CSI data at your university and build a working multifloor indoor navigation app.

### 2.1 Hardware and Infrastructure Setup
- Identify and procure WiFi access points that can export CSI (e.g. Quantenna, Intel 5300, ESP32-S3, Nexmon on Raspberry Pi)
- Build or adapt a data collection platform (MapFind equivalent — could be a wheeled robot with SLAM, or manual collection with a smartphone + LiDAR)
- Set up ground-truth labeling (SLAM-based, or manual measurement on a floor plan)
- Deploy access points in the target building(s)

### 2.2 Data Collection at University
- Map at least one floor of one building first (proof of concept)
- Collect CSI data at thousands of labeled points covering the floor
- Process raw CSI → AoA/ToF features using the MATLAB pipeline (CSI_to_DLocFeatures)
- Create train/test splits
- Validate the data quality by training DLoc and checking error metrics

### 2.3 Multifloor Extension
- Extend the localization to multiple floors within a single building
- Add floor detection (elevation/barometer/staircase detection, or a separate classifier)
- Train per-floor models or a unified model with floor as an additional dimension
- Map additional buildings as capacity allows

### 2.4 Navigation Application
- Build a mobile app (or web app) that:
  - Shows an indoor map of the building (from SLAM or architectural floor plans)
  - Estimates the user's position in real-time using WiFi CSI
  - Provides turn-by-turn indoor navigation (pathfinding on the floor plan)
- Backend: central server runs the DLoc model, receives CSI from APs, returns location
- Frontend: displays position on the map with navigation directions

### 2.5 University-Wide Scale (Stretch Goal)
- Extend coverage to multiple buildings across campus
- Build a unified map with outdoor-to-indoor handoff
- Optimize deployment: minimize number of APs needed per floor
- Study long-term stability (does the model need retraining over months?)

---

## Timeline Suggestion

| Phase | Focus |
|-------|-------|
| Part 1.1 | Understanding (you are here) |
| Part 1.2 | Exact replication |
| Part 1.3 | Model improvements |
| Part 2.1 | Hardware setup |
| Part 2.2 | Data collection (1 floor) |
| Part 2.3 | Multifloor extension |
| Part 2.4 | Navigation app |
| Part 2.5 | Scale-up (stretch) |

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Can't get CSI-capable hardware | Use ESP32-S3 (cheap, CSI-capable) or Nexmon-patched Raspberry Pi |
| Ground truth labeling is hard | Use LiDAR SLAM on phone (iPhone/iPad) or manual measurement on architectural floor plans |
| Model doesn't generalize to new building | Retrain per-building; use transfer learning from WILD pretrained model |
| Multifloor detection is unreliable | Use barometer sensor + WiFi RSSI signature per floor as a separate classifier |
| Real-time inference is too slow | Use the distilled student model (10-15x smaller) for mobile deployment |
