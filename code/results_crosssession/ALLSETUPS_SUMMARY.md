# DLoc vs MobileNetV2-UNet — All Setups, Official Protocol

Both models run under identical data splits + meters metric across every runnable paper setup (Fig 10b, Table 1 folds, Fig 13b). Median / P90 / P99 in meters, mean +/- std across seeds.


## Fig 10b: in-distribution (Jul28)  (`rw_to_rw`)

U-Net seeds: none | Baseline seeds: none

| Model | Median (m) | P90 (m) | P99 (m) |
|-------|-----------|---------|---------|
| MobileNetV2-UNet (ours) | -- | -- | -- |
| DLoc baseline (reproduced) | -- | -- | -- |
| DLoc baseline (paper, ref) | 0.64 | 1.60 | -- |

## Table 1: test Aug16_1 (furniture)  (`rw_to_rw_env2`)

U-Net seeds: [42, 123, 777] | Baseline seeds: none

| Model | Median (m) | P90 (m) | P99 (m) |
|-------|-----------|---------|---------|
| MobileNetV2-UNet (ours) | 0.852 +/- 0.036 | 1.839 +/- 0.055 | 5.197 +/- 0.180 |
| DLoc baseline (reproduced) | -- | -- | -- |
| DLoc baseline (paper, ref) | 0.71 | 1.71 | -- |

## Table 1: test Aug16_3 (diff furniture)  (`rw_to_rw_env3`)

U-Net seeds: [42, 123, 777] | Baseline seeds: none

| Model | Median (m) | P90 (m) | P99 (m) |
|-------|-----------|---------|---------|
| MobileNetV2-UNet (ours) | 0.822 +/- 0.032 | 2.793 +/- 0.114 | 9.457 +/- 0.445 |
| DLoc baseline (reproduced) | -- | -- | -- |
| DLoc baseline (paper, ref) | 0.82 | 2.52 | -- |

## Table 1: test Aug16_4_ref (reflector)  (`rw_to_rw_env4`)

U-Net seeds: [42, 123, 777] | Baseline seeds: none

| Model | Median (m) | P90 (m) | P99 (m) |
|-------|-----------|---------|---------|
| MobileNetV2-UNet (ours) | 1.116 +/- 0.013 | 2.489 +/- 0.025 | 6.834 +/- 0.246 |
| DLoc baseline (reproduced) | -- | -- | -- |
| DLoc baseline (paper, ref) | 1.05 | 2.77 | -- |

## Fig 13b: across space (disjoint)  (`data_segment`)

U-Net seeds: none | Baseline seeds: none

| Model | Median (m) | P90 (m) | P99 (m) |
|-------|-----------|---------|---------|
| MobileNetV2-UNet (ours) | -- | -- | -- |
| DLoc baseline (reproduced) | -- | -- | -- |
| DLoc baseline (paper, ref) | -- | -- | -- |

## Summary — Median (m), mean +/- std across seeds

| Fold | U-Net (ours) | Baseline (repro) | DLoc (paper) |
|------|--------------|------------------|--------------|
| Fig 10b: in-distribution (Jul28) | -- | -- | 0.64 |
| Table 1: test Aug16_1 (furniture) | 0.852 +/- 0.036 | -- | 0.71 |
| Table 1: test Aug16_3 (diff furniture) | 0.822 +/- 0.032 | -- | 0.82 |
| Table 1: test Aug16_4_ref (reflector) | 1.116 +/- 0.013 | -- | 1.05 |
| Fig 13b: across space (disjoint) | -- | -- | -- |
