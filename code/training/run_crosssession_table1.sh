#!/usr/bin/env bash
# run_crosssession_table1.sh
# Official DLoc Table 1 leave-one-session-out protocol for the
# MobileNetV2-UNet student (+ optional reproduced DLoc baseline), across seeds.
#
# Run from anywhere (it cd's to the code/ root, where data/ runs/ results live):
#   tmux new -s table1 'bash code/training/run_crosssession_table1.sh 2>&1 | tee /tmp/table1.log'
#
# PILOT:  SEEDS=(42); ENVS=(rw_to_rw_env2)
# FULL :  SEEDS=(42 123 777); ENVS=(rw_to_rw_env2 rw_to_rw_env3 rw_to_rw_env4)
set -euo pipefail

# ---- locate the code root (folder containing core/ models/ configs/ training/ evaluation/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CODE_ROOT"   # so ./data ./runs ./results_crosssession resolve relative to code/

# ------------------------------------------------------------------ config
SEEDS=(42 123 777)
ENVS=(rw_to_rw_env2 rw_to_rw_env3 rw_to_rw_env4)
UNET_EPOCHS=120
RUN_BASELINE=0   # professor: do NOT re-run baseline; compare to DLoc's published numbers
RUN_UNET=1       # our model only
PARAMS=configs/params_storage/params_tab1_crosssession_single_gpu.py
# -------------------------------------------------------------------------

echo "=== Cross-session Table 1 run (code root: $CODE_ROOT) ==="
echo "Folds:  ${ENVS[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "Baseline=${RUN_BASELINE}  U-Net=${RUN_UNET}"
echo "========================================"

# Sanity: only the Table 1 folds (rw_to_rw_env*) need the Aug16 symlinks.
NEED_AUG16=0
for env in "${ENVS[@]}"; do
    case "$env" in rw_to_rw_env*) NEED_AUG16=1 ;; esac
done
if [ "$NEED_AUG16" -eq 1 ]; then
    for f in data/dataset_train_jacobs_Aug16_1.mat \
             data/dataset_train_jacobs_Aug16_3.mat \
             data/dataset_train_jacobs_Aug16_4_ref.mat; do
        if [ ! -e "$f" ]; then
            echo "ERROR: missing $f -- create the Aug16 symlinks first (see CROSSSESSION_RUNBOOK.md)."
            exit 1
        fi
    done
fi

if [ "$RUN_BASELINE" -eq 1 ]; then
    cp "$PARAMS" configs/params.py
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="baseline_${env}_seed${seed}"
            if [ -f "./runs/${tag}/decoder/best_net_decoder.pth" ]; then
                echo "SKIP (done): $tag"; continue
            fi
            echo ">>> BASELINE $tag"
            RUN_TAG="$tag" DLOC_ENV="$env" DLOC_SEED="$seed" python training/train_and_test.py
        done
    done
fi

if [ "$RUN_UNET" -eq 1 ]; then
    # Seed-major: a complete 3-fold table is produced after the first seed.
    for seed in "${SEEDS[@]}"; do
        for env in "${ENVS[@]}"; do
            if [ -f "./results_crosssession/unet_${env}_seed${seed}.json" ]; then
                echo "SKIP (done): unet_${env}_seed${seed}"; continue
            fi
            echo ">>> U-NET unet_${env}_seed${seed}"
            python training/train_unet_crosssession.py --env "$env" --seed "$seed" --epochs "$UNET_EPOCHS" \
                || { echo "FAILED unet_${env}_seed${seed} -- continuing"; continue; }
        done
    done
fi

echo ">>> Aggregating results"
python evaluation/aggregate_table1.py
echo "=== DONE. See results_crosssession/ALLSETUPS_SUMMARY.md ==="
