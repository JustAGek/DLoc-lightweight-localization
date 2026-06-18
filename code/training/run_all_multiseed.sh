#!/bin/bash
# Master script: Run all multi-seed experiments (120 epochs, 3 seeds each)
# 8 variants: baseline + mobilenet×2 + tinycnn×2 + unet×3
#
# Estimated: ~28 hrs on H100

set -e
cd /workspace/dloc-code

echo "============================================"
echo "  MULTI-SEED PIPELINE START: $(date)"
echo "============================================"

# Helper to update config and run student models
run_student() {
    local STYPE=$1
    local MODE=$2
    local SCHED=$3
    echo ""
    echo "============================================"
    echo "  Starting: ${STYPE} / ${MODE} / sched=${SCHED}"
    echo "  Time: $(date)"
    echo "============================================"
    sed -i "s/STUDENT_TYPE = .*/STUDENT_TYPE = '${STYPE}'/" run_multiple_seeds.py
    sed -i "s/MODE = .*/MODE = '${MODE}'/" run_multiple_seeds.py
    sed -i "s/SCHEDULER_TYPE = .*/SCHEDULER_TYPE = '${SCHED}'/" run_multiple_seeds.py
    sed -i "s/EPOCHS = .*/EPOCHS = 120/" run_multiple_seeds.py
    python3 run_multiple_seeds.py 2>&1 | tee "/tmp/multiseed_${STYPE}_${MODE}_${SCHED}.log"
}

# 1. Baseline (separate script, ~12.8 hrs)
echo ""
echo "============================================"
echo "  Starting: BASELINE"
echo "  Time: $(date)"
echo "============================================"
python3 run_multiseed_baseline.py 2>&1 | tee /tmp/multiseed_baseline.log

# 2. MobileNet standalone (~2.5 hrs)
run_student mobilenet standalone none

# 3. MobileNet distill (~2.5 hrs)
run_student mobilenet distill none

# 4. TinyCNN standalone (~2.1 hrs)
run_student tinycnn standalone none

# 5. TinyCNN distill (~2.1 hrs)
run_student tinycnn distill none

# 6. U-Net cosine standalone (~2.7 hrs)
run_student unet standalone cosine

# 7. U-Net cosine distill (~2.7 hrs)
run_student unet distill cosine

echo ""
echo "============================================"
echo "  ALL MULTI-SEED RUNS COMPLETE: $(date)"
echo "============================================"
echo ""

# Print all reports
echo "=== FINAL SUMMARY ==="
for f in /workspace/dloc-code/runs/multiseed_*/report.txt; do
    echo ""
    echo "--- $(basename $(dirname $f)) ---"
    cat "$f"
done
