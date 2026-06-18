#!/usr/bin/env bash
# run.sh — convenience launcher for the DLoc project.
# Usage:  ./run.sh <command>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd="${1:-help}"

case "$cmd" in
  pilot)
    # quick 1-fold / 1-seed cross-session run to validate the pipeline
    cd "$ROOT/code"
    python training/train_unet_crosssession.py --env rw_to_rw_env2 --seed 42 --epochs 120
    ;;
  crosssession)
    # full official leave-one-session-out sweep (3 folds x 3 seeds) + aggregation
    bash "$ROOT/code/training/run_crosssession_table1.sh"
    ;;
  robustness)
    cd "$ROOT/code"; python evaluation/test_robustness.py
    ;;
  benchmark)
    cd "$ROOT/code"; python evaluation/benchmark_models.py
    ;;
  aggregate)
    cd "$ROOT/code"; python evaluation/aggregate_table1.py
    ;;
  docs)
    # compile the four defense documents (needs pdflatex + pgfplots)
    cd "$ROOT/docs"
    for f in 01_dloc_architecture 02_dataset 03_model_architectures 04_experiments; do
      pdflatex -interaction=nonstopmode "$f.tex" >/dev/null
      pdflatex -interaction=nonstopmode "$f.tex" >/dev/null
      echo "built $f.pdf"
    done
    ;;
  help|*)
    cat <<EOF
DLoc project launcher

  ./run.sh pilot         1-fold/1-seed cross-session run (validate the pipeline)
  ./run.sh crosssession  full official leave-one-out sweep (3 folds x 3 seeds) + aggregate
  ./run.sh robustness    AP-dropout / noise / blur / attenuation evaluation
  ./run.sh benchmark     params / FLOPs / latency / memory of every model
  ./run.sh aggregate     rebuild results_crosssession/ALLSETUPS_SUMMARY.md
  ./run.sh docs          compile the 4 defense PDFs in docs/

Before running experiments: put the WILD feature .mat files in code/data/ and create the
Aug16 'train_' symlinks (see code/CROSSSESSION_RUNBOOK.md).
Data + weights: HuggingFace JustAGeek/dloc-wild-fig10b (data), JustAGeek/dloc-code (weights).
EOF
    ;;
esac
