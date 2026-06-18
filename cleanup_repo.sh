#!/usr/bin/env bash
# ============================================================
# cleanup_repo.sh — remove clutter / unrelated files from the repo.
# REVIEW before running. Run from the repo root:  bash cleanup_repo.sh
# Nothing here touches datasets, weights, code, or git history;
# those large artifacts are excluded by .gitignore and live on HuggingFace.
# ============================================================
set -u
echo "This will delete the files listed below. Ctrl-C now to abort."
echo "Press Enter to continue..."; read -r _

# ---- Unrelated robotics homework (Dr. Askar's CSE 432 sheets) ----
rm -f "5- Sheet 5.pdf" "Sheet 6(1).pdf"

# ---- Junk / scratch images ----
rm -f xxx.png version.png RAG_chatbot.png \
      Screenshot_1.png Screenshot_2.png Screenshot_3.png Screenshot_4.png

# ---- Temporary upload logs ----
rm -f hf_push.log hf_weights.log hf_weights2.log

# ---- Superseded duplicates / archives ----
rm -f "stand alon.zip" student_mobilenet.txt train_distillation.txt
# Ahmed's original buggy scripts (fixed versions live in dloc_run/). Uncomment to remove:
# rm -rf "stand alon"

# ---- Old README (superseded by README.md). Uncomment to remove: ----
# rm -f README_github.md

echo "Done. Review 'git status' before committing."
