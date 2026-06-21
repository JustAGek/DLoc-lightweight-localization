#!/usr/bin/env python
"""Cross-session median error: our U-Net (mean +/- SAMPLE std over 3 seeds, ddof=1)
vs paper DLoc reference. Reads the local result JSONs in code/results_crosssession/.
Runs with NO external data download."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig, PALETTE
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
JSON_DIR = os.path.join(CODE, "results_crosssession")
OUT = os.path.join(REPO, "Figures", "generated")

# Paper DLoc reference (Table 1), median in metres.
PAPER_MED = {"env2": 0.71, "env3": 0.82, "env4": 1.05}
LABELS = {"env2": "Aug16_1\n(furniture)",
          "env3": "Aug16_3\n(diff. furniture)",
          "env4": "Aug16_4_ref\n(reflector)"}


def load_medians():
    by = {}
    for fp in glob.glob(os.path.join(JSON_DIR, "unet_rw_to_rw_env*_seed*.json")):
        d = json.load(open(fp))
        env = d["env"].split("_")[-1]            # rw_to_rw_env2 -> env2
        by.setdefault(env, []).append(d["best_median"])
    return by


def main():
    apply_style()
    by = load_medians()
    envs = ["env2", "env3", "env4"]
    if not all(e in by for e in envs):
        print("Missing JSONs in", JSON_DIR, "->", {e: len(by.get(e, [])) for e in envs})
        return

    ours_mean = [float(np.mean(by[e])) for e in envs]
    ours_std = [float(np.std(by[e], ddof=1)) for e in envs]
    paper = [PAPER_MED[e] for e in envs]

    x = np.arange(len(envs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, ours_mean, w, yerr=ours_std, capsize=4,
           color=PALETTE["unet"], label="MobileNetV2-UNet (ours, 2.41M)")
    ax.bar(x + w/2, paper, w, color=PALETTE["paper"], label="DLoc (paper, 16.5M)")
    for xi, (m, s) in enumerate(zip(ours_mean, ours_std)):
        ax.text(xi - w/2, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    for xi, p in enumerate(paper):
        ax.text(xi + w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[e] for e in envs])
    ax.set_ylabel("Median localization error (m)")
    ax.set_title("Leave-one-session-out cross-session (3 seeds)")
    ax.legend()
    savefig(fig, OUT, "crosssession_median")
    print("seeds per env:", {e: len(by[e]) for e in envs})
    print("ours mean:", [round(m, 3) for m in ours_mean],
          " sample std:", [round(s, 3) for s in ours_std])


if __name__ == "__main__":
    main()
