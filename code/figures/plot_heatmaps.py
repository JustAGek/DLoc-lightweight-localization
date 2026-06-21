#!/usr/bin/env python
"""Qualitative localization panels from heatmaps_<job>.npz (dump_results.py).
Each row = one model; columns = samples spread across error percentiles.
Each panel overlays the predicted heatmap with predicted (x) and ground-truth (o)
peak locations, titled with that sample's localization error.

Run after dump_results.py + scp of Figures/figure_data/.  No data download."""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Figures", "figure_data")
OUT = os.path.join(REPO, "Figures", "generated")

# pretty titles for known jobs; fallback = filename stem
TITLES = {
    "unet_fig10b": "MobileNetV2-UNet (in-dist.)",
    "mobilenet_fig10b": "MobileNetV2 (in-dist.)",
    "tinycnn_fig10b": "TinyCNN (in-dist.)",
    "unet_cross_env2": "U-Net cross-session: Aug16_1 (furniture)",
    "unet_cross_env3": "U-Net cross-session: Aug16_3 (diff. furn.)",
    "unet_cross_env4": "U-Net cross-session: Aug16_4_ref (reflector)",
}


def panel_row(job, npz, ncol=6):
    pred, gt = npz["pred"], npz["gt"]               # (k,H,W)
    xyp, xyg, err = npz["xy_pred"], npz["xy_gt"], npz["err"]
    k = min(ncol, pred.shape[0])
    fig, axes = plt.subplots(1, k, figsize=(2.5 * k, 2.9))
    if k == 1:
        axes = [axes]
    for j in range(k):
        ax = axes[j]
        vmax = float(pred[j].max()) or 1.0
        ax.imshow(pred[j], cmap="viridis", origin="lower", aspect="auto",
                  norm=PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax))
        ax.plot(xyg[j, 0], xyg[j, 1], "o", mfc="none", mec="white", mew=1.8, ms=11, label="truth")
        ax.plot(xyp[j, 0], xyp[j, 1], "x", color="red", mew=2.2, ms=9, label="pred")
        ax.set_title(f"err = {err[j]:.2f} m", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(TITLES.get(job, job), fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.6)
    fig.suptitle(TITLES.get(job, job) + "  —  predicted heatmap (samples by error percentile)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, OUT, f"heatmaps_{job}")
    plt.close(fig)


def main():
    files = sorted(glob.glob(os.path.join(DATA, "heatmaps_*.npz")))
    if not files:
        print("No heatmaps_*.npz in", DATA, "\nRun dump_results.py + scp first."); return
    for fp in files:
        job = os.path.basename(fp)[len("heatmaps_"):-len(".npz")]
        npz = np.load(fp)
        print("panel:", job, "k=", npz["pred"].shape[0])
        apply_style()
        panel_row(job, npz)


if __name__ == "__main__":
    main()
