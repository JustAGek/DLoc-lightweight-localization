#!/usr/bin/env python
"""CDF of per-sample localization error (the paper's signature plot).
Requires the regenerated arrays -> run dump_per_sample_errors.py first."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig, PALETTE
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Figures", "figure_data")
OUT = os.path.join(REPO, "Figures", "generated")

# in-distribution: array file -> (legend label, colour)
SERIES = {
    "errors_unet_fig10b.npy":      ("MobileNetV2-UNet (0.64 m)", PALETTE["unet"]),
    "errors_mobilenet_fig10b.npy": ("MobileNetV2 (0.71 m)",      PALETTE["mobilenet"]),
    "errors_tinycnn_fig10b.npy":   ("TinyCNN",                   PALETTE["tinycnn"]),
}

# cross-session U-Net folds (leave-one-session-out)
CROSS = {
    "errors_unet_cross_env2.npy": ("Aug16_1 (furniture)",     PALETTE["unet"]),
    "errors_unet_cross_env3.npy": ("Aug16_3 (diff. furn.)",   PALETTE["mobilenet"]),
    "errors_unet_cross_env4.npy": ("Aug16_4_ref (reflector)", PALETTE["baseline"]),
}


def _cdf_plot(series, title, fname, xmax=3.0):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    found = 0
    for fn, (label, c) in series.items():
        fp = os.path.join(DATA, fn)
        if not os.path.exists(fp):
            print("missing", fp); continue
        e = np.sort(np.load(fp)); found += 1
        ax.plot(e, np.arange(1, e.size + 1) / e.size, color=c, label=label)
    if not found:
        print("No arrays for", fname, "in", DATA); plt.close(fig); return
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    ax.set_xlabel("Localization error (m)"); ax.set_ylabel("CDF")
    ax.set_title(title); ax.legend(loc="lower right")
    savefig(fig, OUT, fname)
    plt.close(fig)


def main():
    _cdf_plot(SERIES, "Fig. 10b -- error CDF (in-distribution)", "cdf_fig10b", xmax=3.0)
    _cdf_plot(CROSS, "Cross-session error CDF (U-Net, leave-one-session-out)",
              "cdf_crosssession", xmax=4.0)


if __name__ == "__main__":
    main()
