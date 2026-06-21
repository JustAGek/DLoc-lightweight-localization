#!/usr/bin/env python
"""Compression story: (1) accuracy-vs-params Pareto, (2) parameter-count bars.
Numbers are the verified values from reports/ + benchmark_models.py.
Runs with NO external data download.

NOTE: U-Net FLOPs/latency were not in the original benchmark run (benchmark_models.py
get_models() omits U-Net). Add it there and re-run to fill flops for U-Net; the
Pareto (params vs median) is already complete for all five models."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig, PALETTE
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "Figures", "generated")

# params (full model) | median (Fig 10b, m) | test-time FLOPs (G, None=not benchmarked)
M = {
    "DLoc baseline":    dict(params=16.5e6, median=0.707, flops=81.06, color=PALETTE["baseline"]),
    "MobileNetV2":      dict(params=2.27e6, median=0.707, flops=1.28,  color=PALETTE["mobilenet"]),
    "MobileNetV2-UNet": dict(params=2.41e6, median=0.640, flops=1.45,  color=PALETTE["unet"]),
    "TinyCNN":          dict(params=46.7e3, median=3.92,  flops=0.66,  color=PALETTE["tinycnn"]),
    "Mamba":            dict(params=653e3,  median=10.56, flops=8.86,  color=PALETTE["mamba"]),
}


def pareto():
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for name, d in M.items():
        ax.scatter(d["params"], d["median"], s=110, color=d["color"], zorder=3, edgecolor="white")
        ax.annotate(name, (d["params"], d["median"]),
                    textcoords="offset points", xytext=(9, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Median error (m), Fig. 10b")
    ax.set_title("Accuracy vs. model size  (lower-left = better)")
    savefig(fig, OUT, "compression_pareto")


def param_bars():
    apply_style()
    names = list(M)
    params = [M[n]["params"] / 1e6 for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, params, color=[M[n]["color"] for n in names])
    ax.set_yscale("log")
    ax.set_ylabel("Parameters (millions, log)")
    ax.set_title("Model size")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, OUT, "compression_params")


if __name__ == "__main__":
    pareto()
    param_bars()
