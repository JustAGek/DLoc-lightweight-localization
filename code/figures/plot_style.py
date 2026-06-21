"""Shared matplotlib style + palette for all DLoc thesis figures.
Import and call apply_style() at the top of each plot script."""
import os
import matplotlib as mpl

# Consistent, print-friendly palette used across every figure.
PALETTE = {
    "baseline": "#264653",          # dark slate  (DLoc teacher)
    "mobilenet": "#e76f51",         # orange
    "mobilenet_distill": "#f4a261", # light orange
    "unet": "#2a9d8f",              # teal  (ours / headline)
    "tinycnn": "#8d99ae",           # grey
    "mamba": "#9b5de5",             # purple
    "paper": "#6c757d",             # grey  (paper-reference bars)
    "accent": "#e63946",            # red accent
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def savefig(fig, out_dir, name):
    """Save a figure as both PDF (for LaTeX) and PNG (for quick preview)."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    print(f"  wrote {os.path.join(out_dir, name)}.pdf (+.png)")
