#!/usr/bin/env python
"""
dump_results.py -- ONE pass over each .mat: per-sample errors (for CDFs) AND
qualitative heatmap panels (pred vs ground-truth). Loading a 15 GB .mat twice
is the budget killer, so errors + heatmaps share a single data load + forward.

Outputs (default REPO/Figures/figure_data/):
    errors_<job>.npy      1-D float32 per-sample error (metres)
    heatmaps_<job>.npz    pred/gt heatmaps for ~6 samples spread across error
                          percentiles (best, p25, median, p75, p90, worst)
    summary.json          {job: {n, median, mean, p90, p99}}

Run:  python dump_results.py                 # in-distribution students
      python dump_results.py --crosssession  # + U-Net cross-session folds
      python dump_results.py --only unet_cross_env2   # single job
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)                 # .../code
REPO = os.path.dirname(CODE)                 # .../DLOC
for sub in ("core", "models"):
    p = os.path.join(CODE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DEFAULT = os.path.join(REPO, "data")
OUT_DEFAULT  = os.path.join(REPO, "Figures", "figure_data")

FIG10B_TEST = [
    "dataset_fov_test_jacobs_July28_2.mat",
    "dataset_non_fov_test_jacobs_July28_2.mat",
]
CROSS_TEST = {
    "env2": "dataset_jacobs_Aug16_1.mat",
    "env3": "dataset_jacobs_Aug16_3.mat",
    "env4": "dataset_jacobs_Aug16_4_ref.mat",
}
# how many qualitative samples and at which error percentiles
QUAL_PCTL = [2, 25, 50, 75, 90, 98]


def build_model(model_type, weights, device):
    mt = model_type.lower()
    if mt == "mobilenet":
        from student_mobilenet import MobileNetStudent
        m = MobileNetStudent(input_nc=4)
        m.load_state_dict(torch.load(weights[0], map_location=device, weights_only=True))
    elif mt == "tinycnn":
        from student_tinycnn import TinyCNNStudent
        m = TinyCNNStudent(input_nc=4)
        m.load_state_dict(torch.load(weights[0], map_location=device, weights_only=True))
    elif mt == "unet":
        from student_unet import MobileNetUNetStudent
        m = MobileNetUNetStudent(input_nc=4)
        m.load_state_dict(torch.load(weights[0], map_location=device, weights_only=True))
    elif mt == "baseline":
        import torch.nn as nn
        from Generators import ResnetEncoder, ResnetDecoder
        enc = ResnetEncoder(input_nc=4, output_nc=256, ngf=64, n_blocks=6)
        dec = ResnetDecoder(input_nc=256, output_nc=1, ngf=64, n_blocks=9, encoder_blocks=6)
        enc.load_state_dict(torch.load(weights[0], map_location=device, weights_only=True))
        dec.load_state_dict(torch.load(weights[1], map_location=device, weights_only=True))

        class _Baseline(nn.Module):
            def __init__(s):
                super().__init__(); s.encoder = enc; s.decoder = dec
            def forward(s, x):
                return torch.sigmoid(s.decoder(s.encoder(x)))
        m = _Baseline()
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    return m.to(device).eval()


def load_concat(files):
    from data_loader import load_data
    w_list, lbl_list = [], []
    for fp in files:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        w_list.append(w); lbl_list.append(lbl)
    return torch.cat(w_list, 0), torch.cat(lbl_list, 0)


def run_job(model, w, lbl, device, batch=32):
    """Single forward pass -> per-sample errors. Returns (errors, loader-order preserved)."""
    from utils import localization_error
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(w, lbl), batch_size=batch, shuffle=False)
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            errs.extend(localization_error(out.cpu().numpy(), yb.numpy(), scale=0.1))
    return np.asarray(errs, dtype=np.float32)


def capture_heatmaps(model, w, lbl, errs, device):
    """Re-run model on a handful of percentile-selected samples; save pred+gt heatmaps.
    The selected re-forward is ~6 samples = negligible vs the full pass."""
    order = np.argsort(errs)
    n = errs.size
    idx = sorted({int(order[min(n - 1, int(round((p / 100.0) * (n - 1))))]) for p in QUAL_PCTL})
    preds, gts, xy_pred, xy_gt, sel_err = [], [], [], [], []
    with torch.no_grad():
        for i in idx:
            out = model(w[i:i+1].to(device)).cpu().numpy()[0, 0]   # (H,W)
            gt = lbl[i].numpy()[0]                                  # (H,W)
            preds.append(out.astype(np.float32)); gts.append(gt.astype(np.float32))
            py, px = np.unravel_index(int(np.argmax(out)), out.shape)
            gy, gx = np.unravel_index(int(np.argmax(gt)),  gt.shape)
            xy_pred.append([px, py]); xy_gt.append([gx, gy]); sel_err.append(float(errs[i]))
    return dict(pred=np.stack(preds), gt=np.stack(gts),
                xy_pred=np.array(xy_pred), xy_gt=np.array(xy_gt),
                err=np.array(sel_err, np.float32), idx=np.array(idx))


def discover_cross_weights():
    jobs, base = {}, os.path.join(REPO, "dloc_run", "runs_h100")
    for env in CROSS_TEST:
        hits = sorted(glob.glob(os.path.join(base, f"unet_rw_to_rw_{env}_seed42_*", "best_unet.pth")))
        if hits:
            jobs[f"unet_cross_{env}"] = ("unet", [hits[0]], [CROSS_TEST[env]])
    return jobs


def default_jobs():
    w = os.path.join(REPO, "weights")
    return {
        "unet_fig10b":      ("unet",      [os.path.join(w, "unet_cosine_standalone_best.pth")], FIG10B_TEST),
        "mobilenet_fig10b": ("mobilenet", [os.path.join(w, "mobilenet_standalone_best.pth")],   FIG10B_TEST),
        "tinycnn_fig10b":   ("tinycnn",   [os.path.join(w, "tinycnn_standalone_best.pth")],     FIG10B_TEST),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crosssession", action="store_true")
    ap.add_argument("--only", default=None, help="run a single job name")
    ap.add_argument("--no-heatmaps", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  data={args.data}  out={args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    jobs = default_jobs()
    if args.crosssession:
        jobs.update(discover_cross_weights())
    if args.only:
        jobs = {k: v for k, v in jobs.items() if k == args.only}

    # merge into any existing summary so partial/resumed runs accumulate
    spath = os.path.join(args.out, "summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else {}

    for name, (mt, weights, files) in jobs.items():
        fpaths = [os.path.join(args.data, f) for f in files]
        missing = [p for p in list(weights) + fpaths if not os.path.exists(p)]
        if missing:
            print(f"[skip] {name}: missing {missing}", flush=True); continue
        print(f"[run ] {name}: {mt}", flush=True)
        try:
            model = build_model(mt, weights, device)
            w, lbl = load_concat(fpaths)
            print(f"       loaded {tuple(w.shape)}", flush=True)
            errs = run_job(model, w, lbl, device, args.batch)
            np.save(os.path.join(args.out, f"errors_{name}.npy"), errs)
            if not args.no_heatmaps:
                hm = capture_heatmaps(model, w, lbl, errs, device)
                np.savez_compressed(os.path.join(args.out, f"heatmaps_{name}.npz"), **hm)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[FAIL] {name}: {e}", flush=True); continue
        summary[name] = dict(n=int(errs.size), median=float(np.median(errs)),
                             mean=float(errs.mean()), p90=float(np.percentile(errs, 90)),
                             p99=float(np.percentile(errs, 99)))
        json.dump(summary, open(spath, "w"), indent=2)   # write after every job
        print(f"       n={errs.size}  median={summary[name]['median']:.3f}m  saved", flush=True)
        del w, lbl
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("done ->", spath, flush=True)


if __name__ == "__main__":
    main()
