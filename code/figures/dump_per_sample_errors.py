#!/usr/bin/env python
"""
dump_per_sample_errors.py  --  regenerate PER-SAMPLE localization errors (metres)

Why this exists: training/eval only ever logged per-EPOCH median/P90/P99 scalars.
`eval_cross_session.evaluate_on_files()` computes the per-sample array and then
throws it away. CDFs, the error histogram, and qualitative heatmaps all need the
raw per-sample arrays, so we recompute them from the saved weights + test data.

Outputs (default REPO/Figures/figure_data/):
    errors_<job>.npy     1-D float32 array, per-sample error in metres
    summary.json         {job: {n, median, mean, p90, p99}}

Run:  python dump_per_sample_errors.py                 # in-distribution students
      python dump_per_sample_errors.py --crosssession  # + U-Net cross-session folds

Weights:  in-distribution -> REPO/weights/ (download from HF JustAGeek/dloc-code
          if absent).  Cross-session U-Net -> auto-discovered in dloc_run/runs_h100/.
Data:     test .mat files from HF dataset JustAGeek/dloc-wild-fig10b -> --data dir.
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

# in-distribution Fig 10b test set (Jul28) -- small
FIG10B_TEST = [
    "dataset_fov_test_jacobs_July28_2.mat",
    "dataset_non_fov_test_jacobs_July28_2.mat",
]
# cross-session held-out files (big: 7-15 GB each)
CROSS_TEST = {
    "env2": "dataset_jacobs_Aug16_1.mat",
    "env3": "dataset_jacobs_Aug16_3.mat",
    "env4": "dataset_jacobs_Aug16_4_ref.mat",
}


def build_model(model_type, weights, device):
    """weights: list of .pth paths (1 for students, 2 for baseline enc+dec)."""
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
        # DLoc test-time = sigmoid(decoder(encoder(x))); two weight files.
        # NOTE: verify the saved state_dict keys map onto these modules before trusting.
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


def per_sample_errors(model, files, device, batch=32):
    from utils import localization_error
    from data_loader import load_data
    w_list, lbl_list = [], []
    for fp in files:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        w_list.append(w); lbl_list.append(lbl)
    w = torch.cat(w_list, 0); lbl = torch.cat(lbl_list, 0)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(w, lbl), batch_size=batch, shuffle=False)
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            errs.extend(localization_error(out.cpu().numpy(), yb.numpy(), scale=0.1))
    return np.asarray(errs, dtype=np.float32)


def discover_cross_weights():
    """Find local cross-session U-Net best weights (seed 42) for each fold."""
    jobs, base = {}, os.path.join(REPO, "dloc_run", "runs_h100")
    for env in CROSS_TEST:
        hits = sorted(glob.glob(os.path.join(base, f"unet_rw_to_rw_{env}_seed42_*", "best_unet.pth")))
        if hits:
            jobs[f"unet_cross_{env}"] = ("unet", [hits[0]], [CROSS_TEST[env]])
    return jobs


def default_jobs():
    """job -> (model_type, [weights...], [test_files...]).  In-distribution students."""
    w = os.path.join(REPO, "weights")
    return {
        "unet_fig10b":      ("unet",      [os.path.join(w, "unet_cosine_standalone_best.pth")], FIG10B_TEST),
        "mobilenet_fig10b": ("mobilenet", [os.path.join(w, "mobilenet_standalone_best.pth")],   FIG10B_TEST),
        "tinycnn_fig10b":   ("tinycnn",   [os.path.join(w, "tinycnn_standalone_best.pth")],     FIG10B_TEST),
        # baseline needs two files; enable once you've verified the keys load:
        # "baseline_fig10b": ("baseline",
        #     [os.path.join(w, "baseline_encoder_best.pth"), os.path.join(w, "baseline_decoder_best.pth")],
        #     FIG10B_TEST),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crosssession", action="store_true",
                    help="also dump U-Net cross-session folds (needs big Aug16 .mat)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  data={args.data}  out={args.out}")
    os.makedirs(args.out, exist_ok=True)

    jobs = default_jobs()
    if args.crosssession:
        jobs.update(discover_cross_weights())

    summary = {}
    for name, (mt, weights, files) in jobs.items():
        files = [os.path.join(args.data, f) for f in files]
        missing = [p for p in list(weights) + files if not os.path.exists(p)]
        if missing:
            print(f"[skip] {name}: missing {missing}")
            continue
        print(f"[run ] {name}: {mt}")
        try:
            model = build_model(mt, weights, device)
            errs = per_sample_errors(model, files, device, args.batch)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            continue
        np.save(os.path.join(args.out, f"errors_{name}.npy"), errs)
        summary[name] = dict(n=int(errs.size), median=float(np.median(errs)),
                             mean=float(errs.mean()), p90=float(np.percentile(errs, 90)),
                             p99=float(np.percentile(errs, 99)))
        print(f"       n={errs.size}  median={summary[name]['median']:.3f}m  saved")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done ->", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
