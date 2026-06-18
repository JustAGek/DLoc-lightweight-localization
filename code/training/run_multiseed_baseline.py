"""
run_multiseed_baseline.py
Multi-seed training for DLoc baseline (Enc_2Dec_Network).
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os, sys, time, random
import torch
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import localization_error, write_log
from data_loader import load_data
from modelADT import ModelADT
from joint_model import Enc_2Dec_Network

SEEDS = [42, 123, 777]
EPOCHS = 120
BATCH_SIZE = 64
NUM_WORKERS = 4

trainpath = [
    "./data/dataset_jacobs_July28.mat",
    "./data/dataset_non_fov_train_jacobs_July28_2.mat",
    "./data/dataset_fov_train_jacobs_July28_2.mat",
]
testpath = [
    "./data/dataset_fov_test_jacobs_July28_2.mat",
    "./data/dataset_non_fov_test_jacobs_July28_2.mat",
]

SEP1 = "=" * 60
SEP2 = "=" * 65
SEP3 = "-" * 65


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    from params_storage.params_fig10b_single_gpu import opt_exp, opt_encoder, opt_decoder, opt_offset_decoder
    enc = ModelADT(); enc.initialize(opt_encoder); enc.setup(opt_encoder)
    dec = ModelADT(); dec.initialize(opt_decoder); dec.setup(opt_decoder)
    off = ModelADT(); off.initialize(opt_offset_decoder); off.setup(opt_offset_decoder)
    model = Enc_2Dec_Network()
    model.initialize(opt_exp, enc, dec, off, gpu_ids=opt_exp.gpu_ids)
    return model


def train_one_seed(seed, train_loader, test_loader, save_dir):
    set_seed(seed)
    print(f"\n{SEP1}")
    print(f"  SEED {seed} -- BASELINE DLoc ({EPOCHS} epochs)")
    print(SEP1)

    model = build_model()
    seed_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    best_median = float("inf")
    best_p90 = float("inf")
    best_p99 = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # TRAIN
        model.encoder.net.train()
        model.decoder.net.train()
        model.offset_decoder.net.train()
        for data in train_loader:
            model.set_input(data[1], data[2], data[0], shuffle_channel=True)
            model.optimize_parameters()

        # LR scheduling (critical — linear decay from niter to niter+niter_decay)
        model.encoder.update_learning_rate()
        model.decoder.update_learning_rate()
        model.offset_decoder.update_learning_rate()

        # TEST
        model.eval()
        test_errors = []
        with torch.no_grad():
            for data in test_loader:
                model.set_input(data[1], data[2], data[0], shuffle_channel=False)
                model.test()
                pred = model.decoder.output.cpu().numpy()
                lbl = data[2].cpu().numpy()
                if lbl.ndim == 3:
                    lbl = lbl[:, np.newaxis, :, :]
                test_errors.extend(localization_error(pred, lbl, scale=0.1))

        test_median = np.median(test_errors)
        test_p90 = np.percentile(test_errors, 90)
        test_p99 = np.percentile(test_errors, 99)
        elapsed = time.time() - t0

        rl = f"seed{seed}_baseline"
        write_log([str(test_median)], rl, log_dir=seed_dir, log_type="test_median_error")
        write_log([str(test_p90)], rl, log_dir=seed_dir, log_type="test_90_error")
        write_log([str(test_p99)], rl, log_dir=seed_dir, log_type="test_99_error")

        marker = ""
        if test_median < best_median:
            best_median = test_median
            best_p90 = test_p90
            best_p99 = test_p99
            marker = " *"

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch}/{EPOCHS} | {elapsed:.0f}s | "
                  f"med={test_median:.3f}m P90={test_p90:.3f}m P99={test_p99:.3f}m{marker}")

    print(f"  Seed {seed} done. Best: med={best_median:.4f}m P90={best_p90:.3f}m P99={best_p99:.3f}m")
    return {"median": best_median, "p90": best_p90, "p99": best_p99}


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS} | Epochs: {EPOCHS}")

    print("\nLoading training data...")
    train_wo, train_w, train_lbl = [], [], []
    for fp in trainpath:
        wo, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        train_wo.append(wo)
        train_w.append(w)
        train_lbl.append(lbl)
    train_wo = torch.cat(train_wo, 0)
    train_w = torch.cat(train_w, 0)
    train_lbl = torch.cat(train_lbl, 0)
    print(f"Training samples: {train_w.shape[0]}")

    print("Loading testing data...")
    test_wo, test_w, test_lbl = [], [], []
    for fp in testpath:
        wo, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        test_wo.append(wo)
        test_w.append(w)
        test_lbl.append(lbl)
    test_wo = torch.cat(test_wo, 0)
    test_w = torch.cat(test_w, 0)
    test_lbl = torch.cat(test_lbl, 0)
    print(f"Testing samples: {test_w.shape[0]}")

    train_data = torch.utils.data.TensorDataset(train_wo, train_w, train_lbl)
    test_data = torch.utils.data.TensorDataset(test_wo, test_w, test_lbl)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join("./runs", f"multiseed_baseline_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        result = train_one_seed(seed, train_loader, test_loader, save_dir)
        all_results.append(result)

    medians = [r["median"] for r in all_results]
    p90s = [r["p90"] for r in all_results]
    p99s = [r["p99"] for r in all_results]

    print(f"\n\n{SEP2}")
    print(f"  MULTI-SEED RESULTS: BASELINE DLoc")
    print(SEP2)
    for i, seed in enumerate(SEEDS):
        r = all_results[i]
        print(f"  Seed {seed:<5}: median={r['median']:.4f}m  P90={r['p90']:.3f}m  P99={r['p99']:.3f}m")
    print(SEP3)
    print(f"  Median error:  {np.mean(medians):.4f} +/- {np.std(medians):.4f} m")
    print(f"  P90 error:     {np.mean(p90s):.3f} +/- {np.std(p90s):.3f} m")
    print(f"  P99 error:     {np.mean(p99s):.3f} +/- {np.std(p99s):.3f} m")
    print(SEP2)

    report_path = os.path.join(save_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Student: baseline\nMode: standalone\nSeeds: {SEEDS}\nEpochs: {EPOCHS}\n\n")
        for i, seed in enumerate(SEEDS):
            r = all_results[i]
            f.write(f"Seed {seed}: median={r['median']:.4f}m  P90={r['p90']:.3f}m  P99={r['p99']:.3f}m\n")
        f.write(f"\nMedian: {np.mean(medians):.4f} +/- {np.std(medians):.4f} m\n")
        f.write(f"P90:    {np.mean(p90s):.3f} +/- {np.std(p90s):.3f} m\n")
        f.write(f"P99:    {np.mean(p99s):.3f} +/- {np.std(p99s):.3f} m\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
