"""
aggregate_table1.py
Aggregate the cross-session (Table 1) leave-one-out runs into a single
mean +/- std table, comparing the MobileNetV2-UNet student against the
reproduced DLoc baseline and the paper's published DLoc numbers.

Reads:
  - U-Net   : ./results_crosssession/unet_<env>_seed<seed>.json
  - Baseline: ./runs/baseline_<env>_seed<seed>/decoder_test_{median,90,99}_error.txt
              (best epoch = epoch of minimum test median error)

Writes: ./results_crosssession/TABLE1_SUMMARY.md
"""
import os
import re
import glob
import json
import numpy as np

ENVS = ['rw_to_rw', 'rw_to_rw_env2', 'rw_to_rw_env3', 'rw_to_rw_env4', 'data_segment']
ENV_LABELS = {
    'rw_to_rw': 'Fig 10b: in-distribution (Jul28)',
    'rw_to_rw_env2': 'Table 1: test Aug16_1 (furniture)',
    'rw_to_rw_env3': 'Table 1: test Aug16_3 (diff furniture)',
    'rw_to_rw_env4': 'Table 1: test Aug16_4_ref (reflector)',
    'data_segment': 'Fig 13b: across space (disjoint)',
}
# Paper DLoc median / P90 (m) reference, where the paper reports a number.
# Fig 10b ~0.64m/1.6m; Table 1 from the table; Fig 13b is a CDF (no point value).
PAPER_DLOC = {
    'rw_to_rw': {'median': 0.64, 'p90': 1.60},
    'rw_to_rw_env2': {'median': 0.71, 'p90': 1.71},
    'rw_to_rw_env3': {'median': 0.82, 'p90': 2.52},
    'rw_to_rw_env4': {'median': 1.05, 'p90': 2.77},
    'data_segment': {'median': float('nan'), 'p90': float('nan')},
}


def collect_unet():
    """env -> list of dicts {median, p90, p99, seed}."""
    out = {e: [] for e in ENVS}
    for fp in sorted(glob.glob('./results_crosssession/unet_*.json')):
        with open(fp) as f:
            r = json.load(f)
        if r['env'] in out:
            out[r['env']].append({
                'median': r['best_median'], 'p90': r['best_p90'],
                'p99': r['best_p99'], 'seed': r['seed'],
            })
    return out


def collect_baseline():
    """env -> list of dicts {median, p90, p99, seed} at best (min-median) epoch."""
    out = {e: [] for e in ENVS}
    for run_dir in sorted(glob.glob('./runs/baseline_*_seed*')):
        m = re.search(r'baseline_(rw_to_rw(?:_env\d)?|data_segment)_seed(\d+)',
                      os.path.basename(run_dir))
        if not m:
            continue
        env, seed = m.group(1), int(m.group(2))
        med_f = os.path.join(run_dir, 'decoder_test_median_error.txt')
        p90_f = os.path.join(run_dir, 'decoder_test_90_error.txt')
        p99_f = os.path.join(run_dir, 'decoder_test_99_error.txt')
        if not os.path.exists(med_f):
            print(f'  WARNING: no median log in {run_dir}, skipping')
            continue
        med = np.atleast_1d(np.loadtxt(med_f))
        p90 = np.atleast_1d(np.loadtxt(p90_f)) if os.path.exists(p90_f) else None
        p99 = np.atleast_1d(np.loadtxt(p99_f)) if os.path.exists(p99_f) else None
        best = int(np.argmin(med))
        out[env].append({
            'median': float(med[best]),
            'p90': float(p90[best]) if p90 is not None and best < len(p90) else float('nan'),
            'p99': float(p99[best]) if p99 is not None and best < len(p99) else float('nan'),
            'seed': seed,
        })
    return out


def stats(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float('nan'), float('nan'), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(mean, std, n):
    if n == 0:
        return '--'
    if n == 1:
        return f'{mean:.3f}'
    return f'{mean:.3f} +/- {std:.3f}'


def main():
    unet = collect_unet()
    base = collect_baseline()

    lines = []
    lines.append('# DLoc vs MobileNetV2-UNet — All Setups, Official Protocol\n')
    lines.append('Both models run under identical data splits + meters metric across every '
                 'runnable paper setup (Fig 10b, Table 1 folds, Fig 13b). '
                 'Median / P90 / P99 in meters, mean +/- std across seeds.\n')

    for env in ENVS:
        u = unet[env]
        b = base[env]
        seeds_u = sorted(r['seed'] for r in u)
        seeds_b = sorted(r['seed'] for r in b)
        lines.append(f'\n## {ENV_LABELS[env]}  (`{env}`)\n')
        lines.append(f'U-Net seeds: {seeds_u or "none"} | Baseline seeds: {seeds_b or "none"}\n')
        lines.append('| Model | Median (m) | P90 (m) | P99 (m) |')
        lines.append('|-------|-----------|---------|---------|')
        for name, rows in [('MobileNetV2-UNet (ours)', u),
                           ('DLoc baseline (reproduced)', b)]:
            med = fmt(*stats(rows, 'median'))
            p90 = fmt(*stats(rows, 'p90'))
            p99 = fmt(*stats(rows, 'p99'))
            lines.append(f'| {name} | {med} | {p90} | {p99} |')
        pap = PAPER_DLOC[env]
        pm = '--' if np.isnan(pap['median']) else f'{pap["median"]:.2f}'
        pp = '--' if np.isnan(pap['p90']) else f'{pap["p90"]:.2f}'
        lines.append(f'| DLoc baseline (paper, ref) | {pm} | {pp} | -- |')

    # Compact summary table
    lines.append('\n## Summary — Median (m), mean +/- std across seeds\n')
    lines.append('| Fold | U-Net (ours) | Baseline (repro) | DLoc (paper) |')
    lines.append('|------|--------------|------------------|--------------|')
    for env in ENVS:
        u = fmt(*stats(unet[env], 'median'))
        b = fmt(*stats(base[env], 'median'))
        pmed = PAPER_DLOC[env]['median']
        p = '--' if np.isnan(pmed) else f'{pmed:.2f}'
        lines.append(f'| {ENV_LABELS[env]} | {u} | {b} | {p} |')

    text = '\n'.join(lines) + '\n'
    os.makedirs('./results_crosssession', exist_ok=True)
    out_path = './results_crosssession/ALLSETUPS_SUMMARY.md'
    with open(out_path, 'w') as f:
        f.write(text)
    print(text)
    print(f'Summary written to {out_path}')


if __name__ == '__main__':
    main()
