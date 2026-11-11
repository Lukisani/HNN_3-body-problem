# Train baseline and HNN with several random seeds, then compare them on a fixed
# held-out test set (same orbits for every seed), so the HNN-vs-baseline gap can be
# reported as a mean and spread instead of a single run.
#
# Models are saved to experiment-3body/3D/seed_runs/, so they don't overwrite the
# models the notebook loads. All seeds share the same training dataset; only the
# network initialisation and batch order change.

import argparse, os, sys, time
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.append(PARENT_DIR)
sys.path.append(THIS_DIR)

import train3DtestcodeTesting as T
from data3dcanonical import sample_orbits, MAX_DERIVATIVE


def get_run_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--total_steps', type=int, default=10000)
    p.add_argument('--test_trials', type=int, default=300, help='orbits in the held-out test set')
    p.add_argument('--test_seed', type=int, default=12345, help='fixes which orbits are in the test set')
    return p.parse_args()


def training_args(seed, total_steps, baseline):
    # Reuse the training script's defaults so settings match a normal training run
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        args = T.get_args()
    finally:
        sys.argv = old_argv
    args.seed = seed
    args.total_steps = total_steps
    args.baseline = baseline
    args.verbose = False
    return args


def per_sample_error(model, x, dxdt):
    # Mean squared error over the 18 components, one number per sample
    x = x.clone().requires_grad_(True)
    pred = model.time_derivative(x)
    return ((pred - dxdt) ** 2).mean(dim=1).detach().numpy()


def main():
    run = get_run_args()
    out_dir = os.path.join(THIS_DIR, 'seed_runs')
    os.makedirs(out_dir, exist_ok=True)

    # Fixed test set: same orbits for every seed. Filter off, so it includes close
    # encounters; those are scored separately as "extreme".
    np.random.seed(run.test_seed)
    test, _ = sample_orbits(trials=run.test_trials, max_derivative=None, num_workers=1)
    x_test = torch.tensor(test['coords'], dtype=torch.float32)
    dxdt_test = torch.tensor(test['dcoords'], dtype=torch.float32)
    extreme = np.abs(test['dcoords']).max(axis=1) > MAX_DERIVATIVE
    print(f"Test set: {len(extreme)} samples, {extreme.sum()} extreme (|dcoord| > {MAX_DERIVATIVE})\n")

    results = {'baseline': [], 'hnn': []}
    for seed in run.seeds:
        for label in ['baseline', 'hnn']:
            t0 = time.time()
            args = training_args(seed, run.total_steps, baseline=(label == 'baseline'))
            model, stats = T.train(args)
            torch.save(model.state_dict(), os.path.join(out_dir, f'3body-orbits-{label}-seed{seed}.tar'))

            err = per_sample_error(model, x_test, dxdt_test)
            r = {'seed': seed,
                 'median': np.median(err),
                 'in_range_mean': err[~extreme].mean(),
                 'extreme_mean': err[extreme].mean() if extreme.any() else float('nan')}
            results[label].append(r)
            print(f"seed {seed} {label:8s} median {r['median']:.3g} | in-range mean {r['in_range_mean']:.3g} "
                  f"| extreme mean {r['extreme_mean']:.3g} | {time.time() - t0:.0f}s")

    # Summary: mean +/- standard deviation across seeds
    print("\nSummary over seeds", run.seeds)
    for key in ['median', 'in_range_mean', 'extreme_mean']:
        b = np.array([r[key] for r in results['baseline']])
        h = np.array([r[key] for r in results['hnn']])
        ratio = b / h  # how many times lower the HNN's error is, per seed
        print(f"{key:14s} baseline {b.mean():.3g} +/- {b.std(ddof=1) if len(b) > 1 else 0:.2g} | "
              f"hnn {h.mean():.3g} +/- {h.std(ddof=1) if len(h) > 1 else 0:.2g} | "
              f"baseline/hnn per seed: {', '.join(f'{v:.2f}' for v in ratio)}")


if __name__ == '__main__':
    main()