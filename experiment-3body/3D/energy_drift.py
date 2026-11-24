# Energy drift of the trained baseline and HNN models over many test orbits.
#
# Each model is rolled out from the same set of starting states, and the total
# energy is tracked along each rollout. The true dynamics conserve energy exactly,
# so any change in energy is error introduced by the learned model.
#
# Run run_seeds.py first: this script loads the models it saved in seed_runs/.
#
# Usage (from the repo root, with .venv active):
#   python experiment-3body/3D/energy_drift.py
#   python experiment-3body/3D/energy_drift.py --seeds 0 1 2 3 4 --n_orbits 500 --t_end 5

import argparse, os, sys
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.append(PARENT_DIR)
sys.path.append(THIS_DIR)

import train3d as T
from nn_models import MLP
from hnn3d import HNN
from data3dcanonical import random_config, get_orbit, update, MAX_DERIVATIVE

LABELS = ['baseline', 'hnn']
COLORS = {'hnn': '#2a78d6', 'baseline': '#eb6834'}   # blue = HNN, orange = baseline
NAMES = {'hnn': 'HNN', 'baseline': 'Baseline'}


def get_run_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--n_orbits', type=int, default=500, help='number of test orbits')
    p.add_argument('--t_end', type=float, default=5.0, help='rollout length (training orbits span t = 0 to 5)')
    p.add_argument('--dt', type=float, default=0.01, help='RK4 step size')
    p.add_argument('--test_seed', type=int, default=777, help='fixes which orbits are tested')
    return p.parse_args()


def load_model(label, seed):
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        args = T.get_args()
    finally:
        sys.argv = old_argv
    baseline = (label == 'baseline')
    output_dim = args.input_dim if baseline else 1
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=baseline)
    path = os.path.join(THIS_DIR, 'seed_runs', f'3body-orbits-{label}-seed{seed}.tar')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def test_initial_states(n_orbits, test_seed):
    '''Starting states in canonical coordinates [q1, p1, q2, p2, q3, p3], shape (n, 18).
    Uses the same rule as the training filter: an orbit is kept only if no derivative
    component, sampled at 20 times over t = 0 to 5, exceeds MAX_DERIVATIVE.'''
    np.random.seed(test_seed)
    states, rejected = [], 0
    while len(states) < n_orbits:
        state = random_config()
        assert np.allclose(state[:, 0], 1.0), 'energy formula below assumes unit masses'
        orbit, _ = get_orbit(state, t_points=20, t_span=[0, 5])
        derivs = np.stack([update(None, orbit[..., k].flatten()) for k in range(orbit.shape[-1])])
        if np.abs(derivs).max() > MAX_DERIVATIVE:
            rejected += 1
            continue
        states.append(state[:, 1:7].flatten())   # [q, v] per body; with unit masses p = v
    return np.array(states), rejected


def energy(x):
    '''Total energy for canonical states x of shape (..., 18), unit masses, G = 1.'''
    x = x.reshape(*x.shape[:-1], 3, 6)
    q, p = x[..., :3], x[..., 3:]
    kinetic = 0.5 * (p ** 2).sum(axis=(-1, -2))
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            potential = potential - 1.0 / np.linalg.norm(q[..., i, :] - q[..., j, :], axis=-1)
    return kinetic + potential


def rollout(model, x0, n_steps, dt):
    '''Fixed-step RK4 of the learned dynamics, all orbits at once. Returns (n_steps + 1, n, 18).'''
    def f(x):
        x = x.detach().requires_grad_(True)
        return model.time_derivative(x).detach()

    x = torch.tensor(x0, dtype=torch.float32)
    path = [x0.copy()]
    for _ in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Freeze orbits that have already blown up, so they don't overflow
        bad = ~torch.isfinite(x).all(dim=1) | (x.abs().max(dim=1).values > 1e3)
        x = torch.where(bad[:, None], torch.full_like(x, float('nan')), x)
        path.append(x.numpy().astype(np.float64))
    return np.stack(path)


def main():
    run = get_run_args()
    n_steps = int(round(run.t_end / run.dt))
    t = np.arange(n_steps + 1) * run.dt

    x0, rejected = test_initial_states(run.n_orbits, run.test_seed)
    e0 = energy(x0)
    print(f"Test orbits: {len(x0)} (rejected {rejected} with close encounters), rollout t = 0 to {run.t_end}, dt = {run.dt}\n")

    # drift[label][seed index] = relative energy drift, shape (n_steps + 1, n_orbits)
    drift = {label: [] for label in LABELS}
    for seed in run.seeds:
        for label in LABELS:
            path = rollout(load_model(label, seed), x0, n_steps, run.dt)
            d = np.abs(energy(path) - e0) / np.abs(e0)
            d[~np.isfinite(d)] = np.inf                     # blown-up orbits count as infinite drift (and stay that way)
            drift[label].append(d)
            final = d[-1]
            print(f"seed {seed} {label:8s} median drift at t={run.t_end:g}: {np.median(final):.3g} | "
                  f"90th pct: {np.percentile(final, 90):.3g} | blew up: {np.mean(np.isinf(final)):.1%}")

    print(f"\nSummary over seeds {run.seeds} (relative energy drift |E(t) - E(0)| / |E(0)| at t = {run.t_end:g})")
    summary = {}
    for label in LABELS:
        med = np.array([np.median(d[-1]) for d in drift[label]])
        div = np.array([np.mean(np.isinf(d[-1])) for d in drift[label]])
        summary[label] = med
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"{label:8s} median drift {med.mean():.3g} +/- {sd(med):.2g} | blew up {div.mean():.1%} +/- {sd(div):.1%}")
    ratio = summary['baseline'] / summary['hnn']
    print("baseline/hnn median drift per seed:", ', '.join(f'{r:.2f}' for r in ratio))

    plot(t, drift, run, ratio)


def plot(t, drift, run, ratio):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid = '#2b2b2b', '#6b6b6b', '#e6e6e4'
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
    mask = t > 0                                              # drift is 0 at t = 0; skip it on a log axis

    ends = {}
    for label in ['baseline', 'hnn']:
        curves = np.stack([np.median(d, axis=1) for d in drift[label]])   # (seeds, time)
        mid = np.median(curves, axis=0)
        lo, hi = curves.min(axis=0), curves.max(axis=0)
        ax.fill_between(t[mask], lo[mask], hi[mask], color=COLORS[label], alpha=0.15, linewidth=0)
        ax.plot(t[mask], mid[mask], color=COLORS[label], linewidth=2, label=NAMES[label])
        ends[label] = mid[-1]

    # Direct labels at the line ends (legend kept too, so identity is never colour alone)
    for label, y in ends.items():
        if np.isfinite(y) and y > 0:
            ax.annotate(f'{NAMES[label]}  {y:.2g}', xy=(t[-1], y), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=ink)

    m = np.mean(ratio)
    if np.isfinite(m) and m >= 1:
        title = f'HNN energy drift is {m:.0f}× smaller than the baseline at t = {run.t_end:g}'
    elif np.isfinite(m):
        title = f'Baseline energy drift is {1 / m:.1f}× smaller than the HNN at t = {run.t_end:g}'
    else:
        title = 'Relative energy drift over the rollout'
    ax.set_title(title, loc='left', fontsize=11, color=ink, pad=22)
    ax.text(0, 1.02, f'Median over {drift["hnn"][0].shape[1]} test orbits · line = median of {len(run.seeds)} seeds, band = seed range',
            transform=ax.transAxes, fontsize=8, color=muted)

    ax.set_yscale('log')
    ax.set_xlabel('Time', fontsize=9, color=muted)
    ax.set_ylabel('Relative energy drift  |E(t) − E(0)| / |E(0)|', fontsize=9, color=muted)
    ax.set_xlim(0, run.t_end)
    ax.grid(True, axis='y', color=grid, linewidth=0.8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=ink)
    fig.tight_layout()

    out = os.path.join(THIS_DIR, 'seed_runs', 'energy_drift.png')
    fig.savefig(out, facecolor=fig.get_facecolor())
    print('\nPlot saved to', out)


if __name__ == '__main__':
    main()
