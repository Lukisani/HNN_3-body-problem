# Compare four ways of rolling out a three-body orbit from the same starting states:
#
#   baseline NN  - learned derivatives, stepped with RK4
#   HNN          - derivatives from the learned Hamiltonian, stepped with RK4
#   RK4          - TRUE equations of motion, stepped with RK4
#   leapfrog     - TRUE equations of motion, stepped with leapfrog (a symplectic method)
#
# All four use the same fixed step size. They are scored on energy drift, on distance
# from a high-accuracy reference solution, and on run time.
#
# Run run_seeds.py first: this script loads the models it saved in seed_runs/.
# It needs energy_drift.py in the same folder (it reuses its helpers).
#
# Usage (from the repo root, with .venv active):
#   python experiment-3body/3D/compare_methods.py
#   python experiment-3body/3D/compare_methods.py --dt 0.05 --t_end 5

import argparse, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from energy_drift import load_model, test_initial_states, energy, rollout

METHODS = ['baseline', 'hnn', 'rk4', 'leapfrog']
LEARNED = ['baseline', 'hnn']
NAMES = {'baseline': 'Baseline NN', 'hnn': 'HNN', 'rk4': 'RK4, true equations', 'leapfrog': 'Leapfrog, true equations'}
COLORS = {'hnn': '#2a78d6', 'baseline': '#eb6834', 'rk4': '#1a9e7a', 'leapfrog': '#8a5cd6'}
EPSILON = 1e-6     # gravity softening, same as get_accelerations() in data3dcanonical.py
FLOOR = 1e-16      # smallest value shown on the log axes (double-precision rounding level)


def get_run_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--n_orbits', type=int, default=500, help='number of test orbits')
    p.add_argument('--t_end', type=float, default=5.0, help='rollout length (training orbits span t = 0 to 5)')
    p.add_argument('--dt', type=float, default=0.01, help='step size used by all four methods')
    p.add_argument('--test_seed', type=int, default=777, help='fixes which orbits are tested')
    return p.parse_args()


##### TRUE EQUATIONS OF MOTION, in canonical coordinates [q1, p1, q2, p2, q3, p3] #####
def split(x):
    x = x.reshape(*x.shape[:-1], 3, 6)
    return x[..., :3], x[..., 3:]           # positions and momenta, each (..., 3 bodies, 3)


def join(q, p):
    return np.concatenate([q, p], axis=-1).reshape(*q.shape[:-2], 18)


def accelerations(q):
    '''Gravitational acceleration on each body. Unit masses, G = 1.'''
    d = q[..., None, :, :] - q[..., :, None, :]                 # d[i, j] = q_j - q_i
    r = np.linalg.norm(d, axis=-1, keepdims=True)
    return (d / (r ** 3 + EPSILON)).sum(axis=-2)                # the i = j term is 0 / EPSILON = 0


def true_derivative(x):
    q, p = split(x)
    return join(p, accelerations(q))                            # dq/dt = p, dp/dt = force


##### FIXED-STEP INTEGRATORS OF THE TRUE EQUATIONS (all orbits at once) #####
def rk4_true(x0, n_steps, dt):
    x, path = x0.copy(), [x0.copy()]
    for _ in range(n_steps):
        k1 = true_derivative(x)
        k2 = true_derivative(x + 0.5 * dt * k1)
        k3 = true_derivative(x + 0.5 * dt * k2)
        k4 = true_derivative(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(x)
    return np.stack(path)


def leapfrog_true(x0, n_steps, dt):
    '''Kick-drift-kick leapfrog (velocity Verlet). One force evaluation per step.'''
    q, p = split(x0.copy())
    a = accelerations(q)
    path = [x0.copy()]
    for _ in range(n_steps):
        p = p + 0.5 * dt * a
        q = q + dt * p
        a = accelerations(q)
        p = p + 0.5 * dt * a
        path.append(join(q, p))
    return np.stack(path)


##### HIGH-ACCURACY REFERENCE #####
def reference(x0, t):
    '''Adaptive 8th-order Runge-Kutta with very tight tolerances, one orbit at a time.
    (Tighter than the solver settings used to build the training data, so that it is
    more accurate than every method being scored against it.)'''
    out = np.empty((len(t), len(x0), 18))
    for k, x in enumerate(x0):
        sol = solve_ivp(lambda _, y: true_derivative(y), [t[0], t[-1]], x, t_eval=t,
                        method='DOP853', rtol=1e-12, atol=1e-12)
        out[:, k] = sol.y.T
    return out


##### SCORES #####
def energy_drift(path, e0):
    d = np.abs(energy(path) - e0) / np.abs(e0)
    d[~np.isfinite(d)] = np.inf
    return d                                                     # (time, orbits)


def position_error(path, ref):
    '''RMS distance of the three bodies from their reference positions.'''
    q, q_ref = split(path)[0], split(ref)[0]
    e = np.sqrt(((q - q_ref) ** 2).sum(axis=-1).mean(axis=-1))
    e[~np.isfinite(e)] = np.inf
    return e                                                     # (time, orbits)


def main():
    run = get_run_args()
    n_steps = int(round(run.t_end / run.dt))
    t = np.arange(n_steps + 1) * run.dt

    x0, rejected = test_initial_states(run.n_orbits, run.test_seed)
    e0 = energy(x0)
    print(f"Test orbits: {len(x0)} (rejected {rejected} with close encounters), t = 0 to {run.t_end:g}, step {run.dt:g} for all methods")

    t0 = time.perf_counter()
    ref = reference(x0, t)
    ref_curve = np.median(energy_drift(ref, e0), axis=1)
    print(f"Reference solution: {time.perf_counter() - t0:.1f}s, its own median energy drift {ref_curve[-1]:.1e}")
    print("  (energy drift below that level can't be measured: the equations use slightly softened")
    print("   gravity, so the plain energy formula is only conserved to about that accuracy)\n")

    # runs[method] = list of (energy drift, position error, seconds); one entry per seed for learned models
    runs = {m: [] for m in METHODS}
    for name, fn in [('rk4', rk4_true), ('leapfrog', leapfrog_true)]:
        t0 = time.perf_counter()
        path = fn(x0, n_steps, run.dt)
        runs[name].append((energy_drift(path, e0), position_error(path, ref), time.perf_counter() - t0))
    for seed in run.seeds:
        for label in LEARNED:
            model = load_model(label, seed)
            t0 = time.perf_counter()
            path = rollout(model, x0, n_steps, run.dt)
            runs[label].append((energy_drift(path, e0), position_error(path, ref), time.perf_counter() - t0))

    def stat(values):
        v = np.array(values)
        return f"{v.mean():.2e}" + (f" +/- {v.std(ddof=1):.1e}" if len(v) > 1 else "")

    print(f"Median over {len(x0)} orbits at t = {run.t_end:g}" + (f" (+/- = spread over seeds {run.seeds})" if len(run.seeds) > 1 else ""))
    print(f"{'method':26s} {'energy drift':22s} {'position error':22s} seconds per {len(x0)} rollouts")
    for m in METHODS:
        drift = stat([np.median(r[0][-1]) for r in runs[m]])
        pos = stat([np.median(r[1][-1]) for r in runs[m]])
        secs = np.mean([r[2] for r in runs[m]])
        print(f"{NAMES[m]:26s} {drift:22s} {pos:22s} {secs:.2f}")

    plot(t, runs, run, len(x0), ref_curve)


def plot(t, runs, run, n_orbits, ref_curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = '#2b2b2b', '#6b6b6b', '#e6e6e4', '#fcfcfb'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.patch.set_facecolor(surface)
    mask = t > 0                                                 # both scores are 0 at t = 0

    panels = [(0, 'Relative energy drift  |E(t) − E(0)| / |E(0)|'),
              (1, 'Position error against the reference solution')]
    for ax, (idx, ylabel) in zip(axes, panels):
        ax.set_facecolor(surface)
        ends = {}
        for m in METHODS:
            curves = np.stack([np.median(r[idx], axis=1) for r in runs[m]])       # (seeds, time)
            curves = np.maximum(curves, FLOOR)
            mid = np.median(curves, axis=0)
            if len(curves) > 1:
                ax.fill_between(t[mask], curves.min(axis=0)[mask], curves.max(axis=0)[mask],
                                color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(t[mask], mid[mask], color=COLORS[m], linewidth=2)
            ends[m] = mid[-1]

        # Legend below the panel, largest value first, with each method's final value
        order = sorted(METHODS, key=lambda m: -ends[m])
        handles = [plt.Line2D([0], [0], color=COLORS[m], linewidth=2) for m in order]
        labels = [f"{NAMES[m]}:  {ends[m]:.1e}" for m in order]
        if idx == 0:
            # Smallest drift that can be measured (set by the gravity softening)
            floor = np.maximum(ref_curve, FLOOR)
            ax.plot(t[mask], floor[mask], color=muted, linewidth=1, linestyle='--')
            handles.append(plt.Line2D([0], [0], color=muted, linewidth=1, linestyle='--'))
            labels.append(f"Measurement floor:  {floor[-1]:.1e}")
        ax.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=ink, ncol=2,
                  loc='upper center', bbox_to_anchor=(0.5, -0.2), title=f'Value at t = {run.t_end:g}',
                  title_fontsize=8).get_title().set_color(muted)

        ax.set_yscale('log')
        ax.set_title(ylabel, loc='left', fontsize=10, color=ink)
        ax.set_xlabel('Time', fontsize=9, color=muted)
        ax.set_xlim(0, run.t_end)
        ax.grid(True, axis='y', color=grid, linewidth=0.8)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)

    n_seeds = len(runs['hnn'])
    band = f' · networks: line = median of {n_seeds} seeds, band = seed range' if n_seeds > 1 else ''
    fig.suptitle('Learned models against integrators that use the true equations', x=0.02, ha='left',
                 fontsize=12, color=ink)
    fig.text(0.02, 0.9, f'Median over {n_orbits} test orbits · step {run.dt:g} for all four methods{band}',
             fontsize=8.5, color=muted)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = os.path.join(THIS_DIR, 'seed_runs', 'compare_methods.png')
    fig.savefig(out, facecolor=fig.get_facecolor())
    print('\nPlot saved to', out)


if __name__ == '__main__':
    main()
