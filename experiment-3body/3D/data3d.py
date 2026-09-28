# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
solve_ivp = scipy.integrate.solve_ivp

import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

import pickle

# Local pickle helpers. These used to come from utils.py, but utils.py imports
# torch, and on Windows every parallel worker re-imports this module -- so each
# worker was loading torch (hundreds of MB of RAM each). Defining them here keeps
# the workers light.
def to_pickle(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

##### ENERGY #####
def potential_energy(state):
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1,state.shape[0]):
            r_ij = ((state[i:i+1,1:4] - state[j:j+1,1:4])**2).sum(1, keepdims=True)**.5
            m_i = state[i:i+1,0:1]
            m_j = state[j:j+1,0:1]
            tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=\sum_i 0.5 * p^2 / m'''
    mass = state[:, 0:1]  # (3, 1)
    p = state[:, 4:7]     # (3, 3) [px, py, pz]
    energies = 0.5 * (p**2).sum(axis=1, keepdims=True) / mass
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon=1e-3):
    # shape of state is [bodies x properties]
    net_accs = [] # [nbodies x 2]
    for i in range(state.shape[0]): # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 1:4] - state[i, 1:4] # indexes 1:3 -> pxs, pys, pyz
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        masses = other_bodies[:, 0:1] # index 0 -> mass
        pointwise_accs = masses * displacements / (distances**3 + epsilon) # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update(t, state):
    state = state.reshape(-1,7) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,1:4] = state[:,4:7] # dx, dy, dz = vx, vy, dz
    deriv[:,4:7] = get_accelerations(state)
    return deriv.reshape(-1)


##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0,2], nbodies=3, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-6 # was -9 before...

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval,  **kwargs)
    orbit = path['y'].reshape(nbodies, 7, t_points)
    print('data3dgetorbit')
    return orbit, orbit_settings


##### INITIALIZE THE THREE BODIES #####
def rotate3d(p, theta, axis):
    # General 3D rotation matrix for a given axis
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3)
    if axis == 'x':
        R[1:, 1:] = [[c, -s], [s, c]]
    elif axis == 'y':
        R[::2, ::2] = [[c, s], [-s, c]]
    elif axis == 'z':
        R[:2, :2] = [[c, -s], [s, c]]
    return (R @ p.reshape(3, 1)).squeeze()

def random_config(nu=2e-1, min_radius=0.9, max_radius=1.2):
    '''Near-circular equilateral-triangle configuration in a random 3D orientation.

    The triangle is built in the xy-plane, where the circular-orbit velocities are
    correct, and then the whole configuration (positions and velocities) is rotated
    by one random 3D rotation. (The old version rotated a random 3D position vector
    about the z-axis, which only gives a proper equilateral triangle when that vector
    lies in the xy-plane. Otherwise the bodies started too close together and with
    sub-circular speeds, so they collapsed into close encounters with huge forces.)'''
    state = np.zeros((3, 7))
    state[:, 0] = 1  # Masses
    r = np.random.rand() * (max_radius - min_radius) + min_radius

    alpha = 2 * np.pi * np.random.rand()
    p1 = r * np.array([np.cos(alpha), np.sin(alpha), 0.0])
    p2 = rotate3d(p1, theta=2*np.pi/3, axis='z')
    p3 = rotate3d(p2, theta=2*np.pi/3, axis='z')

    v1 = rotate3d(p1, theta=np.pi/2, axis='z') / r**1.5
    v1 *= np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # Circular orbit scaling
    v2 = rotate3d(v1, theta=2*np.pi/3, axis='z')
    v3 = rotate3d(v2, theta=2*np.pi/3, axis='z')

    v1 = v1 * (1 + nu * (2*np.random.rand(3) - 1))
    v2 = v2 * (1 + nu * (2*np.random.rand(3) - 1))
    v3 = v3 * (1 + nu * (2*np.random.rand(3) - 1))

    # One random rotation applied to everything, so orbits point in random directions
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    for i, (p, v) in enumerate([(p1, v1), (p2, v2), (p3, v3)]):
        state[i, 1:4], state[i, 4:7] = Q @ p, Q @ v
    return state


##### ONE TRIAL, RUN IN A WORKER PROCESS #####
def _generate_trial(seed, timesteps, nbodies, orbit_noise, min_radius, max_radius, t_span, kwargs):
    '''Generate a single orbit and return its canonical coords/derivatives/energy
    for every sampled timestep. Runs standalone in a worker process, so it must
    seed its own RNG -- otherwise forked workers can inherit the same numpy
    random state and end up generating duplicate orbits.'''
    np.random.seed(seed)

    state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
    orbit, _ = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
    batch = orbit.transpose(2, 0, 1).reshape(-1, nbodies * 7)

    xs, dxs, es = [], [], []
    for state_flat in batch:
        # Get derivatives first (needs original state with velocities)
        dstate_flat = update(None, state_flat)
        # Convert flat state to [nbodies, 7] where 7 = [mass, qx, qy, qz, vx, vy, vz]
        state_reshaped = state_flat.reshape(nbodies, 7)

        # --- Critical fix: Compute canonical coordinates (q, p) instead of (q, v) ---
        mass = state_reshaped[:, 0]          # Shape (3,)
        positions = state_reshaped[:, 1:4]   # Shape (3, 3)
        velocities = state_reshaped[:, 4:7]  # Shape (3, 3)
        momenta = mass[:, None] * velocities  # p = m*v, shape (3, 3)

        # Concatenate [positions, momenta] and flatten
        canonical_coords = np.hstack([positions, momenta]).flatten()  # Shape (18,)

        # Process derivatives similarly
        dstate_reshaped = dstate_flat.reshape(nbodies, 7)
        d_positions = dstate_reshaped[:, 1:4]     # dq/dt = v (already correct)
        d_velocities = dstate_reshaped[:, 4:7]    # dv/dt = acceleration
        d_momenta = mass[:, None] * d_velocities  # dp/dt = m*dv/dt
        d_canonical_coords = np.hstack([d_positions, d_momenta]).flatten()

        xs.append(canonical_coords)
        dxs.append(d_canonical_coords)

        # Energy computation remains unchanged
        shaped_state = state_flat.copy().reshape(nbodies, 7, 1)
        es.append(total_energy(shaped_state))

    return xs, dxs, es


##### INTEGRATE AN ORBIT OR TWO #####
def sample_orbits(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                  min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False,
                  num_workers=None, **kwargs): #trials were at 5000 before and timesteps at 20 - reduced to make dataset generation faster

    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 3-body orbits:")

    N = timesteps * trials
    # Cap the default: each worker is a full Python process, so using every core
    # can exhaust RAM on machines with many cores. Pass num_workers to override.
    num_workers = num_workers or min(max((os.cpu_count() or 2) - 1, 1), 6)

    # Each trial gets its own seed, derived from a random base seed, so that
    # parallel worker processes don't end up producing correlated orbits.
    base_seed = np.random.randint(0, 2**31 - 1)
    seeds = [int(base_seed + i) for i in range(trials)]

    worker = partial(_generate_trial, timesteps=timesteps, nbodies=nbodies,
                      orbit_noise=orbit_noise, min_radius=min_radius,
                      max_radius=max_radius, t_span=t_span, kwargs=kwargs)

    x, dx, e = [], [], []
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.imap(worker, seeds)
            for xs, dxs, es in tqdm(results, total=trials, desc="Generating orbits"):
                x.extend(xs); dx.extend(dxs); e.extend(es)
    else:
        # num_workers=1 keeps the old single-process behaviour (e.g. for debugging)
        for seed in tqdm(seeds, desc="Generating orbits"):
            xs, dxs, es = worker(seed)
            x.extend(xs); dx.extend(dxs); e.extend(es)

    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N] }
    return data, orbit_settings


##### MAKE A DATASET #####
def make_orbits_dataset(test_split=0.2, **kwargs):
    data, orbit_settings = sample_orbits(**kwargs)
    
    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    data['meta'] = orbit_settings
    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-3Dorbits-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_orbits_dataset(**kwargs)
        print(data)
        to_pickle(data, path)

    return data