# Use only if no additional kwargs needed (its based on default values)

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

# to_pickle comes from data3dcanonical (not utils) so worker processes don't import torch
from data3dcanonical import make_orbits_dataset, to_pickle

# The "if __name__" guard below is required now that dataset generation uses
# multiprocessing: on Windows/macOS (which default to the 'spawn' start method),
# worker processes re-import this file, and without the guard they'd each try
# to regenerate the whole dataset themselves. Harmless on Linux/Colab (which
# default to 'fork'), but needed for this script to be safe to run locally too.
if __name__ == "__main__":
    path = '{}/{}-3Dorbits-dataset.pkl'.format(THIS_DIR, '3body')
    data = make_orbits_dataset(verbose=True)
    print('Dataset created successfully')
    to_pickle(data, path)
    print('...saved in', path)