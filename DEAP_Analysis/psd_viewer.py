import pickle

from fontTools.misc.cython import returns

subject_idx = 1

for trial_idx in range(0, 1):
    try:
        with open(f'psd_results/subject_{subject_idx:02d}_trial_{trial_idx}.pkl', 'rb') as f:
            unpickled_psd = pickle.load(f)
            print(f"{unpickled_psd}")
    except FileNotFoundError:
        print(f"File not found: subject_{subject_idx:02d}_trial_{trial_idx}.pkl")


