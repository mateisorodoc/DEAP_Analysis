import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from joblib import Parallel, delayed
import mne
from scipy.signal import welch
from tqdm import tqdm

# Configuration for the DEAP dataset directory
DEAP_DATA_DIR = r"F:\DEAP\data_preprocessed_python"

# Ensure the cache directory exists
os.makedirs("cache", exist_ok=True)

# Channel names for DEAP dataset (32 EEG channels)
deap_channel_names = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
    'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6',
    'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

# Frequency bands
frequency_bands = {
    'Theta': (4, 7),
    'Alpha': (8, 13),
    'Beta': (14, 29),
    'Gamma': (30, 45)
}

# EEG system setup
sampling_frequency = 128  # DEAP's actual sampling rate
info = mne.create_info(ch_names=deap_channel_names, ch_types="eeg", sfreq=sampling_frequency)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Create directories for storing results
os.makedirs("psd_results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def cpu_psd_array_welch(data, sfreq, fmin, fmax, n_fft=512):
    """Calculate total power in frequency band using Welch's method on CPU."""
    psds = []
    for ch_data in data:
        f, pxx = welch(ch_data, fs=sfreq, nperseg=n_fft)
        mask = (f >= fmin) & (f <= fmax)
        if np.sum(mask) == 0:
            psds.append(0.0)
            continue
        delta_f = f[1] - f[0]  # Frequency resolution
        total_power = np.sum(pxx[mask]) * delta_f
        psds.append(total_power)
    return np.array(psds), f[mask]

def load_deap_data(subject_idx):
    """Load DEAP data for specified subject."""
    dat_file = os.path.join(DEAP_DATA_DIR, f"s{subject_idx:02d}.dat")
    with open(dat_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data['data'][:, :32, :], data['labels']

def process_and_save_psd(subject_idx, trial_idx):
    """Process and save PSD results for a single trial."""
    deap_eeg_data, deap_labels = load_deap_data(subject_idx)
    trial_data = np.array(deap_eeg_data[trial_idx])  # Use numpy for CPU operations
    psd_results = {}

    for band, (fmin, fmax) in frequency_bands.items():
        psd, _ = cpu_psd_array_welch(trial_data, sfreq=sampling_frequency, fmin=fmin, fmax=fmax)
        psd_results[band] = psd  # Store results directly in numpy

    # Save the results to file (on CPU)
    with open(f'psd_results/subject_{subject_idx:02d}_trial_{trial_idx}.pkl', 'wb') as f:
        pickle.dump({"trial_idx": trial_idx, "band_data": psd_results, "labels": deap_labels[trial_idx]}, f)
    return subject_idx, trial_idx

def plot_trial_from_file(subject_idx, trial_idx):
    """Generate and save visualization for a trial with autoscaling."""
    subject_folder = f"plots/subject_{subject_idx:02d}"
    os.makedirs(subject_folder, exist_ok=True)

    with open(f'psd_results/subject_{subject_idx:02d}_trial_{trial_idx}.pkl', 'rb') as f:
        result = pickle.load(f)

    band_data = result['band_data']
    trial_labels = result['labels']
    label_names = ["Valence", "Arousal", "Dominance", "Liking"]

    fig = plt.figure(figsize=(15, 18))
    outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 3])

    # Rating bars
    top_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[0], hspace=0.1)
    for i, (label_name, rating) in enumerate(zip(label_names, trial_labels)):
        ax = fig.add_subplot(top_gs[i, 0])
        ax.barh(0, rating, color='red', height=0.5, align='center')
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 11, 1))
        ax.grid(True, axis='x', linestyle="--", alpha=0.7)
        ax.text(rating + 0.3, 0, f"{label_name}: {rating:.1f}", va='center', fontsize=10)
        ax.spines[:].set_visible(False)

    # Topomaps with autoscaling
    bottom_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[1], wspace=0.4)
    for j, band in enumerate(frequency_bands.keys()):
        ax_topo = fig.add_subplot(bottom_gs[0, j])
        mne.viz.plot_topomap(band_data[band], info, axes=ax_topo, show=False, cmap="inferno", contours=0)
        ax_topo.set_title(f"{band} Band", fontsize=12)

    fig.suptitle(f"Subject {subject_idx:02d} - Trial {trial_idx + 1}", fontsize=16)
    plt.savefig(f"{subject_folder}/trial_{trial_idx + 1}.png")
    plt.close(fig)

def main():
    """Main processing pipeline with progress bars."""
    num_subjects = 32
    num_trials = 40

    # Multi-threading options (consider adjusting based on your system load)
    n_jobs = 10

    print("Starting parallel PSD computation...")
    start_time = time.time()

    psd_tasks = [
        (subject_idx, trial_idx)
        for subject_idx in range(1, num_subjects + 1)
        for trial_idx in range(num_trials)
    ]

    # Use joblib with optimized parallelism
    Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_and_save_psd)(subject_idx, trial_idx)
        for subject_idx, trial_idx in tqdm(psd_tasks, desc="Computing PSD", total=len(psd_tasks))
    )

    print(f"PSD computation completed in {time.time() - start_time:.2f} seconds")

    print("Starting parallel plotting...")
    start_plot = time.time()

    plot_tasks = [
        (subject_idx, trial_idx)
        for subject_idx in range(1, num_subjects + 1)
        for trial_idx in range(num_trials)
    ]

    # Parallel plotting tasks
    Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(plot_trial_from_file)(subject_idx, trial_idx)
        for subject_idx, trial_idx in tqdm(plot_tasks, desc="Plotting trials", total=len(plot_tasks))
    )

    print(f"Plotting completed in {time.time() - start_plot:.2f} seconds")

if __name__ == '__main__':
    main()
