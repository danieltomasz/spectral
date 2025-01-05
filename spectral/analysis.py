"""Functions to do the analysis of the data"""

from typing import List, Optional, Any
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from specparam import SpectralGroupModel

from .spectral import specparam2pandas
from .simulation import create_simulated_epochs, SimulationFunc
from .itpc import compute_itpc, ITPC, average_itpc


def compute_psd_and_fit(
    epochs: mne.Epochs, freq_range: list = None, peak_params: dict = None
) -> tuple:
    # Default frequency range
    if freq_range is None:
        freq_range = [2, 40]

    # Default peak parameters
    # Default peak parameters
    default_peak_params = {
        "peak_width_limits": [3, 8],
        "min_peak_height": 0.2,
        "max_n_peaks": 6,
    }
    peak_params = peak_params or default_peak_params

    # Compute PSD
    psd_results = epochs.compute_psd(
        method="welch",
        picks="misc",
        fmin=1,
        fmax=epochs.info["sfreq"] / 2,
        n_fft=int(epochs.info["sfreq"]),
        average="mean",
    )
    freqs = psd_results.freqs
    psds = psd_results.get_data(picks="misc")
    avg_psds = np.mean(psds, axis=0)

    # Fit spectral model
    fg = SpectralGroupModel(verbose=False, **peak_params)
    fg.fit(freqs, avg_psds, n_jobs=-1, freq_range=freq_range)
    exponents = fg.get_params("aperiodic_params", "exponent")
    fits = specparam2pandas(fg)
    return freqs, avg_psds, exponents, fits


def simulate_and_compute_itpc(
    labels: List[str],
    simulation_func: SimulationFunc,
    freqs: np.ndarray,
    n_cycles: int,
    n_seconds: int = 1,
    fs: int = 1000,
    n_epochs: int = 100,
    exponent: float = -2,
    filter_freq: Optional[float] = 125,
    resample_freq: Optional[int] = 250,
    **kwargs,
) -> ITPC:
    """
    Simulate data using the provided simulation function and compute ITPC.
    """
    # Create simulated epochs
    epochs = create_simulated_epochs(
        labels,
        simulation_func=simulation_func,
        n_seconds=n_seconds,
        fs=fs,
        n_epochs=n_epochs,
        exponent=exponent,
        filter_freq=filter_freq,
        resample_freq=resample_freq,
        **kwargs,
    )
    # Compute ITPC
    itpc = compute_itpc(epochs, freqs, n_cycles)
    return itpc


def plot_itpc_histogram(
    itpc: ITPC, tmin_avg: float = 0.3, tmax_avg: float = 0.7, bins: int = 30
) -> tuple:
    """
    Plot a histogram of averaged ITPC values with the mean value and return the averaged ITPC and plot object.
    """
    avg_itpc = average_itpc(itpc, tmin_avg=tmin_avg, tmax_avg=tmax_avg)
    mean_itpc = np.mean(avg_itpc)

    fig, ax = plt.subplots()
    ax.hist(avg_itpc, bins=bins, alpha=0.75, color="blue", edgecolor="black")
    ax.axvline(mean_itpc, color="red", linestyle="dashed", linewidth=1)
    ax.set_title("Histogram of Averaged ITPC Values")
    ax.set_xlabel("ITPC Value")
    ax.set_ylabel("Frequency")
    plt.show()

    return avg_itpc, fig


def plot_itpc_violinplot(
    itpc: ITPC, tmin_avg: float = 0.3, tmax_avg: float = 0.7
) -> tuple:
    """
    Plot a violin plot of averaged ITPC values showing the standard deviation and distribution of values.
    """
    avg_itpc = average_itpc(itpc, tmin_avg=tmin_avg, tmax_avg=tmax_avg)
    mean_itpc = np.mean(avg_itpc)
    std_itpc = np.std(avg_itpc)

    fig, ax = plt.subplots()
    sns.violinplot(data=avg_itpc, ax=ax, inner="point", scale="width")
    ax.axhline(mean_itpc, color="red", linestyle="dashed", linewidth=1, label=f'Mean: {mean_itpc:.2f}')
    ax.axhline(mean_itpc + std_itpc, color="green", linestyle="dashed", linewidth=1, label=f'Std Dev: {std_itpc:.2f}')
    ax.axhline(mean_itpc - std_itpc, color="green", linestyle="dashed", linewidth=1)
    ax.set_title("Violin Plot of Averaged ITPC Values")
    ax.set_xlabel("ITPC Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    return avg_itpc, fig
