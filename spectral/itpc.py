"""Inter-Trial Phase Consistency (ITPC) analysis."""

from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
from mne_bids import get_entities_from_fname


class ITPC(NamedTuple):
    """Class to store Inter-Trial Phase Consistency (ITPC) data."""

    data: np.ndarray
    times: np.ndarray


def compute_itpc(epochs, freqs, n_cycles) -> ITPC:
    """Compute Inter-Trial Phase Coherence (ITC) using Morlet wavelets."""
    _, itc = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        average=True,
        n_cycles=n_cycles,
        return_itc=True,
        picks="misc",
        n_jobs=-1,
    )

    # Check if the second dimension (frequency) has more than one element
    if itc.data.shape[1] > 1:
        # Average along the frequency dimension
        itc_data = itc.data.mean(axis=1)
    else:
        # Squeeze out the frequency dimension if there's only one frequency
        itc_data = np.squeeze(itc.data, axis=1)

    return ITPC(data=itc_data, times=itc.times)


def extract_itpc_values(itpc: ITPC, tmin: float = -0.5, tmax: float = 1.0) -> ITPC:
    """Extract ITPC values for the entire epoch."""
    mask = (itpc.times >= tmin) & (itpc.times <= tmax)
    return ITPC(data=itpc.data[:, mask], times=itpc.times[mask])


def z_normalize_itpc(
    itpc: ITPC, tmin_baseline: float = -0.5, tmax_baseline: float = -0.2
) -> ITPC:
    """Z-normalize ITPC data based on the baseline period."""
    baseline_mask = (itpc.times >= tmin_baseline) & (itpc.times <= tmax_baseline)
    baseline_mean = itpc.data[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_std = itpc.data[:, baseline_mask].std(axis=1, keepdims=True)
    z_normalized_data = (itpc.data - baseline_mean) / baseline_std
    return ITPC(data=z_normalized_data, times=itpc.times)


def average_itpc(itpc: ITPC, tmin_avg=0.3, tmax_avg=0.7):
    """Average z-normalized ITPC over the specified time window for each channel."""
    avg_mask = (itpc.times >= tmin_avg) & (itpc.times <= tmax_avg)
    return itpc.data[:, avg_mask].mean(
        axis=1
    )  # Average across the time window for each channel


def create_itpc_dataframe(value, epochs, type, metadata):
    """Create a DataFrame with averaged z-normalized ITPC values and metadata."""
    return pd.DataFrame(
        {
            "ch_names": epochs.info["ch_names"],
            "value": value,
            "type": type,
            # print(entities)
            "subject": metadata["subject"],
            "session": metadata["session"],
            "task": metadata["task"],
            "run": metadata["run"],
        }
    )


def process_single_file(file_path, freqs, n_cycles):
    """Process a single file to compute z-normalized and averaged ITPC and create a DataFrame."""
    epochs = mne.read_epochs(file_path)
    metadata = get_entities_from_fname(file_path, on_error="warn")

    itpc = compute_itpc(epochs, freqs, n_cycles)
    itpc_extracted = extract_itpc_values(itpc)

    # Z-normalize ITPC and average over the specified time window
    z_normalized_itpc = z_normalize_itpc(
        itpc_extracted, tmin_baseline=-0.5, tmax_baseline=-0.2
    )
    avg_z_itpc = average_itpc(z_normalized_itpc, tmin_avg=0.3, tmax_avg=0.7)
    df_znorm = create_itpc_dataframe(
        value=avg_z_itpc, epochs=epochs, type="z_normalized_itpc", metadata=metadata
    )

    # Compute ITPC and average over the prestimuli time window
    prestim_itpc = average_itpc(itpc_extracted, tmin_avg=-0.5, tmax_avg=-0.2)
    df_pre = create_itpc_dataframe(
        value=prestim_itpc, epochs=epochs, type="prestim_itpc", metadata=metadata
    )

    # Compute ITPC and average over the prestimuli time window
    prestim_itpc = average_itpc(itpc_extracted, tmin_avg=0.3, tmax_avg=0.7)
    df_stim = create_itpc_dataframe(
        value=prestim_itpc, epochs=epochs, type="stim_itpc", metadata=metadata
    )
    return pd.concat([df_znorm, df_pre, df_stim])


def plot_itpc(
    itpc,
    vmin=None,
    vmax=None,
    title="Inter-Trial Phase Consistency (ITPC)",
    label="ITPC",
):
    """
    Plot the Inter-Trial Phase Consistency (ITPC) data.

    Parameters:
    itpc : object
        The ITPC object containing the data to be plotted.
    vmin : float, optional
        The minimum value for the color scale. If None, it will be set to the minimum value of the data.
    vmax : float, optional
        The maximum value for the color scale. If None, it will be set to the maximum value of the data.
    """
    # Set vmin and vmax to data-driven values if not provided
    if vmin is None:
        vmin = itpc.data.min()
    if vmax is None:
        vmax = itpc.data.max()

    # Plot the ITPC data
    fig, ax = plt.subplots()
    im = ax.imshow(
        itpc.data,
        aspect="auto",
        origin="lower",
        extent=(itpc.times[0], itpc.times[-1], 0, itpc.data.shape[0]),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    fig.colorbar(im, ax=ax, label=label)
    plt.show()
