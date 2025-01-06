import matplotlib.pyplot as plt
import sys
from mne_bids import get_entities_from_fname

from .preprocess_assr import apply_window_and_pad


def plot_psd_pre_post(epochs, metadata):
    """
    Computes and plots the Power Spectral Density (PSD) of EEG data before and after a specific event.

    Parameters:
    epochs (mne.Epochs): The EEG data segmented into epochs.
    metadata (dict): A dictionary containing metadata about the subject and session.
                    It should have keys 'sub' and 'ses' corresponding to subject ID and session ID.

    The function creates two subplots: one for the PSD before the event (from -1 to 0 seconds)
    and one for the PSD after the event (from 0 to 1 seconds). The PSD is computed using the
    Welch method on the 'misc' channels, and the frequency range of interest is from 3.0 to 45 Hz.

    The title of the figure includes the subject ID and session ID extracted from the metadata.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"This is extracted values for sub-{metadata['subject']}_ses-{metadata['session']}",
        fontsize=16,
    )
    epochs.compute_psd(
        method="welch", picks="misc", tmin=-1, tmax=0, fmin=3.0, fmax=45
    ).plot(axes=axs[0], picks="misc")
    epochs.compute_psd(
        method="welch", picks="misc", tmin=0, tmax=1, fmin=3.0, fmax=45
    ).plot(axes=axs[1], picks="misc")
    return fig


def plot_psd_padded_pre_post(epochs, metadata):
    """Plot padded PSD for pre and post stimulus."""
    times = [(-1, 0), (0, 1)]
    conditions = ["prestim", "stim"]

    fig, axs = plt.subplots(2, 2, figsize=(24, 9))
    axs = axs.flatten()
    fig.suptitle(
        f"This is extracted values for sub-{metadata['subject']}_ses-{metadata['session']}_run-{metadata['run']}",
        fontsize=16,
    )
    for i, (time, condition) in enumerate(zip(times, conditions)):
        print(2 * i)
        epochs_temp = epochs.copy().crop(tmin=time[0], tmax=time[1])
        epochs_padded = apply_window_and_pad(
            epochs_temp,
            desired_length_sec=2,
            window_type="identity",
            use_reverse_padding=False,
        )
        epochs_temp.compute_psd(method="welch", picks="misc", fmin=3.0, fmax=45).plot(
            axes=axs[2 * i], picks="misc"
        )
        epochs_padded.compute_psd(method="welch", picks="misc", fmin=3.0, fmax=45).plot(
            axes=axs[2 * i + 1], picks="misc"
        )
