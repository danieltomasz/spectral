import mne
import numpy as np
from scipy.signal.windows import gaussian, tukey
from scipy.stats import zscore


def normalize_epochs_per_channel(epochs):
    """
    Applies z-score normalization per channel for each epoch.
    """
    normalized_data = []
    for epoch in epochs.get_data(copy=True):
        # Z-score normalization for each channel in the epoch
        normalized_epoch = zscore(epoch, axis=1)
        normalized_data.append(normalized_epoch)

    # Creating a new EpochsArray with normalized data
    return mne.EpochsArray(normalized_data, epochs.info, tmin=epochs.tmin)


def apply_window_and_pad(
    epochs,
    desired_length_sec,
    window_type="hamming",
    use_reverse_padding=False,
    alpha=0.1,
):
    """Pad epochs with a window function and zero-padding."""
    sfreq = epochs.info["sfreq"]

    # Check to ensure desired length is greater than current length
    if (desired_length_samples := int(np.round(desired_length_sec * sfreq))) <= (
        current_length_samples := epochs.get_data(copy=False).shape[2]
    ):
        raise ValueError(
            "Desired length must be greater than the current length of the epochs."
        )

    new_data = []
    for epoch in epochs.get_data(copy=True):
        # Apply window
        if window_type == "hamming":
            window = np.hamming(current_length_samples)
        elif window_type == "gaussian":
            std = current_length_samples / 10  # Smaller std for a narrower window
            window = gaussian(current_length_samples, std=std)
        elif window_type == "tukey":
            window = tukey(current_length_samples, alpha=alpha)
        else:
            window = np.ones(current_length_samples)

        windowed_epoch = epoch * window[None, :]
        # Padding
        if use_reverse_padding:
            reverse_data = np.flip(windowed_epoch, axis=1)
            padding_length = (desired_length_samples - current_length_samples) // 2
            padded_epoch = np.hstack(
                (
                    reverse_data[:, :padding_length],
                    epoch,
                    reverse_data[:, :padding_length],
                )
            )
        else:
            padding_length = (desired_length_samples - epoch.shape[1]) // 2
            padded_epoch = np.pad(
                epoch, ((0, 0), (padding_length, padding_length)), "constant"
            )
        new_data.append(padded_epoch)

    return mne.EpochsArray(new_data, epochs.info, tmin=epochs.tmin)
