"""
This module contains functions for spectral analysis of EEG data.
"""

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mne
from mne_bids import get_entities_from_fname


from specparam.plts.spectra import plot_spectra
from specparam import SpectralGroupModel
from specparam.core.funcs import infer_ap_func
from specparam.core.info import get_ap_indices

plt.rcParams["figure.figsize"] = [10, 6]  # Width, Height in inches

def compute_psd_fit_spectral_model(
    loaded_epochs: Dict[str, mne.BaseEpochs],
    fg: SpectralGroupModel,
    freq_range: List[float],
    n_fft: int | None = None,
) -> pd.DataFrame:
    """
    Process loaded MEG epochs by computing power spectral density (PSD),
    fitting a spectral group model, and creating a consolidated DataFrame
    with results and metadata.
    """
    dataframes_list: List[pd.DataFrame] = []

    for epoch_name, epoch_data in loaded_epochs.items():
        print(f"Processing {epoch_name}...")

        # Compute PSD and fit spectral model
        psd_data, spectra, freqs = process_epoch_data(epoch_data, n_fft)
        fg.fit(freqs, spectra, freq_range)

        # Create and populate DataFrame
        df = create_and_populate_dataframe(fg, epoch_data, epoch_name)
        dataframes_list.append(df)

    return pd.concat(dataframes_list, ignore_index=True)


def process_epoch_data(epoch_data: mne.BaseEpochs, n_fft: int | None = None) -> tuple:
    """
    Compute PSD and extract spectra and frequencies from epoch data.
    """
    fs = epoch_data.info["sfreq"]
    if n_fft is None:
        n_fft = int(2 * fs)

    welch_params = {
        "method": "welch",
        "n_overlap": int(fs / 2),
        "n_jobs": -1,
        "n_fft": n_fft,
        "picks": "misc",
        "average": "median",
    }
    psd = epoch_data.compute_psd(**welch_params).average()
    spectra, freqs = psd.get_data(picks="misc", return_freqs=True)

    return psd, spectra, freqs


def create_and_populate_dataframe(
    fg: SpectralGroupModel, epoch_data: mne.BaseEpochs, epoch_name: str
) -> pd.DataFrame:
    """
    Create a DataFrame with spectral model results and metadata.
    """
    df = specparam2pandas(fg)

    # Add channel information
    ch_df = pd.DataFrame(
        {
            "label": epoch_data.info["ch_names"],
            "ID": range(len(epoch_data.info["ch_names"])),
        }
    )
    df = pd.merge(df, ch_df, left_on="ID", right_on="ID", how="left")

    # Add metadata from filename
    entities = get_entities_from_fname(epoch_name, on_error="warn")
    for key in ["subject", "session", "task", "run"]:
        df[key] = entities.get(key)

    return df


def specparam2pandas(fg):
    """
    Converts a SpectralGroupModel object into a pandas DataFrame, with peak parameters and
    corresponding aperiodic fit information.

    Args:
    -----
    fg : specpramGroup
        The SpectralGroupModel object containing the fitting results.

    Returns:
    --------
    peaks_df : pandas.DataFrame
        A DataFrame with the peak parameters and corresponding aperiodic fit information.
        The columns are:
        - 'CF': center frequency of each peak
        - 'PW': power of each peak
        - 'BW': bandwidth of each peak
        - 'error': fitting error of the aperiodic component
        - 'r_squared': R-squared value of the aperiodic fit
        - 'exponent': exponent of the aperiodic component
        - 'offset': offset of the aperiodic component
        - 'knee': knee parameter of the aperiodic component [if is initially present in the fg object]
    Notes:
    ------
    This function creates two DataFrames. The first DataFrame `specparam_aperiodic`
    contains the aperiodic fit information and is based on the `aperiodic_params`
    attribute of the SpectralGroupModel object. The columns are inferred using the
    `get_ap_indices()` and `infer_ap_func()` functions from the specparam package.
    The second DataFrame `peak_df` contains the peak parameters and is based on the
    `peak_params` attribute of the SpectralGroupModel object. The column names are renamed
    to match the headers of `fooof_aperiodic`, and the 'ID' column is cast to integer.
    The two DataFrames are then merged based on a shared 'ID' column.
    """
    peaks_df = (
        pd.DataFrame(fg.get_params("peak_params"))  # prepare peaks dataframe
        .set_axis(["CF", "PW", "BW", "ID"], axis=1)  # rename cols
        .astype({"ID": int})
    )
    specparam_aperiodic = (
        pd.DataFrame(
            fg.get_params("aperiodic_params"),
            columns=get_ap_indices(
                infer_ap_func(np.transpose(fg.get_params("aperiodic_params")))
            ),
        )
        .assign(error=fg.get_params("error"), r_squared=fg.get_params("r_squared"))
        .reset_index(names=["ID"])
    )

    # Now, let's merge the dataframes
    return pd.merge(specparam_aperiodic, peaks_df, on="ID", how="left")


def examine_spectra(fg, subject):
    """Compare the power spectra between low and high exponent channels"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    def argmedian(arr):
        return np.argsort(arr)[len(arr) // 2]

    exps = fg.get_params("aperiodic_params", "exponent")
    r_squared = fg.get_params("r_squared")
    spectra_exp = [
        fg.get_model(np.argmin(exps)).power_spectrum,
        fg.get_model(argmedian(exps)).power_spectrum,
        fg.get_model(np.argmax(exps)).power_spectrum,
    ]

    labels_spectra_exp = [
        f"Low Exponent {format(np.min(exps), '.2f')}",
        f"Median Exponent {format(np.median(exps), '.2f')}",
        f"High Exponent {format(np.max(exps), '.2f')}",
    ]

    plot_spectra(
        fg.freqs,
        spectra_exp,
        ax=ax[0],
        labels=labels_spectra_exp,
    )

    spectra_r_squared = [
        fg.get_model(np.argmin(r_squared)).power_spectrum,
        fg.get_model(argmedian(r_squared)).power_spectrum,
        fg.get_model(np.argmax(r_squared)).power_spectrum,
    ]

    labels_spectra_r_squared = [
        f"Low R_squared  {format(np.min(r_squared), '.2f')}",
        f"Median R_squared {format(np.median(r_squared), '.2f')}",
        f"High R_squared {format(np.max(r_squared), '.2f')}",
    ]

    my_colors = ["blue", "green", "red"]
    plot_spectra(
        fg.freqs,
        spectra_r_squared,
        ax=ax[1],
        labels=labels_spectra_r_squared,
        colors=my_colors,
    )
    ylim1 = ax[0].get_ylim()
    ylim2 = ax[1].get_ylim()
    # Set the same limits on the y-axis for both plots
    ax[0].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    ax[1].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    fig.suptitle(
        f"sub-{subject} - Power spectra comparison between low, median and high exponent and R_squared values"
    )
    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig  # Return the figure object


def plot_spectra_models_generalized(fg, data, data_type="exps"):
    """
    Plots spectra models based on specified data type (experimental data or r_squared).

    Parameters:
    - fg: The FOOOFGroup object.
    - data: Array-like, data to determine models (experimental data or r_squared).
    - labels: Labels for each plotted spectra model.
    - data_type: Type of data to plot ('exps' for experimental data, 'r_squared' for r_squared values).
    """

    # Define a helper function for median
    def argmedian(data):
        return np.argsort(data)[len(data) // 2]

    # Select the appropriate data for model generation
    if data_type == "exps":
        indices = [np.argmin(data), argmedian(data), np.argmax(data)]
        labels = [
            f"Low {data_type} {format(np.min(data), '.2f')}",
            f"Median {data_type} {format(np.median(data), '.2f')}",
            f"High {data_type} {format(np.max(data), '.2f')}",
        ]
    elif data_type == "r_squared":
        indices = [np.argmin(data), argmedian(data), np.argmax(data)]
        labels = [
            f"Low R_squared {format(np.min(data), '.2f')}",
            f"Median R_squared {format(np.median(data), '.2f')}",
            f"High R_squared {format(np.max(data), '.2f')}",
        ]
    else:
        raise ValueError("data_type must be 'exps' or 'r_squared'")

    # Generate models based on the selected data
    spectra_models = [fg.get_model(idx, regenerate=True) for idx in indices]

    # Iterate over each model and its corresponding label
    for model, label in zip(spectra_models, labels):
        # Print results and plot extracted model fit
        model.print_results()
        model.plot()
        print(label)


def plot_models(fg, param_choice="exponent"):
    """
    Plot models from a FOOOF group object based on exponent or R-squared values.

    This function generates three plots (low, median, and high) for the specified
    parameter, prints the results for each model, and displays the corresponding label.

    Parameters:
    -----------
    fg : FOOOFGroup
        The FOOOF group object containing the models to plot.
    param_choice : str, optional
        The parameter to use for selecting models. Must be either 'exponent' or 'r_squared'.
        Default is 'exponent'.

    Raises:
    -------
    ValueError
        If param_choice is not 'exponent' or 'r_squared'.
    """
    if param_choice.lower() == "exponent":
        param = fg.get_params("aperiodic_params", "exponent")
        param_name = "Exponent"
    elif param_choice.lower() == "r_squared":
        param = fg.get_params("r_squared")
        param_name = "R-squared"
    else:
        raise ValueError("param_choice must be either 'exponent' or 'r_squared'")

    def argmedian(arr):
        return np.argsort(arr)[len(arr) // 2]

    labels_spectra = [
        f"Low {param_name} {format(np.min(param), '.2f')}",
        f"Median {param_name} {format(np.median(param), '.2f')}",
        f"High {param_name} {format(np.max(param), '.2f')}",
    ]

    spectra_models = [
        fg.get_model(np.argmin(param), regenerate=True),
        fg.get_model(argmedian(param), regenerate=True),
        fg.get_model(np.argmax(param), regenerate=True),
    ]

    for fm, label in zip(spectra_models, labels_spectra):
        fm.print_results()
        fm.plot()
        plt.title(label)
        plt.show()
        print(label)
