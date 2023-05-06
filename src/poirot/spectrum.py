import pandas as pd
import numpy as np

from fooof.core.info import get_ap_indices
from fooof.core.funcs import infer_ap_func
from fooof import FOOOFGroup

from neurodsp.spectral import compute_spectrum


def my_compute_spectrum(data, fs, nperseg):
    """
    Computes the power spectrum of a signal.

    Args:
        data (_type_): _description_
        fs (_type_): _description_
        nperseg (_type_): _description_

    Returns:
        _type_: _description_
    """
    sig , spectrum = compute_spectrum(data, fs=fs,nperseg=nperseg)
    return  spectrum

def psd_fooof(freqs, spectra, fg=None, freq_range=None):
    """
    Applies the FOOOF algorithm on power spectral density (PSD) data.

    Args:
        freqs (array): Frequency array.
        spectra (array): 2d array, shape: [n_power_spectra, n_freqs]
        fg (FOOOFGroup object, optional):
            FOOOFGroup object. Defaults to None.
        freq_range (array, optional):
            Frequency range. Defaults to None.

    Returns:
        FOOOFGroup object:
            Fitting results in the form of a FOOOFGroup object.
    """
    if fg is None:
        fg = FOOOFGroup(
            peak_width_limits=[2, 8],
            min_peak_height=0.1,
            max_n_peaks=6,
        )
    if freq_range is None:
        freq_range = [2, 48]
    fg.fit(freqs, spectra, freq_range=freq_range, n_jobs=-1, progress="tqdm")
    return fg

def specparam_xr(xs, stacked_cols, fg, freq_range):
    
    xs = xs.stack(point=stacked_cols).transpose("point", "freqs")
    # assumption stacked dim is called point
    freqs = xs.freqs.values
    spectra = xs.values
    fg.fit(freqs, spectra, freq_range=freq_range, n_jobs=-1, progress="tqdm")
    df = fooof2pandas(fg)
    return (
        pd.DataFrame.from_records(xs.point.values, columns=stacked_cols)
        .assign(ID=np.arange(len(xs.point.values)))
        .merge(df, on="ID")
        .drop(columns=["ID"])
    )
    
def specparam_attributes(xs, stacked_cols, fg, freq_range):
    
    xs = xs.stack(point=stacked_cols).transpose("point", "freqs")
    # assumption stacked dim is called point
    freqs = xs.freqs.values
    spectra = xs.values
    fg.fit(freqs, spectra, freq_range=freq_range, n_jobs=-1, progress="tqdm")
    df = fooof2pandas(fg)
    # get an list of additioanl columns
    l1 = ["freqs", "point"]
    l1.extend(stacked_cols)
    l2 = xs._coords.keys()
    list_of_columns = list(filter(lambda x: x not in l1, l2))
    return (
        pd.DataFrame.from_records(xs.point.values, columns=stacked_cols)
        .assign(ID=np.arange(len(xs.point.values)))
        .assign(**{column: xs[column].values for column in list_of_columns})
        .merge(df, on="ID")
        .drop(columns=["ID"])
    )


def fooof2pandas(fg):
    """
    Converts a FOOOFGroup object into a pandas DataFrame, with peak parameters and
    corresponding aperiodic fit information.

    Args:
    -----
    fg : FOOOFGroup
        The FOOOFGroup object containing the fitting results.

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
    This function creates two DataFrames. The first DataFrame `fooof_aperiodic`
    contains the aperiodic fit information and is based on the `aperiodic_params`
    attribute of the FOOOFGroup object. The columns are inferred using the
    `get_ap_indices()` and `infer_ap_func()` functions from the FOOOF package.
    The second DataFrame `peak_df` contains the peak parameters and is based on the
    `peak_params` attribute of the FOOOFGroup object. The column names are renamed
    to match the headers of `fooof_aperiodic`, and the 'ID' column is cast to integer.
    The two DataFrames are then merged based on a shared 'ID' column.
    """

    fooof_aperiodic = (
        pd.DataFrame(
            fg.get_params("aperiodic_params"),
            columns=get_ap_indices(
                infer_ap_func(np.transpose(fg.get_params("aperiodic_params")))
            ),
        )
        .assign(error=fg.get_params("error"), r_squared=fg.get_params("r_squared"))
        .reset_index(names=["ID"])
    )
    return (
        pd.DataFrame(fg.get_params("peak_params"))  # prepare peaks dataframe
        .set_axis(["CF", "PW", "BW", "ID"], axis=1)  # rename cols
        .astype({"ID": int})
        .join(fooof_aperiodic.set_index("ID"), on="ID")
    )

