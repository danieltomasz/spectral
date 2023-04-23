from __future__ import annotations

from pathlib import Path, PurePath
from glob import glob
import scipy.io as sio
import numpy as np
import re
import xarray as xr
import pandas as pd

import scipy.io as sio
import xarray as xr
import matplotlib.pyplot as plt

from poirot.translate import prepare_grouping

def load_mat_file(filepath: str, sub: list):
    mat_content = sio.loadmat(filepath)
    roi_names = [l.flatten()[0]
                for l in mat_content["RowNames"].flatten()]  # flatten

    freqs = np.ndarray.squeeze(mat_content["Freqs"])
    tf = mat_content["TF"]
    return xr.DataArray(
        tf,
        dims=["roi_names", "sub", "freqs"],
        coords={"roi_names": roi_names, "sub": sub, "freqs": freqs},
    )


def plot_psd_vs_freq_roi(
    psd, dim, rois: list | None = None, region: str | None = None, logfreq=True
):
    """
    Creates a plot of power spectral density (PSD) values for a given region and specified ROIs.

    Args:
        psd (Array): A multidimensional array containing PSD values, with dimensions (n_rois, n_epochs, n_freqs).
        rois (List[str]): A list of strings corresponding to the regions of interest (ROIs) for which the PSD will be plotted.
        region (str): A string indicating the overall region being plotted.

    Returns:
        fig (Figure): A Figure object representing the plot created by the function.
    """
    # Calculate the power for the specified ROIs
    if region:
        rois_idx = [list(psd[dim].values).index(coord) for coord in rois]
        power_rois = psd.values[rois_idx, 0, :]
    else:
        power_rois = psd.values[:, 0, :]
        rois = list(psd[dim].values)
    freqs = psd.freqs.values
    # Create the plot as an object
    fig, ax = plt.subplots()
    ax.plot(freqs, power_rois.T)
    ax.set_xlim([2, 48])
    ax.set_ylim(
        [np.min(power_rois), np.max(power_rois)]
    )  # set ylim to same range for all plots
    ax.set_yscale("log")
    if logfreq:
        ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"PSD vs Freq for {region} ROIs for {psd.sub.values[0]}")
    ax.legend(rois, loc="center left", bbox_to_anchor=(1, 0.5))
    # Return the plot as an object
    return fig
