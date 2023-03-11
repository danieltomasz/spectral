import pytest
import pandas as pd
from fooof import FOOOFGroup

from lambic.spectrum import psd_fooof, fooof2pandas
from lambic.utils import generate_example_spectra

def fooof2pandas_shape() ->  tuple:
    freqs, spectra, sim_params = generate_example_spectra()
    fparams =  FOOOFGroup(
        peak_width_limits=[2, 12],
        min_peak_height=0.1,
        max_n_peaks=6,
        aperiodic_mode="fixed",
    )
    fg = psd_fooof(freqs, spectra, fparams)
    df = fooof2pandas(fg)
    return df.shape

def test_fooof2pandas():
    assert fooof2pandas_shape() == (3, 8)

#%%
