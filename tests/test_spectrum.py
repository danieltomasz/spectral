import pytest
import pandas as pd
from lambic.spectrum import generate_example_spectra, psd_fooof, fooof2pandas

def fooof2pandas_shape() ->  tuple:
    freqs, spectra, sim_params = generate_example_spectra()
    fg = psd_fooof(freqs, spectra)
    df = fooof2pandas(fg)
    return df.shape

def test_fooof2pandas():
    assert fooof2pandas_shape() == (3, 8)

#%%
