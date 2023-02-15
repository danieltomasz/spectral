
from fooof import FOOOFGroup,  FOOOF
import pandas as pd
import numpy as np
from neurodsp.spectral import compute_spectrum
from fooof import FOOOFGroup
import mne
from fooof.sim.gen import gen_freqs, gen_group_power_spectra
import functools
import inspect
from fooof.objs.utils import combine_fooofs

def generate_example_spectra():
    n_spectra = 2
    freq_range = [3, 40]
    ap_params = [[0.5, 1], [1, 1.5]]
    pe_params = [[10, 0.4, 1], [10, 0.2, 1, 22, 0.1, 3]]
    nlv = 0.02
    # Simulate a group of power spectra
    freqs, powers, sim_params = gen_group_power_spectra(n_spectra, freq_range, ap_params,
                                                    pe_params, nlv, return_params=True)
    return freqs, powers, sim_params
