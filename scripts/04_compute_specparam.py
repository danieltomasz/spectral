# %%
import xarray as xr
import pandas as pd
import numpy as np
import tqdm as tqdm
from pathlib import Path


import datetime
import json

from fooof import FOOOF, FOOOFGroup

from poirot.spectrum import specparam_attributes


# %%
def get_specparam_dataframe(folder :str):
    fg = FOOOFGroup(**spec_params)
    spectra = list(Path(folder).glob("*.nc"))
    df_list = []
    stacked_cols = ['labels', 'sub', 'session']
    for spectrum in spectra:
        ds = xr.open_dataarray(spectra)
        ds.close()


# def get_specpraram_single(ds : xr.DataArray):
    

# for subject in SUBS:
#     ds = xr.open_dataarray(
#         f"{DATA_FOLDER }/interim/timeseries/{subject}_MEG_ASSR_times.nc")
#     ds.close()
#     df = (
#         process_spectrum(ds, fs = FS, nperseg = NPERSEG, method=METHOD)
#         .stack(trial_iter_number=('trial', 'iter_number'))
#         .mean("trial_iter_number")
#         .pipe(
#             specparam_attributes,
#             stacked_cols=stacked_cols,
#             fg=fg,
#             freq_range=FREQ_RANGE,
#         )
#     )
#     df_list.append(df)
# df_concat = pd.concat(df_list)


# %%

# PSD Computation
FS = 600
NPERSEG = 2*FS
METHOD = 'medfilt'

# SPECPARAMS
FREQ_RANGE = [2, 45]

spec_params = {
    'peak_width_limits': [2, 10.0],
    'max_n_peaks': 6,
    'min_peak_height': 0.1,
    'peak_threshold': 2.0,
    'aperiodic_mode': 'fixed'
}

base_folder = "/Volumes/ExtremePro/Analyses/tDCS_MEG/"
psd_folder_aspo = f'{base_folder}/interim/psd/aspo'
specparam_folder_aspo = f'{base_folder}/interim/specparam/aspo'


# %%
fg = FOOOFGroup(**spec_params)
spectras = list(Path(psd_folder_aspo).glob("*.nc"))
spectrum = spectras[0]
stacked_cols = ['labels', 'sub', 'session', "iter_number"]
# ds = xr.open_dataarray(spectrum).mean("trial")
# xs = ds.stack(point=stacked_cols).transpose("point", "freqs")
# freqs = xs.freqs.values
# spectra = xs.values
# fg.fit(freqs, spectra, freq_range=FREQ_RANGE, n_jobs=-1, progress="tqdm")
da = (xr.open_dataarray(spectrum)
    .mean("trial")
    .pipe(
        specparam_attributes,
        stacked_cols=stacked_cols,
        fg=fg,
        freq_range=FREQ_RANGE,
    )
      )
# %%
