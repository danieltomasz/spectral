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

def get_single_specparam_estimation(run: Path, spec_params :dict, stacked_cols : list):
    "Compute the specparam for a single run and return a dataframe"
    fg = FOOOFGroup(**spec_params)
    return (xr.open_dataarray(run)
    .mean("trial")
    .pipe(
        specparam_attributes,
        stacked_cols=stacked_cols,
        fg=fg,
        freq_range=FREQ_RANGE,
    )
      )

def get_all_specparam_estimation(folder: str, spec_params :dict, stacked_cols : list):
    """Return a dataframe with all specparam estimation for a given folder"""
    list_of_dataframes = [] 
    runs = list(Path(folder).glob("*.nc"))
    for run in tqdm.tqdm(runs):
        ds =  get_single_specparam_estimation(run, spec_params, stacked_cols)
        list_of_dataframes.append(ds)
    return pd.concat(list_of_dataframes)

# %%

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



specparam_folder = f'{base_folder}/interim/specparam/'

dataset = "assr"

if __name__ == '__main__':
    match dataset:
        case "aspo":
            psd_folder_aspo = f'{base_folder}/interim/psd/aspo'
            stacked_cols = ['labels', 'sub', 'session', "iter_number"]
            da = get_all_specparam_estimation(psd_folder_aspo, spec_params, stacked_cols)
            da.to_csv(f"{specparam_folder}/specparam_ASPO.csv")
        case "assr":
            psd_folder_assr = f'{base_folder}/interim/psd/assr/padded'
            #  add condition to the stacked cols
            stacked_cols = ['labels', 'sub', 'session', "iter_number", "condition"]
            da = get_all_specparam_estimation(psd_folder_assr, spec_params, stacked_cols)
            da.to_csv(f"{specparam_folder}/specparam_ASSR.csv")


# %%
