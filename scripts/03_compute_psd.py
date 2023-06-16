# %%
import argparse
import json
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


from fooof import FOOOFGroup
from fooof.core.info import get_ap_indices
from fooof.core.funcs import infer_ap_func
from fooof import FOOOFGroup

from neurodsp.spectral import compute_spectrum

from poirot.spectrum import process_spectrum

# %%

def save_trial(trial : Path , output_folder: str):
    """Save the trial to disk"""
    def compute_trial(trial):
        """Compute the PSD for a single trial"""
        ds = xr.open_dataarray(trial)
        time_dim = "time" if "time" in ds.dims else "time_padded"
        return ds.pipe(process_spectrum, fs=FS, nperseg = NPERSEG, method=METHOD,  time_dim = time_dim)
    
    ds = compute_trial(trial)
    subject = ds.sub.values[0]
    session = ds.session.values[0]
    condition = ds.condition.values[0] if "condition" in ds.coords else ""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(f"{output_folder}/{subject}_{session}_{condition}_PSD.nc")


def  compute_all_trials(input_folder_assr: str, output_folder: str):
    """Compute the PSD for all trial in input folder and save the result to output_folder"""
    trials_assr = list(Path(input_folder_assr).glob("*.nc"))
    for trial in tqdm(trials_assr):
        save_trial(trial , output_folder)

FS = 600
NPERSEG = 2*FS
METHOD = 'welch'

base_folder = "/Volumes/ExtremePro/Analyses/tDCS_MEG/"
input_folder_assr = f'{base_folder}/interim/timeseries/assr/padded/'
output_folder_assr = f'{base_folder}/interim/psd/assr/padded/'

input_folder_aspo = f'{base_folder}/interim/timeseries/aspo'
output_folder_aspo = f'{base_folder}/interim/psd/aspo'

if __name__ == "__main__":
    compute_all_trials(input_folder_aspo, output_folder_aspo)
    compute_all_trials(input_folder_assr, output_folder_assr)

#ds.pipe(process_spectrum, fs=FS, nperseg = NPERSEG, method=METHOD,  time_dim = "time_padded").isel(labels=25).isel(trial=14).plot(xlim=(2, 100))

