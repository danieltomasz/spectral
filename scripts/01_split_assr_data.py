# %%
import pandas as pd
import xarray as xr
from fooof import FOOOF, FOOOFGroup
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from poirot.spectrum import specparam_attributes, process_spectrum
from poirot.specparam import create_specparam_object, plot_fit


def split_pad_time(ds, fs, time_range, condition='STIM', pad_ratio=0.5 ):
    """Split the time series in trials and pad them with zeros"""
    zscore = lambda x: (x - x.mean()) / x.std()
    padding = lambda x: np.pad(x, pad_width=int(pad_ratio*fs))
    return (
        ds
        .pipe(lambda x: xr.apply_ufunc( 
        zscore, x,
        vectorize=True,
        input_core_dims=[["time"]],
        output_core_dims=[['time']]
        ))
        .sel(time=slice(*time_range))
        .pipe(lambda x: xr.apply_ufunc( 
                padding, x,
                vectorize=True,
                input_core_dims=[["time"]],
                output_core_dims=[['time_padded']]
        ))
        .assign_coords(condition=condition)
        .expand_dims('condition')
    )

def preview_time_series(ds):
    """Preview the time series"""
    fig, ax = plt.subplots()
    (ds
     .squeeze()
     .isel(trial=1)
     .isel(labels=0)
     .isel(iter_number=0)
     .plot.line(x="time_padded", ax=ax)
    )
    ax.set_title("Time series")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig, ax



def save_trial(trial : Path , output_folder, conditions: list[dict]):
    """Save the trial to disk"""
    
    def create_conditions(data_array, fs, time_range, condition, pad_ratio=0.5 ):
        """Create the conditions and save them to disk"""
        Path(output_folder_assr).mkdir(parents=True, exist_ok=True)
        return split_pad_time(data_array, fs, time_range ,condition, pad_ratio)

    ds = xr.open_dataarray(trial)
    stem = trial.stem
    for cond in conditions:
        ds_cond = create_conditions(ds, fs, cond["time_range"], cond["condition"], pad_ratio=0.5 )
        ds_cond.to_netcdf(f"{output_folder}/{cond['condition']}_{stem}.nc")
    
# %%

fs = 600

base_folder = "/Volumes/ExtremePro/Analyses/tDCS_MEG/"
input_folder_assr = f'{base_folder}/interim/timeseries/assr/'

output_folder_assr = f'{base_folder}/interim/timeseries/assr/padded/'

STIM = {
    'time_range': (0, 1),
    'condition': 'STIM'}
PRE ={
    'time_range': (-1, 0),
    'condition': 'PRE'
}
conditions = [STIM, PRE]



if __name__ == "__main__":
    trials = list(Path(input_folder_assr).glob("*.nc"))
    for trial in trials:
        save_trial(trial, output_folder_assr, conditions)

# %%
# %%
