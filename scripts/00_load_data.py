"""
Copy the matlab data to the data folder and run this script to process it.
"""

import re
import argparse
from typing import List, Union
import numpy as np
import xarray as xr
import scipy.io as sio
from tqdm import tqdm
from pathlib import Path

def load_matlab_file(trial: Path) -> dict:
    """
    Load a matlab file and return a dictionary
    """
    return sio.loadmat(trial)

def process_trial(trial: Path, time: np.ndarray,  sub: str, session: str) -> xr.DataArray:
    """
    Load a single trial of MEG data for a single subject.
    """
    trial_number = re.findall(r"\d+", str(trial))[-1]
    iter_number = re.findall(r"\d+", str(trial))[-2]

    mstruct = load_matlab_file(trial)
    values = mstruct["Value"]

    label_struct = mstruct["Atlas"][0, 0]["Scouts"]["Label"].ravel().tolist()
    labels = np.array([item.item() for item in label_struct])

    region_struct = mstruct["Atlas"][0, 0]["Scouts"]["Region"].ravel().tolist()
    regions = np.array([item.item() for item in region_struct])
    return (
        xr.DataArray(
            values.T,
            dims=["time", "labels"],
            coords={
                "labels": labels,
                "time": time,
                "sub": sub,
                "session": session,
                "iter_number": int(iter_number),
                "trial": int(trial_number)
            })
        .expand_dims('trial')
        .expand_dims("iter_number")
        .expand_dims("sub")
        .expand_dims("session")
        .assign_coords(regions=("labels", regions))
    )


def process_sub_session( data_folder: str, sub: str, time: np.ndarray, session: str) -> xr.DataArray:
    """
    Process all trials for a given subject and return a xarray DataArrays"""
    list_of_trials = []
    pattern = f"*{sub}_{session}*.mat"
    trials = Path(data_folder).glob(pattern)
    for trial in trials:
        assert trial.exists(), f"File {trial} does not exist"
        single_trial = process_trial(trial, time, sub, session)
        list_of_trials.append(single_trial.compute())
    return xr.combine_by_coords(list_of_trials )

def process_subjects(data_folder :str, subjects: np.ndarray, time : np.ndarray, sessions: List[str], output_folder :str):
    """
    Process all subjects and sessions
    """
    for sub in tqdm(subjects):
        for single_session in sessions:
            print(f"Processing {sub} {single_session}")
            data_subject_session =  process_sub_session(data_folder, sub, time, single_session)
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            data_subject_session.to_netcdf(f"{output_folder}/{sub}_{single_session}_times.nc")


base_folder = "/Volumes/ExtremePro/Analyses/tDCS_MEG/"

time_assr = time = np.linspace(-2, 2, 2401)
sessions_assr = ["A", "B"]
data_folder_assr = f'{base_folder}/raw/brainstorm/ASSR/'
output_folder_assr = f'{base_folder}/interim/timeseries/assr/'
subjects_assr = [f"Subject{str(i).zfill(2)}" for i in range(1, 16)]

time_aspo= np.linspace(0, 5, 3000)
sessions_aspo = ["Session1", "Session2"]
data_folder_aspo = f'{base_folder}/raw/brainstorm/ASPO/'
output_folder_aspo = f'{base_folder}/interim/timeseries/aspo/'
subjects_aspo = [f"S{str(i).zfill(3)}" for i in range(1, 16)]

if __name__ == "__main__":
    process_subjects(data_folder_assr, subjects_assr, time_assr, sessions_assr, output_folder_assr)
    process_subjects(data_folder_aspo, subjects_aspo, time_aspo, sessions_aspo, output_folder_aspo)