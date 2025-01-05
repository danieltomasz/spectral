"""This is used to preprocess the ASSR part of the data"""

import os
import sys
import pathlib
import scipy.io as sio
import numpy as np
import mne

from .utils import suppress_stdout


def get_trials_in_session(raw_data_dir, pattern):
    """Get trial files satisfying specyfic pattern."""
    # if not (trials := list(pathlib.Path(raw_data_dir).glob(pattern))):
    #     raise FileNotFoundError(f"No trial files found with pattern: {pattern}")
    print(f"Searching for trials with pattern: {pattern}")
    try:
        trials = list(pathlib.Path(raw_data_dir).glob(pattern))
    except Exception as e:
        print(f"Error: {str(e)} with pattern: {pattern}")
        trials = []
    return trials


def create_raw(trial, times=[-2, 2, 2401], sfreq=600):
    """Create MNE Raw object from a MATLAB file."""

    def extract_from_mstruct(matlab_struct: dict, key: str) -> np.array:
        item_list = matlab_struct["Atlas"][0, 0]["Scouts"][key].ravel().tolist()
        return np.array([item.item() for item in item_list])

    times = np.linspace(*times)
    struct = sio.loadmat(trial)
    labels = extract_from_mstruct(struct, "Label")
    regions = extract_from_mstruct(struct, "Region")
    data = struct["Value"]
    ch_names = labels.tolist()
    ch_types = ["misc"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)


def create_epochs(trials):
    raw_trials = []
    with suppress_stdout():
        for trial in trials:
            raw = create_raw(trial)
            raw_trials.append(raw)
    datas = [r.get_data() for r in raw_trials]

    info = raw_trials[0].info
    return mne.EpochsArray(datas, info, tmin=-2.0)
