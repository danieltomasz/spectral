"""This is used to preprocess the ASSR part of the data"""

import os
import sys
import pathlib
import scipy.io as sio
import numpy as np
import mne
from typing import List, Dict
import re
import os
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import scipy.io as sio
from tqdm import tqdm
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



def get_trials(sub: str, session: str, data_folder: str) -> List[Path]:
    """Get trial files for a given subject and session."""
    return [Path(p) for p in glob(f"{data_folder}/ASP0_{sub}_{session}_*.mat")]


def sort_trials(file_paths: List[Path]) -> Dict[str, List[Path]]:
    """Sort trial files by iteration number."""
    iter_files = defaultdict(list)
    for file in file_paths:
        if match := re.search(r"(\d+)_clean_resample", file.name):
            iter_number = match[1]
            iter_files[iter_number].append(file)
    return dict(iter_files)


def load_matlab_file(trial: Path) -> dict:
    """Load a MATLAB file and return its contents."""
    return sio.loadmat(trial)


def build_mne_epochs_from_matlab_files(file_paths: List[Path]):
    """Build MNE Epochs object from multiple MATLAB files."""
    all_data = []
    ch_names = None
    sfreq = 600.0  # Assuming this is constant for all files

    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            mat_data = load_matlab_file(file_path)
            data = mat_data["Value"]  # Shape should be (68, 3000)
            all_data.append(data)

            if ch_names is None:
                region_struct = (
                    mat_data["Atlas"][0, 0]["Scouts"]["Label"].ravel().tolist()
                )
                ch_names = [item.item() for item in region_struct]

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_data:
        raise ValueError("No valid epochs were loaded.")

    # Stack all data into a 3D array (n_epochs, n_channels, n_times)
    all_data = np.stack(all_data, axis=0)

    # Create MNE info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")

    # Create events array
    events = np.column_stack(
        [
            np.arange(len(all_data)) * 3000,
            np.zeros(len(all_data), dtype=int),
            np.ones(len(all_data), dtype=int),
        ]
    )

    # Create MNE Epochs object
    return mne.EpochsArray(all_data, info, events, tmin=0)


def process_and_save_meg_epochs_by_session(
    sub: str, sessions: List[str], data_folder: str, output_folder: str
):
    """
    This function performs the following steps for each session:
    1. Retrieves trial files for the subject and session
    2. Sorts the trials
    3. For each iteration (run):
        a. Builds MNE epoch objects from MATLAB files
        b. Saves the epoch objects in BIDS-compliant .fif format

    Parameters:
    - sub (str): Subject identifier
    - sessions (List[str]): List of session identifiers
    - data_folder (str): Path to the folder containing raw data
    - output_folder (str): Path to the folder where processed data will be saved
    """
    for session in sessions:
        print(f"\nProcessing {sub} - {session}")
        files = get_trials(sub, session, data_folder)
        sorted_dict = sort_trials(files)

        for iter_number, file_list in sorted_dict.items():
            print(f"\nProcessing iteration {iter_number}")
            epochs = build_mne_epochs_from_matlab_files(file_list)
            print(f"Created Epochs object: {epochs}")

            # Save the Epochs object
            output_file = os.path.join(
                output_folder,
                f"sub-{sub}_ses-{session}_task-rest_run-{iter_number}-epo.fif",
            )
            epochs.save(output_file, overwrite=True)
            print(f"Saved Epochs to: {output_file}")

    print("\nProcessing complete.")


def load_subject_meg_epochs_to_dict(
    data_folder: str, sub: str = "S001", pattern: str = "sub-{}_*-epo.fif"
) -> Dict[str, mne.BaseEpochs]:
    """
    Load MEG Epochs files for a given subject into a flat dictionary.

    Parameters:
    - data_folder (str): Path to the folder containing the -epo.fif files
    - subject (str): Subject identifier (default is "S001")

    Returns:
    - Dict[str, mne.Epochs]: Dictionary with filenames (without extension) as keys and loaded Epochs as values
    """
    loaded_data = {}

    # Create the glob pattern to match files for this subject
    file_pattern = os.path.join(data_folder, pattern.format(sub))

    # Find all matching files
    epoch_files = sorted(glob(file_pattern))

    if not epoch_files:
        print(f"No files found for {sub}")
        return loaded_data

    # Load each file and store in the dictionary
    for file in epoch_files:
        try:
            epochs = mne.read_epochs(file, preload=True)
            # Get the filename without path and extension
            key = os.path.splitext(os.path.basename(file))[0]
            loaded_data[key] = epochs
            print(f"Loaded: {key}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return loaded_data
