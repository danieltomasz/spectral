import pandas as pd
import numpy as np
from fooof.sim.gen import gen_freqs, gen_group_power_spectra
from fooof.core.info import get_ap_indices, get_peak_indices
from fooof.core.funcs import infer_ap_func
from fooof import FOOOFGroup


def generate_example_spectra():
    n_spectra = 2
    freq_range = [1.5, 50]
    ap_params = [[0.5, 1], [1, 1.5]]
    pe_params = [[10, 0.4, 1], [10, 0.2, 1, 22, 0.1, 3]]
    nlv = 0.02
    # Simulate a group of power spectra
    [freqs, powers, sim_params] = gen_group_power_spectra(
        n_spectra, freq_range, ap_params, pe_params, nlv, return_params=True
    )
    # type: ignore
    return freqs, powers, sim_params


def psd_fooof(freqs, spectra, fg: object = None, freq_range=None):
    if fg is None:
        fg = FOOOFGroup(
            peak_width_limits=[2, 8],
            min_peak_height=0.1,
            max_n_peaks=6,
        )
    if fg is None:
        freq_range=[2, 48]
    fg.fit(freqs, spectra, freq_range=freq_range, n_jobs=-1, progress="tqdm")
    return fg


def fooof2pandas(fg):
    temp_df = pd.DataFrame(fg.get_params("aperiodic_params"),
                           columns=get_ap_indices(infer_ap_func(np.transpose(fg.get_params("aperiodic_params")))))
    temp_df["error"] = fg.get_params("error")
    temp_df["r_squared"] = fg.get_params("r_squared")
    temp_df.insert(0, "ID", temp_df.index)
    peaks = fg.get_params("peak_params")  # prepare peaks dataframe
    peaks_df = pd.DataFrame(peaks)
    peaks_df.columns = ["CF", "PW", "BW", "ID"]
    peaks_df["ID"] = peaks_df["ID"].astype(int)
    peaks_df = peaks_df.join(temp_df.set_index("ID"), on="ID")
    return peaks_df
