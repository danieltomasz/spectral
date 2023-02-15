import pandas as pd
import numpy as np

from fooof import Bands
from fooof.core.funcs import infer_ap_func
from fooof.core.info import get_ap_indices, get_peak_indices
from fooof.core.modutils import safe_import, check_dependency
from fooof.analysis.periodic import get_band_peak

def fooof2pandas(input_func):
    def wrapper(*args, **kwargs):

        fg = input_func(*args, **kwargs)
        print('Number of model fits: ', len(fg))
        temp_df = pd.DataFrame() # prepare error and slope dataframe
        #temp_df['knee'] = fg.get_params('aperiodic_params', 'knee')
        temp_df['offset'] = fg.get_params('aperiodic_params', 'offset')
        temp_df['exponent'] = fg.get_params('aperiodic_params', 'exponent')

        temp_df['errors']= fg.get_params('error')
        temp_df['r2s']=fg.get_params('r_squared')
        temp_df.insert(0, 'ID', temp_df.index)

        peaks = fg.get_params('peak_params') # prepare peaks dataframe
        peaks_df = pd.DataFrame(peaks)
        peaks_df.columns = ['CF', 'PW', 'BW', 'ID']
        peaks_df['ID'] = peaks_df['ID'].astype(int)
        peaks_df = peaks_df.join(temp_df.set_index('ID'), on='ID')
        return peaks_df
    return wrapper



def psd_fooof(freqs, spectra):
    fg = FOOOFGroup(peak_width_limits=[2.5, 8],min_peak_height=0.05, max_n_peaks=6)
    fg.fit(freqs, spectra, freq_range=[3, 48],n_jobs=-1, progress='tqdm')
    return fg

def model_to_dict(fit_results):
    """Convert model fit results to a dictionary.
    Parameters
    ----------
    fit_results : FOOOFResults
        Results of a model fit.
    Returns
    -------
    dict
        Model results organized into a dictionary.
    """

    fr_dict = dict(
        zip(
            get_ap_indices(infer_ap_func(fit_results.aperiodic_params)),
            fit_results.aperiodic_params,
        )
    )
    # periodic parameters
    peaks = fit_results.peak_params

    for ind, peak in enumerate(peaks):
        for pe_label, pe_param in zip(get_peak_indices(), peak):
            fr_dict[f'{pe_label.lower()}_{str(ind)}'] = pe_param

    # goodness-of-fit metrics
    fr_dict['error'] = fit_results.error
    fr_dict['r_squared'] = fit_results.r_squared

    return fr_dict


def model_to_dataframe(fit_results):
    """Convert model fit results to a dataframe.
    Parameters
    ----------
    fit_results : FOOOFResults
        Results of a model fit.
    peak_org : int or Bands
        How to organize peaks.
        If int, extracts the first n peaks.
        If Bands, extracts peaks based on band definitions.
    Returns
    -------
    pd.Series
        Model results organized into a dataframe.
    """

    return pd.Series(model_to_dict(fit_results))

def group_to_dataframe(fit_results, peak_org):
    """Convert a group of model fit results into a dataframe.
    Parameters
    ----------
    fit_results : list of FOOOFResults
        List of FOOOFResults objects.
    peak_org : int or Bands
        How to organize peaks.
        If int, extracts the first n peaks.
        If Bands, extracts peaks based on band definitions.
    Returns
    -------
    pd.DataFrame
        Model results organized into a dataframe.
    """

    return pd.DataFrame([model_to_dataframe(f_res, peak_org) for f_res in fit_results])
