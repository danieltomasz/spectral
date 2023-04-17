from pathlib import Path
import fooof
import shutil
from pandas.core.frame import DataFrame
import scipy.io  as sio
import numpy as np
import pandas as pd
from fooof import FOOOFGroup,  FOOOF

def process_single_recording_session(path: str) -> DataFrame:
    """[summary]

    Args:
        path (str): [description]

    Returns:
        DataFrame: [description]
    """

    try:    
        data_dict =  mat2dict_scout_psd(path)
        freqs_data  = data_dict["freqs"] 
        powers_data = data_dict["powers"]
        #plot_spectra(freqs_data, powers_data, log_powers=True)
        peaks_df = psd_fooof(freqs_data, powers_data)
        peaks_df = add_columns(peaks_df, data_dict)
        return peaks_df
    except OSError as err:
        print("OS error: {0}".format(err))


def mat2dict_scout_psd(path: str) -> dict:
    """Load content of matfile containg PSD of scouts into Python dict

    Args:
        mat_content (str): Path to file, may be a string or pathlike object

    Returns:
        dict: Dictionary containing all the information  extracted from mat file
    """

    def get_real_names(row_names):
        row_names=  [l.flatten()[0] for l in row_names.flatten()] #flatten
        row_names = [i.split('@')[0] for i in row_names] #delete location
        row_names = [x.strip(' ') for x in row_names]
        row_names = [x.replace(' ', '_') for x in row_names] #remove 
        return row_names

    def split(strng, sep, pos):
        strng = strng.split(sep)
        return sep.join(strng[:pos]), sep.join(strng[pos:])

    #print(sorted(mat_content.keys()))
    #print(mat_content["TF"])
    mat_content = sio.loadmat(path)
    tf = np.ndarray.squeeze(mat_content["TF"])
    row_names = mat_content["RowNames"]
    row_names = get_real_names(row_names)
    freqs = np.ndarray.squeeze(mat_content["Freqs"])
    comment = mat_content["Comment"].tolist()[0].split('|')[1]
    subject_cond = split(comment, "_", 2)[0]
    resample = split(split(comment, "_", 5)[1],"_",2)[0]
    condition = comment.split('_Destrieux_')[1]
    data_dict = {'freqs' : freqs, 'powers' : tf, 'row_names' : row_names, 'subject_cond' : subject_cond, 'resample' : resample, 'condition': condition}
    if len(tf)==0:
        print ("TF is empty")
    else:
        return  data_dict  