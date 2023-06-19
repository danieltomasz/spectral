# %%
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io  as sio

base_folder = "/Volumes/ExtremePro/Analyses/tDCS_MEG/"
metadata_ASSR = f"{base_folder}/metadata/keys.mat"
metadata_ASOP = f"{base_folder}/metadata/SessionAssignment.csv"
specparam_folder = f'{base_folder}/interim/specparam/'

# %% indentify condition in the ASSR dataset
def load_conditions_coding(keys_path: str) -> pd.DataFrame:
    """ Load coding telling us what conditions was representing specific recording (Sham or real TDCS)"""
    keys = sio.loadmat(keys_path)
    row_names = keys["keys"].tolist()[0]
    temp_df = pd.DataFrame()   
    temp_df["subject_cond"]  = [l[0].flatten()[0] for l in row_names]
    temp_df["subject_cond"] = temp_df['subject_cond'].astype('category').str.strip()
    temp_df["T"] =  [l[1].flatten()[0] for l in row_names]  
    return temp_df

assr_keys = load_conditions_coding(metadata_ASSR)

asop_keys = (
    pd.read_csv(metadata_ASOP,  delimiter=";")
    .melt(id_vars='Subject', var_name='Session', value_name='type')
    .assign(type=lambda x: x['type'].str.replace(' ', ''))
    .assign(type=lambda x: x['type'].replace(u'\xa0', u'', regex=True))
    .rename(columns={'Subject': 'subject', 'Session': 'session'})
)

# %%
# subjectID S T P ROI value
# Where S is pre post sound, T is sham or real, P is pre post tDCS.

columns_order = ['subjectID', 'S', 'T', 'P', 'ROI', 'regions']

def standarize_roi_names(df: pd.DataFrame) -> pd.DataFrame:
    return (
    df
    .pipe(lambda df: df.assign(region= df['ROI'].apply(lambda x: x.rsplit(' ', 1)[0]),
                               hemi=df['ROI'].apply(lambda x: x.rsplit(' ', 1)[1])))
    .replace({'hemi': {"L" :'l', "R" :'r'}})
    .assign(label=lambda x: x['hemi'].str.cat(x['region'], sep='h_'))
    .drop([ "region", "hemi"], axis=1)
     )

df_assr_standarized  = (
    pd.read_csv(f"{specparam_folder}/specparam_ASSR.csv")
    .replace({'iter_number': {1 :'pre', 3 :'pre', 2 :'post',4:'post'}})
    .rename(columns={"iter_number": "P", "condition": "S", "labels": "ROI"})
    .assign(subject_cond=lambda x: x['sub'].str.cat(x['session'], sep='_'))
    .merge(assr_keys, on='subject_cond')
    .assign(subjectID=lambda x: x['sub'].str.extract('(\d+)').astype(int))
    .drop([ "Unnamed: 0", "sub", "session", "subject_cond"], axis=1)
    .pipe(lambda df: df.loc[:, [*columns_order, *df.columns.difference(columns_order)]])
    .sort_values(by=['subjectID', 'ROI'], ascending=[True, False])
    .pipe(standarize_roi_names)
)
df_assr_standarized.to_csv(f"{specparam_folder}/specparam_ASSR_standarized.csv", index=False)

df_aspo_standarized= (
    pd.read_csv(f"{specparam_folder}/specparam_ASPO.csv")
    .replace({'iter_number': {1 :'pre', 3 :'pre', 2 :'post',4:'post'}})
    .assign(S='REST')
    .assign(subjectID=lambda x: x['sub'].str.extract('(\d+)').astype(int))
    .rename(columns={'sub': 'subject'})
    .merge(asop_keys, on=['subject', 'session'], how='left')
    .rename(columns={"iter_number": "P",  "labels": "ROI", "type": "T"})
    .drop([ "Unnamed: 0", "session", "subject"], axis=1)
    .pipe(lambda df: df.loc[:, [*columns_order, *df.columns.difference(columns_order)]])
    .sort_values(by=['subjectID', 'ROI'], ascending=[True, False])
    .pipe(standarize_roi_names)

)
df_aspo_standarized.to_csv(f"{specparam_folder}/specparam_ASPO_standarized.csv", index=False)
# %%
