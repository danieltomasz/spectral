"""Plots for fitted  parameters."""

#%% import packages

import numpy as np
import xarray as xr
import pandas as pd
from fooof import FOOOF, FOOOFGroup
from fooof.core.funcs import gaussian_function, expo_nk_function

import matplotlib.pyplot as plt
import seaborn as sns
sns.set( font_scale = 1.5)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set(style="ticks", rc={"lines.linewidth": 0.6})


#%% Define functions

def create_slope_from_parameters(aps, columns):
    freq =  np.arange(2, 45, 1)
    list_data = []
    for index, row in aps.iterrows():
        power = expo_nk_function(freq, row["offset"], row["exponent"])
        data = pd.DataFrame({'power': power, 'frequency':freq, })
        for variable in columns:
            data[variable]  = row[variable]
        list_data.append(data)

    return pd.concat(list_data,  ignore_index=True)


def plot_lines(df, condition = 'condition', subject = 'subjectID'):
    fig = plt.figure()
    b = sns.lineplot(data=df, x="frequency", y="power", hue=condition,  units= subject, estimator=None,)
    #b.set_xlim([min(freq)-0.5, max(freq) + 0.5])
    plt.xlabel("Frequency", size=20)
    plt.ylabel("Power", size=20)
    plt.title("Power Spectrum", size=20)
    return fig

def plot_group_lines(df, condition = 'condition'):
    print(condition)
    fig = plt.figure()
    b = sns.lineplot(data=df, x="frequency", y="power", hue=condition, estimator=None,)
    #b.set_xlim([min(freq)-0.5, max(freq) + 0.5])
    plt.xlabel("Frequency", size=20)
    plt.ylabel("Power", size=20)
    plt.title("Power Spectrum", size=20)
    return fig


def plot_fit(df: xr.DataArray, fm :FOOOF, freq_range, file_name :str = None, file_path :str = None):
    freqs = df.freqs.values
    spectrum = df.values[0]
    try:
        fm.add_data(freqs, spectrum, freq_range)
        roi_name = df.labels.values.item().replace(" ", "")
        fm.fit()
        # Filename and path to save the figure
        if file_name is None:
            file_name = f"{df.sub.values.item()}_{roi_name}_spectrum.png"
        if file_path is None:
            file_path = 'figures'
        
        fm.plot(save_fig=True, file_name= file_name, file_path=file_path)
    except Exception as e:
        print("Caught an error: ", e)    
# %%
