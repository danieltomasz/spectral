#| output: false
#| warning: false


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from fooof.core.funcs import gaussian_function, expo_nk_function


def plot_group_lines(df, file_path=None, condition='condition', dpi=300, fig_size=None, font_size=16):
    """
    Create a line plot with grouped lines.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe with the data to plot. Should have columns "frequency", "power", and "condition".
    filepath : str, optional
        The path to save the resulting plot. Default is None which means no file is saved.
    condition : str, optional
        The name of the column in the DataFrame that specifies the grouping variable. Default is 'condition'.\n    
    dpi : int, optional
        The resolution of the saved plot. Default is 300.   
    fig_size : list, optional      
        The size of the figure in inches as a list of two values. Default is None which means [10, 5].
    fontsize : int, optional
        The font size of the plot. Default is 16.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting matplotlib figure object.

    """
    if fig_size is None:
        fig_size = [10, 5]
    fig = plt.figure()
    b = sns.lineplot(data=df, x="frequency", y="power", hue=condition, estimator=None,)
    plt.xlabel("Frequency", fontsize=font_size)
    plt.ylabel("Log(Power)", fontsize=font_size)
    #plt.yscale("log")
    #plt.xscale("log")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Power Spectrum", fontsize=font_size)
    b.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)
    fig.set_size_inches(fig_size) # type: ignore
    fig.set_dpi(dpi)
    # Save the figure
    if file_path is not None:
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return fig



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
