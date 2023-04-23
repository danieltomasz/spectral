# %%
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import seaborn as sns

import poirot
from poirot.utils import reload_package

reload_package(poirot)

# %%

def test_plot_lines():
    # Get frequency axis (x-axis)
    taps = {
        "offset": [-16.237085, -15.772295],
        "exponent": [1.18727834, 2.367788],
        "condition": ["OLD", "YOUNG"],
        "subjectID": ["sub-032528", "sub-032448"],
    }
    aps = pd.DataFrame(taps)
    final_list = poirot.visualize.create_slope_from_parameters(
        aps, ["condition", "subjectID"]
    )

    poirot.visualize.plot_lines(final_list)

def test_plot_group_lines():
    # Get frequency axis (x-axis)
    taps = {
        "offset": [-16.237085, -15.772295, -16.23885],
        "exponent": [1.18727834, 2.367788, 1.996912],
        "Y_OH_OL": ["O_H", "O_L", "Y_H" ],
    }
    aps = pd.DataFrame(taps)
    final_list = poirot.visualize.create_slope_from_parameters(
        aps, ["Y_OH_OL"]
    )

    poirot.visualize.plot_group_lines(final_list, condition = "Y_OH_OL")
    assert True


# %%
if __name__ == '__main__':
    test_plot_group_lines()
# %%
