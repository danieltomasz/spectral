# %%
from pathlib import Path
import importlib
import pytest
import numpy as np
import pandas as pd

import lambic
importlib.reload(lambic)


def test_plot_lines():
    # Get frequency axis (x-axis)
    taps = {
        "offset": [-16.237085, -15.772295],
        "exponent": [1.18727834, 2.367788],
        "condition": ["OLD", "YOUNG"],
        "subjectID": ["sub-032528", "sub-032448"],
    }
    aps = pd.DataFrame(taps)
    final_list = lambic.visualize.create_slope_from_parameters(
        aps, ["condition", "subjectID"]
    )

    lambic.visualize.plot_lines(final_list)
    assert True


# %%
if __name__ == '__main__':
    test_plot_lines()
# %%
