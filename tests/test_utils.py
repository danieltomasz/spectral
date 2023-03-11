# %%
import lambic
import importlib
importlib.reload(lambic)
from lambic.utils import generate_example_spectra

def test_generate_example_spectra():
    generate_example_spectra()
    assert True
# %%
if __name__ == '__main__':
    freqs, powers, sim_params = generate_example_spectra()
    assert True
# %%
