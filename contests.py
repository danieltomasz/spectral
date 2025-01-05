"""configure pytest for notebook testing:"""

import pytest


def pytest_collect_file(parent, path):
    """Collect IPython notebooks for testing."""
    if path.ext == ".ipynb":
        return pytest.File.from_parent(parent, path=path)
