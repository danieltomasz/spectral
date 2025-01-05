"""Tests for the brain_atlas module."""

import pytest
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from spectral.brain_atlas import (
    dk_atlas,
    plot_brain,
)  # Replace 'your_module' with the actual module name


@pytest.fixture
def atlas():
    """Return the Desikan-Killiany brain atlas."""
    return dk_atlas()


def test_dk_atlas(atlas):
    """Test the dk_atlas function."""
    assert isinstance(atlas, gpd.GeoDataFrame)
    expected_columns = ["label", "side", "hemi", "geometry"]
    assert all(col in atlas.columns for col in expected_columns)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_plot_brain_basic(atlas, theme, mocker):
    """Test the themes of the plot_brain function."""
    fig = plot_brain(atlas, theme=theme)
    assert isinstance(fig, plt.Figure)
    if theme == "dark":
        assert plt.rcParams["axes.facecolor"] == "black"
    else:
        assert plt.rcParams["axes.facecolor"] == "white"
    plt.close(fig)


def test_plot_brain_with_data(atlas):
    """Test the plot_brain function with data."""
    data_df = pd.DataFrame({"roi": atlas["label"], "value": range(len(atlas))})
    fig = plot_brain(atlas, data_df=data_df, value_column="value")

    # Verify figure was created
    assert isinstance(fig, plt.Figure)

    # Verify we have 4 subplots + colorbar (5 axes total)
    assert len(fig.axes) == 5

    # Verify colorbar exists and has data
    colorbar_ax = fig.axes[-1]
    # Get the colorbar label using get_ylabel instead
    assert colorbar_ax.get_ylabel().strip() == "value"
    # Alternative checks we could add:
    # assert colorbar_ax.collections  # Check if colorbar has collections
    # assert colorbar_ax.get_ylabel() != ""  # Check if label exists

    plt.close(fig)


def test_plot_brain_save(atlas, tmp_path):
    """Test saving the plot."""
    save_path = tmp_path / "test_plot.png"
    fig = plot_brain(atlas)
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    assert save_path.exists()  # Verify file was created
    assert save_path.is_file()

    plt.close(fig)


def test_plot_brain_missing_data(atlas):
    """Test plotting with incomplete data."""
    data_df = pd.DataFrame(
        {
            "roi": atlas["label"][: len(atlas) // 2],  # Only half of the labels
            "value": range(len(atlas) // 2),
        }
    )
    fig = plot_brain(atlas, data_df=data_df, value_column="value")
    assert isinstance(fig, plt.Figure)
    # Verify we have 4 subplots + colorbar
    assert len(fig.axes) == 5
    # Verify colorbar
    colorbar_ax = fig.axes[-1]
    assert colorbar_ax.get_ylabel().strip() == "value"

    plt.close(fig)


def test_plot_brain_custom_cmap(atlas):
    """Test plotting with custom colormap."""
    fig = plot_brain(atlas, cmap="coolwarm")

    # Verify figure was created
    assert isinstance(fig, plt.Figure)

    # Verify we have 4 subplots (no colorbar without data)
    assert len(fig.axes) == 4

    plt.close(fig)
