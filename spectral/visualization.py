import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import (
    ggplot,
    aes,
    geom_histogram,
    facet_wrap,
    labs,
    theme_minimal,
    theme,
    scale_x_continuous,
    theme_matplotlib,
)


def plot_psd_and_exponents(
    freqs: np.ndarray,
    avg_psds: np.ndarray,
    exponents: np.ndarray,
    labels: list,
    figsize: tuple = (15, 6),
) -> None:
    """Plot PSDs and exponent histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot PSDs
    for i in range(len(labels)):
        ax1.loglog(freqs, avg_psds[i])
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax1.set_title("Average PSD across epochs")

    # Plot histogram
    ax2.hist(exponents, bins=25, edgecolor="black")
    ax2.set_xlabel("Exponent Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram of Estimated Exponents")

    plt.tight_layout()
    plt.show()

    # Print statistics
    mean_exp = np.mean(exponents)
    std_exp = np.std(exponents)
    print(f"Exponent mean: {mean_exp:.2f} ± {std_exp:.2f}")


def plot_peak_parameters_distribution(
    fits: pd.DataFrame,
    bin_count: int = 30,
    fill_color: str = "blue",
    alpha: float = 0.6,
    figure_size: tuple = (8, 3),
) -> "ggplot":
    """
    Create faceted histogram plots for CF, PW, and BW peak parameters.
    """
    # Select and reshape data using Pandas methods
    plot_data = fits[["CF", "PW", "BW"]].melt(
        value_vars=["CF", "PW", "BW"], var_name="parameter", value_name="value"
    )

    # Parameter labels
    labels = {
        "CF": "Center Frequency (Hz)",
        "PW": "Power",
        "BW": "Bandwidth (Hz)",
    }

    # Create plot with smaller size
    plot = (
        ggplot(plot_data, aes(x="value"))
        + geom_histogram(bins=bin_count, fill=fill_color, alpha=alpha)
        + facet_wrap("~parameter", scales="free", labeller=labels)
        + labs(title="Distribution of Peak Parameters", x="Value", y="Count")
        + theme_minimal()
        + theme(figure_size=figure_size)  # Set figure size
    )

    return plot


def plot_peak_relationships(
    fits: pd.DataFrame, bins_number: int = 25, figsize: tuple = (12, 5)
) -> tuple:
    """
    Plot 2D histograms of CF vs BW and CF vs PW with shared colorbar.

    Parameters
    ----------
    fits : pd.DataFrame
        DataFrame containing CF, BW, and PW columns
    bins_number : int
        Number of bins for 2D histogram
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    tuple
        Figure and axes objects
    """
    # Clean data
    clean_fits = fits.dropna(subset=["CF", "BW", "PW"])

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate common range for colorbar
    h1 = np.histogram2d(clean_fits["CF"], clean_fits["BW"], bins=bins_number)[0]
    h2 = np.histogram2d(clean_fits["CF"], clean_fits["PW"], bins=bins_number)[0]
    vmin = min(h1.min(), h2.min())
    vmax = max(h1.max(), h2.max())

    # Create histplots
    sns.histplot(
        data=clean_fits,
        x="CF",
        y="BW",
        bins=bins_number,
        ax=ax1,
        cbar=False,  # No colorbar for first plot
        vmin=vmin,
        vmax=vmax,
    )
    g = sns.histplot(
        data=clean_fits,
        x="CF",
        y="PW",
        bins=bins_number,
        ax=ax2,
        cbar=True,  # Only one colorbar
        vmin=vmin,
        vmax=vmax,
    )

    # Set titles
    ax1.set_title("Center Frequency vs Bandwidth")
    ax2.set_title("Center Frequency vs Power")

    plt.tight_layout()
    return fig, (ax1, ax2)