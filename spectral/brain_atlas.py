"""Functions to load and plot the Desikan-Killiany brain atlas."""

import os
from importlib import resources
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def dk_atlas():
    """Load the Desikan-Killiany brain atlas."""
    gpkg_path = resources.files("spectral.data").joinpath("brain.gpkg")
    return gpd.read_file(gpkg_path)


def convert_roi_names(rois_df, column="label", atlas="dk", new_column="roi"):
    """Create a dictionary to map old names to new names
    Usage
    rois = convert_roi_names(rois_df, column="label")
    """
    if atlas == "dk":
        br_df = dk_atlas()

    name_map = {}

    for _, row in rois_df.iterrows():
        old_name = row[column]

        # Remove the hemisphere indicator (L or R) from the end
        base_name = old_name[:-2] if old_name.endswith((" L", " R")) else old_name

        # Convert to lowercase and replace spaces with underscores
        new_name = base_name.lower().replace(" ", "_")

        # Add 'lh_' or 'rh_' prefix based on the original name
        hemisphere = "lh_" if old_name.endswith(" L") else "rh_"
        new_name = hemisphere + new_name

        name_map[old_name] = new_name

    # Apply the name mapping to the 'label' column
    rois_df[new_column] = rois_df[column].map(name_map)

    # Verify that all new names exist in br_df
    br_labels = set(br_df["label"].dropna())
    if missing_labels := set(rois_df[new_column]) - br_labels:
        print(
            f"Warning: The following converted labels are not in br_df: {missing_labels}"
        )

    return rois_df


def plot_brain(
    atlas,
    data_df=None,
    value_column=None,
    title="Brain Region Plot",
    cmap="viridis",
    theme="dark",
    default_color="lightblue",
    dpi=300,
    figsize=(16, 12),
):
    """
    Plot the brain regions with the specified data column or just the atlas.
    If data_df and value_column are not provided, it plots the atlas itself.
    Returns the figure for further customization or saving.
    """
    # Set the style based on the theme
    if theme == "dark":
        plt.style.use("dark_background")
        title_color = "white"
        edge_color = "white"
        missing_color = "darkgray"
    else:
        plt.style.use("default")
        title_color = "black"
        edge_color = "black"
        missing_color = "lightgray"

    # Create a copy of the atlas
    br_copy = atlas.copy()

    # If data_df and value_column are provided, merge the data
    if data_df is not None and value_column is not None:
        br_copy = br_copy.merge(data_df, left_on="label", right_on="roi", how="left")
        vmin = br_copy[value_column].min()
        vmax = br_copy[value_column].max()
        cmap = plt.get_cmap(cmap)
    else:
        # If no data provided, use a constant value for coloring
        br_copy["constant"] = 1
        value_column = "constant"
        vmin = 0
        vmax = 1
        cmap = mcolors.ListedColormap([default_color])

    # Set up the colormap
    cmap.set_bad(color=missing_color)

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=16, y=0.95, color=title_color)
    axs = axs.ravel()  # Flatten axs for easier indexing

    # Filter and plot for each combination of 'side' and 'hemi'
    plots = []
    for i, (side, hemi) in enumerate(
        [
            ("lateral", "left"),
            ("lateral", "right"),
            ("medial", "left"),
            ("medial", "right"),
        ]
    ):
        plot_data = br_copy[(br_copy["side"] == side) & (br_copy["hemi"] == hemi)]
        plot = plot_data.plot(
            column=value_column,
            ax=axs[i],
            cmap=cmap,
            edgecolor=edge_color,
            linewidth=0.5,
            missing_kwds={"color": missing_color},
            vmin=vmin,
            vmax=vmax,
        )
        plots.append(plot)
        axs[i].axis("off")
        axs[i].set_title(f"{side.capitalize()} {hemi.capitalize()}", color=title_color)

    # Add colorbar only if data is provided
    if data_df is not None and value_column is not None:
        cax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(value_column, rotation=270, labelpad=20, color=title_color)
        cbar.ax.yaxis.set_tick_params(color=title_color)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=title_color)

    # Adjust layout without using tight_layout
    fig.subplots_adjust(
        top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.1, hspace=0.1
    )

    return fig
