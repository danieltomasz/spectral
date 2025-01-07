"""Inter-Trial Phase Consistency (ITPC) analysis."""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
from mne_bids import get_entities_from_fname
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def default_freqs() -> np.ndarray:
    """Default frequency range for analysis."""
    return np.arange(39, 41, 1)

@dataclass
class ITPCConfig:
    """Configuration parameters for ITPC analysis."""
    
    tmin_baseline: float = -0.5
    tmax_baseline: float = -0.2
    tmin_analysis: float = -0.5
    tmax_analysis: float = 1.0
    tmin_avg: float = 0.3
    tmax_avg: float = 0.7
    freqs: np.ndarray = field(default_factory=default_freqs)

    n_cycles: int = 20


@dataclass
class ITPCData:
    """Class to store Inter-Trial Phase Consistency (ITPC) data."""
    data: np.ndarray
    times: np.ndarray
    metadata: Dict


    def __post_init__(self):
        """Validate data dimensions."""
        if self.data.shape[1] != len(self.times):
            raise ValueError("Data time dimension must match times length")


def compute_itpc(epochs: mne.Epochs, config: ITPCConfig) -> ITPCData:
    """
    Compute Inter-Trial Phase Coherence using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data to analyze
    config : ITPCConfig
        Configuration parameters
        
    Returns
    -------
    ITPCData
        Computed ITPC values with associated metadata
        
    Raises
    ------
    ValueError
        If input data dimensions are invalid
    """
    try:
        _, itc = epochs.compute_tfr(
            method="morlet",
            freqs=config.freqs,
            average=True,
            n_cycles=config.n_cycles,
            return_itc=True,
            picks="misc",
            n_jobs=-1,
        )
        
        # Average across frequencies if multiple frequencies present
        if itc.data.shape[1] > 1:
            itc_data = itc.data.mean(axis=1)
        else:
            itc_data = np.squeeze(itc.data, axis=1)
            
        metadata = {
            "n_epochs": len(epochs),
            "freqs": config.freqs,
            "n_cycles": config.n_cycles
        }
        
        return ITPCData(data=itc_data, times=itc.times, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Error computing ITPC: {str(e)}")
        raise


def extract_time_window(
    itpc: ITPCData, 
    tmin: float, 
    tmax: float,
    copy: bool = True
) -> ITPCData:
    """Extract ITPC values for a specific time window."""
    mask = (itpc.times >= tmin) & (itpc.times <= tmax)
    data = itpc.data[:, mask].copy() if copy else itpc.data[:, mask]
    return ITPCData(
        data=data,
        times=itpc.times[mask],
        metadata=itpc.metadata
    )

def compute_zscore(
    data: np.ndarray,
    baseline_mask: np.ndarray,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute z-scores with baseline correction.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    baseline_mask : np.ndarray
        Boolean mask for baseline period
    eps : float
        Small constant to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Z-scored data
    """
    baseline_mean = data[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_std = data[:, baseline_mask].std(axis=1, keepdims=True) + eps
    return (data - baseline_mean) / baseline_std

def z_normalize_itpc(
    itpc: ITPCData,
    config: ITPCConfig
) -> ITPCData:
    """Z-normalize ITPC data based on the baseline period."""
    baseline_mask = (itpc.times >= config.tmin_baseline) & (itpc.times <= config.tmax_baseline)
    
    z_scores = compute_zscore(itpc.data, baseline_mask)
    
    metadata = itpc.metadata.copy()
    metadata.update({
        "baseline_period": (config.tmin_baseline, config.tmax_baseline),
        "normalization": "z-score"
    })
    
    return ITPCData(data=z_scores, times=itpc.times, metadata=metadata)

def create_itpc_dataframe(
    itpc: ITPCData,
    epochs: mne.Epochs,
    value_type: str,
    file_metadata: Dict
) -> pd.DataFrame:
    """
    Create a DataFrame with ITPC values and metadata.
    
    Parameters
    ----------
    itpc : ITPCData
        ITPC data to convert
    epochs : mne.Epochs
        Original epochs object for channel information
    value_type : str
        Type of ITPC values (e.g., 'z_normalized_itpc')
    file_metadata : Dict
        Additional metadata from the file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing ITPC values and metadata
    """
    if len(epochs.info["ch_names"]) != itpc.data.shape[0]:
        raise ValueError("Number of channels in epochs doesn't match ITPC data")
        
    return pd.DataFrame({
        "ch_names": epochs.info["ch_names"],
        "value": itpc.data.mean(axis=1),  # Average across time
        "type": value_type,
        "subject": file_metadata["subject"],
        "session": file_metadata["session"],
        "task": file_metadata["task"],
        "run": file_metadata["run"],
    })



def process_single_file(
    file_path: str,
    config: ITPCConfig
) -> pd.DataFrame:
    """
    Process a single file to compute ITPC metrics.
    
    Parameters
    ----------
    file_path : str
        Path to the epochs file
    config : ITPCConfig
        Analysis configuration parameters
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all ITPC metrics
    """
    try:
        epochs = mne.read_epochs(file_path)
        metadata = get_entities_from_fname(file_path, on_error="warn")
        
        # Compute initial ITPC
        itpc = compute_itpc(epochs, config)
        
        # Extract relevant time window
        itpc_extracted = extract_time_window(
            itpc, config.tmin_analysis, config.tmax_analysis
        )
        
        # Compute different ITPC metrics
        z_norm_itpc = z_normalize_itpc(itpc_extracted, config)
        
        # Create DataFrames for each metric
        df_list = []
        
        # Z-normalized ITPC
        df_znorm = create_itpc_dataframe(
            z_norm_itpc, epochs, "z_normalized_itpc", metadata
        )
        df_list.append(df_znorm)
        
        # Pre-stimulus ITPC
        prestim_itpc = extract_time_window(
            itpc_extracted, config.tmin_baseline, config.tmax_baseline
        )
        df_pre = create_itpc_dataframe(
            prestim_itpc, epochs, "prestim_itpc", metadata
        )
        df_list.append(df_pre)
        
        # Stimulus ITPC
        stim_itpc = extract_time_window(
            itpc_extracted, config.tmin_avg, config.tmax_avg
        )
        df_stim = create_itpc_dataframe(
            stim_itpc, epochs, "stim_itpc", metadata
        )
        df_list.append(df_stim)
        
        return pd.concat(df_list, ignore_index=True)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise


def plot_itpc(
    itpc,
    vmin=None,
    vmax=None,
    title="Inter-Trial Phase Consistency (ITPC)",
    label="ITPC",
):
    """
    Plot the Inter-Trial Phase Consistency (ITPC) data.

    Parameters:
    itpc : object
        The ITPC object containing the data to be plotted.
    vmin : float, optional
        The minimum value for the color scale. If None, it will be set to the minimum value of the data.
    vmax : float, optional
        The maximum value for the color scale. If None, it will be set to the maximum value of the data.
    """
    # Set vmin and vmax to data-driven values if not provided
    if vmin is None:
        vmin = itpc.data.min()
    if vmax is None:
        vmax = itpc.data.max()

    # Plot the ITPC data
    fig, ax = plt.subplots()
    im = ax.imshow(
        itpc.data,
        aspect="auto",
        origin="lower",
        extent=(itpc.times[0], itpc.times[-1], 0, itpc.data.shape[0]),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    fig.colorbar(im, ax=ax, label=label)
    plt.show()
