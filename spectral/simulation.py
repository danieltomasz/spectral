"""functions to create simulation"""

from typing import Callable, List, Optional, Any
import numpy as np
import neurodsp
import mne
import neurodsp.sim
from typing import Optional, Union, Dict, List, Tuple
import numpy as np
from neurodsp.sim import sim_powerlaw

SimulationFunc = Callable[..., np.ndarray]


def default_simulation(
    n_seconds: int, fs: int, n_epochs: int, exponent: float
) -> np.ndarray:
    """Default 1/f noise simulation function"""
    n_points = int(n_seconds * fs)
    epochs_data = np.zeros((n_epochs, n_points))
    for i in range(n_epochs):
        signal = neurodsp.sim.sim_powerlaw(n_seconds, fs, exponent)
        epochs_data[i, :] = signal
    return epochs_data


def generate_ssep(
    t: np.ndarray,
    f: float,
    H: int,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    window: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate Steady State Evoked Potentials signal.

    Parameters:
    -----------
    t : np.ndarray
        Time points
    f : float
        Fundamental frequency
    H : int
        Number of harmonics
    amplitudes : np.ndarray
        Array of amplitudes (a_h^k) for each harmonic
    phases : np.ndarray
        Array of phases (φ_h^k) for each harmonic
    window : bool, optional
        Apply Hanning window to signal (default=True)

    Returns:
    --------
    s_k : np.ndarray
        The generated SSEP signal
    harmonics : list
        List of individual harmonic components
    """
    if len(amplitudes) != H or len(phases) != H:
        raise ValueError(
            "Length of amplitudes and phases must match the number of harmonics H"
        )

    s_k = np.zeros_like(t, dtype=float)
    harmonics = []

    # Calculate the sum of harmonics
    for h in range(H):
        h_idx = h + 1  # harmonic index (1-based)
        # Calculate individual harmonic
        harmonic = amplitudes[h] * np.cos(2 * np.pi * h_idx * f * t + phases[h])
        harmonics.append(harmonic)
        # Add to total signal
        s_k += harmonic

    if window:
        window_func = np.hanning(len(s_k))
        s_k *= window_func

    return s_k, harmonics


def normalize_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    """Normalize signal to have similar total power."""
    psd, freqs = mne.time_frequency.psd_array_welch(signal, sfreq=fs, fmin=0, fmax=fs/2, n_fft=fs)
    total_power = np.sum(psd)
    return signal / np.sqrt(total_power)


def simulate_ssep(
    n_seconds: int,
    fs: int,
    n_epochs: int,
    exponent: float,
    ssep_params: Optional[Dict] = None,
    add_powerlaw: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Simulate epochs with SSEP signal, optionally combined with 1/f noise.

    Parameters
    ----------
    n_seconds : int
        Duration of each epoch in seconds
    fs : int
        Sampling frequency
    n_epochs : int
        Number of epochs to generate
    exponent : float
        Exponent for 1/f noise (only used if add_powerlaw=True)
    ssep_params : dict, optional
        Dictionary with SSEP parameters:
        - freq : float, fundamental frequency
        - n_harmonics : int, number of harmonics
        - amplitudes : array-like, amplitudes for each harmonic
        - phases : array-like, phases for each harmonic
        - scale : float, scaling factor for SSEP signal
    add_powerlaw : bool, optional
        Whether to add 1/f noise to the SSEP signal (default=True)
    normalize : bool, optional
        Whether to normalize the 1/f noise (default=False)

    Returns
    -------
    epochs_data : np.ndarray
        Array of simulated epochs (n_epochs × n_timepoints)
    """
    # Default SSEP parameters
    default_params = {
        "freq": 40,
        "n_harmonics": 3,
        "amplitudes": [1.0, 0.5, 0.25],
        "phases": [0, np.pi / 4, np.pi / 2],
        "scale": 0.3,
    }
    ssep_params = ssep_params or default_params

    n_points = int(n_seconds * fs)
    t = np.linspace(0, n_seconds, n_points)
    epochs_data = np.zeros((n_epochs, n_points))

    for i in range(n_epochs):
        # Generate SSEP signal
        ssep_signal, _ = generate_ssep(
            t,
            ssep_params["freq"],
            ssep_params["n_harmonics"],
            ssep_params["amplitudes"],
            ssep_params["phases"],
        )

        if add_powerlaw:
            # Generate 1/f noise
            noise = sim_powerlaw(n_seconds, fs, exponent)
            if normalize:
                noise = normalize_signal(noise, fs)
            # Combine signals
            signal = noise + ssep_params["scale"] * ssep_signal
        else:
            signal = ssep_params["scale"] * ssep_signal

        epochs_data[i, :] = signal

    return epochs_data


def simulate_signal_with_peaks(
    n_seconds: int,
    fs: int,
    n_epochs: int,
    exponent: float,
    peak_params: dict = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Simulate epochs with 1/f noise and oscillatory peaks.

    Parameters
    ----------
    n_seconds : int
        Duration of each epoch in seconds
    fs : int
        Sampling frequency
    n_epochs : int
        Number of epochs to generate
    exponent : float
        Exponent for 1/f noise
    peak_params : dict, optional
        Dictionary with peak parameters:
        - freq : float, oscillation frequency
        - bw : float, bandwidth
        - height : float, peak height
    normalize : bool, optional
        Whether to normalize the 1/f noise (default=False)
    """
    # Default peak parameters
    default_peaks = {"freq": 10, "bw": 3, "height": 2}
    peak_params = peak_params or default_peaks

    n_points = int(n_seconds * fs)
    epochs_data = np.zeros((n_epochs, n_points))

    for i in range(n_epochs):
        # Generate aperiodic component
        ap_sig = neurodsp.sim.sim_powerlaw(n_seconds, fs, exponent)
        if normalize:
            ap_sig = normalize_signal(ap_sig, fs)

        # Add oscillatory peak
        sig = neurodsp.sim.sim_peak_oscillation(
            ap_sig,
            fs,
            peak_params["freq"],
            peak_params["bw"],
            peak_params["height"],
        )
        epochs_data[i, :] = sig

    return epochs_data


def create_simulated_epochs(
    labels: List[str],
    simulation_func: SimulationFunc = default_simulation,
    n_seconds: int = 1,
    fs: int = 1000,
    n_epochs: int = 100,
    exponent: float = -2,
    filter_freq: Optional[float] = 125,
    resample_freq: Optional[int] = 250,
    **kwargs
) -> mne.Epochs:
    """
    Create simulated EEG epochs using a provided simulation function.
    """
    # Simulate epochs for each label
    simulated_data = {}
    for label in labels:
        simulated_data[label] = simulation_func(
            n_seconds, fs, n_epochs, exponent, **kwargs
        )

    # Combine data into single array
    combined_data = np.stack(
        [simulated_data[label] for label in labels], axis=1
    )

    # Create MNE info and events
    info = mne.create_info(
        ch_names=labels, sfreq=fs, ch_types=["misc"] * len(labels)
    )
    events = np.array([[i, 0, 1] for i in range(n_epochs)])

    # Create and process epochs
    epochs = mne.EpochsArray(combined_data, info, events)

    if filter_freq:
        epochs = epochs.filter(l_freq=None, picks="misc", h_freq=filter_freq)
    if resample_freq and resample_freq != fs:
        epochs = epochs.resample(sfreq=resample_freq)

    return epochs
