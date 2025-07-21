"""
Signal Filtering Utilities
=========================

Common filtering functions for electrophysiology data preprocessing.
Supports various filter types and methods with consistent interface.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import signal
from typing import Optional, Union, Tuple
import logging

from ...core.exceptions import PreprocessingError, InvalidParameterError

logger = logging.getLogger(__name__)


def design_filter(filter_type: str,
                 cutoff_freq: Union[float, Tuple[float, float]],
                 sampling_rate: float,
                 order: int = 4,
                 method: str = 'butter') -> Tuple[np.ndarray, np.ndarray]:
    """
    Design digital filter coefficients.
    
    Parameters
    ----------
    filter_type : str
        Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    cutoff_freq : float or tuple
        Cutoff frequency(ies) in Hz. Single value for low/highpass,
        tuple (low, high) for band filters
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
        
    Returns
    -------
    sos : ndarray
        Second-order sections representation of the filter
        
    Raises
    ------
    InvalidParameterError
        If parameters are invalid
    """
    # Validate inputs
    valid_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    if filter_type not in valid_types:
        raise InvalidParameterError(f"Invalid filter type: {filter_type}")
        
    valid_methods = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    if method not in valid_methods:
        raise InvalidParameterError(f"Invalid filter method: {method}")
        
    # Calculate Nyquist frequency
    nyquist = sampling_rate / 2
    
    # Normalize frequencies
    if filter_type in ['lowpass', 'highpass']:
        if isinstance(cutoff_freq, (list, tuple)):
            cutoff_freq = cutoff_freq[0]
        Wn = cutoff_freq / nyquist
        
        if not 0 < Wn < 1:
            raise InvalidParameterError(
                f"Cutoff frequency {cutoff_freq} Hz must be between 0 and {nyquist} Hz"
            )
    else:  # bandpass or bandstop
        if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
            raise InvalidParameterError(
                f"Band filters require two cutoff frequencies, got {cutoff_freq}"
            )
        Wn = [cutoff_freq[0] / nyquist, cutoff_freq[1] / nyquist]
        
        if not (0 < Wn[0] < Wn[1] < 1):
            raise InvalidParameterError(
                f"Invalid cutoff frequencies: {cutoff_freq}. "
                f"Must satisfy 0 < low < high < {nyquist} Hz"
            )
    
    # Design filter based on method
    try:
        if method == 'butter':
            sos = signal.butter(order, Wn, btype=filter_type, output='sos')
        elif method == 'cheby1':
            sos = signal.cheby1(order, 0.5, Wn, btype=filter_type, output='sos')
        elif method == 'cheby2':
            sos = signal.cheby2(order, 40, Wn, btype=filter_type, output='sos')
        elif method == 'ellip':
            sos = signal.ellip(order, 0.5, 40, Wn, btype=filter_type, output='sos')
        elif method == 'bessel':
            sos = signal.bessel(order, Wn, btype=filter_type, output='sos')
    except Exception as e:
        raise PreprocessingError(f"Failed to design filter: {e}")
        
    return sos


def apply_filter(data: np.ndarray,
                filter_type: str,
                cutoff_freq: Union[float, Tuple[float, float]],
                sampling_rate: float,
                order: int = 4,
                method: str = 'butter',
                axis: int = -1,
                padlen: Optional[int] = None) -> np.ndarray:
    """
    Apply digital filter to data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    filter_type : str
        Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    cutoff_freq : float or tuple
        Cutoff frequency(ies) in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method
    axis : int
        Axis along which to filter
    padlen : int, optional
        Padding length for filtfilt. If None, uses default
        
    Returns
    -------
    filtered : ndarray
        Filtered data
    """
    # Design filter
    sos = design_filter(filter_type, cutoff_freq, sampling_rate, order, method)
    
    # Apply filter
    try:
        if padlen is None:
            # Use default padding
            filtered = signal.sosfiltfilt(sos, data, axis=axis)
        else:
            filtered = signal.sosfiltfilt(sos, data, axis=axis, padlen=padlen)
    except Exception as e:
        raise PreprocessingError(f"Failed to apply filter: {e}")
        
    return filtered


def bandpass_filter(data: np.ndarray,
                   low_freq: float,
                   high_freq: float,
                   sampling_rate: float,
                   order: int = 4,
                   method: str = 'butter',
                   **kwargs) -> np.ndarray:
    """
    Apply bandpass filter to data.
    
    Convenience function for bandpass filtering.
    
    Parameters
    ----------
    data : ndarray
        Input data
    low_freq : float
        Low cutoff frequency in Hz
    high_freq : float
        High cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method
    **kwargs
        Additional arguments passed to apply_filter
        
    Returns
    -------
    filtered : ndarray
        Bandpass filtered data
    """
    return apply_filter(data, 'bandpass', (low_freq, high_freq), 
                       sampling_rate, order, method, **kwargs)


def lowpass_filter(data: np.ndarray,
                  cutoff_freq: float,
                  sampling_rate: float,
                  order: int = 4,
                  method: str = 'butter',
                  **kwargs) -> np.ndarray:
    """
    Apply lowpass filter to data.
    
    Convenience function for lowpass filtering.
    
    Parameters
    ----------
    data : ndarray
        Input data
    cutoff_freq : float
        Cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method
    **kwargs
        Additional arguments passed to apply_filter
        
    Returns
    -------
    filtered : ndarray
        Lowpass filtered data
    """
    return apply_filter(data, 'lowpass', cutoff_freq, 
                       sampling_rate, order, method, **kwargs)


def highpass_filter(data: np.ndarray,
                   cutoff_freq: float,
                   sampling_rate: float,
                   order: int = 4,
                   method: str = 'butter',
                   **kwargs) -> np.ndarray:
    """
    Apply highpass filter to data.
    
    Convenience function for highpass filtering.
    
    Parameters
    ----------
    data : ndarray
        Input data
    cutoff_freq : float
        Cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method
    **kwargs
        Additional arguments passed to apply_filter
        
    Returns
    -------
    filtered : ndarray
        Highpass filtered data
    """
    return apply_filter(data, 'highpass', cutoff_freq, 
                       sampling_rate, order, method, **kwargs)


def notch_filter(data: np.ndarray,
                notch_freq: float,
                sampling_rate: float,
                quality_factor: float = 30.0,
                **kwargs) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency.
    
    Useful for removing line noise (50/60 Hz).
    
    Parameters
    ----------
    data : ndarray
        Input data
    notch_freq : float
        Frequency to remove in Hz
    sampling_rate : float
        Sampling rate in Hz
    quality_factor : float
        Quality factor Q = freq/bandwidth
    **kwargs
        Additional arguments passed to filtfilt
        
    Returns
    -------
    filtered : ndarray
        Notch filtered data
    """
    # Design notch filter
    b, a = signal.iirnotch(notch_freq / (sampling_rate / 2), quality_factor)
    
    # Apply filter
    try:
        filtered = signal.filtfilt(b, a, data, **kwargs)
    except Exception as e:
        raise PreprocessingError(f"Failed to apply notch filter: {e}")
        
    return filtered


def adaptive_filter(data: np.ndarray,
                   sampling_rate: float,
                   min_freq: float = 0.1,
                   max_freq: Optional[float] = None) -> np.ndarray:
    """
    Apply adaptive filtering based on signal characteristics.
    
    Automatically determines appropriate filter parameters based
    on signal spectrum analysis.
    
    Parameters
    ----------
    data : ndarray
        Input data
    sampling_rate : float
        Sampling rate in Hz
    min_freq : float
        Minimum frequency to preserve
    max_freq : float, optional
        Maximum frequency to preserve. If None, uses Nyquist/2
        
    Returns
    -------
    filtered : ndarray
        Adaptively filtered data
    """
    if max_freq is None:
        max_freq = sampling_rate / 4  # Conservative default
        
    # Analyze signal spectrum
    freqs, psd = signal.welch(data, sampling_rate, nperseg=min(len(data), 4096))
    
    # Find dominant frequency range
    total_power = np.sum(psd)
    cumulative_power = np.cumsum(psd)
    
    # Find frequency range containing 95% of power
    idx_low = np.where(cumulative_power >= 0.025 * total_power)[0][0]
    idx_high = np.where(cumulative_power >= 0.975 * total_power)[0][0]
    
    freq_low = max(freqs[idx_low], min_freq)
    freq_high = min(freqs[idx_high], max_freq)
    
    logger.info(f"Adaptive filter: {freq_low:.1f} - {freq_high:.1f} Hz")
    
    # Apply bandpass filter
    if freq_low < freq_high:
        return bandpass_filter(data, freq_low, freq_high, sampling_rate)
    else:
        logger.warning("Could not determine appropriate filter range")
        return data


def filter_bank(data: np.ndarray,
               frequency_bands: list,
               sampling_rate: float,
               order: int = 4,
               method: str = 'butter') -> dict:
    """
    Apply multiple bandpass filters to decompose signal.
    
    Parameters
    ----------
    data : ndarray
        Input data
    frequency_bands : list
        List of (low, high) frequency tuples
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order
    method : str
        Filter design method
        
    Returns
    -------
    filtered_bands : dict
        Dictionary mapping frequency bands to filtered data
    """
    filtered_bands = {}
    
    for low, high in frequency_bands:
        band_name = f"{low}-{high}Hz"
        try:
            filtered_bands[band_name] = bandpass_filter(
                data, low, high, sampling_rate, order, method
            )
        except Exception as e:
            logger.warning(f"Failed to filter band {band_name}: {e}")
            
    return filtered_bands


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate test signal
    fs = 1000  # Hz
    t = np.linspace(0, 1, fs)
    
    # Signal with multiple components
    signal_clean = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz
                   0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz
                   0.3 * np.sin(2 * np.pi * 120 * t))  # 120 Hz
    
    # Add noise
    signal_noisy = signal_clean + 0.2 * np.random.randn(len(t))
    
    # Apply different filters
    filtered_low = lowpass_filter(signal_noisy, 30, fs)
    filtered_band = bandpass_filter(signal_noisy, 5, 15, fs)
    filtered_notch = notch_filter(signal_noisy, 50, fs)
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    
    axes[0].plot(t, signal_noisy, 'b-', alpha=0.7)
    axes[0].set_title('Original Noisy Signal')
    axes[0].set_ylabel('Amplitude')
    
    axes[1].plot(t, filtered_low, 'g-')
    axes[1].set_title('Lowpass Filtered (30 Hz)')
    axes[1].set_ylabel('Amplitude')
    
    axes[2].plot(t, filtered_band, 'r-')
    axes[2].set_title('Bandpass Filtered (5-15 Hz)')
    axes[2].set_ylabel('Amplitude')
    
    axes[3].plot(t, filtered_notch, 'm-')
    axes[3].set_title('Notch Filtered (50 Hz removed)')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()