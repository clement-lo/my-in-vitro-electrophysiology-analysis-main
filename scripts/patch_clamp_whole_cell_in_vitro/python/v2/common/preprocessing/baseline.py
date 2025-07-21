"""
Baseline Correction Utilities
=============================

Functions for baseline correction and offset removal in
electrophysiology recordings.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import signal, stats
from typing import Optional, Union, Tuple, List
import logging

from ...core.exceptions import PreprocessingError, InvalidParameterError

logger = logging.getLogger(__name__)


def baseline_mean(data: np.ndarray,
                 baseline_window: Union[float, Tuple[float, float]],
                 sampling_rate: float,
                 axis: int = -1) -> np.ndarray:
    """
    Correct baseline using mean of baseline window.
    
    Parameters
    ----------
    data : ndarray
        Input data
    baseline_window : float or tuple
        If float: duration from start (seconds)
        If tuple: (start, end) times in seconds
    sampling_rate : float
        Sampling rate in Hz
    axis : int
        Axis along which to correct baseline
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Get baseline indices
    baseline_indices = _get_baseline_indices(data.shape[axis], baseline_window, sampling_rate)
    
    # Calculate baseline
    if data.ndim == 1:
        baseline = np.mean(data[baseline_indices])
        return data - baseline
    else:
        # Extract baseline region
        baseline_data = np.take(data, baseline_indices, axis=axis)
        baseline = np.mean(baseline_data, axis=axis, keepdims=True)
        return data - baseline


def baseline_median(data: np.ndarray,
                   baseline_window: Union[float, Tuple[float, float]],
                   sampling_rate: float,
                   axis: int = -1) -> np.ndarray:
    """
    Correct baseline using median of baseline window.
    
    More robust to outliers than mean baseline correction.
    
    Parameters
    ----------
    data : ndarray
        Input data
    baseline_window : float or tuple
        If float: duration from start (seconds)
        If tuple: (start, end) times in seconds
    sampling_rate : float
        Sampling rate in Hz
    axis : int
        Axis along which to correct baseline
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Get baseline indices
    baseline_indices = _get_baseline_indices(data.shape[axis], baseline_window, sampling_rate)
    
    # Calculate baseline
    if data.ndim == 1:
        baseline = np.median(data[baseline_indices])
        return data - baseline
    else:
        # Extract baseline region
        baseline_data = np.take(data, baseline_indices, axis=axis)
        baseline = np.median(baseline_data, axis=axis, keepdims=True)
        return data - baseline


def baseline_mode(data: np.ndarray,
                 baseline_window: Union[float, Tuple[float, float]],
                 sampling_rate: float,
                 axis: int = -1,
                 bins: int = 50) -> np.ndarray:
    """
    Correct baseline using mode (most common value) of baseline window.
    
    Useful when baseline contains a dominant value.
    
    Parameters
    ----------
    data : ndarray
        Input data
    baseline_window : float or tuple
        If float: duration from start (seconds)
        If tuple: (start, end) times in seconds
    sampling_rate : float
        Sampling rate in Hz
    axis : int
        Axis along which to correct baseline
    bins : int
        Number of bins for histogram-based mode estimation
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Get baseline indices
    baseline_indices = _get_baseline_indices(data.shape[axis], baseline_window, sampling_rate)
    
    # Calculate baseline
    if data.ndim == 1:
        baseline_data = data[baseline_indices]
        baseline = _estimate_mode(baseline_data, bins)
        return data - baseline
    else:
        # Apply along axis
        return np.apply_along_axis(
            lambda x: x - _estimate_mode(x[baseline_indices], bins),
            axis, data
        )


def baseline_polynomial(data: np.ndarray,
                       baseline_window: Union[float, Tuple[float, float]],
                       sampling_rate: float,
                       order: int = 1,
                       axis: int = -1) -> np.ndarray:
    """
    Correct baseline by fitting and subtracting polynomial.
    
    Parameters
    ----------
    data : ndarray
        Input data
    baseline_window : float or tuple
        If float: duration from start (seconds)
        If tuple: (start, end) times in seconds
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Polynomial order
    axis : int
        Axis along which to correct baseline
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Get baseline indices
    baseline_indices = _get_baseline_indices(data.shape[axis], baseline_window, sampling_rate)
    
    if data.ndim == 1:
        return _baseline_poly_1d(data, baseline_indices, order)
    else:
        return np.apply_along_axis(
            _baseline_poly_1d, axis, data, baseline_indices, order
        )


def baseline_percentile(data: np.ndarray,
                       baseline_window: Union[float, Tuple[float, float]],
                       sampling_rate: float,
                       percentile: float = 10,
                       axis: int = -1) -> np.ndarray:
    """
    Correct baseline using percentile of baseline window.
    
    Useful when baseline is contaminated with positive-going artifacts.
    
    Parameters
    ----------
    data : ndarray
        Input data
    baseline_window : float or tuple
        If float: duration from start (seconds)
        If tuple: (start, end) times in seconds
    sampling_rate : float
        Sampling rate in Hz
    percentile : float
        Percentile to use (0-100)
    axis : int
        Axis along which to correct baseline
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    if not 0 <= percentile <= 100:
        raise InvalidParameterError("Percentile must be between 0 and 100")
        
    # Get baseline indices
    baseline_indices = _get_baseline_indices(data.shape[axis], baseline_window, sampling_rate)
    
    # Calculate baseline
    if data.ndim == 1:
        baseline = np.percentile(data[baseline_indices], percentile)
        return data - baseline
    else:
        # Extract baseline region
        baseline_data = np.take(data, baseline_indices, axis=axis)
        baseline = np.percentile(baseline_data, percentile, axis=axis, keepdims=True)
        return data - baseline


def baseline_als(data: np.ndarray,
                lam: float = 1e6,
                p: float = 0.01,
                niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline correction.
    
    This method is particularly good for removing slowly varying
    baseline from spectra or similar data.
    
    Parameters
    ----------
    data : ndarray
        Input data (1D)
    lam : float
        Smoothness parameter (larger = smoother)
    p : float
        Asymmetry parameter (typically 0.001-0.1)
    niter : int
        Number of iterations
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
        
    References
    ----------
    Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline Correction with
    Asymmetric Least Squares Smoothing.
    """
    if data.ndim != 1:
        raise InvalidParameterError("ALS baseline correction only supports 1D data")
        
    L = len(data)
    D = _sparse_diff_matrix(L, 2)
    D = lam * D.T @ D
    w = np.ones(L)
    
    for i in range(niter):
        W = _sparse_diag_matrix(w)
        Z = W + D
        z = np.linalg.solve(Z, w * data)
        w = p * (data > z) + (1 - p) * (data < z)
        
    return data - z


def baseline_rolling(data: np.ndarray,
                    window_size: Union[int, float],
                    sampling_rate: Optional[float] = None,
                    method: str = 'median',
                    axis: int = -1) -> np.ndarray:
    """
    Rolling baseline correction.
    
    Subtracts a rolling window statistic from the signal.
    
    Parameters
    ----------
    data : ndarray
        Input data
    window_size : int or float
        Window size in samples (int) or seconds (float, requires sampling_rate)
    sampling_rate : float, optional
        Sampling rate in Hz (required if window_size is in seconds)
    method : str
        Statistic to use: 'mean', 'median', 'min'
    axis : int
        Axis along which to correct baseline
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Convert window size to samples
    if isinstance(window_size, float):
        if sampling_rate is None:
            raise InvalidParameterError("sampling_rate required when window_size is in seconds")
        window_samples = int(window_size * sampling_rate)
    else:
        window_samples = window_size
        
    # Ensure odd window for symmetry
    if window_samples % 2 == 0:
        window_samples += 1
        
    # Apply rolling baseline
    if method == 'mean':
        baseline = _rolling_mean(data, window_samples, axis)
    elif method == 'median':
        baseline = _rolling_median(data, window_samples, axis)
    elif method == 'min':
        baseline = _rolling_min(data, window_samples, axis)
    else:
        raise InvalidParameterError(f"Unknown method: {method}")
        
    return data - baseline


def baseline_morphological(data: np.ndarray,
                          struct_size: Union[int, float],
                          sampling_rate: Optional[float] = None,
                          method: str = 'opening') -> np.ndarray:
    """
    Morphological baseline correction.
    
    Uses mathematical morphology operations to estimate baseline.
    Good for removing peaks while preserving baseline.
    
    Parameters
    ----------
    data : ndarray
        Input data (1D)
    struct_size : int or float
        Structuring element size in samples (int) or seconds (float)
    sampling_rate : float, optional
        Sampling rate in Hz (required if struct_size is in seconds)
    method : str
        Morphological operation: 'opening', 'closing', 'tophat'
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    if data.ndim != 1:
        raise InvalidParameterError("Morphological baseline correction only supports 1D data")
        
    # Convert structure size to samples
    if isinstance(struct_size, float):
        if sampling_rate is None:
            raise InvalidParameterError("sampling_rate required when struct_size is in seconds")
        struct_samples = int(struct_size * sampling_rate)
    else:
        struct_samples = struct_size
        
    # Create structuring element
    struct = np.ones(struct_samples)
    
    try:
        from scipy import ndimage
        
        if method == 'opening':
            # Opening: erosion followed by dilation
            baseline = ndimage.grey_opening(data, size=struct_samples)
        elif method == 'closing':
            # Closing: dilation followed by erosion
            baseline = ndimage.grey_closing(data, size=struct_samples)
        elif method == 'tophat':
            # Top-hat: data minus opening
            opening = ndimage.grey_opening(data, size=struct_samples)
            return data - opening
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
            
        return data - baseline
        
    except Exception as e:
        raise PreprocessingError(f"Morphological baseline correction failed: {e}")


def baseline_adaptive(data: np.ndarray,
                     sampling_rate: float,
                     method: str = 'auto') -> Tuple[np.ndarray, str]:
    """
    Adaptive baseline correction.
    
    Automatically selects appropriate baseline correction method
    based on signal characteristics.
    
    Parameters
    ----------
    data : ndarray
        Input data
    sampling_rate : float
        Sampling rate in Hz
    method : str
        Method selection: 'auto' or specific method
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    method_used : str
        The baseline correction method that was applied
    """
    if method != 'auto':
        # Use specified method with default parameters
        if method == 'mean':
            baseline_window = min(0.1, len(data) / (10 * sampling_rate))
            return baseline_mean(data, baseline_window, sampling_rate), method
        elif method == 'median':
            baseline_window = min(0.1, len(data) / (10 * sampling_rate))
            return baseline_median(data, baseline_window, sampling_rate), method
        elif method == 'polynomial':
            baseline_window = min(0.1, len(data) / (10 * sampling_rate))
            return baseline_polynomial(data, baseline_window, sampling_rate), method
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
            
    # Auto-detect best method
    # Analyze signal characteristics
    signal_std = np.std(data)
    signal_range = np.ptp(data)
    
    # Check for drift
    n_segments = 10
    segment_size = len(data) // n_segments
    segment_means = [np.mean(data[i*segment_size:(i+1)*segment_size]) 
                    for i in range(n_segments)]
    drift = np.std(segment_means) / signal_std if signal_std > 0 else 0
    
    # Select method based on characteristics
    if drift > 0.5:
        # Significant drift - use polynomial
        method_used = 'polynomial'
        baseline_window = min(0.2, len(data) / (5 * sampling_rate))
        corrected = baseline_polynomial(data, baseline_window, sampling_rate, order=2)
    elif signal_range / signal_std > 10:
        # Large outliers - use median
        method_used = 'median'
        baseline_window = min(0.1, len(data) / (10 * sampling_rate))
        corrected = baseline_median(data, baseline_window, sampling_rate)
    else:
        # Default to mean
        method_used = 'mean'
        baseline_window = min(0.1, len(data) / (10 * sampling_rate))
        corrected = baseline_mean(data, baseline_window, sampling_rate)
        
    logger.info(f"Auto-selected baseline correction: {method_used}")
    return corrected, method_used


def baseline_correct(data: np.ndarray,
                    method: str = 'mean',
                    baseline_window: Optional[Union[float, Tuple[float, float]]] = None,
                    sampling_rate: Optional[float] = None,
                    **kwargs) -> np.ndarray:
    """
    General baseline correction function.
    
    Parameters
    ----------
    data : ndarray
        Input data
    method : str
        Baseline correction method
    baseline_window : float or tuple, optional
        Baseline window specification
    sampling_rate : float, optional
        Sampling rate in Hz
    **kwargs
        Additional method-specific parameters
        
    Returns
    -------
    corrected : ndarray
        Baseline-corrected data
    """
    # Methods requiring baseline window
    window_methods = ['mean', 'median', 'mode', 'polynomial', 'percentile']
    
    if method in window_methods:
        if baseline_window is None or sampling_rate is None:
            raise InvalidParameterError(
                f"Method '{method}' requires baseline_window and sampling_rate"
            )
            
        if method == 'mean':
            return baseline_mean(data, baseline_window, sampling_rate, **kwargs)
        elif method == 'median':
            return baseline_median(data, baseline_window, sampling_rate, **kwargs)
        elif method == 'mode':
            return baseline_mode(data, baseline_window, sampling_rate, **kwargs)
        elif method == 'polynomial':
            order = kwargs.get('order', 1)
            return baseline_polynomial(data, baseline_window, sampling_rate, order, **kwargs)
        elif method == 'percentile':
            percentile = kwargs.get('percentile', 10)
            return baseline_percentile(data, baseline_window, sampling_rate, percentile, **kwargs)
            
    elif method == 'als':
        return baseline_als(data, **kwargs)
    elif method == 'rolling':
        return baseline_rolling(data, **kwargs)
    elif method == 'morphological':
        return baseline_morphological(data, **kwargs)
    elif method == 'adaptive':
        if sampling_rate is None:
            raise InvalidParameterError("Adaptive method requires sampling_rate")
        corrected, _ = baseline_adaptive(data, sampling_rate, **kwargs)
        return corrected
    else:
        raise InvalidParameterError(f"Unknown baseline correction method: {method}")


# Helper functions

def _get_baseline_indices(data_length: int,
                         baseline_window: Union[float, Tuple[float, float]],
                         sampling_rate: float) -> np.ndarray:
    """Get indices for baseline window."""
    if isinstance(baseline_window, (int, float)):
        # Duration from start
        n_samples = int(baseline_window * sampling_rate)
        indices = np.arange(min(n_samples, data_length))
    else:
        # Start and end times
        start_idx = int(baseline_window[0] * sampling_rate)
        end_idx = int(baseline_window[1] * sampling_rate)
        indices = np.arange(max(0, start_idx), min(end_idx, data_length))
        
    if len(indices) == 0:
        raise InvalidParameterError("Baseline window contains no samples")
        
    return indices


def _estimate_mode(data: np.ndarray, bins: int = 50) -> float:
    """Estimate mode using histogram."""
    counts, bin_edges = np.histogram(data, bins=bins)
    max_bin = np.argmax(counts)
    mode = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    return mode


def _baseline_poly_1d(data: np.ndarray, 
                     baseline_indices: np.ndarray,
                     order: int) -> np.ndarray:
    """Fit and subtract polynomial baseline from 1D data."""
    # Fit polynomial to baseline region
    baseline_data = data[baseline_indices]
    t_baseline = baseline_indices
    
    coeffs = np.polyfit(t_baseline, baseline_data, order)
    
    # Evaluate over entire signal
    t_full = np.arange(len(data))
    baseline_full = np.polyval(coeffs, t_full)
    
    return data - baseline_full


def _sparse_diff_matrix(n: int, order: int) -> np.ndarray:
    """Create sparse difference matrix for ALS."""
    # Simple implementation - for production use scipy.sparse
    D = np.zeros((n - order, n))
    for i in range(n - order):
        for j in range(order + 1):
            D[i, i + j] = (-1) ** j * np.math.comb(order, j)
    return D


def _sparse_diag_matrix(diag: np.ndarray) -> np.ndarray:
    """Create sparse diagonal matrix for ALS."""
    # Simple implementation - for production use scipy.sparse
    return np.diag(diag)


def _rolling_mean(data: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """Calculate rolling mean."""
    return signal.convolve(data, np.ones(window)/window, mode='same')


def _rolling_median(data: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """Calculate rolling median."""
    from scipy.signal import medfilt
    return medfilt(data, kernel_size=window)


def _rolling_min(data: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """Calculate rolling minimum."""
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(data, size=window, mode='nearest')


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate test signal with baseline drift
    t = np.linspace(0, 10, 1000)
    fs = 100  # Hz
    
    # Signal with different baseline issues
    # 1. Linear drift
    signal_drift = np.sin(2 * np.pi * 2 * t) + 0.5 * t + 2
    
    # 2. Step change
    signal_step = np.sin(2 * np.pi * 2 * t) + 1
    signal_step[500:] += 2
    
    # 3. Exponential drift
    signal_exp = np.sin(2 * np.pi * 2 * t) + 2 * np.exp(0.1 * t)
    
    # Apply different baseline corrections
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Linear drift correction
    corrected_mean = baseline_mean(signal_drift, 0.5, fs)
    corrected_poly = baseline_polynomial(signal_drift, 0.5, fs, order=1)
    
    axes[0, 0].plot(t, signal_drift)
    axes[0, 0].set_title('Linear Drift - Original')
    axes[0, 1].plot(t, corrected_mean)
    axes[0, 1].set_title('Mean Baseline Correction')
    axes[0, 2].plot(t, corrected_poly)
    axes[0, 2].set_title('Polynomial Baseline Correction')
    
    # Step change correction
    corrected_median = baseline_median(signal_step, (0, 0.5), fs)
    corrected_rolling = baseline_rolling(signal_step, 2.0, fs, method='median')
    
    axes[1, 0].plot(t, signal_step)
    axes[1, 0].set_title('Step Change - Original')
    axes[1, 1].plot(t, corrected_median)
    axes[1, 1].set_title('Median Baseline Correction')
    axes[1, 2].plot(t, corrected_rolling)
    axes[1, 2].set_title('Rolling Median Correction')
    
    # Exponential drift correction
    corrected_percentile = baseline_percentile(signal_exp, 0.5, fs, percentile=10)
    corrected_als = baseline_als(signal_exp)
    
    axes[2, 0].plot(t, signal_exp)
    axes[2, 0].set_title('Exponential Drift - Original')
    axes[2, 1].plot(t, corrected_percentile)
    axes[2, 1].set_title('Percentile Baseline Correction')
    axes[2, 2].plot(t, corrected_als)
    axes[2, 2].set_title('ALS Baseline Correction')
    
    # Add zero line for reference
    for ax in axes.flat:
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()