"""
Signal Detrending Utilities
===========================

Common detrending functions for removing slow drifts and trends
from electrophysiology recordings.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import signal, optimize
from typing import Optional, Union, Tuple
import logging

from ...core.exceptions import PreprocessingError, InvalidParameterError

logger = logging.getLogger(__name__)


def detrend_linear(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove linear trend from data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    try:
        return signal.detrend(data, axis=axis, type='linear')
    except Exception as e:
        raise PreprocessingError(f"Failed to apply linear detrending: {e}")


def detrend_constant(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove constant offset (mean) from data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Mean-subtracted data
    """
    try:
        return signal.detrend(data, axis=axis, type='constant')
    except Exception as e:
        raise PreprocessingError(f"Failed to apply constant detrending: {e}")


def detrend_polynomial(data: np.ndarray, order: int, axis: int = -1) -> np.ndarray:
    """
    Remove polynomial trend from data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    order : int
        Polynomial order (1=linear, 2=quadratic, etc.)
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    if order < 1:
        raise InvalidParameterError("Polynomial order must be >= 1")
        
    # Handle different array dimensions
    if data.ndim == 1:
        return _detrend_poly_1d(data, order)
    else:
        # Apply along specified axis
        return np.apply_along_axis(_detrend_poly_1d, axis, data, order)


def _detrend_poly_1d(data: np.ndarray, order: int) -> np.ndarray:
    """
    Remove polynomial trend from 1D data.
    
    Parameters
    ----------
    data : ndarray
        1D input data
    order : int
        Polynomial order
        
    Returns
    -------
    detrended : ndarray
        Detrended 1D data
    """
    try:
        # Create time vector
        t = np.arange(len(data))
        
        # Fit polynomial
        coeffs = np.polyfit(t, data, order)
        
        # Evaluate polynomial
        trend = np.polyval(coeffs, t)
        
        # Remove trend
        return data - trend
        
    except Exception as e:
        raise PreprocessingError(f"Failed to fit polynomial of order {order}: {e}")


def detrend_savgol(data: np.ndarray, 
                   window_length: Optional[int] = None,
                   polyorder: int = 3,
                   axis: int = -1) -> np.ndarray:
    """
    Remove trend using Savitzky-Golay filter.
    
    This method fits a polynomial to local windows and subtracts
    the smoothed result as the trend.
    
    Parameters
    ----------
    data : ndarray
        Input data
    window_length : int, optional
        Length of the filter window. If None, uses len(data)//10
    polyorder : int
        Order of the polynomial to fit
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    # Determine window length
    if window_length is None:
        window_length = data.shape[axis] // 10
        # Ensure odd window length
        if window_length % 2 == 0:
            window_length += 1
            
    # Ensure minimum window length
    window_length = max(window_length, polyorder + 2)
    
    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1
        
    try:
        # Calculate trend using Savitzky-Golay filter
        trend = signal.savgol_filter(data, window_length, polyorder, axis=axis)
        
        # Remove trend
        return data - trend
        
    except Exception as e:
        raise PreprocessingError(f"Failed to apply Savitzky-Golay detrending: {e}")


def detrend_breakpoints(data: np.ndarray,
                       breakpoints: Union[int, np.ndarray],
                       axis: int = -1) -> np.ndarray:
    """
    Remove piecewise linear trend with breakpoints.
    
    Parameters
    ----------
    data : ndarray
        Input data
    breakpoints : int or array-like
        Number of breakpoints or array of breakpoint indices
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    if isinstance(breakpoints, int):
        # Create evenly spaced breakpoints
        n_points = data.shape[axis]
        bp_indices = np.linspace(0, n_points-1, breakpoints+2, dtype=int)[1:-1]
    else:
        bp_indices = np.asarray(breakpoints, dtype=int)
        
    # Handle different dimensions
    if data.ndim == 1:
        return _detrend_breakpoints_1d(data, bp_indices)
    else:
        return np.apply_along_axis(_detrend_breakpoints_1d, axis, data, bp_indices)


def _detrend_breakpoints_1d(data: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """
    Remove piecewise linear trend from 1D data.
    
    Parameters
    ----------
    data : ndarray
        1D input data
    breakpoints : ndarray
        Breakpoint indices
        
    Returns
    -------
    detrended : ndarray
        Detrended 1D data
    """
    try:
        n_points = len(data)
        
        # Add endpoints
        all_breakpoints = np.concatenate([[0], breakpoints, [n_points-1]])
        
        # Initialize trend
        trend = np.zeros_like(data)
        
        # Fit linear segments
        for i in range(len(all_breakpoints)-1):
            start_idx = all_breakpoints[i]
            end_idx = all_breakpoints[i+1] + 1
            
            # Get segment
            segment = data[start_idx:end_idx]
            t_segment = np.arange(len(segment))
            
            # Fit linear trend to segment
            coeffs = np.polyfit(t_segment, segment, 1)
            trend[start_idx:end_idx] = np.polyval(coeffs, t_segment)
            
        return data - trend
        
    except Exception as e:
        raise PreprocessingError(f"Failed to apply breakpoint detrending: {e}")


def detrend_exponential(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove exponential trend from data.
    
    Fits and removes trend of the form: a * exp(b * t) + c
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    if data.ndim == 1:
        return _detrend_exponential_1d(data)
    else:
        return np.apply_along_axis(_detrend_exponential_1d, axis, data)


def _detrend_exponential_1d(data: np.ndarray) -> np.ndarray:
    """
    Remove exponential trend from 1D data.
    
    Parameters
    ----------
    data : ndarray
        1D input data
        
    Returns
    -------
    detrended : ndarray
        Detrended 1D data
    """
    try:
        t = np.arange(len(data))
        
        # Define exponential function
        def exp_func(t, a, b, c):
            return a * np.exp(b * t) + c
            
        # Initial guess
        a_guess = np.ptp(data) / 2
        c_guess = np.mean(data)
        b_guess = np.log(2) / len(data)  # Reasonable decay
        
        # Fit exponential
        popt, _ = optimize.curve_fit(
            exp_func, t, data, 
            p0=[a_guess, b_guess, c_guess],
            maxfev=5000
        )
        
        # Calculate trend
        trend = exp_func(t, *popt)
        
        return data - trend
        
    except Exception as e:
        logger.warning(f"Exponential fit failed, falling back to polynomial: {e}")
        # Fall back to polynomial detrending
        return _detrend_poly_1d(data, order=2)


def detrend_adaptive(data: np.ndarray,
                    method: str = 'auto',
                    axis: int = -1) -> Tuple[np.ndarray, str]:
    """
    Apply adaptive detrending based on data characteristics.
    
    Automatically selects the best detrending method based on
    analysis of the signal characteristics.
    
    Parameters
    ----------
    data : ndarray
        Input data
    method : str
        Detrending method or 'auto' for automatic selection
    axis : int
        Axis along which to detrend
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    method_used : str
        The detrending method that was applied
    """
    if method != 'auto':
        # Use specified method
        if method == 'linear':
            return detrend_linear(data, axis), 'linear'
        elif method == 'constant':
            return detrend_constant(data, axis), 'constant'
        elif method == 'polynomial':
            return detrend_polynomial(data, order=2, axis=axis), 'polynomial'
        elif method == 'savgol':
            return detrend_savgol(data, axis=axis), 'savgol'
        else:
            raise InvalidParameterError(f"Unknown detrending method: {method}")
            
    # Auto-detect best method
    if data.ndim == 1:
        method_used = _detect_trend_type_1d(data)
    else:
        # Use representative slice
        slice_idx = data.shape[axis] // 2
        slice_data = np.take(data, slice_idx, axis=axis)
        method_used = _detect_trend_type_1d(slice_data.flatten())
        
    logger.info(f"Auto-detected trend type: {method_used}")
    
    # Apply detected method
    if method_used == 'linear':
        return detrend_linear(data, axis), method_used
    elif method_used == 'polynomial':
        return detrend_polynomial(data, order=2, axis=axis), method_used
    elif method_used == 'exponential':
        return detrend_exponential(data, axis), method_used
    else:  # constant
        return detrend_constant(data, axis), method_used


def _detect_trend_type_1d(data: np.ndarray) -> str:
    """
    Detect the type of trend in 1D data.
    
    Parameters
    ----------
    data : ndarray
        1D input data
        
    Returns
    -------
    trend_type : str
        Detected trend type
    """
    t = np.arange(len(data))
    
    # Fit different models and compare residuals
    residuals = {}
    
    # Constant (mean)
    residuals['constant'] = np.sum((data - np.mean(data))**2)
    
    # Linear
    try:
        coeffs = np.polyfit(t, data, 1)
        linear_fit = np.polyval(coeffs, t)
        residuals['linear'] = np.sum((data - linear_fit)**2)
    except:
        residuals['linear'] = np.inf
        
    # Polynomial (quadratic)
    try:
        coeffs = np.polyfit(t, data, 2)
        poly_fit = np.polyval(coeffs, t)
        residuals['polynomial'] = np.sum((data - poly_fit)**2)
    except:
        residuals['polynomial'] = np.inf
        
    # Exponential
    try:
        def exp_func(t, a, b, c):
            return a * np.exp(b * t) + c
        popt, _ = optimize.curve_fit(exp_func, t, data, maxfev=1000)
        exp_fit = exp_func(t, *popt)
        residuals['exponential'] = np.sum((data - exp_fit)**2)
    except:
        residuals['exponential'] = np.inf
        
    # Select method with lowest residual
    best_method = min(residuals, key=residuals.get)
    
    # Check if improvement is significant
    const_residual = residuals['constant']
    best_residual = residuals[best_method]
    
    # If improvement is less than 5%, use simpler method
    if best_residual > 0.95 * const_residual:
        return 'constant'
        
    return best_method


def detrend(data: np.ndarray,
           type: str = 'linear',
           axis: int = -1,
           **kwargs) -> np.ndarray:
    """
    General detrending function with multiple methods.
    
    Parameters
    ----------
    data : ndarray
        Input data
    type : str
        Detrending type: 'constant', 'linear', 'polynomial',
        'savgol', 'breakpoints', 'exponential', 'adaptive'
    axis : int
        Axis along which to detrend
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    detrended : ndarray
        Detrended data
    """
    if type == 'constant':
        return detrend_constant(data, axis)
    elif type == 'linear':
        return detrend_linear(data, axis)
    elif type == 'polynomial':
        order = kwargs.get('order', 2)
        return detrend_polynomial(data, order, axis)
    elif type == 'savgol':
        window_length = kwargs.get('window_length', None)
        polyorder = kwargs.get('polyorder', 3)
        return detrend_savgol(data, window_length, polyorder, axis)
    elif type == 'breakpoints':
        breakpoints = kwargs.get('breakpoints', 5)
        return detrend_breakpoints(data, breakpoints, axis)
    elif type == 'exponential':
        return detrend_exponential(data, axis)
    elif type == 'adaptive':
        detrended, _ = detrend_adaptive(data, 'auto', axis)
        return detrended
    else:
        raise InvalidParameterError(f"Unknown detrending type: {type}")


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate test signal with various trends
    t = np.linspace(0, 10, 1000)
    
    # Signal with linear trend
    linear_trend = 0.5 * t
    signal_linear = np.sin(2 * np.pi * 2 * t) + linear_trend + 0.1 * np.random.randn(len(t))
    
    # Signal with exponential trend
    exp_trend = 2 * np.exp(0.1 * t)
    signal_exp = np.sin(2 * np.pi * 2 * t) + exp_trend + 0.1 * np.random.randn(len(t))
    
    # Signal with polynomial trend
    poly_trend = 0.1 * t**2 - 0.5 * t + 1
    signal_poly = np.sin(2 * np.pi * 2 * t) + poly_trend + 0.1 * np.random.randn(len(t))
    
    # Apply different detrending methods
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Linear trend
    axes[0, 0].plot(t, signal_linear, 'b-', alpha=0.7, label='Original')
    axes[0, 0].plot(t, linear_trend, 'r--', label='Trend')
    axes[0, 0].set_title('Linear Trend - Original')
    axes[0, 0].legend()
    
    detrended_linear = detrend_linear(signal_linear)
    axes[0, 1].plot(t, detrended_linear, 'g-')
    axes[0, 1].set_title('Linear Trend - Detrended')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Exponential trend
    axes[1, 0].plot(t, signal_exp, 'b-', alpha=0.7, label='Original')
    axes[1, 0].plot(t, exp_trend, 'r--', label='Trend')
    axes[1, 0].set_title('Exponential Trend - Original')
    axes[1, 0].legend()
    
    detrended_exp = detrend_exponential(signal_exp)
    axes[1, 1].plot(t, detrended_exp, 'g-')
    axes[1, 1].set_title('Exponential Trend - Detrended')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Polynomial trend
    axes[2, 0].plot(t, signal_poly, 'b-', alpha=0.7, label='Original')
    axes[2, 0].plot(t, poly_trend, 'r--', label='Trend')
    axes[2, 0].set_title('Polynomial Trend - Original')
    axes[2, 0].legend()
    
    detrended_poly = detrend_polynomial(signal_poly, order=2)
    axes[2, 1].plot(t, detrended_poly, 'g-')
    axes[2, 1].set_title('Polynomial Trend - Detrended')
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test adaptive detrending
    print("\nAdaptive Detrending Test:")
    for signal, name in [(signal_linear, 'Linear'), 
                        (signal_exp, 'Exponential'),
                        (signal_poly, 'Polynomial')]:
        _, method = detrend_adaptive(signal)
        print(f"{name} signal: detected {method} trend")