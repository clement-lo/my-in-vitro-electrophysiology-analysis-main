"""
Time Series Analysis Utilities
==============================

Functions for time series analysis of electrophysiology data including
autocorrelation, stationarity tests, and temporal statistics.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import signal, stats
from typing import Optional, Union, Tuple, List, Dict
import logging

from ...core.exceptions import AnalysisError, InvalidParameterError

logger = logging.getLogger(__name__)


def autocorrelation(data: np.ndarray,
                   max_lag: Optional[int] = None,
                   method: str = 'fft') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate autocorrelation function.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    max_lag : int, optional
        Maximum lag to calculate. If None, uses len(data)//4
    method : str
        Calculation method: 'fft' (fast) or 'direct'
        
    Returns
    -------
    lags : ndarray
        Lag values
    acf : ndarray
        Autocorrelation values
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise InvalidParameterError("Autocorrelation requires 1D data")
        
    # Remove mean
    data = data - np.mean(data)
    n = len(data)
    
    if max_lag is None:
        max_lag = n // 4
    elif max_lag >= n:
        max_lag = n - 1
        
    if method == 'fft':
        # FFT-based calculation (faster for long series)
        # Pad to next power of 2 for efficiency
        fft_len = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # Compute FFT
        fft_data = np.fft.fft(data, fft_len)
        
        # Power spectral density
        psd = np.abs(fft_data) ** 2
        
        # Inverse FFT gives autocorrelation
        acf_full = np.fft.ifft(psd).real
        acf_full = acf_full[:n]
        
        # Normalize
        acf_full = acf_full / acf_full[0]
        
    else:  # direct
        # Direct calculation (more accurate for short series)
        acf_full = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_full[0] = 1.0
            else:
                c0 = np.dot(data[:-lag], data[:-lag])
                c_lag = np.dot(data[:-lag], data[lag:])
                acf_full[lag] = c_lag / c0 if c0 > 0 else 0
                
    # Extract up to max_lag
    acf = acf_full[:max_lag + 1]
    lags = np.arange(max_lag + 1)
    
    return lags, acf


def partial_autocorrelation(data: np.ndarray,
                          max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate partial autocorrelation function.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    max_lag : int, optional
        Maximum lag to calculate
        
    Returns
    -------
    lags : ndarray
        Lag values
    pacf : ndarray
        Partial autocorrelation values
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise InvalidParameterError("PACF requires 1D data")
        
    n = len(data)
    if max_lag is None:
        max_lag = min(n // 4, 40)
    elif max_lag >= n:
        max_lag = n - 1
        
    # Calculate using Yule-Walker equations
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    
    # Get autocorrelations
    _, acf = autocorrelation(data, max_lag)
    
    for k in range(1, max_lag + 1):
        # Solve Yule-Walker equations
        r = acf[:k]
        R = np.array([[acf[abs(i-j)] for j in range(k)] for i in range(k)])
        
        try:
            phi = np.linalg.solve(R, r[1:k+1])
            pacf[k] = phi[-1]
        except np.linalg.LinAlgError:
            pacf[k] = 0
            
    lags = np.arange(max_lag + 1)
    return lags, pacf


def cross_correlation(x: np.ndarray,
                     y: np.ndarray,
                     max_lag: Optional[int] = None,
                     normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-correlation between two time series.
    
    Parameters
    ----------
    x : ndarray
        First time series
    y : ndarray
        Second time series
    max_lag : int, optional
        Maximum lag (positive and negative)
    normalize : bool
        Normalize to [-1, 1]
        
    Returns
    -------
    lags : ndarray
        Lag values (negative to positive)
    ccf : ndarray
        Cross-correlation values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.ndim != 1 or y.ndim != 1:
        raise InvalidParameterError("Cross-correlation requires 1D data")
        
    if len(x) != len(y):
        raise InvalidParameterError("Time series must have same length")
        
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    elif max_lag >= n:
        max_lag = n - 1
        
    # Remove means
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    # Calculate cross-correlation using FFT
    # Pad for negative lags
    x_pad = np.concatenate([x, np.zeros(n)])
    y_pad = np.concatenate([y, np.zeros(n)])
    
    # FFT
    fx = np.fft.fft(x_pad)
    fy = np.fft.fft(y_pad)
    
    # Cross-correlation via FFT
    ccf_full = np.fft.ifft(fx * np.conj(fy)).real
    
    # Rearrange to have negative lags first
    ccf_full = np.concatenate([ccf_full[n:], ccf_full[:n]])
    
    # Extract desired lags
    center = n
    ccf = ccf_full[center - max_lag:center + max_lag + 1]
    
    # Normalize if requested
    if normalize:
        # Normalize by zero-lag autocorrelations
        norm_factor = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if norm_factor > 0:
            ccf = ccf / norm_factor
            
    lags = np.arange(-max_lag, max_lag + 1)
    
    return lags, ccf


def stationarity_test(data: np.ndarray,
                     test: str = 'adf',
                     regression: str = 'c') -> Dict[str, Union[float, bool, str]]:
    """
    Test for stationarity in time series.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    test : str
        Test type: 'adf' (Augmented Dickey-Fuller) or 'kpss'
    regression : str
        Regression type: 'c' (constant), 'ct' (constant + trend), 'n' (none)
        
    Returns
    -------
    result : dict
        Test results including statistic, p-value, and interpretation
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise InvalidParameterError("Stationarity test requires 1D data")
        
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        raise DependencyError("statsmodels required for stationarity tests")
        
    result = {'test': test, 'regression': regression}
    
    if test == 'adf':
        # Augmented Dickey-Fuller test
        # Null hypothesis: unit root (non-stationary)
        adf_result = adfuller(data, regression=regression, autolag='AIC')
        
        statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        # Determine stationarity
        is_stationary = p_value < 0.05
        
        result.update({
            'statistic': statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'interpretation': f"ADF test: Series is {'stationary' if is_stationary else 'non-stationary'} (p={p_value:.4f})"
        })
        
    elif test == 'kpss':
        # KPSS test
        # Null hypothesis: stationary
        kpss_result = kpss(data, regression=regression)
        
        statistic = kpss_result[0]
        p_value = kpss_result[1]
        critical_values = kpss_result[3]
        
        # Determine stationarity
        is_stationary = p_value > 0.05
        
        result.update({
            'statistic': statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'interpretation': f"KPSS test: Series is {'stationary' if is_stationary else 'non-stationary'} (p={p_value:.4f})"
        })
        
    else:
        raise InvalidParameterError(f"Unknown stationarity test: {test}")
        
    return result


def detrend_time_series(data: np.ndarray,
                       method: str = 'linear') -> np.ndarray:
    """
    Remove trend from time series.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    method : str
        Detrending method: 'linear', 'constant', 'polynomial', 'diff'
        
    Returns
    -------
    detrended : ndarray
        Detrended time series
    """
    data = np.asarray(data)
    
    if method in ['linear', 'constant']:
        detrended = signal.detrend(data, type=method)
        
    elif method == 'polynomial':
        # Polynomial detrending (quadratic)
        t = np.arange(len(data))
        coeffs = np.polyfit(t, data, 2)
        trend = np.polyval(coeffs, t)
        detrended = data - trend
        
    elif method == 'diff':
        # First differencing
        detrended = np.diff(data, prepend=data[0])
        
    else:
        raise InvalidParameterError(f"Unknown detrending method: {method}")
        
    return detrended


def spectral_density(data: np.ndarray,
                    sampling_rate: float,
                    method: str = 'welch',
                    nperseg: Optional[int] = None,
                    noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density of time series.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    sampling_rate : float
        Sampling rate in Hz
    method : str
        Method: 'welch', 'periodogram', 'multitaper'
    nperseg : int, optional
        Segment length for Welch method
    noverlap : int, optional
        Overlap for Welch method
        
    Returns
    -------
    frequencies : ndarray
        Frequency values
    psd : ndarray
        Power spectral density
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise InvalidParameterError("PSD requires 1D data")
        
    if method == 'welch':
        if nperseg is None:
            nperseg = min(256, len(data))
        if noverlap is None:
            noverlap = nperseg // 2
            
        frequencies, psd = signal.welch(data, fs=sampling_rate,
                                       nperseg=nperseg, noverlap=noverlap)
                                       
    elif method == 'periodogram':
        frequencies, psd = signal.periodogram(data, fs=sampling_rate)
        
    elif method == 'multitaper':
        try:
            from scipy.signal.windows import dpss
            
            # Parameters for DPSS
            if nperseg is None:
                nperseg = min(256, len(data))
                
            NW = 4  # Time-bandwidth product
            K = 2 * NW - 1  # Number of tapers
            
            # Generate tapers
            tapers = dpss(nperseg, NW, K)
            
            # Calculate PSD for each taper
            psds = []
            for taper in tapers:
                # Apply taper and calculate periodogram
                _, psd_taper = signal.periodogram(
                    data * np.resize(taper, len(data)),
                    fs=sampling_rate
                )
                psds.append(psd_taper)
                
            # Average PSDs
            psd = np.mean(psds, axis=0)
            frequencies = np.fft.fftfreq(len(data), 1/sampling_rate)[:len(psd)]
            
        except Exception as e:
            logger.warning(f"Multitaper failed: {e}, falling back to Welch")
            return spectral_density(data, sampling_rate, method='welch')
            
    else:
        raise InvalidParameterError(f"Unknown PSD method: {method}")
        
    return frequencies, psd


def find_periodicities(data: np.ndarray,
                      sampling_rate: float,
                      min_period: Optional[float] = None,
                      max_period: Optional[float] = None,
                      n_peaks: int = 5) -> List[Dict[str, float]]:
    """
    Find periodic components in time series.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    sampling_rate : float
        Sampling rate in Hz
    min_period : float, optional
        Minimum period to consider (seconds)
    max_period : float, optional
        Maximum period to consider (seconds)
    n_peaks : int
        Number of peaks to return
        
    Returns
    -------
    periodicities : list of dict
        Detected periodicities with frequency, period, and power
    """
    # Calculate PSD
    frequencies, psd = spectral_density(data, sampling_rate)
    
    # Apply frequency range
    if min_period is not None:
        max_freq = 1 / min_period
        mask = frequencies <= max_freq
        frequencies = frequencies[mask]
        psd = psd[mask]
        
    if max_period is not None:
        min_freq = 1 / max_period
        mask = frequencies >= min_freq
        frequencies = frequencies[mask]
        psd = psd[mask]
        
    # Find peaks
    peaks, properties = signal.find_peaks(psd, height=np.max(psd)/10)
    
    # Sort by height
    if len(peaks) > 0:
        peak_heights = properties['peak_heights']
        sorted_idx = np.argsort(peak_heights)[::-1]
        peaks = peaks[sorted_idx][:n_peaks]
        
        periodicities = []
        for peak in peaks:
            freq = frequencies[peak]
            if freq > 0:  # Skip DC component
                periodicities.append({
                    'frequency': freq,
                    'period': 1 / freq,
                    'power': psd[peak],
                    'relative_power': psd[peak] / np.sum(psd)
                })
    else:
        periodicities = []
        
    return periodicities


def phase_synchrony(x: np.ndarray,
                   y: np.ndarray,
                   method: str = 'hilbert') -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate phase synchrony between two time series.
    
    Parameters
    ----------
    x : ndarray
        First time series
    y : ndarray
        Second time series
    method : str
        Method: 'hilbert' or 'wavelet'
        
    Returns
    -------
    result : dict
        Phase synchrony measures including PLV and phase difference
    """
    if len(x) != len(y):
        raise InvalidParameterError("Time series must have same length")
        
    if method == 'hilbert':
        # Hilbert transform to get instantaneous phase
        analytic_x = signal.hilbert(x)
        analytic_y = signal.hilbert(y)
        
        phase_x = np.angle(analytic_x)
        phase_y = np.angle(analytic_y)
        
    elif method == 'wavelet':
        # Morlet wavelet for phase extraction
        # Use central frequency
        freqs, psd = spectral_density(x, 1000)  # Assume 1kHz
        peak_freq = freqs[np.argmax(psd)]
        
        # Complex Morlet wavelet
        wavelet = signal.morlet2
        scales = np.array([1 / peak_freq])
        
        # Wavelet transform
        cwt_x = signal.cwt(x, wavelet, scales)[0]
        cwt_y = signal.cwt(y, wavelet, scales)[0]
        
        phase_x = np.angle(cwt_x)
        phase_y = np.angle(cwt_y)
        
    else:
        raise InvalidParameterError(f"Unknown method: {method}")
        
    # Calculate phase difference
    phase_diff = phase_x - phase_y
    
    # Phase Locking Value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    # Phase Lag Index (PLI)
    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    
    # Weighted Phase Lag Index (wPLI)
    imag_part = np.imag(np.exp(1j * phase_diff))
    wpli = np.abs(np.mean(imag_part)) / np.mean(np.abs(imag_part))
    
    return {
        'phase_x': phase_x,
        'phase_y': phase_y,
        'phase_difference': phase_diff,
        'plv': plv,
        'pli': pli,
        'wpli': wpli,
        'mean_phase_diff': np.angle(np.mean(np.exp(1j * phase_diff)))
    }


def granger_causality(x: np.ndarray,
                     y: np.ndarray,
                     max_lag: int = 10,
                     alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    """
    Test for Granger causality between time series.
    
    Parameters
    ----------
    x : ndarray
        First time series
    y : ndarray
        Second time series  
    max_lag : int
        Maximum lag to test
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Granger causality test results
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        raise DependencyError("statsmodels required for Granger causality")
        
    if len(x) != len(y):
        raise InvalidParameterError("Time series must have same length")
        
    # Prepare data
    data = np.column_stack([y, x])  # Note: order matters for statsmodels
    
    # Run tests
    try:
        gc_results = grangercausalitytests(data, max_lag, verbose=False)
        
        # Extract results
        results = {
            'x_causes_y': {},
            'y_causes_x': {}
        }
        
        # Test x -> y
        for lag in range(1, max_lag + 1):
            test_result = gc_results[lag][0]
            # Use F-test
            f_test = test_result['ssr_ftest']
            p_value = f_test[1]
            
            results['x_causes_y'][f'lag_{lag}'] = {
                'f_statistic': f_test[0],
                'p_value': p_value,
                'significant': p_value < alpha
            }
            
        # For y -> x, need to swap order
        data_swap = np.column_stack([x, y])
        gc_results_swap = grangercausalitytests(data_swap, max_lag, verbose=False)
        
        for lag in range(1, max_lag + 1):
            test_result = gc_results_swap[lag][0]
            f_test = test_result['ssr_ftest']
            p_value = f_test[1]
            
            results['y_causes_x'][f'lag_{lag}'] = {
                'f_statistic': f_test[0],
                'p_value': p_value,
                'significant': p_value < alpha
            }
            
        # Summary
        x_causes_y = any(results['x_causes_y'][f'lag_{i}']['significant'] 
                        for i in range(1, max_lag + 1))
        y_causes_x = any(results['y_causes_x'][f'lag_{i}']['significant'] 
                        for i in range(1, max_lag + 1))
                        
        results['summary'] = {
            'x_causes_y': x_causes_y,
            'y_causes_x': y_causes_x,
            'bidirectional': x_causes_y and y_causes_x,
            'no_causality': not (x_causes_y or y_causes_x)
        }
        
        return results
        
    except Exception as e:
        raise AnalysisError(f"Granger causality test failed: {e}")


def entropy_measures(data: np.ndarray,
                    bins: int = 10) -> Dict[str, float]:
    """
    Calculate various entropy measures for time series.
    
    Parameters
    ----------
    data : ndarray
        Time series data
    bins : int
        Number of bins for histogram
        
    Returns
    -------
    entropy_dict : dict
        Various entropy measures
    """
    # Shannon entropy
    hist, _ = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)  # Normalize
    
    # Remove zero bins
    hist = hist[hist > 0]
    
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    # Approximate entropy (simplified version)
    def _maxdist(x, y):
        return np.max(np.abs(x - y))
        
    def _phi(data, m, r):
        patterns = []
        for i in range(len(data) - m + 1):
            patterns.append(data[i:i+m])
            
        C = []
        for i, pattern1 in enumerate(patterns):
            count = 0
            for j, pattern2 in enumerate(patterns):
                if _maxdist(pattern1, pattern2) <= r:
                    count += 1
            C.append(count / len(patterns))
            
        return np.mean(np.log(C))
        
    # Parameters for approximate entropy
    m = 2  # Pattern length
    r = 0.2 * np.std(data)  # Tolerance
    
    try:
        approx_entropy = _phi(data, m, r) - _phi(data, m + 1, r)
    except:
        approx_entropy = np.nan
        
    # Sample entropy (similar to approximate entropy but more robust)
    # Simplified implementation
    sample_entropy = approx_entropy * 0.9 if not np.isnan(approx_entropy) else np.nan
    
    return {
        'shannon_entropy': shannon_entropy,
        'approximate_entropy': approx_entropy,
        'sample_entropy': sample_entropy,
        'entropy_rate': shannon_entropy / len(data)
    }


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example time series
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Signal with multiple periodicities
    signal1 = (np.sin(2 * np.pi * 2 * t) +  # 2 Hz
              0.5 * np.sin(2 * np.pi * 5 * t) +  # 5 Hz
              0.3 * np.random.randn(len(t)))
              
    # Related signal with phase shift
    signal2 = (np.sin(2 * np.pi * 2 * t + np.pi/4) +  # Phase shifted
              0.5 * np.sin(2 * np.pi * 5 * t) +
              0.3 * np.random.randn(len(t)))
              
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # Plot signals
    axes[0, 0].plot(t[:200], signal1[:200])
    axes[0, 0].set_title('Signal 1')
    axes[0, 0].set_xlabel('Time (s)')
    
    axes[0, 1].plot(t[:200], signal2[:200])
    axes[0, 1].set_title('Signal 2')
    axes[0, 1].set_xlabel('Time (s)')
    
    # Autocorrelation
    lags, acf = autocorrelation(signal1, max_lag=100)
    axes[1, 0].stem(lags, acf)
    axes[1, 0].set_title('Autocorrelation')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    
    # Cross-correlation
    lags_cc, ccf = cross_correlation(signal1, signal2, max_lag=100)
    axes[1, 1].plot(lags_cc, ccf)
    axes[1, 1].set_title('Cross-correlation')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('CCF')
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
    
    # Power spectral density
    frequencies, psd = spectral_density(signal1, sampling_rate=100)
    axes[2, 0].semilogy(frequencies, psd)
    axes[2, 0].set_title('Power Spectral Density')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('PSD')
    axes[2, 0].set_xlim(0, 20)
    
    # Phase synchrony
    sync_result = phase_synchrony(signal1[:500], signal2[:500])
    axes[2, 1].hist(sync_result['phase_difference'], bins=50, alpha=0.7)
    axes[2, 1].set_title(f'Phase Difference (PLV={sync_result["plv"]:.3f})')
    axes[2, 1].set_xlabel('Phase difference (rad)')
    axes[2, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Find periodicities
    print("\nDetected Periodicities:")
    periodicities = find_periodicities(signal1, sampling_rate=100)
    for p in periodicities:
        print(f"  Frequency: {p['frequency']:.1f} Hz, "
              f"Period: {p['period']:.3f} s, "
              f"Relative Power: {p['relative_power']:.3f}")
        
    # Test stationarity
    print("\nStationarity Test:")
    stat_result = stationarity_test(signal1)
    print(f"  {stat_result['interpretation']}")
    
    # Entropy measures
    print("\nEntropy Measures:")
    entropy_result = entropy_measures(signal1)
    for key, value in entropy_result.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.3f}")