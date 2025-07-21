"""
Descriptive Statistics Utilities
================================

Functions for calculating descriptive statistics of electrophysiology data
including central tendency, variability, and distribution characteristics.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple, Dict, List
import logging

from ...core.exceptions import AnalysisError, InvalidParameterError

logger = logging.getLogger(__name__)


def calculate_basic_stats(data: np.ndarray,
                         axis: Optional[int] = None,
                         ddof: int = 1) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate basic descriptive statistics.
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to compute statistics
    ddof : int
        Degrees of freedom for standard deviation
        
    Returns
    -------
    stats_dict : dict
        Dictionary containing:
        - mean: Mean value
        - median: Median value
        - std: Standard deviation
        - var: Variance
        - sem: Standard error of mean
        - min: Minimum value
        - max: Maximum value
        - range: Range (max - min)
        - cv: Coefficient of variation
    """
    # Remove NaN values if present
    if np.any(np.isnan(data)):
        logger.warning("Data contains NaN values, they will be ignored")
        if axis is None:
            data = data[~np.isnan(data)]
        else:
            # Use nanmean, nanstd, etc.
            return {
                'mean': np.nanmean(data, axis=axis),
                'median': np.nanmedian(data, axis=axis),
                'std': np.nanstd(data, axis=axis, ddof=ddof),
                'var': np.nanvar(data, axis=axis, ddof=ddof),
                'sem': np.nanstd(data, axis=axis, ddof=ddof) / 
                       np.sqrt(np.sum(~np.isnan(data), axis=axis)),
                'min': np.nanmin(data, axis=axis),
                'max': np.nanmax(data, axis=axis),
                'range': np.nanmax(data, axis=axis) - np.nanmin(data, axis=axis),
                'cv': np.nanstd(data, axis=axis, ddof=ddof) / 
                      np.nanmean(data, axis=axis)
            }
    
    mean_val = np.mean(data, axis=axis)
    std_val = np.std(data, axis=axis, ddof=ddof)
    
    # Sample size for SEM
    if axis is None:
        n = len(data)
    else:
        n = data.shape[axis]
    
    stats_dict = {
        'mean': mean_val,
        'median': np.median(data, axis=axis),
        'std': std_val,
        'var': np.var(data, axis=axis, ddof=ddof),
        'sem': std_val / np.sqrt(n),
        'min': np.min(data, axis=axis),
        'max': np.max(data, axis=axis),
        'range': np.ptp(data, axis=axis),
        'cv': std_val / mean_val if np.all(mean_val != 0) else np.inf
    }
    
    return stats_dict


def calculate_percentiles(data: np.ndarray,
                         percentiles: List[float] = [5, 25, 50, 75, 95],
                         axis: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate percentiles of data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    percentiles : list of float
        Percentile values to calculate (0-100)
    axis : int, optional
        Axis along which to compute percentiles
        
    Returns
    -------
    percentile_dict : dict
        Dictionary mapping percentile values to results
    """
    percentile_dict = {}
    
    for p in percentiles:
        if not 0 <= p <= 100:
            raise InvalidParameterError(f"Percentile {p} must be between 0 and 100")
        
        percentile_dict[f'p{p}'] = np.percentile(data, p, axis=axis)
        
    # Add IQR
    if 25 in percentiles and 75 in percentiles:
        percentile_dict['iqr'] = percentile_dict['p75'] - percentile_dict['p25']
        
    return percentile_dict


def calculate_distribution_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics related to data distribution shape.
    
    Parameters
    ----------
    data : ndarray
        Input data (1D)
        
    Returns
    -------
    dist_stats : dict
        Dictionary containing:
        - skewness: Measure of asymmetry
        - kurtosis: Measure of tail heaviness
        - jarque_bera: Test statistic for normality
        - jarque_bera_p: p-value for normality test
        - shapiro_w: Shapiro-Wilk test statistic
        - shapiro_p: Shapiro-Wilk p-value
    """
    if data.ndim != 1:
        data = data.flatten()
        
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    if len(data) < 8:
        logger.warning("Not enough data points for distribution statistics")
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'jarque_bera': np.nan,
            'jarque_bera_p': np.nan,
            'shapiro_w': np.nan,
            'shapiro_p': np.nan
        }
    
    # Calculate skewness and kurtosis
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    
    # Normality tests
    jb_stat, jb_p = stats.jarque_bera(data)
    
    # Shapiro-Wilk test (for smaller samples)
    if len(data) <= 5000:
        sw_stat, sw_p = stats.shapiro(data)
    else:
        sw_stat, sw_p = np.nan, np.nan
        
    return {
        'skewness': skew,
        'kurtosis': kurt,
        'jarque_bera': jb_stat,
        'jarque_bera_p': jb_p,
        'shapiro_w': sw_stat,
        'shapiro_p': sw_p
    }


def calculate_mad(data: np.ndarray,
                 axis: Optional[int] = None,
                 center: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
    """
    Calculate Median Absolute Deviation (MAD).
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to compute MAD
    center : ndarray, optional
        Center point (default: median)
        
    Returns
    -------
    mad : float or ndarray
        Median absolute deviation
    """
    if center is None:
        center = np.median(data, axis=axis, keepdims=True)
        
    return np.median(np.abs(data - center), axis=axis)


def calculate_robust_stats(data: np.ndarray,
                          axis: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate robust statistics less sensitive to outliers.
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to compute statistics
        
    Returns
    -------
    robust_stats : dict
        Dictionary containing:
        - median: Median (robust center)
        - mad: Median absolute deviation
        - trimmed_mean: 10% trimmed mean
        - winsorized_mean: 10% winsorized mean
        - iqr: Interquartile range
    """
    # Basic robust statistics
    median = np.median(data, axis=axis)
    mad = calculate_mad(data, axis=axis)
    
    # IQR
    q1 = np.percentile(data, 25, axis=axis)
    q3 = np.percentile(data, 75, axis=axis)
    iqr = q3 - q1
    
    # Trimmed and winsorized means
    if axis is None:
        trimmed = stats.trim_mean(data, 0.1)
        winsorized = np.mean(stats.mstats.winsorize(data, limits=0.1))
    else:
        # Apply along axis
        trimmed = np.apply_along_axis(lambda x: stats.trim_mean(x, 0.1), 
                                     axis, data)
        winsorized = np.apply_along_axis(
            lambda x: np.mean(stats.mstats.winsorize(x, limits=0.1)), 
            axis, data)
        
    return {
        'median': median,
        'mad': mad,
        'trimmed_mean': trimmed,
        'winsorized_mean': winsorized,
        'iqr': iqr
    }


def calculate_ci(data: np.ndarray,
                confidence: float = 0.95,
                axis: Optional[int] = None,
                method: str = 'normal') -> Tuple[Union[float, np.ndarray], 
                                                Union[float, np.ndarray]]:
    """
    Calculate confidence interval.
    
    Parameters
    ----------
    data : ndarray
        Input data
    confidence : float
        Confidence level (0-1)
    axis : int, optional
        Axis along which to compute CI
    method : str
        Method for CI calculation: 'normal', 'bootstrap', 'percentile'
        
    Returns
    -------
    ci_low : float or ndarray
        Lower confidence interval
    ci_high : float or ndarray
        Upper confidence interval
    """
    if not 0 < confidence < 1:
        raise InvalidParameterError("Confidence must be between 0 and 1")
        
    alpha = 1 - confidence
    
    if method == 'normal':
        # Parametric CI assuming normal distribution
        mean = np.mean(data, axis=axis)
        sem = stats.sem(data, axis=axis)
        
        # Get t-value
        if axis is None:
            n = len(data)
        else:
            n = data.shape[axis]
            
        t_val = stats.t.ppf(1 - alpha/2, n - 1)
        
        ci_low = mean - t_val * sem
        ci_high = mean + t_val * sem
        
    elif method == 'percentile':
        # Non-parametric percentile method
        ci_low = np.percentile(data, 100 * alpha/2, axis=axis)
        ci_high = np.percentile(data, 100 * (1 - alpha/2), axis=axis)
        
    elif method == 'bootstrap':
        # Bootstrap confidence interval
        if axis is not None:
            raise NotImplementedError("Bootstrap CI not implemented for axis != None")
            
        from scipy.stats import bootstrap
        
        # Define statistic function
        def statistic(x):
            return np.mean(x)
            
        # Perform bootstrap
        res = bootstrap((data,), statistic, confidence_level=confidence,
                       n_resamples=10000, method='percentile')
        ci_low, ci_high = res.confidence_interval
        
    else:
        raise InvalidParameterError(f"Unknown CI method: {method}")
        
    return ci_low, ci_high


def detect_outliers(data: np.ndarray,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data.
    
    Parameters
    ----------
    data : ndarray
        Input data
    method : str
        Detection method: 'iqr', 'zscore', 'mad', 'isolation'
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    outlier_mask : ndarray
        Boolean array where True indicates outlier
    """
    data_1d = data.flatten()
    outlier_mask = np.zeros_like(data, dtype=bool)
    
    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(data_1d, 25)
        q3 = np.percentile(data_1d, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(data_1d))
        outlier_mask = z_scores.reshape(data.shape) > threshold
        
    elif method == 'mad':
        # Median absolute deviation method
        median = np.median(data_1d)
        mad = calculate_mad(data_1d)
        
        # Modified z-score
        modified_z = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z) > threshold
        
    elif method == 'isolation':
        # Isolation Forest method (requires sklearn)
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape data for sklearn
            X = data_1d.reshape(-1, 1)
            
            # Fit isolation forest
            clf = IsolationForest(contamination=0.1, random_state=42)
            outliers = clf.fit_predict(X)
            
            # Convert to boolean mask
            outlier_mask = (outliers == -1).reshape(data.shape)
            
        except ImportError:
            raise DependencyError("sklearn required for isolation forest method")
            
    else:
        raise InvalidParameterError(f"Unknown outlier method: {method}")
        
    n_outliers = np.sum(outlier_mask)
    logger.info(f"Detected {n_outliers} outliers ({100*n_outliers/data.size:.1f}%)")
    
    return outlier_mask


def calculate_effect_size(group1: np.ndarray,
                         group2: np.ndarray,
                         method: str = 'cohen') -> float:
    """
    Calculate effect size between two groups.
    
    Parameters
    ----------
    group1 : ndarray
        First group data
    group2 : ndarray
        Second group data
    method : str
        Effect size method: 'cohen', 'glass', 'hedges'
        
    Returns
    -------
    effect_size : float
        Calculated effect size
    """
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    if method == 'cohen':
        # Cohen's d - pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / pooled_std
        
    elif method == 'glass':
        # Glass's delta - uses control group SD
        effect_size = (mean1 - mean2) / np.std(group2, ddof=1)
        
    elif method == 'hedges':
        # Hedges' g - corrected Cohen's d
        n1, n2 = len(group1), len(group2)
        
        # Calculate Cohen's d first
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std
        
        # Apply Hedges correction
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        effect_size = d * correction
        
    else:
        raise InvalidParameterError(f"Unknown effect size method: {method}")
        
    return effect_size


def moving_statistics(data: np.ndarray,
                     window_size: int,
                     statistic: str = 'mean',
                     min_periods: Optional[int] = None) -> np.ndarray:
    """
    Calculate moving window statistics.
    
    Parameters
    ----------
    data : ndarray
        Input data (1D)
    window_size : int
        Size of moving window
    statistic : str
        Statistic to calculate: 'mean', 'median', 'std', 'var'
    min_periods : int, optional
        Minimum number of observations in window
        
    Returns
    -------
    moving_stat : ndarray
        Moving statistic array
    """
    if data.ndim != 1:
        raise InvalidParameterError("Moving statistics only support 1D data")
        
    n = len(data)
    moving_stat = np.full(n, np.nan)
    
    if min_periods is None:
        min_periods = window_size
        
    # Calculate for each window position
    for i in range(n):
        start = max(0, i - window_size + 1)
        end = i + 1
        
        window_data = data[start:end]
        
        if len(window_data) >= min_periods:
            if statistic == 'mean':
                moving_stat[i] = np.mean(window_data)
            elif statistic == 'median':
                moving_stat[i] = np.median(window_data)
            elif statistic == 'std':
                moving_stat[i] = np.std(window_data, ddof=1)
            elif statistic == 'var':
                moving_stat[i] = np.var(window_data, ddof=1)
            else:
                raise InvalidParameterError(f"Unknown statistic: {statistic}")
                
    return moving_stat


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example data
    np.random.seed(42)
    
    # Normal distribution with outliers
    normal_data = np.random.normal(10, 2, 1000)
    outliers = np.random.uniform(20, 30, 50)
    data = np.concatenate([normal_data, outliers])
    
    # Calculate statistics
    basic_stats = calculate_basic_stats(data)
    print("Basic Statistics:")
    for key, value in basic_stats.items():
        print(f"  {key}: {value:.3f}")
        
    print("\nRobust Statistics:")
    robust_stats = calculate_robust_stats(data)
    for key, value in robust_stats.items():
        print(f"  {key}: {value:.3f}")
        
    # Distribution tests
    dist_stats = calculate_distribution_stats(data)
    print("\nDistribution Statistics:")
    print(f"  Skewness: {dist_stats['skewness']:.3f}")
    print(f"  Kurtosis: {dist_stats['kurtosis']:.3f}")
    print(f"  Normality (Shapiro p-value): {dist_stats['shapiro_p']:.4f}")
    
    # Detect outliers
    outlier_mask = detect_outliers(data, method='iqr')
    print(f"\nDetected {np.sum(outlier_mask)} outliers")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with statistics
    axes[0, 0].hist(data, bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(basic_stats['mean'], color='r', 
                      linestyle='--', label='Mean')
    axes[0, 0].axvline(basic_stats['median'], color='g', 
                      linestyle='--', label='Median')
    axes[0, 0].set_title('Data Distribution')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(data, vert=True)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Outlier detection
    axes[1, 1].scatter(np.arange(len(data)), data, 
                      c=outlier_mask, cmap='RdYlBu', alpha=0.5)
    axes[1, 1].set_title('Outlier Detection')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()