"""
Statistical Analysis Utilities
==============================

Common statistical functions for electrophysiology data analysis including
descriptive statistics, hypothesis testing, and time series analysis.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

# Descriptive statistics
from .descriptive import (
    calculate_basic_stats,
    calculate_percentiles,
    calculate_distribution_stats,
    calculate_mad,
    calculate_robust_stats,
    calculate_ci,
    detect_outliers,
    calculate_effect_size,
    moving_statistics
)

# Hypothesis testing
from .hypothesis_testing import (
    normality_test,
    compare_two_groups,
    compare_multiple_groups,
    perform_post_hoc,
    paired_samples_test,
    correlation_test,
    multiple_testing_correction
)

# Time series analysis
from .time_series import (
    autocorrelation,
    partial_autocorrelation,
    cross_correlation,
    stationarity_test,
    detrend_time_series,
    spectral_density,
    find_periodicities,
    phase_synchrony,
    granger_causality,
    entropy_measures
)

__all__ = [
    # Descriptive statistics
    'calculate_basic_stats',
    'calculate_percentiles',
    'calculate_distribution_stats',
    'calculate_mad',
    'calculate_robust_stats',
    'calculate_ci',
    'detect_outliers',
    'calculate_effect_size',
    'moving_statistics',
    
    # Hypothesis testing
    'normality_test',
    'compare_two_groups',
    'compare_multiple_groups',
    'perform_post_hoc',
    'paired_samples_test',
    'correlation_test',
    'multiple_testing_correction',
    
    # Time series analysis
    'autocorrelation',
    'partial_autocorrelation',
    'cross_correlation',
    'stationarity_test',
    'detrend_time_series',
    'spectral_density',
    'find_periodicities',
    'phase_synchrony',
    'granger_causality',
    'entropy_measures'
]