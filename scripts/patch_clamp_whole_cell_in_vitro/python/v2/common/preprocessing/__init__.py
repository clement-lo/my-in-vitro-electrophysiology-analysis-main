"""
Preprocessing Utilities
======================

Common preprocessing functions for electrophysiology data including
filtering, detrending, and baseline correction.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

# Filtering functions
from .filters import (
    design_filter,
    apply_filter,
    bandpass_filter,
    lowpass_filter,
    highpass_filter,
    notch_filter,
    adaptive_filter,
    filter_bank
)

# Detrending functions
from .detrend import (
    detrend,
    detrend_linear,
    detrend_constant,
    detrend_polynomial,
    detrend_savgol,
    detrend_breakpoints,
    detrend_exponential,
    detrend_adaptive
)

# Baseline correction functions
from .baseline import (
    baseline_correct,
    baseline_mean,
    baseline_median,
    baseline_mode,
    baseline_polynomial,
    baseline_percentile,
    baseline_als,
    baseline_rolling,
    baseline_morphological,
    baseline_adaptive
)

__all__ = [
    # Filtering
    'design_filter',
    'apply_filter',
    'bandpass_filter',
    'lowpass_filter',
    'highpass_filter',
    'notch_filter',
    'adaptive_filter',
    'filter_bank',
    
    # Detrending
    'detrend',
    'detrend_linear',
    'detrend_constant',
    'detrend_polynomial',
    'detrend_savgol',
    'detrend_breakpoints',
    'detrend_exponential',
    'detrend_adaptive',
    
    # Baseline correction
    'baseline_correct',
    'baseline_mean',
    'baseline_median',
    'baseline_mode',
    'baseline_polynomial',
    'baseline_percentile',
    'baseline_als',
    'baseline_rolling',
    'baseline_morphological',
    'baseline_adaptive'
]