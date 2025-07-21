"""
Visualization Utilities
=======================

Common visualization functions for electrophysiology data including
trace plots, event visualization, and spectral analysis plots.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

# Trace plotting functions
from .trace_plots import (
    plot_trace,
    plot_multi_trace,
    plot_trace_with_events,
    plot_trace_heatmap,
    plot_trace_colored,
    plot_raster,
    plot_trace_shaded,
    plot_phase_plane,
    create_trace_figure
)

# Event plotting functions
from .event_plots import (
    plot_spike_overlay,
    plot_spike_histogram,
    plot_isi_histogram,
    plot_burst_detection,
    plot_event_raster,
    plot_psth,
    plot_event_triggered_average,
    plot_event_statistics
)

# Spectral plotting functions
from .spectral_plots import (
    plot_power_spectrum,
    plot_spectrogram,
    plot_coherence,
    plot_wavelet_transform,
    plot_frequency_bands,
    plot_cross_frequency_coupling,
    plot_multitaper_spectrum,
    plot_spectral_comparison
)

__all__ = [
    # Trace plots
    'plot_trace',
    'plot_multi_trace',
    'plot_trace_with_events',
    'plot_trace_heatmap',
    'plot_trace_colored',
    'plot_raster',
    'plot_trace_shaded',
    'plot_phase_plane',
    'create_trace_figure',
    
    # Event plots
    'plot_spike_overlay',
    'plot_spike_histogram',
    'plot_isi_histogram',
    'plot_burst_detection',
    'plot_event_raster',
    'plot_psth',
    'plot_event_triggered_average',
    'plot_event_statistics',
    
    # Spectral plots
    'plot_power_spectrum',
    'plot_spectrogram',
    'plot_coherence',
    'plot_wavelet_transform',
    'plot_frequency_bands',
    'plot_cross_frequency_coupling',
    'plot_multitaper_spectrum',
    'plot_spectral_comparison'
]