"""
Spectral Analysis Plotting Utilities
====================================

Functions for visualizing frequency domain analysis including
power spectra, spectrograms, and coherence plots.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns
from scipy import signal
from typing import Optional, Union, List, Tuple, Dict, Any
import logging

from ...core.exceptions import VisualizationError, InvalidParameterError

logger = logging.getLogger(__name__)


def plot_power_spectrum(frequencies: np.ndarray,
                       power: np.ndarray,
                       ax: Optional[plt.Axes] = None,
                       log_scale: bool = True,
                       freq_range: Optional[Tuple[float, float]] = None,
                       highlight_bands: Optional[Dict[str, Tuple[float, float]]] = None,
                       **kwargs) -> plt.Axes:
    """
    Plot power spectral density.
    
    Parameters
    ----------
    frequencies : ndarray
        Frequency values in Hz
    power : ndarray
        Power spectral density values
    ax : matplotlib.Axes, optional
        Axes to plot on
    log_scale : bool
        Use log scale for power axis
    freq_range : tuple, optional
        Frequency range to display (min, max)
    highlight_bands : dict, optional
        Frequency bands to highlight {'name': (low, high)}
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with power spectrum
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Apply frequency range
    if freq_range is not None:
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[mask]
        power = power[mask]
        
    # Plot power spectrum
    if log_scale:
        ax.semilogy(frequencies, power, **kwargs)
    else:
        ax.plot(frequencies, power, **kwargs)
        
    # Highlight frequency bands
    if highlight_bands is not None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(highlight_bands)))
        
        for i, (band_name, (low, high)) in enumerate(highlight_bands.items()):
            ax.axvspan(low, high, alpha=0.2, color=colors[i], 
                      label=f'{band_name} ({low}-{high} Hz)')
                      
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density' + 
                  (' (log scale)' if log_scale else ''))
    ax.set_title('Power Spectrum')
    ax.grid(True, alpha=0.3)
    
    if highlight_bands:
        ax.legend(loc='upper right')
        
    return ax


def plot_spectrogram(time: np.ndarray,
                    frequencies: np.ndarray,
                    power: np.ndarray,
                    ax: Optional[plt.Axes] = None,
                    cmap: str = 'viridis',
                    log_scale: bool = True,
                    freq_range: Optional[Tuple[float, float]] = None,
                    time_range: Optional[Tuple[float, float]] = None,
                    colorbar: bool = True,
                    **kwargs) -> plt.Axes:
    """
    Plot time-frequency spectrogram.
    
    Parameters
    ----------
    time : ndarray
        Time values in seconds
    frequencies : ndarray
        Frequency values in Hz
    power : ndarray
        2D power array (frequencies x time)
    ax : matplotlib.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    log_scale : bool
        Use log scale for power values
    freq_range : tuple, optional
        Frequency range to display
    time_range : tuple, optional
        Time range to display
    colorbar : bool
        Add colorbar
    **kwargs
        Additional arguments for imshow
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with spectrogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Apply ranges
    freq_mask = np.ones(len(frequencies), dtype=bool)
    time_mask = np.ones(len(time), dtype=bool)
    
    if freq_range is not None:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        
    if time_range is not None:
        time_mask = (time >= time_range[0]) & (time <= time_range[1])
        
    # Subset data
    plot_power = power[freq_mask, :][:, time_mask]
    plot_freq = frequencies[freq_mask]
    plot_time = time[time_mask]
    
    # Apply log scale if requested
    if log_scale:
        plot_power = 10 * np.log10(plot_power + 1e-10)
        
    # Create image
    im = ax.imshow(plot_power, aspect='auto', origin='lower',
                   extent=[plot_time[0], plot_time[-1], 
                          plot_freq[0], plot_freq[-1]],
                   cmap=cmap, **kwargs)
                   
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)' if log_scale else 'Power')
        
    return ax


def plot_coherence(frequencies: np.ndarray,
                  coherence: np.ndarray,
                  phase: Optional[np.ndarray] = None,
                  ax: Optional[plt.Axes] = None,
                  freq_range: Optional[Tuple[float, float]] = None,
                  significance_level: Optional[float] = None,
                  **kwargs) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot coherence between two signals.
    
    Parameters
    ----------
    frequencies : ndarray
        Frequency values
    coherence : ndarray
        Coherence values (0-1)
    phase : ndarray, optional
        Phase values in radians
    ax : matplotlib.Axes or tuple, optional
        Axes to plot on. If phase provided, expects tuple of 2 axes
    freq_range : tuple, optional
        Frequency range to display
    significance_level : float, optional
        Significance threshold to display
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes or tuple
        Axes with coherence (and phase) plot
    """
    # Handle axes
    if phase is not None:
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                          sharex=True)
        else:
            ax1, ax2 = ax
    else:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        else:
            ax1 = ax
            
    # Apply frequency range
    if freq_range is not None:
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[mask]
        coherence = coherence[mask]
        if phase is not None:
            phase = phase[mask]
            
    # Plot coherence
    ax1.plot(frequencies, coherence, **kwargs)
    ax1.set_ylabel('Coherence')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Coherence Analysis')
    
    # Add significance level
    if significance_level is not None:
        ax1.axhline(significance_level, color='r', linestyle='--',
                   label=f'Significance ({significance_level:.2f})')
        ax1.legend()
        
    # Plot phase if provided
    if phase is not None:
        # Convert to degrees
        phase_deg = np.degrees(phase)
        ax2.plot(frequencies, phase_deg, **kwargs)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_ylim(-180, 180)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        return ax1, ax2
    else:
        ax1.set_xlabel('Frequency (Hz)')
        return ax1


def plot_wavelet_transform(time: np.ndarray,
                          frequencies: np.ndarray,
                          cwt_matrix: np.ndarray,
                          ax: Optional[plt.Axes] = None,
                          cmap: str = 'jet',
                          power: bool = True,
                          contour: bool = False,
                          **kwargs) -> plt.Axes:
    """
    Plot continuous wavelet transform.
    
    Parameters
    ----------
    time : ndarray
        Time values
    frequencies : ndarray
        Frequency values
    cwt_matrix : ndarray
        Complex wavelet transform coefficients
    ax : matplotlib.Axes, optional
        Axes to plot on
    cmap : str
        Colormap
    power : bool
        Plot power (magnitude squared) instead of magnitude
    contour : bool
        Add contour lines
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with wavelet plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Calculate magnitude or power
    if power:
        plot_data = np.abs(cwt_matrix)**2
        label = 'Power'
    else:
        plot_data = np.abs(cwt_matrix)
        label = 'Magnitude'
        
    # Create mesh
    T, F = np.meshgrid(time, frequencies)
    
    # Plot
    im = ax.pcolormesh(T, F, plot_data, shading='auto', 
                      cmap=cmap, **kwargs)
                      
    # Add contours if requested
    if contour:
        levels = np.percentile(plot_data.flatten(), [50, 75, 90, 95])
        ax.contour(T, F, plot_data, levels=levels, 
                  colors='white', alpha=0.5, linewidths=0.5)
                  
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Transform')
    
    # Use log scale for frequency
    ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
    
    return ax


def plot_frequency_bands(time: np.ndarray,
                        band_powers: Dict[str, np.ndarray],
                        ax: Optional[plt.Axes] = None,
                        normalize: bool = True,
                        stacked: bool = False,
                        **kwargs) -> plt.Axes:
    """
    Plot power in different frequency bands over time.
    
    Parameters
    ----------
    time : ndarray
        Time values
    band_powers : dict
        Dictionary mapping band names to power time series
    ax : matplotlib.Axes, optional
        Axes to plot on
    normalize : bool
        Normalize each band to sum to 1
    stacked : bool
        Create stacked area plot
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with band power plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Prepare data
    if normalize:
        # Normalize so bands sum to 1 at each time point
        total_power = np.sum(list(band_powers.values()), axis=0)
        total_power[total_power == 0] = 1  # Avoid division by zero
        band_powers_norm = {name: power / total_power 
                           for name, power in band_powers.items()}
        plot_data = band_powers_norm
        ylabel = 'Relative Power'
    else:
        plot_data = band_powers
        ylabel = 'Power'
        
    # Plot
    if stacked:
        # Stacked area plot
        ax.stackplot(time, *plot_data.values(), 
                    labels=plot_data.keys(), alpha=0.7, **kwargs)
    else:
        # Line plot
        for band_name, power in plot_data.items():
            ax.plot(time, power, label=band_name, **kwargs)
            
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title('Frequency Band Power')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_cross_frequency_coupling(phase_freq: np.ndarray,
                                 amp_freq: np.ndarray,
                                 coupling_matrix: np.ndarray,
                                 ax: Optional[plt.Axes] = None,
                                 cmap: str = 'hot',
                                 **kwargs) -> plt.Axes:
    """
    Plot phase-amplitude coupling matrix.
    
    Parameters
    ----------
    phase_freq : ndarray
        Frequencies for phase
    amp_freq : ndarray
        Frequencies for amplitude
    coupling_matrix : ndarray
        Coupling values (phase_freq x amp_freq)
    ax : matplotlib.Axes, optional
        Axes to plot on
    cmap : str
        Colormap
    **kwargs
        Additional arguments for imshow
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with coupling plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # Plot coupling matrix
    im = ax.imshow(coupling_matrix, aspect='auto', origin='lower',
                   extent=[amp_freq[0], amp_freq[-1],
                          phase_freq[0], phase_freq[-1]],
                   cmap=cmap, **kwargs)
                   
    ax.set_xlabel('Amplitude Frequency (Hz)')
    ax.set_ylabel('Phase Frequency (Hz)')
    ax.set_title('Phase-Amplitude Coupling')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coupling Strength')
    
    # Add diagonal line
    ax.plot([amp_freq[0], amp_freq[-1]], 
           [phase_freq[0], phase_freq[-1]], 
           'k--', alpha=0.5)
           
    return ax


def plot_multitaper_spectrum(frequencies: np.ndarray,
                           tapers: np.ndarray,
                           ax: Optional[plt.Axes] = None,
                           log_scale: bool = True,
                           show_individual: bool = True,
                           **kwargs) -> plt.Axes:
    """
    Plot multitaper power spectrum with individual tapers.
    
    Parameters
    ----------
    frequencies : ndarray
        Frequency values
    tapers : ndarray
        Power spectra for each taper (n_tapers x n_freq)
    ax : matplotlib.Axes, optional
        Axes to plot on
    log_scale : bool
        Use log scale for power
    show_individual : bool
        Show individual taper spectra
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with multitaper spectrum
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Plot individual tapers
    if show_individual:
        for i, taper_spec in enumerate(tapers):
            if log_scale:
                ax.semilogy(frequencies, taper_spec, 'gray', 
                           alpha=0.3, linewidth=0.5)
            else:
                ax.plot(frequencies, taper_spec, 'gray', 
                       alpha=0.3, linewidth=0.5)
                       
    # Plot average
    avg_spectrum = np.mean(tapers, axis=0)
    if log_scale:
        ax.semilogy(frequencies, avg_spectrum, 'k-', 
                   linewidth=2, label='Average', **kwargs)
    else:
        ax.plot(frequencies, avg_spectrum, 'k-', 
               linewidth=2, label='Average', **kwargs)
               
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power' + (' (log scale)' if log_scale else ''))
    ax.set_title(f'Multitaper Spectrum ({len(tapers)} tapers)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_spectral_comparison(freq_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           ax: Optional[plt.Axes] = None,
                           log_scale: bool = True,
                           normalize: bool = False,
                           freq_range: Optional[Tuple[float, float]] = None,
                           **kwargs) -> plt.Axes:
    """
    Compare multiple power spectra.
    
    Parameters
    ----------
    freq_data : dict
        Dictionary mapping labels to (frequencies, power) tuples
    ax : matplotlib.Axes, optional
        Axes to plot on
    log_scale : bool
        Use log scale for power
    normalize : bool
        Normalize each spectrum to max=1
    freq_range : tuple, optional
        Frequency range to display
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with comparison plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Plot each spectrum
    for label, (frequencies, power) in freq_data.items():
        # Apply frequency range
        if freq_range is not None:
            mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            frequencies = frequencies[mask]
            power = power[mask]
            
        # Normalize if requested
        if normalize:
            power = power / np.max(power)
            
        # Plot
        if log_scale:
            ax.semilogy(frequencies, power, label=label, **kwargs)
        else:
            ax.plot(frequencies, power, label=label, **kwargs)
            
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power' + (' (normalized)' if normalize else '') +
                  (' (log scale)' if log_scale else ''))
    ax.set_title('Spectral Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # Generate example signal
    fs = 1000  # Hz
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Multi-component signal
    signal_data = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz
                  0.5 * np.sin(2 * np.pi * 25 * t) +  # 25 Hz
                  0.3 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz
                  0.2 * np.random.randn(len(t)))
                  
    # Calculate power spectrum
    frequencies, power = signal.welch(signal_data, fs, nperseg=1024)
    
    # Calculate spectrogram
    f_spec, t_spec, Sxx = signal.spectrogram(signal_data, fs, nperseg=256)
    
    # Create example plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Power spectrum with bands
    bands = {
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 80)
    }
    plot_power_spectrum(frequencies, power, ax=axes[0],
                       freq_range=(0, 100), highlight_bands=bands)
                       
    # 2. Spectrogram
    plot_spectrogram(t_spec, f_spec, Sxx, ax=axes[1],
                    freq_range=(0, 100))
                    
    # 3. Band power over time
    # Calculate band powers
    band_powers = {}
    for band_name, (low, high) in bands.items():
        # Simple band power calculation
        band_mask = (f_spec >= low) & (f_spec <= high)
        band_powers[band_name] = np.mean(Sxx[band_mask, :], axis=0)
        
    plot_frequency_bands(t_spec, band_powers, ax=axes[2], 
                        stacked=True)
                        
    # 4. Spectral comparison
    # Add some variation to create multiple spectra
    spec_data = {
        'Original': (frequencies, power),
        'Filtered': (frequencies, power * np.exp(-frequencies/50))
    }
    plot_spectral_comparison(spec_data, ax=axes[3],
                           freq_range=(0, 100))
    
    plt.tight_layout()
    plt.show()