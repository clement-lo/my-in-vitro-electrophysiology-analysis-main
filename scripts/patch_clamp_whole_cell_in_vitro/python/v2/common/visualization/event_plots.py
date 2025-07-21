"""
Event Plotting Utilities
========================

Functions for visualizing detected events in electrophysiology data
including spikes, bursts, and other discrete events.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict, Any
import logging

from ...core.exceptions import VisualizationError, InvalidParameterError

logger = logging.getLogger(__name__)


def plot_spike_overlay(time: np.ndarray,
                      data: np.ndarray,
                      spike_times: np.ndarray,
                      window: Tuple[float, float] = (-0.002, 0.003),
                      ax: Optional[plt.Axes] = None,
                      normalize: bool = True,
                      color_map: bool = True,
                      alpha: float = 0.5,
                      **kwargs) -> plt.Axes:
    """
    Overlay detected spikes on a single plot.
    
    Parameters
    ----------
    time : ndarray
        Time vector in seconds
    data : ndarray
        Signal data
    spike_times : ndarray
        Times of detected spikes
    window : tuple
        Time window around spike (pre, post) in seconds
    ax : matplotlib.Axes, optional
        Axes to plot on
    normalize : bool
        Normalize spike waveforms
    color_map : bool
        Color code spikes by time
    alpha : float
        Transparency for overlaid spikes
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with spike overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Convert window to samples
    dt = time[1] - time[0]
    pre_samples = int(abs(window[0]) / dt)
    post_samples = int(window[1] / dt)
    
    # Extract spike waveforms
    waveforms = []
    valid_spikes = []
    
    for spike_time in spike_times:
        # Find nearest sample
        spike_idx = np.argmin(np.abs(time - spike_time))
        
        # Check bounds
        if spike_idx - pre_samples >= 0 and spike_idx + post_samples < len(data):
            waveform = data[spike_idx - pre_samples:spike_idx + post_samples]
            waveforms.append(waveform)
            valid_spikes.append(spike_time)
            
    if not waveforms:
        logger.warning("No valid spike waveforms found")
        return ax
        
    waveforms = np.array(waveforms)
    
    # Create time axis for waveforms
    waveform_time = np.linspace(window[0], window[1], pre_samples + post_samples)
    
    # Normalize if requested
    if normalize:
        # Normalize by peak
        peaks = np.max(np.abs(waveforms), axis=1, keepdims=True)
        peaks[peaks == 0] = 1  # Avoid division by zero
        waveforms = waveforms / peaks
        
    # Plot waveforms
    if color_map:
        # Color code by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(waveforms)))
        for i, waveform in enumerate(waveforms):
            ax.plot(waveform_time * 1000, waveform, 
                   color=colors[i], alpha=alpha, **kwargs)
    else:
        # Plot all in same color
        for waveform in waveforms:
            ax.plot(waveform_time * 1000, waveform, 
                   'k-', alpha=alpha, **kwargs)
            
    # Add mean waveform
    mean_waveform = np.mean(waveforms, axis=0)
    ax.plot(waveform_time * 1000, mean_waveform, 'r-', 
           linewidth=2, label='Mean')
           
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude' + (' (normalized)' if normalize else ''))
    ax.set_title(f'Spike Overlay (n={len(waveforms)})')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_spike_histogram(spike_times: Union[np.ndarray, List[np.ndarray]],
                        bin_size: float = 0.01,
                        ax: Optional[plt.Axes] = None,
                        normalize: bool = False,
                        cumulative: bool = False,
                        **kwargs) -> plt.Axes:
    """
    Plot spike time histogram.
    
    Parameters
    ----------
    spike_times : array or list of arrays
        Spike times. If list, creates stacked histogram
    bin_size : float
        Bin size in seconds
    ax : matplotlib.Axes, optional
        Axes to plot on
    normalize : bool
        Normalize to firing rate (Hz)
    cumulative : bool
        Plot cumulative histogram
    **kwargs
        Additional arguments for hist()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with histogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    # Handle single array
    if isinstance(spike_times, np.ndarray):
        spike_times = [spike_times]
        
    # Find time range
    all_spikes = np.concatenate(spike_times)
    t_min, t_max = all_spikes.min(), all_spikes.max()
    
    # Create bins
    bins = np.arange(t_min, t_max + bin_size, bin_size)
    
    # Plot histogram(s)
    if len(spike_times) == 1:
        counts, _, _ = ax.hist(spike_times[0], bins=bins, 
                              cumulative=cumulative, **kwargs)
    else:
        # Stacked histogram for multiple units
        ax.hist(spike_times, bins=bins, stacked=True,
               cumulative=cumulative, label=[f'Unit {i+1}' 
               for i in range(len(spike_times))], **kwargs)
        ax.legend()
        
    if normalize:
        # Convert to firing rate
        ax.set_ylabel('Firing Rate (Hz)')
        # Scale y-axis
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0]/bin_size, ylim[1]/bin_size)
    else:
        ax.set_ylabel('Spike Count')
        
    ax.set_xlabel('Time (s)')
    ax.set_title('Spike Time Histogram')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_isi_histogram(spike_times: np.ndarray,
                      bin_size: float = 0.001,
                      max_isi: Optional[float] = None,
                      ax: Optional[plt.Axes] = None,
                      log_scale: bool = False,
                      **kwargs) -> plt.Axes:
    """
    Plot inter-spike interval (ISI) histogram.
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times in seconds
    bin_size : float
        Bin size in seconds
    max_isi : float, optional
        Maximum ISI to include
    ax : matplotlib.Axes, optional
        Axes to plot on
    log_scale : bool
        Use log scale for x-axis
    **kwargs
        Additional arguments for hist()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with ISI histogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    if len(isis) == 0:
        logger.warning("Not enough spikes for ISI calculation")
        return ax
        
    # Filter by max ISI if specified
    if max_isi is not None:
        isis = isis[isis <= max_isi]
        
    # Create bins
    if log_scale:
        # Logarithmic bins
        min_isi = isis[isis > 0].min() if np.any(isis > 0) else 0.0001
        bins = np.logspace(np.log10(min_isi), np.log10(isis.max()), 50)
    else:
        # Linear bins
        bins = np.arange(0, isis.max() + bin_size, bin_size)
        
    # Plot histogram
    counts, bins, patches = ax.hist(isis * 1000, bins=bins * 1000, **kwargs)
    
    # Add statistics
    mean_isi = np.mean(isis) * 1000
    cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
    
    ax.axvline(mean_isi, color='r', linestyle='--', 
              label=f'Mean: {mean_isi:.1f} ms')
              
    ax.set_xlabel('Inter-Spike Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title(f'ISI Distribution (CV={cv:.2f})')
    
    if log_scale:
        ax.set_xscale('log')
        
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_burst_detection(time: np.ndarray,
                        data: np.ndarray,
                        bursts: List[Tuple[float, float]],
                        spike_times: Optional[np.ndarray] = None,
                        ax: Optional[plt.Axes] = None,
                        trace_color: str = 'black',
                        burst_color: str = 'red',
                        spike_color: str = 'blue',
                        **kwargs) -> plt.Axes:
    """
    Plot trace with burst periods highlighted.
    
    Parameters
    ----------
    time : ndarray
        Time vector
    data : ndarray
        Signal data
    bursts : list of tuples
        List of (start_time, end_time) for each burst
    spike_times : ndarray, optional
        Times of individual spikes
    ax : matplotlib.Axes, optional
        Axes to plot on
    trace_color : str
        Color for trace
    burst_color : str
        Color for burst highlighting
    spike_color : str
        Color for spike markers
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with burst visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        
    # Plot trace
    ax.plot(time, data, color=trace_color, linewidth=0.8)
    
    # Highlight burst periods
    for start, end in bursts:
        ax.axvspan(start, end, color=burst_color, alpha=0.3)
        
    # Plot spikes if provided
    if spike_times is not None:
        # Find spike amplitudes
        spike_indices = np.searchsorted(time, spike_times)
        spike_indices = np.clip(spike_indices, 0, len(data)-1)
        spike_amplitudes = data[spike_indices]
        
        ax.scatter(spike_times, spike_amplitudes, 
                  color=spike_color, s=30, zorder=5)
                  
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Burst Detection ({len(bursts)} bursts)')
    ax.grid(True, alpha=0.3)
    
    # Add burst statistics
    if bursts:
        burst_durations = [end - start for start, end in bursts]
        mean_duration = np.mean(burst_durations)
        ax.text(0.02, 0.95, f'Mean burst duration: {mean_duration*1000:.1f} ms',
               transform=ax.transAxes, verticalalignment='top')
               
    return ax


def plot_event_raster(event_dict: Dict[str, np.ndarray],
                     ax: Optional[plt.Axes] = None,
                     colors: Optional[Dict[str, str]] = None,
                     labels: Optional[Dict[str, str]] = None,
                     time_range: Optional[Tuple[float, float]] = None,
                     **kwargs) -> plt.Axes:
    """
    Plot raster of different event types.
    
    Parameters
    ----------
    event_dict : dict
        Dictionary mapping event types to time arrays
    ax : matplotlib.Axes, optional
        Axes to plot on
    colors : dict, optional
        Colors for each event type
    labels : dict, optional
        Labels for each event type
    time_range : tuple, optional
        Time range to display
    **kwargs
        Additional arguments for eventplot()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with event raster
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Default colors
    if colors is None:
        colors = {}
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(event_dict)))
        for i, key in enumerate(event_dict.keys()):
            colors[key] = color_cycle[i]
            
    # Prepare data for eventplot
    positions = []
    linecolors = []
    linelabels = []
    
    for i, (event_type, times) in enumerate(event_dict.items()):
        if time_range is not None:
            # Filter by time range
            mask = (times >= time_range[0]) & (times <= time_range[1])
            times = times[mask]
            
        positions.append(times)
        linecolors.append(colors.get(event_type, 'black'))
        
        if labels:
            linelabels.append(labels.get(event_type, event_type))
        else:
            linelabels.append(event_type)
            
    # Create event plot
    ax.eventplot(positions, colors=linecolors, 
                lineoffsets=np.arange(len(positions)),
                linelengths=0.8, **kwargs)
                
    # Set labels
    ax.set_yticks(np.arange(len(positions)))
    ax.set_yticklabels(linelabels)
    ax.set_xlabel('Time (s)')
    ax.set_title('Event Raster')
    
    if time_range is not None:
        ax.set_xlim(time_range)
        
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_psth(spike_times: np.ndarray,
             event_times: np.ndarray,
             window: Tuple[float, float] = (-0.5, 1.0),
             bin_size: float = 0.01,
             ax: Optional[plt.Axes] = None,
             baseline_subtract: bool = True,
             error_bars: bool = True,
             **kwargs) -> Tuple[plt.Axes, np.ndarray, np.ndarray]:
    """
    Plot peri-stimulus time histogram (PSTH).
    
    Parameters
    ----------
    spike_times : ndarray
        Times of all spikes
    event_times : ndarray
        Times of stimulus/events
    window : tuple
        Time window around events (pre, post) in seconds
    bin_size : float
        Bin size in seconds
    ax : matplotlib.Axes, optional
        Axes to plot on
    baseline_subtract : bool
        Subtract baseline firing rate
    error_bars : bool
        Show standard error bars
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with PSTH
    bins : ndarray
        Bin edges
    psth : ndarray
        PSTH values (spikes/s)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Create bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Collect spike times relative to events
    relative_times = []
    
    for event_time in event_times:
        # Find spikes in window
        mask = (spike_times >= event_time + window[0]) & \
               (spike_times <= event_time + window[1])
        event_spikes = spike_times[mask] - event_time
        relative_times.extend(event_spikes)
        
    relative_times = np.array(relative_times)
    
    # Calculate histogram
    counts, _ = np.histogram(relative_times, bins=bins)
    
    # Convert to firing rate (spikes/s)
    psth = counts / (len(event_times) * bin_size)
    
    # Baseline subtraction
    if baseline_subtract and window[0] < 0:
        baseline_mask = bin_centers < 0
        if np.any(baseline_mask):
            baseline_rate = np.mean(psth[baseline_mask])
            psth = psth - baseline_rate
            
    # Calculate error bars
    if error_bars:
        # Standard error assuming Poisson statistics
        se = np.sqrt(counts) / (len(event_times) * bin_size)
        ax.bar(bin_centers, psth, width=bin_size, 
              yerr=se, capsize=2, **kwargs)
    else:
        ax.bar(bin_centers, psth, width=bin_size, **kwargs)
        
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Peri-Stimulus Time Histogram')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax, bins, psth


def plot_event_triggered_average(time: np.ndarray,
                                data: np.ndarray,
                                event_times: np.ndarray,
                                window: Tuple[float, float] = (-0.1, 0.2),
                                ax: Optional[plt.Axes] = None,
                                show_traces: bool = True,
                                n_traces: Optional[int] = None,
                                **kwargs) -> Tuple[plt.Axes, np.ndarray, np.ndarray]:
    """
    Plot event-triggered average of continuous signal.
    
    Parameters
    ----------
    time : ndarray
        Time vector for data
    data : ndarray
        Continuous signal data
    event_times : ndarray
        Times of trigger events
    window : tuple
        Time window around events (pre, post)
    ax : matplotlib.Axes, optional
        Axes to plot on
    show_traces : bool
        Show individual traces
    n_traces : int, optional
        Number of individual traces to show
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with triggered average
    avg_time : ndarray
        Time vector for average
    avg_signal : ndarray
        Average signal
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Sample rate
    dt = time[1] - time[0]
    
    # Window in samples
    pre_samples = int(abs(window[0]) / dt)
    post_samples = int(window[1] / dt)
    total_samples = pre_samples + post_samples
    
    # Collect triggered segments
    segments = []
    
    for event_time in event_times:
        # Find nearest sample
        event_idx = np.argmin(np.abs(time - event_time))
        
        # Check bounds
        if event_idx - pre_samples >= 0 and event_idx + post_samples < len(data):
            segment = data[event_idx - pre_samples:event_idx + post_samples]
            segments.append(segment)
            
    if not segments:
        logger.warning("No valid segments found")
        return ax, np.array([]), np.array([])
        
    segments = np.array(segments)
    
    # Calculate average
    avg_signal = np.mean(segments, axis=0)
    std_signal = np.std(segments, axis=0)
    
    # Time vector
    avg_time = np.linspace(window[0], window[1], total_samples)
    
    # Plot individual traces
    if show_traces:
        n_show = n_traces if n_traces else min(len(segments), 20)
        for i in range(n_show):
            ax.plot(avg_time * 1000, segments[i], 'gray', 
                   alpha=0.3, linewidth=0.5)
            
    # Plot average
    ax.plot(avg_time * 1000, avg_signal, 'k-', linewidth=2, 
           label='Average', **kwargs)
           
    # Add standard deviation
    ax.fill_between(avg_time * 1000, 
                   avg_signal - std_signal,
                   avg_signal + std_signal,
                   color='gray', alpha=0.3, label='±1 SD')
                   
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time from event (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Event-Triggered Average (n={len(segments)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax, avg_time, avg_signal


def plot_event_statistics(event_times: Union[np.ndarray, Dict[str, np.ndarray]],
                         duration: Optional[float] = None,
                         ax: Optional[plt.Axes] = None,
                         plot_type: str = 'bar',
                         **kwargs) -> plt.Axes:
    """
    Plot summary statistics of events.
    
    Parameters
    ----------
    event_times : array or dict
        Event times or dictionary of event types
    duration : float, optional
        Recording duration for rate calculation
    ax : matplotlib.Axes, optional
        Axes to plot on
    plot_type : str
        Type of plot: 'bar', 'box', 'violin'
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes with statistics plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Convert to dict if necessary
    if isinstance(event_times, np.ndarray):
        event_dict = {'Events': event_times}
    else:
        event_dict = event_times
        
    # Calculate statistics
    stats = {}
    for name, times in event_dict.items():
        if len(times) > 1:
            isis = np.diff(times)
            stats[name] = {
                'count': len(times),
                'rate': len(times) / duration if duration else np.nan,
                'mean_isi': np.mean(isis),
                'cv_isi': np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0,
                'isis': isis
            }
        else:
            stats[name] = {
                'count': len(times),
                'rate': len(times) / duration if duration else np.nan,
                'mean_isi': np.nan,
                'cv_isi': np.nan,
                'isis': np.array([])
            }
            
    # Create plot based on type
    if plot_type == 'bar':
        # Bar plot of rates
        names = list(stats.keys())
        rates = [stats[n]['rate'] for n in names]
        
        ax.bar(names, rates, **kwargs)
        ax.set_ylabel('Event Rate (Hz)')
        ax.set_title('Event Rates')
        
    elif plot_type == 'box':
        # Box plot of ISIs
        isi_data = [stats[n]['isis'] * 1000 for n in stats.keys() 
                   if len(stats[n]['isis']) > 0]
        labels = [n for n in stats.keys() if len(stats[n]['isis']) > 0]
        
        ax.boxplot(isi_data, labels=labels, **kwargs)
        ax.set_ylabel('Inter-Event Interval (ms)')
        ax.set_title('ISI Distributions')
        
    elif plot_type == 'violin':
        # Violin plot of ISIs
        isi_data = []
        labels = []
        
        for name, stat in stats.items():
            if len(stat['isis']) > 0:
                isi_data.extend(stat['isis'] * 1000)
                labels.extend([name] * len(stat['isis']))
                
        if isi_data:
            import pandas as pd
            df = pd.DataFrame({'ISI': isi_data, 'Type': labels})
            
            # Use seaborn for violin plot
            import seaborn as sns
            sns.violinplot(data=df, x='Type', y='ISI', ax=ax, **kwargs)
            ax.set_ylabel('Inter-Event Interval (ms)')
            ax.set_title('ISI Distributions')
            
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example data
    np.random.seed(42)
    
    # Simulate recording
    duration = 10.0
    fs = 20000
    time = np.linspace(0, duration, int(duration * fs))
    
    # Generate spikes
    spike_rate = 20  # Hz
    n_spikes = int(duration * spike_rate)
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
    
    # Add bursts
    burst_times = []
    for i in range(5):
        burst_start = i * 2
        burst_spikes = burst_start + np.cumsum(
            np.random.exponential(0.005, 10))
        burst_times.extend(burst_spikes[burst_spikes < burst_start + 0.1])
    
    all_spikes = np.sort(np.concatenate([spike_times, burst_times]))
    
    # Generate signal with spikes
    signal = np.random.randn(len(time)) * 0.1
    for spike_time in all_spikes:
        spike_idx = int(spike_time * fs)
        if spike_idx < len(signal):
            signal[spike_idx] = 1.0
            
    # Create figure with examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Spike overlay
    plot_spike_overlay(time, signal, all_spikes[:50], 
                      ax=axes[0], window=(-0.002, 0.003))
                      
    # 2. Spike histogram
    plot_spike_histogram(all_spikes, bin_size=0.1, ax=axes[1])
    
    # 3. ISI histogram
    plot_isi_histogram(all_spikes, ax=axes[2])
    
    # 4. Burst detection
    bursts = [(i*2, i*2+0.1) for i in range(5)]
    plot_burst_detection(time[:int(5*fs)], signal[:int(5*fs)], 
                        bursts, all_spikes[all_spikes < 5], ax=axes[3])
                        
    # 5. Event raster
    event_dict = {
        'Regular spikes': spike_times,
        'Burst spikes': np.array(burst_times)
    }
    plot_event_raster(event_dict, ax=axes[4], time_range=(0, 5))
    
    # 6. Event statistics
    plot_event_statistics(event_dict, duration=duration, 
                         ax=axes[5], plot_type='bar')
    
    plt.tight_layout()
    plt.show()