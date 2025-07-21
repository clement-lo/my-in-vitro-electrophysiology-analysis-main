"""
Trace Plotting Utilities
========================

Functions for plotting electrophysiology traces with various
visualization options and annotations.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict, Any
import logging

from ...core.exceptions import VisualizationError, InvalidParameterError

logger = logging.getLogger(__name__)


def plot_trace(time: np.ndarray,
               data: np.ndarray,
               ax: Optional[plt.Axes] = None,
               title: Optional[str] = None,
               xlabel: str = 'Time (s)',
               ylabel: str = 'Amplitude',
               color: str = 'blue',
               linewidth: float = 1.0,
               alpha: float = 1.0,
               **kwargs) -> plt.Axes:
    """
    Plot a single electrophysiology trace.
    
    Parameters
    ----------
    time : ndarray
        Time vector in seconds
    data : ndarray
        Signal data
    ax : matplotlib.Axes, optional
        Axes to plot on. If None, creates new figure
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    color : str or tuple
        Line color
    linewidth : float
        Line width
    alpha : float
        Line transparency
    **kwargs
        Additional arguments passed to plot()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    try:
        ax.plot(time, data, color=color, linewidth=linewidth, 
                alpha=alpha, **kwargs)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
            
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    except Exception as e:
        raise VisualizationError(f"Failed to plot trace: {e}")
        
    return ax


def plot_multi_trace(time: np.ndarray,
                    data: Union[np.ndarray, List[np.ndarray]],
                    labels: Optional[List[str]] = None,
                    ax: Optional[plt.Axes] = None,
                    colors: Optional[List] = None,
                    offset: float = 0,
                    normalize: bool = False,
                    **kwargs) -> plt.Axes:
    """
    Plot multiple traces on the same axes.
    
    Parameters
    ----------
    time : ndarray
        Time vector in seconds
    data : ndarray or list of arrays
        Signal data. If 2D array, each row is a trace
    labels : list of str, optional
        Labels for each trace
    ax : matplotlib.Axes, optional
        Axes to plot on
    colors : list, optional
        Colors for each trace. If None, uses color cycle
    offset : float
        Vertical offset between traces
    normalize : bool
        Normalize each trace to unit variance
    **kwargs
        Additional arguments passed to plot()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with plots
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Convert to list of arrays
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = [data]
        else:
            data = [data[i] for i in range(data.shape[0])]
            
    n_traces = len(data)
    
    # Set up colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_traces))
        
    # Plot each trace
    for i, trace in enumerate(data):
        if normalize:
            trace = (trace - np.mean(trace)) / (np.std(trace) + 1e-8)
            
        # Apply offset
        y_data = trace + i * offset
        
        label = labels[i] if labels else None
        color = colors[i] if i < len(colors) else 'blue'
        
        ax.plot(time, y_data, color=color, label=label, **kwargs)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude' + (' (normalized)' if normalize else ''))
    
    if labels:
        ax.legend(loc='best')
        
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_trace_with_events(time: np.ndarray,
                          data: np.ndarray,
                          events: np.ndarray,
                          event_labels: Optional[List[str]] = None,
                          ax: Optional[plt.Axes] = None,
                          trace_color: str = 'blue',
                          event_color: str = 'red',
                          event_style: str = 'markers',
                          **kwargs) -> plt.Axes:
    """
    Plot trace with event markers.
    
    Parameters
    ----------
    time : ndarray
        Time vector in seconds
    data : ndarray
        Signal data
    events : ndarray
        Event times or indices
    event_labels : list of str, optional
        Labels for events
    ax : matplotlib.Axes, optional
        Axes to plot on
    trace_color : str
        Color for trace
    event_color : str
        Color for events
    event_style : str
        How to display events: 'markers', 'lines', 'spans'
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    # Plot trace
    ax.plot(time, data, color=trace_color, linewidth=1.0)
    
    # Convert event indices to times if needed
    if events.dtype == int:
        event_times = time[events]
        event_values = data[events]
    else:
        event_times = events
        # Interpolate to get values
        event_values = np.interp(event_times, time, data)
        
    # Plot events
    if event_style == 'markers':
        ax.scatter(event_times, event_values, color=event_color, 
                  s=50, zorder=5, **kwargs)
    elif event_style == 'lines':
        for et in event_times:
            ax.axvline(et, color=event_color, alpha=0.5, 
                      linestyle='--', **kwargs)
    elif event_style == 'spans':
        # Assume pairs of events
        for i in range(0, len(event_times)-1, 2):
            ax.axvspan(event_times[i], event_times[i+1], 
                      color=event_color, alpha=0.2, **kwargs)
            
    # Add labels if provided
    if event_labels and event_style == 'markers':
        for i, (et, ev) in enumerate(zip(event_times, event_values)):
            if i < len(event_labels):
                ax.annotate(event_labels[i], (et, ev), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8)
                          
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_trace_heatmap(time: np.ndarray,
                      data: np.ndarray,
                      ax: Optional[plt.Axes] = None,
                      cmap: str = 'viridis',
                      aspect: str = 'auto',
                      interpolation: str = 'nearest',
                      colorbar: bool = True,
                      **kwargs) -> plt.Axes:
    """
    Plot multiple traces as a heatmap.
    
    Parameters
    ----------
    time : ndarray
        Time vector in seconds
    data : ndarray
        2D array where each row is a trace
    ax : matplotlib.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    aspect : str
        Aspect ratio setting
    interpolation : str
        Interpolation method
    colorbar : bool
        Whether to add colorbar
    **kwargs
        Additional arguments passed to imshow()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with heatmap
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    if data.ndim != 2:
        raise InvalidParameterError("Data must be 2D array for heatmap")
        
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect=aspect,
                   interpolation=interpolation,
                   extent=[time[0], time[-1], data.shape[0], 0],
                   **kwargs)
                   
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trace #')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude')
        
    return ax


def plot_trace_colored(time: np.ndarray,
                      data: np.ndarray,
                      color_data: np.ndarray,
                      ax: Optional[plt.Axes] = None,
                      cmap: str = 'viridis',
                      linewidth: float = 2.0,
                      colorbar: bool = True,
                      **kwargs) -> plt.Axes:
    """
    Plot trace with color-coded segments.
    
    Parameters
    ----------
    time : ndarray
        Time vector
    data : ndarray
        Signal data
    color_data : ndarray
        Data for color coding (same length as time/data)
    ax : matplotlib.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    linewidth : float
        Line width
    colorbar : bool
        Whether to add colorbar
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with colored trace
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    # Create line segments
    points = np.array([time, data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth)
    lc.set_array(color_data)
    
    # Add to axes
    line = ax.add_collection(lc)
    
    # Set limits
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(data.min() - 0.1*np.ptp(data), 
                data.max() + 0.1*np.ptp(data))
                
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    if colorbar:
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label('Color value')
        
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_raster(spike_times: List[np.ndarray],
               ax: Optional[plt.Axes] = None,
               colors: Optional[List] = None,
               marker: str = '|',
               markersize: float = 10,
               linewidth: float = 1.0,
               **kwargs) -> plt.Axes:
    """
    Plot spike raster.
    
    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit/trial
    ax : matplotlib.Axes, optional
        Axes to plot on
    colors : list, optional
        Colors for each unit
    marker : str
        Marker style
    markersize : float
        Marker size
    linewidth : float
        Line width for markers
    **kwargs
        Additional arguments passed to eventplot()
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with raster plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    n_units = len(spike_times)
    
    # Set up colors
    if colors is None:
        colors = ['black'] * n_units
        
    # Create raster plot
    ax.eventplot(spike_times, colors=colors, 
                linelengths=0.8, linewidths=linewidth, **kwargs)
                
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit #')
    ax.set_ylim(-0.5, n_units - 0.5)
    
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_trace_shaded(time: np.ndarray,
                     mean: np.ndarray,
                     error: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                     ax: Optional[plt.Axes] = None,
                     color: str = 'blue',
                     alpha: float = 0.3,
                     label: Optional[str] = None,
                     **kwargs) -> plt.Axes:
    """
    Plot trace with shaded error region.
    
    Parameters
    ----------
    time : ndarray
        Time vector
    mean : ndarray
        Mean trace
    error : ndarray or tuple
        If array: symmetric error (e.g., std)
        If tuple: (lower_error, upper_error)
    ax : matplotlib.Axes, optional
        Axes to plot on
    color : str
        Color for trace and shading
    alpha : float
        Transparency for shading
    label : str, optional
        Label for legend
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with shaded plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    # Plot mean
    ax.plot(time, mean, color=color, label=label, **kwargs)
    
    # Add shading
    if isinstance(error, tuple):
        lower = mean - error[0]
        upper = mean + error[1]
    else:
        lower = mean - error
        upper = mean + error
        
    ax.fill_between(time, lower, upper, color=color, alpha=alpha)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    if label:
        ax.legend()
        
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_phase_plane(x: np.ndarray,
                    y: np.ndarray,
                    ax: Optional[plt.Axes] = None,
                    color: Union[str, np.ndarray] = 'blue',
                    cmap: str = 'viridis',
                    xlabel: str = 'X',
                    ylabel: str = 'Y',
                    arrows: bool = False,
                    arrow_spacing: int = 50,
                    **kwargs) -> plt.Axes:
    """
    Plot phase plane trajectory.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    ax : matplotlib.Axes, optional
        Axes to plot on
    color : str or array
        Color specification. If array, color-codes trajectory
    cmap : str
        Colormap for trajectory coloring
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    arrows : bool
        Add direction arrows
    arrow_spacing : int
        Spacing between arrows
    **kwargs
        Additional arguments
        
    Returns
    -------
    ax : matplotlib.Axes
        Axes object with phase plane plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Plot trajectory
    if isinstance(color, np.ndarray):
        # Color-coded trajectory
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=cmap, **kwargs)
        lc.set_array(color)
        ax.add_collection(lc)
        
        ax.set_xlim(x.min() - 0.1*np.ptp(x), x.max() + 0.1*np.ptp(x))
        ax.set_ylim(y.min() - 0.1*np.ptp(y), y.max() + 0.1*np.ptp(y))
    else:
        ax.plot(x, y, color=color, **kwargs)
        
    # Add direction arrows
    if arrows:
        indices = np.arange(0, len(x)-1, arrow_spacing)
        for i in indices:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ax.arrow(x[i], y[i], dx*0.5, dy*0.5,
                    head_width=0.02*np.ptp(x),
                    head_length=0.02*np.ptp(x),
                    fc=color if isinstance(color, str) else 'black',
                    ec=color if isinstance(color, str) else 'black',
                    alpha=0.5)
                    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    return ax


def create_trace_figure(n_subplots: int,
                       layout: Optional[Tuple[int, int]] = None,
                       figsize: Optional[Tuple[float, float]] = None,
                       sharex: bool = True,
                       sharey: bool = False) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create figure with subplots for trace plotting.
    
    Parameters
    ----------
    n_subplots : int
        Number of subplots
    layout : tuple, optional
        (n_rows, n_cols). If None, automatically determined
    figsize : tuple, optional
        Figure size. If None, automatically determined
    sharex : bool
        Share x-axis across subplots
    sharey : bool
        Share y-axis across subplots
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    axes : ndarray
        Array of axes objects
    """
    # Determine layout
    if layout is None:
        n_cols = int(np.ceil(np.sqrt(n_subplots)))
        n_rows = int(np.ceil(n_subplots / n_cols))
    else:
        n_rows, n_cols = layout
        
    # Determine figure size
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
        
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            sharex=sharex, sharey=sharey)
                            
    # Flatten axes array
    if n_subplots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    # Remove extra subplots
    for i in range(n_subplots, len(axes)):
        fig.delaxes(axes[i])
        
    return fig, axes[:n_subplots]


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example data
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    noise = 0.2 * np.random.randn(len(t))
    data = signal + noise
    
    # Create figure with multiple examples
    fig, axes = create_trace_figure(6, layout=(3, 2), figsize=(12, 10))
    
    # 1. Simple trace
    plot_trace(t, data, ax=axes[0], title='Simple Trace')
    
    # 2. Multiple traces
    data_multi = np.array([data + i*0.5 for i in range(3)])
    plot_multi_trace(t, data_multi, ax=axes[1], 
                    labels=['Trace 1', 'Trace 2', 'Trace 3'],
                    title='Multiple Traces')
                    
    # 3. Trace with events
    events = np.array([2, 4, 6, 8])
    plot_trace_with_events(t, data, events, ax=axes[2],
                          event_style='markers',
                          title='Trace with Events')
                          
    # 4. Heatmap
    data_2d = np.array([data + i*0.2 + 0.1*np.random.randn(len(t)) 
                       for i in range(20)])
    plot_trace_heatmap(t, data_2d, ax=axes[3], 
                      title='Trace Heatmap')
                      
    # 5. Colored trace
    color_data = np.abs(data)
    plot_trace_colored(t, data, color_data, ax=axes[4],
                      title='Color-coded Trace')
                      
    # 6. Shaded error
    mean_trace = signal
    error = 0.2 * np.ones_like(t)
    plot_trace_shaded(t, mean_trace, error, ax=axes[5],
                     title='Trace with Error Shading')
                     
    plt.tight_layout()
    plt.show()