"""
Visualization Utilities for Electrophysiology Analysis
Common plotting functions and styles
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_figure_style(style: str = 'paper'):
    """Setup figure style for publications or presentations."""
    styles = {
        'paper': {
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
        },
        'presentation': {
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2.0,
            'patch.linewidth': 1.5,
        },
        'poster': {
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'figure.titlesize': 24,
            'axes.linewidth': 2.0,
            'lines.linewidth': 3.0,
            'patch.linewidth': 2.0,
        }
    }
    
    if style in styles:
        plt.rcParams.update(styles[style])

def create_multi_panel_figure(n_panels: int, 
                            fig_size: Optional[Tuple[float, float]] = None,
                            layout: Union[str, Tuple[int, int]] = 'auto') -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a multi-panel figure with optimal layout."""
    if layout == 'auto':
        # Determine best layout
        if n_panels <= 3:
            rows, cols = 1, n_panels
        elif n_panels == 4:
            rows, cols = 2, 2
        elif n_panels <= 6:
            rows, cols = 2, 3
        elif n_panels <= 9:
            rows, cols = 3, 3
        elif n_panels <= 12:
            rows, cols = 3, 4
        else:
            cols = int(np.ceil(np.sqrt(n_panels)))
            rows = int(np.ceil(n_panels / cols))
    else:
        rows, cols = layout
        
    if fig_size is None:
        fig_size = (cols * 4, rows * 3)
        
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    
    # Flatten axes array for easy indexing
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
    # Hide extra axes
    all_axes = axes if isinstance(axes, list) else [axes]
    for i in range(n_panels, len(all_axes)):
        all_axes[i].set_visible(False)
        
    return fig, all_axes[:n_panels]

def plot_trace_with_events(ax: plt.Axes, 
                          time: np.ndarray, 
                          signal: np.ndarray,
                          events: Optional[Union[np.ndarray, List]] = None,
                          event_color: str = 'red',
                          signal_color: str = 'black',
                          signal_label: str = 'Signal',
                          event_label: str = 'Events',
                          event_marker: str = 'o',
                          event_size: float = 50):
    """Plot signal trace with marked events."""
    # Plot signal
    ax.plot(time, signal, color=signal_color, linewidth=0.8, 
            label=signal_label, alpha=0.8)
    
    # Plot events if provided
    if events is not None and len(events) > 0:
        if isinstance(events[0], (int, np.integer)):
            # Events are indices
            event_times = time[events]
            event_values = signal[events]
        else:
            # Events are times
            event_times = np.array(events)
            # Find nearest signal values
            event_values = []
            for t in event_times:
                idx = np.argmin(np.abs(time - t))
                event_values.append(signal[idx])
            event_values = np.array(event_values)
        
        ax.scatter(event_times, event_values, color=event_color, 
                  s=event_size, marker=event_marker,
                  zorder=5, label=event_label, alpha=0.8,
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    ax.legend(loc='best', framealpha=0.8)
    ax.grid(True, alpha=0.3)

def plot_histogram_with_stats(ax: plt.Axes, 
                             data: np.ndarray, 
                             bins: Union[int, str, np.ndarray] = 'auto',
                             show_mean: bool = True, 
                             show_median: bool = True,
                             show_std: bool = False,
                             color: str = 'skyblue', 
                             alpha: float = 0.7,
                             label: Optional[str] = None):
    """Plot histogram with statistical indicators."""
    # Remove NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center')
        return None, None
    
    # Plot histogram
    counts, bins, patches = ax.hist(data_clean, bins=bins, color=color, 
                                   alpha=alpha, edgecolor='black', 
                                   linewidth=0.5, label=label)
    
    # Add statistical indicators
    y_max = ax.get_ylim()[1]
    
    if show_mean:
        mean_val = np.mean(data_clean)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        
    if show_median:
        median_val = np.median(data_clean)
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.2f}')
        
    if show_std:
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        ax.axvspan(mean_val - std_val, mean_val + std_val, 
                  alpha=0.2, color='gray', label=f'±1 SD')
    
    ax.set_ylabel('Count')
    ax.legend(loc='best', framealpha=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    return counts, bins

def plot_raster(ax: plt.Axes, 
                spike_times_list: List[np.ndarray], 
                duration: Optional[float] = None,
                color: Union[str, List[str]] = 'black', 
                marker: str = '|',
                markersize: float = 10,
                linewidth: float = 1.5,
                trial_labels: Optional[List[str]] = None):
    """Plot raster plot of spike trains."""
    n_trials = len(spike_times_list)
    
    # Handle colors
    if isinstance(color, str):
        colors = [color] * n_trials
    else:
        colors = color
        
    # Plot each trial
    for i, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            y_vals = np.ones_like(spike_times) * i
            ax.scatter(spike_times, y_vals, marker=marker, 
                      s=markersize**2, c=colors[i % len(colors)],
                      linewidths=linewidth, edgecolors='none')
    
    # Set limits and labels
    ax.set_ylim(-0.5, n_trials - 0.5)
    ax.set_ylabel('Trial/Unit #')
    ax.set_xlabel('Time (s)')
    
    if duration is not None:
        ax.set_xlim(0, duration)
    
    # Add trial labels if provided
    if trial_labels is not None:
        ax.set_yticks(range(n_trials))
        ax.set_yticklabels(trial_labels)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis so trial 0 is at top
    ax.invert_yaxis()

def plot_heatmap(ax: plt.Axes, 
                data: np.ndarray, 
                x_labels: Optional[List] = None,
                y_labels: Optional[List] = None, 
                cmap: str = 'viridis',
                cbar_label: str = '',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                aspect: str = 'auto',
                interpolation: str = 'nearest'):
    """Plot heatmap with optional labels."""
    # Create the heatmap
    im = ax.imshow(data, cmap=cmap, aspect=aspect, 
                   interpolation=interpolation,
                   vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    
    # Add grid
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2)
    
    return im

def plot_phase_space(ax: plt.Axes, 
                    x: np.ndarray, 
                    y: np.ndarray,
                    trajectory_color: Union[str, np.ndarray] = 'blue',
                    colormap: str = 'viridis',
                    start_marker: bool = True,
                    end_marker: bool = True,
                    arrow_frequency: int = 0):
    """Plot phase space trajectory."""
    # Plot trajectory
    if isinstance(trajectory_color, str):
        ax.plot(x, y, color=trajectory_color, linewidth=1, alpha=0.7)
    else:
        # Color by time or other variable
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap=colormap)
        lc.set_array(trajectory_color)
        lc.set_linewidth(1)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Time')
    
    # Add markers
    if start_marker:
        ax.scatter(x[0], y[0], color='green', s=100, marker='o', 
                  zorder=5, label='Start', edgecolors='black')
        
    if end_marker:
        ax.scatter(x[-1], y[-1], color='red', s=100, marker='s',
                  zorder=5, label='End', edgecolors='black')
    
    # Add arrows to show direction
    if arrow_frequency > 0:
        indices = np.arange(0, len(x)-1, arrow_frequency)
        for i in indices:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ax.arrow(x[i], y[i], dx*0.5, dy*0.5,
                    head_width=0.02, head_length=0.03,
                    fc='black', ec='black', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    if start_marker or end_marker:
        ax.legend(loc='best', framealpha=0.8)

def add_scale_bar(ax: plt.Axes, 
                 x_size: Optional[float] = None, 
                 y_size: Optional[float] = None,
                 x_label: str = '', 
                 y_label: str = '',
                 location: str = 'lower right', 
                 pad: float = 0.05,
                 bar_color: str = 'black',
                 text_color: str = 'black',
                 linewidth: float = 2):
    """Add scale bar to plot."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Default sizes if not specified
    if x_size is None and y_size is None:
        x_size = x_range * 0.1
    
    # Determine position based on location
    locations = {
        'lower right': (1 - pad, pad),
        'lower left': (pad, pad),
        'upper right': (1 - pad, 1 - pad),
        'upper left': (pad, 1 - pad)
    }
    
    if location not in locations:
        location = 'lower right'
        
    rel_x, rel_y = locations[location]
    x_pos = xlim[0] + x_range * rel_x
    y_pos = ylim[0] + y_range * rel_y
    
    # Adjust for bar position
    if 'right' in location and x_size:
        x_pos -= x_size
    if 'upper' in location and y_size:
        y_pos -= y_size
    
    # Draw scale bars
    if x_size is not None:
        ax.plot([x_pos, x_pos + x_size], [y_pos, y_pos], 
               color=bar_color, linewidth=linewidth, solid_capstyle='butt')
        if x_label:
            ax.text(x_pos + x_size/2, y_pos - y_range*0.02,
                   x_label, ha='center', va='top', color=text_color)
    
    if y_size is not None:
        ax.plot([x_pos, x_pos], [y_pos, y_pos + y_size], 
               color=bar_color, linewidth=linewidth, solid_capstyle='butt')
        if y_label:
            ax.text(x_pos - x_range*0.02, y_pos + y_size/2,
                   y_label, ha='right', va='center', 
                   color=text_color, rotation=90)

def save_figure(fig: plt.Figure, 
               filename: str, 
               dpi: int = 300, 
               tight_layout: bool = True,
               transparent: bool = False,
               **kwargs):
    """Save figure with consistent settings."""
    if tight_layout:
        try:
            fig.tight_layout()
        except:
            pass  # Sometimes tight_layout fails with complex layouts
        
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', 
                exist_ok=True)
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                transparent=transparent, **kwargs)
    
def create_summary_table(ax: plt.Axes, 
                        data: Dict[str, Any], 
                        title: str = 'Summary Statistics',
                        row_colors: Optional[List[str]] = None,
                        col_widths: Optional[List[float]] = None):
    """Create a summary statistics table in an axis."""
    ax.axis('off')
    
    # Prepare data for table
    rows = []
    row_labels = []
    
    for key, value in data.items():
        label = key.replace('_', ' ').title()
        
        if isinstance(value, (int, float, np.number)):
            if abs(value) < 0.01 or abs(value) > 1000:
                value_str = f'{value:.2e}'
            else:
                value_str = f'{value:.3f}'
        elif isinstance(value, (list, np.ndarray)) and len(value) <= 3:
            value_str = ', '.join([f'{v:.2f}' for v in value])
        else:
            value_str = str(value)
            
        rows.append([value_str])
        row_labels.append(label)
    
    # Create table
    if col_widths is None:
        col_widths = [0.7, 0.3]
        
    table = ax.table(cellText=rows, 
                    rowLabels=row_labels,
                    cellLoc='center',
                    rowLoc='right',
                    loc='center',
                    colWidths=col_widths)
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Apply row colors if specified
    if row_colors:
        for i, color in enumerate(row_colors):
            if i < len(rows):
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, -1)].set_facecolor(color)
    
    # Style header
    for i in range(len(row_labels)):
        table[(i+1, -1)].set_text_props(weight='bold')
    
    # Add title
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
           ha='center', va='top', fontsize=12, weight='bold')
    
    return table

# Predefined color schemes
COLOR_SCHEMES = {
    'categorical': {
        'colors': sns.color_palette('Set2'),
        'description': 'Distinct colors for categorical data'
    },
    'sequential': {
        'colors': sns.color_palette('viridis'),
        'description': 'Sequential colors for ordered data'
    },
    'diverging': {
        'colors': sns.color_palette('RdBu'),
        'description': 'Diverging colors with neutral center'
    },
    'neural': {
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'description': 'Colors optimized for neural data'
    },
    'conditions': {
        'colors': sns.color_palette('Set1'),
        'description': 'High contrast for experimental conditions'
    }
}

def get_color_palette(name: str = 'categorical', n_colors: Optional[int] = None) -> List:
    """Get a color palette by name."""
    if name in COLOR_SCHEMES:
        colors = COLOR_SCHEMES[name]['colors']
    else:
        # Try to use seaborn palette
        try:
            colors = sns.color_palette(name)
        except:
            colors = COLOR_SCHEMES['categorical']['colors']
    
    if n_colors is not None:
        # Repeat colors if needed
        while len(colors) < n_colors:
            colors = colors + colors
        colors = colors[:n_colors]
    
    return colors

# Export commonly used functions
__all__ = [
    'setup_figure_style',
    'create_multi_panel_figure',
    'plot_trace_with_events',
    'plot_histogram_with_stats',
    'plot_raster',
    'plot_heatmap',
    'plot_phase_space',
    'add_scale_bar',
    'save_figure',
    'create_summary_table',
    'get_color_palette',
    'COLOR_SCHEMES'
]
