"""
Advanced Synaptic Analysis Visualization Module
==============================================

Enhanced visualization utilities for synaptic analysis using Matplotlib and Seaborn.
Provides publication-quality figures with statistical overlays.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from scipy import stats

# Set default style
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

# Custom color palettes for electrophysiology
EPHYS_COLORS = {
    'trace': '#2C3E50',
    'events': '#E74C3C',
    'baseline': '#95A5A6',
    'fit': '#27AE60',
    'confidence': '#3498DB',
    'histogram': '#9B59B6'
}


class SynapticVisualizer:
    """Advanced visualization class for synaptic analysis."""
    
    def __init__(self, style: str = 'publication', dpi: int = 300):
        """
        Initialize visualizer with style preferences.
        
        Parameters
        ----------
        style : str
            Visualization style ('publication', 'presentation', 'notebook')
        dpi : int
            Resolution for saved figures
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
        
    def _setup_style(self):
        """Configure visualization style based on preference."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 8,
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'figure.titlesize': 12,
                'lines.linewidth': 1.0,
                'lines.markersize': 4
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'lines.linewidth': 2.0,
                'lines.markersize': 8
            })
        else:  # notebook
            plt.rcParams.update({
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 1.5,
                'lines.markersize': 6
            })
            
    def plot_event_summary(self, events: List[Any], 
                          signal: np.ndarray,
                          sampling_rate: float,
                          figsize: Tuple[int, int] = (14, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive event analysis summary figure.
        
        Parameters
        ----------
        events : list
            Detected synaptic events
        signal : np.ndarray
            Preprocessed signal
        sampling_rate : float
            Sampling rate in Hz
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1])
        
        # Main trace
        ax_trace = fig.add_subplot(gs[0, :])
        self._plot_trace_with_events(ax_trace, signal, events, sampling_rate)
        
        # Event properties
        ax_amp = fig.add_subplot(gs[1, 0])
        ax_iei = fig.add_subplot(gs[1, 1])
        ax_freq = fig.add_subplot(gs[1, 2])
        
        self._plot_amplitude_distribution(ax_amp, events)
        self._plot_iei_distribution(ax_iei, events)
        self._plot_frequency_histogram(ax_freq, events, signal.shape[0] / sampling_rate)
        
        # Kinetics
        ax_rise = fig.add_subplot(gs[2, 0])
        ax_decay = fig.add_subplot(gs[2, 1])
        ax_area = fig.add_subplot(gs[2, 2])
        
        self._plot_rise_time_distribution(ax_rise, events)
        self._plot_decay_distribution(ax_decay, events)
        self._plot_area_distribution(ax_area, events)
        
        # Correlations
        ax_corr1 = fig.add_subplot(gs[3, 0])
        ax_corr2 = fig.add_subplot(gs[3, 1])
        ax_summary = fig.add_subplot(gs[3, 2])
        
        self._plot_amplitude_vs_iei(ax_corr1, events)
        self._plot_amplitude_vs_decay(ax_corr2, events)
        self._plot_summary_table(ax_summary, events, signal, sampling_rate)
        
        # Overall title
        fig.suptitle('Synaptic Event Analysis Summary', fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def _plot_trace_with_events(self, ax: plt.Axes, signal: np.ndarray, 
                               events: List[Any], sampling_rate: float):
        """Plot signal trace with detected events marked."""
        time = np.arange(len(signal)) / sampling_rate
        
        # Plot signal
        ax.plot(time, signal, color=EPHYS_COLORS['trace'], linewidth=0.5, 
                alpha=0.8, label='Signal')
        
        # Mark events
        if events:
            event_times = [e.time for e in events]
            event_indices = [int(e.time * sampling_rate) for e in events]
            event_amplitudes = [signal[idx] if idx < len(signal) else np.nan 
                              for idx in event_indices]
            
            ax.scatter(event_times, event_amplitudes, color=EPHYS_COLORS['events'], 
                      s=30, zorder=5, alpha=0.7, label=f'Events (n={len(events)})')
            
            # Add example event windows
            if len(events) > 0:
                for i in range(min(3, len(events))):  # Show first 3 events
                    event = events[i]
                    window_start = max(0, event.time - 0.01)
                    window_end = min(time[-1], event.time + 0.04)
                    
                    rect = Rectangle((window_start, ax.get_ylim()[0]), 
                                   window_end - window_start,
                                   ax.get_ylim()[1] - ax.get_ylim()[0],
                                   alpha=0.1, color=EPHYS_COLORS['events'],
                                   zorder=1)
                    ax.add_patch(rect)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (pA)')
        ax.set_title('Synaptic Current Recording with Detected Events')
        ax.legend(loc='upper right')
        
        # Add scale bar
        self._add_scale_bar(ax, signal, sampling_rate)
        
    def _plot_amplitude_distribution(self, ax: plt.Axes, events: List[Any]):
        """Plot amplitude distribution with statistical overlay."""
        amplitudes = [e.amplitude for e in events if e.amplitude is not None]
        
        if not amplitudes:
            ax.text(0.5, 0.5, 'No amplitude data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Amplitude (pA)')
            ax.set_title('Amplitude Distribution')
            return
            
        # Histogram with KDE
        sns.histplot(amplitudes, bins=30, kde=True, color=EPHYS_COLORS['histogram'],
                    stat='density', ax=ax)
        
        # Add statistics
        mean_amp = np.mean(amplitudes)
        median_amp = np.median(amplitudes)
        
        ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_amp:.1f} pA')
        ax.axvline(median_amp, color='orange', linestyle=':', linewidth=2,
                  label=f'Median: {median_amp:.1f} pA')
        
        # Test for normality
        if len(amplitudes) > 20:
            _, p_value = stats.normaltest(amplitudes)
            normality_text = f'Normality test p={p_value:.3f}'
            ax.text(0.95, 0.95, normality_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Amplitude (pA)')
        ax.set_ylabel('Density')
        ax.set_title('Amplitude Distribution')
        ax.legend()
        
    def _plot_iei_distribution(self, ax: plt.Axes, events: List[Any]):
        """Plot inter-event interval distribution."""
        if len(events) < 2:
            ax.text(0.5, 0.5, 'Insufficient events', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('IEI (ms)')
            ax.set_title('Inter-Event Intervals')
            return
            
        times = [e.time for e in events]
        ieis = np.diff(times) * 1000  # Convert to ms
        
        # Use log scale if range is large
        if np.max(ieis) / np.min(ieis) > 100:
            bins = np.logspace(np.log10(np.min(ieis)), np.log10(np.max(ieis)), 30)
            ax.set_xscale('log')
        else:
            bins = 30
            
        sns.histplot(ieis, bins=bins, kde=False, color=EPHYS_COLORS['histogram'],
                    stat='count', ax=ax)
        
        # Add CV
        cv = np.std(ieis) / np.mean(ieis)
        ax.text(0.95, 0.95, f'CV = {cv:.2f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_xlabel('Inter-Event Interval (ms)')
        ax.set_ylabel('Count')
        ax.set_title('IEI Distribution')
        
    def _plot_frequency_histogram(self, ax: plt.Axes, events: List[Any], 
                                duration: float):
        """Plot instantaneous frequency over time."""
        if len(events) < 2:
            ax.text(0.5, 0.5, 'Insufficient events', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Time (s)')
            ax.set_title('Event Frequency')
            return
            
        # Calculate instantaneous frequency
        times = np.array([e.time for e in events])
        
        # Bin the recording into segments
        n_bins = min(20, len(events) // 5)
        if n_bins < 3:
            n_bins = 3
            
        bins = np.linspace(0, duration, n_bins + 1)
        freq, bin_edges = np.histogram(times, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Convert to Hz
        freq = freq / bin_width
        
        # Plot as step function
        ax.step(bin_centers, freq, where='mid', color=EPHYS_COLORS['histogram'],
               linewidth=2)
        ax.fill_between(bin_centers, freq, step='mid', alpha=0.3,
                       color=EPHYS_COLORS['histogram'])
        
        # Add mean frequency line
        mean_freq = len(events) / duration
        ax.axhline(mean_freq, color='red', linestyle='--',
                  label=f'Mean: {mean_freq:.1f} Hz')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Event Frequency Over Time')
        ax.legend()
        
    def _plot_rise_time_distribution(self, ax: plt.Axes, events: List[Any]):
        """Plot rise time distribution."""
        rise_times = [e.rise_time * 1000 for e in events 
                     if hasattr(e, 'rise_time') and e.rise_time is not None]
        
        if not rise_times:
            ax.text(0.5, 0.5, 'No rise time data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Rise Time (ms)')
            ax.set_title('Rise Time Distribution')
            return
            
        sns.histplot(rise_times, bins=20, kde=True, color=EPHYS_COLORS['histogram'],
                    ax=ax)
        
        ax.axvline(np.mean(rise_times), color='red', linestyle='--',
                  label=f'Mean: {np.mean(rise_times):.2f} ms')
        
        ax.set_xlabel('Rise Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Rise Time Distribution')
        ax.legend()
        
    def _plot_decay_distribution(self, ax: plt.Axes, events: List[Any]):
        """Plot decay tau distribution."""
        decay_taus = [e.decay_tau * 1000 for e in events 
                     if hasattr(e, 'decay_tau') and e.decay_tau is not None]
        
        if not decay_taus:
            ax.text(0.5, 0.5, 'No decay data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Decay τ (ms)')
            ax.set_title('Decay Time Constant')
            return
            
        sns.histplot(decay_taus, bins=20, kde=True, color=EPHYS_COLORS['histogram'],
                    ax=ax)
        
        ax.axvline(np.mean(decay_taus), color='red', linestyle='--',
                  label=f'Mean: {np.mean(decay_taus):.2f} ms')
        
        ax.set_xlabel('Decay τ (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Decay Time Constant Distribution')
        ax.legend()
        
    def _plot_area_distribution(self, ax: plt.Axes, events: List[Any]):
        """Plot event area/charge distribution."""
        areas = [e.area for e in events 
                if hasattr(e, 'area') and e.area is not None]
        
        if not areas:
            ax.text(0.5, 0.5, 'No area data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Area (pA·s)')
            ax.set_title('Event Charge')
            return
            
        sns.histplot(areas, bins=20, kde=True, color=EPHYS_COLORS['histogram'],
                    ax=ax)
        
        ax.axvline(np.mean(areas), color='red', linestyle='--',
                  label=f'Mean: {np.mean(areas):.3f} pA·s')
        
        ax.set_xlabel('Area (pA·s)')
        ax.set_ylabel('Count')
        ax.set_title('Event Charge Distribution')
        ax.legend()
        
    def _plot_amplitude_vs_iei(self, ax: plt.Axes, events: List[Any]):
        """Plot correlation between amplitude and preceding IEI."""
        if len(events) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Preceding IEI (ms)')
            ax.set_title('Amplitude vs IEI')
            return
            
        amplitudes = []
        preceding_ieis = []
        
        for i in range(1, len(events)):
            if events[i].amplitude is not None:
                amplitudes.append(abs(events[i].amplitude))
                preceding_ieis.append((events[i].time - events[i-1].time) * 1000)
                
        if len(amplitudes) > 2:
            # Scatter with regression
            sns.scatterplot(x=preceding_ieis, y=amplitudes, alpha=0.6, ax=ax)
            
            # Add regression line if significant correlation
            if len(amplitudes) > 10:
                r, p = stats.pearsonr(preceding_ieis, amplitudes)
                if p < 0.05:
                    sns.regplot(x=preceding_ieis, y=amplitudes, scatter=False,
                              color='red', ax=ax)
                    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}',
                           transform=ax.transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                           
        ax.set_xlabel('Preceding IEI (ms)')
        ax.set_ylabel('Amplitude (pA)')
        ax.set_title('Short-term Plasticity')
        
    def _plot_amplitude_vs_decay(self, ax: plt.Axes, events: List[Any]):
        """Plot correlation between amplitude and decay tau."""
        amplitudes = []
        decay_taus = []
        
        for event in events:
            if (hasattr(event, 'amplitude') and event.amplitude is not None and
                hasattr(event, 'decay_tau') and event.decay_tau is not None):
                amplitudes.append(abs(event.amplitude))
                decay_taus.append(event.decay_tau * 1000)
                
        if len(amplitudes) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xlabel('Amplitude (pA)')
            ax.set_title('Amplitude vs Decay')
            return
            
        # Scatter with marginal distributions
        sns.scatterplot(x=amplitudes, y=decay_taus, alpha=0.6, ax=ax)
        
        # Add correlation if significant
        if len(amplitudes) > 10:
            r, p = stats.pearsonr(amplitudes, decay_taus)
            if p < 0.05:
                sns.regplot(x=amplitudes, y=decay_taus, scatter=False,
                          color='red', ax=ax)
                ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                       
        ax.set_xlabel('Amplitude (pA)')
        ax.set_ylabel('Decay τ (ms)')
        ax.set_title('Amplitude-Kinetics Relationship')
        
    def _plot_summary_table(self, ax: plt.Axes, events: List[Any], 
                          signal: np.ndarray, sampling_rate: float):
        """Create summary statistics table."""
        ax.axis('off')
        
        # Calculate statistics
        stats_data = []
        
        # Basic stats
        duration = signal.shape[0] / sampling_rate
        stats_data.append(['Recording Duration', f'{duration:.1f} s'])
        stats_data.append(['Number of Events', f'{len(events)}'])
        stats_data.append(['Event Frequency', f'{len(events)/duration:.2f} Hz'])
        
        # Amplitude stats
        amplitudes = [e.amplitude for e in events if e.amplitude is not None]
        if amplitudes:
            stats_data.append(['Mean Amplitude', f'{np.mean(amplitudes):.1f} ± {np.std(amplitudes):.1f} pA'])
            
        # IEI stats
        if len(events) > 1:
            ieis = np.diff([e.time for e in events]) * 1000
            stats_data.append(['Mean IEI', f'{np.mean(ieis):.1f} ± {np.std(ieis):.1f} ms'])
            stats_data.append(['IEI CV', f'{np.std(ieis)/np.mean(ieis):.2f}'])
            
        # Kinetic stats
        decay_taus = [e.decay_tau * 1000 for e in events 
                     if hasattr(e, 'decay_tau') and e.decay_tau is not None]
        if decay_taus:
            stats_data.append(['Mean Decay τ', f'{np.mean(decay_taus):.1f} ± {np.std(decay_taus):.1f} ms'])
            
        # Create table
        table = ax.table(cellText=stats_data,
                        colWidths=[0.6, 0.4],
                        cellLoc='left',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i, (param, value) in enumerate(stats_data):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#F5F5F5')
            
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)
        
    def _add_scale_bar(self, ax: plt.Axes, signal: np.ndarray, 
                     sampling_rate: float):
        """Add scale bar to trace plot."""
        # Calculate appropriate scale
        y_range = np.ptp(signal)
        x_range = signal.shape[0] / sampling_rate
        
        # Y scale (current)
        if y_range > 100:
            y_scale = 50
        elif y_range > 50:
            y_scale = 20
        elif y_range > 20:
            y_scale = 10
        else:
            y_scale = 5
            
        # X scale (time)
        if x_range > 10:
            x_scale = 1.0
        elif x_range > 1:
            x_scale = 0.1
        else:
            x_scale = 0.01
            
        # Position scale bar
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.1
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.1
        
        # Draw scale bar
        ax.plot([x_pos - x_scale, x_pos], [y_pos, y_pos], 'k-', linewidth=2)
        ax.plot([x_pos, x_pos], [y_pos, y_pos + y_scale], 'k-', linewidth=2)
        
        # Labels
        ax.text(x_pos - x_scale/2, y_pos - (ylim[1] - ylim[0]) * 0.02,
               f'{x_scale*1000:.0f} ms' if x_scale < 1 else f'{x_scale:.0f} s',
               ha='center', va='top', fontsize=8)
        ax.text(x_pos + (xlim[1] - xlim[0]) * 0.02, y_pos + y_scale/2,
               f'{y_scale:.0f} pA', ha='left', va='center', fontsize=8)
    
    def plot_io_curve_comparison(self, io_results_list: List[Dict[str, Any]],
                               labels: List[str],
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple input-output curves.
        
        Parameters
        ----------
        io_results_list : list
            List of I/O analysis results
        labels : list
            Labels for each curve
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        colors = sns.color_palette("husl", len(io_results_list))
        
        # Plot each curve
        for i, (results, label) in enumerate(zip(io_results_list, labels)):
            # Data points
            ax1.scatter(results['input'], results['output_normalized'],
                       color=colors[i], s=50, alpha=0.6, edgecolor='black',
                       label=f'{label} (data)')
            
            # Fitted curve
            x_fit = np.linspace(np.min(results['input']), 
                              np.max(results['input']), 200)
            y_fit = results['model_function'](x_fit, *results['parameters'])
            
            ax1.plot(x_fit, y_fit, color=colors[i], linewidth=2,
                    label=f'{label} (R²={results["r_squared"]:.3f})')
            
        ax1.set_xlabel('Input Intensity')
        ax1.set_ylabel('Normalized Output')
        ax1.set_title('Input-Output Curve Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Parameter comparison
        param_names = {
            'sigmoid': ['IC₅₀', 'Slope', 'Max'],
            'hill': ['Vmax', 'Km', 'n'],
            'boltzmann': ['Vmax', 'V₅₀', 'Slope']
        }
        
        # Create parameter comparison dataframe
        param_data = []
        for results, label in zip(io_results_list, labels):
            model = results['model']
            params = results['parameters']
            params_ci = results['parameters_ci']
            
            for i, param in enumerate(params):
                if model in param_names and i < len(param_names[model]):
                    param_name = param_names[model][i]
                else:
                    param_name = f'p{i}'
                    
                param_data.append({
                    'Curve': label,
                    'Parameter': param_name,
                    'Value': param,
                    'CI': params_ci[i] if i < len(params_ci) else np.nan
                })
                
        df_params = pd.DataFrame(param_data)
        
        # Plot parameter comparison
        param_pivot = df_params.pivot(index='Parameter', columns='Curve', values='Value')
        param_pivot.plot(kind='bar', ax=ax2, color=colors[:len(labels)])
        
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameter Comparison')
        ax2.legend(title='Curve')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_publication_figure(self, results: Dict[str, Any],
                                figsize: Tuple[int, int] = (7, 5),
                                panels: List[str] = ['trace', 'histogram'],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-ready figure with selected panels.
        
        Parameters
        ----------
        results : dict
            Analysis results
        figsize : tuple
            Figure size in inches
        panels : list
            Which panels to include
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        # Switch to publication style temporarily
        old_style = self.style
        self.style = 'publication'
        self._setup_style()
        
        n_panels = len(panels)
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, 
                               constrained_layout=True)
        
        if n_panels == 1:
            axes = [axes]
            
        for ax, panel_type in zip(axes, panels):
            if panel_type == 'trace' and 'preprocessed_signal' in results:
                self._plot_trace_with_events(
                    ax, 
                    results['preprocessed_signal'],
                    results.get('events', []),
                    results.get('sampling_rate', 1.0)
                )
            elif panel_type == 'histogram' and 'events' in results:
                self._plot_amplitude_distribution(ax, results['events'])
            elif panel_type == 'io_curve' and 'model_function' in results:
                self._plot_io_curve(ax, results)
                
        # Label panels
        for i, ax in enumerate(axes):
            ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top')
            
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        # Restore original style
        self.style = old_style
        self._setup_style()
        
        return fig
    
    def _plot_io_curve(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot single I/O curve for publication figure."""
        ax.scatter(results['input'], results['output_normalized'],
                  s=30, alpha=0.6, edgecolor='black')
        
        x_fit = np.linspace(np.min(results['input']), 
                          np.max(results['input']), 200)
        y_fit = results['model_function'](x_fit, *results['parameters'])
        
        ax.plot(x_fit, y_fit, 'r-', linewidth=2)
        
        ax.set_xlabel('Stimulation Intensity')
        ax.set_ylabel('Response Amplitude')
        ax.set_title(f'{results["model"].capitalize()} Fit (R² = {results["r_squared"]:.3f})')


# Convenience functions
def plot_synaptic_summary(events: List[Any], signal: np.ndarray, 
                        sampling_rate: float, **kwargs) -> plt.Figure:
    """Quick function to create synaptic analysis summary."""
    visualizer = SynapticVisualizer()
    return visualizer.plot_event_summary(events, signal, sampling_rate, **kwargs)


def compare_io_curves(io_results_list: List[Dict[str, Any]], 
                    labels: List[str], **kwargs) -> plt.Figure:
    """Quick function to compare I/O curves."""
    visualizer = SynapticVisualizer()
    return visualizer.plot_io_curve_comparison(io_results_list, labels, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create example data
    from dataclasses import dataclass
    
    @dataclass
    class MockEvent:
        time: float
        amplitude: float
        decay_tau: float = None
        rise_time: float = None
        area: float = None
        
    # Generate mock events
    n_events = 100
    events = []
    current_time = 0.5
    
    for _ in range(n_events):
        current_time += np.random.exponential(0.1)  # 10 Hz average
        
        event = MockEvent(
            time=current_time,
            amplitude=np.random.normal(-50, 15),
            decay_tau=np.random.normal(0.01, 0.003),
            rise_time=np.random.normal(0.001, 0.0003),
            area=np.random.normal(-1, 0.3)
        )
        events.append(event)
        
    # Create mock signal
    sampling_rate = 20000
    duration = current_time + 0.5
    signal = np.random.randn(int(duration * sampling_rate)) * 5 - 40
    
    # Create visualizer and plot
    viz = SynapticVisualizer(style='notebook')
    fig = viz.plot_event_summary(events, signal, sampling_rate)
    plt.show()