"""
Merged Synaptic Analysis Module
Combines event detection from classic with I/O analysis from previous
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
import logging
from typing import Dict, List, Tuple, Optional
import os

# Import the unified loader
from utils.unified_loader import load_data

logger = logging.getLogger(__name__)

# ============= Event Detection (from Classic) =============

def detect_synaptic_events(signal: np.ndarray, 
                          sampling_rate: float,
                          event_type: str = 'both',
                          threshold_factor: float = 3.0) -> Dict:
    """
    Detect synaptic events (EPSCs/IPSCs) using advanced threshold detection.
    
    Args:
        signal: Voltage or current trace
        sampling_rate: Sampling frequency in Hz
        event_type: 'epsc', 'ipsc', or 'both'
        threshold_factor: Detection threshold as multiple of noise SD
        
    Returns:
        Dictionary containing event times, amplitudes, and properties
    """
    # Baseline correction
    baseline = np.median(signal)
    signal_corrected = signal - baseline
    
    # Calculate noise level
    noise_std = np.std(signal_corrected[:int(0.1 * len(signal_corrected))])
    
    # Set thresholds based on event type
    if event_type == 'epsc':
        threshold = -threshold_factor * noise_std
        peaks, properties = find_peaks(-signal_corrected, height=-threshold)
    elif event_type == 'ipsc':
        threshold = threshold_factor * noise_std
        peaks, properties = find_peaks(signal_corrected, height=threshold)
    else:  # both
        epsc_peaks, epsc_props = find_peaks(-signal_corrected, height=threshold_factor * noise_std)
        ipsc_peaks, ipsc_props = find_peaks(signal_corrected, height=threshold_factor * noise_std)
        
        return {
            'epsc_events': analyze_event_properties(signal_corrected, epsc_peaks, sampling_rate, 'epsc'),
            'ipsc_events': analyze_event_properties(signal_corrected, ipsc_peaks, sampling_rate, 'ipsc')
        }
    
    return analyze_event_properties(signal_corrected, peaks, sampling_rate, event_type)

def analyze_event_properties(signal: np.ndarray, 
                           event_indices: np.ndarray,
                           sampling_rate: float,
                           event_type: str) -> pd.DataFrame:
    """Analyze detailed properties of detected synaptic events."""
    events = []
    
    for idx in event_indices:
        # Define analysis window
        window_samples = int(0.1 * sampling_rate)  # 100ms window
        start = max(0, idx - window_samples // 2)
        end = min(len(signal), idx + window_samples // 2)
        
        event_trace = signal[start:end]
        
        # Calculate properties
        amplitude = np.abs(signal[idx])
        
        # Rise time (10-90%)
        if event_type == 'epsc':
            peak_val = np.min(event_trace)
        else:
            peak_val = np.max(event_trace)
            
        rise_10 = 0.1 * peak_val
        rise_90 = 0.9 * peak_val
        
        # Find rise time points
        rise_indices = np.where((event_trace > rise_10) & (event_trace < rise_90))[0]
        if len(rise_indices) > 1:
            rise_time = (rise_indices[-1] - rise_indices[0]) / sampling_rate * 1000  # ms
        else:
            rise_time = np.nan
            
        # Decay time constant (simplified)
        decay_start = idx - start
        if decay_start < len(event_trace) - 1:
            decay_trace = event_trace[decay_start:]
            decay_to_37 = peak_val * 0.37
            decay_indices = np.where(np.abs(decay_trace) < np.abs(decay_to_37))[0]
            if len(decay_indices) > 0:
                decay_tau = decay_indices[0] / sampling_rate * 1000  # ms
            else:
                decay_tau = np.nan
        else:
            decay_tau = np.nan
            
        events.append({
            'time': idx / sampling_rate,
            'amplitude': amplitude,
            'rise_time_ms': rise_time,
            'decay_tau_ms': decay_tau,
            'event_type': event_type
        })
    
    return pd.DataFrame(events)

# ============= Input-Output Analysis (from Previous) =============

def sigmoidal(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Sigmoidal function for I/O curve fitting.
    y = a + (b - a) / (1 + exp((c - x) / d))
    
    Parameters:
        a: minimum response
        b: maximum response
        c: half-maximal stimulus (EC50/IC50)
        d: slope factor
    """
    return a + (b - a) / (1 + np.exp((c - x) / d))

def analyze_input_output(stimulus_intensities: np.ndarray,
                        responses: np.ndarray,
                        response_errors: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze synaptic input-output relationships.
    
    Args:
        stimulus_intensities: Array of stimulus intensities
        responses: Array of response amplitudes
        response_errors: Optional array of response standard errors
        
    Returns:
        Dictionary with fit parameters and analysis results
    """
    # Initial parameter guess
    a0 = np.min(responses)
    b0 = np.max(responses)
    c0 = np.median(stimulus_intensities)
    d0 = (np.max(stimulus_intensities) - np.min(stimulus_intensities)) / 10
    
    try:
        # Fit sigmoidal curve
        popt, pcov = curve_fit(sigmoidal, stimulus_intensities, responses,
                              p0=[a0, b0, c0, d0],
                              bounds=([0, 0, 0, 0.1], [np.inf, np.inf, np.inf, np.inf]))
        
        # Calculate fit quality
        residuals = responses - sigmoidal(stimulus_intensities, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((responses - np.mean(responses))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate dynamic range
        dynamic_range = popt[1] - popt[0]
        
        # Calculate slope at half-max
        slope_at_half = (popt[1] - popt[0]) / (4 * popt[3])
        
        results = {
            'fit_params': {
                'min_response': popt[0],
                'max_response': popt[1],
                'half_max_stimulus': popt[2],
                'slope_factor': popt[3]
            },
            'derived_params': {
                'dynamic_range': dynamic_range,
                'slope_at_half_max': slope_at_half,
                'r_squared': r_squared
            },
            'fit_function': lambda x: sigmoidal(x, *popt),
            'covariance': pcov
        }
        
        logger.info(f"I/O fit successful: R² = {r_squared:.3f}, EC50 = {popt[2]:.2f}")
        
    except Exception as e:
        logger.warning(f"Sigmoidal fit failed: {e}. Using linear fit.")
        # Fallback to linear fit
        z = np.polyfit(stimulus_intensities, responses, 1)
        p = np.poly1d(z)
        
        results = {
            'fit_params': {'slope': z[0], 'intercept': z[1]},
            'fit_function': p,
            'fit_type': 'linear'
        }
    
    return results

# ============= Paired-Pulse Analysis =============

def analyze_paired_pulse(signal: np.ndarray,
                        sampling_rate: float,
                        pulse_interval_ms: float,
                        n_pulses: int = 2) -> Dict:
    """
    Analyze paired-pulse facilitation or depression.
    
    Args:
        signal: Recording containing paired pulse responses
        sampling_rate: Sampling frequency
        pulse_interval_ms: Interval between pulses in milliseconds
        n_pulses: Number of pulses in the train
        
    Returns:
        Dictionary with PPR and analysis results
    """
    # Detect all events
    events = detect_synaptic_events(signal, sampling_rate, event_type='epsc')
    
    if 'epsc_events' in events:
        event_df = events['epsc_events']
    else:
        event_df = events
        
    if len(event_df) < n_pulses:
        logger.warning(f"Found only {len(event_df)} events, expected {n_pulses}")
        return {}
    
    # Calculate expected time between pulses
    expected_interval = pulse_interval_ms / 1000  # Convert to seconds
    
    # Group events into pulse trains
    pulse_trains = []
    i = 0
    while i < len(event_df) - n_pulses + 1:
        train = [event_df.iloc[i]]
        for j in range(1, n_pulses):
            next_event_time = event_df.iloc[i]['time'] + j * expected_interval
            # Find event closest to expected time
            time_diffs = np.abs(event_df['time'] - next_event_time)
            if time_diffs.min() < expected_interval * 0.2:  # 20% tolerance
                closest_idx = time_diffs.idxmin()
                train.append(event_df.iloc[closest_idx])
            else:
                break
                
        if len(train) == n_pulses:
            pulse_trains.append(train)
            i += n_pulses
        else:
            i += 1
    
    # Calculate PPR for each train
    pprs = []
    for train in pulse_trains:
        amplitudes = [event['amplitude'] for event in train]
        ppr = amplitudes[1] / amplitudes[0]
        pprs.append(ppr)
        
    results = {
        'mean_ppr': np.mean(pprs),
        'ppr_sem': np.std(pprs) / np.sqrt(len(pprs)),
        'n_trains': len(pulse_trains),
        'all_pprs': pprs,
        'facilitation': np.mean(pprs) > 1.0
    }
    
    return results

# ============= Comprehensive Visualization =============

def plot_synaptic_analysis(results: Dict,
                          signal: np.ndarray,
                          sampling_rate: float,
                          time: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
    """Create comprehensive synaptic analysis visualization."""
    
    if time is None:
        time = np.arange(len(signal)) / sampling_rate
        
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Raw trace with detected events
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, signal, 'k-', linewidth=0.5, alpha=0.7)
    
    # Mark detected events
    if 'events' in results:
        events = results['events']
        if isinstance(events, pd.DataFrame):
            ax1.scatter(events['time'], 
                       signal[np.array(events['time'] * sampling_rate, dtype=int)],
                       c='r', s=30, alpha=0.6, label='Detected events')
        ax1.legend()
        
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current (pA)')
    ax1.set_title('Raw Trace with Detected Synaptic Events')
    
    # 2. Event amplitude histogram
    ax2 = fig.add_subplot(gs[1, 0])
    if 'events' in results and isinstance(results['events'], pd.DataFrame):
        ax2.hist(results['events']['amplitude'], bins=30, alpha=0.7, color='blue')
        ax2.set_xlabel('Amplitude (pA)')
        ax2.set_ylabel('Count')
        ax2.set_title('Event Amplitude Distribution')
    
    # 3. Inter-event interval histogram
    ax3 = fig.add_subplot(gs[1, 1])
    if 'events' in results and isinstance(results['events'], pd.DataFrame):
        if len(results['events']) > 1:
            ieis = np.diff(results['events']['time'])
            ax3.hist(ieis, bins=30, alpha=0.7, color='green')
            ax3.set_xlabel('Inter-event Interval (s)')
            ax3.set_ylabel('Count')
            ax3.set_title('IEI Distribution')
    
    # 4. Input-Output curve
    ax4 = fig.add_subplot(gs[1, 2])
    if 'io_analysis' in results:
        io = results['io_analysis']
        if 'stimulus' in io and 'response' in io:
            # Plot data points
            ax4.scatter(io['stimulus'], io['response'], s=50, alpha=0.6)
            
            # Plot fit
            if 'fit_function' in io:
                x_fit = np.linspace(np.min(io['stimulus']), np.max(io['stimulus']), 100)
                y_fit = io['fit_function'](x_fit)
                ax4.plot(x_fit, y_fit, 'r-', linewidth=2)
                
            ax4.set_xlabel('Stimulus Intensity')
            ax4.set_ylabel('Response Amplitude')
            ax4.set_title('Input-Output Relationship')
    
    # 5. Event kinetics
    ax5 = fig.add_subplot(gs[2, 0])
    if 'events' in results and isinstance(results['events'], pd.DataFrame):
        events_df = results['events']
        if 'rise_time_ms' in events_df.columns:
            valid_rise = events_df['rise_time_ms'].dropna()
            valid_decay = events_df['decay_tau_ms'].dropna()
            
            if len(valid_rise) > 0 and len(valid_decay) > 0:
                ax5.scatter(valid_rise, valid_decay, alpha=0.5)
                ax5.set_xlabel('Rise Time (ms)')
                ax5.set_ylabel('Decay Tau (ms)')
                ax5.set_title('Event Kinetics')
    
    # 6. Cumulative event plot
    ax6 = fig.add_subplot(gs[2, 1])
    if 'events' in results and isinstance(results['events'], pd.DataFrame):
        event_times = results['events']['time']
        cumulative = np.arange(1, len(event_times) + 1)
        ax6.plot(event_times, cumulative, 'k-', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Cumulative Events')
        ax6.set_title('Cumulative Event Count')
        
        # Add frequency annotation
        if len(event_times) > 1:
            freq = len(event_times) / (event_times.iloc[-1] - event_times.iloc[0])
            ax6.text(0.05, 0.95, f'Mean frequency: {freq:.1f} Hz',
                    transform=ax6.transAxes, va='top')
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*20 + "\n"
    
    if 'events' in results and isinstance(results['events'], pd.DataFrame):
        events_df = results['events']
        summary_text += f"Total events: {len(events_df)}\n"
        summary_text += f"Mean amplitude: {events_df['amplitude'].mean():.1f} pA\n"
        summary_text += f"Amplitude CV: {events_df['amplitude'].std()/events_df['amplitude'].mean():.2f}\n"
        
        if len(events_df) > 1:
            duration = events_df['time'].iloc[-1] - events_df['time'].iloc[0]
            summary_text += f"Mean frequency: {len(events_df)/duration:.1f} Hz\n"
    
    if 'ppr_analysis' in results:
        ppr = results['ppr_analysis']
        summary_text += f"\nPaired-Pulse Ratio: {ppr.get('mean_ppr', 0):.2f}\n"
        summary_text += f"PPR type: {'Facilitation' if ppr.get('facilitation', False) else 'Depression'}\n"
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
             fontsize=11, va='top', family='monospace')
    
    plt.suptitle('Comprehensive Synaptic Analysis', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()

# ============= Main Analysis Pipeline =============

def run_synaptic_analysis(file_path: str,
                         analysis_type: str = 'spontaneous',
                         config: Optional[Dict] = None,
                         output_dir: str = './results') -> Dict:
    """
    Run complete synaptic analysis pipeline.
    
    Args:
        file_path: Path to data file
        analysis_type: 'spontaneous', 'evoked', 'input_output', 'paired_pulse'
        config: Optional configuration dictionary
        output_dir: Directory for saving results
        
    Returns:
        Dictionary with all analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    signal, sampling_rate, time, metadata = load_data(file_path)
    
    results = {
        'metadata': metadata,
        'sampling_rate': sampling_rate
    }
    
    # Run appropriate analysis
    if analysis_type == 'spontaneous':
        # Detect spontaneous events
        events = detect_synaptic_events(signal, sampling_rate, 
                                       event_type=config.get('event_type', 'both'),
                                       threshold_factor=config.get('threshold_factor', 3.0))
        results['events'] = events
        
    elif analysis_type == 'input_output':
        # For I/O analysis, expect config to contain stimulus and response data
        if config and 'stimulus_intensities' in config and 'responses' in config:
            io_results = analyze_input_output(
                config['stimulus_intensities'],
                config['responses'],
                config.get('response_errors')
            )
            results['io_analysis'] = io_results
        else:
            logger.warning("Input-output analysis requires stimulus and response data in config")
            
    elif analysis_type == 'paired_pulse':
        # Paired pulse analysis
        interval_ms = config.get('pulse_interval_ms', 50)
        ppr_results = analyze_paired_pulse(signal, sampling_rate, interval_ms)
        results['ppr_analysis'] = ppr_results
        
    elif analysis_type == 'evoked':
        # Similar to spontaneous but with different parameters
        events = detect_synaptic_events(signal, sampling_rate,
                                       event_type='epsc',
                                       threshold_factor=config.get('threshold_factor', 2.0))
        results['events'] = events
        
    # Create visualizations
    plot_path = os.path.join(output_dir, f'{analysis_type}_analysis.png')
    plot_synaptic_analysis(results, signal, sampling_rate, time, plot_path)
    
    # Save results
    import pickle
    with open(os.path.join(output_dir, 'synaptic_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save event data as CSV if available
    if 'events' in results:
        if isinstance(results['events'], pd.DataFrame):
            results['events'].to_csv(os.path.join(output_dir, 'detected_events.csv'), index=False)
        elif isinstance(results['events'], dict):
            for event_type, events_df in results['events'].items():
                if isinstance(events_df, pd.DataFrame):
                    events_df.to_csv(os.path.join(output_dir, f'{event_type}.csv'), index=False)
    
    logger.info(f"Synaptic analysis complete. Results saved to {output_dir}")
    
    return results

# ============= Example Usage =============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Synaptic Analysis')
    parser.add_argument('file_path', help='Path to data file')
    parser.add_argument('--analysis', default='spontaneous',
                       choices=['spontaneous', 'evoked', 'input_output', 'paired_pulse'],
                       help='Type of analysis')
    parser.add_argument('--output', default='./results', help='Output directory')
    parser.add_argument('--event-type', default='both',
                       choices=['epsc', 'ipsc', 'both'],
                       help='Type of synaptic events to detect')
    
    args = parser.parse_args()
    
    config = {
        'event_type': args.event_type,
        'threshold_factor': 3.0
    }
    
    results = run_synaptic_analysis(args.file_path, args.analysis, config, args.output)
