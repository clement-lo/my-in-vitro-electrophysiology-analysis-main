"""
Merged Action Potential Analysis Module
Combines functionality from classic and previous versions
"""

import logging
import yaml
import pyabf
import neo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import detrend, butter, sosfilt, find_peaks
from scipy.optimize import curve_fit
from elephant.statistics import isi
import quantities as pq
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Configuration and Data Loading (from Classic) =============
def load_config(config_file):
    """Load configuration settings from a YAML file and validate."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def load_data(file_path, file_type='abf'):
    """Load intracellular action potential data using appropriate loaders."""
    logger.info(f"Loading data from {file_path} as {file_type} format")
    
    if file_type == 'abf':
        return load_abf_data(file_path)
    elif file_type == 'neo':
        return load_neo_data(file_path)
    elif file_type == 'csv':
        return load_csv_data(file_path)  # Added from previous version
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def load_csv_data(file_path):
    """Load electrophysiology data from a CSV file (from previous version)."""
    try:
        data = pd.read_csv(file_path)
        # Standardize column names
        if 'time' in data.columns:
            data.rename(columns={'time': 'Time'}, inplace=True)
        if 'voltage' in data.columns:
            data.rename(columns={'voltage': 'Voltage'}, inplace=True)
        
        # Convert to format compatible with classic version
        sampling_rate = 1.0 / (data['Time'].iloc[1] - data['Time'].iloc[0])
        signal = data['Voltage'].values
        time = data['Time'].values
        
        logger.info(f"CSV data loaded: {len(signal)} samples at {sampling_rate} Hz")
        return signal, sampling_rate, time
    except Exception as e:
        logger.error(f"Failed to load CSV data: {e}")
        raise

# ============= Detection Methods (from Classic) =============
def detect_action_potentials(signal, sampling_rate, method='threshold', **kwargs):
    """Detect action potentials using a specified detection method."""
    if method == 'threshold':
        threshold = kwargs.get('threshold', -30.0)
        return threshold_spike_detection(signal, sampling_rate, threshold)
    elif method == 'template_matching':
        template = kwargs.get('template')
        return template_matching_spike_detection(signal, template)
    elif method == 'wavelet':
        wavelet = kwargs.get('wavelet', 'db4')
        return wavelet_transform_spike_detection(signal, wavelet)
    elif method == 'machine_learning':
        model = kwargs.get('model')
        return ml_based_spike_detection(signal, model)
    else:
        raise NotImplementedError(f"Spike detection method '{method}' not implemented")

# ============= Analysis Functions (from Previous) =============
def analyze_action_potential_properties(signal, spike_times, sampling_rate):
    """
    Analyze detailed action potential properties (merged from previous version).
    
    Returns:
        DataFrame with AP properties including amplitude, half-width, rise time, etc.
    """
    ap_properties = []
    
    for i, spike_time in enumerate(spike_times):
        spike_idx = int(spike_time * sampling_rate)
        
        # Define window around spike
        window_ms = 10  # 10ms window
        window_samples = int(window_ms * sampling_rate / 1000)
        start_idx = max(0, spike_idx - window_samples)
        end_idx = min(len(signal), spike_idx + window_samples)
        
        ap_waveform = signal[start_idx:end_idx]
        
        if len(ap_waveform) > 0:
            # Calculate properties
            baseline = np.mean(ap_waveform[:window_samples//2])
            peak_value = np.max(ap_waveform)
            amplitude = peak_value - baseline
            
            # Half-width calculation
            half_amp = baseline + amplitude / 2
            above_half = ap_waveform > half_amp
            if np.any(above_half):
                half_width_samples = np.sum(above_half)
                half_width_ms = half_width_samples * 1000 / sampling_rate
            else:
                half_width_ms = np.nan
            
            ap_properties.append({
                'AP_Index': i,
                'Time': spike_time,
                'Amplitude': amplitude,
                'Half_Width_ms': half_width_ms,
                'Peak_Value': peak_value,
                'Baseline': baseline
            })
    
    return pd.DataFrame(ap_properties)

def analyze_spike_train_dynamics(spike_times, duration):
    """
    Analyze spike train dynamics (from previous version).
    
    Returns:
        Dictionary with firing rate, ISI statistics, and other metrics
    """
    if len(spike_times) < 2:
        return {
            'mean_firing_rate': 0,
            'isi_mean': np.nan,
            'isi_cv': np.nan,
            'burst_index': 0
        }
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    # Calculate metrics
    mean_firing_rate = len(spike_times) / duration
    isi_mean = np.mean(isis)
    isi_std = np.std(isis)
    isi_cv = isi_std / isi_mean if isi_mean > 0 else np.nan
    
    # Simple burst detection
    burst_threshold = isi_mean * 0.5
    burst_isis = isis < burst_threshold
    burst_index = np.sum(burst_isis) / len(isis) if len(isis) > 0 else 0
    
    return {
        'mean_firing_rate': mean_firing_rate,
        'isi_mean': isi_mean,
        'isi_cv': isi_cv,
        'burst_index': burst_index,
        'total_spikes': len(spike_times)
    }

# ============= Visualization (Enhanced) =============
def plot_comprehensive_analysis(signal, spike_times, ap_properties, spike_train_metrics, 
                               sampling_rate, save_path=None):
    """Create comprehensive visualization of AP analysis results."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Action Potential Analysis', fontsize=16)
    
    time_vector = np.arange(len(signal)) / sampling_rate
    
    # 1. Raw signal with detected spikes
    ax1 = axes[0, 0]
    ax1.plot(time_vector, signal, 'b-', alpha=0.7, label='Signal')
    ax1.scatter(spike_times, signal[np.array(spike_times * sampling_rate, dtype=int)], 
                c='r', marker='o', s=50, label='Detected APs')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Raw Signal with Detected Action Potentials')
    ax1.legend()
    
    # 2. AP properties scatter
    ax2 = axes[0, 1]
    if not ap_properties.empty:
        scatter = ax2.scatter(ap_properties['Time'], ap_properties['Amplitude'],
                            c=ap_properties['Half_Width_ms'], cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Half-width (ms)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (mV)')
        ax2.set_title('AP Properties Over Time')
    
    # 3. ISI histogram
    ax3 = axes[1, 0]
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        ax3.hist(isis, bins=30, alpha=0.7, color='g')
        ax3.set_xlabel('ISI (s)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'ISI Distribution (CV={spike_train_metrics["isi_cv"]:.2f})')
    
    # 4. Firing rate over time
    ax4 = axes[1, 1]
    if len(spike_times) > 5:
        window = 1.0  # 1 second window
        bins = np.arange(0, spike_times[-1] + window, window)
        hist, _ = np.histogram(spike_times, bins=bins)
        ax4.plot(bins[:-1], hist / window, 'k-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Firing Rate (Hz)')
        ax4.set_title('Instantaneous Firing Rate')
    
    # 5. AP waveform overlay
    ax5 = axes[2, 0]
    plot_ap_waveform_overlay(signal, spike_times, sampling_rate, ax5)
    
    # 6. Phase plot
    ax6 = axes[2, 1]
    plot_phase_plot(signal, spike_times, sampling_rate, ax6)
    
    # 7. Amplitude vs Half-width
    ax7 = axes[3, 0]
    if not ap_properties.empty:
        ax7.scatter(ap_properties['Amplitude'], ap_properties['Half_Width_ms'])
        ax7.set_xlabel('Amplitude (mV)')
        ax7.set_ylabel('Half-width (ms)')
        ax7.set_title('AP Amplitude vs Half-width')
    
    # 8. Summary statistics text
    ax8 = axes[3, 1]
    ax8.axis('off')
    summary_text = f"""
    Summary Statistics:
    ------------------
    Total APs: {spike_train_metrics['total_spikes']}
    Mean Firing Rate: {spike_train_metrics['mean_firing_rate']:.2f} Hz
    Mean ISI: {spike_train_metrics['isi_mean']*1000:.1f} ms
    ISI CV: {spike_train_metrics['isi_cv']:.2f}
    Burst Index: {spike_train_metrics['burst_index']:.2f}
    
    Mean Amplitude: {ap_properties['Amplitude'].mean():.1f} mV
    Mean Half-width: {ap_properties['Half_Width_ms'].mean():.2f} ms
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=12, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()
    
def plot_ap_waveform_overlay(signal, spike_times, sampling_rate, ax):
    """Overlay detected AP waveforms."""
    window_ms = 5
    window_samples = int(window_ms * sampling_rate / 1000)
    
    for spike_time in spike_times[:50]:  # Limit to first 50
        spike_idx = int(spike_time * sampling_rate)
        start_idx = max(0, spike_idx - window_samples)
        end_idx = min(len(signal), spike_idx + window_samples)
        
        waveform = signal[start_idx:end_idx]
        time_ms = np.arange(len(waveform)) * 1000 / sampling_rate - window_ms
        ax.plot(time_ms, waveform, 'b-', alpha=0.3)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('AP Waveform Overlay')

def plot_phase_plot(signal, spike_times, sampling_rate, ax):
    """Create phase plot (dV/dt vs V)."""
    # Calculate derivative
    dv_dt = np.gradient(signal) * sampling_rate
    
    # Plot phase trajectory
    ax.plot(signal, dv_dt, 'k-', alpha=0.3, linewidth=0.5)
    
    # Highlight AP trajectories
    for spike_time in spike_times[:20]:  # Limit to first 20
        spike_idx = int(spike_time * sampling_rate)
        window = int(0.005 * sampling_rate)  # 5ms window
        start_idx = max(0, spike_idx - window)
        end_idx = min(len(signal), spike_idx + window)
        
        ax.plot(signal[start_idx:end_idx], dv_dt[start_idx:end_idx], 'r-', linewidth=2)
    
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('dV/dt (mV/ms)')
    ax.set_title('Phase Plot')

# ============= Main Analysis Pipeline =============
def run_complete_analysis(file_path, config_path=None, output_dir='./results'):
    """
    Run complete action potential analysis pipeline.
    
    This is the main entry point that combines all functionality.
    """
    # Load configuration if provided
    config = load_config(config_path) if config_path else {}
    
    # Determine file type
    file_type = 'csv' if file_path.endswith('.csv') else config.get('file_type', 'abf')
    
    # Load data
    signal, sampling_rate, time = load_data(file_path, file_type)
    
    # Preprocess signal
    if config.get('preprocess', True):
        signal = preprocess_signal(signal, sampling_rate)
    
    # Detect action potentials
    method = config.get('detection_method', 'threshold')
    detection_params = config.get('detection_params', {})
    spike_times = detect_action_potentials(signal, sampling_rate, method, **detection_params)
    
    # Analyze AP properties
    ap_properties = analyze_action_potential_properties(signal, spike_times, sampling_rate)
    
    # Analyze spike train dynamics
    duration = len(signal) / sampling_rate
    spike_train_metrics = analyze_spike_train_dynamics(spike_times, duration)
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'comprehensive_analysis.png')
    plot_comprehensive_analysis(signal, spike_times, ap_properties, 
                               spike_train_metrics, sampling_rate, plot_path)
    
    # Save results
    results = {
        'spike_times': spike_times,
        'ap_properties': ap_properties,
        'spike_train_metrics': spike_train_metrics,
        'sampling_rate': sampling_rate,
        'file_path': file_path
    }
    
    # Save as pickle for full data preservation
    import pickle
    with open(os.path.join(output_dir, 'analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as CSV
    ap_properties.to_csv(os.path.join(output_dir, 'ap_properties.csv'), index=False)
    
    # Save summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"Action Potential Analysis Summary\n")
        f.write(f"================================\n")
        f.write(f"File: {file_path}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
        f.write(f"Sampling Rate: {sampling_rate} Hz\n")
        f.write(f"\nSpike Train Metrics:\n")
        for key, value in spike_train_metrics.items():
            f.write(f"  {key}: {value:.3f}\n")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Action Potential Analysis')
    parser.add_argument('file_path', help='Path to data file')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_complete_analysis(args.file_path, args.config, args.output)
