"""
Complete Synaptic Analysis Module
=================================

A comprehensive module for analyzing synaptic currents and input-output relationships
in electrophysiology data. Supports multiple file formats and analysis methods.

Features:
- Multi-format data loading (PyNWB, PyABF, Neo, CSV, HDF5)
- Event detection and kinetic analysis
- Input-output curve fitting
- Advanced visualization with Matplotlib and Seaborn
- Statistical analysis with confidence intervals
- Comprehensive error handling and validation

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats, optimize
from scipy.signal import butter, sosfilt, find_peaks, detrend
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports with fallback
try:
    import pyabf
    HAS_PYABF = True
except ImportError:
    HAS_PYABF = False
    warnings.warn("PyABF not available. ABF file support disabled.")

try:
    import neo
    HAS_NEO = True
except ImportError:
    HAS_NEO = False
    warnings.warn("Neo not available. Some file formats may not be supported.")

try:
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    HAS_PYNWB = False
    warnings.warn("PyNWB not available. NWB file support disabled.")

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    warnings.warn("h5py not available. HDF5 file support disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for all plots
sns.set_style("whitegrid")
sns.set_context("notebook")


class FileFormat(Enum):
    """Supported file formats for electrophysiology data."""
    ABF = "abf"
    NEO = "neo"
    NWB = "nwb"
    CSV = "csv"
    HDF5 = "hdf5"
    UNKNOWN = "unknown"


class AnalysisType(Enum):
    """Types of synaptic analysis."""
    EVENT_DETECTION = "event_detection"
    INPUT_OUTPUT = "input_output"
    KINETICS = "kinetics"
    COMBINED = "combined"


@dataclass
class SynapticEvent:
    """Data class for individual synaptic events."""
    time: float
    amplitude: float
    rise_time: Optional[float] = None
    decay_tau: Optional[float] = None
    area: Optional[float] = None
    baseline: Optional[float] = None
    peak_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Configuration for synaptic analysis."""
    # Preprocessing
    detrend: bool = True
    filter_type: str = "bandpass"  # "bandpass", "lowpass", "highpass", "none"
    filter_freq_low: float = 1.0  # Hz
    filter_freq_high: float = 1000.0  # Hz
    filter_order: int = 4
    
    # Event detection
    detection_method: str = "threshold"  # "threshold", "template", "deconvolution"
    threshold_std: float = 3.0  # Standard deviations above baseline
    min_event_interval: float = 0.005  # seconds
    baseline_window: float = 0.1  # seconds for baseline estimation
    
    # Kinetics
    kinetic_model: str = "exponential"  # "exponential", "biexponential", "alpha"
    fit_window: float = 0.05  # seconds around event
    
    # Input-Output
    io_model: str = "sigmoid"  # "sigmoid", "linear", "hill", "boltzmann"
    io_normalize: bool = True
    
    # Visualization
    plot_style: str = "seaborn"
    figure_dpi: int = 300
    save_figures: bool = False
    figure_format: str = "png"
    
    # Statistics
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Performance
    parallel_processing: bool = False
    chunk_size: int = 10000  # samples per chunk for large files


class DataLoader:
    """Unified data loader for multiple electrophysiology file formats."""
    
    @staticmethod
    def detect_format(file_path: str) -> FileFormat:
        """Automatically detect file format based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.abf': FileFormat.ABF,
            '.nwb': FileFormat.NWB,
            '.csv': FileFormat.CSV,
            '.h5': FileFormat.HDF5,
            '.hdf5': FileFormat.HDF5,
            '.dat': FileFormat.NEO,
            '.smr': FileFormat.NEO,
            '.axgx': FileFormat.NEO,
            '.axgd': FileFormat.NEO
        }
        return format_map.get(ext, FileFormat.UNKNOWN)
    
    @staticmethod
    def load(file_path: str, 
             format: Optional[FileFormat] = None,
             channel: int = 0) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Load electrophysiology data from various formats.
        
        Parameters
        ----------
        file_path : str
            Path to the data file
        format : FileFormat, optional
            File format. If None, will be auto-detected
        channel : int, default=0
            Channel index for multi-channel recordings
            
        Returns
        -------
        data : np.ndarray
            Signal data
        sampling_rate : float
            Sampling rate in Hz
        metadata : dict
            Additional metadata from the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if format is None:
            format = DataLoader.detect_format(file_path)
            
        logger.info(f"Loading {format.value} file: {file_path}")
        
        if format == FileFormat.ABF:
            return DataLoader._load_abf(file_path, channel)
        elif format == FileFormat.NWB:
            return DataLoader._load_nwb(file_path, channel)
        elif format == FileFormat.CSV:
            return DataLoader._load_csv(file_path)
        elif format == FileFormat.HDF5:
            return DataLoader._load_hdf5(file_path, channel)
        elif format == FileFormat.NEO:
            return DataLoader._load_neo(file_path, channel)
        else:
            raise ValueError(f"Unsupported file format: {format}")
    
    @staticmethod
    def _load_abf(file_path: str, channel: int) -> Tuple[np.ndarray, float, Dict]:
        """Load ABF file using PyABF."""
        if not HAS_PYABF:
            raise ImportError("PyABF is required for ABF file support")
            
        abf = pyabf.ABF(file_path)
        abf.setSweep(0, channel=channel)
        
        metadata = {
            'protocol': abf.protocol,
            'units': abf.sweepUnitsY,
            'channel_count': abf.channelCount,
            'sweep_count': abf.sweepCount,
            'creator': abf.creator,
            'creation_date': abf.abfDateTime
        }
        
        return abf.sweepY, abf.dataRate, metadata
    
    @staticmethod
    def _load_nwb(file_path: str, channel: int) -> Tuple[np.ndarray, float, Dict]:
        """Load NWB file using PyNWB."""
        if not HAS_PYNWB:
            raise ImportError("PyNWB is required for NWB file support")
            
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Try different acquisition types
            acquisition = None
            for acq_name in ['CurrentClampSeries', 'VoltageClampSeries', 
                           'PatchClampSeries', 'TimeSeries']:
                if acq_name in nwbfile.acquisition:
                    acquisition = nwbfile.acquisition[acq_name]
                    break
                    
            if acquisition is None:
                # Try to get the first available acquisition
                acq_keys = list(nwbfile.acquisition.keys())
                if acq_keys:
                    acquisition = nwbfile.acquisition[acq_keys[0]]
                else:
                    raise ValueError("No acquisition data found in NWB file")
            
            data = acquisition.data[:]
            rate = acquisition.rate if hasattr(acquisition, 'rate') else acquisition.starting_time_rate
            
            metadata = {
                'description': acquisition.description,
                'unit': acquisition.unit,
                'electrode': str(acquisition.electrode) if hasattr(acquisition, 'electrode') else None,
                'session_id': nwbfile.session_id,
                'experimenter': nwbfile.experimenter
            }
            
        return data, rate, metadata
    
    @staticmethod
    def _load_csv(file_path: str) -> Tuple[np.ndarray, float, Dict]:
        """Load CSV file with time and signal columns."""
        df = pd.read_csv(file_path)
        
        # Try to identify columns
        time_col = None
        signal_col = None
        
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
            elif any(x in col.lower() for x in ['current', 'voltage', 'signal', 'amplitude']):
                signal_col = col
                
        if time_col is None or signal_col is None:
            # Fallback: assume first column is time, second is signal
            if len(df.columns) >= 2:
                time_col = df.columns[0]
                signal_col = df.columns[1]
            else:
                raise ValueError("CSV must have at least 2 columns (time and signal)")
        
        time = df[time_col].values
        signal = df[signal_col].values
        
        # Calculate sampling rate
        dt = np.mean(np.diff(time))
        sampling_rate = 1.0 / dt
        
        metadata = {
            'columns': list(df.columns),
            'time_column': time_col,
            'signal_column': signal_col,
            'duration': time[-1] - time[0]
        }
        
        return signal, sampling_rate, metadata
    
    @staticmethod
    def _load_hdf5(file_path: str, channel: int) -> Tuple[np.ndarray, float, Dict]:
        """Load HDF5 file."""
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 file support")
            
        with h5py.File(file_path, 'r') as f:
            # Look for common dataset names
            data_paths = ['data', 'signal', 'recording', 'traces']
            data = None
            
            for path in data_paths:
                if path in f:
                    data = f[path][:]
                    if len(data.shape) > 1:
                        data = data[channel]
                    break
                    
            if data is None:
                raise ValueError("No recognized data array found in HDF5 file")
            
            # Look for sampling rate
            rate_attrs = ['sampling_rate', 'fs', 'rate']
            sampling_rate = None
            
            for attr in rate_attrs:
                if attr in f.attrs:
                    sampling_rate = float(f.attrs[attr])
                    break
                    
            if sampling_rate is None:
                raise ValueError("Sampling rate not found in HDF5 file")
            
            metadata = dict(f.attrs)
            
        return data, sampling_rate, metadata
    
    @staticmethod
    def _load_neo(file_path: str, channel: int) -> Tuple[np.ndarray, float, Dict]:
        """Load file using Neo."""
        if not HAS_NEO:
            raise ImportError("Neo is required for this file format")
            
        # Detect Neo IO class based on extension
        ext = os.path.splitext(file_path)[1].lower()
        io_map = {
            '.dat': neo.io.RawBinarySignalIO,
            '.smr': neo.io.Spike2IO,
            '.axgx': neo.io.AxographIO,
            '.axgd': neo.io.AxographIO
        }
        
        io_class = io_map.get(ext)
        if io_class is None:
            raise ValueError(f"Unsupported Neo format: {ext}")
            
        reader = io_class(file_path)
        block = reader.read_block()
        segment = block.segments[0]
        
        if channel >= len(segment.analogsignals):
            raise ValueError(f"Channel {channel} not found. File has {len(segment.analogsignals)} channels.")
            
        signal_obj = segment.analogsignals[channel]
        data = np.array(signal_obj.magnitude).flatten()
        sampling_rate = float(signal_obj.sampling_rate.magnitude)
        
        metadata = {
            'units': str(signal_obj.units),
            'channel_count': len(segment.analogsignals),
            'duration': float(signal_obj.duration.magnitude)
        }
        
        return data, sampling_rate, metadata


class SynapticAnalyzer(ABC):
    """Abstract base class for synaptic analysis."""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.data = None
        self.sampling_rate = None
        self.metadata = {}
        self.results = {}
        
    @abstractmethod
    def analyze(self, data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
        """Perform analysis on the data."""
        pass
    
    @abstractmethod
    def visualize(self, **kwargs) -> None:
        """Visualize analysis results."""
        pass
    
    def preprocess(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Preprocess signal with detrending and filtering.
        
        Parameters
        ----------
        data : np.ndarray
            Raw signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        np.ndarray
            Preprocessed signal
        """
        signal = data.copy()
        
        # Detrend
        if self.config.detrend:
            signal = detrend(signal)
            
        # Filter
        if self.config.filter_type != "none":
            nyquist = sampling_rate / 2
            
            if self.config.filter_type == "bandpass":
                low = self.config.filter_freq_low / nyquist
                high = self.config.filter_freq_high / nyquist
                if low >= 1 or high >= 1:
                    logger.warning("Filter frequencies exceed Nyquist frequency, skipping filter")
                else:
                    sos = butter(self.config.filter_order, [low, high], 
                               btype='bandpass', output='sos')
                    signal = sosfilt(sos, signal)
                    
            elif self.config.filter_type == "lowpass":
                high = self.config.filter_freq_high / nyquist
                if high >= 1:
                    logger.warning("Filter frequency exceeds Nyquist frequency, skipping filter")
                else:
                    sos = butter(self.config.filter_order, high, 
                               btype='lowpass', output='sos')
                    signal = sosfilt(sos, signal)
                    
            elif self.config.filter_type == "highpass":
                low = self.config.filter_freq_low / nyquist
                if low >= 1:
                    logger.warning("Filter frequency exceeds Nyquist frequency, skipping filter")
                else:
                    sos = butter(self.config.filter_order, low, 
                               btype='highpass', output='sos')
                    signal = sosfilt(sos, signal)
                    
        return signal
    
    def calculate_baseline_stats(self, signal: np.ndarray, 
                                window_size: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate baseline mean and standard deviation.
        
        Parameters
        ----------
        signal : np.ndarray
            Signal data
        window_size : int, optional
            Window size in samples for baseline calculation
            
        Returns
        -------
        baseline_mean : float
        baseline_std : float
        """
        if window_size is None:
            window_size = int(self.config.baseline_window * self.sampling_rate)
            
        # Use the first window_size samples for baseline
        baseline = signal[:window_size]
        return np.mean(baseline), np.std(baseline)


class EventDetectionAnalyzer(SynapticAnalyzer):
    """Analyzer for detecting and characterizing synaptic events."""
    
    def analyze(self, data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
        """
        Detect synaptic events and analyze their properties.
        
        Parameters
        ----------
        data : np.ndarray
            Signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        dict
            Analysis results including detected events
        """
        self.data = data
        self.sampling_rate = sampling_rate
        
        # Preprocess
        signal = self.preprocess(data, sampling_rate)
        
        # Detect events
        events = self._detect_events(signal, sampling_rate)
        
        # Analyze event properties
        if len(events) > 0:
            self._analyze_event_properties(events, signal, sampling_rate)
            
        # Calculate statistics
        stats = self._calculate_event_statistics(events)
        
        self.results = {
            'events': events,
            'statistics': stats,
            'preprocessed_signal': signal
        }
        
        return self.results
    
    def _detect_events(self, signal: np.ndarray, sampling_rate: float) -> List[SynapticEvent]:
        """Detect synaptic events using configured method."""
        if self.config.detection_method == "threshold":
            return self._threshold_detection(signal, sampling_rate)
        elif self.config.detection_method == "template":
            return self._template_detection(signal, sampling_rate)
        elif self.config.detection_method == "deconvolution":
            return self._deconvolution_detection(signal, sampling_rate)
        else:
            raise ValueError(f"Unknown detection method: {self.config.detection_method}")
    
    def _threshold_detection(self, signal: np.ndarray, 
                           sampling_rate: float) -> List[SynapticEvent]:
        """Threshold-based event detection."""
        # Calculate baseline statistics
        baseline_mean, baseline_std = self.calculate_baseline_stats(signal)
        
        # Set threshold
        threshold = baseline_mean - (self.config.threshold_std * baseline_std)
        
        # Find peaks (negative for IPSCs/EPSCs)
        min_distance = int(self.config.min_event_interval * sampling_rate)
        peaks, properties = find_peaks(-signal, 
                                     height=-threshold,
                                     distance=min_distance,
                                     prominence=baseline_std)
        
        # Create event objects
        events = []
        for i, peak_idx in enumerate(peaks):
            event = SynapticEvent(
                time=peak_idx / sampling_rate,
                amplitude=signal[peak_idx] - baseline_mean,
                baseline=baseline_mean,
                peak_index=peak_idx,
                metadata={'prominence': properties['prominences'][i]}
            )
            events.append(event)
            
        return events
    
    def _template_detection(self, signal: np.ndarray, 
                          sampling_rate: float) -> List[SynapticEvent]:
        """Template matching for event detection."""
        # This is a placeholder for template-based detection
        # In practice, you would use a library template or learn from data
        logger.warning("Template detection not fully implemented, using threshold method")
        return self._threshold_detection(signal, sampling_rate)
    
    def _deconvolution_detection(self, signal: np.ndarray, 
                               sampling_rate: float) -> List[SynapticEvent]:
        """Deconvolution-based event detection."""
        # This is a placeholder for deconvolution-based detection
        logger.warning("Deconvolution detection not fully implemented, using threshold method")
        return self._threshold_detection(signal, sampling_rate)
    
    def _analyze_event_properties(self, events: List[SynapticEvent], 
                                signal: np.ndarray, 
                                sampling_rate: float) -> None:
        """Analyze kinetic properties of detected events."""
        window_samples = int(self.config.fit_window * sampling_rate)
        
        for event in events:
            if event.peak_index is None:
                continue
                
            # Extract event window
            start = max(0, event.peak_index - window_samples // 2)
            end = min(len(signal), event.peak_index + window_samples // 2)
            event_signal = signal[start:end]
            time_vector = np.arange(len(event_signal)) / sampling_rate
            
            # Fit kinetics
            try:
                if self.config.kinetic_model == "exponential":
                    params = self._fit_exponential(time_vector, event_signal, event)
                    if params is not None:
                        event.decay_tau = params[1]
                elif self.config.kinetic_model == "biexponential":
                    params = self._fit_biexponential(time_vector, event_signal, event)
                    if params is not None:
                        event.decay_tau = params[1]  # Fast component
                        event.metadata['decay_tau_slow'] = params[3]
                elif self.config.kinetic_model == "alpha":
                    params = self._fit_alpha(time_vector, event_signal, event)
                    if params is not None:
                        event.rise_time = params[1]
                        event.decay_tau = params[2]
            except Exception as e:
                logger.debug(f"Failed to fit kinetics for event at {event.time}: {e}")
                
            # Calculate area under the curve
            if event.baseline is not None:
                event.area = np.trapz(event_signal - event.baseline, 
                                    dx=1/sampling_rate)
    
    def _fit_exponential(self, t: np.ndarray, signal: np.ndarray, 
                       event: SynapticEvent) -> Optional[np.ndarray]:
        """Fit single exponential decay."""
        def exp_func(x, A, tau, C):
            return A * np.exp(-x / tau) + C
            
        try:
            # Initial guess
            A0 = signal[0] - signal[-1]
            tau0 = 0.01  # 10 ms
            C0 = signal[-1]
            
            popt, _ = optimize.curve_fit(exp_func, t, signal, 
                                       p0=[A0, tau0, C0],
                                       bounds=([-np.inf, 0.0001, -np.inf],
                                             [np.inf, 1.0, np.inf]))
            return popt
        except:
            return None
    
    def _fit_biexponential(self, t: np.ndarray, signal: np.ndarray, 
                         event: SynapticEvent) -> Optional[np.ndarray]:
        """Fit double exponential decay."""
        def biexp_func(x, A1, tau1, A2, tau2, C):
            return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + C
            
        try:
            # Initial guess
            A0 = (signal[0] - signal[-1]) / 2
            tau1_0 = 0.005  # 5 ms fast
            tau2_0 = 0.05   # 50 ms slow
            C0 = signal[-1]
            
            popt, _ = optimize.curve_fit(biexp_func, t, signal, 
                                       p0=[A0, tau1_0, A0, tau2_0, C0],
                                       bounds=([-np.inf, 0.0001, -np.inf, 0.001, -np.inf],
                                             [np.inf, 0.1, np.inf, 1.0, np.inf]))
            return popt
        except:
            return None
    
    def _fit_alpha(self, t: np.ndarray, signal: np.ndarray, 
                 event: SynapticEvent) -> Optional[np.ndarray]:
        """Fit alpha function (rise and decay)."""
        def alpha_func(x, A, rise, decay, t0, C):
            shifted = x - t0
            return np.where(shifted > 0,
                          A * (1 - np.exp(-shifted / rise)) * np.exp(-shifted / decay) + C,
                          C)
            
        try:
            # Find peak time
            peak_idx = np.argmin(signal)
            t0 = t[peak_idx]
            
            # Initial guess
            A0 = signal[peak_idx] - signal[-1]
            rise0 = 0.001   # 1 ms
            decay0 = 0.01   # 10 ms
            C0 = signal[-1]
            
            popt, _ = optimize.curve_fit(alpha_func, t, signal, 
                                       p0=[A0, rise0, decay0, t0, C0])
            return popt
        except:
            return None
    
    def _calculate_event_statistics(self, events: List[SynapticEvent]) -> Dict[str, Any]:
        """Calculate statistics for detected events."""
        if len(events) == 0:
            return {
                'count': 0,
                'frequency': 0.0
            }
            
        amplitudes = [e.amplitude for e in events if e.amplitude is not None]
        times = [e.time for e in events]
        
        stats = {
            'count': len(events),
            'frequency': len(events) / (self.data.shape[0] / self.sampling_rate),
            'amplitude_mean': np.mean(amplitudes) if amplitudes else np.nan,
            'amplitude_std': np.std(amplitudes) if amplitudes else np.nan,
            'amplitude_min': np.min(amplitudes) if amplitudes else np.nan,
            'amplitude_max': np.max(amplitudes) if amplitudes else np.nan,
            'inter_event_interval_mean': np.mean(np.diff(times)) if len(times) > 1 else np.nan,
            'inter_event_interval_cv': (np.std(np.diff(times)) / np.mean(np.diff(times))) if len(times) > 1 else np.nan
        }
        
        # Add kinetic statistics if available
        decay_taus = [e.decay_tau for e in events if e.decay_tau is not None]
        if decay_taus:
            stats['decay_tau_mean'] = np.mean(decay_taus)
            stats['decay_tau_std'] = np.std(decay_taus)
            
        rise_times = [e.rise_time for e in events if e.rise_time is not None]
        if rise_times:
            stats['rise_time_mean'] = np.mean(rise_times)
            stats['rise_time_std'] = np.std(rise_times)
            
        return stats
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8), 
                 show_events: bool = True,
                 show_histogram: bool = True,
                 save_path: Optional[str] = None) -> None:
        """
        Visualize event detection results.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        show_events : bool
            Whether to mark detected events on the trace
        show_histogram : bool
            Whether to show amplitude histogram
        save_path : str, optional
            Path to save figure
        """
        if self.results is None:
            raise ValueError("No analysis results to visualize. Run analyze() first.")
            
        events = self.results['events']
        signal = self.results['preprocessed_signal']
        stats = self.results['statistics']
        
        # Create time vector
        time = np.arange(len(signal)) / self.sampling_rate
        
        # Setup figure
        if show_histogram:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                               gridspec_kw={'height_ratios': [2, 1, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//2))
            
        # Plot trace
        ax1.plot(time, signal, 'k-', linewidth=0.5, alpha=0.8)
        
        if show_events and len(events) > 0:
            event_times = [e.time for e in events]
            event_amplitudes = [signal[int(e.time * self.sampling_rate)] for e in events]
            ax1.scatter(event_times, event_amplitudes, c='red', s=50, 
                       zorder=5, label=f'{len(events)} events')
            ax1.legend()
            
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (pA)')
        ax1.set_title(f'Synaptic Events (Frequency: {stats["frequency"]:.2f} Hz)')
        
        if show_histogram and len(events) > 0:
            # Amplitude histogram
            amplitudes = [e.amplitude for e in events if e.amplitude is not None]
            if amplitudes:
                ax2.hist(amplitudes, bins=30, edgecolor='black', alpha=0.7)
                ax2.axvline(np.mean(amplitudes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(amplitudes):.2f} pA')
                ax2.set_xlabel('Amplitude (pA)')
                ax2.set_ylabel('Count')
                ax2.set_title('Amplitude Distribution')
                ax2.legend()
                
            # Inter-event interval histogram
            if len(events) > 1:
                intervals = np.diff([e.time for e in events])
                ax3.hist(intervals, bins=30, edgecolor='black', alpha=0.7)
                ax3.axvline(np.mean(intervals), color='red', linestyle='--',
                          label=f'Mean: {np.mean(intervals)*1000:.1f} ms')
                ax3.set_xlabel('Inter-event Interval (s)')
                ax3.set_ylabel('Count')
                ax3.set_title('Inter-event Interval Distribution')
                ax3.legend()
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, 
                       format=self.config.figure_format)
            
        plt.show()


class InputOutputAnalyzer(SynapticAnalyzer):
    """Analyzer for synaptic input-output relationships."""
    
    def analyze(self, input_data: np.ndarray, output_data: np.ndarray, 
               sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Analyze input-output relationships.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input values (e.g., stimulation intensity)
        output_data : np.ndarray
            Output values (e.g., response amplitude)
        sampling_rate : float
            Not used for I/O analysis, included for API consistency
            
        Returns
        -------
        dict
            Analysis results including fitted parameters
        """
        self.sampling_rate = sampling_rate
        
        # Normalize if requested
        if self.config.io_normalize:
            output_norm = (output_data - np.min(output_data)) / (np.max(output_data) - np.min(output_data))
        else:
            output_norm = output_data
            
        # Fit model
        if self.config.io_model == "sigmoid":
            params, params_ci = self._fit_sigmoid(input_data, output_norm)
            model_func = self._sigmoid
        elif self.config.io_model == "linear":
            params, params_ci = self._fit_linear(input_data, output_norm)
            model_func = lambda x, *p: p[0] * x + p[1]
        elif self.config.io_model == "hill":
            params, params_ci = self._fit_hill(input_data, output_norm)
            model_func = self._hill
        elif self.config.io_model == "boltzmann":
            params, params_ci = self._fit_boltzmann(input_data, output_norm)
            model_func = self._boltzmann
        else:
            raise ValueError(f"Unknown I/O model: {self.config.io_model}")
            
        # Calculate goodness of fit
        fitted_values = model_func(input_data, *params)
        r_squared = 1 - np.sum((output_norm - fitted_values)**2) / np.sum((output_norm - np.mean(output_norm))**2)
        
        self.results = {
            'input': input_data,
            'output': output_data,
            'output_normalized': output_norm,
            'model': self.config.io_model,
            'parameters': params,
            'parameters_ci': params_ci,
            'r_squared': r_squared,
            'model_function': model_func
        }
        
        return self.results
    
    def _sigmoid(self, x, a, b, c):
        """Sigmoid function."""
        return c / (1 + np.exp(-(x - a) / b))
    
    def _hill(self, x, vmax, km, n):
        """Hill equation."""
        return vmax * (x**n) / (km**n + x**n)
    
    def _boltzmann(self, x, vmax, v50, slope):
        """Boltzmann function."""
        return vmax / (1 + np.exp((v50 - x) / slope))
    
    def _fit_sigmoid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit sigmoid function with confidence intervals."""
        # Initial guess
        a0 = np.median(x)  # Midpoint
        b0 = (np.max(x) - np.min(x)) / 10  # Slope
        c0 = np.max(y)  # Maximum
        
        try:
            popt, pcov = optimize.curve_fit(self._sigmoid, x, y, 
                                          p0=[a0, b0, c0],
                                          maxfev=5000)
            
            # Calculate confidence intervals
            perr = np.sqrt(np.diag(pcov))
            ci = 1.96 * perr  # 95% confidence interval
            
            return popt, ci
        except Exception as e:
            logger.warning(f"Sigmoid fitting failed: {e}")
            return np.array([a0, b0, c0]), np.array([np.nan, np.nan, np.nan])
    
    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit linear function with confidence intervals."""
        # Use scipy.stats for linear regression with confidence intervals
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate confidence intervals
        n = len(x)
        t_val = stats.t.ppf(0.975, n-2)  # 95% confidence
        
        ci_slope = t_val * std_err
        
        # For intercept CI, we need more calculation
        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean)**2)
        mse = np.sum((y - (slope * x + intercept))**2) / (n - 2)
        ci_intercept = t_val * np.sqrt(mse * (1/n + x_mean**2/ss_x))
        
        return np.array([slope, intercept]), np.array([ci_slope, ci_intercept])
    
    def _fit_hill(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit Hill equation with confidence intervals."""
        # Initial guess
        vmax0 = np.max(y)
        km0 = np.median(x)
        n0 = 1.0
        
        try:
            popt, pcov = optimize.curve_fit(self._hill, x, y, 
                                          p0=[vmax0, km0, n0],
                                          bounds=([0, 0, 0.1], [np.inf, np.inf, 10]),
                                          maxfev=5000)
            
            perr = np.sqrt(np.diag(pcov))
            ci = 1.96 * perr
            
            return popt, ci
        except Exception as e:
            logger.warning(f"Hill fitting failed: {e}")
            return np.array([vmax0, km0, n0]), np.array([np.nan, np.nan, np.nan])
    
    def _fit_boltzmann(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit Boltzmann function with confidence intervals."""
        # Initial guess
        vmax0 = np.max(y)
        v50_0 = np.median(x)
        slope0 = (np.max(x) - np.min(x)) / 10
        
        try:
            popt, pcov = optimize.curve_fit(self._boltzmann, x, y, 
                                          p0=[vmax0, v50_0, slope0],
                                          maxfev=5000)
            
            perr = np.sqrt(np.diag(pcov))
            ci = 1.96 * perr
            
            return popt, ci
        except Exception as e:
            logger.warning(f"Boltzmann fitting failed: {e}")
            return np.array([vmax0, v50_0, slope0]), np.array([np.nan, np.nan, np.nan])
    
    def visualize(self, figsize: Tuple[int, int] = (10, 6),
                 show_confidence: bool = True,
                 save_path: Optional[str] = None) -> None:
        """
        Visualize input-output relationship.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        show_confidence : bool
            Whether to show confidence bands
        save_path : str, optional
            Path to save figure
        """
        if self.results is None:
            raise ValueError("No analysis results to visualize. Run analyze() first.")
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot data points
        ax.scatter(self.results['input'], self.results['output_normalized'], 
                  s=50, alpha=0.6, edgecolor='black', label='Data')
        
        # Plot fitted curve
        x_fit = np.linspace(np.min(self.results['input']), 
                          np.max(self.results['input']), 200)
        y_fit = self.results['model_function'](x_fit, *self.results['parameters'])
        
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
               label=f'{self.config.io_model.capitalize()} fit (R² = {self.results["r_squared"]:.3f})')
        
        # Add confidence bands if requested
        if show_confidence and not np.any(np.isnan(self.results['parameters_ci'])):
            # Bootstrap confidence bands
            n_bootstrap = self.config.bootstrap_iterations
            y_bootstrap = []
            
            for _ in range(n_bootstrap):
                # Resample data
                idx = np.random.choice(len(self.results['input']), 
                                     size=len(self.results['input']), 
                                     replace=True)
                x_boot = self.results['input'][idx]
                y_boot = self.results['output_normalized'][idx]
                
                # Refit
                try:
                    if self.config.io_model == "sigmoid":
                        params_boot, _ = self._fit_sigmoid(x_boot, y_boot)
                    elif self.config.io_model == "linear":
                        params_boot, _ = self._fit_linear(x_boot, y_boot)
                    elif self.config.io_model == "hill":
                        params_boot, _ = self._fit_hill(x_boot, y_boot)
                    elif self.config.io_model == "boltzmann":
                        params_boot, _ = self._fit_boltzmann(x_boot, y_boot)
                        
                    y_pred = self.results['model_function'](x_fit, *params_boot)
                    y_bootstrap.append(y_pred)
                except:
                    continue
                    
            if y_bootstrap:
                y_bootstrap = np.array(y_bootstrap)
                ci_low = np.percentile(y_bootstrap, (1 - self.config.confidence_level) * 100 / 2, axis=0)
                ci_high = np.percentile(y_bootstrap, 100 - (1 - self.config.confidence_level) * 100 / 2, axis=0)
                
                ax.fill_between(x_fit, ci_low, ci_high, alpha=0.2, color='red', 
                              label=f'{self.config.confidence_level*100:.0f}% CI')
        
        ax.set_xlabel('Input')
        ax.set_ylabel('Output (normalized)' if self.config.io_normalize else 'Output')
        ax.set_title(f'Synaptic Input-Output Relationship ({self.config.io_model.capitalize()})')
        ax.legend()
        
        # Add parameter text
        param_text = f"Parameters:\n"
        param_names = {
            'sigmoid': ['x₅₀', 'slope', 'max'],
            'linear': ['slope', 'intercept'],
            'hill': ['Vmax', 'Km', 'n'],
            'boltzmann': ['Vmax', 'V₅₀', 'slope']
        }
        
        for i, (param, ci) in enumerate(zip(self.results['parameters'], 
                                           self.results['parameters_ci'])):
            if self.config.io_model in param_names and i < len(param_names[self.config.io_model]):
                name = param_names[self.config.io_model][i]
            else:
                name = f'p{i}'
                
            if not np.isnan(ci):
                param_text += f"{name} = {param:.3f} ± {ci:.3f}\n"
            else:
                param_text += f"{name} = {param:.3f}\n"
                
        ax.text(0.05, 0.95, param_text.strip(), transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
                                                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi,
                       format=self.config.figure_format)
            
        plt.show()


class CombinedSynapticAnalyzer(SynapticAnalyzer):
    """Combined analyzer for comprehensive synaptic analysis."""
    
    def __init__(self, config: AnalysisConfig = None):
        super().__init__(config)
        self.event_analyzer = EventDetectionAnalyzer(config)
        self.io_analyzer = InputOutputAnalyzer(config)
        
    def analyze(self, data: np.ndarray, sampling_rate: float,
               input_data: Optional[np.ndarray] = None,
               output_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform combined synaptic analysis.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data for event detection
        sampling_rate : float
            Sampling rate in Hz
        input_data : np.ndarray, optional
            Input values for I/O analysis
        output_data : np.ndarray, optional
            Output values for I/O analysis
            
        Returns
        -------
        dict
            Combined analysis results
        """
        results = {}
        
        # Event detection analysis
        results['event_analysis'] = self.event_analyzer.analyze(data, sampling_rate)
        
        # Input-output analysis if data provided
        if input_data is not None and output_data is not None:
            results['io_analysis'] = self.io_analyzer.analyze(input_data, output_data, sampling_rate)
            
        self.results = results
        return results
    
    def visualize(self, figsize: Tuple[int, int] = (14, 10),
                 save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization."""
        if self.results is None:
            raise ValueError("No analysis results to visualize. Run analyze() first.")
            
        # Determine subplot layout
        has_events = 'event_analysis' in self.results
        has_io = 'io_analysis' in self.results
        
        if has_events and has_io:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
            ax1 = fig.add_subplot(gs[0, :])  # Event trace
            ax2 = fig.add_subplot(gs[1, 0])  # Amplitude histogram
            ax3 = fig.add_subplot(gs[1, 1])  # IEI histogram
            ax4 = fig.add_subplot(gs[2, :])  # I/O curve
        elif has_events:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
            gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = None
        else:
            fig, ax4 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.4))
            ax1 = ax2 = ax3 = None
            
        # Event analysis plots
        if has_events:
            events = self.results['event_analysis']['events']
            signal = self.results['event_analysis']['preprocessed_signal']
            stats = self.results['event_analysis']['statistics']
            time = np.arange(len(signal)) / self.event_analyzer.sampling_rate
            
            # Trace with events
            ax1.plot(time, signal, 'k-', linewidth=0.5, alpha=0.8)
            if len(events) > 0:
                event_times = [e.time for e in events]
                event_amplitudes = [signal[int(e.time * self.event_analyzer.sampling_rate)] for e in events]
                ax1.scatter(event_times, event_amplitudes, c='red', s=50, 
                           zorder=5, label=f'{len(events)} events')
                
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude (pA)')
            ax1.set_title(f'Synaptic Events (Frequency: {stats["frequency"]:.2f} Hz)')
            ax1.legend()
            
            # Amplitude histogram
            if len(events) > 0:
                amplitudes = [e.amplitude for e in events if e.amplitude is not None]
                if amplitudes:
                    ax2.hist(amplitudes, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                    ax2.axvline(np.mean(amplitudes), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(amplitudes):.2f} pA')
                    ax2.set_xlabel('Amplitude (pA)')
                    ax2.set_ylabel('Count')
                    ax2.set_title('Amplitude Distribution')
                    ax2.legend()
                    
                # IEI histogram
                if len(events) > 1:
                    intervals = np.diff([e.time for e in events]) * 1000  # Convert to ms
                    ax3.hist(intervals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
                    ax3.axvline(np.mean(intervals), color='red', linestyle='--',
                              label=f'Mean: {np.mean(intervals):.1f} ms')
                    ax3.set_xlabel('Inter-event Interval (ms)')
                    ax3.set_ylabel('Count')
                    ax3.set_title('IEI Distribution')
                    ax3.legend()
                    
        # I/O analysis plot
        if has_io and ax4 is not None:
            io_results = self.results['io_analysis']
            
            # Data points
            ax4.scatter(io_results['input'], io_results['output_normalized'], 
                       s=50, alpha=0.6, edgecolor='black', label='Data')
            
            # Fitted curve
            x_fit = np.linspace(np.min(io_results['input']), 
                              np.max(io_results['input']), 200)
            y_fit = io_results['model_function'](x_fit, *io_results['parameters'])
            
            ax4.plot(x_fit, y_fit, 'g-', linewidth=2, 
                    label=f'{io_results["model"].capitalize()} fit (R² = {io_results["r_squared"]:.3f})')
            
            ax4.set_xlabel('Input')
            ax4.set_ylabel('Output (normalized)' if self.config.io_normalize else 'Output')
            ax4.set_title('Input-Output Relationship')
            ax4.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi,
                       format=self.config.figure_format)
            
        plt.show()


def create_example_data():
    """Create example data for testing the module."""
    # Create synthetic synaptic current data
    sampling_rate = 20000  # Hz
    duration = 10  # seconds
    time = np.arange(0, duration, 1/sampling_rate)
    
    # Baseline current with noise
    baseline = -50  # pA
    noise = np.random.normal(0, 5, len(time))
    signal = baseline + noise
    
    # Add synaptic events
    n_events = 50
    event_times = np.sort(np.random.uniform(0.5, duration-0.5, n_events))
    
    for event_time in event_times:
        # Create event with exponential decay
        event_start = int(event_time * sampling_rate)
        event_duration = int(0.1 * sampling_rate)  # 100 ms window
        
        if event_start + event_duration < len(signal):
            event_time_vector = np.arange(event_duration) / sampling_rate
            amplitude = np.random.uniform(-50, -150)  # pA
            tau = np.random.uniform(0.005, 0.02)  # 5-20 ms
            event_shape = amplitude * np.exp(-event_time_vector / tau)
            
            signal[event_start:event_start+event_duration] += event_shape
            
    # Create I/O data
    input_intensities = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    output_responses = 100 / (1 + np.exp(-(input_intensities - 50) / 10)) + np.random.normal(0, 5, len(input_intensities))
    
    return signal, sampling_rate, input_intensities, output_responses


# Main execution example
if __name__ == "__main__":
    # Create example data
    signal, sampling_rate, input_vals, output_vals = create_example_data()
    
    # Configure analysis
    config = AnalysisConfig(
        filter_freq_high=2000,
        threshold_std=3.0,
        kinetic_model="exponential",
        io_model="sigmoid",
        plot_style="seaborn"
    )
    
    # Perform combined analysis
    analyzer = CombinedSynapticAnalyzer(config)
    results = analyzer.analyze(signal, sampling_rate, input_vals, output_vals)
    
    # Print results summary
    print("\n=== Synaptic Analysis Results ===")
    if 'event_analysis' in results:
        stats = results['event_analysis']['statistics']
        print(f"\nEvent Detection:")
        print(f"  Events detected: {stats['count']}")
        print(f"  Frequency: {stats['frequency']:.2f} Hz")
        print(f"  Mean amplitude: {stats['amplitude_mean']:.2f} ± {stats['amplitude_std']:.2f} pA")
        
    if 'io_analysis' in results:
        io = results['io_analysis']
        print(f"\nInput-Output Analysis:")
        print(f"  Model: {io['model']}")
        print(f"  R²: {io['r_squared']:.3f}")
        print(f"  Parameters: {io['parameters']}")
    
    # Visualize results
    analyzer.visualize()