"""
Unified Data Pipeline for Electrophysiology Analysis
Handles multiple formats with automatic detection and standardized output
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UnifiedDataPipeline:
    """Unified data loading and preprocessing pipeline."""
    
    # Supported formats and their extensions
    FORMAT_EXTENSIONS = {
        'nwb': ['.nwb'],
        'abf': ['.abf'],
        'csv': ['.csv', '.txt', '.tsv'],
        'mat': ['.mat'],
        'hdf5': ['.h5', '.hdf5', '.hdf'],
        'neo': ['.smr', '.axgx', '.axgd'],
        'binary': ['.dat', '.bin']
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metadata_cache = {}
        
    def load_data(self, file_path: Union[str, Path], 
                  format: Optional[str] = None,
                  **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        """
        Load electrophysiology data from various formats.
        
        Returns:
            tuple: (signal, sampling_rate, time, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Auto-detect format if not specified
        if format is None or format == 'auto':
            format = self._detect_format(file_path)
            
        logger.info(f"Loading {format} file: {file_path}")
        
        # Load based on format
        loaders = {
            'nwb': self._load_nwb,
            'abf': self._load_abf,
            'csv': self._load_csv,
            'mat': self._load_mat,
            'hdf5': self._load_hdf5,
            'neo': self._load_neo,
            'binary': self._load_binary
        }
        
        if format not in loaders:
            raise ValueError(f"Unsupported format: {format}")
            
        # Load data
        signal, sampling_rate, time, metadata = loaders[format](file_path, **kwargs)
        
        # Store metadata
        self.metadata_cache[str(file_path)] = metadata
        
        # Apply preprocessing if configured
        if self.config.get('preprocessing', {}).get('enabled', False):
            signal = self._preprocess_signal(signal, sampling_rate)
            
        return signal, sampling_rate, time, metadata
    
    def _detect_format(self, file_path: Path) -> str:
        """Automatically detect file format based on extension and content."""
        ext = file_path.suffix.lower()
        
        # Check extension first
        for format, extensions in self.FORMAT_EXTENSIONS.items():
            if ext in extensions:
                return format
                
        # Try to detect by content
        try:
            # Check for HDF5
            import h5py
            with h5py.File(file_path, 'r') as f:
                if 'nwb' in str(f.attrs.get('namespace', '')).lower():
                    return 'nwb'
                return 'hdf5'
        except:
            pass
            
        # Default to binary
        return 'binary'
    
    def _load_nwb(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load NWB file."""
        try:
            from pynwb import NWBHDF5IO
        except ImportError:
            raise ImportError("PyNWB not installed. Install with: pip install pynwb")
            
        with NWBHDF5IO(str(file_path), 'r') as io:
            nwbfile = io.read()
            
            # Extract first acquisition (customize as needed)
            acquisition_names = list(nwbfile.acquisition.keys())
            if not acquisition_names:
                raise ValueError("No acquisition data found in NWB file")
                
            # Get series based on type
            series_name = kwargs.get('series_name', acquisition_names[0])
            series = nwbfile.acquisition[series_name]
            
            # Extract data
            signal = series.data[:]
            
            # Get sampling rate
            if hasattr(series, 'rate'):
                sampling_rate = float(series.rate)
            elif hasattr(series, 'starting_time_rate'):
                sampling_rate = float(series.starting_time_rate)
            else:
                sampling_rate = 1.0 / (series.timestamps[1] - series.timestamps[0])
                
            # Generate time vector
            if hasattr(series, 'timestamps') and series.timestamps is not None:
                time = series.timestamps[:]
            else:
                time = np.arange(len(signal)) / sampling_rate
                
            # Extract metadata
            metadata = {
                'format': 'nwb',
                'session_id': nwbfile.identifier,
                'session_description': nwbfile.session_description,
                'series_name': series_name,
                'series_description': getattr(series, 'description', ''),
                'units': getattr(series, 'unit', 'unknown'),
                'electrode': getattr(series, 'electrode', None),
                'file_path': str(file_path)
            }
            
        return signal, sampling_rate, time, metadata
    
    def _load_abf(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load ABF file."""
        try:
            import pyabf
        except ImportError:
            raise ImportError("PyABF not installed. Install with: pip install pyabf")
            
        abf = pyabf.ABF(str(file_path))
        
        # Get channel and sweep
        channel = kwargs.get('channel', 0)
        sweep = kwargs.get('sweep', 0)
        
        abf.setSweep(sweep, channel=channel)
        
        signal = abf.sweepY
        sampling_rate = float(abf.dataRate)
        time = abf.sweepX
        
        metadata = {
            'format': 'abf',
            'protocol': abf.protocol,
            'units': abf.sweepUnitsY,
            'channel_count': abf.channelCount,
            'sweep_count': abf.sweepCount,
            'sweep_number': sweep,
            'channel_number': channel,
            'file_path': str(file_path)
        }
        
        return signal, sampling_rate, time, metadata
    
    def _load_csv(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load CSV file with automatic column detection."""
        # Try different delimiters
        for sep in [',', '\t', ' ', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) >= 2:
                    break
            except:
                continue
        else:
            raise ValueError("Could not parse CSV file")
            
        # Detect columns
        time_col = kwargs.get('time_column')
        signal_col = kwargs.get('signal_column')
        
        if time_col is None:
            # Auto-detect time column
            time_candidates = [col for col in df.columns if 'time' in col.lower()]
            time_col = time_candidates[0] if time_candidates else df.columns[0]
            
        if signal_col is None:
            # Auto-detect signal column
            signal_candidates = [col for col in df.columns 
                               if any(term in col.lower() 
                                     for term in ['voltage', 'current', 'signal', 'potential'])]
            signal_col = signal_candidates[0] if signal_candidates else df.columns[1]
            
        # Extract data
        time = df[time_col].values
        signal = df[signal_col].values
        
        # Calculate sampling rate
        dt = np.mean(np.diff(time))
        sampling_rate = 1.0 / dt
        
        metadata = {
            'format': 'csv',
            'columns': list(df.columns),
            'time_column': time_col,
            'signal_column': signal_col,
            'shape': df.shape,
            'file_path': str(file_path)
        }
        
        return signal, sampling_rate, time, metadata
    
    def _load_mat(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load MATLAB file."""
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("SciPy not installed")
            
        mat_data = loadmat(str(file_path))
        
        # Get variable names (excluding private)
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]
        
        # Get signal variable
        signal_var = kwargs.get('signal_variable')
        if signal_var is None:
            # Try common names
            for name in ['data', 'signal', 'voltage', 'current', 'trace']:
                if name in var_names:
                    signal_var = name
                    break
            else:
                signal_var = var_names[0]
                
        signal = mat_data[signal_var].flatten()
        
        # Get sampling rate
        fs_var = kwargs.get('fs_variable', 'fs')
        if fs_var in mat_data:
            sampling_rate = float(mat_data[fs_var])
        else:
            sampling_rate = kwargs.get('sampling_rate', 20000.0)
            
        # Generate time vector
        time = np.arange(len(signal)) / sampling_rate
        
        metadata = {
            'format': 'mat',
            'variables': var_names,
            'signal_variable': signal_var,
            'matlab_metadata': {k: v for k, v in mat_data.items() 
                              if k.startswith('__')},
            'file_path': str(file_path)
        }
        
        return signal, sampling_rate, time, metadata
    
    def _load_hdf5(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load generic HDF5 file."""
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Navigate to data
            data_path = kwargs.get('data_path', '/data/signal')
            
            if data_path not in f:
                # Try to find signal data
                for key in f.keys():
                    if 'signal' in key.lower() or 'data' in key.lower():
                        data_path = f"/{key}"
                        break
                else:
                    raise ValueError("Could not find signal data in HDF5 file")
                    
            signal = f[data_path][:]
            
            # Get sampling rate from attributes
            if 'sampling_rate' in f[data_path].attrs:
                sampling_rate = float(f[data_path].attrs['sampling_rate'])
            elif 'fs' in f.attrs:
                sampling_rate = float(f.attrs['fs'])
            else:
                sampling_rate = kwargs.get('sampling_rate', 20000.0)
                
            # Time vector
            time_path = kwargs.get('time_path')
            if time_path and time_path in f:
                time = f[time_path][:]
            else:
                time = np.arange(len(signal)) / sampling_rate
                
            metadata = {
                'format': 'hdf5',
                'datasets': list(f.keys()),
                'attributes': dict(f.attrs),
                'data_path': data_path,
                'file_path': str(file_path)
            }
            
        return signal, sampling_rate, time, metadata
    
    def _load_neo(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load Neo-compatible file."""
        try:
            import neo
        except ImportError:
            raise ImportError("Neo not installed. Install with: pip install neo")
            
        # Determine IO class based on extension
        ext = file_path.suffix.lower()
        io_map = {
            '.smr': neo.Spike2IO,
            '.axgx': neo.AxographIO,
            '.axgd': neo.AxographIO
        }
        
        if ext not in io_map:
            raise ValueError(f"Unsupported Neo format: {ext}")
            
        # Load file
        reader = io_map[ext](str(file_path))
        block = reader.read_block()
        
        # Get first segment and analog signal
        segment = block.segments[0]
        analog_signal = segment.analogsignals[0]
        
        signal = analog_signal.magnitude.flatten()
        sampling_rate = float(analog_signal.sampling_rate)
        time = analog_signal.times.magnitude
        
        metadata = {
            'format': 'neo',
            'neo_type': ext,
            'units': str(analog_signal.units),
            'channel_count': len(segment.analogsignals),
            'segment_count': len(block.segments),
            'file_path': str(file_path)
        }
        
        return signal, sampling_rate, time, metadata
    
    def _load_binary(self, file_path: Path, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """Load raw binary file."""
        # Get parameters
        dtype = kwargs.get('dtype', 'float32')
        sampling_rate = kwargs.get('sampling_rate', 20000.0)
        offset = kwargs.get('offset', 0)
        
        # Read file
        signal = np.fromfile(file_path, dtype=dtype, offset=offset)
        
        # Reshape if needed
        n_channels = kwargs.get('n_channels', 1)
        if n_channels > 1:
            signal = signal.reshape(-1, n_channels)
            channel = kwargs.get('channel', 0)
            signal = signal[:, channel]
            
        time = np.arange(len(signal)) / sampling_rate
        
        metadata = {
            'format': 'binary',
            'dtype': dtype,
            'sampling_rate': sampling_rate,
            'n_channels': n_channels,
            'file_path': str(file_path)
        }
        
        return signal, sampling_rate, time, metadata
    
    def _preprocess_signal(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply preprocessing based on configuration."""
        # Detrending
        if self.config.get('preprocessing', {}).get('detrend', True):
            from scipy.signal import detrend
            signal = detrend(signal)
            
        # Filtering
        if self.config.get('preprocessing', {}).get('filter', True):
            from scipy.signal import butter, filtfilt
            
            low_cut = self.config.get('preprocessing', {}).get('low_cutoff', 1.0)
            high_cut = self.config.get('preprocessing', {}).get('high_cutoff', 5000.0)
            order = self.config.get('preprocessing', {}).get('filter_order', 4)
            
            nyquist = sampling_rate / 2
            low = low_cut / nyquist
            high = high_cut / nyquist
            
            if 0 < low < 1 and 0 < high < 1:
                b, a = butter(order, [low, high], btype='band')
                signal = filtfilt(b, a, signal)
                
        return signal
    
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a file."""
        return self.metadata_cache.get(str(file_path))
    
    def clear_cache(self) -> None:
        """Clear metadata cache."""
        self.metadata_cache.clear()


# Convenience function
def load_data(file_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
    """Load electrophysiology data using the unified pipeline."""
    pipeline = UnifiedDataPipeline()
    return pipeline.load_data(file_path, **kwargs)
