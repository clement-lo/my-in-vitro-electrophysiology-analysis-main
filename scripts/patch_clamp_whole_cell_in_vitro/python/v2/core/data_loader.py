"""
Unified Data Loader for Electrophysiology Analysis
=================================================

Comprehensive data loading module supporting multiple file formats
with automatic detection and standardized output.

Supported formats:
- ABF (Axon Binary Format) via PyABF
- NWB (Neurodata Without Borders) via PyNWB
- CSV/TXT (Comma/Tab separated values)
- HDF5 (Hierarchical Data Format)
- Neo-compatible formats (Spike2, AxoGraph, etc.)

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
import pandas as pd

# Optional imports with availability checking
try:
    import pyabf
    HAS_PYABF = True
except ImportError:
    HAS_PYABF = False
    warnings.warn("PyABF not installed. ABF file support disabled.")

try:
    import neo
    HAS_NEO = True
except ImportError:
    HAS_NEO = False
    warnings.warn("Neo not installed. Some formats may not be supported.")

try:
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    HAS_PYNWB = False
    warnings.warn("PyNWB not installed. NWB file support disabled.")

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    warnings.warn("h5py not installed. HDF5 file support disabled.")

logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported file formats."""
    ABF = "abf"
    NWB = "nwb"
    CSV = "csv"
    HDF5 = "hdf5"
    NEO = "neo"
    UNKNOWN = "unknown"


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class DataLoader:
    """
    Unified data loader for multiple electrophysiology file formats.
    
    This class provides a consistent interface for loading data from
    various file formats commonly used in electrophysiology.
    """
    
    # Format detection mapping
    FORMAT_EXTENSIONS = {
        '.abf': FileFormat.ABF,
        '.nwb': FileFormat.NWB,
        '.csv': FileFormat.CSV,
        '.txt': FileFormat.CSV,
        '.tsv': FileFormat.CSV,
        '.h5': FileFormat.HDF5,
        '.hdf5': FileFormat.HDF5,
        '.hdf': FileFormat.HDF5,
        '.dat': FileFormat.NEO,
        '.smr': FileFormat.NEO,
        '.axgx': FileFormat.NEO,
        '.axgd': FileFormat.NEO,
        '.cfs': FileFormat.NEO
    }
    
    def __init__(self):
        """Initialize data loader."""
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check available dependencies and log status."""
        dependencies = {
            'PyABF': HAS_PYABF,
            'PyNWB': HAS_PYNWB,
            'Neo': HAS_NEO,
            'h5py': HAS_HDF5
        }
        
        available = [name for name, status in dependencies.items() if status]
        missing = [name for name, status in dependencies.items() if not status]
        
        if available:
            logger.debug(f"Available dependencies: {', '.join(available)}")
        if missing:
            logger.info(f"Missing optional dependencies: {', '.join(missing)}")
    
    @classmethod
    def detect_format(cls, file_path: Union[str, Path]) -> FileFormat:
        """
        Detect file format based on extension.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the file
            
        Returns
        -------
        FileFormat
            Detected file format
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        return cls.FORMAT_EXTENSIONS.get(ext, FileFormat.UNKNOWN)
    
    @classmethod
    def get_supported_formats(cls) -> Dict[str, bool]:
        """
        Get dictionary of supported formats and their availability.
        
        Returns
        -------
        dict
            Format names and availability status
        """
        return {
            'ABF': HAS_PYABF,
            'NWB': HAS_PYNWB,
            'CSV': True,  # Always available via pandas
            'HDF5': HAS_HDF5,
            'Neo formats': HAS_NEO
        }
    
    def load(self, file_path: Union[str, Path], 
             format: Optional[FileFormat] = None,
             channel: int = 0,
             **kwargs) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Load electrophysiology data from file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        format : FileFormat, optional
            File format. If None, will be auto-detected
        channel : int, default=0
            Channel index for multi-channel recordings
        **kwargs
            Additional format-specific parameters
            
        Returns
        -------
        data : np.ndarray
            Signal data (1D or 2D array)
        sampling_rate : float
            Sampling rate in Hz
        metadata : dict
            File metadata and additional information
            
        Raises
        ------
        DataLoaderError
            If file cannot be loaded
        """
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            raise DataLoaderError(f"File not found: {path}")
            
        # Detect format if not specified
        if format is None:
            format = self.detect_format(path)
            
        if format == FileFormat.UNKNOWN:
            raise DataLoaderError(f"Unknown file format: {path.suffix}")
            
        logger.info(f"Loading {format.value} file: {path}")
        
        # Load based on format
        try:
            if format == FileFormat.ABF:
                return self._load_abf(path, channel, **kwargs)
            elif format == FileFormat.NWB:
                return self._load_nwb(path, channel, **kwargs)
            elif format == FileFormat.CSV:
                return self._load_csv(path, **kwargs)
            elif format == FileFormat.HDF5:
                return self._load_hdf5(path, channel, **kwargs)
            elif format == FileFormat.NEO:
                return self._load_neo(path, channel, **kwargs)
            else:
                raise DataLoaderError(f"Unsupported format: {format}")
        except Exception as e:
            raise DataLoaderError(f"Failed to load {path}: {str(e)}")
    
    def _load_abf(self, path: Path, channel: int, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Load ABF file using PyABF."""
        if not HAS_PYABF:
            raise DataLoaderError("PyABF is required for ABF file support. Install with: pip install pyabf")
            
        abf = pyabf.ABF(str(path))
        
        # Check channel availability
        if channel >= abf.channelCount:
            raise DataLoaderError(f"Channel {channel} not available. File has {abf.channelCount} channels.")
            
        # Get sweep data
        sweep = kwargs.get('sweep', 0)
        if sweep >= abf.sweepCount:
            raise DataLoaderError(f"Sweep {sweep} not available. File has {abf.sweepCount} sweeps.")
            
        abf.setSweep(sweep, channel=channel)
        
        # Extract data
        data = abf.sweepY
        sampling_rate = float(abf.dataRate)
        
        # Build metadata
        metadata = {
            'format': 'ABF',
            'protocol': abf.protocol,
            'units': abf.sweepUnitsY,
            'channel_count': abf.channelCount,
            'sweep_count': abf.sweepCount,
            'current_channel': channel,
            'current_sweep': sweep,
            'creator': abf.creator if hasattr(abf, 'creator') else None,
            'creation_date': str(abf.abfDateTime) if hasattr(abf, 'abfDateTime') else None,
            'file_path': str(path)
        }
        
        return data, sampling_rate, metadata
    
    def _load_nwb(self, path: Path, channel: int, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Load NWB file using PyNWB."""
        if not HAS_PYNWB:
            raise DataLoaderError("PyNWB is required for NWB file support. Install with: pip install pynwb")
            
        with NWBHDF5IO(str(path), 'r') as io:
            nwbfile = io.read()
            
            # Try to find acquisition data
            acquisition_names = list(nwbfile.acquisition.keys())
            if not acquisition_names:
                raise DataLoaderError("No acquisition data found in NWB file")
                
            # Get specified acquisition or first available
            acq_name = kwargs.get('acquisition', acquisition_names[0])
            if acq_name not in nwbfile.acquisition:
                raise DataLoaderError(f"Acquisition '{acq_name}' not found. Available: {acquisition_names}")
                
            acquisition = nwbfile.acquisition[acq_name]
            
            # Extract data
            data = acquisition.data[:]
            
            # Handle multi-channel data
            if data.ndim > 1:
                if channel >= data.shape[1]:
                    raise DataLoaderError(f"Channel {channel} not available. Data has {data.shape[1]} channels.")
                data = data[:, channel]
                
            # Get sampling rate
            if hasattr(acquisition, 'rate'):
                sampling_rate = float(acquisition.rate)
            elif hasattr(acquisition, 'starting_time_rate'):
                sampling_rate = float(acquisition.starting_time_rate)
            else:
                raise DataLoaderError("Could not determine sampling rate from NWB file")
                
            # Build metadata
            metadata = {
                'format': 'NWB',
                'acquisition': acq_name,
                'available_acquisitions': acquisition_names,
                'description': acquisition.description if hasattr(acquisition, 'description') else None,
                'unit': acquisition.unit if hasattr(acquisition, 'unit') else None,
                'session_id': nwbfile.session_id if hasattr(nwbfile, 'session_id') else None,
                'experimenter': nwbfile.experimenter if hasattr(nwbfile, 'experimenter') else None,
                'file_path': str(path)
            }
            
        return data, sampling_rate, metadata
    
    def _load_csv(self, path: Path, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Load CSV/TXT file with automatic column detection."""
        # Determine delimiter
        delimiter = kwargs.get('delimiter', None)
        if delimiter is None:
            # Try to auto-detect delimiter
            with open(path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ' '
                    
        # Read data
        df = pd.read_csv(path, delimiter=delimiter)
        
        # Identify columns
        time_col = kwargs.get('time_column', None)
        data_col = kwargs.get('data_column', None)
        
        if time_col is None:
            # Look for time column
            time_candidates = [col for col in df.columns if 'time' in col.lower()]
            if time_candidates:
                time_col = time_candidates[0]
            else:
                # Assume first column is time
                time_col = df.columns[0]
                
        if data_col is None:
            # Look for data column
            data_candidates = [col for col in df.columns 
                             if any(term in col.lower() 
                                   for term in ['voltage', 'current', 'signal', 'data', 'amplitude'])]
            if data_candidates:
                data_col = data_candidates[0]
            else:
                # Assume second column is data
                if len(df.columns) >= 2:
                    data_col = df.columns[1]
                else:
                    raise DataLoaderError("CSV must have at least 2 columns")
                    
        # Extract arrays
        time = df[time_col].values
        data = df[data_col].values
        
        # Calculate sampling rate
        dt = np.mean(np.diff(time))
        sampling_rate = 1.0 / dt
        
        # Build metadata
        metadata = {
            'format': 'CSV',
            'delimiter': delimiter,
            'columns': list(df.columns),
            'time_column': time_col,
            'data_column': data_col,
            'shape': df.shape,
            'duration': time[-1] - time[0],
            'file_path': str(path)
        }
        
        return data, sampling_rate, metadata
    
    def _load_hdf5(self, path: Path, channel: int, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Load HDF5 file."""
        if not HAS_HDF5:
            raise DataLoaderError("h5py is required for HDF5 file support. Install with: pip install h5py")
            
        with h5py.File(path, 'r') as f:
            # Get data location
            data_path = kwargs.get('data_path', None)
            if data_path is None:
                # Look for common data paths
                candidates = ['data', 'signal', 'recording', 'traces', 'voltage', 'current']
                for candidate in candidates:
                    if candidate in f:
                        data_path = candidate
                        break
                        
                if data_path is None:
                    raise DataLoaderError(f"No data found. Available datasets: {list(f.keys())}")
                    
            # Load data
            data = f[data_path][:]
            
            # Handle multi-channel
            if data.ndim > 1:
                if channel >= data.shape[1]:
                    raise DataLoaderError(f"Channel {channel} not available. Data has {data.shape[1]} channels.")
                data = data[:, channel]
                
            # Get sampling rate
            rate_path = kwargs.get('rate_path', None)
            if rate_path:
                sampling_rate = float(f[rate_path][()])
            else:
                # Look in attributes
                rate_attrs = ['sampling_rate', 'fs', 'rate', 'sample_rate']
                sampling_rate = None
                
                for attr in rate_attrs:
                    if attr in f.attrs:
                        sampling_rate = float(f.attrs[attr])
                        break
                    elif data_path in f and attr in f[data_path].attrs:
                        sampling_rate = float(f[data_path].attrs[attr])
                        break
                        
                if sampling_rate is None:
                    sampling_rate = kwargs.get('sampling_rate', 20000.0)
                    logger.warning(f"Sampling rate not found in file, using default: {sampling_rate} Hz")
                    
            # Build metadata
            metadata = {
                'format': 'HDF5',
                'datasets': list(f.keys()),
                'data_path': data_path,
                'attributes': dict(f.attrs),
                'shape': data.shape,
                'file_path': str(path)
            }
            
        return data, sampling_rate, metadata
    
    def _load_neo(self, path: Path, channel: int, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Load Neo-compatible file."""
        if not HAS_NEO:
            raise DataLoaderError("Neo is required for this file format. Install with: pip install neo")
            
        # Determine IO class based on extension
        ext = path.suffix.lower()
        io_mapping = {
            '.smr': neo.io.Spike2IO,
            '.axgx': neo.io.AxographIO,
            '.axgd': neo.io.AxographIO,
            '.cfs': neo.io.CFSIO,
            '.dat': neo.io.RawBinarySignalIO
        }
        
        io_class = io_mapping.get(ext)
        if io_class is None:
            raise DataLoaderError(f"Unsupported Neo format: {ext}")
            
        # Special handling for RawBinarySignal
        if io_class == neo.io.RawBinarySignalIO:
            # Need additional parameters
            dtype = kwargs.get('dtype', 'float32')
            n_channels = kwargs.get('n_channels', 1)
            sampling_rate = kwargs.get('sampling_rate', 20000.0)
            
            reader = io_class(
                filename=str(path),
                dtype=dtype,
                n_channels=n_channels,
                sampling_rate=sampling_rate
            )
        else:
            reader = io_class(filename=str(path))
            
        # Read data
        block = reader.read_block()
        if not block.segments:
            raise DataLoaderError("No segments found in Neo file")
            
        segment = block.segments[0]
        if not segment.analogsignals:
            raise DataLoaderError("No analog signals found in Neo file")
            
        # Get specified channel
        if channel >= len(segment.analogsignals):
            raise DataLoaderError(f"Channel {channel} not available. File has {len(segment.analogsignals)} channels.")
            
        signal_obj = segment.analogsignals[channel]
        data = np.array(signal_obj.magnitude).flatten()
        sampling_rate = float(signal_obj.sampling_rate.magnitude)
        
        # Build metadata
        metadata = {
            'format': 'Neo',
            'neo_io': io_class.__name__,
            'units': str(signal_obj.units),
            'channel_count': len(segment.analogsignals),
            'duration': float(signal_obj.duration.magnitude) if hasattr(signal_obj, 'duration') else None,
            'file_path': str(path)
        }
        
        return data, sampling_rate, metadata


# Convenience function for backward compatibility
def load_data(file_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Load electrophysiology data from file.
    
    This is a convenience function that creates a DataLoader instance
    and loads the specified file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the data file
    **kwargs
        Additional parameters passed to DataLoader.load()
        
    Returns
    -------
    data : np.ndarray
        Signal data
    sampling_rate : float
        Sampling rate in Hz
    metadata : dict
        File metadata
    """
    loader = DataLoader()
    return loader.load(file_path, **kwargs)


if __name__ == "__main__":
    # Example usage and format support check
    print("Electrophysiology Data Loader - Format Support")
    print("=" * 50)
    
    loader = DataLoader()
    formats = loader.get_supported_formats()
    
    print("\nSupported formats:")
    for format_name, available in formats.items():
        status = "✓" if available else "✗"
        print(f"  {status} {format_name}")
        
    print("\nFile extensions:")
    for ext, format in DataLoader.FORMAT_EXTENSIONS.items():
        print(f"  {ext} → {format.value}")
        
    # Example loading
    print("\nExample usage:")
    print("  data, rate, metadata = load_data('recording.abf')")
    print("  data, rate, metadata = load_data('data.nwb', acquisition='CurrentClampSeries')")
    print("  data, rate, metadata = load_data('traces.csv', time_column='Time', data_column='Voltage')")