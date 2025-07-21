"""
Unified Data Loader for Electrophysiology Analysis
Supports multiple file formats with automatic detection
"""

import os
import numpy as np
import pandas as pd
import pyabf
import neo
import logging

logger = logging.getLogger(__name__)

class UnifiedDataLoader:
    """Unified data loader supporting multiple formats."""
    
    @staticmethod
    def detect_file_type(file_path):
        """Automatically detect file type based on extension and content."""
        ext = os.path.splitext(file_path)[1].lower()
        
        type_map = {
            '.abf': 'abf',
            '.csv': 'csv',
            '.txt': 'csv',
            '.dat': 'neo',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.smr': 'neo',
            '.axgx': 'neo',
            '.axgd': 'neo'
        }
        
        return type_map.get(ext, 'unknown')
    
    @staticmethod
    def load_file(file_path, file_type=None):
        """
        Load electrophysiology data from various formats.
        
        Returns:
            tuple: (signal, sampling_rate, time, metadata)
        """
        if file_type is None:
            file_type = UnifiedDataLoader.detect_file_type(file_path)
            
        logger.info(f"Loading {file_type} file: {file_path}")
        
        loaders = {
            'abf': UnifiedDataLoader._load_abf,
            'csv': UnifiedDataLoader._load_csv,
            'neo': UnifiedDataLoader._load_neo,
            'hdf5': UnifiedDataLoader._load_hdf5
        }
        
        loader = loaders.get(file_type)
        if loader is None:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        return loader(file_path)
    
    @staticmethod
    def _load_abf(file_path):
        """Load ABF file using PyABF."""
        abf = pyabf.ABF(file_path)
        
        # Get first channel by default
        abf.setSweep(0, channel=0)
        signal = abf.sweepY
        sampling_rate = abf.dataRate
        time = abf.sweepX
        
        metadata = {
            'protocol': abf.protocol,
            'units': abf.sweepUnitsY,
            'channel_count': abf.channelCount,
            'sweep_count': abf.sweepCount,
            'file_path': file_path
        }
        
        return signal, sampling_rate, time, metadata
    
    @staticmethod
    def _load_csv(file_path):
        """Load CSV file with automatic column detection."""
        # Try to read with common delimiters
        for sep in [',', '\t', ' ', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) >= 2:
                    break
            except:
                continue
                
        # Detect time and voltage columns
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        voltage_cols = [col for col in df.columns if any(v in col.lower() for v in ['volt', 'potential', 'signal'])]
        
        if not time_cols or not voltage_cols:
            # Use first two columns as time and voltage
            time_col = df.columns[0]
            voltage_col = df.columns[1]
        else:
            time_col = time_cols[0]
            voltage_col = voltage_cols[0]
            
        time = df[time_col].values
        signal = df[voltage_col].values
        
        # Calculate sampling rate
        dt = np.mean(np.diff(time))
        sampling_rate = 1.0 / dt
        
        metadata = {
            'columns': list(df.columns),
            'shape': df.shape,
            'file_path': file_path
        }
        
        return signal, sampling_rate, time, metadata
    
    @staticmethod
    def _load_neo(file_path):
        """Load Neo-compatible file."""
        # This is a placeholder - implement based on specific Neo format
        raise NotImplementedError("Neo loading to be implemented based on specific format")
    
    @staticmethod
    def _load_hdf5(file_path):
        """Load HDF5 file."""
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # This is a basic implementation - adjust based on your HDF5 structure
            signal = f['data/signal'][:]
            sampling_rate = f['data'].attrs.get('sampling_rate', 20000)
            time = np.arange(len(signal)) / sampling_rate
            
            metadata = {
                'datasets': list(f.keys()),
                'attributes': dict(f.attrs),
                'file_path': file_path
            }
            
        return signal, sampling_rate, time, metadata


def load_data(file_path, file_type=None, **kwargs):
    """
    Convenience function for loading electrophysiology data.
    
    Args:
        file_path: Path to data file
        file_type: Optional file type specification
        **kwargs: Additional arguments passed to specific loaders
        
    Returns:
        tuple: (signal, sampling_rate, time, metadata)
    """
    loader = UnifiedDataLoader()
    return loader.load_file(file_path, file_type)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            signal, fs, time, metadata = load_data(file_path)
            print(f"Loaded data: {len(signal)} samples at {fs} Hz")
            print(f"Duration: {len(signal)/fs:.2f} seconds")
            print(f"Metadata: {metadata}")
        except Exception as e:
            print(f"Error loading file: {e}")
