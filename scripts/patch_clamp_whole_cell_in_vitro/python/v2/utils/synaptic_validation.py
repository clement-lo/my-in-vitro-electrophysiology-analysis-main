"""
Synaptic Analysis Validation Module
===================================

Comprehensive validation utilities for synaptic analysis data and parameters.
Ensures data quality and parameter validity before analysis.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SynapticDataValidator:
    """Validator for synaptic electrophysiology data."""
    
    @staticmethod
    def validate_signal(signal: np.ndarray, 
                       sampling_rate: float,
                       min_duration: float = 0.1,
                       max_duration: float = 3600.0,
                       expected_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Validate time series signal data.
        
        Parameters
        ----------
        signal : np.ndarray
            Signal data to validate
        sampling_rate : float
            Sampling rate in Hz
        min_duration : float
            Minimum acceptable duration in seconds
        max_duration : float
            Maximum acceptable duration in seconds
        expected_range : tuple, optional
            Expected (min, max) range for signal values
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        # Type checks
        if not isinstance(signal, np.ndarray):
            raise ValidationError(f"Signal must be numpy array, got {type(signal)}")
            
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValidationError(f"Sampling rate must be positive number, got {sampling_rate}")
            
        # Dimension checks
        if signal.ndim > 2:
            raise ValidationError(f"Signal must be 1D or 2D array, got {signal.ndim}D")
            
        if signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
            logger.warning("Signal appears to be transposed (more rows than columns)")
            
        # Size checks
        if signal.size == 0:
            raise ValidationError("Signal is empty")
            
        duration = signal.shape[-1] / sampling_rate
        if duration < min_duration:
            raise ValidationError(f"Signal duration {duration:.2f}s is less than minimum {min_duration}s")
            
        if duration > max_duration:
            raise ValidationError(f"Signal duration {duration:.2f}s exceeds maximum {max_duration}s")
            
        # Data quality checks
        if np.any(np.isnan(signal)):
            raise ValidationError("Signal contains NaN values")
            
        if np.any(np.isinf(signal)):
            raise ValidationError("Signal contains infinite values")
            
        # Range checks
        if expected_range is not None:
            min_val, max_val = expected_range
            if np.min(signal) < min_val or np.max(signal) > max_val:
                raise ValidationError(
                    f"Signal values [{np.min(signal):.2f}, {np.max(signal):.2f}] "
                    f"outside expected range [{min_val}, {max_val}]"
                )
                
        # Check for constant signal
        if np.std(signal) < 1e-10:
            logger.warning("Signal appears to be constant (no variation)")
            
        # Check for clipping
        unique_vals = np.unique(signal)
        if len(unique_vals) < 10:
            logger.warning(f"Signal has only {len(unique_vals)} unique values, may be clipped or digitized")
            
    @staticmethod
    def validate_io_data(input_data: np.ndarray, 
                        output_data: np.ndarray,
                        min_points: int = 3) -> None:
        """
        Validate input-output relationship data.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input values
        output_data : np.ndarray
            Output values
        min_points : int
            Minimum number of data points required
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        # Type checks
        if not isinstance(input_data, np.ndarray):
            raise ValidationError(f"Input data must be numpy array, got {type(input_data)}")
            
        if not isinstance(output_data, np.ndarray):
            raise ValidationError(f"Output data must be numpy array, got {type(output_data)}")
            
        # Shape checks
        if input_data.shape != output_data.shape:
            raise ValidationError(
                f"Input and output shapes must match: {input_data.shape} != {output_data.shape}"
            )
            
        if input_data.ndim > 1:
            raise ValidationError(f"Input/output data must be 1D, got {input_data.ndim}D")
            
        # Size checks
        if len(input_data) < min_points:
            raise ValidationError(
                f"Need at least {min_points} data points, got {len(input_data)}"
            )
            
        # Data quality checks
        if np.any(np.isnan(input_data)) or np.any(np.isnan(output_data)):
            raise ValidationError("Input/output data contains NaN values")
            
        if np.any(np.isinf(input_data)) or np.any(np.isinf(output_data)):
            raise ValidationError("Input/output data contains infinite values")
            
        # Check for duplicate inputs
        unique_inputs = np.unique(input_data)
        if len(unique_inputs) < len(input_data):
            logger.warning(
                f"Input data contains duplicates: {len(input_data)} points, "
                f"{len(unique_inputs)} unique values"
            )
            
        # Check for negative inputs (common issue)
        if np.any(input_data < 0):
            logger.info("Input data contains negative values")
            
        # Check data ordering
        if not np.all(np.diff(input_data) >= 0):
            logger.warning("Input data is not monotonically increasing")
            
    @staticmethod
    def validate_events(events: List[Any],
                       signal_duration: float,
                       min_amplitude: Optional[float] = None,
                       max_amplitude: Optional[float] = None) -> None:
        """
        Validate detected synaptic events.
        
        Parameters
        ----------
        events : list
            List of detected events
        signal_duration : float
            Duration of the signal in seconds
        min_amplitude : float, optional
            Minimum acceptable amplitude
        max_amplitude : float, optional
            Maximum acceptable amplitude
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        if not isinstance(events, list):
            raise ValidationError(f"Events must be a list, got {type(events)}")
            
        for i, event in enumerate(events):
            # Check event attributes
            if not hasattr(event, 'time'):
                raise ValidationError(f"Event {i} missing 'time' attribute")
                
            if not hasattr(event, 'amplitude'):
                raise ValidationError(f"Event {i} missing 'amplitude' attribute")
                
            # Check time validity
            if event.time < 0 or event.time > signal_duration:
                raise ValidationError(
                    f"Event {i} time {event.time:.3f}s outside signal duration [0, {signal_duration:.3f}]"
                )
                
            # Check amplitude validity
            if event.amplitude is not None:
                if np.isnan(event.amplitude) or np.isinf(event.amplitude):
                    raise ValidationError(f"Event {i} has invalid amplitude: {event.amplitude}")
                    
                if min_amplitude is not None and event.amplitude < min_amplitude:
                    logger.warning(f"Event {i} amplitude {event.amplitude} below minimum {min_amplitude}")
                    
                if max_amplitude is not None and event.amplitude > max_amplitude:
                    logger.warning(f"Event {i} amplitude {event.amplitude} above maximum {max_amplitude}")


class ParameterValidator:
    """Validator for analysis parameters."""
    
    @staticmethod
    def validate_filter_params(freq_low: float, 
                             freq_high: float,
                             sampling_rate: float,
                             filter_type: str) -> None:
        """
        Validate filter parameters.
        
        Parameters
        ----------
        freq_low : float
            Low frequency cutoff in Hz
        freq_high : float
            High frequency cutoff in Hz
        sampling_rate : float
            Sampling rate in Hz
        filter_type : str
            Type of filter
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        nyquist = sampling_rate / 2
        
        valid_types = ['bandpass', 'lowpass', 'highpass', 'none']
        if filter_type not in valid_types:
            raise ValidationError(f"Filter type must be one of {valid_types}, got '{filter_type}'")
            
        if filter_type == 'none':
            return
            
        if filter_type in ['bandpass', 'highpass']:
            if freq_low <= 0:
                raise ValidationError(f"Low frequency must be positive, got {freq_low}")
            if freq_low >= nyquist:
                raise ValidationError(
                    f"Low frequency {freq_low} Hz must be less than Nyquist frequency {nyquist} Hz"
                )
                
        if filter_type in ['bandpass', 'lowpass']:
            if freq_high <= 0:
                raise ValidationError(f"High frequency must be positive, got {freq_high}")
            if freq_high >= nyquist:
                raise ValidationError(
                    f"High frequency {freq_high} Hz must be less than Nyquist frequency {nyquist} Hz"
                )
                
        if filter_type == 'bandpass':
            if freq_low >= freq_high:
                raise ValidationError(
                    f"Low frequency {freq_low} Hz must be less than high frequency {freq_high} Hz"
                )
                
    @staticmethod
    def validate_detection_params(threshold_std: float,
                                min_event_interval: float,
                                baseline_window: float,
                                detection_method: str) -> None:
        """
        Validate event detection parameters.
        
        Parameters
        ----------
        threshold_std : float
            Threshold in standard deviations
        min_event_interval : float
            Minimum interval between events in seconds
        baseline_window : float
            Window for baseline calculation in seconds
        detection_method : str
            Detection method name
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        valid_methods = ['threshold', 'template', 'deconvolution']
        if detection_method not in valid_methods:
            raise ValidationError(
                f"Detection method must be one of {valid_methods}, got '{detection_method}'"
            )
            
        if threshold_std <= 0:
            raise ValidationError(f"Threshold must be positive, got {threshold_std}")
            
        if threshold_std > 10:
            logger.warning(f"Very high threshold {threshold_std} STD may miss many events")
            
        if min_event_interval <= 0:
            raise ValidationError(f"Minimum event interval must be positive, got {min_event_interval}")
            
        if min_event_interval > 1:
            logger.warning(f"Large minimum interval {min_event_interval}s may miss closely spaced events")
            
        if baseline_window <= 0:
            raise ValidationError(f"Baseline window must be positive, got {baseline_window}")
            
        if baseline_window < 0.01:
            logger.warning(f"Very short baseline window {baseline_window}s may be unreliable")
            
    @staticmethod
    def validate_kinetic_params(kinetic_model: str,
                              fit_window: float) -> None:
        """
        Validate kinetic analysis parameters.
        
        Parameters
        ----------
        kinetic_model : str
            Kinetic model name
        fit_window : float
            Window for fitting in seconds
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        valid_models = ['exponential', 'biexponential', 'alpha']
        if kinetic_model not in valid_models:
            raise ValidationError(
                f"Kinetic model must be one of {valid_models}, got '{kinetic_model}'"
            )
            
        if fit_window <= 0:
            raise ValidationError(f"Fit window must be positive, got {fit_window}")
            
        if fit_window < 0.005:
            logger.warning(f"Very short fit window {fit_window}s may not capture full kinetics")
            
        if fit_window > 0.5:
            logger.warning(f"Very long fit window {fit_window}s may include multiple events")
            
    @staticmethod
    def validate_io_params(io_model: str,
                         io_normalize: bool) -> None:
        """
        Validate input-output analysis parameters.
        
        Parameters
        ----------
        io_model : str
            I/O model name
        io_normalize : bool
            Whether to normalize output
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        valid_models = ['sigmoid', 'linear', 'hill', 'boltzmann']
        if io_model not in valid_models:
            raise ValidationError(
                f"I/O model must be one of {valid_models}, got '{io_model}'"
            )
            
        if not isinstance(io_normalize, bool):
            raise ValidationError(f"io_normalize must be boolean, got {type(io_normalize)}")


class FileValidator:
    """Validator for input files."""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path],
                         must_exist: bool = True,
                         expected_extensions: Optional[List[str]] = None) -> Path:
        """
        Validate file path.
        
        Parameters
        ----------
        file_path : str or Path
            File path to validate
        must_exist : bool
            Whether file must exist
        expected_extensions : list, optional
            List of acceptable file extensions
            
        Returns
        -------
        Path
            Validated Path object
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(f"File not found: {path}")
            
        if must_exist and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
            
        if expected_extensions:
            if path.suffix.lower() not in expected_extensions:
                raise ValidationError(
                    f"File extension '{path.suffix}' not in allowed extensions: {expected_extensions}"
                )
                
        return path
    
    @staticmethod
    def validate_data_format(file_path: Union[str, Path]) -> str:
        """
        Validate and identify data file format.
        
        Parameters
        ----------
        file_path : str or Path
            File path to check
            
        Returns
        -------
        str
            Identified file format
            
        Raises
        ------
        ValidationError
            If format is not supported
        """
        path = Path(file_path)
        
        supported_formats = {
            '.abf': 'ABF (Axon Binary Format)',
            '.nwb': 'NWB (Neurodata Without Borders)',
            '.csv': 'CSV (Comma Separated Values)',
            '.h5': 'HDF5',
            '.hdf5': 'HDF5',
            '.dat': 'Binary/Neo compatible',
            '.smr': 'Spike2',
            '.axgx': 'AxoGraph X',
            '.axgd': 'AxoGraph'
        }
        
        ext = path.suffix.lower()
        if ext not in supported_formats:
            raise ValidationError(
                f"Unsupported file format '{ext}'. Supported formats: {list(supported_formats.keys())}"
            )
            
        return supported_formats[ext]


def validate_analysis_config(config: Any) -> None:
    """
    Validate entire analysis configuration.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration object to validate
        
    Raises
    ------
    ValidationError
        If any validation fails
    """
    # Filter parameters
    ParameterValidator.validate_filter_params(
        config.filter_freq_low,
        config.filter_freq_high,
        1000,  # Dummy sampling rate for validation
        config.filter_type
    )
    
    # Detection parameters
    ParameterValidator.validate_detection_params(
        config.threshold_std,
        config.min_event_interval,
        config.baseline_window,
        config.detection_method
    )
    
    # Kinetic parameters
    ParameterValidator.validate_kinetic_params(
        config.kinetic_model,
        config.fit_window
    )
    
    # I/O parameters
    ParameterValidator.validate_io_params(
        config.io_model,
        config.io_normalize
    )
    
    # Statistical parameters
    if config.bootstrap_iterations < 100:
        logger.warning(f"Low bootstrap iterations ({config.bootstrap_iterations}) may give unreliable CIs")
        
    if not 0 < config.confidence_level < 1:
        raise ValidationError(f"Confidence level must be between 0 and 1, got {config.confidence_level}")
        
    # Visualization parameters
    if config.figure_dpi < 72:
        logger.warning(f"Low DPI ({config.figure_dpi}) may result in poor quality figures")
        
    valid_formats = ['png', 'pdf', 'svg', 'jpg']
    if config.figure_format not in valid_formats:
        raise ValidationError(
            f"Figure format must be one of {valid_formats}, got '{config.figure_format}'"
        )


def create_validation_report(signal: np.ndarray,
                           sampling_rate: float,
                           events: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Create a comprehensive validation report for the data.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal data
    sampling_rate : float
        Sampling rate in Hz
    events : list, optional
        Detected events
        
    Returns
    -------
    dict
        Validation report with statistics and warnings
    """
    report = {
        'signal_stats': {
            'duration': signal.shape[-1] / sampling_rate,
            'sampling_rate': sampling_rate,
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'has_nan': bool(np.any(np.isnan(signal))),
            'has_inf': bool(np.any(np.isinf(signal))),
            'unique_values': len(np.unique(signal))
        },
        'quality_checks': {
            'is_constant': np.std(signal) < 1e-10,
            'is_clipped': len(np.unique(signal)) < 100,
            'has_outliers': bool(np.any(np.abs(stats.zscore(signal)) > 5))
        },
        'warnings': []
    }
    
    # Add warnings
    if report['quality_checks']['is_constant']:
        report['warnings'].append("Signal appears to be constant")
        
    if report['quality_checks']['is_clipped']:
        report['warnings'].append("Signal may be clipped or heavily digitized")
        
    if report['quality_checks']['has_outliers']:
        report['warnings'].append("Signal contains extreme outliers")
        
    # Event statistics if provided
    if events is not None:
        report['event_stats'] = {
            'count': len(events),
            'frequency': len(events) / report['signal_stats']['duration'] if report['signal_stats']['duration'] > 0 else 0,
            'valid_times': all(0 <= e.time <= report['signal_stats']['duration'] for e in events if hasattr(e, 'time'))
        }
        
    return report


# Example usage
if __name__ == "__main__":
    # Test validation functions
    
    # Create test data
    signal = np.random.randn(10000) * 10 - 50
    sampling_rate = 20000
    
    # Validate signal
    try:
        SynapticDataValidator.validate_signal(signal, sampling_rate, expected_range=(-200, 50))
        print("✓ Signal validation passed")
    except ValidationError as e:
        print(f"✗ Signal validation failed: {e}")
        
    # Create validation report
    report = create_validation_report(signal, sampling_rate)
    print("\nValidation Report:")
    print(f"  Duration: {report['signal_stats']['duration']:.2f} seconds")
    print(f"  Signal range: [{report['signal_stats']['min']:.2f}, {report['signal_stats']['max']:.2f}]")
    print(f"  Warnings: {report['warnings']}")