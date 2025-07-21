"""
Custom Exceptions for Electrophysiology Analysis
===============================================

Centralized exception definitions for better error handling
and debugging across all modules.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""


class EphysAnalysisError(Exception):
    """Base exception for all electrophysiology analysis errors."""
    pass


# Data-related exceptions
class DataError(EphysAnalysisError):
    """Base exception for data-related errors."""
    pass


class DataLoadError(DataError):
    """Error loading data from file."""
    pass


class DataFormatError(DataError):
    """Error with data format or structure."""
    pass


class DataValidationError(DataError):
    """Data validation failed."""
    pass


# Analysis-related exceptions
class AnalysisError(EphysAnalysisError):
    """Base exception for analysis-related errors."""
    pass


class PreprocessingError(AnalysisError):
    """Error during data preprocessing."""
    pass


class DetectionError(AnalysisError):
    """Error during event/feature detection."""
    pass


class FittingError(AnalysisError):
    """Error during curve fitting or parameter estimation."""
    pass


class ConvergenceError(FittingError):
    """Optimization/fitting failed to converge."""
    pass


# Configuration-related exceptions
class ConfigurationError(EphysAnalysisError):
    """Base exception for configuration errors."""
    pass


class InvalidParameterError(ConfigurationError):
    """Invalid parameter value or combination."""
    pass


class MissingParameterError(ConfigurationError):
    """Required parameter is missing."""
    pass


# File I/O exceptions
class FileIOError(EphysAnalysisError):
    """Base exception for file I/O errors."""
    pass


class UnsupportedFormatError(FileIOError):
    """File format is not supported."""
    pass


class CorruptedFileError(FileIOError):
    """File appears to be corrupted or unreadable."""
    pass


# Dependency exceptions
class DependencyError(EphysAnalysisError):
    """Base exception for dependency-related errors."""
    pass


class MissingDependencyError(DependencyError):
    """Required dependency is not installed."""
    pass


class IncompatibleVersionError(DependencyError):
    """Dependency version is incompatible."""
    pass


# Runtime exceptions
class RuntimeError(EphysAnalysisError):
    """Base exception for runtime errors."""
    pass


class MemoryError(RuntimeError):
    """Insufficient memory for operation."""
    pass


class TimeoutError(RuntimeError):
    """Operation timed out."""
    pass


class ResourceError(RuntimeError):
    """Resource (file, network, etc.) is unavailable."""
    pass


# Visualization exceptions
class VisualizationError(EphysAnalysisError):
    """Base exception for visualization errors."""
    pass


class PlottingError(VisualizationError):
    """Error creating plot or figure."""
    pass


class ExportError(VisualizationError):
    """Error exporting figure or visualization."""
    pass


# Validation exceptions
class ValidationError(EphysAnalysisError):
    """Base exception for validation errors."""
    pass


class DataQualityError(ValidationError):
    """Data quality is below acceptable threshold."""
    pass


class ResultValidationError(ValidationError):
    """Analysis results failed validation."""
    pass


# Helper functions for exception handling
def format_exception_message(exception: Exception, context: dict = None) -> str:
    """
    Format exception message with optional context.
    
    Parameters
    ----------
    exception : Exception
        The exception to format
    context : dict, optional
        Additional context information
        
    Returns
    -------
    str
        Formatted error message
    """
    message = f"{exception.__class__.__name__}: {str(exception)}"
    
    if context:
        message += "\nContext:"
        for key, value in context.items():
            message += f"\n  {key}: {value}"
            
    return message


def handle_missing_dependency(package_name: str, install_command: str = None) -> None:
    """
    Raise appropriate error for missing dependency.
    
    Parameters
    ----------
    package_name : str
        Name of the missing package
    install_command : str, optional
        Installation command (if not provided, uses pip)
        
    Raises
    ------
    MissingDependencyError
        Always raised with helpful message
    """
    if install_command is None:
        install_command = f"pip install {package_name}"
        
    raise MissingDependencyError(
        f"{package_name} is required but not installed. "
        f"Install with: {install_command}"
    )


def validate_data_quality(data, sampling_rate: float, min_duration: float = 0.1) -> None:
    """
    Common data quality validation.
    
    Parameters
    ----------
    data : array-like
        Data to validate
    sampling_rate : float
        Sampling rate in Hz
    min_duration : float
        Minimum acceptable duration in seconds
        
    Raises
    ------
    DataQualityError
        If data quality is insufficient
    """
    import numpy as np
    
    if len(data) == 0:
        raise DataQualityError("Data is empty")
        
    duration = len(data) / sampling_rate
    if duration < min_duration:
        raise DataQualityError(
            f"Data duration ({duration:.3f}s) is less than "
            f"minimum required ({min_duration}s)"
        )
        
    if np.all(np.isnan(data)):
        raise DataQualityError("Data contains only NaN values")
        
    if np.all(data == data[0]):
        raise DataQualityError("Data is constant (no variation)")


# Exception context manager
class ExceptionContext:
    """
    Context manager for adding context to exceptions.
    
    Example
    -------
    >>> with ExceptionContext(file_path="data.abf", operation="loading"):
    ...     # code that might raise exception
    """
    
    def __init__(self, **context):
        """Initialize with context information."""
        self.context = context
        
    def __enter__(self):
        """Enter context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, potentially modifying exception."""
        if exc_val is not None and isinstance(exc_val, EphysAnalysisError):
            # Add context to our custom exceptions
            if not hasattr(exc_val, 'context'):
                exc_val.context = {}
            exc_val.context.update(self.context)
            
            # Update exception message
            exc_val.args = (format_exception_message(exc_val, self.context),)
            
        return False  # Don't suppress exception