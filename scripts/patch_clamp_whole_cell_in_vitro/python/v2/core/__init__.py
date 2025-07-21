"""
Core Infrastructure for Electrophysiology Analysis
=================================================

This module provides the foundational components for all analysis modules:
- Base analyzer classes
- Data loading utilities
- Configuration management
- Custom exceptions

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

# Base classes
from .base_analyzer import (
    BaseAnalyzer,
    BaseVisualizer,
    BaseConfig,
    AnalysisResult,
    BatchAnalyzer
)

# Data loading
from .data_loader import (
    DataLoader,
    FileFormat,
    load_data,
    DataLoaderError
)

# Configuration management
from .config_manager import (
    ConfigManager,
    ConfigError,
    PreprocessingConfig,
    AnalysisConfig,
    config_manager,
    load_config,
    save_config,
    create_default_config
)

# Exceptions
from .exceptions import (
    EphysAnalysisError,
    DataError,
    DataLoadError,
    DataFormatError,
    DataValidationError,
    AnalysisError,
    PreprocessingError,
    DetectionError,
    FittingError,
    ConvergenceError,
    ConfigurationError,
    InvalidParameterError,
    MissingParameterError,
    FileIOError,
    UnsupportedFormatError,
    CorruptedFileError,
    DependencyError,
    MissingDependencyError,
    IncompatibleVersionError,
    RuntimeError,
    MemoryError,
    TimeoutError,
    ResourceError,
    VisualizationError,
    PlottingError,
    ExportError,
    ValidationError,
    DataQualityError,
    ResultValidationError,
    ExceptionContext,
    format_exception_message,
    handle_missing_dependency,
    validate_data_quality
)

__all__ = [
    # Base classes
    'BaseAnalyzer',
    'BaseVisualizer', 
    'BaseConfig',
    'AnalysisResult',
    'BatchAnalyzer',
    
    # Data loading
    'DataLoader',
    'FileFormat',
    'load_data',
    'DataLoaderError',
    
    # Configuration
    'ConfigManager',
    'ConfigError',
    'PreprocessingConfig',
    'AnalysisConfig',
    'config_manager',
    'load_config',
    'save_config',
    'create_default_config',
    
    # Exceptions
    'EphysAnalysisError',
    'DataError',
    'DataLoadError',
    'DataFormatError',
    'DataValidationError',
    'AnalysisError',
    'PreprocessingError',
    'DetectionError',
    'FittingError',
    'ConvergenceError',
    'ConfigurationError',
    'InvalidParameterError',
    'MissingParameterError',
    'FileIOError',
    'UnsupportedFormatError',
    'CorruptedFileError',
    'DependencyError',
    'MissingDependencyError',
    'IncompatibleVersionError',
    'RuntimeError',
    'MemoryError',
    'TimeoutError',
    'ResourceError',
    'VisualizationError',
    'PlottingError',
    'ExportError',
    'ValidationError',
    'DataQualityError',
    'ResultValidationError',
    'ExceptionContext',
    'format_exception_message',
    'handle_missing_dependency',
    'validate_data_quality'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Electrophysiology Analysis System'