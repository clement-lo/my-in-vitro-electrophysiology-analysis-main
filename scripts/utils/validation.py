# Last reviewed by Claude Desktop on 2025-07-17 22:37:57
"""
Validation utilities for electrophysiology analysis scripts.
Provides reusable validation functions for parameters, configurations, and data.
"""

import os
import yaml
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If file doesn't exist when required
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    if must_exist and not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return True


def validate_numeric_range(value: Union[int, float], 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          param_name: str = "value") -> bool:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for error messages
        
    Returns:
        bool: True if valid
        
    Raises:
        TypeError: If value is not numeric
        ValueError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be numeric, got {type(value).__name__}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
    
    return True


def validate_array(arr: Any, 
                  ndim: Optional[int] = None,
                  shape: Optional[Tuple[int, ...]] = None,
                  dtype: Optional[type] = None,
                  param_name: str = "array") -> np.ndarray:
    """
    Validate numpy array properties.
    
    Args:
        arr: Array to validate
        ndim: Expected number of dimensions
        shape: Expected shape
        dtype: Expected data type
        param_name: Parameter name for error messages
        
    Returns:
        np.ndarray: Validated array
        
    Raises:
        TypeError: If not array-like
        ValueError: If properties don't match
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception as e:
            raise TypeError(f"{param_name} must be array-like: {e}")
    
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{param_name} must have {ndim} dimensions, got {arr.ndim}")
    
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{param_name} must have shape {shape}, got {arr.shape}")
    
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        raise TypeError(f"{param_name} must have dtype {dtype}, got {arr.dtype}")
    
    return arr


def validate_probability(value: float, param_name: str = "probability") -> bool:
    """Validate probability value is between 0 and 1."""
    return validate_numeric_range(value, 0.0, 1.0, param_name)


def validate_positive_int(value: Any, param_name: str = "value") -> int:
    """Validate positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{param_name} must be a positive integer, got {value}")
    return value


def load_and_validate_config(config_path: str, 
                           required_fields: Optional[Dict[str, List[str]]] = None) -> Dict:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to YAML config file
        required_fields: Dict of required fields per section
        
    Returns:
        Dict: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
        ValueError: If required fields missing
    """
    validate_file_path(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}")
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config).__name__}")
    
    # Validate required fields
    if required_fields:
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
            
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing required field '{field}' in section '{section}'")
    
    logger.info(f"Successfully loaded and validated config from {config_path}")
    return config


def safe_file_operation(operation: str = "read"):
    """
    Decorator for safe file operations with error handling.
    
    Args:
        operation: Type of operation ('read', 'write')
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"File not found during {operation}: {e}")
                raise
            except PermissionError as e:
                logger.error(f"Permission denied during {operation}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during {operation}: {e}")
                raise
        return wrapper
    return decorator