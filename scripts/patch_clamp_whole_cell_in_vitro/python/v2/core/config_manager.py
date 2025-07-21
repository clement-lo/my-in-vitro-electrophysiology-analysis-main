"""
Configuration Management System
==============================

Centralized configuration management for all analysis modules.
Supports YAML, JSON, and programmatic configuration with validation.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    # General settings
    verbose: bool = False
    debug: bool = False
    
    # Output settings
    save_results: bool = True
    output_dir: Optional[str] = None
    output_format: str = 'json'  # 'json', 'npz', 'mat', 'hdf5'
    
    # Processing settings
    parallel: bool = False
    n_jobs: int = -1  # -1 means use all available cores
    chunk_size: Optional[int] = None
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_formats = ['json', 'npz', 'mat', 'hdf5']
        if self.output_format not in valid_formats:
            raise ConfigError(f"Invalid output format: {self.output_format}. Must be one of {valid_formats}")
            
        if self.n_jobs == 0:
            raise ConfigError("n_jobs cannot be 0")
            
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ConfigError("chunk_size must be positive")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass 
class PreprocessingConfig(BaseConfig):
    """Configuration for signal preprocessing."""
    # Detrending
    detrend: bool = True
    detrend_type: str = 'linear'  # 'linear', 'constant', 'polynomial'
    detrend_order: int = 1  # For polynomial detrending
    
    # Filtering
    filter_data: bool = True
    filter_type: str = 'bandpass'  # 'lowpass', 'highpass', 'bandpass', 'bandstop'
    filter_low_cutoff: float = 1.0  # Hz
    filter_high_cutoff: float = 1000.0  # Hz
    filter_order: int = 4
    filter_method: str = 'butter'  # 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
    
    # Baseline correction
    baseline_correct: bool = True
    baseline_window: float = 0.1  # seconds
    baseline_method: str = 'mean'  # 'mean', 'median', 'polynomial'
    
    # Artifact removal
    remove_artifacts: bool = False
    artifact_threshold: float = 5.0  # Standard deviations
    artifact_method: str = 'threshold'  # 'threshold', 'template', 'ica'
    
    # Resampling
    resample: bool = False
    resample_rate: Optional[float] = None  # Target sampling rate in Hz
    
    def validate(self) -> None:
        """Validate preprocessing parameters."""
        super().validate()
        
        # Validate filter parameters
        if self.filter_data:
            valid_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
            if self.filter_type not in valid_types:
                raise ConfigError(f"Invalid filter type: {self.filter_type}")
                
            if self.filter_type in ['bandpass', 'bandstop']:
                if self.filter_low_cutoff >= self.filter_high_cutoff:
                    raise ConfigError("Low cutoff must be less than high cutoff")
                    
            valid_methods = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
            if self.filter_method not in valid_methods:
                raise ConfigError(f"Invalid filter method: {self.filter_method}")
                
        # Validate detrend parameters
        valid_detrend = ['linear', 'constant', 'polynomial']
        if self.detrend_type not in valid_detrend:
            raise ConfigError(f"Invalid detrend type: {self.detrend_type}")
            
        # Validate baseline parameters
        if self.baseline_window <= 0:
            raise ConfigError("Baseline window must be positive")
            
        valid_baseline = ['mean', 'median', 'polynomial']
        if self.baseline_method not in valid_baseline:
            raise ConfigError(f"Invalid baseline method: {self.baseline_method}")


@dataclass
class AnalysisConfig(PreprocessingConfig):
    """Extended configuration for analysis modules."""
    # Additional analysis-specific parameters can be added here
    # This serves as a base for module-specific configurations
    pass


class ConfigManager:
    """
    Centralized configuration management system.
    
    Handles loading, saving, validation, and merging of configurations
    from various sources (files, dictionaries, command line).
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self._configs: Dict[str, BaseConfig] = {}
        self._config_classes: Dict[str, Type[BaseConfig]] = {
            'base': BaseConfig,
            'preprocessing': PreprocessingConfig,
            'analysis': AnalysisConfig
        }
        
    def register_config_class(self, name: str, config_class: Type[BaseConfig]) -> None:
        """
        Register a new configuration class.
        
        Parameters
        ----------
        name : str
            Name to register the class under
        config_class : Type[BaseConfig]
            Configuration class (must inherit from BaseConfig)
        """
        if not issubclass(config_class, BaseConfig):
            raise ConfigError(f"Config class must inherit from BaseConfig")
        self._config_classes[name] = config_class
        
    def create_config(self, config_type: str = 'base', **kwargs) -> BaseConfig:
        """
        Create a new configuration instance.
        
        Parameters
        ----------
        config_type : str
            Type of configuration to create
        **kwargs
            Configuration parameters
            
        Returns
        -------
        BaseConfig
            Configuration instance
        """
        if config_type not in self._config_classes:
            raise ConfigError(f"Unknown config type: {config_type}")
            
        config_class = self._config_classes[config_type]
        config = config_class(**kwargs)
        config.validate()
        return config
        
    def load_from_file(self, file_path: Union[str, Path], 
                      config_type: str = 'base') -> BaseConfig:
        """
        Load configuration from file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to configuration file (YAML or JSON)
        config_type : str
            Type of configuration to create
            
        Returns
        -------
        BaseConfig
            Loaded configuration
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
            
        # Load based on extension
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ConfigError(f"Unsupported config format: {path.suffix}")
            
        # Create configuration
        config_class = self._config_classes[config_type]
        config = config_class.from_dict(data)
        config.validate()
        
        logger.info(f"Loaded {config_type} configuration from {path}")
        return config
        
    def save_to_file(self, config: BaseConfig, file_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        config : BaseConfig
            Configuration to save
        file_path : str or Path
            Output file path
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = config.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ConfigError(f"Unsupported config format: {path.suffix}")
            
        logger.info(f"Saved configuration to {path}")
        
    def merge_configs(self, base_config: BaseConfig, 
                     override_config: Union[BaseConfig, Dict[str, Any]]) -> BaseConfig:
        """
        Merge two configurations, with override taking precedence.
        
        Parameters
        ----------
        base_config : BaseConfig
            Base configuration
        override_config : BaseConfig or dict
            Configuration to override with
            
        Returns
        -------
        BaseConfig
            Merged configuration
        """
        # Convert to dictionaries
        base_dict = base_config.to_dict()
        
        if isinstance(override_config, BaseConfig):
            override_dict = override_config.to_dict()
        else:
            override_dict = override_config
            
        # Merge dictionaries
        merged_dict = deepcopy(base_dict)
        merged_dict.update(override_dict)
        
        # Create new config instance
        config_class = type(base_config)
        merged_config = config_class.from_dict(merged_dict)
        merged_config.validate()
        
        return merged_config
        
    def get_default_config(self, config_type: str = 'base') -> BaseConfig:
        """
        Get default configuration for a given type.
        
        Parameters
        ----------
        config_type : str
            Configuration type
            
        Returns
        -------
        BaseConfig
            Default configuration instance
        """
        if config_type not in self._config_classes:
            raise ConfigError(f"Unknown config type: {config_type}")
            
        return self._config_classes[config_type]()
        
    def validate_config_file(self, file_path: Union[str, Path], 
                           config_type: str = 'base') -> bool:
        """
        Validate a configuration file without loading it.
        
        Parameters
        ----------
        file_path : str or Path
            Path to configuration file
        config_type : str
            Expected configuration type
            
        Returns
        -------
        bool
            True if valid, raises ConfigError if invalid
        """
        try:
            self.load_from_file(file_path, config_type)
            return True
        except Exception as e:
            raise ConfigError(f"Invalid configuration file: {e}")


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def load_config(file_path: Union[str, Path], config_type: str = 'analysis') -> AnalysisConfig:
    """Load analysis configuration from file."""
    return config_manager.load_from_file(file_path, config_type)


def save_config(config: BaseConfig, file_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config_manager.save_to_file(config, file_path)


def create_default_config() -> AnalysisConfig:
    """Create default analysis configuration."""
    return config_manager.get_default_config('analysis')


if __name__ == "__main__":
    # Example usage
    print("Configuration Manager Example")
    print("=" * 50)
    
    # Create default configuration
    config = create_default_config()
    print("\nDefault configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save to YAML
    yaml_path = Path("example_config.yaml")
    save_config(config, yaml_path)
    print(f"\nSaved configuration to {yaml_path}")
    
    # Modify and save
    config.filter_low_cutoff = 0.5
    config.filter_high_cutoff = 2000.0
    config.parallel = True
    
    json_path = Path("example_config.json")
    save_config(config, json_path)
    print(f"Saved modified configuration to {json_path}")
    
    # Clean up example files
    yaml_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)