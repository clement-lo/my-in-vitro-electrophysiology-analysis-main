"""
Base Analyzer Classes for Electrophysiology Analysis
===================================================

Abstract base classes that define the common interface and functionality
for all analysis modules in the electrophysiology analysis system.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results with metadata."""
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "2.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version
        }
    
    def save(self, path: Union[str, Path], format: str = 'json') -> None:
        """Save results to file."""
        path = Path(path)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format == 'npz':
            np.savez_compressed(path, **self.data, metadata=json.dumps(self.metadata))
        else:
            raise ValueError(f"Unknown format: {format}")
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnalysisResult':
        """Load results from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(
                data=data['data'],
                metadata=data['metadata'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                version=data['version']
            )
        else:
            raise ValueError(f"Unknown format: {path.suffix}")


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    Provides common functionality for data validation, preprocessing,
    analysis execution, and result management.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize analyzer with configuration.
        
        Parameters
        ----------
        config : Any, optional
            Configuration object specific to the analyzer type
        """
        self.config = config
        self._results = None
        self._data = None
        self._sampling_rate = None
        self._metadata = {}
        
    @property
    def results(self) -> Optional[AnalysisResult]:
        """Get analysis results."""
        return self._results
        
    @abstractmethod
    def analyze(self, data: np.ndarray, sampling_rate: float, 
                **kwargs) -> AnalysisResult:
        """
        Perform analysis on the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        sampling_rate : float
            Sampling rate in Hz
        **kwargs
            Additional analysis-specific parameters
            
        Returns
        -------
        AnalysisResult
            Analysis results with metadata
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: np.ndarray, sampling_rate: float) -> None:
        """
        Validate input data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to validate
        sampling_rate : float
            Sampling rate to validate
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        pass
    
    def preprocess(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Preprocess data before analysis.
        
        Default implementation returns data unchanged.
        Override in subclasses for specific preprocessing.
        
        Parameters
        ----------
        data : np.ndarray
            Raw data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        np.ndarray
            Preprocessed data
        """
        return data
    
    def run_analysis(self, data: np.ndarray, sampling_rate: float,
                    validate: bool = True, preprocess: bool = True,
                    **kwargs) -> AnalysisResult:
        """
        Complete analysis pipeline with validation and preprocessing.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        sampling_rate : float
            Sampling rate in Hz
        validate : bool
            Whether to validate data
        preprocess : bool
            Whether to preprocess data
        **kwargs
            Additional parameters for analysis
            
        Returns
        -------
        AnalysisResult
            Analysis results
        """
        # Store original data info
        self._data = data
        self._sampling_rate = sampling_rate
        
        # Validation
        if validate:
            self.validate_data(data, sampling_rate)
            
        # Preprocessing
        if preprocess:
            data = self.preprocess(data, sampling_rate)
            
        # Analysis
        self._results = self.analyze(data, sampling_rate, **kwargs)
        
        # Add standard metadata
        self._results.metadata.update({
            'sampling_rate': sampling_rate,
            'data_shape': data.shape,
            'duration': data.shape[-1] / sampling_rate,
            'analyzer': self.__class__.__name__,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        })
        
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis results.
        
        Returns
        -------
        dict
            Summary statistics and key findings
        """
        if self._results is None:
            return {"error": "No analysis results available"}
            
        return {
            "analyzer": self.__class__.__name__,
            "timestamp": self._results.timestamp,
            "duration": self._results.metadata.get('duration', 'N/A'),
            "key_metrics": self._get_key_metrics()
        }
        
    def _get_key_metrics(self) -> Dict[str, Any]:
        """
        Extract key metrics from results.
        
        Override in subclasses to provide specific metrics.
        """
        return {}


class BaseVisualizer(ABC):
    """Abstract base class for visualization components."""
    
    def __init__(self, style: str = 'default', dpi: int = 100):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        style : str
            Visualization style
        dpi : int
            Figure resolution
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
        
    @abstractmethod
    def _setup_style(self) -> None:
        """Configure visualization style."""
        pass
    
    @abstractmethod
    def plot(self, results: AnalysisResult, **kwargs) -> Any:
        """
        Create visualization from results.
        
        Parameters
        ----------
        results : AnalysisResult
            Analysis results to visualize
        **kwargs
            Plotting parameters
            
        Returns
        -------
        Any
            Figure or axis object
        """
        pass


class BatchAnalyzer:
    """Helper class for batch processing multiple files."""
    
    def __init__(self, analyzer: BaseAnalyzer):
        """
        Initialize batch processor.
        
        Parameters
        ----------
        analyzer : BaseAnalyzer
            Analyzer instance to use for processing
        """
        self.analyzer = analyzer
        self.results = {}
        self.errors = {}
        
    def process_files(self, file_paths: List[Union[str, Path]],
                     data_loader_func: callable,
                     parallel: bool = False,
                     **kwargs) -> Dict[str, AnalysisResult]:
        """
        Process multiple files.
        
        Parameters
        ----------
        file_paths : list
            List of file paths to process
        data_loader_func : callable
            Function to load data from file
        parallel : bool
            Whether to process in parallel
        **kwargs
            Additional parameters for analysis
            
        Returns
        -------
        dict
            Results keyed by file path
        """
        if parallel:
            # TODO: Implement parallel processing
            logger.warning("Parallel processing not yet implemented, using sequential")
            
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                # Load data
                data, sampling_rate, metadata = data_loader_func(file_path)
                
                # Run analysis
                result = self.analyzer.run_analysis(data, sampling_rate, **kwargs)
                result.metadata['source_file'] = str(file_path)
                result.metadata['file_metadata'] = metadata
                
                self.results[str(file_path)] = result
                logger.info(f"Successfully processed: {file_path}")
                
            except Exception as e:
                self.errors[str(file_path)] = str(e)
                logger.error(f"Failed to process {file_path}: {e}")
                
        return self.results
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for file_path, result in self.results.items():
            output_name = Path(file_path).stem + '_results.json'
            result.save(output_dir / output_name)
            
        # Save summary
        summary = {
            'processed': len(self.results),
            'errors': len(self.errors),
            'files': list(self.results.keys()),
            'error_files': self.errors,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# Configuration base classes
@dataclass
class BaseConfig:
    """Base configuration class for all analyzers."""
    # Common parameters
    verbose: bool = False
    save_intermediate: bool = False
    output_format: str = 'json'
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.output_format not in ['json', 'npz', 'mat']:
            raise ValueError(f"Unknown output format: {self.output_format}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)