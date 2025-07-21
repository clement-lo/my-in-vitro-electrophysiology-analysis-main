"""
Base Analysis Classes for Electrophysiology Analysis
Provides abstract base classes and common functionality
"""

from abc import ABC, abstractmethod
import logging
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class AbstractAnalysis(ABC):
    """Abstract base class for all analysis modules."""
    
    def __init__(self, name: str, version: str = "2.0"):
        self.name = name
        self.version = version
        self.results = {}
        self.metadata = {
            'analysis_name': name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'parameters': {}
        }
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from file. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate analysis parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_analysis(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analysis. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def generate_report(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
        """Generate analysis report. Must be implemented by subclasses."""
        pass
    
    def export_results(self, output_path: Union[str, Path], format: str = 'pickle') -> Path:
        """Export analysis results in specified format."""
        output_path = Path(output_path)
        
        if format == 'pickle':
            with open(output_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump({
                    'results': self.results,
                    'metadata': self.metadata
                }, f)
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(self.results)
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump({
                    'results': json_results,
                    'metadata': self.metadata
                }, f, indent=2)
        elif format == 'csv':
            # Export tabular data as CSV
            if 'dataframe' in self.results:
                self.results['dataframe'].to_csv(output_path.with_suffix('.csv'), index=False)
            else:
                self.logger.warning("No dataframe found in results for CSV export")
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Results exported to {output_path}")
        return output_path
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Recursively convert numpy arrays and other non-JSON types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> Path:
        """Save analysis checkpoint for recovery."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{self.name}_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'metadata': self.metadata,
                'state': getattr(self, 'state', {})
            }, f)
            
        self.logger.info(f"Checkpoint saved to {checkpoint_file}")
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: Union[str, Path]) -> None:
        """Load analysis from checkpoint."""
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.results = checkpoint['results']
        self.metadata = checkpoint['metadata']
        if 'state' in checkpoint:
            self.state = checkpoint['state']
            
        self.logger.info(f"Checkpoint loaded from {checkpoint_file}")


class BatchAnalysis(AbstractAnalysis):
    """Base class for batch processing multiple files."""
    
    def __init__(self, name: str, version: str = "2.0"):
        super().__init__(name, version)
        self.batch_results = []
        
    def run_batch(self, file_list: list, parameters: Dict[str, Any], 
                  parallel: bool = False, n_jobs: int = -1) -> list:
        """Run analysis on multiple files."""
        if parallel:
            from joblib import Parallel, delayed
            
            self.batch_results = Parallel(n_jobs=n_jobs)(
                delayed(self._process_single_file)(file_path, parameters)
                for file_path in file_list
            )
        else:
            self.batch_results = []
            for file_path in file_list:
                result = self._process_single_file(file_path, parameters)
                self.batch_results.append(result)
                
        return self.batch_results
    
    def _process_single_file(self, file_path: Union[str, Path], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file."""
        try:
            data = self.load_data(file_path)
            results = self.run_analysis(data, parameters)
            results['file_path'] = str(file_path)
            results['success'] = True
            return results
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'success': False,
                'error': str(e)
            }
    
    def summarize_batch_results(self) -> pd.DataFrame:
        """Create summary DataFrame of batch results."""
        summary_data = []
        
        for result in self.batch_results:
            summary_row = {
                'file_path': result['file_path'],
                'success': result['success']
            }
            
            if result['success']:
                # Extract key metrics (subclasses should override)
                summary_row.update(self._extract_summary_metrics(result))
            else:
                summary_row['error'] = result.get('error', 'Unknown error')
                
            summary_data.append(summary_row)
            
        return pd.DataFrame(summary_data)
    
    def _extract_summary_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics from a single result. Override in subclasses."""
        return {}


class ValidationMixin:
    """Mixin class providing common validation methods."""
    
    @staticmethod
    def validate_numeric_parameter(value: Any, min_val: Optional[float] = None,
                                 max_val: Optional[float] = None, 
                                 param_name: str = "parameter") -> float:
        """Validate a numeric parameter."""
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{param_name} must be numeric, got {type(value)}")
            
        if min_val is not None and num_value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {num_value}")
            
        if max_val is not None and num_value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {num_value}")
            
        return num_value
    
    @staticmethod
    def validate_array_parameter(value: Any, expected_shape: Optional[tuple] = None,
                               expected_ndim: Optional[int] = None,
                               param_name: str = "array") -> np.ndarray:
        """Validate an array parameter."""
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except Exception:
                raise ValueError(f"{param_name} must be array-like")
                
        if expected_ndim is not None and value.ndim != expected_ndim:
            raise ValueError(f"{param_name} must have {expected_ndim} dimensions, got {value.ndim}")
            
        if expected_shape is not None and value.shape != expected_shape:
            raise ValueError(f"{param_name} must have shape {expected_shape}, got {value.shape}")
            
        return value
    
    @staticmethod
    def validate_choice_parameter(value: Any, choices: list, 
                                param_name: str = "parameter") -> Any:
        """Validate a choice parameter."""
        if value not in choices:
            raise ValueError(f"{param_name} must be one of {choices}, got {value}")
        return value
