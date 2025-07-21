"""
Template for creating new analysis modules
Copy this file and implement the abstract methods
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.base_analysis import AbstractAnalysis, ValidationMixin
from ..core.data_pipeline import load_data
from ..core.config_manager import ConfigManager

class TemplateAnalysis(AbstractAnalysis, ValidationMixin):
    """Template analysis module - replace with your analysis."""
    
    def __init__(self, config: Union[Dict, ConfigManager, Path, str, None] = None):
        super().__init__(name="template_analysis", version="2.0")
        
        # Handle different config types
        if isinstance(config, ConfigManager):
            self.config_manager = config
        elif isinstance(config, (Path, str)):
            self.config_manager = ConfigManager(config)
        elif isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config_manager.config = config
        else:
            self.config_manager = ConfigManager()
            
        self.config = self.config_manager.create_analysis_config('template')
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data using unified pipeline."""
        signal, rate, time, metadata = load_data(file_path, **kwargs)
        
        # Store as structured data
        self.data = {
            'signal': signal,
            'sampling_rate': rate,
            'time': time,
            'metadata': metadata
        }
        
        return self.data
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate analysis parameters."""
        # Example validation
        required = ['param1', 'param2']
        
        for param in required:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
        # Use validation mixin methods
        parameters['param1'] = self.validate_numeric_parameter(
            parameters['param1'], 
            min_val=0, 
            max_val=100,
            param_name='param1'
        )
        
        return True
    
    def run_analysis(self, data: Any = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the analysis."""
        if data is None:
            data = self.data
            
        if parameters is None:
            parameters = self.config.get('template', {})
            
        # Validate parameters
        self.validate_parameters(parameters)
        
        # Store parameters in metadata
        self.metadata['parameters'] = parameters
        
        # Perform analysis
        results = self._perform_analysis(data, parameters)
        
        # Store results
        self.results = results
        
        return results
    
    def _perform_analysis(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Actual analysis implementation."""
        # Example analysis
        signal = data['signal']
        rate = data['sampling_rate']
        
        # Calculate some metrics
        results = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'duration': len(signal) / rate
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any] = None, output_dir: Union[str, Path] = None) -> Path:
        """Generate analysis report."""
        if results is None:
            results = self.results
            
        if output_dir is None:
            output_dir = Path('./results')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot signal
        ax1 = axes[0]
        ax1.plot(self.data['time'], self.data['signal'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')
        ax1.set_title('Raw Signal')
        
        # Plot metrics
        ax2 = axes[1]
        metrics = ['mean', 'std', 'min', 'max']
        values = [results[m] for m in metrics]
        ax2.bar(metrics, values)
        ax2.set_ylabel('Value')
        ax2.set_title('Signal Metrics')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / f"{self.name}_report.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save text report
        report_path = output_dir / f"{self.name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"{self.name.replace('_', ' ').title()} Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Parameters:\n")
            for key, value in self.metadata.get('parameters', {}).items():
                f.write(f"  {key}: {value}\n")
                
            f.write("\nResults:\n")
            for key, value in results.items():
                f.write(f"  {key}: {value:.4f}\n")
                
        self.logger.info(f"Report saved to {output_dir}")
        
        return report_path
    
    # Additional analysis-specific methods
    def custom_method(self) -> Any:
        """Add your custom methods here."""
        pass


# Example usage
if __name__ == "__main__":
    # Create analysis instance
    analysis = TemplateAnalysis()
    
    # Load data
    data = analysis.load_data('path/to/data.nwb')
    
    # Run analysis
    results = analysis.run_analysis(parameters={'param1': 50, 'param2': 'value'})
    
    # Generate report
    analysis.generate_report(output_dir='./results')
    
    # Export results
    analysis.export_results('./results/template_results', format='json')
