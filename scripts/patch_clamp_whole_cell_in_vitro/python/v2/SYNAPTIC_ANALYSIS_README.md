# Complete Synaptic Analysis Module

## Overview

The Complete Synaptic Analysis Module provides a comprehensive, production-ready solution for analyzing synaptic currents in electrophysiology data. It integrates event detection, kinetic analysis, and input-output relationships with support for multiple file formats and advanced visualization.

## Features

### 🔌 Multi-Format Data Loading
- **PyNWB** - Neurodata Without Borders standard format
- **PyABF** - Axon Binary Format from Molecular Devices
- **Neo** - Support for Spike2, AxoGraph, and other formats
- **CSV** - Time-series and tabular data
- **HDF5** - Generic hierarchical data format

### 📊 Analysis Capabilities
- **Event Detection**
  - Threshold-based detection with adaptive baseline
  - Template matching (extensible)
  - Deconvolution methods (extensible)
  
- **Kinetic Analysis**
  - Single exponential decay fitting
  - Bi-exponential decay fitting
  - Alpha function fitting (rise and decay)
  
- **Input-Output Analysis**
  - Sigmoid curve fitting
  - Hill equation fitting
  - Boltzmann function fitting
  - Linear regression with confidence intervals

### 📈 Advanced Visualization
- Publication-quality figures with Matplotlib and Seaborn
- Comprehensive event summary dashboards
- Statistical overlays and confidence intervals
- Multi-condition comparison plots
- Customizable styles (publication, presentation, notebook)

### ✅ Validation & Quality Control
- Comprehensive data validation
- Parameter validation with warnings
- Quality metrics and reports
- File format verification

## Installation

```bash
# Core dependencies
pip install numpy scipy pandas matplotlib seaborn

# Optional dependencies for file formats
pip install pyabf        # For ABF files
pip install pynwb        # For NWB files
pip install neo          # For Neo-compatible formats
pip install h5py         # For HDF5 files
```

## Quick Start

### Basic Event Detection

```python
from synaptic_analysis_complete import DataLoader, EventDetectionAnalyzer, AnalysisConfig

# Load data
data, sampling_rate, metadata = DataLoader.load('recording.abf')

# Configure analysis
config = AnalysisConfig(
    threshold_std=3.0,
    kinetic_model='exponential'
)

# Analyze
analyzer = EventDetectionAnalyzer(config)
results = analyzer.analyze(data, sampling_rate)

# Visualize
analyzer.visualize()
```

### Input-Output Analysis

```python
from synaptic_analysis_complete import InputOutputAnalyzer

# Prepare I/O data
input_intensities = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
output_responses = np.array([5, 12, 25, 45, 65, 78, 85, 90, 93, 95, 96])

# Analyze
analyzer = InputOutputAnalyzer(AnalysisConfig(io_model='sigmoid'))
results = analyzer.analyze(input_intensities, output_responses)

# Visualize with confidence intervals
analyzer.visualize(show_confidence=True)
```

### Combined Analysis

```python
from synaptic_analysis_complete import CombinedSynapticAnalyzer

# Perform both event detection and I/O analysis
analyzer = CombinedSynapticAnalyzer()
results = analyzer.analyze(
    data, sampling_rate,
    input_data=input_vals,
    output_data=output_vals
)

# Create comprehensive visualization
analyzer.visualize()
```

## Configuration Options

The `AnalysisConfig` class provides extensive customization:

```python
config = AnalysisConfig(
    # Preprocessing
    detrend=True,
    filter_type='bandpass',  # 'bandpass', 'lowpass', 'highpass', 'none'
    filter_freq_low=1.0,     # Hz
    filter_freq_high=1000.0, # Hz
    
    # Event detection
    detection_method='threshold',  # 'threshold', 'template', 'deconvolution'
    threshold_std=3.0,
    min_event_interval=0.005,      # seconds
    
    # Kinetics
    kinetic_model='exponential',   # 'exponential', 'biexponential', 'alpha'
    fit_window=0.05,               # seconds
    
    # Input-Output
    io_model='sigmoid',            # 'sigmoid', 'linear', 'hill', 'boltzmann'
    io_normalize=True,
    
    # Statistics
    bootstrap_iterations=1000,
    confidence_level=0.95,
    
    # Visualization
    plot_style='seaborn',
    figure_dpi=300,
    save_figures=True
)
```

## Advanced Usage

### Custom Event Detection

```python
class CustomEventDetector(EventDetectionAnalyzer):
    def _detect_events(self, signal, sampling_rate):
        # Implement custom detection algorithm
        events = []
        # ... your detection logic ...
        return events
```

### Batch Processing

```python
from pathlib import Path

# Process multiple files
data_dir = Path('data/')
results_dict = {}

for file_path in data_dir.glob('*.abf'):
    try:
        data, rate, _ = DataLoader.load(file_path)
        analyzer = EventDetectionAnalyzer()
        results = analyzer.analyze(data, rate)
        results_dict[file_path.name] = results
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
```

### Multi-Condition Comparison

```python
from utils.synaptic_visualization import SynapticVisualizer

# Analyze multiple conditions
conditions = ['Control', 'Drug A', 'Drug B']
io_results = []

for condition_data in [control_data, drug_a_data, drug_b_data]:
    analyzer = InputOutputAnalyzer()
    results = analyzer.analyze(condition_data['input'], condition_data['output'])
    io_results.append(results)

# Compare results
visualizer = SynapticVisualizer()
fig = visualizer.plot_io_curve_comparison(io_results, conditions)
```

## Validation Framework

### Data Validation

```python
from utils.synaptic_validation import SynapticDataValidator

# Validate before analysis
SynapticDataValidator.validate_signal(
    data, sampling_rate,
    expected_range=(-500, 100)  # pA
)

# Create validation report
from utils.synaptic_validation import create_validation_report
report = create_validation_report(data, sampling_rate)
print(f"Data quality: {report['quality_checks']}")
```

### Parameter Validation

```python
from utils.synaptic_validation import validate_analysis_config

# Validate configuration
try:
    validate_analysis_config(config)
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

## Visualization Options

### Publication Figures

```python
from utils.synaptic_visualization import SynapticVisualizer

# Create publication-ready figure
visualizer = SynapticVisualizer(style='publication', dpi=600)
fig = visualizer.create_publication_figure(
    results,
    panels=['trace', 'histogram'],
    figsize=(7, 5)
)
fig.savefig('figure1.pdf', format='pdf')
```

### Custom Visualizations

```python
# Event summary dashboard
fig = visualizer.plot_event_summary(
    events, signal, sampling_rate,
    figsize=(16, 12)
)

# Statistical overlays
visualizer.plot_amplitude_distribution(ax, events)
```

## File Format Examples

### Loading NWB Files

```python
# NWB files with rich metadata
data, rate, metadata = DataLoader.load('experiment.nwb')
print(f"Experimenter: {metadata['experimenter']}")
print(f"Session: {metadata['session_id']}")
```

### Loading ABF Files

```python
# ABF files from pCLAMP
data, rate, metadata = DataLoader.load('voltage_clamp.abf')
print(f"Protocol: {metadata['protocol']}")
print(f"Channels: {metadata['channel_count']}")
```

### Custom HDF5 Format

```python
# HDF5 with custom structure
import h5py

# Save data
with h5py.File('custom_data.h5', 'w') as f:
    f.create_dataset('data', data=signal)
    f.attrs['sampling_rate'] = 20000
    f.attrs['units'] = 'pA'

# Load with DataLoader
data, rate, metadata = DataLoader.load('custom_data.h5')
```

## Performance Considerations

- **Large Files**: Data is loaded into memory. For files >2GB, consider chunked processing
- **Parallel Processing**: Set `config.parallel_processing = True` for multi-core support
- **Caching**: Results are not cached by default. Implement caching for repeated analyses
- **Memory Usage**: Event detection stores all events. For very long recordings (>1 hour), consider segmented analysis

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Check installed packages
   import synaptic_analysis_complete
   print(synaptic_analysis_complete.__version__)
   ```

2. **File Loading Failures**
   ```python
   # Verify file format support
   from synaptic_analysis_complete import DataLoader, FileFormat
   print(DataLoader.detect_format('myfile.abf'))
   ```

3. **Event Detection Issues**
   - Check signal polarity (inward currents should be negative)
   - Adjust threshold_std parameter (try 2.0-5.0)
   - Verify filtering parameters don't remove signal

4. **Fitting Failures**
   - Check data quality with validation tools
   - Try different models (exponential vs biexponential)
   - Adjust fit_window parameter

## API Reference

### Core Classes

- `DataLoader` - Unified file loading
- `EventDetectionAnalyzer` - Synaptic event detection
- `InputOutputAnalyzer` - I/O relationship analysis
- `CombinedSynapticAnalyzer` - Combined analysis
- `AnalysisConfig` - Configuration parameters

### Validation Classes

- `SynapticDataValidator` - Data validation
- `ParameterValidator` - Parameter validation
- `FileValidator` - File format validation

### Visualization Classes

- `SynapticVisualizer` - Advanced plotting
- Helper functions for quick visualizations

## Contributing

This module is designed for extensibility. To add new features:

1. **New File Format**: Add loader method to `DataLoader`
2. **New Detection Method**: Override `_detect_events` in `EventDetectionAnalyzer`
3. **New Kinetic Model**: Add fitting function to event analyzer
4. **New Visualization**: Extend `SynapticVisualizer`

## Citation

If you use this module in your research, please cite:
```
Electrophysiology Analysis System v2.0
Complete Synaptic Analysis Module
https://github.com/your-repo/electrophysiology-analysis
```

## License

This module is part of the Electrophysiology Analysis System.
See LICENSE file for details.