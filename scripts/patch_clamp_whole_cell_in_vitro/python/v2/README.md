# v2 - Research-Grade Electrophysiology Analysis Framework

Advanced, extensible framework for patch-clamp and whole-cell electrophysiology analysis.

## Overview

This directory contains a comprehensive, object-oriented framework designed for:
- Production research environments
- Complex analysis pipelines
- Multi-format data handling
- Extensible architecture
- Publication-ready outputs

## Architecture

```
v2/
├── core/               # Core infrastructure
│   ├── base_analyzer.py    # Abstract base classes
│   ├── data_loader.py      # Multi-format loader
│   ├── config_manager.py   # Configuration system
│   └── exceptions.py       # Custom exceptions
├── common/             # Shared utilities
│   ├── preprocessing/      # Signal processing
│   ├── statistics/         # Statistical analysis
│   ├── validation/         # Data validation
│   └── visualization/      # Plotting utilities
├── modules/            # Analysis modules
│   ├── action_potential/   # AP analysis
│   ├── synaptic/          # Synaptic analysis
│   ├── ion_channels/      # Channel analysis
│   ├── network_connectivity/ # Network analysis
│   ├── pharmacology/      # Drug modulation
│   └── time_frequency/    # Spectral analysis
└── workflows/          # Complete pipelines
```

## Key Features

### 1. Multi-Format Data Loading
```python
from core.data_loader import DataLoader

loader = DataLoader()
# Automatically detects format
data = loader.load_file("recording.abf")  # ABF
data = loader.load_file("data.nwb")       # NWB
data = loader.load_file("trace.csv")      # CSV
```

### 2. Comprehensive Configuration
```python
from core.config_manager import ConfigManager

config = ConfigManager.from_file("config.yaml")
# Or programmatically
config = ConfigManager()
config.preprocessing.filter_type = "butterworth"
config.analysis.spike_threshold = -20
```

### 3. Object-Oriented Analysis
```python
from modules.action_potential import ActionPotentialAnalyzer
from modules.synaptic import SynapticAnalyzer

# Initialize analyzers
ap_analyzer = ActionPotentialAnalyzer(config)
syn_analyzer = SynapticAnalyzer(config)

# Run analyses
ap_results = ap_analyzer.analyze(data)
syn_results = syn_analyzer.analyze(data)
```

### 4. Batch Processing
```python
from core.base_analyzer import BatchAnalyzer

batch = BatchAnalyzer(analyzer, config)
results = batch.analyze_directory("data/recordings/")
batch.save_results("results/batch_analysis.h5")
```

### 5. Advanced Visualization
```python
from common.visualization import create_analysis_figure

fig = create_analysis_figure(results, style="publication")
fig.save("figure_1.pdf", dpi=300)
```

## Module Details

### Action Potential Analysis
- Spike detection with multiple algorithms
- Waveform analysis and classification
- Burst detection and characterization
- Adaptation and accommodation metrics

### Synaptic Analysis
- Event detection (mEPSCs, mIPSCs)
- Kinetic analysis (rise time, decay)
- Paired-pulse analysis
- Synaptic plasticity quantification

### Ion Channel Analysis
- I-V curve fitting
- Activation/inactivation kinetics
- State model fitting
- Conductance analysis

### Network Analysis
- Connectivity mapping
- Synchronization metrics
- Graph theoretical measures
- Dynamic network states

### Pharmacology
- Dose-response curves
- Time-series modulation
- Kinetic modeling
- Multi-drug interactions

### Time-Frequency Analysis
- Wavelet transforms
- Multitaper spectrograms
- Cross-frequency coupling
- Phase-amplitude relationships

## Data Formats Supported

- **ABF**: Axon Binary Format (pyABF)
- **NWB**: Neurodata Without Borders (PyNWB)
- **CSV**: Comma-separated values
- **HDF5**: Hierarchical Data Format
- **Neo**: Neo-compatible formats

## Configuration System

Hierarchical configuration with validation:
```yaml
preprocessing:
  detrend: true
  filter:
    type: butterworth
    low_freq: 1
    high_freq: 3000
    order: 4

analysis:
  action_potential:
    threshold: -20
    min_amplitude: 40
    refractory_period: 2
    
  synaptic:
    event_threshold: 3  # z-score
    min_event_interval: 10  # ms

visualization:
  style: publication
  color_scheme: colorblind_safe
```

## Error Handling

Comprehensive exception hierarchy:
```python
try:
    results = analyzer.analyze(data)
except DataValidationError as e:
    logger.error(f"Invalid data: {e}")
except AnalysisError as e:
    logger.error(f"Analysis failed: {e}")
```

## Performance Optimization

- Parallel processing support
- Efficient memory usage
- Caching for repeated operations
- Optimized algorithms

## Quality Assurance

- Extensive unit tests
- Integration tests
- Validation against published results
- Continuous integration

## Extending the Framework

### Creating Custom Analyzers
```python
from core.base_analyzer import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, data, **kwargs):
        # Implementation
        return AnalysisResult(...)
```

### Adding New Formats
```python
from core.data_loader import register_loader

@register_loader('.custom')
def load_custom_format(file_path):
    # Implementation
    return StandardizedData(...)
```

## Dependencies

```
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
pandas>=1.1.0
pyabf>=2.3.0
pynwb>=2.0.0
neo>=0.10.0
h5py>=3.0.0
pyyaml>=5.3.0
tqdm>=4.62.0
```

## Best Practices

1. Always validate data before analysis
2. Use configuration files for reproducibility
3. Log all operations for debugging
4. Save intermediate results
5. Document custom modifications

## Migration from v1

```python
# v1 style
spikes = detect_action_potentials(data, threshold=-20)

# v2 style
analyzer = ActionPotentialAnalyzer(config)
results = analyzer.analyze(data)
spikes = results.spike_times
```

## See Also
- [Main README](../../README.md) - Overview of v1 vs v2
- [v1 README](../v1/README.md) - Simple implementations
- [API Documentation](docs/api.md) - Detailed API reference
- [Examples](examples/) - Usage examples