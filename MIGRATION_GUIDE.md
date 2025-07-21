# Migration Guide: From v1 to v2

This guide helps you transition from the educational v1 implementation to the research-grade v2 framework.

## Overview

The v2 framework provides:
- Object-oriented architecture
- Multi-format data support
- Comprehensive error handling
- Advanced analysis features
- Configuration management
- Batch processing capabilities

## Quick Migration Examples

### 1. Action Potential Analysis

**v1 (Simple function-based):**
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import action_potential_analysis

# Load CSV data
data = action_potential_analysis.load_data('recording.csv')

# Detect spikes
spikes = action_potential_analysis.detect_action_potentials(data, threshold=-20)

# Analyze properties
properties = action_potential_analysis.analyze_ap_properties(data, spikes)

# Plot results
action_potential_analysis.plot_results(data, spikes, properties)
```

**v2 (Framework-based):**
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.action_potential_analysis_merged import run_complete_analysis

# One-line comprehensive analysis (supports multiple formats)
results = run_complete_analysis('recording.abf', output_dir='./results')

# Or use individual components
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_loader import DataLoader
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.modules.action_potential import ActionPotentialAnalyzer

loader = DataLoader()
data = loader.load_file('recording.nwb')  # Auto-detects format

analyzer = ActionPotentialAnalyzer(config)
results = analyzer.analyze(data)
```

### 2. Synaptic Current Analysis

**v1:**
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import synaptic_current_analysis

data = synaptic_current_analysis.load_data('recording.csv')
events = synaptic_current_analysis.detect_synaptic_events(data, threshold=3)
kinetics = synaptic_current_analysis.analyze_kinetics(events)
```

**v2:**
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.synaptic_analysis_merged import run_synaptic_analysis

# Comprehensive analysis with validation
results = run_synaptic_analysis(
    data_file='recording.abf',
    analysis_type='comprehensive',
    config_file='synaptic_config.yaml'
)

# Access detailed results
events = results['events']
kinetics = results['kinetics']
statistics = results['statistics']
```

### 3. Configuration Management

**v1 (Direct parameters):**
```python
# Parameters scattered throughout function calls
spikes = detect_spikes(data, threshold=-20, refractory=2.0)
filtered = filter_data(data, low_freq=1, high_freq=3000)
```

**v2 (Centralized configuration):**
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.config_manager import ConfigManager

# Load from YAML
config = ConfigManager.from_file('analysis_config.yaml')

# Or create programmatically
config = ConfigManager()
config.analysis.spike_threshold = -20
config.analysis.refractory_period = 2.0
config.preprocessing.filter.low_freq = 1
config.preprocessing.filter.high_freq = 3000

# Use throughout analysis
analyzer = ActionPotentialAnalyzer(config)
```

## Key API Changes

### Data Loading

| v1 Function | v2 Method | Key Differences |
|-------------|-----------|-----------------|
| `load_data('file.csv')` | `DataLoader().load_file('file.abf')` | Multi-format support |
| `pd.read_csv()` | Automatic format detection | Handles ABF, NWB, HDF5 |
| Manual parsing | Standardized data structure | Consistent interface |

### Analysis Functions

| v1 Pattern | v2 Pattern | Benefits |
|------------|------------|----------|
| Function-based | Class-based analyzers | Stateful, extensible |
| Direct returns | Result objects | Rich metadata |
| Single analysis | Pipeline support | Batch processing |
| Fixed parameters | Configuration system | Reproducibility |

### Visualization

| v1 Approach | v2 Approach | Improvements |
|-------------|-------------|--------------|
| `plot_results()` | `create_publication_figure()` | Publication-ready |
| Basic matplotlib | Styled figures | Consistent aesthetics |
| Single plots | Multi-panel layouts | Comprehensive views |

## Migration Strategies

### 1. Gradual Migration
```python
# Wrap v1 functions with v2 data loading
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_loader import DataLoader
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import action_potential_analysis

# Use v2 loader with v1 analysis
loader = DataLoader()
data = loader.load_file('recording.abf')
data_array = data.as_array()  # Convert for v1 compatibility

# Continue with v1 analysis
spikes = action_potential_analysis.detect_action_potentials(data_array, threshold=-20)
```

### 2. Feature-by-Feature
Start with data loading, then migrate analysis modules one at a time:
1. Data loading (biggest immediate benefit)
2. Configuration management
3. Core analysis modules
4. Visualization
5. Batch processing

### 3. New Projects
For new analyses, start directly with v2 to leverage all features.

## Feature Comparison

| Feature | v1 | v2 |
|---------|----|----|
| **Data Formats** | CSV only | ABF, NWB, CSV, HDF5, Neo |
| **Architecture** | Functions | OOP Framework |
| **Configuration** | Function params | YAML/Programmatic |
| **Error Handling** | Basic | Comprehensive |
| **Validation** | Limited | Extensive |
| **Batch Processing** | Manual | Built-in |
| **Extensibility** | Limited | Designed for it |
| **Testing** | Basic | Comprehensive |

## Common Migration Issues

### 1. Import Paths
```python
# v1
from action_potential_analysis import detect_spikes

# v2
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.modules.action_potential import ActionPotentialAnalyzer
```

### 2. Data Format
```python
# v1 expects numpy arrays or pandas DataFrames
# v2 uses standardized data objects

# Convert v2 to v1 compatible:
data_array = v2_data.as_array()

# Convert v1 to v2 compatible:
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_structures import StandardizedData
v2_data = StandardizedData(v1_array, sampling_rate=20000)
```

### 3. Result Access
```python
# v1: Direct returns
spikes = detect_spikes(data)

# v2: Result objects
results = analyzer.analyze(data)
spikes = results.spike_times
metadata = results.metadata
```

## Getting Help

- See example scripts in `python/v2/examples/`
- Check module documentation in `python/v2/README.md`
- Review test cases for usage patterns

---
*Migration Guide - In Vitro Electrophysiology Analysis Framework*

