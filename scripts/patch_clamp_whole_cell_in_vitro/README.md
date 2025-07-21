# Patch-Clamp and Whole-Cell In Vitro Electrophysiology Analysis

This directory contains two complementary implementations for analyzing patch-clamp and whole-cell electrophysiology data:

## Directory Structure

```
patch_clamp_whole_cell_in_vitro/
├── python/
│   ├── v1/        # Simple, educational implementations
│   └── v2/        # Research-grade, extensible framework
├── matlab/
│   ├── v1/        # Original MATLAB implementations
│   └── v2/        # Enhanced MATLAB framework
└── notebooks/
    └── v1/        # Jupyter notebooks for interactive analysis
```

## Version Comparison

### v1 - Educational & Prototyping
- **Purpose**: Learning, teaching, quick analyses, prototyping
- **Target Users**: Students, educators, researchers doing simple analyses
- **Characteristics**:
  - Simple, direct implementations
  - Minimal dependencies
  - CSV-focused data handling
  - Function-based architecture
  - Clear, readable code
  - Limited error handling

### v2 - Research-Grade Framework
- **Purpose**: Production research, complex analyses, extensible pipelines
- **Target Users**: Research labs, advanced users, pipeline development
- **Characteristics**:
  - Object-oriented architecture
  - Multi-format data support (ABF, NWB, HDF5, Neo)
  - Comprehensive error handling
  - Extensive validation
  - Configurable and extensible
  - Publication-ready outputs

## Quick Start Guide

### For Educational Use (v1)
```python
# Simple spike analysis
from python.v1.action_potential_analysis import detect_action_potentials
spikes = detect_action_potentials(data, threshold=-20)
```

### For Research Use (v2)
```python
# Comprehensive analysis pipeline
from python.v2.core.data_loader import DataLoader
from python.v2.modules.action_potential import ActionPotentialAnalyzer

loader = DataLoader()
data = loader.load_file("recording.abf")
analyzer = ActionPotentialAnalyzer(config)
results = analyzer.analyze(data)
```

## Module Overview

| Module | v1 (Simple) | v2 (Advanced) |
|--------|-------------|---------------|
| Action Potentials | Basic detection & analysis | Full waveform analysis, classification |
| Synaptic Analysis | I/O relationships | Event detection, kinetics, plasticity |
| Ion Channels | Simple kinetics | State models, conductance analysis |
| Network Analysis | Basic connectivity | Graph theory, dynamics, synchronization |
| Pharmacology | Dose-response | Time-series modulation, kinetic modeling |
| Time-Frequency | Basic spectrograms | Wavelet, multitaper, cross-frequency coupling |

## Choosing Between v1 and v2

**Use v1 when:**
- Learning electrophysiology analysis
- Teaching or demonstrating concepts
- Doing quick, simple analyses
- Working primarily with CSV data
- Needing minimal setup

**Use v2 when:**
- Building analysis pipelines
- Working with multiple data formats
- Needing extensive validation
- Publishing research results
- Requiring extensibility

## Data Compatibility

- **v1**: Primarily CSV files, simple format
- **v2**: ABF, NWB, CSV, HDF5, Neo formats

## Configuration

- **v1**: Direct function parameters or simple YAML configs
- **v2**: Comprehensive configuration system with validation

## See Also
- [v1 Python README](python/v1/README.md) - Detailed v1 documentation
- [v2 Python README](python/v2/README.md) - Detailed v2 documentation
- [Notebooks README](notebooks/v1/README.md) - Interactive examples