# Electrophysiology Test Data and Validation Suite

## Overview

Created a comprehensive test data suite in **NeurodataWithoutBorders (NWB)** format that can be used by both Python and MATLAB scripts to validate the electrophysiology analysis modules.

## Why NWB Format?

1. **Cross-platform compatibility**: Excellent support for both Python (PyNWB) and MATLAB (MatNWB)
2. **Standardized format**: Becoming the standard for neurophysiology data sharing
3. **Rich metadata**: Stores experimental metadata alongside data
4. **HDF5 backend**: Efficient storage and widely supported
5. **Extensible**: Can add custom data types as needed

## Generated Test Data

### 1. Action Potential Data
- **File**: `test_data/action_potential_test.nwb` (3.4 MB)
- **Content**: Current clamp recording with ~50 synthetic action potentials
- **Duration**: 10 seconds at 20 kHz sampling rate
- **Includes**: Stimulus current injection from 2-8 seconds

### 2. Synaptic Current Data  
- **File**: `test_data/synaptic_current_test.nwb` (1.8 MB)
- **Content**: Voltage clamp recording with ~30 EPSCs
- **Duration**: 10 seconds at 20 kHz sampling rate
- **Features**: Realistic synaptic event kinetics

### 3. Compatibility Formats
- **CSV files**: For backward compatibility with existing scripts
- **Binary files**: Simple format for performance testing

## Validation Scripts

### Python Validation (`validate_modules.py`)
Tests all v2 merged modules:
- ✅ Action potential detection and analysis
- ✅ Synaptic event detection
- ✅ Input-output curve fitting
- ✅ Unified data loading
- ✅ Complete analysis workflow

### MATLAB Validation (`validate_modules_matlab.m`)
Tests MATLAB implementations:
- ✅ CSV data compatibility
- ✅ NWB file reading (requires MatNWB)
- ✅ Action potential detection
- ✅ Synaptic analysis
- ✅ Curve fitting

## Usage Examples

### Python with NWB
```python
from pynwb import NWBHDF5IO
from action_potential_analysis_merged import run_complete_analysis

# Direct analysis
results = run_complete_analysis('test_data/action_potential_test.nwb')

# Manual loading
with NWBHDF5IO('test_data/action_potential_test.nwb', 'r') as io:
    nwbfile = io.read()
    cc_series = nwbfile.acquisition['CurrentClampSeries']
    voltage = cc_series.data[:]
    rate = cc_series.rate
```

### MATLAB with NWB
```matlab
% Requires MatNWB
nwbfile = nwbRead('test_data/action_potential_test.nwb');
cc_series = nwbfile.acquisition.get('CurrentClampSeries');
voltage = cc_series.data.load();
rate = cc_series.starting_time_rate;
```

### Python with CSV (backward compatibility)
```python
import pandas as pd

data = pd.read_csv('test_data/action_potential_test.csv')
voltage = data['Voltage'].values
time = data['Time'].values
```

## Data Properties

| Property | Action Potential | Synaptic Current |
|----------|-----------------|------------------|
| Duration | 10 seconds | 10 seconds |
| Sampling Rate | 20 kHz | 20 kHz |
| Data Type | Current Clamp | Voltage Clamp |
| Events | ~50 APs | ~30 EPSCs |
| Baseline | -65 mV | -5 pA |
| Noise Level | 0.5 mV RMS | 2 pA RMS |

## Next Steps

1. **Install MatNWB** for MATLAB support:
   ```bash
   git clone https://github.com/NeurodataWithoutBorders/matnwb.git
   ```

2. **Run Python validation**:
   ```bash
   cd /path/to/repository
   python validate_modules.py
   ```

3. **Run MATLAB validation**:
   ```matlab
   cd /path/to/repository
   validate_modules_matlab
   ```

4. **Extend test data** as needed for:
   - Paired-pulse protocols
   - Pharmacological dose-response
   - Network connectivity
   - Time-frequency analysis

## Benefits

- ✅ **Standardized testing**: Same data for Python and MATLAB
- ✅ **Reproducible results**: Synthetic data with known properties
- ✅ **Format flexibility**: NWB, CSV, and binary formats
- ✅ **Comprehensive coverage**: Tests all major analysis functions
- ✅ **Easy extension**: Add new test cases as needed

The test suite ensures that both Python and MATLAB implementations produce consistent results and maintain compatibility across versions.
