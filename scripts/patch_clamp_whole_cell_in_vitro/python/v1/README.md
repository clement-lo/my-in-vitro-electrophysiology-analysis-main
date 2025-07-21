# v1 - Simple Electrophysiology Analysis

Simple, educational implementations for patch-clamp and whole-cell electrophysiology analysis.

## Overview

This directory contains straightforward Python implementations designed for:
- Educational purposes
- Quick prototyping
- Understanding core concepts
- Simple data analysis

## Features

- **Minimal Dependencies**: NumPy, SciPy, Matplotlib
- **Clear Code**: Function-based, well-commented
- **CSV Support**: Easy data import/export
- **Direct Analysis**: No complex abstractions

## Modules

### 1. Action Potential Analysis
```python
from action_potential_analysis import detect_action_potentials, analyze_spike_shape
from action_potential_spike_train_analysis import analyze_spike_trains

# Detect spikes
spikes = detect_action_potentials(voltage_data, threshold=-20)

# Analyze spike trains
isi_stats = analyze_spike_trains(spike_times)
```

### 2. Synaptic Analysis
```python
from synaptic_input_output_analysis import analyze_io_relationship
from synaptic_current_analysis import detect_synaptic_events

# Analyze input-output curves
io_curve = analyze_io_relationship(stimulus_intensities, response_amplitudes)

# Detect synaptic events
events = detect_synaptic_events(current_data, threshold=10)
```

### 3. Ion Channel Kinetics
```python
from ion_channels_kinetics import fit_activation_curve, calculate_reversal_potential

# Fit activation curves
params = fit_activation_curve(voltages, currents)

# Calculate reversal potential
e_rev = calculate_reversal_potential(ion_concentrations)
```

### 4. Network Analysis
```python
from network_connectivity_plasticity_analysis import calculate_connectivity, analyze_plasticity

# Calculate connectivity
connectivity_matrix = calculate_connectivity(spike_trains)

# Analyze plasticity
plasticity_index = analyze_plasticity(pre_responses, post_responses)
```

### 5. Pharmacological Modulation
```python
from pharmacological_modulation import analyze_dose_response, calculate_ic50

# Dose-response analysis
dr_params = analyze_dose_response(concentrations, responses)

# Calculate IC50
ic50 = calculate_ic50(dr_params)
```

### 6. Time-Frequency Analysis
```python
from time_frequency_analysis import compute_spectrogram, calculate_coherence

# Compute spectrogram
freqs, times, power = compute_spectrogram(signal, fs=1000)

# Calculate coherence
coherence = calculate_coherence(signal1, signal2, fs=1000)
```

## Data Format

All modules expect simple formats:
- **Time series**: 1D NumPy arrays
- **Sampling rate**: Specified in Hz
- **Units**: mV for voltage, pA for current

Example CSV format:
```
Time(s),Voltage(mV),Current(pA)
0.000,−65.0,0.0
0.001,−64.8,0.5
...
```

## Configuration

Simple YAML configuration files available for some modules:
```yaml
# action_potential_analysis_config.yaml
detection:
  threshold: -20  # mV
  min_amplitude: 40  # mV
  
analysis:
  pre_spike_ms: 2
  post_spike_ms: 5
```

## Quick Examples

### Complete Analysis Pipeline
```python
import numpy as np
import pandas as pd
from action_potential_analysis import detect_and_analyze_aps
from synaptic_current_analysis import analyze_synaptic_currents

# Load data
data = pd.read_csv('recording.csv')
voltage = data['Voltage(mV)'].values
current = data['Current(pA)'].values
time = data['Time(s)'].values
fs = 1000  # Hz

# Analyze action potentials
ap_results = detect_and_analyze_aps(voltage, fs)
print(f"Found {len(ap_results['spike_times'])} spikes")

# Analyze synaptic currents
syn_results = analyze_synaptic_currents(current, fs)
print(f"Found {len(syn_results['event_times'])} synaptic events")
```

## Limitations

- Basic error handling
- Limited to CSV data format
- No automatic unit conversions
- Single-channel analysis only
- No parallel processing

## When to Upgrade to v2

Consider using v2 when you need:
- Multiple data format support
- Batch processing
- Advanced error handling
- Extensible architecture
- Publication-quality analysis

## Dependencies

```
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
pandas>=1.1.0
pyyaml>=5.3.0
```

## See Also
- [Main README](../../README.md) - Overview of v1 vs v2
- [v2 README](../v2/README.md) - Advanced framework documentation
- [Notebooks](../../notebooks/v1/) - Interactive examples