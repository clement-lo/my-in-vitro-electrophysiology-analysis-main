# Migration Guide: From Previous to Classic V2

## Quick Start

### For Action Potential Analysis

**Previous way:**
```python
from action_potential_spike_train_analysis import load_data, analyze_action_potentials
data = load_data('recording.csv')
ap_properties = analyze_action_potentials(data)
```

**New V2 way:**
```python
from action_potential_analysis_merged import run_complete_analysis
results = run_complete_analysis('recording.csv', output_dir='./results')
# Access results
ap_properties = results['ap_properties']
spike_times = results['spike_times']
```

### For Synaptic Analysis

**Previous way:**
```python
from synaptic_input_output_analysis import analyze_synaptic_io
io_results = analyze_synaptic_io(data)
```

**New V2 way (coming soon):**
```python
from synaptic_analysis_merged import run_synaptic_analysis
results = run_synaptic_analysis('recording.abf', analysis_type='input_output')
```

## Key Differences

1. **Unified data loading** - No need to worry about file formats
2. **Comprehensive output** - All analyses return structured results
3. **Better visualization** - Multi-panel figures with all metrics
4. **Configuration support** - Use YAML configs or direct parameters

## API Mapping

| Previous Function | V2 Equivalent | Notes |
|------------------|---------------|-------|
| `load_data()` | `UnifiedDataLoader.load_file()` | Auto-detects format |
| `analyze_action_potentials()` | `analyze_action_potential_properties()` | Enhanced metrics |
| `analyze_spike_train()` | `analyze_spike_train_dynamics()` | More metrics |
| `plot_results()` | `plot_comprehensive_analysis()` | 8-panel figure |

