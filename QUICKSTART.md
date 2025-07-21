# Quick Start Guide

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/clement-lo/my-in-vitro-electrophysiology-analysis-main.git
cd my-in-vitro-electrophysiology-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Choose Your Implementation Level

### For Learning/Quick Analysis (v1)
```python
# Simple action potential analysis
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import action_potential_analysis

# Load and analyze
data = action_potential_analysis.load_data('test_data/action_potential_test.csv')
spikes = action_potential_analysis.detect_action_potentials(data, threshold=-20)
action_potential_analysis.plot_results(data, spikes)
```

### For Research/Production (v2)
```python
# Advanced analysis pipeline
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.action_potential_analysis_merged import run_complete_analysis

# Run comprehensive analysis
results = run_complete_analysis(
    'test_data/action_potential_test.nwb',
    output_dir='results/'
)
```

## 3. Available Analysis Scripts

### Patch-Clamp & Whole-Cell Analysis
**Path**: `scripts/patch_clamp_whole_cell_in_vitro/`

| Analysis | v1 Location | v2 Location |
|----------|-------------|-------------|
| Action Potentials | `python/v1/action_potential_analysis.py` | `python/v2/action_potential_analysis_merged.py` |
| Synaptic Currents | `python/v1/synaptic_current_analysis.py` | `python/v2/synaptic_analysis_merged.py` |
| Ion Channels | `python/v1/ion_channels_kinetics.py` | `python/v2/modules/ion_channels.py` |
| Network Analysis | `python/v1/network_connectivity_plasticity_analysis.py` | `python/v2/modules/network_analysis.py` |
| Pharmacology | `python/v1/pharmacological_modulation.py` | `python/v2/modules/pharmacology.py` |
| Time-Frequency | `python/v1/time_frequency_analysis.py` | `python/v2/modules/time_frequency.py` |

### MEA Analysis
**Path**: `scripts/mea_in_vitro/python/`
- `advanced_spike_sorting_clustering.py`
- `connectivity_analysis.py`
- `network_dynamics_analysis.py`
- `pharmacological_modulation_analysis.py`

## 4. Data Formats Supported

| Format | Extension | v1 Support | v2 Support |
|--------|-----------|------------|------------|
| CSV | .csv | ✅ | ✅ |
| ABF | .abf | ❌ | ✅ |
| NWB | .nwb | ❌ | ✅ |
| HDF5 | .h5/.hdf5 | ❌ | ✅ |
| Neo | various | ❌ | ✅ |

## 5. Quick Examples

### Detect Synaptic Events (v1)
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import synaptic_current_analysis

# Simple event detection
data = synaptic_current_analysis.load_data('recording.csv')
events = synaptic_current_analysis.detect_synaptic_events(data, threshold=3)
synaptic_current_analysis.plot_events(data, events)
```

### Multi-Format Loading (v2)
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_loader import DataLoader

loader = DataLoader()
# Automatically detects format
data = loader.load_file('recording.abf')  # or .nwb, .csv, .h5
```

### Run MEA Analysis
```python
from scripts.mea_in_vitro.python import network_dynamics_analysis

# Analyze network dynamics
results = network_dynamics_analysis.analyze_network_dynamics('mea_recording.h5')
network_dynamics_analysis.plot_network_metrics(results)
```

## 6. Configuration Files

### Using YAML Configs (v1)
```yaml
# Example: action_potential_analysis_config.yaml
analysis:
  threshold: -20
  min_amplitude: 40
  refractory_period: 2
```

### Programmatic Config (v2)
```python
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.config_manager import ConfigManager

config = ConfigManager()
config.analysis.spike_threshold = -20
config.preprocessing.filter.low_freq = 1
```

## 7. Jupyter Notebooks

Interactive tutorials available in:
```
scripts/patch_clamp_whole_cell_in_vitro/notebooks/v1/
├── 01_Synaptic_Input_Output_Analysis.ipynb
├── 02_Action_Potential_Spike_Train_Analysis.ipynb
├── 03_Ion_Channel_Kinetics.ipynb
├── 04_Pharmacological_Modulation.ipynb
├── 05_Network_Connectivity_Plasticity_Analysis.ipynb
└── 06_Time_Frequency_Analysis.ipynb
```

## 8. Validation Tools

```bash
# Validate installation
python scripts/utils/validate_modules.py

# Generate test data
python scripts/test_data_generator.py

# Run MATLAB validation
matlab -batch "run('scripts/utils/validate_modules_matlab.m')"
```

## Need Help?

- See [README.md](README.md) for comprehensive documentation
- Check module-specific READMEs in each directory
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---
*Quick Start Guide for In Vitro Electrophysiology Analysis Framework*
