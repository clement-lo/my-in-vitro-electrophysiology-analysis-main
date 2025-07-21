# v1 Notebooks - Interactive Electrophysiology Analysis

Interactive Jupyter notebooks for learning and exploring electrophysiology analysis.

## Overview

These notebooks provide hands-on tutorials for analyzing patch-clamp and whole-cell electrophysiology data. Each notebook focuses on a specific analysis type with explanations, visualizations, and exercises.

## Notebooks

1. **01_Synaptic_Input_Output_Analysis.ipynb**
   - Input-output relationship analysis
   - Curve fitting (sigmoid, linear)
   - Synaptic strength quantification

2. **02_Action_Potential_Spike_Train_Analysis.ipynb**
   - Spike detection and characterization
   - Inter-spike interval analysis
   - Firing rate calculations

3. **03_Ion_Channel_Kinetics.ipynb**
   - I-V curve analysis
   - Activation/inactivation curves
   - Time constant calculations

4. **04_Pharmacological_Modulation.ipynb**
   - Dose-response curves
   - Time-series drug effects
   - IC50/EC50 calculations

5. **05_Network_Connectivity_Plasticity_Analysis.ipynb**
   - Connectivity matrices
   - Synaptic plasticity protocols
   - Network metrics

6. **06_Time_Frequency_Analysis.ipynb**
   - Spectrograms
   - Power spectral density
   - Coherence analysis

## Setup

### Required Dependencies
```bash
pip install numpy scipy matplotlib pandas jupyter
```

### Data Access
Notebooks are configured to use test data from the project:
```python
# Example path in notebooks
file_path = '../../../test_data/synaptic_current_test.csv'
```

### Module Imports
Notebooks can import v1 Python modules:
```python
import sys
sys.path.append('../../python/v1')
from synaptic_input_output_analysis import analyze_io_relationship
```

## Usage Instructions

1. **Start Jupyter**:
   ```bash
   cd scripts/patch_clamp_whole_cell_in_vitro/notebooks/v1
   jupyter notebook
   ```

2. **Run Notebooks**:
   - Open desired notebook
   - Run cells sequentially (Shift+Enter)
   - Modify parameters to explore

3. **Use Your Data**:
   - Update file paths to point to your data
   - Ensure data format matches expected structure
   - See each notebook for format requirements

## Learning Path

### Beginner
1. Start with 01_Synaptic_Input_Output_Analysis
2. Move to 02_Action_Potential_Spike_Train_Analysis
3. Try modifying parameters and observing changes

### Intermediate
1. Work through 03_Ion_Channel_Kinetics
2. Explore 04_Pharmacological_Modulation
3. Combine techniques from multiple notebooks

### Advanced
1. Tackle 05_Network_Connectivity_Plasticity_Analysis
2. Master 06_Time_Frequency_Analysis
3. Create custom analysis pipelines

## Tips

- Each notebook is self-contained with explanations
- Experiment with different parameters
- Use the markdown cells to take notes
- Save modified notebooks with new names

## Troubleshooting

### Import Errors
```python
# If modules can't be found, check the path:
import os
print(os.getcwd())  # Current directory
```

### Data Loading Issues
- Verify file paths are correct
- Check data format matches expectations
- Use absolute paths if needed

### Memory Issues
- Reduce data size for initial exploration
- Clear outputs of cells not in use
- Restart kernel if needed

## Next Steps

After mastering these notebooks:
1. Try the v2 framework for advanced analysis
2. Create your own analysis notebooks
3. Contribute improvements back to the project

## See Also
- [v1 Python Modules](../../python/v1/README.md)
- [v2 Advanced Framework](../../python/v2/README.md)
- [Main Documentation](../../README.md)