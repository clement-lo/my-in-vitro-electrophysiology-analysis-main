# Using NWB Test Data with Your Analysis Scripts

## Python Example

```python
from pynwb import NWBHDF5IO
from action_potential_analysis_merged import run_complete_analysis

# Load NWB file
nwb_file = './test_data/action_potential_test.nwb'

# Option 1: Direct analysis with merged module
results = run_complete_analysis(nwb_file, output_dir='./results')

# Option 2: Manual loading and analysis
with NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile = io.read()
    
    # Get current clamp data
    cc_series = nwbfile.acquisition['CurrentClampSeries']
    voltage = cc_series.data[:]
    sampling_rate = cc_series.rate
    
    # Now use with your analysis functions
    # ...
```

## MATLAB Example

```matlab
% Ensure MatNWB is in your path
% addpath('/path/to/matnwb');

% Load NWB file
nwbfile = nwbRead('./test_data/action_potential_test.nwb');

% Get current clamp data
cc_series = nwbfile.acquisition.get('CurrentClampSeries');
voltage = cc_series.data.load();
sampling_rate = cc_series.starting_time_rate;

% Use with your MATLAB analysis scripts
% ...
```

## Available Test Datasets

1. **action_potential_test.nwb**
   - Current clamp recording with ~50 action potentials
   - 10 seconds duration at 20 kHz
   - Includes stimulus current injection

2. **synaptic_current_test.nwb**
   - Voltage clamp recording with EPSCs and IPSCs
   - Mixed spontaneous synaptic events
   - Realistic kinetics and amplitudes

3. **paired_pulse_test.nwb**
   - Multiple voltage clamp series for different intervals
   - Tests paired-pulse facilitation/depression
   - Intervals: 20, 50, 100, 200 ms

4. **input_output_test.nwb**
   - Multiple trials at different stimulus intensities
   - Includes summary table with mean responses
   - Tests I/O curve fitting functionality
