# Test Data Metadata

## Files Generated

1. **NWB Format** (Recommended)
   - action_potential_test.nwb - Action potential recording
   - synaptic_current_test.nwb - Synaptic current recording

2. **CSV Format** (Backward compatibility)
   - action_potential_test.csv - Time, Voltage columns
   - synaptic_current_test.csv - Time, Current columns

3. **Binary Format**
   - action_potential_test.dat - Simple binary format

## Data Properties

- Sampling rate: 20,000 Hz
- Duration: 10 seconds
- Action potential data: ~50 spikes, current clamp
- Synaptic data: ~30 EPSCs, voltage clamp

## Usage

### Python with NWB
```python
from pynwb import NWBHDF5IO

with NWBHDF5IO('action_potential_test.nwb', 'r') as io:
    nwbfile = io.read()
    cc_series = nwbfile.acquisition['CurrentClampSeries']
    voltage = cc_series.data[:]
    rate = cc_series.rate
```

### MATLAB with NWB
```matlab
nwbfile = nwbRead('action_potential_test.nwb');
cc_series = nwbfile.acquisition.get('CurrentClampSeries');
voltage = cc_series.data.load();
rate = cc_series.starting_time_rate;
```

### Python with CSV
```python
import pandas as pd

data = pd.read_csv('action_potential_test.csv')
time = data['Time'].values
voltage = data['Voltage'].values
```
