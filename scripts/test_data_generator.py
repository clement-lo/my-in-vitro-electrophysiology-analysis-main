"""
NWB Test Dataset Generator for Electrophysiology Analysis Validation
Creates comprehensive test datasets that can be used by both Python and MATLAB
"""

import numpy as np
from datetime import datetime
from dateutil import tz
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.icephys import (VoltageClampSeries, CurrentClampSeries, 
                           VoltageClampStimulusSeries, CurrentClampStimulusSeries,
                           IZeroClampSeries)
from pynwb.behavior import SpatialSeries, Position
from pynwb.file import Subject
import os

class ElectrophysiologyTestDataGenerator:
    """Generate comprehensive test datasets for all electrophysiology analysis modules."""
    
    def __init__(self, output_dir='./test_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sampling_rate = 20000.0  # 20 kHz
        
    def create_base_nwb_file(self, identifier, description):
        """Create a base NWB file with common metadata."""
        nwbfile = NWBFile(
            session_description=description,
            identifier=identifier,
            session_start_time=datetime.now(tz.tzlocal()),
            experimenter='Test Generator',
            lab='Computational Neuroscience Lab',
            institution='Test Institution',
            experiment_description='Synthetic data for testing analysis pipelines',
            session_id='test_session_001'
        )
        
        # Add subject information
        nwbfile.subject = Subject(
            subject_id='test_subject_001',
            age='P90D',  # 90 days old
            description='Synthetic test subject',
            species='Mus musculus',
            sex='M'
        )
        
        return nwbfile
    
    def generate_action_potential_data(self, duration=10.0):
        """Generate synthetic action potential data for testing AP analysis."""
        print("Generating action potential data...")
        
        nwbfile = self.create_base_nwb_file(
            'action_potential_test',
            'Synthetic action potential data for testing'
        )
        
        # Create time vector
        time = np.arange(0, duration, 1/self.sampling_rate)
        n_samples = len(time)
        
        # Generate baseline with noise
        baseline = -65.0  # mV
        noise_std = 0.5
        voltage = baseline + np.random.normal(0, noise_std, n_samples)
        
        # Add action potentials
        spike_times = np.sort(np.random.uniform(0.5, duration-0.5, 50))  # 50 spikes
        
        for spike_time in spike_times:
            spike_idx = int(spike_time * self.sampling_rate)
            
            # Create realistic AP waveform
            ap_duration = int(0.003 * self.sampling_rate)  # 3ms AP
            if spike_idx + ap_duration < n_samples:
                t_ap = np.arange(ap_duration) / self.sampling_rate
                
                # AP shape: fast depolarization, slower repolarization
                depol_phase = 55 * np.exp(-t_ap/0.0002) * (t_ap < 0.0005)
                repol_phase = -20 * np.exp(-(t_ap-0.0005)/0.001) * (t_ap >= 0.0005)
                ap_waveform = depol_phase + repol_phase
                
                voltage[spike_idx:spike_idx+ap_duration] += ap_waveform
        
        # Add the recording as CurrentClampSeries
        electrode = nwbfile.create_icephys_electrode(
            name='electrode_0',
            description='Synthetic patch electrode',
            device=nwbfile.create_device(name='Amplifier_1')
        )
        
        cc_series = CurrentClampSeries(
            name='CurrentClampSeries',
            data=voltage,
            starting_time=0.0,
            rate=self.sampling_rate,
            electrode=electrode,
            gain=1.0,
            bias_current=0.0,
            bridge_balance=0.0,
            capacitance_compensation=0.0
        )
        
        nwbfile.add_acquisition(cc_series)
        
        # Add stimulus (current injection)
        stimulus = np.zeros(n_samples)
        stim_start = int(2.0 * self.sampling_rate)
        stim_end = int(8.0 * self.sampling_rate)
        stimulus[stim_start:stim_end] = 100.0  # 100 pA current injection
        
        stim_series = CurrentClampStimulusSeries(
            name='CurrentClampStimulusSeries',
            data=stimulus,
            starting_time=0.0,
            rate=self.sampling_rate,
            electrode=electrode,
            gain=1.0
        )
        
        nwbfile.add_stimulus(stim_series)
        
        # Save file
        filename = os.path.join(self.output_dir, 'action_potential_test.nwb')
        with NWBHDF5IO(filename, 'w') as io:
            io.write(nwbfile)
            
        print(f"✓ Saved action potential data to {filename}")
        return filename
    
    def generate_synaptic_current_data(self, duration=10.0):
        """Generate synthetic synaptic current data (EPSCs/IPSCs)."""
        print("Generating synaptic current data...")
        
        nwbfile = self.create_base_nwb_file(
            'synaptic_current_test',
            'Synthetic synaptic current data for testing'
        )
        
        time = np.arange(0, duration, 1/self.sampling_rate)
        n_samples = len(time)
        
        # Baseline current with noise
        baseline = -5.0  # pA
        noise_std = 2.0
        current = baseline + np.random.normal(0, noise_std, n_samples)
        
        # Add EPSCs (downward deflections)
        n_epscs = 30
        epsc_times = np.sort(np.random.uniform(0.5, duration-0.5, n_epscs))
        
        for epsc_time in epsc_times:
            epsc_idx = int(epsc_time * self.sampling_rate)
            
            # EPSC kinetics: fast rise, exponential decay
            epsc_duration = int(0.05 * self.sampling_rate)  # 50ms
            if epsc_idx + epsc_duration < n_samples:
                t_epsc = np.arange(epsc_duration) / self.sampling_rate
                amplitude = np.random.uniform(20, 100)  # pA
                
                # Double exponential for realistic EPSC shape
                rise_tau = 0.001  # 1ms rise
                decay_tau = 0.01  # 10ms decay
                epsc_waveform = amplitude * (np.exp(-t_epsc/decay_tau) - np.exp(-t_epsc/rise_tau))
                epsc_waveform *= (rise_tau * decay_tau) / (decay_tau - rise_tau)
                
                current[epsc_idx:epsc_idx+epsc_duration] -= epsc_waveform
        
        # Add some IPSCs (upward deflections)
        n_ipscs = 15
        ipsc_times = np.sort(np.random.uniform(0.5, duration-0.5, n_ipscs))
        
        for ipsc_time in ipsc_times:
            ipsc_idx = int(ipsc_time * self.sampling_rate)
            
            # IPSC kinetics: slower than EPSCs
            ipsc_duration = int(0.1 * self.sampling_rate)  # 100ms
            if ipsc_idx + ipsc_duration < n_samples:
                t_ipsc = np.arange(ipsc_duration) / self.sampling_rate
                amplitude = np.random.uniform(10, 50)  # pA
                
                rise_tau = 0.002  # 2ms rise
                decay_tau = 0.02  # 20ms decay
                ipsc_waveform = amplitude * (np.exp(-t_ipsc/decay_tau) - np.exp(-t_ipsc/rise_tau))
                ipsc_waveform *= (rise_tau * decay_tau) / (decay_tau - rise_tau)
                
                current[ipsc_idx:ipsc_idx+ipsc_duration] += ipsc_waveform
        
        # Create voltage clamp recording
        electrode = nwbfile.create_icephys_electrode(
            name='electrode_0',
            description='Synthetic patch electrode for voltage clamp',
            device=nwbfile.create_device(name='Amplifier_1')
        )
        
        vc_series = VoltageClampSeries(
            name='VoltageClampSeries',
            data=current,
            starting_time=0.0,
            rate=self.sampling_rate,
            electrode=electrode,
            gain=1.0,
            capacitance_slow=0.0,
            capacitance_fast=0.0,
            resistance_comp_correction=0.0,
            resistance_comp_bandwidth=0.0,
            resistance_comp_prediction=0.0,
            whole_cell_series_resistance_comp=0.0,
            whole_cell_capacitance_comp=0.0
        )
        
        nwbfile.add_acquisition(vc_series)
        
        # Add voltage command
        voltage_command = np.full(n_samples, -70.0)  # Hold at -70mV
        
        stim_series = VoltageClampStimulusSeries(
            name='VoltageClampStimulusSeries',
            data=voltage_command,
            starting_time=0.0,
            rate=self.sampling_rate,
            electrode=electrode,
            gain=1.0
        )
        
        nwbfile.add_stimulus(stim_series)
        
        # Save file
        filename = os.path.join(self.output_dir, 'synaptic_current_test.nwb')
        with NWBHDF5IO(filename, 'w') as io:
            io.write(nwbfile)
            
        print(f"✓ Saved synaptic current data to {filename}")
        return filename
    
    def generate_paired_pulse_data(self, n_trials=10, intervals_ms=[20, 50, 100, 200]):
        """Generate paired-pulse facilitation/depression data."""
        print("Generating paired-pulse data...")
        
        nwbfile = self.create_base_nwb_file(
            'paired_pulse_test',
            'Synthetic paired-pulse data for testing'
        )
        
        electrode = nwbfile.create_icephys_electrode(
            name='electrode_0',
            description='Synthetic patch electrode',
            device=nwbfile.create_device(name='Amplifier_1')
        )
        
        # Generate data for each interval
        for interval_ms in intervals_ms:
            interval_s = interval_ms / 1000.0
            
            # Each trial is 1 second long
            trial_duration = 1.0
            time = np.arange(0, trial_duration, 1/self.sampling_rate)
            n_samples = len(time)
            
            all_trials_current = []
            
            for trial in range(n_trials):
                # Baseline current with noise
                current = -5.0 + np.random.normal(0, 2.0, n_samples)
                
                # First pulse at 0.2s
                pulse1_time = 0.2
                pulse1_idx = int(pulse1_time * self.sampling_rate)
                
                # Second pulse after interval
                pulse2_time = pulse1_time + interval_s
                pulse2_idx = int(pulse2_time * self.sampling_rate)
                
                # Generate EPSC responses
                epsc_duration = int(0.05 * self.sampling_rate)
                t_epsc = np.arange(epsc_duration) / self.sampling_rate
                
                # First EPSC
                amplitude1 = 50 + np.random.normal(0, 5)  # pA
                rise_tau = 0.001
                decay_tau = 0.01
                epsc1 = amplitude1 * (np.exp(-t_epsc/decay_tau) - np.exp(-t_epsc/rise_tau))
                
                # Second EPSC (with facilitation or depression)
                ppr = 1.2 if interval_ms < 50 else 0.8  # Facilitation for short intervals
                amplitude2 = amplitude1 * ppr + np.random.normal(0, 5)
                epsc2 = amplitude2 * (np.exp(-t_epsc/decay_tau) - np.exp(-t_epsc/rise_tau))
                
                # Add EPSCs to current trace
                current[pulse1_idx:pulse1_idx+epsc_duration] -= epsc1
                if pulse2_idx + epsc_duration < n_samples:
                    current[pulse2_idx:pulse2_idx+epsc_duration] -= epsc2
                
                all_trials_current.append(current)
            
            # Average across trials
            mean_current = np.mean(all_trials_current, axis=0)
            
            # Create voltage clamp series for this interval
            vc_series = VoltageClampSeries(
                name=f'VoltageClampSeries_{interval_ms}ms',
                data=mean_current,
                starting_time=0.0,
                rate=self.sampling_rate,
                electrode=electrode,
                gain=1.0,
                capacitance_slow=0.0,
                capacitance_fast=0.0,
                resistance_comp_correction=0.0,
                resistance_comp_bandwidth=0.0,
                resistance_comp_prediction=0.0,
                whole_cell_series_resistance_comp=0.0,
                whole_cell_capacitance_comp=0.0
            )
            
            nwbfile.add_acquisition(vc_series)
        
        # Save file
        filename = os.path.join(self.output_dir, 'paired_pulse_test.nwb')
        with NWBHDF5IO(filename, 'w') as io:
            io.write(nwbfile)
            
        print(f"✓ Saved paired-pulse data to {filename}")
        return filename
    
    def generate_input_output_data(self):
        """Generate input-output curve data."""
        print("Generating input-output curve data...")
        
        nwbfile = self.create_base_nwb_file(
            'input_output_test',
            'Synthetic input-output curve data'
        )
        
        # Define stimulus intensities
        stim_intensities = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        n_intensities = len(stim_intensities)
        n_trials = 5
        
        electrode = nwbfile.create_icephys_electrode(
            name='electrode_0',
            description='Synthetic patch electrode',
            device=nwbfile.create_device(name='Amplifier_1')
        )
        
        # Generate responses following sigmoidal relationship
        # Response = a + (b-a) / (1 + exp((c-x)/d))
        a = 5  # min response
        b = 80  # max response  
        c = 45  # half-max stimulus
        d = 10  # slope factor
        
        # Store summary data
        mean_responses = []
        sem_responses = []
        
        for i, stim in enumerate(stim_intensities):
            trial_responses = []
            
            # Generate multiple trials for each intensity
            for trial in range(n_trials):
                # Calculate expected response
                expected_response = a + (b - a) / (1 + np.exp((c - stim) / d))
                
                # Add biological variability
                actual_response = expected_response + np.random.normal(0, 5)
                actual_response = max(0, actual_response)  # Can't be negative
                
                trial_responses.append(actual_response)
                
                # Create a 1-second recording for this trial
                time = np.arange(0, 1.0, 1/self.sampling_rate)
                n_samples = len(time)
                
                # Baseline current
                current = -5.0 + np.random.normal(0, 2.0, n_samples)
                
                # Add evoked response at 0.1s
                response_idx = int(0.1 * self.sampling_rate)
                response_duration = int(0.05 * self.sampling_rate)
                
                if response_idx + response_duration < n_samples:
                    t_response = np.arange(response_duration) / self.sampling_rate
                    rise_tau = 0.002
                    decay_tau = 0.015
                    response_waveform = actual_response * (np.exp(-t_response/decay_tau) - 
                                                          np.exp(-t_response/rise_tau))
                    current[response_idx:response_idx+response_duration] -= response_waveform
                
                # Create voltage clamp series
                vc_series = VoltageClampSeries(
                    name=f'VoltageClampSeries_stim{stim}_trial{trial}',
                    data=current,
                    starting_time=0.0,
                    rate=self.sampling_rate,
                    electrode=electrode,
                    gain=1.0,
                    capacitance_slow=0.0,
                    capacitance_fast=0.0,
                    resistance_comp_correction=0.0,
                    resistance_comp_bandwidth=0.0,
                    resistance_comp_prediction=0.0,
                    whole_cell_series_resistance_comp=0.0,
                    whole_cell_capacitance_comp=0.0
                )
                
                nwbfile.add_acquisition(vc_series)
            
            mean_responses.append(np.mean(trial_responses))
            sem_responses.append(np.std(trial_responses) / np.sqrt(n_trials))
        
        # Add analysis results as a table
        from pynwb.core import DynamicTable, VectorData
        
        io_table = DynamicTable(
            name='input_output_curve',
            description='Summary of input-output relationship'
        )
        
        io_table.add_column(
            name='stimulus_intensity',
            description='Stimulus intensity values'
        )
        io_table.add_column(
            name='mean_response',
            description='Mean response amplitude'
        )
        io_table.add_column(
            name='sem_response',
            description='Standard error of response'
        )
        
        for i in range(n_intensities):
            io_table.add_row(
                stimulus_intensity=stim_intensities[i],
                mean_response=mean_responses[i],
                sem_response=sem_responses[i]
            )
        
        # Add table to analysis
        if not hasattr(nwbfile, 'analysis'):
            nwbfile.create_processing_module(
                name='analysis',
                description='Analysis results'
            )
        nwbfile.processing['analysis'].add(io_table)
        
        # Save file
        filename = os.path.join(self.output_dir, 'input_output_test.nwb')
        with NWBHDF5IO(filename, 'w') as io:
            io.write(nwbfile)
            
        print(f"✓ Saved input-output data to {filename}")
        return filename
    
    def generate_all_datasets(self):
        """Generate all test datasets."""
        print("\n🔄 Generating all test datasets...")
        
        files = []
        files.append(self.generate_action_potential_data())
        files.append(self.generate_synaptic_current_data())
        files.append(self.generate_paired_pulse_data())
        files.append(self.generate_input_output_data())
        
        print(f"\n✅ Generated {len(files)} test datasets in {self.output_dir}")
        return files


def create_validation_script():
    """Create a validation script that works with both Python and MATLAB."""
    
    validation_code = """
# Python validation script for NWB test data
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

def validate_action_potential_data(filename):
    """Validate action potential test data."""
    print(f"\nValidating {filename}...")
    
    with NWBHDF5IO(filename, 'r') as io:
        nwbfile = io.read()
        
        # Check that we have current clamp data
        assert 'CurrentClampSeries' in nwbfile.acquisition
        cc_series = nwbfile.acquisition['CurrentClampSeries']
        
        # Get data and time
        voltage = cc_series.data[:]
        rate = cc_series.rate
        time = np.arange(len(voltage)) / rate
        
        # Basic checks
        assert len(voltage) > 0, "No voltage data found"
        assert rate == 20000.0, f"Expected 20kHz sampling rate, got {rate}"
        
        # Check for action potentials
        threshold = -20  # mV
        peaks = voltage > threshold
        n_spikes = np.sum(np.diff(peaks.astype(int)) > 0)
        assert n_spikes > 10, f"Expected >10 spikes, found {n_spikes}"
        
        print(f"✓ Found {n_spikes} action potentials")
        print(f"✓ Voltage range: {voltage.min():.1f} to {voltage.max():.1f} mV")
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, voltage, 'b-', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.title('Action Potential Test Data')
        plt.tight_layout()
        plt.show()

def validate_synaptic_current_data(filename):
    """Validate synaptic current test data."""
    print(f"\nValidating {filename}...")
    
    with NWBHDF5IO(filename, 'r') as io:
        nwbfile = io.read()
        
        # Check that we have voltage clamp data
        assert 'VoltageClampSeries' in nwbfile.acquisition
        vc_series = nwbfile.acquisition['VoltageClampSeries']
        
        # Get data
        current = vc_series.data[:]
        rate = vc_series.rate
        time = np.arange(len(current)) / rate
        
        # Basic checks
        assert len(current) > 0, "No current data found"
        
        # Detect events (simplified)
        baseline = np.median(current)
        threshold = baseline - 3 * np.std(current[:int(0.1*len(current))])
        events = current < threshold
        n_events = np.sum(np.diff(events.astype(int)) > 0)
        
        print(f"✓ Found approximately {n_events} synaptic events")
        print(f"✓ Current range: {current.min():.1f} to {current.max():.1f} pA")

# Run validation if called directly
if __name__ == "__main__":
    import os
    test_dir = './test_data'
    
    if os.path.exists(os.path.join(test_dir, 'action_potential_test.nwb')):
        validate_action_potential_data(os.path.join(test_dir, 'action_potential_test.nwb'))
    
    if os.path.exists(os.path.join(test_dir, 'synaptic_current_test.nwb')):
        validate_synaptic_current_data(os.path.join(test_dir, 'synaptic_current_test.nwb'))
"""
    
    with open('./test_data/validate_nwb_data.py', 'w') as f:
        f.write(validation_code)
    
    # Also create MATLAB validation script
    matlab_code = """% MATLAB validation script for NWB test data
% Requires MatNWB: https://github.com/NeurodataWithoutBorders/matnwb

function validate_nwb_data()
    % Add MatNWB to path (adjust as needed)
    % addpath('/path/to/matnwb');
    
    test_dir = './test_data';
    
    % Validate action potential data
    if exist(fullfile(test_dir, 'action_potential_test.nwb'), 'file')
        validate_action_potential_data(fullfile(test_dir, 'action_potential_test.nwb'));
    end
    
    % Validate synaptic current data
    if exist(fullfile(test_dir, 'synaptic_current_test.nwb'), 'file')
        validate_synaptic_current_data(fullfile(test_dir, 'synaptic_current_test.nwb'));
    end
end

function validate_action_potential_data(filename)
    fprintf('\nValidating %s...\n', filename);
    
    % Read NWB file
    nwbfile = nwbRead(filename);
    
    % Get current clamp data
    cc_series = nwbfile.acquisition.get('CurrentClampSeries');
    voltage = cc_series.data.load();
    rate = cc_series.starting_time_rate;
    time = (0:length(voltage)-1) / rate;
    
    % Basic checks
    assert(~isempty(voltage), 'No voltage data found');
    assert(rate == 20000, sprintf('Expected 20kHz sampling rate, got %g', rate));
    
    % Check for action potentials
    threshold = -20; % mV
    peaks = voltage > threshold;
    n_spikes = sum(diff([0; peaks(:); 0]) > 0);
    assert(n_spikes > 10, sprintf('Expected >10 spikes, found %d', n_spikes));
    
    fprintf('✓ Found %d action potentials\n', n_spikes);
    fprintf('✓ Voltage range: %.1f to %.1f mV\n', min(voltage), max(voltage));
    
    % Plot
    figure;
    plot(time, voltage, 'b-', 'LineWidth', 0.5);
    xlabel('Time (s)');
    ylabel('Voltage (mV)');
    title('Action Potential Test Data');
end

function validate_synaptic_current_data(filename)
    fprintf('\nValidating %s...\n', filename);
    
    % Read NWB file
    nwbfile = nwbRead(filename);
    
    % Get voltage clamp data
    vc_series = nwbfile.acquisition.get('VoltageClampSeries');
    current = vc_series.data.load();
    rate = vc_series.starting_time_rate;
    time = (0:length(current)-1) / rate;
    
    % Basic checks
    assert(~isempty(current), 'No current data found');
    
    % Detect events (simplified)
    baseline = median(current);
    noise_std = std(current(1:floor(0.1*length(current))));
    threshold = baseline - 3 * noise_std;
    events = current < threshold;
    n_events = sum(diff([0; events(:); 0]) > 0);
    
    fprintf('✓ Found approximately %d synaptic events\n', n_events);
    fprintf('✓ Current range: %.1f to %.1f pA\n', min(current), max(current));
end
"""
    
    with open('./test_data/validate_nwb_data.m', 'w') as f:
        f.write(matlab_code)
    
    print("\n📝 Created validation scripts:")
    print("  - validate_nwb_data.py (Python)")
    print("  - validate_nwb_data.m (MATLAB)")


if __name__ == "__main__":
    # Generate all test datasets
    generator = ElectrophysiologyTestDataGenerator()
    generator.generate_all_datasets()
    
    # Create validation scripts
    create_validation_script()
    
    print("\n✅ Test data generation complete!")
    print("\nTo use these datasets:")
    print("1. In Python: Load with PyNWB")
    print("2. In MATLAB: Load with MatNWB")
    print("3. Run validation scripts to verify data integrity")
