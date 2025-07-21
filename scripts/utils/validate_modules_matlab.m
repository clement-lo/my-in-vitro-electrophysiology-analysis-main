%% Comprehensive MATLAB Validation Script
% Tests electrophysiology analysis modules with NWB test data

function validate_electrophysiology_modules()
    % Add paths
    addpath(genpath('scripts/classic_in_vitro/matlab'));
    % addpath('/path/to/matnwb'); % Add MatNWB path
    
    fprintf('================================================================================\n');
    fprintf('MATLAB ELECTROPHYSIOLOGY ANALYSIS VALIDATION\n');
    fprintf('================================================================================\n');
    
    % Create output directory
    if ~exist('validation_results_matlab', 'dir')
        mkdir('validation_results_matlab');
    end
    
    % Run validations
    validate_action_potential_analysis_matlab();
    validate_synaptic_analysis_matlab();
    validate_nwb_compatibility();
    
    fprintf('\n================================================================================\n');
    fprintf('✅ MATLAB VALIDATION COMPLETE\n');
    fprintf('================================================================================\n');
end

function validate_action_potential_analysis_matlab()
    fprintf('\n🔬 VALIDATING ACTION POTENTIAL ANALYSIS (MATLAB)\n');
    fprintf('------------------------------------------------------------\n');
    
    % Test with CSV data
    fprintf('\n1. Testing with CSV file...\n');
    try
        % Load CSV data
        data = readtable('test_data/action_potential_test.csv');
        time = data.Time;
        voltage = data.Voltage;
        sampling_rate = 1 / (time(2) - time(1));
        
        % Run v1 MATLAB analysis
        cd('scripts/classic_in_vitro/matlab/v1');
        [spike_times, spike_indices] = detect_action_potentials(voltage, sampling_rate);
        
        fprintf('✓ Detected %d spikes\n', length(spike_times));
        fprintf('✓ Mean firing rate: %.2f Hz\n', length(spike_times) / (time(end) - time(1)));
        
        % Plot results
        figure('Name', 'MATLAB AP Analysis');
        plot(time, voltage, 'b-');
        hold on;
        plot(spike_times, voltage(spike_indices), 'ro', 'MarkerSize', 8);
        xlabel('Time (s)');
        ylabel('Voltage (mV)');
        title('Action Potential Detection - MATLAB');
        legend('Voltage', 'Detected Spikes');
        
        saveas(gcf, 'validation_results_matlab/ap_detection_matlab.png');
        cd('../../../..');
        
    catch ME
        fprintf('✗ Error with CSV analysis: %s\n', ME.message);
    end
    
    % Test with NWB data
    fprintf('\n2. Testing with NWB file...\n');
    try
        % Check if MatNWB is available
        if exist('nwbRead', 'file')
            nwbfile = nwbRead('test_data/action_potential_test.nwb');
            cc_series = nwbfile.acquisition.get('CurrentClampSeries');
            voltage_nwb = cc_series.data.load();
            rate_nwb = cc_series.starting_time_rate;
            
            fprintf('✓ Loaded NWB file successfully\n');
            fprintf('✓ Data length: %d samples at %.0f Hz\n', length(voltage_nwb), rate_nwb);
        else
            fprintf('⚠ MatNWB not found - skipping NWB test\n');
        end
    catch ME
        fprintf('✗ Error with NWB: %s\n', ME.message);
    end
end

function validate_synaptic_analysis_matlab()
    fprintf('\n🔬 VALIDATING SYNAPTIC ANALYSIS (MATLAB)\n');
    fprintf('------------------------------------------------------------\n');
    
    % Test with CSV data
    fprintf('\n1. Testing synaptic event detection...\n');
    try
        % Load CSV data
        data = readtable('test_data/synaptic_current_test.csv');
        time = data.Time;
        current = data.Current;
        sampling_rate = 1 / (time(2) - time(1));
        
        % Simple event detection
        baseline = median(current);
        noise_std = std(current(1:floor(0.1*length(current))));
        threshold = baseline - 3 * noise_std;
        
        events = find(current < threshold);
        % Find event onsets
        event_onsets = events([true; diff(events) > 1]);
        n_events = length(event_onsets);
        
        fprintf('✓ Detected %d synaptic events\n', n_events);
        fprintf('✓ Mean event rate: %.2f Hz\n', n_events / (time(end) - time(1)));
        
        % Plot
        figure('Name', 'MATLAB Synaptic Analysis');
        plot(time, current, 'k-');
        hold on;
        plot(time(event_onsets), current(event_onsets), 'ro', 'MarkerSize', 6);
        xlabel('Time (s)');
        ylabel('Current (pA)');
        title('Synaptic Event Detection - MATLAB');
        legend('Current', 'Detected Events');
        
        saveas(gcf, 'validation_results_matlab/synaptic_detection_matlab.png');
        
    catch ME
        fprintf('✗ Error with synaptic analysis: %s\n', ME.message);
    end
    
    % Test I/O curve fitting
    fprintf('\n2. Testing input-output curve fitting...\n');
    try
        % Generate test I/O data
        stim = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]';
        resp = 5 + 75 ./ (1 + exp((45 - stim) / 10));
        resp = resp + randn(size(resp)) * 3;
        
        % Fit sigmoidal curve
        ft = fittype('a + (b-a)/(1+exp((c-x)/d))', 'independent', 'x');
        opts = fitoptions(ft);
        opts.StartPoint = [min(resp), max(resp), median(stim), 10];
        opts.Lower = [0, 0, 0, 0.1];
        
        [fitresult, gof] = fit(stim, resp, ft, opts);
        
        fprintf('✓ I/O curve fit successful\n');
        fprintf('✓ EC50 = %.1f, R² = %.3f\n', fitresult.c, gof.rsquare);
        
        % Plot
        figure('Name', 'MATLAB I/O Curve');
        plot(fitresult, stim, resp);
        xlabel('Stimulus Intensity');
        ylabel('Response Amplitude');
        title('Input-Output Curve - MATLAB');
        legend('Data', 'Fitted Curve', 'Location', 'Best');
        
        saveas(gcf, 'validation_results_matlab/io_curve_matlab.png');
        
    catch ME
        fprintf('✗ Error with I/O fitting: %s\n', ME.message);
    end
end

function validate_nwb_compatibility()
    fprintf('\n🔬 VALIDATING NWB COMPATIBILITY\n');
    fprintf('------------------------------------------------------------\n');
    
    if ~exist('nwbRead', 'file')
        fprintf('⚠ MatNWB not installed - skipping NWB compatibility tests\n');
        fprintf('\nTo install MatNWB:\n');
        fprintf('1. git clone https://github.com/NeurodataWithoutBorders/matnwb.git\n');
        fprintf('2. Add matnwb to MATLAB path\n');
        return;
    end
    
    % Test reading both NWB files
    files = {'action_potential_test.nwb', 'synaptic_current_test.nwb'};
    
    for i = 1:length(files)
        fprintf('\nTesting %s...\n', files{i});
        try
            nwbfile = nwbRead(fullfile('test_data', files{i}));
            
            % Check basic properties
            fprintf('✓ Session ID: %s\n', nwbfile.general_session_id);
            fprintf('✓ Session description: %s\n', nwbfile.session_description);
            
            % List acquisition data
            acq_keys = keys(nwbfile.acquisition);
            fprintf('✓ Acquisition data: ');
            for j = 1:length(acq_keys)
                fprintf('%s ', acq_keys{j});
            end
            fprintf('\n');
            
        catch ME
            fprintf('✗ Error reading %s: %s\n', files{i}, ME.message);
        end
    end
end

%% Helper Functions

function [spike_times, spike_indices] = detect_action_potentials(voltage, sampling_rate)
    % Simple threshold-based spike detection
    threshold = -20; % mV
    
    % Find threshold crossings
    above_threshold = voltage > threshold;
    crossings = find(diff([0; above_threshold; 0]) > 0);
    
    spike_indices = crossings;
    spike_times = (crossings - 1) / sampling_rate;
end
