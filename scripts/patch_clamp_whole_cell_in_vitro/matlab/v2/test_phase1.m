% Test Phase 1: Core Infrastructure
% Test with neural_data_1.mat

clear; clc;

% Add paths
addpath('scripts/enhanced/core');

% Configuration
% Get the project root directory (5 levels up from current file)
current_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(current_dir, '..', '..', '..', '..', '..');
file_path = fullfile(project_root, 'data', 'neural_data_1.mat');
sampling_rate = 40000; % Hz - adjust based on your data

try
    fprintf('=== Testing Phase 1: Core Infrastructure ===\n\n');
    
    % Test 1: Configuration management
    fprintf('Test 1: Configuration management...\n');
    config = create_analysis_config();
    fprintf('✓ Configuration created successfully\n\n');
    
    % Test 2: Data loading
    fprintf('Test 2: Enhanced data loading...\n');
    neural_data = load_electrophysiology_data(file_path, sampling_rate);
    fprintf('✓ Data loaded successfully\n\n');
    
    % Test 3: Signal preprocessing
    fprintf('Test 3: Signal preprocessing...\n');
    processed_data = advanced_signal_preprocessing(neural_data, config);
    fprintf('✓ Signal preprocessing completed\n\n');
    
    % Quick visualization
    figure('Name', 'Phase 1 Test Results');
    subplot(3,1,1);
    plot(processed_data.Time(1:min(10000, height(processed_data))), ...
         processed_data.raw(1:min(10000, height(processed_data))));
    title('Raw Signal');
    ylabel('Amplitude (V)');
    
    subplot(3,1,2);
    plot(processed_data.Time(1:min(10000, height(processed_data))), ...
         processed_data.spikes(1:min(10000, height(processed_data))));
    title('Spike Component (High-pass filtered)');
    ylabel('Amplitude (V)');
    
    subplot(3,1,3);
    plot(processed_data.Time(1:min(10000, height(processed_data))), ...
         processed_data.LFP(1:min(10000, height(processed_data))));
    title('LFP Component (Low-pass filtered)');
    ylabel('Amplitude (V)');
    xlabel('Time (s)');
    
    % Save test results
    save(fullfile(config.paths.data_processed, 'phase1_test_results.mat'), ...
         'processed_data', 'config');
    
    fprintf('=== Phase 1 Testing Completed Successfully ===\n');
    
catch ME
    fprintf('❌ Phase 1 test failed: %s\n', ME.message);
    rethrow(ME);
end