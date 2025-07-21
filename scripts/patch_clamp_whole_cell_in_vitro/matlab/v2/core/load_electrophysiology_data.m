function neural_data = load_electrophysiology_data(file_path, sampling_rate)
    % Enhanced data loading with timetable integration
    % 
    % Args:
    %   file_path (str): Path to the data file
    %   sampling_rate (double): Sampling rate in Hz
    %
    % Returns:
    %   neural_data (timetable): Loaded data with metadata
    
    % Validate inputs
    if ~exist(file_path, 'file')
        error('File not found: %s', file_path);
    end
    
    % Load MAT file
    loaded_data = load(file_path);
    field_names = fieldnames(loaded_data);
    
    % Extract signal data (assuming first field contains the signal)
    if isfield(loaded_data, 'data')
        signal = loaded_data.data;
    else
        signal = loaded_data.(field_names{1});
    end
    
    % Handle different data shapes
    if size(signal, 2) > size(signal, 1)
        signal = signal'; % Ensure column vector
    end
    
    % If multi-channel, take first channel
    if size(signal, 2) > 1
        signal = signal(:, 1);
        warning('Multi-channel data detected. Using first channel only.');
    end
    
    % Create timetable with proper sampling rate
    neural_data = timetable(signal, 'SampleRate', sampling_rate);
    neural_data.Properties.VariableNames{1} = 'raw';
    neural_data.Properties.VariableUnits{1} = 'Volts';
    
    % Add metadata
    neural_data.Properties.UserData.file_path = file_path;
    neural_data.Properties.UserData.load_timestamp = datetime('now');
    neural_data.Properties.UserData.duration_seconds = height(neural_data) / sampling_rate;
    
    fprintf('Loaded %d samples (%.2f seconds) at %.1f Hz\n', ...
            height(neural_data), neural_data.Properties.UserData.duration_seconds, sampling_rate);
end