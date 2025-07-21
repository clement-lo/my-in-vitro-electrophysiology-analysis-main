function processed_data = advanced_signal_preprocessing(neural_data, config)
    % Advanced signal processing with spike and LFP separation
    %
    % Args:
    %   neural_data (timetable): Raw neural data
    %   config (struct): Analysis configuration
    %
    % Returns:
    %   processed_data (timetable): Processed data with spike and LFP components
    
    sampling_rate = neural_data.Properties.SampleRate;
    
    % Validate sampling rate
    validate_sampling_rate(sampling_rate, config);
    
    % Design filters
    [b_high, a_high] = butter(config.filtering.filter_order, ...
                             config.filtering.spike_highpass/sampling_rate, "high");
    [b_low, a_low] = butter(config.filtering.filter_order, ...
                           config.filtering.lfp_lowpass/sampling_rate, "low");
    
    % Apply filtering
    if config.filtering.use_zero_phase
        neural_data.spikes = filtfilt(b_high, a_high, neural_data.raw);
        neural_data.LFP = filtfilt(b_low, a_low, neural_data.raw);
    else
        neural_data.spikes = filter(b_high, a_high, neural_data.raw);
        neural_data.LFP = filter(b_low, a_low, neural_data.raw);
    end
    
    % Update metadata
    neural_data.Properties.VariableUnits(2:3) = {'Volts','Volts'};
    neural_data.Properties.VariableDescriptions = {...
        'Raw signal', 'High-pass filtered (spikes)', 'Low-pass filtered (LFP)'};
    
    processed_data = neural_data;
    
    fprintf('Signal preprocessing completed: spike filter >%.0f Hz, LFP filter <%.0f Hz\n', ...
            config.filtering.spike_highpass, config.filtering.lfp_lowpass);
end

function validate_sampling_rate(sampling_rate, config)
    % Validate sampling rate against requirements
    min_required = config.filtering.spike_highpass * config.sampling.nyquist_safety_factor;
    if sampling_rate < min_required
        warning('Sampling rate (%.1f Hz) may be insufficient. Recommended: >%.1f Hz', ...
                sampling_rate, min_required);
    end
end