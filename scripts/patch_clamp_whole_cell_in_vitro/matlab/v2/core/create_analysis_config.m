function config = create_analysis_config()
    % Comprehensive configuration for electrophysiology analysis
    
    % Sampling parameters
    config.sampling.default_rate = 40000; % Hz
    config.sampling.min_rate = 10000; % Hz
    config.sampling.nyquist_safety_factor = 2.5;
    
    % Signal processing parameters
    config.filtering.spike_highpass = 500;   % Hz
    config.filtering.lfp_lowpass = 200;      % Hz
    config.filtering.filter_order = 5;
    config.filtering.use_zero_phase = true;
    
    % Spike detection parameters
    config.spike_detection.threshold = -0.000033; % V
    config.spike_detection.min_peak_distance = 0.001; % s
    config.spike_detection.method = 'threshold';
    
    % Spike extraction parameters
    config.spike_extraction.pre_peak_ms = 0.5;  % ms
    config.spike_extraction.post_peak_ms = 0.5; % ms
    
    % Clustering parameters
    config.clustering.method = 'pca_gmm';
    config.clustering.pca_components = 3;
    config.clustering.n_clusters = 2;
    config.clustering.gmm_options = struct(...
        'Start', 'plus', ...
        'CovarianceType', 'diagonal', ...
        'SharedCovariance', true);
    
    % LFP analysis parameters
    config.lfp.resample_rate = 1000;  % Hz
    config.lfp.spectrogram_window = 220; % samples
    config.lfp.frequency_range = [1, 100]; % Hz
    
    % Visualization options
    config.visualization.save_plots = true;
    config.visualization.plot_format = 'png';
    config.visualization.figure_dpi = 300;
    
    % Output directories
    config.paths.results = 'results';
    config.paths.figures = 'results/figures';
    config.paths.data_processed = 'results/processed_data';
    
    % Create directories if they don't exist
    create_output_directories(config.paths);
end

function create_output_directories(paths)
    % Create output directories
    dirs = struct2cell(paths);
    for i = 1:length(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
end