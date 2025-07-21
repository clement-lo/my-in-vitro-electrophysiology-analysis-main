"""
v1 - Simple Electrophysiology Analysis Package

Educational and prototyping implementations for patch-clamp and whole-cell
electrophysiology data analysis.

Modules:
    - action_potential_analysis: AP detection and characterization
    - action_potential_spike_train_analysis: Spike train analysis
    - synaptic_current_analysis: Synaptic event detection
    - synaptic_input_output_analysis: I/O relationship analysis
    - ion_channels_kinetics: Channel kinetics modeling
    - network_connectivity_plasticity_analysis: Network analysis
    - pharmacological_modulation: Drug effect analysis
    - time_frequency_analysis: Spectral analysis
"""

__version__ = "1.0.0"
__author__ = "Neuroscience Research Lab"

# Import main functions for convenience
try:
    from .action_potential_analysis import detect_action_potentials
    from .synaptic_current_analysis import detect_synaptic_events
    from .ion_channels_kinetics import fit_activation_curve
    from .network_connectivity_plasticity_analysis import calculate_connectivity
    from .pharmacological_modulation import analyze_dose_response
    from .time_frequency_analysis import compute_spectrogram
except ImportError:
    # Handle case where dependencies might not be installed
    pass

__all__ = [
    'detect_action_potentials',
    'detect_synaptic_events',
    'fit_activation_curve',
    'calculate_connectivity',
    'analyze_dose_response',
    'compute_spectrogram'
]