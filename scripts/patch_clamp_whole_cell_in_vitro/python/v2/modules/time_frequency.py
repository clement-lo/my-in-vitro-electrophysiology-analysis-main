"""
Time-Frequency Analysis Module v2
Enhanced with wavelet analysis and cross-frequency coupling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import pywt
from typing import Dict, Any, Union, Tuple, Optional, List
from pathlib import Path
import logging

from ..core.base_analysis import AbstractAnalysis, ValidationMixin
from ..core.data_pipeline import load_data
from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TimeFrequencyAnalysis(AbstractAnalysis, ValidationMixin):
    """
    Advanced time-frequency analysis for electrophysiology data.
    
    Features:
    - Multiple spectral analysis methods (FFT, Welch, multitaper)
    - Wavelet transform (continuous and discrete)
    - Cross-frequency coupling analysis
    - Event-triggered spectral analysis
    - Coherence and phase analysis
    """
    
    def __init__(self, config: Union[Dict, ConfigManager, Path, str, None] = None):
        super().__init__(name="time_frequency_analysis", version="2.0")
        
        # Handle configuration
        if isinstance(config, ConfigManager):
            self.config_manager = config
        elif isinstance(config, (Path, str)):
            self.config_manager = ConfigManager(config)
        elif isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config_manager.config = config
        else:
            self.config_manager = ConfigManager()
            
        self.config = self.config_manager.create_analysis_config('time_frequency')
        
        # Available methods
        self.methods = {
            'fft': self._compute_fft,
            'welch': self._compute_welch_psd,
            'multitaper': self._compute_multitaper,
            'wavelet': self._compute_wavelet_transform,
            'spectrogram': self._compute_spectrogram,
            'coherence': self._compute_coherence
        }
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load electrophysiology data for time-frequency analysis."""
        signal_data, rate, time, metadata = load_data(file_path, **kwargs)
        
        # Handle multi-channel data
        if signal_data.ndim > 1:
            n_channels = signal_data.shape[1]
            logger.info(f"Loaded {n_channels} channels")
        else:
            signal_data = signal_data.reshape(-1, 1)
            n_channels = 1
            
        self.data = {
            'signals': signal_data,
            'sampling_rate': rate,
            'time': time,
            'n_channels': n_channels,
            'metadata': metadata
        }
        
        return self.data
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate time-frequency analysis parameters."""
        required = ['method']
        
        for param in required:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
        # Validate method
        parameters['method'] = self.validate_choice_parameter(
            parameters['method'],
            list(self.methods.keys()),
            'method'
        )
        
        # Method-specific validation
        method = parameters['method']
        
        if method in ['fft', 'welch', 'multitaper']:
            if 'freq_range' in parameters:
                freq_range = parameters['freq_range']
                if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                    raise ValueError("freq_range must be [min, max]")
                    
        if method == 'wavelet':
            if 'wavelet' not in parameters:
                parameters['wavelet'] = 'morlet'
            if 'frequencies' not in parameters:
                parameters['frequencies'] = np.logspace(0, 2.3, 50)  # 1-200 Hz
                
        return True
    
    def run_analysis(self, data: Any = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run time-frequency analysis."""
        if data is None:
            data = self.data
            
        if parameters is None:
            parameters = self.config.get('time_frequency', {
                'method': 'wavelet',
                'wavelet': 'morlet',
                'frequencies': [1, 200],
                'n_frequencies': 50,
                'freq_range': [1, 100],
                'compute_phase': True,
                'compute_coupling': False
            })
            
        self.validate_parameters(parameters)
        self.metadata['parameters'] = parameters
        
        method = parameters['method']
        method_func = self.methods[method]
        
        results = {}
        
        # Run primary analysis
        logger.info(f"Running {method} analysis...")
        primary_results = method_func(data, parameters)
        results.update(primary_results)
        
        # Additional analyses
        if parameters.get('compute_phase', False) and method == 'wavelet':
            logger.info("Computing phase analysis...")
            phase_results = self._analyze_phase(primary_results['wavelet_coeffs'])
            results['phase'] = phase_results
            
        if parameters.get('compute_coupling', False):
            logger.info("Computing cross-frequency coupling...")
            coupling_results = self._compute_cross_frequency_coupling(data, parameters)
            results['coupling'] = coupling_results
            
        if parameters.get('event_triggered', False):
            logger.info("Computing event-triggered spectra...")
            event_results = self._compute_event_triggered_spectra(data, parameters)
            results['event_triggered'] = event_results
            
        self.results = results
        return results
    
    def _compute_fft(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Fast Fourier Transform."""
        signals = data['signals']
        fs = data['sampling_rate']
        
        # FFT parameters
        nperseg = parameters.get('nperseg', None)
        if nperseg is None:
            nperseg = min(len(signals), int(fs * 2))  # 2 second window
            
        results = {}
        
        for ch in range(data['n_channels']):
            signal = signals[:, ch]
            
            # Remove DC component
            signal = signal - np.mean(signal)
            
            # Apply window
            window = parameters.get('window', 'hann')
            if window:
                w = signal.get_window(window, len(signal))
                signal = signal * w
                
            # Compute FFT
            freqs = rfftfreq(len(signal), 1/fs)
            fft_vals = rfft(signal)
            
            # Power spectral density
            psd = np.abs(fft_vals) ** 2 / (fs * len(signal))
            
            # Apply frequency range
            if 'freq_range' in parameters:
                fmin, fmax = parameters['freq_range']
                mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[mask]
                psd = psd[mask]
                
            results[f'channel_{ch}'] = {
                'frequencies': freqs,
                'psd': psd,
                'fft_complex': fft_vals if parameters.get('keep_complex', False) else None
            }
            
        # Average across channels if requested
        if parameters.get('average_channels', False) and data['n_channels'] > 1:
            avg_psd = np.mean([results[f'channel_{ch}']['psd'] 
                              for ch in range(data['n_channels'])], axis=0)
            results['average'] = {
                'frequencies': results['channel_0']['frequencies'],
                'psd': avg_psd
            }
            
        return results
    
    def _compute_welch_psd(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute power spectral density using Welch's method."""
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Welch parameters
        nperseg = parameters.get('nperseg', int(fs * 2))  # 2 second window
        noverlap = parameters.get('noverlap', nperseg // 2)
        
        results = {}
        
        for ch in range(data['n_channels']):
            signal = signals[:, ch]
            
            # Compute Welch PSD
            freqs, psd = signal.welch(signal, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap,
                                     window=parameters.get('window', 'hann'),
                                     detrend=parameters.get('detrend', 'constant'))
            
            # Apply frequency range
            if 'freq_range' in parameters:
                fmin, fmax = parameters['freq_range']
                mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[mask]
                psd = psd[mask]
                
            results[f'channel_{ch}'] = {
                'frequencies': freqs,
                'psd': psd
            }
            
        # Compute band powers
        if parameters.get('compute_band_powers', True):
            band_powers = self._compute_band_powers(results, fs)
            results['band_powers'] = band_powers
            
        return results
    
    def _compute_multitaper(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute multitaper spectral estimate."""
        try:
            from scipy.signal.windows import dpss
        except ImportError:
            logger.warning("Multitaper requires scipy >= 1.8.0")
            return self._compute_welch_psd(data, parameters)
            
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Multitaper parameters
        nw = parameters.get('bandwidth', 4)  # Time-bandwidth product
        n_tapers = parameters.get('n_tapers', 7)
        
        results = {}
        
        for ch in range(data['n_channels']):
            signal = signals[:, ch]
            
            # Create DPSS tapers
            tapers, ratios = dpss(len(signal), nw, n_tapers, return_ratios=True)
            
            # Compute spectrum for each taper
            spectra = []
            for taper in tapers:
                windowed = signal * taper
                spectrum = np.abs(rfft(windowed)) ** 2
                spectra.append(spectrum)
                
            # Average across tapers
            psd = np.mean(spectra, axis=0)
            freqs = rfftfreq(len(signal), 1/fs)
            
            # Normalize
            psd = psd * 2 / (fs * len(signal))
            
            results[f'channel_{ch}'] = {
                'frequencies': freqs,
                'psd': psd,
                'taper_ratios': ratios
            }
            
        return results
    
    def _compute_wavelet_transform(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute continuous wavelet transform."""
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Wavelet parameters
        wavelet = parameters.get('wavelet', 'morlet')
        
        # Get frequencies
        if 'frequencies' in parameters:
            if isinstance(parameters['frequencies'], (list, tuple)) and len(parameters['frequencies']) == 2:
                # Generate frequency array from range
                fmin, fmax = parameters['frequencies']
                n_freqs = parameters.get('n_frequencies', 50)
                frequencies = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
            else:
                frequencies = np.array(parameters['frequencies'])
        else:
            frequencies = np.logspace(0, 2.3, 50)  # 1-200 Hz
            
        # Convert frequencies to scales for wavelet
        if wavelet == 'morlet':
            # For Morlet wavelet: frequency = 1 / (scale * wavelet_central_freq)
            central_freq = pywt.central_frequency(wavelet)
            scales = central_freq * fs / frequencies
        else:
            scales = fs / frequencies
            
        results = {
            'frequencies': frequencies,
            'time': data['time'],
            'wavelet_type': wavelet
        }
        
        for ch in range(data['n_channels']):
            signal = signals[:, ch]
            
            # Compute CWT
            coeffs, freqs_cwt = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
            
            # Power
            power = np.abs(coeffs) ** 2
            
            results[f'channel_{ch}'] = {
                'coefficients': coeffs,
                'power': power,
                'phase': np.angle(coeffs) if parameters.get('compute_phase', True) else None
            }
            
        # Store for later use
        results['wavelet_coeffs'] = coeffs
        
        return results
    
    def _compute_spectrogram(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute spectrogram using STFT."""
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Spectrogram parameters
        nperseg = parameters.get('nperseg', int(fs * 0.5))  # 0.5 second window
        noverlap = parameters.get('noverlap', int(nperseg * 0.9))  # 90% overlap
        
        results = {}
        
        for ch in range(data['n_channels']):
            signal = signals[:, ch]
            
            # Compute spectrogram
            freqs, times, Sxx = signal.spectrogram(signal, fs=fs,
                                                   nperseg=nperseg,
                                                   noverlap=noverlap,
                                                   window=parameters.get('window', 'hann'))
            
            # Apply frequency range
            if 'freq_range' in parameters:
                fmin, fmax = parameters['freq_range']
                mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[mask]
                Sxx = Sxx[mask, :]
                
            results[f'channel_{ch}'] = {
                'frequencies': freqs,
                'time': times,
                'power': Sxx
            }
            
        return results
    
    def _compute_coherence(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute coherence between channels."""
        if data['n_channels'] < 2:
            logger.warning("Coherence requires at least 2 channels")
            return {}
            
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Coherence parameters
        nperseg = parameters.get('nperseg', int(fs * 2))
        
        # Get channel pairs
        channel_pairs = parameters.get('channel_pairs', [(0, 1)])
        
        results = {}
        
        for ch1, ch2 in channel_pairs:
            if ch1 >= data['n_channels'] or ch2 >= data['n_channels']:
                continue
                
            # Compute coherence
            freqs, Cxy = signal.coherence(signals[:, ch1], signals[:, ch2],
                                         fs=fs, nperseg=nperseg)
            
            # Compute phase difference
            freqs, Pxy = signal.csd(signals[:, ch1], signals[:, ch2],
                                   fs=fs, nperseg=nperseg)
            phase_diff = np.angle(Pxy)
            
            results[f'pair_{ch1}_{ch2}'] = {
                'frequencies': freqs,
                'coherence': Cxy,
                'phase_difference': phase_diff
            }
            
        return results
    
    def _analyze_phase(self, wavelet_coeffs: np.ndarray) -> Dict[str, Any]:
        """Analyze instantaneous phase from wavelet coefficients."""
        phase = np.angle(wavelet_coeffs)
        
        # Phase synchronization metrics
        # Kuramoto order parameter
        kuramoto_r = np.abs(np.mean(np.exp(1j * phase), axis=1))
        
        # Phase coherence across time
        phase_coherence = np.abs(np.mean(np.exp(1j * phase), axis=0))
        
        return {
            'instantaneous_phase': phase,
            'kuramoto_order': kuramoto_r,
            'phase_coherence': phase_coherence,
            'mean_phase': np.mean(phase, axis=1)
        }
    
    def _compute_cross_frequency_coupling(self, data: Dict[str, Any], 
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute phase-amplitude coupling and other cross-frequency metrics."""
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Define frequency bands
        phase_freqs = parameters.get('phase_frequencies', [(4, 8), (8, 12)])  # Theta, Alpha
        amp_freqs = parameters.get('amplitude_frequencies', [(30, 50), (50, 100)])  # Gamma
        
        results = {}
        
        for ch in range(min(data['n_channels'], 1)):  # Limit to first channel for now
            signal_ch = signals[:, ch]
            
            pac_matrix = np.zeros((len(phase_freqs), len(amp_freqs)))
            
            for i, (f_low_phase, f_high_phase) in enumerate(phase_freqs):
                # Extract phase
                phase_signal = self._bandpass_filter(signal_ch, f_low_phase, f_high_phase, fs)
                phase = np.angle(signal.hilbert(phase_signal))
                
                for j, (f_low_amp, f_high_amp) in enumerate(amp_freqs):
                    # Extract amplitude
                    amp_signal = self._bandpass_filter(signal_ch, f_low_amp, f_high_amp, fs)
                    amplitude = np.abs(signal.hilbert(amp_signal))
                    
                    # Compute PAC using Modulation Index
                    pac_value = self._compute_modulation_index(phase, amplitude)
                    pac_matrix[i, j] = pac_value
                    
            results[f'channel_{ch}'] = {
                'pac_matrix': pac_matrix,
                'phase_bands': phase_freqs,
                'amplitude_bands': amp_freqs
            }
            
        return results
    
    def _bandpass_filter(self, signal: np.ndarray, low_freq: float, 
                        high_freq: float, fs: float) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if low >= 1 or high >= 1:
            logger.warning(f"Frequency out of range: {low_freq}-{high_freq} Hz")
            return signal
            
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, signal)
    
    def _compute_modulation_index(self, phase: np.ndarray, amplitude: np.ndarray) -> float:
        """Compute modulation index for phase-amplitude coupling."""
        # Bin phase
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Calculate mean amplitude in each phase bin
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                mean_amp[i] = np.mean(amplitude[mask])
                
        # Normalize
        mean_amp = mean_amp / np.sum(mean_amp)
        
        # Calculate entropy
        entropy = -np.sum(mean_amp * np.log(mean_amp + 1e-10))
        
        # Normalize to get MI
        max_entropy = np.log(n_bins)
        mi = (max_entropy - entropy) / max_entropy
        
        return mi
    
    def _compute_band_powers(self, psd_results: Dict[str, Any], fs: float) -> Dict[str, Any]:
        """Compute power in standard frequency bands."""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200)
        }
        
        band_powers = {}
        
        for ch_key, ch_data in psd_results.items():
            if not ch_key.startswith('channel_'):
                continue
                
            freqs = ch_data['frequencies']
            psd = ch_data['psd']
            
            ch_bands = {}
            for band_name, (f_low, f_high) in bands.items():
                mask = (freqs >= f_low) & (freqs < f_high)
                if np.any(mask):
                    # Integrate power in band
                    band_power = np.trapz(psd[mask], freqs[mask])
                    ch_bands[band_name] = band_power
                    
            band_powers[ch_key] = ch_bands
            
        return band_powers
    
    def _compute_event_triggered_spectra(self, data: Dict[str, Any], 
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute spectral analysis triggered on specific events."""
        event_times = parameters.get('event_times', [])
        if not event_times:
            return {}
            
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Window parameters
        pre_event = parameters.get('pre_event_ms', 500) / 1000  # Convert to seconds
        post_event = parameters.get('post_event_ms', 500) / 1000
        
        pre_samples = int(pre_event * fs)
        post_samples = int(post_event * fs)
        total_samples = pre_samples + post_samples
        
        # Collect event-aligned segments
        segments = []
        
        for event_time in event_times:
            event_idx = int(event_time * fs)
            
            if event_idx - pre_samples >= 0 and event_idx + post_samples < len(signals):
                segment = signals[event_idx - pre_samples:event_idx + post_samples, 0]
                segments.append(segment)
                
        if not segments:
            return {}
            
        segments = np.array(segments)
        
        # Compute average spectrogram
        all_spectrograms = []
        
        for segment in segments:
            freqs, times, Sxx = signal.spectrogram(segment, fs=fs,
                                                   nperseg=int(fs * 0.1),
                                                   noverlap=int(fs * 0.09))
            all_spectrograms.append(Sxx)
            
        # Average across events
        mean_spectrogram = np.mean(all_spectrograms, axis=0)
        
        # Adjust time axis to be relative to event
        times = times - pre_event
        
        return {
            'frequencies': freqs,
            'time': times,
            'mean_spectrogram': mean_spectrogram,
            'n_events': len(segments),
            'all_spectrograms': np.array(all_spectrograms) if parameters.get('keep_all', False) else None
        }
    
    def generate_report(self, results: Dict[str, Any] = None, 
                       output_dir: Union[str, Path] = None) -> Path:
        """Generate comprehensive time-frequency analysis report."""
        if results is None:
            results = self.results
            
        if output_dir is None:
            output_dir = Path('./results')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine figure layout based on analyses performed
        n_plots = sum([
            'frequencies' in results,
            'wavelet_coeffs' in results,
            'coherence' in results,
            'coupling' in results,
            'event_triggered' in results
        ])
        
        fig = plt.figure(figsize=(16, 4 * n_plots))
        plot_idx = 1
        
        # 1. Power Spectral Density
        if 'frequencies' in results or 'channel_0' in results:
            ax1 = plt.subplot(n_plots, 1, plot_idx)
            self._plot_psd(ax1, results)
            plot_idx += 1
            
        # 2. Wavelet/Spectrogram
        if 'wavelet_coeffs' in results or any('power' in v for v in results.values() if isinstance(v, dict)):
            ax2 = plt.subplot(n_plots, 1, plot_idx)
            self._plot_time_frequency(ax2, results)
            plot_idx += 1
            
        # 3. Coherence
        if any('coherence' in str(k) for k in results.keys()):
            ax3 = plt.subplot(n_plots, 1, plot_idx)
            self._plot_coherence(ax3, results)
            plot_idx += 1
            
        # 4. Cross-frequency coupling
        if 'coupling' in results:
            ax4 = plt.subplot(n_plots, 1, plot_idx)
            self._plot_coupling(ax4, results['coupling'])
            plot_idx += 1
            
        # 5. Event-triggered spectra
        if 'event_triggered' in results:
            ax5 = plt.subplot(n_plots, 1, plot_idx)
            self._plot_event_triggered(ax5, results['event_triggered'])
            
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / 'time_frequency_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_path = output_dir / 'time_frequency_report.txt'
        self._generate_text_report(results, report_path)
        
        # Export results
        self.export_results(output_dir / 'time_frequency_results', format='json')
        
        logger.info(f"Time-frequency analysis report saved to {output_dir}")
        
        return report_path
    
    def _plot_psd(self, ax, results):
        """Plot power spectral density."""
        # Find PSD data in results
        for key, value in results.items():
            if isinstance(value, dict) and 'psd' in value:
                freqs = value['frequencies']
                psd = value['psd']
                
                # Convert to dB
                psd_db = 10 * np.log10(psd + 1e-20)
                
                label = key.replace('_', ' ').title()
                ax.plot(freqs, psd_db, label=label)
                
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to log scale if wide range
        if ax.get_xlim()[1] / ax.get_xlim()[0] > 100:
            ax.set_xscale('log')
    
    def _plot_time_frequency(self, ax, results):
        """Plot time-frequency representation."""
        # Find time-frequency data
        tf_data = None
        time = None
        freqs = None
        
        for key, value in results.items():
            if isinstance(value, dict) and 'power' in value and value['power'].ndim == 2:
                tf_data = value['power']
                time = value.get('time', results.get('time'))
                freqs = value.get('frequencies', results.get('frequencies'))
                break
                
        if tf_data is not None:
            # Plot spectrogram
            im = ax.imshow(10 * np.log10(tf_data + 1e-20), 
                          aspect='auto', origin='lower',
                          extent=[time[0], time[-1], freqs[0], freqs[-1]],
                          cmap='viridis')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Time-Frequency Representation')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power (dB)')
    
    def _plot_coherence(self, ax, results):
        """Plot coherence results."""
        for key, value in results.items():
            if 'coherence' in key and isinstance(value, dict):
                freqs = value['frequencies']
                coherence = value['coherence']
                
                pair_label = key.replace('pair_', 'Channels ')
                ax.plot(freqs, coherence, label=pair_label)
                
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title('Channel Coherence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_coupling(self, ax, coupling_data):
        """Plot cross-frequency coupling."""
        if 'channel_0' in coupling_data:
            ch_data = coupling_data['channel_0']
            pac_matrix = ch_data['pac_matrix']
            
            im = ax.imshow(pac_matrix, aspect='auto', origin='lower', cmap='hot')
            
            # Set tick labels
            phase_labels = [f"{f[0]}-{f[1]}" for f in ch_data['phase_bands']]
            amp_labels = [f"{f[0]}-{f[1]}" for f in ch_data['amplitude_bands']]
            
            ax.set_xticks(range(len(amp_labels)))
            ax.set_xticklabels(amp_labels)
            ax.set_yticks(range(len(phase_labels)))
            ax.set_yticklabels(phase_labels)
            
            ax.set_xlabel('Amplitude Frequency (Hz)')
            ax.set_ylabel('Phase Frequency (Hz)')
            ax.set_title('Phase-Amplitude Coupling')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Modulation Index')
    
    def _plot_event_triggered(self, ax, event_data):
        """Plot event-triggered spectrogram."""
        if 'mean_spectrogram' in event_data:
            spectrogram = event_data['mean_spectrogram']
            time = event_data['time']
            freqs = event_data['frequencies']
            
            im = ax.imshow(10 * np.log10(spectrogram + 1e-20),
                          aspect='auto', origin='lower',
                          extent=[time[0], time[-1], freqs[0], freqs[-1]],
                          cmap='viridis')
            
            # Mark event time
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Time relative to event (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Event-Triggered Spectrogram (n={event_data["n_events"]})')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power (dB)')
    
    def _generate_text_report(self, results: Dict[str, Any], report_path: Path):
        """Generate detailed text report."""
        with open(report_path, 'w') as f:
            f.write("Time-Frequency Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Analysis parameters
            f.write("Analysis Parameters\n")
            f.write("-" * 30 + "\n")
            for key, value in self.metadata.get('parameters', {}).items():
                f.write(f"{key}: {value}\n")
                
            # Results summary
            f.write("\n\nResults Summary\n")
            f.write("-" * 30 + "\n")
            
            # Band powers if available
            if 'band_powers' in results:
                f.write("\nFrequency Band Powers:\n")
                for ch_key, bands in results['band_powers'].items():
                    f.write(f"\n{ch_key}:\n")
                    for band, power in bands.items():
                        f.write(f"  {band}: {power:.2e}\n")
                        
            # Coherence results
            for key, value in results.items():
                if 'coherence' in key and isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    mean_coh = np.mean(value['coherence'])
                    max_coh = np.max(value['coherence'])
                    f.write(f"  Mean coherence: {mean_coh:.3f}\n")
                    f.write(f"  Max coherence: {max_coh:.3f}\n")


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = TimeFrequencyAnalysis()
    
    # Load data
    data = analyzer.load_data('test_data/lfp_recording.nwb')
    
    # Run wavelet analysis
    results = analyzer.run_analysis(
        parameters={
            'method': 'wavelet',
            'wavelet': 'morlet',
            'frequencies': [1, 200],
            'n_frequencies': 100,
            'compute_phase': True,
            'compute_coupling': True
        }
    )
    
    # Generate report
    analyzer.generate_report(output_dir='./results/time_frequency')
