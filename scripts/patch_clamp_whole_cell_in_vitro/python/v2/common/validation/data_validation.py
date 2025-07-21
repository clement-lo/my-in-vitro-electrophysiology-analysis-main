"""
Data Validation Utilities
=========================

Functions for validating electrophysiology data quality, integrity,
and compatibility with analysis requirements.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import signal, stats
from typing import Optional, Union, Tuple, Dict, List, Any
import logging

from ...core.exceptions import ValidationError, DataQualityError, InvalidParameterError

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for electrophysiology recordings.
    
    This class provides methods to check data quality, detect artifacts,
    and ensure data meets requirements for analysis.
    """
    
    def __init__(self,
                 sampling_rate: float,
                 expected_units: Optional[str] = None,
                 min_duration: float = 0.1,
                 max_duration: Optional[float] = None):
        """
        Initialize data validator.
        
        Parameters
        ----------
        sampling_rate : float
            Expected sampling rate in Hz
        expected_units : str, optional
            Expected units (e.g., 'mV', 'pA')
        min_duration : float
            Minimum acceptable duration in seconds
        max_duration : float, optional
            Maximum acceptable duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.expected_units = expected_units
        self.min_duration = min_duration
        self.max_duration = max_duration
        
    def validate(self, data: np.ndarray,
                time: Optional[np.ndarray] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Parameters
        ----------
        data : ndarray
            Signal data to validate
        time : ndarray, optional
            Time vector
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        validation_report : dict
            Detailed validation results
            
        Raises
        ------
        ValidationError
            If critical validation fails
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Basic validation
        self._validate_shape(data, report)
        self._validate_duration(data, report)
        self._validate_data_type(data, report)
        
        # Quality checks
        self._check_missing_data(data, report)
        self._check_data_range(data, report)
        self._check_noise_level(data, report)
        self._check_artifacts(data, report)
        self._check_saturation(data, report)
        
        # Time vector validation if provided
        if time is not None:
            self._validate_time_vector(time, data, report)
            
        # Metadata validation if provided
        if metadata is not None:
            self._validate_metadata(metadata, report)
            
        # Calculate overall quality score
        report['quality_score'] = self._calculate_quality_score(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        # Determine overall validity
        report['valid'] = len(report['errors']) == 0
        
        if not report['valid']:
            error_msg = "; ".join(report['errors'])
            raise ValidationError(f"Data validation failed: {error_msg}")
            
        return report
        
    def _validate_shape(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Validate data shape and dimensions."""
        if data.ndim > 2:
            report['errors'].append(f"Data has too many dimensions ({data.ndim})")
        elif data.ndim == 2:
            n_channels, n_samples = data.shape
            if n_channels > n_samples:
                report['warnings'].append(
                    "Data might be transposed (channels > samples)"
                )
                
        if data.size == 0:
            report['errors'].append("Data is empty")
            
    def _validate_duration(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Validate recording duration."""
        n_samples = data.shape[-1] if data.ndim > 1 else len(data)
        duration = n_samples / self.sampling_rate
        
        report['quality_metrics']['duration'] = duration
        
        if duration < self.min_duration:
            report['errors'].append(
                f"Duration ({duration:.3f}s) below minimum ({self.min_duration}s)"
            )
            
        if self.max_duration and duration > self.max_duration:
            report['warnings'].append(
                f"Duration ({duration:.3f}s) exceeds maximum ({self.max_duration}s)"
            )
            
    def _validate_data_type(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Validate data type and format."""
        if not np.issubdtype(data.dtype, np.number):
            report['errors'].append(f"Data type ({data.dtype}) is not numeric")
            
        if np.issubdtype(data.dtype, np.complexfloating):
            report['warnings'].append("Data contains complex values")
            
    def _check_missing_data(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Check for missing or invalid data."""
        # Check for NaN
        nan_count = np.sum(np.isnan(data))
        nan_percent = 100 * nan_count / data.size
        
        report['quality_metrics']['nan_percent'] = nan_percent
        
        if nan_count > 0:
            if nan_percent > 10:
                report['errors'].append(
                    f"Too many NaN values ({nan_percent:.1f}%)"
                )
            else:
                report['warnings'].append(
                    f"Data contains {nan_count} NaN values ({nan_percent:.1f}%)"
                )
                
        # Check for Inf
        inf_count = np.sum(np.isinf(data))
        if inf_count > 0:
            report['errors'].append(f"Data contains {inf_count} infinite values")
            
    def _check_data_range(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Check if data is in reasonable range."""
        # Remove NaN for statistics
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return
            
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        data_range = data_max - data_min
        
        report['quality_metrics']['data_range'] = data_range
        report['quality_metrics']['data_min'] = data_min
        report['quality_metrics']['data_max'] = data_max
        
        # Check for reasonable ranges based on expected units
        if self.expected_units:
            if self.expected_units == 'mV':
                if data_max > 200 or data_min < -200:
                    report['warnings'].append(
                        f"Data range ({data_min:.1f} to {data_max:.1f} mV) "
                        "seems unusually large"
                    )
            elif self.expected_units == 'pA':
                if data_max > 10000 or data_min < -10000:
                    report['warnings'].append(
                        f"Data range ({data_min:.1f} to {data_max:.1f} pA) "
                        "seems unusually large"
                    )
                    
        # Check for zero range
        if data_range == 0:
            report['errors'].append("Data has zero range (constant values)")
        elif data_range < 1e-10:
            report['warnings'].append("Data range is extremely small")
            
    def _check_noise_level(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Estimate and check noise level."""
        # Use robust MAD estimator for noise
        if data.ndim == 1:
            # High-pass filter to isolate noise
            if len(data) > 100:
                sos = signal.butter(4, 100, 'hp', fs=self.sampling_rate, output='sos')
                filtered = signal.sosfiltfilt(sos, data)
                
                # MAD estimate of noise
                noise_mad = np.median(np.abs(filtered - np.median(filtered)))
                noise_std = 1.4826 * noise_mad  # Convert to std equivalent
                
                # Signal estimate (low frequency)
                sos_lp = signal.butter(4, 10, 'lp', fs=self.sampling_rate, output='sos')
                signal_est = signal.sosfiltfilt(sos_lp, data)
                signal_power = np.std(signal_est)
                
                if signal_power > 0:
                    snr = 20 * np.log10(signal_power / noise_std)
                else:
                    snr = 0
                    
                report['quality_metrics']['noise_std'] = noise_std
                report['quality_metrics']['snr_db'] = snr
                
                if snr < 10:
                    report['warnings'].append(f"Low SNR ({snr:.1f} dB)")
                    
    def _check_artifacts(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Check for common artifacts."""
        if data.ndim == 1:
            # Check for spikes (outliers)
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            spike_count = np.sum(z_scores > 5)
            spike_percent = 100 * spike_count / len(data)
            
            report['quality_metrics']['spike_percent'] = spike_percent
            
            if spike_percent > 5:
                report['warnings'].append(
                    f"High spike artifact rate ({spike_percent:.1f}%)"
                )
                
            # Check for sudden jumps
            diff = np.diff(data)
            diff_std = np.std(diff)
            jump_threshold = 10 * diff_std
            jumps = np.sum(np.abs(diff) > jump_threshold)
            
            if jumps > 10:
                report['warnings'].append(f"Detected {jumps} sudden jumps")
                
            # Check for 50/60 Hz interference
            freqs, psd = signal.welch(data, self.sampling_rate, nperseg=1024)
            
            # Check 50 Hz
            idx_50 = np.argmin(np.abs(freqs - 50))
            if idx_50 > 0 and idx_50 < len(psd) - 1:
                peak_50 = psd[idx_50] / np.mean(psd[idx_50-5:idx_50+5])
                if peak_50 > 10:
                    report['warnings'].append("Strong 50 Hz interference detected")
                    
            # Check 60 Hz
            idx_60 = np.argmin(np.abs(freqs - 60))
            if idx_60 > 0 and idx_60 < len(psd) - 1:
                peak_60 = psd[idx_60] / np.mean(psd[idx_60-5:idx_60+5])
                if peak_60 > 10:
                    report['warnings'].append("Strong 60 Hz interference detected")
                    
    def _check_saturation(self, data: np.ndarray, report: Dict[str, Any]) -> None:
        """Check for amplifier saturation."""
        # Count consecutive identical values (potential saturation)
        if data.ndim == 1:
            # Find runs of identical values
            diff = np.diff(data)
            zero_runs = []
            current_run = 0
            
            for d in diff:
                if d == 0:
                    current_run += 1
                else:
                    if current_run > 5:  # Minimum run length
                        zero_runs.append(current_run)
                    current_run = 0
                    
            if zero_runs:
                max_run = max(zero_runs)
                if max_run > 50:
                    report['warnings'].append(
                        f"Possible saturation detected (max flat region: {max_run} samples)"
                    )
                    
            # Check for values at extremes
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                data_range = np.ptp(valid_data)
                if data_range > 0:
                    extreme_low = np.sum(valid_data == np.min(valid_data))
                    extreme_high = np.sum(valid_data == np.max(valid_data))
                    
                    if extreme_low > len(valid_data) * 0.01:
                        report['warnings'].append(
                            f"Many values at minimum ({extreme_low} samples)"
                        )
                    if extreme_high > len(valid_data) * 0.01:
                        report['warnings'].append(
                            f"Many values at maximum ({extreme_high} samples)"
                        )
                        
    def _validate_time_vector(self, time: np.ndarray, data: np.ndarray,
                            report: Dict[str, Any]) -> None:
        """Validate time vector consistency."""
        # Check length match
        expected_len = data.shape[-1] if data.ndim > 1 else len(data)
        if len(time) != expected_len:
            report['errors'].append(
                f"Time vector length ({len(time)}) doesn't match data ({expected_len})"
            )
            return
            
        # Check monotonicity
        if not np.all(np.diff(time) > 0):
            report['errors'].append("Time vector is not monotonically increasing")
            
        # Check regular sampling
        dt = np.diff(time)
        dt_std = np.std(dt)
        dt_mean = np.mean(dt)
        
        if dt_std / dt_mean > 0.01:  # More than 1% variation
            report['warnings'].append("Time vector has irregular sampling")
            
        # Check sampling rate match
        actual_rate = 1 / dt_mean
        rate_error = abs(actual_rate - self.sampling_rate) / self.sampling_rate
        
        if rate_error > 0.05:  # More than 5% error
            report['warnings'].append(
                f"Actual sampling rate ({actual_rate:.1f} Hz) differs from "
                f"expected ({self.sampling_rate:.1f} Hz)"
            )
            
    def _validate_metadata(self, metadata: Dict[str, Any],
                          report: Dict[str, Any]) -> None:
        """Validate metadata consistency."""
        # Check units if specified
        if self.expected_units and 'units' in metadata:
            if metadata['units'] != self.expected_units:
                report['warnings'].append(
                    f"Metadata units ({metadata['units']}) differ from "
                    f"expected ({self.expected_units})"
                )
                
        # Check sampling rate if specified
        if 'sampling_rate' in metadata:
            meta_rate = metadata['sampling_rate']
            if abs(meta_rate - self.sampling_rate) > 1:
                report['warnings'].append(
                    f"Metadata sampling rate ({meta_rate} Hz) differs from "
                    f"expected ({self.sampling_rate} Hz)"
                )
                
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct for errors (major issues)
        score -= len(report['errors']) * 20
        
        # Deduct for warnings (minor issues)
        score -= len(report['warnings']) * 5
        
        # Deduct based on metrics
        metrics = report['quality_metrics']
        
        if 'nan_percent' in metrics:
            score -= min(metrics['nan_percent'], 20)
            
        if 'snr_db' in metrics:
            if metrics['snr_db'] < 20:
                score -= (20 - metrics['snr_db'])
                
        if 'spike_percent' in metrics:
            score -= min(metrics['spike_percent'], 10)
            
        return max(0, min(100, score))
        
    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Based on warnings and errors
        for warning in report['warnings']:
            if '50 Hz interference' in warning:
                recommendations.append(
                    "Apply notch filter at 50 Hz to remove interference"
                )
            elif '60 Hz interference' in warning:
                recommendations.append(
                    "Apply notch filter at 60 Hz to remove interference"
                )
            elif 'Low SNR' in warning:
                recommendations.append(
                    "Consider filtering or averaging to improve SNR"
                )
            elif 'spike artifact' in warning:
                recommendations.append(
                    "Use artifact removal or spike detection algorithms"
                )
            elif 'NaN values' in warning:
                recommendations.append(
                    "Interpolate or remove segments with missing data"
                )
                
        # Based on quality score
        if report['quality_score'] < 50:
            recommendations.append(
                "Data quality is poor - consider re-recording if possible"
            )
        elif report['quality_score'] < 70:
            recommendations.append(
                "Data quality is marginal - preprocessing recommended"
            )
            
        report['recommendations'] = list(set(recommendations))  # Remove duplicates


def validate_spike_train(spike_times: np.ndarray,
                        recording_duration: float,
                        min_rate: float = 0.1,
                        max_rate: float = 200.0,
                        max_isi_cv: float = 3.0) -> Dict[str, Any]:
    """
    Validate spike train data.
    
    Parameters
    ----------
    spike_times : ndarray
        Spike times in seconds
    recording_duration : float
        Total recording duration
    min_rate : float
        Minimum acceptable firing rate (Hz)
    max_rate : float
        Maximum acceptable firing rate (Hz)
    max_isi_cv : float
        Maximum acceptable ISI coefficient of variation
        
    Returns
    -------
    validation_report : dict
        Validation results
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'metrics': {}
    }
    
    # Check if empty
    if len(spike_times) == 0:
        report['errors'].append("No spikes detected")
        report['valid'] = False
        return report
        
    # Check ordering
    if not np.all(np.diff(spike_times) >= 0):
        report['errors'].append("Spike times are not sorted")
        report['valid'] = False
        
    # Check bounds
    if np.any(spike_times < 0):
        report['errors'].append("Negative spike times found")
        report['valid'] = False
        
    if np.any(spike_times > recording_duration):
        report['errors'].append("Spike times exceed recording duration")
        report['valid'] = False
        
    # Calculate metrics
    n_spikes = len(spike_times)
    firing_rate = n_spikes / recording_duration
    
    report['metrics']['n_spikes'] = n_spikes
    report['metrics']['firing_rate'] = firing_rate
    
    # Check firing rate
    if firing_rate < min_rate:
        report['warnings'].append(
            f"Low firing rate ({firing_rate:.2f} Hz)"
        )
    elif firing_rate > max_rate:
        report['warnings'].append(
            f"Very high firing rate ({firing_rate:.2f} Hz) - possible noise"
        )
        
    # Check ISI statistics
    if n_spikes > 2:
        isis = np.diff(spike_times)
        isi_mean = np.mean(isis)
        isi_std = np.std(isis)
        isi_cv = isi_std / isi_mean if isi_mean > 0 else 0
        
        report['metrics']['isi_mean'] = isi_mean
        report['metrics']['isi_cv'] = isi_cv
        
        if isi_cv > max_isi_cv:
            report['warnings'].append(
                f"High ISI variability (CV={isi_cv:.2f})"
            )
            
        # Check for refractory period violations
        min_isi = np.min(isis)
        if min_isi < 0.001:  # Less than 1 ms
            report['warnings'].append(
                f"Possible refractory period violations (min ISI={min_isi*1000:.2f} ms)"
            )
            
    return report


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example data with various quality issues
    fs = 1000  # Hz
    duration = 10
    t = np.linspace(0, duration, int(fs * duration))
    
    # Good quality signal
    good_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    # Poor quality signal with artifacts
    poor_signal = good_signal.copy()
    # Add NaN values
    poor_signal[100:150] = np.nan
    # Add spikes
    poor_signal[::500] = 10
    # Add saturation
    poor_signal[2000:2100] = 5.0
    # Add 60 Hz noise
    poor_signal += 0.5 * np.sin(2 * np.pi * 60 * t)
    
    # Create validator
    validator = DataValidator(sampling_rate=fs, expected_units='mV',
                            min_duration=0.5)
    
    # Validate good signal
    print("Good Signal Validation:")
    try:
        good_report = validator.validate(good_signal, time=t)
        print(f"  Quality Score: {good_report['quality_score']:.1f}/100")
        print(f"  Warnings: {len(good_report['warnings'])}")
        print(f"  Errors: {len(good_report['errors'])}")
    except ValidationError as e:
        print(f"  Validation failed: {e}")
        
    # Validate poor signal
    print("\nPoor Signal Validation:")
    try:
        poor_report = validator.validate(poor_signal, time=t)
        print(f"  Quality Score: {poor_report['quality_score']:.1f}/100")
        print(f"  Warnings: {len(poor_report['warnings'])}")
        for warning in poor_report['warnings']:
            print(f"    - {warning}")
        print(f"  Recommendations:")
        for rec in poor_report['recommendations']:
            print(f"    - {rec}")
    except ValidationError as e:
        print(f"  Validation failed: {e}")
        
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(t[:1000], good_signal[:1000])
    axes[0].set_title('Good Quality Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (mV)')
    
    axes[1].plot(t[:1000], poor_signal[:1000])
    axes[1].set_title('Poor Quality Signal (with artifacts)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (mV)')
    
    plt.tight_layout()
    plt.show()