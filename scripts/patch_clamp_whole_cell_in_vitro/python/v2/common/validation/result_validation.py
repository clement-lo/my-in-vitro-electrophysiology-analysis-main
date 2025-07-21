"""
Result Validation Utilities
===========================

Functions for validating analysis results, ensuring consistency,
and checking for common analysis errors.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from typing import Optional, Union, Dict, List, Any, Tuple
import logging

from ...core.exceptions import ResultValidationError, InvalidParameterError
from ...core.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class ResultValidator:
    """
    Validator for analysis results to ensure quality and consistency.
    """
    
    def __init__(self,
                 expected_type: Optional[str] = None,
                 parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 required_fields: Optional[List[str]] = None):
        """
        Initialize result validator.
        
        Parameters
        ----------
        expected_type : str, optional
            Expected analysis type
        parameter_bounds : dict, optional
            Expected bounds for parameters {name: (min, max)}
        required_fields : list, optional
            Required fields in results
        """
        self.expected_type = expected_type
        self.parameter_bounds = parameter_bounds or {}
        self.required_fields = required_fields or []
        
    def validate(self, result: Union[AnalysisResult, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate analysis result.
        
        Parameters
        ----------
        result : AnalysisResult or dict
            Analysis result to validate
            
        Returns
        -------
        validation_report : dict
            Validation results
            
        Raises
        ------
        ResultValidationError
            If validation fails
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'checks_passed': [],
            'checks_failed': []
        }
        
        # Convert to dict if AnalysisResult
        if isinstance(result, AnalysisResult):
            result_dict = {
                'analysis_type': result.analysis_type,
                'parameters': result.parameters,
                'metrics': result.metrics,
                'metadata': result.metadata,
                'data': result.data
            }
        else:
            result_dict = result
            
        # Check analysis type
        if self.expected_type:
            self._check_analysis_type(result_dict, report)
            
        # Check required fields
        self._check_required_fields(result_dict, report)
        
        # Check parameter bounds
        self._check_parameter_bounds(result_dict, report)
        
        # Check data consistency
        self._check_data_consistency(result_dict, report)
        
        # Type-specific validation
        if 'analysis_type' in result_dict:
            self._type_specific_validation(result_dict, report)
            
        # Determine overall validity
        report['valid'] = len(report['errors']) == 0
        
        if not report['valid']:
            error_msg = "; ".join(report['errors'])
            raise ResultValidationError(f"Result validation failed: {error_msg}")
            
        return report
        
    def _check_analysis_type(self, result: Dict[str, Any],
                           report: Dict[str, Any]) -> None:
        """Check if analysis type matches expected."""
        if 'analysis_type' in result:
            if result['analysis_type'] != self.expected_type:
                report['errors'].append(
                    f"Analysis type '{result['analysis_type']}' doesn't match "
                    f"expected '{self.expected_type}'"
                )
            else:
                report['checks_passed'].append("Analysis type matches")
        else:
            report['errors'].append("Missing analysis_type field")
            
    def _check_required_fields(self, result: Dict[str, Any],
                              report: Dict[str, Any]) -> None:
        """Check for required fields."""
        for field in self.required_fields:
            if field not in result:
                report['errors'].append(f"Missing required field: {field}")
                report['checks_failed'].append(f"Required field '{field}'")
            else:
                report['checks_passed'].append(f"Required field '{field}' present")
                
    def _check_parameter_bounds(self, result: Dict[str, Any],
                               report: Dict[str, Any]) -> None:
        """Check if parameters are within expected bounds."""
        if 'parameters' not in result:
            return
            
        params = result['parameters']
        
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            if param_name in params:
                value = params[param_name]
                
                if value < min_val or value > max_val:
                    report['warnings'].append(
                        f"Parameter '{param_name}' ({value}) outside expected "
                        f"range [{min_val}, {max_val}]"
                    )
                    report['checks_failed'].append(
                        f"Parameter bounds for '{param_name}'"
                    )
                else:
                    report['checks_passed'].append(
                        f"Parameter '{param_name}' within bounds"
                    )
                    
    def _check_data_consistency(self, result: Dict[str, Any],
                               report: Dict[str, Any]) -> None:
        """Check internal consistency of data."""
        # Check for NaN or Inf in numeric fields
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    report['errors'].append(f"Field '{key}' contains NaN")
                elif np.isinf(value):
                    report['errors'].append(f"Field '{key}' contains Inf")
                    
            elif isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    report['warnings'].append(f"Field '{key}' contains NaN values")
                if np.any(np.isinf(value)):
                    report['warnings'].append(f"Field '{key}' contains Inf values")
                    
    def _type_specific_validation(self, result: Dict[str, Any],
                                report: Dict[str, Any]) -> None:
        """Perform validation specific to analysis type."""
        analysis_type = result.get('analysis_type', '').lower()
        
        if 'spike' in analysis_type:
            self._validate_spike_analysis(result, report)
        elif 'synaptic' in analysis_type:
            self._validate_synaptic_analysis(result, report)
        elif 'action_potential' in analysis_type:
            self._validate_ap_analysis(result, report)
        elif 'spectral' in analysis_type:
            self._validate_spectral_analysis(result, report)
            
    def _validate_spike_analysis(self, result: Dict[str, Any],
                               report: Dict[str, Any]) -> None:
        """Validate spike detection results."""
        if 'data' in result and 'spike_times' in result['data']:
            spike_times = result['data']['spike_times']
            
            # Check ordering
            if len(spike_times) > 1:
                if not np.all(np.diff(spike_times) >= 0):
                    report['errors'].append("Spike times not sorted")
                    
            # Check for duplicates
            unique_times = np.unique(spike_times)
            if len(unique_times) < len(spike_times):
                report['warnings'].append("Duplicate spike times found")
                
        if 'metrics' in result:
            metrics = result['metrics']
            
            # Check firing rate
            if 'firing_rate' in metrics:
                rate = metrics['firing_rate']
                if rate < 0:
                    report['errors'].append("Negative firing rate")
                elif rate > 1000:
                    report['warnings'].append(
                        f"Extremely high firing rate ({rate} Hz)"
                    )
                    
    def _validate_synaptic_analysis(self, result: Dict[str, Any],
                                  report: Dict[str, Any]) -> None:
        """Validate synaptic event analysis results."""
        if 'data' in result and 'event_times' in result['data']:
            event_times = result['data']['event_times']
            
            # Similar checks as spike analysis
            if len(event_times) > 1:
                if not np.all(np.diff(event_times) >= 0):
                    report['errors'].append("Event times not sorted")
                    
        if 'parameters' in result:
            params = result['parameters']
            
            # Check amplitude
            if 'amplitude' in params:
                amp = params['amplitude']
                if amp == 0:
                    report['warnings'].append("Zero amplitude detected")
                    
            # Check kinetics
            if 'rise_time' in params and 'decay_time' in params:
                rise = params['rise_time']
                decay = params['decay_time']
                
                if rise <= 0:
                    report['errors'].append("Invalid rise time (<= 0)")
                if decay <= 0:
                    report['errors'].append("Invalid decay time (<= 0)")
                if decay < rise:
                    report['warnings'].append("Decay time shorter than rise time")
                    
    def _validate_ap_analysis(self, result: Dict[str, Any],
                            report: Dict[str, Any]) -> None:
        """Validate action potential analysis results."""
        if 'parameters' in result:
            params = result['parameters']
            
            # Check threshold
            if 'threshold' in params and 'peak' in params:
                threshold = params['threshold']
                peak = params['peak']
                
                if peak <= threshold:
                    report['errors'].append("Peak not above threshold")
                    
            # Check widths
            if 'half_width' in params:
                hw = params['half_width']
                if hw <= 0:
                    report['errors'].append("Invalid half-width (<= 0)")
                elif hw > 0.01:  # > 10 ms
                    report['warnings'].append(
                        f"Unusually wide action potential ({hw*1000:.1f} ms)"
                    )
                    
    def _validate_spectral_analysis(self, result: Dict[str, Any],
                                  report: Dict[str, Any]) -> None:
        """Validate spectral analysis results."""
        if 'data' in result:
            data = result['data']
            
            # Check frequency array
            if 'frequencies' in data:
                freqs = data['frequencies']
                
                if len(freqs) > 1:
                    if not np.all(np.diff(freqs) > 0):
                        report['errors'].append("Frequencies not monotonic")
                        
                if np.any(freqs < 0):
                    report['errors'].append("Negative frequencies found")
                    
            # Check power values
            if 'power' in data:
                power = data['power']
                
                if np.any(power < 0):
                    report['warnings'].append("Negative power values found")
                    
                if np.all(power == 0):
                    report['warnings'].append("All power values are zero")


def validate_parameter_consistency(results: List[Dict[str, Any]],
                                 parameter_name: str,
                                 max_cv: float = 0.5,
                                 outlier_method: str = 'iqr') -> Dict[str, Any]:
    """
    Check consistency of a parameter across multiple results.
    
    Parameters
    ----------
    results : list of dict
        List of analysis results
    parameter_name : str
        Name of parameter to check
    max_cv : float
        Maximum acceptable coefficient of variation
    outlier_method : str
        Method for outlier detection
        
    Returns
    -------
    consistency_report : dict
        Consistency analysis results
    """
    # Extract parameter values
    values = []
    for i, result in enumerate(results):
        if 'parameters' in result and parameter_name in result['parameters']:
            values.append(result['parameters'][parameter_name])
        else:
            logger.warning(f"Result {i} missing parameter '{parameter_name}'")
            
    if len(values) < 2:
        return {
            'consistent': True,
            'n_values': len(values),
            'message': "Not enough values to assess consistency"
        }
        
    values = np.array(values)
    
    # Calculate statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val != 0 else 0
    
    # Detect outliers
    from ..statistics.descriptive import detect_outliers
    outlier_mask = detect_outliers(values, method=outlier_method)
    n_outliers = np.sum(outlier_mask)
    
    # Assess consistency
    consistent = cv <= max_cv and n_outliers == 0
    
    return {
        'consistent': consistent,
        'n_values': len(values),
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'n_outliers': n_outliers,
        'outlier_indices': np.where(outlier_mask)[0].tolist(),
        'message': f"{'Consistent' if consistent else 'Inconsistent'} "
                   f"(CV={cv:.2f}, outliers={n_outliers})"
    }


def validate_time_series_results(times: np.ndarray,
                               values: np.ndarray,
                               expected_duration: Optional[float] = None,
                               expected_rate: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate time series analysis results.
    
    Parameters
    ----------
    times : ndarray
        Time points
    values : ndarray
        Corresponding values
    expected_duration : float, optional
        Expected total duration
    expected_rate : float, optional
        Expected sampling/event rate
        
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
    
    # Check lengths match
    if len(times) != len(values):
        report['errors'].append(
            f"Length mismatch: times ({len(times)}) vs values ({len(values)})"
        )
        report['valid'] = False
        return report
        
    # Check time ordering
    if len(times) > 1:
        if not np.all(np.diff(times) >= 0):
            report['errors'].append("Times not monotonically increasing")
            report['valid'] = False
            
        # Check for duplicates
        unique_times = np.unique(times)
        if len(unique_times) < len(times):
            report['warnings'].append(
                f"{len(times) - len(unique_times)} duplicate time points"
            )
            
    # Check duration
    if len(times) > 0:
        actual_duration = times[-1] - times[0]
        report['metrics']['duration'] = actual_duration
        
        if expected_duration is not None:
            duration_error = abs(actual_duration - expected_duration) / expected_duration
            if duration_error > 0.1:  # More than 10% error
                report['warnings'].append(
                    f"Duration mismatch: {actual_duration:.2f} vs "
                    f"expected {expected_duration:.2f}"
                )
                
    # Check rate
    if len(times) > 1:
        # Calculate actual rate
        if expected_rate is not None:
            # For regular sampling
            dt = np.diff(times)
            actual_rate = 1 / np.mean(dt)
            report['metrics']['sampling_rate'] = actual_rate
            
            rate_error = abs(actual_rate - expected_rate) / expected_rate
            if rate_error > 0.05:  # More than 5% error
                report['warnings'].append(
                    f"Rate mismatch: {actual_rate:.2f} Hz vs "
                    f"expected {expected_rate:.2f} Hz"
                )
                
    # Check for NaN/Inf in values
    if np.any(np.isnan(values)):
        report['warnings'].append(f"{np.sum(np.isnan(values))} NaN values found")
    if np.any(np.isinf(values)):
        report['errors'].append(f"{np.sum(np.isinf(values))} Inf values found")
        report['valid'] = False
        
    return report


def cross_validate_results(result1: Dict[str, Any],
                         result2: Dict[str, Any],
                         tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Cross-validate two analysis results for consistency.
    
    Parameters
    ----------
    result1 : dict
        First analysis result
    result2 : dict
        Second analysis result
    tolerance : float
        Relative tolerance for parameter comparison
        
    Returns
    -------
    cross_validation_report : dict
        Cross-validation results
    """
    report = {
        'consistent': True,
        'matches': [],
        'mismatches': [],
        'missing': []
    }
    
    # Check analysis types match
    type1 = result1.get('analysis_type', 'unknown')
    type2 = result2.get('analysis_type', 'unknown')
    
    if type1 != type2:
        report['mismatches'].append({
            'field': 'analysis_type',
            'value1': type1,
            'value2': type2
        })
        report['consistent'] = False
        
    # Compare parameters
    if 'parameters' in result1 and 'parameters' in result2:
        params1 = result1['parameters']
        params2 = result2['parameters']
        
        # Find common parameters
        common_params = set(params1.keys()) & set(params2.keys())
        
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric comparison with tolerance
                rel_diff = abs(val1 - val2) / (abs(val1) + 1e-10)
                
                if rel_diff <= tolerance:
                    report['matches'].append({
                        'parameter': param,
                        'value1': val1,
                        'value2': val2,
                        'rel_diff': rel_diff
                    })
                else:
                    report['mismatches'].append({
                        'parameter': param,
                        'value1': val1,
                        'value2': val2,
                        'rel_diff': rel_diff
                    })
                    report['consistent'] = False
            else:
                # Direct comparison for non-numeric
                if val1 == val2:
                    report['matches'].append({
                        'parameter': param,
                        'value': val1
                    })
                else:
                    report['mismatches'].append({
                        'parameter': param,
                        'value1': val1,
                        'value2': val2
                    })
                    report['consistent'] = False
                    
        # Find missing parameters
        only_in_1 = set(params1.keys()) - set(params2.keys())
        only_in_2 = set(params2.keys()) - set(params1.keys())
        
        for param in only_in_1:
            report['missing'].append({
                'parameter': param,
                'missing_from': 'result2',
                'value': params1[param]
            })
            
        for param in only_in_2:
            report['missing'].append({
                'parameter': param,
                'missing_from': 'result1',
                'value': params2[param]
            })
            
    return report


if __name__ == "__main__":
    # Example usage
    
    # Create example results
    good_result = {
        'analysis_type': 'spike_detection',
        'parameters': {
            'threshold': -20.0,
            'refractory_period': 0.002
        },
        'metrics': {
            'n_spikes': 150,
            'firing_rate': 15.0,
            'mean_amplitude': -45.2
        },
        'data': {
            'spike_times': np.sort(np.random.uniform(0, 10, 150))
        }
    }
    
    bad_result = {
        'analysis_type': 'spike_detection',
        'parameters': {
            'threshold': -20.0,
            'refractory_period': -0.001  # Invalid negative value
        },
        'metrics': {
            'n_spikes': 150,
            'firing_rate': -5.0,  # Invalid negative rate
            'mean_amplitude': np.nan  # NaN value
        },
        'data': {
            'spike_times': np.random.uniform(0, 10, 150)  # Unsorted
        }
    }
    
    # Create validator
    validator = ResultValidator(
        expected_type='spike_detection',
        parameter_bounds={
            'threshold': (-100, 0),
            'refractory_period': (0, 0.01)
        },
        required_fields=['analysis_type', 'parameters', 'metrics']
    )
    
    # Validate good result
    print("Good Result Validation:")
    try:
        good_report = validator.validate(good_result)
        print(f"  Valid: {good_report['valid']}")
        print(f"  Checks passed: {len(good_report['checks_passed'])}")
    except ResultValidationError as e:
        print(f"  Validation failed: {e}")
        
    # Validate bad result
    print("\nBad Result Validation:")
    try:
        bad_report = validator.validate(bad_result)
        print(f"  Valid: {bad_report['valid']}")
    except ResultValidationError as e:
        print(f"  Validation failed: {e}")
        
    # Test parameter consistency
    print("\nParameter Consistency Check:")
    results = [
        {'parameters': {'threshold': -20.0}},
        {'parameters': {'threshold': -21.0}},
        {'parameters': {'threshold': -19.5}},
        {'parameters': {'threshold': -40.0}}  # Outlier
    ]
    
    consistency = validate_parameter_consistency(results, 'threshold')
    print(f"  Consistent: {consistency['consistent']}")
    print(f"  CV: {consistency['cv']:.3f}")
    print(f"  Outliers: {consistency['n_outliers']}")
    
    # Cross-validation
    print("\nCross-validation:")
    result2 = good_result.copy()
    result2['parameters']['threshold'] = -21.0  # Slight difference
    
    cross_val = cross_validate_results(good_result, result2, tolerance=0.1)
    print(f"  Consistent: {cross_val['consistent']}")
    print(f"  Matches: {len(cross_val['matches'])}")
    print(f"  Mismatches: {len(cross_val['mismatches'])}")