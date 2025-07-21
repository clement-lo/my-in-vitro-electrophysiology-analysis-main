"""
Pharmacological Analysis Module v2
Enhanced dose-response analysis with multi-drug interactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import ttest_ind, f_oneway
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import logging
import warnings

from ..core.base_analysis import AbstractAnalysis, ValidationMixin
from ..core.data_pipeline import load_data
from ..core.config_manager import ConfigManager
from ..utils.visualization import create_multi_panel_figure, save_figure

logger = logging.getLogger(__name__)

class PharmacologicalAnalysis(AbstractAnalysis, ValidationMixin):
    """
    Advanced pharmacological analysis for electrophysiology data.
    
    Features:
    - Dose-response curve fitting (Hill equation)
    - IC50/EC50 determination with confidence intervals
    - Multi-drug interaction analysis
    - Time-dependent drug effects
    - Competitive/non-competitive inhibition models
    """
    
    def __init__(self, config: Union[Dict, ConfigManager, Path, str, None] = None):
        super().__init__(name="pharmacological_analysis", version="2.0")
        
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
            
        self.config = self.config_manager.create_analysis_config('pharmacology')
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load dose-response or time-series pharmacology data."""
        
        # Check if it's a dose-response dataset or time-series
        if kwargs.get('data_type') == 'dose_response':
            # Load from CSV or similar format
            df = pd.read_csv(file_path)
            
            self.data = {
                'concentrations': df['concentration'].values,
                'responses': df['response'].values,
                'errors': df.get('error', np.zeros_like(df['response'])).values,
                'n_replicates': df.get('n', np.ones_like(df['response'])).values,
                'drug_name': kwargs.get('drug_name', 'Unknown'),
                'data_type': 'dose_response'
            }
        else:
            # Load time-series data
            signal, rate, time, metadata = load_data(file_path, **kwargs)
            
            self.data = {
                'signal': signal,
                'time': time,
                'sampling_rate': rate,
                'metadata': metadata,
                'data_type': 'time_series',
                'drug_application_times': kwargs.get('drug_times', [])
            }
            
        return self.data
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate pharmacological analysis parameters."""
        
        if parameters.get('analysis_type') == 'dose_response':
            # Validate dose-response parameters
            if 'fit_type' in parameters:
                parameters['fit_type'] = self.validate_choice_parameter(
                    parameters['fit_type'],
                    ['hill', 'logistic', 'exponential'],
                    'fit_type'
                )
                
            if 'ec50_bounds' in parameters:
                bounds = parameters['ec50_bounds']
                if len(bounds) != 2 or bounds[0] >= bounds[1]:
                    raise ValueError("ec50_bounds must be [min, max] with min < max")
                    
        elif parameters.get('analysis_type') == 'time_dependent':
            # Validate time-dependent parameters
            required = ['baseline_period', 'drug_period', 'washout_period']
            for param in required:
                if param not in parameters:
                    parameters[param] = [0, 60]  # Default 1-minute periods
                    
        return True
    
    def run_analysis(self, data: Any = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run pharmacological analysis."""
        
        if data is None:
            data = self.data
            
        if parameters is None:
            parameters = self.config.get('pharmacology', {})
            
        self.validate_parameters(parameters)
        self.metadata['parameters'] = parameters
        
        results = {}
        
        analysis_type = parameters.get('analysis_type', 'dose_response')
        
        if analysis_type == 'dose_response':
            results['dose_response'] = self._analyze_dose_response(data, parameters)
            
        elif analysis_type == 'time_dependent':
            results['time_dependent'] = self._analyze_time_dependent(data, parameters)
            
        elif analysis_type == 'multi_drug':
            results['interactions'] = self._analyze_drug_interactions(data, parameters)
            
        elif analysis_type == 'inhibition':
            results['inhibition'] = self._analyze_inhibition_type(data, parameters)
            
        # Calculate statistics if requested
        if parameters.get('calculate_statistics', True):
            results['statistics'] = self._calculate_statistics(data, results)
            
        self.results = results
        return results
    
    def _analyze_dose_response(self, data: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dose-response relationship."""
        
        concentrations = data['concentrations']
        responses = data['responses']
        errors = data.get('errors', np.zeros_like(responses))
        
        # Log-transform concentrations
        log_conc = np.log10(concentrations + 1e-12)  # Avoid log(0)
        
        # Fit Hill equation
        if parameters.get('fit_type', 'hill') == 'hill':
            fit_result = self._fit_hill_equation(log_conc, responses, errors)
        else:
            # Other fit types...
            fit_result = {}
            
        # Calculate derived parameters
        if fit_result:
            fit_result['potency'] = self._calculate_potency(fit_result)
            fit_result['efficacy'] = fit_result['top'] - fit_result['bottom']
            fit_result['hill_coefficient'] = fit_result['slope']
            
            # Bootstrap confidence intervals
            if parameters.get('bootstrap_ci', True):
                ci_results = self._bootstrap_confidence_intervals(
                    log_conc, responses, errors, 
                    n_iterations=parameters.get('bootstrap_iterations', 1000)
                )
                fit_result['confidence_intervals'] = ci_results
        
        return fit_result
    
    def _fit_hill_equation(self, log_conc: np.ndarray, responses: np.ndarray,
                          errors: np.ndarray) -> Dict[str, Any]:
        """Fit four-parameter Hill equation."""
        
        def hill_equation(x, bottom, top, ec50, slope):
            return bottom + (top - bottom) / (1 + 10**((ec50 - x) * slope))
        
        # Initial parameter estimates
        bottom_init = np.min(responses)
        top_init = np.max(responses)
        ec50_init = log_conc[np.argmin(np.abs(responses - (bottom_init + top_init)/2))]
        slope_init = 1.0
        
        try:
            # Weighted least squares if errors provided
            if np.any(errors > 0):
                sigma = errors
                sigma[sigma == 0] = 1.0
            else:
                sigma = None
                
            popt, pcov = curve_fit(
                hill_equation, log_conc, responses,
                p0=[bottom_init, top_init, ec50_init, slope_init],
                sigma=sigma,
                bounds=([0, 0, np.min(log_conc)-2, 0.1],
                        [np.inf, np.inf, np.max(log_conc)+2, 10]),
                maxfev=5000
            )
            
            # Calculate R-squared
            fitted = hill_equation(log_conc, *popt)
            ss_res = np.sum((responses - fitted)**2)
            ss_tot = np.sum((responses - np.mean(responses))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate standard errors
            perr = np.sqrt(np.diag(pcov))
            
            result = {
                'bottom': popt[0],
                'top': popt[1],
                'log_ec50': popt[2],
                'ec50': 10**popt[2],
                'slope': popt[3],
                'covariance': pcov,
                'std_errors': {
                    'bottom': perr[0],
                    'top': perr[1],
                    'log_ec50': perr[2],
                    'slope': perr[3]
                },
                'r_squared': r_squared,
                'fit_function': lambda x: hill_equation(x, *popt),
                'parameters': popt
            }
            
            logger.info(f"Hill fit successful: EC50={result['ec50']:.3e}, Hill={result['slope']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Hill equation fitting failed: {e}")
            return {}
    
    def _calculate_potency(self, fit_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various potency measures."""
        
        ec50 = fit_result['ec50']
        
        # pEC50 = -log10(EC50)
        pec50 = -np.log10(ec50)
        
        # Calculate EC20 and EC80
        bottom = fit_result['bottom']
        top = fit_result['top']
        slope = fit_result['slope']
        
        ec20_response = bottom + 0.2 * (top - bottom)
        ec80_response = bottom + 0.8 * (top - bottom)
        
        # Solve for concentrations
        ec20 = ec50 * ((0.2 / 0.8) ** (1 / slope))
        ec80 = ec50 * ((0.8 / 0.2) ** (1 / slope))
        
        return {
            'pEC50': pec50,
            'EC20': ec20,
            'EC80': ec80,
            'dynamic_range': np.log10(ec80 / ec20)
        }
    
    def _bootstrap_confidence_intervals(self, log_conc: np.ndarray, 
                                      responses: np.ndarray,
                                      errors: np.ndarray,
                                      n_iterations: int = 1000) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using bootstrap."""
        
        n_points = len(responses)
        bootstrap_params = []
        
        for _ in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(n_points, n_points, replace=True)
            boot_conc = log_conc[indices]
            boot_resp = responses[indices]
            boot_err = errors[indices] if len(errors) > 0 else None
            
            # Fit model
            try:
                fit = self._fit_hill_equation(boot_conc, boot_resp, boot_err)
                if fit:
                    bootstrap_params.append([
                        fit['bottom'], fit['top'], fit['ec50'], fit['slope']
                    ])
            except:
                continue
        
        if len(bootstrap_params) > 100:  # Need sufficient successful fits
            bootstrap_params = np.array(bootstrap_params)
            
            # Calculate 95% CI
            ci_low = np.percentile(bootstrap_params, 2.5, axis=0)
            ci_high = np.percentile(bootstrap_params, 97.5, axis=0)
            
            return {
                'bottom': (ci_low[0], ci_high[0]),
                'top': (ci_low[1], ci_high[1]),
                'ec50': (ci_low[2], ci_high[2]),
                'slope': (ci_low[3], ci_high[3])
            }
        else:
            logger.warning("Insufficient bootstrap iterations succeeded")
            return {}
    
    def _analyze_time_dependent(self, data: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time-dependent drug effects."""
        
        signal = data['signal']
        time = data['time']
        rate = data['sampling_rate']
        
        # Define periods
        baseline_period = parameters['baseline_period']
        drug_period = parameters['drug_period']
        washout_period = parameters.get('washout_period', [])
        
        # Extract data for each period
        baseline_mask = (time >= baseline_period[0]) & (time < baseline_period[1])
        drug_mask = (time >= drug_period[0]) & (time < drug_period[1])
        
        baseline_data = signal[baseline_mask]
        drug_data = signal[drug_mask]
        
        results = {
            'baseline_mean': np.mean(baseline_data),
            'baseline_std': np.std(baseline_data),
            'drug_mean': np.mean(drug_data),
            'drug_std': np.std(drug_data),
            'percent_change': 100 * (np.mean(drug_data) - np.mean(baseline_data)) / np.mean(baseline_data)
        }
        
        # Statistical test
        if len(baseline_data) > 30 and len(drug_data) > 30:
            t_stat, p_value = ttest_ind(baseline_data, drug_data)
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            results['significant'] = p_value < 0.05
        
        # Analyze washout if provided
        if washout_period and len(washout_period) == 2:
            washout_mask = (time >= washout_period[0]) & (time < washout_period[1])
            washout_data = signal[washout_mask]
            
            results['washout_mean'] = np.mean(washout_data)
            results['washout_std'] = np.std(washout_data)
            results['percent_recovery'] = 100 * (np.mean(washout_data) - np.mean(drug_data)) / (np.mean(baseline_data) - np.mean(drug_data))
        
        # Time course analysis
        if parameters.get('analyze_time_course', True):
            results['time_course'] = self._analyze_effect_time_course(
                signal, time, drug_period[0]
            )
        
        return results
    
    def _analyze_effect_time_course(self, signal: np.ndarray, time: np.ndarray,
                                   drug_onset_time: float) -> Dict[str, Any]:
        """Analyze the time course of drug effect."""
        
        # Calculate running average
        window_size = int(len(signal) * 0.01)  # 1% of data
        running_mean = pd.Series(signal).rolling(window_size, center=True).mean().values
        
        # Find onset time (when effect starts)
        baseline_end = int(drug_onset_time * len(time) / time[-1])
        baseline_mean = np.mean(running_mean[:baseline_end])
        baseline_std = np.std(running_mean[:baseline_end])
        
        # Detect significant change
        threshold = baseline_mean + 3 * baseline_std
        onset_idx = np.where(running_mean[baseline_end:] > threshold)[0]
        
        if len(onset_idx) > 0:
            onset_time = time[baseline_end + onset_idx[0]]
            onset_latency = onset_time - drug_onset_time
        else:
            onset_time = None
            onset_latency = None
        
        # Find peak effect
        peak_idx = np.argmax(np.abs(running_mean - baseline_mean))
        peak_time = time[peak_idx]
        peak_value = running_mean[peak_idx]
        
        return {
            'running_mean': running_mean,
            'onset_time': onset_time,
            'onset_latency': onset_latency,
            'peak_time': peak_time,
            'peak_value': peak_value,
            'peak_effect': peak_value - baseline_mean
        }
    
    def _analyze_drug_interactions(self, data: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interactions between multiple drugs."""
        
        # This would implement isobologram analysis, Loewe additivity, etc.
        # For now, return placeholder
        
        results = {
            'interaction_type': 'Not implemented',
            'combination_index': None,
            'synergy_score': None
        }
        
        logger.warning("Multi-drug interaction analysis not yet implemented")
        
        return results
    
    def _analyze_inhibition_type(self, data: Dict[str, Any], 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Determine type of inhibition (competitive vs non-competitive)."""
        
        # This would implement Schild analysis, double-reciprocal plots, etc.
        # For now, return placeholder
        
        results = {
            'inhibition_type': 'Not determined',
            'ki': None,
            'schild_slope': None
        }
        
        logger.warning("Inhibition type analysis not yet implemented")
        
        return results
    
    def _calculate_statistics(self, data: Dict[str, Any], 
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical measures for the analysis."""
        
        stats = {}
        
        if 'dose_response' in results and results['dose_response']:
            dr = results['dose_response']
            
            # Quality metrics
            stats['r_squared'] = dr.get('r_squared', 0)
            stats['n_concentrations'] = len(data.get('concentrations', []))
            
            # Parameter statistics
            if 'std_errors' in dr:
                se = dr['std_errors']
                stats['ec50_cv'] = se['log_ec50'] / np.abs(dr['log_ec50']) * 100
                stats['slope_cv'] = se['slope'] / np.abs(dr['slope']) * 100
        
        return stats
    
    def generate_report(self, results: Dict[str, Any] = None, 
                       output_dir: Union[str, Path] = None) -> Path:
        """Generate comprehensive pharmacological analysis report."""
        
        if results is None:
            results = self.results
            
        if output_dir is None:
            output_dir = Path('./results')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multi-panel figure
        n_panels = sum([
            'dose_response' in results,
            'time_dependent' in results,
            'statistics' in results,
            True  # Summary panel
        ])
        
        fig, axes = create_multi_panel_figure(n_panels, fig_size=(12, 3*n_panels))
        panel_idx = 0
        
        # Dose-response panel
        if 'dose_response' in results and results['dose_response']:
            ax = axes[panel_idx]
            self._plot_dose_response(ax, self.data, results['dose_response'])
            panel_idx += 1
        
        # Time-dependent panel
        if 'time_dependent' in results:
            ax = axes[panel_idx]
            self._plot_time_dependent(ax, self.data, results['time_dependent'])
            panel_idx += 1
        
        # Statistics panel
        if 'statistics' in results:
            ax = axes[panel_idx]
            self._plot_statistics(ax, results['statistics'])
            panel_idx += 1
        
        # Summary panel
        ax = axes[panel_idx]
        self._plot_summary(ax, results)
        
        # Overall title
        drug_name = self.data.get('drug_name', 'Unknown Drug')
        fig.suptitle(f'Pharmacological Analysis: {drug_name}', fontsize=16)
        
        # Save figure
        fig_path = output_dir / 'pharmacological_analysis.png'
        save_figure(fig, fig_path)
        
        # Generate text report
        report_path = output_dir / 'pharmacological_report.txt'
        self._generate_text_report(results, report_path)
        
        # Export results
        self.export_results(output_dir / 'pharmacological_results', format='json')
        
        logger.info(f"Pharmacological analysis report saved to {output_dir}")
        
        return report_path
    
    def _plot_dose_response(self, ax, data, dr_results):
        """Plot dose-response curve."""
        
        conc = data['concentrations']
        resp = data['responses']
        errors = data.get('errors', np.zeros_like(resp))
        
        # Plot data points
        ax.errorbar(conc, resp, yerr=errors, fmt='o', capsize=5, 
                   label='Data', markersize=8)
        
        # Plot fitted curve
        if dr_results and 'fit_function' in dr_results:
            conc_fit = np.logspace(np.log10(np.min(conc))-1, 
                                  np.log10(np.max(conc))+1, 200)
            resp_fit = dr_results['fit_function'](np.log10(conc_fit))
            ax.plot(conc_fit, resp_fit, 'r-', linewidth=2, label='Hill fit')
            
            # Add EC50 line
            ec50 = dr_results['ec50']
            ax.axvline(x=ec50, color='green', linestyle='--', alpha=0.5)
            ax.text(ec50, ax.get_ylim()[0], f'EC50={ec50:.2e}', 
                   rotation=90, va='bottom', ha='right')
        
        ax.set_xscale('log')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Response')
        ax.set_title('Dose-Response Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_time_dependent(self, ax, data, td_results):
        """Plot time-dependent effects."""
        
        if 'time_course' in td_results and td_results['time_course']:
            tc = td_results['time_course']
            time = data['time']
            
            # Plot running mean
            ax.plot(time, tc['running_mean'], 'b-', linewidth=1, 
                   label='Response')
            
            # Mark drug application
            if tc['onset_time']:
                ax.axvline(x=tc['onset_time'], color='red', 
                          linestyle='--', label='Drug onset')
            
            # Mark peak
            if tc['peak_time']:
                ax.scatter([tc['peak_time']], [tc['peak_value']], 
                          color='orange', s=100, marker='*', 
                          label='Peak effect', zorder=5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Response')
            ax.set_title('Time-Dependent Drug Effect')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_statistics(self, ax, stats):
        """Plot statistical summary."""
        ax.axis('off')
        
        text = "Statistical Summary\n" + "=" * 30 + "\n\n"
        
        for key, value in stats.items():
            if isinstance(value, float):
                text += f"{key}: {value:.3f}\n"
            else:
                text += f"{key}: {value}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
               fontsize=10, va='top', family='monospace')
    
    def _plot_summary(self, ax, results):
        """Plot analysis summary."""
        ax.axis('off')
        
        summary = []
        
        if 'dose_response' in results and results['dose_response']:
            dr = results['dose_response']
            summary.append(f"EC50: {dr['ec50']:.3e}")
            summary.append(f"Hill coefficient: {dr['slope']:.2f}")
            summary.append(f"Efficacy: {dr.get('efficacy', 0):.1f}")
        
        if 'time_dependent' in results:
            td = results['time_dependent']
            summary.append(f"Effect: {td['percent_change']:.1f}%")
            if 'p_value' in td:
                summary.append(f"p-value: {td['p_value']:.4f}")
        
        text = "Key Results\n" + "-" * 20 + "\n" + "\n".join(summary)
        
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
               ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _generate_text_report(self, results, report_path):
        """Generate detailed text report."""
        
        with open(report_path, 'w') as f:
            f.write("Pharmacological Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Drug information
            f.write(f"Drug: {self.data.get('drug_name', 'Unknown')}\n")
            f.write(f"Analysis date: {self.metadata.get('created_at', 'Unknown')}\n\n")
            
            # Results sections
            if 'dose_response' in results and results['dose_response']:
                f.write("Dose-Response Analysis\n")
                f.write("-" * 30 + "\n")
                dr = results['dose_response']
                f.write(f"EC50: {dr['ec50']:.3e}\n")
                f.write(f"pEC50: {dr.get('potency', {}).get('pEC50', 0):.2f}\n")
                f.write(f"Hill coefficient: {dr['slope']:.2f}\n")
                f.write(f"Bottom: {dr['bottom']:.2f}\n")
                f.write(f"Top: {dr['top']:.2f}\n")
                f.write(f"R²: {dr.get('r_squared', 0):.3f}\n\n")
            
            # Add other sections...


# Example usage
if __name__ == "__main__":
    # Example dose-response analysis
    analyzer = PharmacologicalAnalysis()
    
    # Create sample dose-response data
    concentrations = np.logspace(-9, -3, 8)  # 1nM to 1mM
    responses = [5, 12, 25, 50, 75, 88, 95, 98]  # Percent response
    errors = [2, 3, 4, 5, 4, 3, 2, 2]
    
    # Save as CSV for testing
    df = pd.DataFrame({
        'concentration': concentrations,
        'response': responses,
        'error': errors
    })
    df.to_csv('test_dose_response.csv', index=False)
    
    # Load and analyze
    data = analyzer.load_data('test_dose_response.csv', 
                             data_type='dose_response',
                             drug_name='Test Drug')
    
    results = analyzer.run_analysis(
        parameters={
            'analysis_type': 'dose_response',
            'bootstrap_ci': True,
            'bootstrap_iterations': 100
        }
    )
    
    # Generate report
    analyzer.generate_report(output_dir='./pharm_results')
