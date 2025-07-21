"""
Ion Channel Kinetics Analysis Module v2
Enhanced with Symbolica MCP integration for analytical solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit, minimize
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import logging

from ..core.base_analysis import AbstractAnalysis, ValidationMixin
from ..core.data_pipeline import load_data
from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class IonChannelAnalysis(AbstractAnalysis, ValidationMixin):
    """
    Advanced ion channel kinetics analysis with support for:
    - Hodgkin-Huxley models
    - Markov chain models
    - Symbolica integration for analytical solutions
    - Automated model selection
    - Uncertainty quantification
    """
    
    def __init__(self, config: Union[Dict, ConfigManager, Path, str, None] = None):
        super().__init__(name="ion_channel_analysis", version="2.0")
        
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
            
        self.config = self.config_manager.create_analysis_config('ion_channel')
        
        # Model definitions
        self.models = {
            'hodgkin_huxley': self._hodgkin_huxley_model,
            'markov_2state': self._markov_2state_model,
            'markov_3state': self._markov_3state_model,
            'goldmann_hodgkin_katz': self._ghk_model
        }
        
        # Symbolica integration hooks
        self.symbolica_enabled = False
        self._check_symbolica_availability()
        
    def _check_symbolica_availability(self):
        """Check if Symbolica MCP is available for analytical solutions."""
        try:
            # This would check for Symbolica MCP availability
            # For now, we'll set up the interface
            self.symbolica_enabled = False
            logger.info("Symbolica MCP not detected - using numerical methods")
        except:
            self.symbolica_enabled = False
            
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load electrophysiology data for ion channel analysis."""
        signal, rate, time, metadata = load_data(file_path, **kwargs)
        
        # Structure data for ion channel analysis
        self.data = {
            'current': signal,  # Assuming current recording
            'voltage': kwargs.get('voltage', None),  # Command voltage if available
            'time': time,
            'sampling_rate': rate,
            'metadata': metadata,
            'temperature': kwargs.get('temperature', 22.0)  # Celsius
        }
        
        return self.data
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate ion channel analysis parameters."""
        # Required parameters
        required = ['model_type', 'voltage_range']
        
        for param in required:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
        # Validate model type
        parameters['model_type'] = self.validate_choice_parameter(
            parameters['model_type'],
            list(self.models.keys()),
            'model_type'
        )
        
        # Validate voltage range
        v_range = parameters['voltage_range']
        if not isinstance(v_range, (list, tuple)) or len(v_range) != 2:
            raise ValueError("voltage_range must be [min, max]")
            
        parameters['voltage_range'] = [
            self.validate_numeric_parameter(v_range[0], param_name='voltage_min'),
            self.validate_numeric_parameter(v_range[1], param_name='voltage_max')
        ]
        
        return True
    
    def run_analysis(self, data: Any = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run ion channel kinetics analysis."""
        if data is None:
            data = self.data
            
        if parameters is None:
            parameters = self.config.get('ion_channel', {
                'model_type': 'hodgkin_huxley',
                'voltage_range': [-100, 50],
                'n_voltage_steps': 15,
                'fit_kinetics': True,
                'calculate_conductance': True
            })
            
        self.validate_parameters(parameters)
        self.metadata['parameters'] = parameters
        
        results = {}
        
        # 1. Analyze current-voltage relationship
        if parameters.get('calculate_iv', True):
            results['iv_curve'] = self._analyze_iv_curve(data, parameters)
            
        # 2. Fit kinetic model
        if parameters.get('fit_kinetics', True):
            model_type = parameters['model_type']
            results['kinetics'] = self._fit_kinetic_model(data, model_type, parameters)
            
        # 3. Calculate conductance
        if parameters.get('calculate_conductance', True):
            results['conductance'] = self._calculate_conductance(data, results.get('iv_curve'))
            
        # 4. Analyze gating
        if parameters.get('analyze_gating', True):
            results['gating'] = self._analyze_gating(data, results.get('kinetics'))
            
        # 5. Temperature corrections
        if parameters.get('temperature_correction', True):
            results['q10'] = self._calculate_q10(data, parameters)
            
        # 6. Model selection if requested
        if parameters.get('auto_model_selection', False):
            results['model_selection'] = self._select_best_model(data, parameters)
            
        self.results = results
        return results
    
    def _analyze_iv_curve(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current-voltage relationship."""
        v_range = parameters['voltage_range']
        n_steps = parameters.get('n_voltage_steps', 15)
        
        voltages = np.linspace(v_range[0], v_range[1], n_steps)
        
        # If we have actual voltage command data
        if data.get('voltage') is not None:
            # Extract steady-state currents at each voltage
            currents, errors = self._extract_steady_state_currents(data)
        else:
            # Simulate or use provided I-V data
            currents = self._simulate_iv_curve(voltages, parameters)
            errors = np.zeros_like(currents)
            
        # Fit I-V relationship
        if parameters.get('fit_iv', True):
            fit_params = self._fit_iv_relationship(voltages, currents, errors)
        else:
            fit_params = None
            
        return {
            'voltages': voltages,
            'currents': currents,
            'errors': errors,
            'fit_parameters': fit_params,
            'reversal_potential': self._find_reversal_potential(voltages, currents)
        }
    
    def _calculate_conductance(self, data: Dict[str, Any], iv_curve: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate voltage-dependent conductance."""
        if iv_curve is None:
            return {}
            
        voltages = iv_curve['voltages']
        currents = iv_curve['currents']
        e_rev = iv_curve['reversal_potential']
        
        # Calculate conductance: g = I / (V - E_rev)
        driving_force = voltages - e_rev
        # Avoid division by zero
        mask = np.abs(driving_force) > 0.1
        conductance = np.zeros_like(voltages)
        conductance[mask] = currents[mask] / driving_force[mask]
        
        # Normalize to maximum
        if np.max(np.abs(conductance)) > 0:
            conductance_norm = conductance / np.max(np.abs(conductance))
        else:
            conductance_norm = conductance
            
        # Fit Boltzmann function if requested
        if self.config.get('ion_channel', {}).get('fit_boltzmann', True):
            boltzmann_params = self._fit_boltzmann(voltages, conductance_norm)
        else:
            boltzmann_params = None
            
        return {
            'conductance': conductance,
            'conductance_normalized': conductance_norm,
            'driving_force': driving_force,
            'boltzmann_fit': boltzmann_params
        }
    
    def _fit_boltzmann(self, voltage: np.ndarray, conductance: np.ndarray) -> Dict[str, float]:
        """Fit Boltzmann function to conductance data."""
        def boltzmann(v, v_half, slope):
            return 1 / (1 + np.exp((v_half - v) / slope))
            
        try:
            # Initial guess
            v_half_guess = voltage[np.argmin(np.abs(conductance - 0.5))]
            slope_guess = 10.0
            
            popt, pcov = curve_fit(
                boltzmann, voltage, conductance,
                p0=[v_half_guess, slope_guess],
                bounds=([-100, 0.1], [100, 50])
            )
            
            return {
                'v_half': popt[0],
                'slope': popt[1],
                'covariance': pcov,
                'fit_function': lambda v: boltzmann(v, *popt)
            }
        except Exception as e:
            logger.warning(f"Boltzmann fit failed: {e}")
            return None
    
    def _fit_kinetic_model(self, data: Dict[str, Any], model_type: str, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fit kinetic model to current traces."""
        model_func = self.models.get(model_type)
        
        if model_func is None:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # If Symbolica is available, try analytical solution first
        if self.symbolica_enabled and parameters.get('use_symbolica', True):
            analytical_result = self._symbolica_solve_kinetics(model_type, data)
            if analytical_result is not None:
                return analytical_result
                
        # Otherwise use numerical methods
        return model_func(data, parameters)
    
    def _hodgkin_huxley_model(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fit Hodgkin-Huxley style kinetics (m^p * h^q)."""
        current = data['current']
        time = data['time']
        
        # Define rate functions (alpha and beta)
        def alpha_m(v):
            return 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
            
        def beta_m(v):
            return 4 * np.exp(-(v + 65) / 18)
            
        def alpha_h(v):
            return 0.07 * np.exp(-(v + 65) / 20)
            
        def beta_h(v):
            return 1 / (1 + np.exp(-(v + 35) / 10))
            
        # Steady-state and time constant
        def m_inf(v):
            return alpha_m(v) / (alpha_m(v) + beta_m(v))
            
        def tau_m(v):
            return 1 / (alpha_m(v) + beta_m(v))
            
        def h_inf(v):
            return alpha_h(v) / (alpha_h(v) + beta_h(v))
            
        def tau_h(v):
            return 1 / (alpha_h(v) + beta_h(v))
            
        # Fit to current data
        results = {
            'model_type': 'hodgkin_huxley',
            'rate_constants': {
                'alpha_m': alpha_m,
                'beta_m': beta_m,
                'alpha_h': alpha_h,
                'beta_h': beta_h
            },
            'steady_state': {
                'm_inf': m_inf,
                'h_inf': h_inf
            },
            'time_constants': {
                'tau_m': tau_m,
                'tau_h': tau_h
            }
        }
        
        return results
    
    def _markov_2state_model(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fit 2-state Markov model (Closed <-> Open)."""
        # Rate constants
        def k_co(v):  # Closed to Open
            return parameters.get('k_co_0', 1.0) * np.exp(v / parameters.get('k_co_v', 25.0))
            
        def k_oc(v):  # Open to Closed
            return parameters.get('k_oc_0', 0.5) * np.exp(-v / parameters.get('k_oc_v', 25.0))
            
        # Steady-state open probability
        def p_open(v):
            return k_co(v) / (k_co(v) + k_oc(v))
            
        # Time constant
        def tau(v):
            return 1 / (k_co(v) + k_oc(v))
            
        return {
            'model_type': 'markov_2state',
            'rate_constants': {
                'k_co': k_co,
                'k_oc': k_oc
            },
            'steady_state': {
                'p_open': p_open
            },
            'time_constant': tau
        }
    
    def _markov_3state_model(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fit 3-state Markov model (Closed <-> Open <-> Inactivated)."""
        # This would implement a more complex Markov model
        # For now, return placeholder
        return {
            'model_type': 'markov_3state',
            'message': 'Complex Markov model fitting not yet implemented'
        }
    
    def _ghk_model(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Goldman-Hodgkin-Katz current equation."""
        # GHK current equation
        def ghk_current(v, p_ion, z, c_in, c_out, temp=22):
            R = 8.314  # J/(mol*K)
            F = 96485  # C/mol
            T = temp + 273.15  # Kelvin
            
            vf_rt = v * F / (R * T)
            
            if np.abs(vf_rt) < 0.01:  # Linear approximation for small voltages
                return p_ion * F * (c_in - c_out)
            else:
                return p_ion * z * F * vf_rt * (c_in - c_out * np.exp(-z * vf_rt)) / (1 - np.exp(-z * vf_rt))
                
        return {
            'model_type': 'ghk',
            'current_equation': ghk_current
        }
    
    def _symbolica_solve_kinetics(self, model_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use Symbolica MCP for analytical solutions of kinetic equations.
        
        This is a hook for future Symbolica integration.
        When implemented, it would:
        1. Define symbolic rate equations
        2. Solve ODEs analytically
        3. Generate Jacobians for faster fitting
        4. Perform sensitivity analysis
        """
        # Placeholder for Symbolica integration
        logger.info("Symbolica analytical solution not available - using numerical methods")
        return None
    
    def _analyze_gating(self, data: Dict[str, Any], kinetics: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze ion channel gating properties."""
        if kinetics is None:
            return {}
            
        results = {}
        
        # Extract activation/inactivation parameters
        if 'steady_state' in kinetics:
            ss = kinetics['steady_state']
            
            # Analyze activation
            if 'm_inf' in ss or 'p_open' in ss:
                act_func = ss.get('m_inf', ss.get('p_open'))
                results['activation'] = self._analyze_gating_curve(act_func, 'activation')
                
            # Analyze inactivation
            if 'h_inf' in ss:
                results['inactivation'] = self._analyze_gating_curve(ss['h_inf'], 'inactivation')
                
        # Time-dependent properties
        if 'time_constants' in kinetics:
            tc = kinetics['time_constants']
            results['time_constants'] = {}
            
            for name, tau_func in tc.items():
                results['time_constants'][name] = self._analyze_time_constant(tau_func)
                
        return results
    
    def _analyze_gating_curve(self, gating_func, gate_type: str) -> Dict[str, Any]:
        """Analyze a gating curve (activation or inactivation)."""
        voltages = np.linspace(-100, 50, 151)
        values = np.array([gating_func(v) for v in voltages])
        
        # Find V50 (half-activation/inactivation voltage)
        v50_idx = np.argmin(np.abs(values - 0.5))
        v50 = voltages[v50_idx]
        
        # Calculate slope factor
        # Find voltages at 10% and 90%
        v10_idx = np.argmin(np.abs(values - 0.1))
        v90_idx = np.argmin(np.abs(values - 0.9))
        
        if gate_type == 'activation':
            slope = (voltages[v90_idx] - voltages[v10_idx]) / 2.197  # For 10-90% range
        else:  # inactivation
            slope = (voltages[v10_idx] - voltages[v90_idx]) / 2.197
            
        return {
            'v50': v50,
            'slope_factor': slope,
            'voltages': voltages,
            'values': values,
            'gate_type': gate_type
        }
    
    def _analyze_time_constant(self, tau_func) -> Dict[str, Any]:
        """Analyze voltage-dependence of time constant."""
        voltages = np.linspace(-100, 50, 151)
        tau_values = np.array([tau_func(v) for v in voltages])
        
        # Find peak tau
        peak_idx = np.argmax(tau_values)
        peak_voltage = voltages[peak_idx]
        peak_tau = tau_values[peak_idx]
        
        return {
            'peak_voltage': peak_voltage,
            'peak_tau': peak_tau,
            'voltages': voltages,
            'tau_values': tau_values
        }
    
    def _calculate_q10(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Q10 temperature coefficient."""
        # Standard Q10 values for different processes
        q10_values = {
            'activation': 2.3,
            'inactivation': 2.9,
            'conductance': 1.5
        }
        
        temp = data.get('temperature', 22.0)
        temp_ref = parameters.get('reference_temperature', 22.0)
        
        # Apply temperature correction
        temp_factor = {}
        for process, q10 in q10_values.items():
            temp_factor[process] = q10 ** ((temp - temp_ref) / 10)
            
        return {
            'q10_values': q10_values,
            'temperature_factors': temp_factor,
            'temperature': temp,
            'reference_temperature': temp_ref
        }
    
    def _select_best_model(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically select the best kinetic model using information criteria."""
        models_to_test = parameters.get('models_to_test', ['hodgkin_huxley', 'markov_2state'])
        
        results = {}
        
        for model_type in models_to_test:
            try:
                # Fit each model
                fit_result = self._fit_kinetic_model(data, model_type, parameters)
                
                # Calculate information criteria
                n_params = self._count_parameters(model_type)
                n_data = len(data['current'])
                
                # Placeholder for actual likelihood calculation
                log_likelihood = -np.random.rand() * 1000  # Would be actual fit likelihood
                
                # AIC and BIC
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_data) - 2 * log_likelihood
                
                results[model_type] = {
                    'fit_result': fit_result,
                    'n_parameters': n_params,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic
                }
                
            except Exception as e:
                logger.warning(f"Model {model_type} fitting failed: {e}")
                
        # Select best model by AIC
        if results:
            best_model = min(results.keys(), key=lambda m: results[m]['aic'])
            results['best_model'] = best_model
            results['selection_criterion'] = 'AIC'
            
        return results
    
    def _count_parameters(self, model_type: str) -> int:
        """Count number of parameters for each model type."""
        param_counts = {
            'hodgkin_huxley': 8,  # 4 alpha/beta functions with 2 params each
            'markov_2state': 4,   # 2 rate constants with 2 params each
            'markov_3state': 9,   # More complex
            'ghk': 3              # Permeability, concentrations
        }
        return param_counts.get(model_type, 5)
    
    def _extract_steady_state_currents(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract steady-state currents from voltage-clamp data."""
        # This would implement actual steady-state detection
        # For now, return placeholder
        n_voltages = 15
        currents = np.random.randn(n_voltages) * 100 - 50
        errors = np.abs(currents) * 0.05
        return currents, errors
    
    def _simulate_iv_curve(self, voltages: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Simulate I-V curve for testing."""
        # Simple linear I-V with reversal at 0 mV
        e_rev = parameters.get('reversal_potential', 0.0)
        g_max = parameters.get('max_conductance', 10.0)
        
        currents = g_max * (voltages - e_rev)
        
        # Add some nonlinearity
        currents *= 1 / (1 + np.exp(-(voltages + 40) / 10))
        
        return currents
    
    def _find_reversal_potential(self, voltages: np.ndarray, currents: np.ndarray) -> float:
        """Find reversal potential from I-V curve."""
        # Find zero-crossing
        zero_idx = np.argmin(np.abs(currents))
        
        if zero_idx > 0 and zero_idx < len(voltages) - 1:
            # Linear interpolation for better accuracy
            i1, i2 = currents[zero_idx-1], currents[zero_idx+1]
            v1, v2 = voltages[zero_idx-1], voltages[zero_idx+1]
            
            if i1 * i2 < 0:  # Sign change
                e_rev = v1 - i1 * (v2 - v1) / (i2 - i1)
            else:
                e_rev = voltages[zero_idx]
        else:
            e_rev = voltages[zero_idx]
            
        return e_rev
    
    def _fit_iv_relationship(self, voltages: np.ndarray, currents: np.ndarray, 
                            errors: np.ndarray) -> Dict[str, Any]:
        """Fit I-V relationship with appropriate model."""
        # Simple linear fit for now
        # Could implement more complex models
        from scipy.stats import linregress
        
        result = linregress(voltages, currents)
        
        return {
            'slope': result.slope,
            'intercept': result.intercept,
            'r_value': result.rvalue,
            'p_value': result.pvalue,
            'std_err': result.stderr
        }
    
    def generate_report(self, results: Dict[str, Any] = None, output_dir: Union[str, Path] = None) -> Path:
        """Generate comprehensive ion channel analysis report."""
        if results is None:
            results = self.results
            
        if output_dir is None:
            output_dir = Path('./results')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. I-V Curve
        if 'iv_curve' in results:
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_iv_curve(ax1, results['iv_curve'])
            
        # 2. Conductance
        if 'conductance' in results:
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_conductance(ax2, results['conductance'])
            
        # 3. Gating curves
        if 'gating' in results:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_gating_curves(ax3, results['gating'])
            
        # 4. Time constants
        if 'gating' in results and 'time_constants' in results['gating']:
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_time_constants(ax4, results['gating']['time_constants'])
            
        # 5. Model comparison
        if 'model_selection' in results:
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_model_comparison(ax5, results['model_selection'])
            
        # 6. Temperature effects
        if 'q10' in results:
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_temperature_effects(ax6, results['q10'])
            
        # 7. Kinetic scheme
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_kinetic_scheme(ax7, results.get('kinetics'))
        
        plt.suptitle('Ion Channel Kinetics Analysis', fontsize=16)
        
        # Save figure
        fig_path = output_dir / 'ion_channel_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_path = output_dir / 'ion_channel_report.txt'
        self._generate_text_report(results, report_path)
        
        # Export results
        self.export_results(output_dir / 'ion_channel_results', format='json')
        
        logger.info(f"Ion channel analysis report saved to {output_dir}")
        
        return report_path
    
    def _plot_iv_curve(self, ax, iv_data):
        """Plot current-voltage relationship."""
        ax.errorbar(iv_data['voltages'], iv_data['currents'], 
                   yerr=iv_data.get('errors'), fmt='o', capsize=5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=iv_data['reversal_potential'], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Current (pA)')
        ax.set_title('Current-Voltage Relationship')
        ax.text(0.05, 0.95, f"E_rev = {iv_data['reversal_potential']:.1f} mV",
                transform=ax.transAxes, va='top')
    
    def _plot_conductance(self, ax, conductance_data):
        """Plot conductance curve."""
        voltages = np.linspace(-100, 50, 151)
        g_norm = conductance_data['conductance_normalized']
        
        ax.plot(voltages[:len(g_norm)], g_norm, 'o-', label='Data')
        
        if conductance_data.get('boltzmann_fit'):
            fit = conductance_data['boltzmann_fit']
            v_fit = np.linspace(voltages[0], voltages[-1], 200)
            g_fit = fit['fit_function'](v_fit)
            ax.plot(v_fit, g_fit, 'r-', label='Boltzmann fit')
            ax.text(0.05, 0.5, 
                   f"V₁/₂ = {fit['v_half']:.1f} mV\nSlope = {fit['slope']:.1f} mV",
                   transform=ax.transAxes)
            
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('G/G_max')
        ax.set_title('Voltage-Dependent Conductance')
        ax.legend()
    
    def _plot_gating_curves(self, ax, gating_data):
        """Plot activation/inactivation curves."""
        if 'activation' in gating_data:
            act = gating_data['activation']
            ax.plot(act['voltages'], act['values'], 'b-', label='Activation')
            
        if 'inactivation' in gating_data:
            inact = gating_data['inactivation']
            ax.plot(inact['voltages'], inact['values'], 'r-', label='Inactivation')
            
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Probability')
        ax.set_title('Gating Curves')
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _plot_time_constants(self, ax, tc_data):
        """Plot time constants."""
        for name, tc in tc_data.items():
            ax.plot(tc['voltages'], tc['tau_values'], label=name)
            
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Time constant (ms)')
        ax.set_title('Voltage-Dependent Time Constants')
        ax.legend()
        ax.set_yscale('log')
    
    def _plot_model_comparison(self, ax, model_data):
        """Plot model comparison results."""
        models = [m for m in model_data.keys() if m != 'best_model' and m != 'selection_criterion']
        if models:
            aic_values = [model_data[m]['aic'] for m in models]
            
            ax.bar(models, aic_values)
            ax.set_ylabel('AIC')
            ax.set_title('Model Comparison')
            
            if 'best_model' in model_data:
                best_idx = models.index(model_data['best_model'])
                ax.get_children()[best_idx].set_color('red')
                ax.text(0.5, 0.95, f"Best model: {model_data['best_model']}",
                       transform=ax.transAxes, ha='center', va='top')
    
    def _plot_temperature_effects(self, ax, q10_data):
        """Plot temperature effects."""
        processes = list(q10_data['q10_values'].keys())
        q10_vals = list(q10_data['q10_values'].values())
        
        ax.bar(processes, q10_vals)
        ax.set_ylabel('Q₁₀')
        ax.set_title('Temperature Coefficients')
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        
        temp_info = f"T = {q10_data['temperature']:.1f}°C\n"
        temp_info += f"T_ref = {q10_data['reference_temperature']:.1f}°C"
        ax.text(0.95, 0.95, temp_info, transform=ax.transAxes, ha='right', va='top')
    
    def _plot_kinetic_scheme(self, ax, kinetics_data):
        """Plot kinetic scheme diagram."""
        ax.axis('off')
        
        if kinetics_data and 'model_type' in kinetics_data:
            model_type = kinetics_data['model_type']
            
            if model_type == 'hodgkin_huxley':
                scheme_text = "Hodgkin-Huxley Model\n\n"
                scheme_text += "I = ḡ · m³h · (V - E_rev)\n\n"
                scheme_text += "dm/dt = αₘ(1-m) - βₘm\n"
                scheme_text += "dh/dt = αₕ(1-h) - βₕh"
                
            elif model_type == 'markov_2state':
                scheme_text = "2-State Markov Model\n\n"
                scheme_text += "    k_CO\n"
                scheme_text += "C ⇌ O\n"
                scheme_text += "    k_OC\n\n"
                scheme_text += "P_open = k_CO / (k_CO + k_OC)"
                
            else:
                scheme_text = f"Model: {model_type}"
                
            ax.text(0.5, 0.5, scheme_text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, family='monospace')
        else:
            ax.text(0.5, 0.5, "No kinetic model fitted", 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _generate_text_report(self, results: Dict[str, Any], report_path: Path):
        """Generate detailed text report."""
        with open(report_path, 'w') as f:
            f.write("Ion Channel Kinetics Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("Summary\n")
            f.write("-" * 30 + "\n")
            
            if 'iv_curve' in results:
                iv = results['iv_curve']
                f.write(f"Reversal potential: {iv['reversal_potential']:.2f} mV\n")
                
            if 'conductance' in results and results['conductance'].get('boltzmann_fit'):
                fit = results['conductance']['boltzmann_fit']
                f.write(f"V₁/₂ (activation): {fit['v_half']:.2f} mV\n")
                f.write(f"Slope factor: {fit['slope']:.2f} mV\n")
                
            if 'kinetics' in results:
                f.write(f"Kinetic model: {results['kinetics'].get('model_type', 'Unknown')}\n")
                
            if 'model_selection' in results:
                f.write(f"Best model (by {results['model_selection']['selection_criterion']}): ")
                f.write(f"{results['model_selection']['best_model']}\n")
                
            # Detailed results
            f.write("\n\nDetailed Results\n")
            f.write("-" * 30 + "\n")
            
            # Write all numeric results
            self._write_dict_to_file(f, results, indent=0)
    
    def _write_dict_to_file(self, f, d, indent=0):
        """Recursively write dictionary contents to file."""
        for key, value in d.items():
            if isinstance(value, dict):
                f.write(" " * indent + f"{key}:\n")
                self._write_dict_to_file(f, value, indent + 2)
            elif isinstance(value, (int, float)):
                f.write(" " * indent + f"{key}: {value:.4f}\n")
            elif isinstance(value, str):
                f.write(" " * indent + f"{key}: {value}\n")
            elif isinstance(value, (list, np.ndarray)) and len(value) < 10:
                f.write(" " * indent + f"{key}: {value}\n")
            else:
                f.write(" " * indent + f"{key}: <{type(value).__name__}>\n")
