"""
Statistical Framework for Electrophysiology Analysis
Provides statistical tests, confidence intervals, and effect sizes
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import warnings
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
import logging

logger = logging.getLogger(__name__)

class StatisticalFramework:
    """Comprehensive statistical analysis framework."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alpha = self.config.get('alpha', 0.05)
        self.correction_method = self.config.get('correction_method', 'bonferroni')
        
    def compare_groups(self, *groups, test_type: str = 'auto') -> Dict[str, Any]:
        """
        Compare multiple groups using appropriate statistical test.
        
        Args:
            *groups: Variable number of data arrays
            test_type: 'auto', 't-test', 'mann-whitney', 'anova', 'kruskal'
            
        Returns:
            Dictionary with test results
        """
        
        # Remove NaN values
        groups = [np.array(g)[~np.isnan(g)] for g in groups]
        
        # Auto-select test if needed
        if test_type == 'auto':
            test_type = self._select_test(groups)
            
        if len(groups) == 2:
            # Two-group comparison
            if test_type in ['t-test', 'welch']:
                result = self._ttest(groups[0], groups[1])
            elif test_type == 'mann-whitney':
                result = self._mann_whitney(groups[0], groups[1])
            else:
                raise ValueError(f"Invalid test type for 2 groups: {test_type}")
                
        else:
            # Multi-group comparison
            if test_type == 'anova':
                result = self._anova(*groups)
            elif test_type == 'kruskal':
                result = self._kruskal(*groups)
            else:
                raise ValueError(f"Invalid test type for {len(groups)} groups: {test_type}")
                
        # Add effect size
        result['effect_size'] = self.calculate_effect_size(*groups, test_type=test_type)
        
        # Add power analysis
        if len(groups) == 2 and len(groups[0]) > 3 and len(groups[1]) > 3:
            result['power'] = self.calculate_power(groups[0], groups[1])
            
        return result
    
    def _select_test(self, groups: List[np.ndarray]) -> str:
        """Automatically select appropriate test based on data properties."""
        
        # Check normality for each group
        normal_groups = []
        for g in groups:
            if len(g) < 20:
                # Too few samples for reliable normality test
                normal_groups.append(False)
            else:
                _, p_value = stats.shapiro(g)
                normal_groups.append(p_value > 0.05)
                
        # Check homogeneity of variance
        if len(groups) == 2:
            _, levene_p = stats.levene(groups[0], groups[1])
            equal_var = levene_p > 0.05
        else:
            _, levene_p = stats.levene(*groups)
            equal_var = levene_p > 0.05
            
        # Select test
        if len(groups) == 2:
            if all(normal_groups) and equal_var:
                return 't-test'
            elif all(normal_groups) and not equal_var:
                return 'welch'
            else:
                return 'mann-whitney'
        else:
            if all(normal_groups) and equal_var:
                return 'anova'
            else:
                return 'kruskal'
    
    def _ttest(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """Perform t-test."""
        
        # Check for equal variances
        _, levene_p = stats.levene(group1, group2)
        equal_var = levene_p > 0.05
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        return {
            'test': 'Student t-test' if equal_var else 'Welch t-test',
            'statistic': t_stat,
            'p_value': p_value,
            'equal_variance': equal_var,
            'mean_difference': np.mean(group1) - np.mean(group2),
            'ci_difference': self._confidence_interval_difference(group1, group2),
            'group_stats': {
                'group1': {'mean': np.mean(group1), 'std': np.std(group1, ddof=1), 'n': len(group1)},
                'group2': {'mean': np.mean(group2), 'std': np.std(group2, ddof=1), 'n': len(group2)}
            }
        }
    
    def _mann_whitney(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """Perform Mann-Whitney U test."""
        
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Calculate median difference
        median_diff = np.median(group1) - np.median(group2)
        
        # Hodges-Lehmann estimator
        differences = []
        for x in group1:
            for y in group2:
                differences.append(x - y)
        hl_estimator = np.median(differences)
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': u_stat,
            'p_value': p_value,
            'median_difference': median_diff,
            'hodges_lehmann': hl_estimator,
            'group_stats': {
                'group1': {'median': np.median(group1), 'iqr': stats.iqr(group1), 'n': len(group1)},
                'group2': {'median': np.median(group2), 'iqr': stats.iqr(group2), 'n': len(group2)}
            }
        }
    
    def _anova(self, *groups) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate eta squared
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            'test': 'One-way ANOVA',
            'statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'group_stats': []
        }
        
        # Add group statistics
        for i, g in enumerate(groups):
            result['group_stats'].append({
                'group': i+1,
                'mean': np.mean(g),
                'std': np.std(g, ddof=1),
                'n': len(g)
            })
            
        # Post-hoc tests if significant
        if p_value < self.alpha:
            result['post_hoc'] = self._tukey_hsd(*groups)
            
        return result
    
    def _kruskal(self, *groups) -> Dict[str, Any]:
        """Perform Kruskal-Wallis test."""
        
        h_stat, p_value = stats.kruskal(*groups)
        
        result = {
            'test': 'Kruskal-Wallis',
            'statistic': h_stat,
            'p_value': p_value,
            'group_stats': []
        }
        
        # Add group statistics
        for i, g in enumerate(groups):
            result['group_stats'].append({
                'group': i+1,
                'median': np.median(g),
                'iqr': stats.iqr(g),
                'n': len(g)
            })
            
        # Post-hoc tests if significant
        if p_value < self.alpha:
            result['post_hoc'] = self._dunn_test(*groups)
            
        return result
    
    def _tukey_hsd(self, *groups) -> List[Dict[str, Any]]:
        """Perform Tukey's HSD post-hoc test."""
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Prepare data for statsmodels
        data = []
        labels = []
        for i, g in enumerate(groups):
            data.extend(g)
            labels.extend([i] * len(g))
            
        # Perform test
        tukey = pairwise_tukeyhsd(data, labels, alpha=self.alpha)
        
        # Extract results
        results = []
        for row in tukey.summary().data[1:]:  # Skip header
            results.append({
                'group1': int(row[0]),
                'group2': int(row[1]),
                'mean_diff': float(row[2]),
                'p_adj': float(row[5]),
                'significant': row[6]
            })
            
        return results
    
    def _dunn_test(self, *groups) -> List[Dict[str, Any]]:
        """Perform Dunn's test for post-hoc analysis."""
        # Simplified implementation - would use proper Dunn's test
        results = []
        n_groups = len(groups)
        
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                _, p_value = stats.mannwhitneyu(groups[i], groups[j])
                # Bonferroni correction
                p_adj = min(p_value * (n_groups * (n_groups - 1) / 2), 1.0)
                
                results.append({
                    'group1': i,
                    'group2': j,
                    'p_value': p_value,
                    'p_adj': p_adj,
                    'significant': p_adj < self.alpha
                })
                
        return results
    
    def calculate_effect_size(self, *groups, test_type: str = 'auto') -> Dict[str, float]:
        """Calculate appropriate effect size measure."""
        
        if len(groups) == 2:
            g1, g2 = groups[0], groups[1]
            
            # Cohen's d
            pooled_std = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + 
                                 (len(g2)-1)*np.var(g2, ddof=1)) / 
                                (len(g1) + len(g2) - 2))
            
            if pooled_std > 0:
                cohen_d = (np.mean(g1) - np.mean(g2)) / pooled_std
            else:
                cohen_d = 0
                
            # Hedges' g (bias-corrected)
            n = len(g1) + len(g2)
            hedges_g = cohen_d * (1 - 3/(4*n - 9))
            
            # Glass's delta (when variances unequal)
            glass_delta = (np.mean(g1) - np.mean(g2)) / np.std(g2, ddof=1)
            
            # Rank-biserial correlation (for non-parametric)
            u_stat, _ = stats.mannwhitneyu(g1, g2)
            r_rb = 1 - (2*u_stat) / (len(g1) * len(g2))
            
            return {
                'cohen_d': cohen_d,
                'hedges_g': hedges_g,
                'glass_delta': glass_delta,
                'rank_biserial': r_rb
            }
            
        else:
            # Eta squared (calculated in ANOVA)
            return {'eta_squared': 0}  # Would be calculated in ANOVA
    
    def calculate_power(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate statistical power for two-group comparison."""
        
        # Calculate effect size
        effect_size_dict = self.calculate_effect_size(group1, group2)
        d = effect_size_dict['cohen_d']
        
        # Sample sizes
        n1, n2 = len(group1), len(group2)
        
        # Calculate power
        try:
            power = ttest_power(d, n1, self.alpha, alternative='two-sided')
            return power
        except:
            return np.nan
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic: Callable,
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        
        # Use scipy.stats.bootstrap
        res = bootstrap((data,), statistic, n_resamples=n_bootstrap,
                       confidence_level=confidence_level, method='BCa')
        
        return res.confidence_interval.low, res.confidence_interval.high
    
    def _confidence_interval_difference(self, group1: np.ndarray, 
                                      group2: np.ndarray) -> Tuple[float, float]:
        """Calculate CI for difference between means."""
        
        mean_diff = np.mean(group1) - np.mean(group2)
        
        # Standard error of difference
        se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + 
                         np.var(group2, ddof=1)/len(group2))
        
        # Degrees of freedom (Welch-Satterthwaite)
        df = ((np.var(group1, ddof=1)/len(group1) + 
               np.var(group2, ddof=1)/len(group2))**2 /
              ((np.var(group1, ddof=1)/len(group1))**2/(len(group1)-1) +
               (np.var(group2, ddof=1)/len(group2))**2/(len(group2)-1)))
        
        # Critical value
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        
        ci_low = mean_diff - t_crit * se_diff
        ci_high = mean_diff + t_crit * se_diff
        
        return (ci_low, ci_high)
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: Optional[str] = None) -> Dict[str, Any]:
        """Apply multiple comparison correction."""
        
        method = method or self.correction_method
        
        if method == 'none':
            return {
                'method': 'none',
                'corrected_p': p_values,
                'significant': [p < self.alpha for p in p_values]
            }
            
        # Use statsmodels for correction
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method=method
        )
        
        return {
            'method': method,
            'original_p': p_values,
            'corrected_p': p_corrected,
            'significant': reject,
            'alpha_corrected': alpha_bonf if method == 'bonferroni' else alpha_sidak
        }
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray,
                           method: str = 'pearson') -> Dict[str, Any]:
        """Perform correlation analysis."""
        
        # Remove pairs with NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return {'error': 'Insufficient data for correlation'}
            
        if method == 'pearson':
            r, p_value = stats.pearsonr(x_clean, y_clean)
            test_name = "Pearson's r"
        elif method == 'spearman':
            r, p_value = stats.spearmanr(x_clean, y_clean)
            test_name = "Spearman's ρ"
        elif method == 'kendall':
            r, p_value = stats.kendalltau(x_clean, y_clean)
            test_name = "Kendall's τ"
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        # Calculate confidence interval for correlation
        if method == 'pearson' and len(x_clean) > 3:
            # Fisher z-transformation
            z = np.arctanh(r)
            se = 1 / np.sqrt(len(x_clean) - 3)
            z_crit = stats.norm.ppf(1 - self.alpha/2)
            
            ci_low = np.tanh(z - z_crit * se)
            ci_high = np.tanh(z + z_crit * se)
        else:
            ci_low, ci_high = np.nan, np.nan
            
        # R-squared
        r_squared = r**2
        
        return {
            'method': test_name,
            'correlation': r,
            'p_value': p_value,
            'r_squared': r_squared,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(x_clean),
            'significant': p_value < self.alpha
        }
    
    def regression_diagnostics(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform regression diagnostics."""
        
        # Normality of residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Homoscedasticity (Breusch-Pagan test would be better)
        # Simple approach: correlation between abs(residuals) and fitted values
        
        # Durbin-Watson for autocorrelation
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        
        return {
            'normality': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'normal': shapiro_p > self.alpha
            },
            'durbin_watson': dw,
            'autocorrelation': 'positive' if dw < 1.5 else 'negative' if dw > 2.5 else 'none'
        }


# Convenience functions
def compare_groups(*groups, **kwargs):
    """Quick group comparison."""
    framework = StatisticalFramework()
    return framework.compare_groups(*groups, **kwargs)

def calculate_effect_size(group1, group2):
    """Quick effect size calculation."""
    framework = StatisticalFramework()
    return framework.calculate_effect_size(group1, group2)

def bootstrap_ci(data, statistic, **kwargs):
    """Quick bootstrap confidence interval."""
    framework = StatisticalFramework()
    return framework.bootstrap_confidence_interval(data, statistic, **kwargs)

def correlation(x, y, method='pearson', **kwargs):
    """Quick correlation analysis."""
    framework = StatisticalFramework()
    return framework.correlation_analysis(x, y, method=method)


# Example usage
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    control = np.random.normal(100, 15, 30)
    treatment = np.random.normal(110, 20, 35)
    
    # Compare groups
    framework = StatisticalFramework()
    result = framework.compare_groups(control, treatment)
    
    print("Statistical Comparison")
    print("=" * 40)
    print(f"Test: {result['test']}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']['cohen_d']:.3f}")
    print(f"Power: {result['power']:.3f}")
