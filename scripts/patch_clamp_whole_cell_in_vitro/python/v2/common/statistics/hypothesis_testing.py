"""
Hypothesis Testing Utilities
============================

Statistical hypothesis testing functions for electrophysiology data
including parametric and non-parametric tests.

Author: Electrophysiology Analysis System
Version: 2.0.0
"""

import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple, Dict, List
import logging

from ...core.exceptions import AnalysisError, InvalidParameterError

logger = logging.getLogger(__name__)


def normality_test(data: np.ndarray,
                  method: str = 'shapiro',
                  alpha: float = 0.05) -> Dict[str, Union[float, bool, str]]:
    """
    Test for normality of data distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data (1D)
    method : str
        Test method: 'shapiro', 'jarque_bera', 'anderson', 'kstest'
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Test results including:
        - statistic: Test statistic
        - p_value: p-value (if applicable)
        - is_normal: Boolean indicating normality
        - method: Test method used
        - interpretation: Text interpretation
    """
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) < 8:
        raise InvalidParameterError("Need at least 8 data points for normality test")
        
    result = {'method': method, 'alpha': alpha}
    
    if method == 'shapiro':
        statistic, p_value = stats.shapiro(data)
        is_normal = p_value > alpha
        
        result.update({
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'interpretation': f"Data is {'normally' if is_normal else 'not normally'} distributed (p={p_value:.4f})"
        })
        
    elif method == 'jarque_bera':
        statistic, p_value = stats.jarque_bera(data)
        is_normal = p_value > alpha
        
        result.update({
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'interpretation': f"Data is {'normally' if is_normal else 'not normally'} distributed (p={p_value:.4f})"
        })
        
    elif method == 'anderson':
        statistic, critical_values, significance_levels = stats.anderson(data)
        
        # Find appropriate critical value
        alpha_percent = int(alpha * 100)
        if alpha_percent in [15, 10, 5, 2.5, 1]:
            idx = [15, 10, 5, 2.5, 1].index(alpha_percent)
            critical_value = critical_values[idx]
            is_normal = statistic < critical_value
        else:
            # Use 5% as default
            idx = 2
            critical_value = critical_values[idx]
            is_normal = statistic < critical_value
            
        result.update({
            'statistic': statistic,
            'critical_value': critical_value,
            'is_normal': is_normal,
            'interpretation': f"Data is {'normally' if is_normal else 'not normally'} distributed (stat={statistic:.4f}, crit={critical_value:.4f})"
        })
        
    elif method == 'kstest':
        statistic, p_value = stats.kstest(data, 'norm', 
                                         args=(np.mean(data), np.std(data)))
        is_normal = p_value > alpha
        
        result.update({
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'interpretation': f"Data is {'normally' if is_normal else 'not normally'} distributed (p={p_value:.4f})"
        })
        
    else:
        raise InvalidParameterError(f"Unknown normality test method: {method}")
        
    return result


def compare_two_groups(group1: np.ndarray,
                      group2: np.ndarray,
                      test: str = 'auto',
                      paired: bool = False,
                      alternative: str = 'two-sided',
                      alpha: float = 0.05) -> Dict[str, Union[float, bool, str]]:
    """
    Compare two groups using appropriate statistical test.
    
    Parameters
    ----------
    group1 : ndarray
        First group data
    group2 : ndarray
        Second group data
    test : str
        Test to use: 'auto', 't-test', 'welch', 'mann-whitney', 'wilcoxon'
    paired : bool
        Whether samples are paired
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', 'greater'
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Test results including statistics, p-value, and interpretation
    """
    # Remove NaN values
    if paired:
        mask = ~(np.isnan(group1) | np.isnan(group2))
        group1 = group1[mask]
        group2 = group2[mask]
        
        if len(group1) != len(group2):
            raise InvalidParameterError("Groups must have same length for paired test")
    else:
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
    # Auto-select test if requested
    if test == 'auto':
        # Check normality
        norm1 = normality_test(group1)['is_normal']
        norm2 = normality_test(group2)['is_normal']
        
        if norm1 and norm2:
            # Check variance equality
            _, p_var = stats.levene(group1, group2)
            if p_var > 0.05:
                test = 't-test' if not paired else 'paired-t'
            else:
                test = 'welch' if not paired else 'paired-t'
        else:
            test = 'mann-whitney' if not paired else 'wilcoxon'
            
        logger.info(f"Auto-selected test: {test}")
        
    result = {
        'test': test,
        'paired': paired,
        'alternative': alternative,
        'alpha': alpha,
        'n1': len(group1),
        'n2': len(group2),
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'median1': np.median(group1),
        'median2': np.median(group2)
    }
    
    # Perform test
    if test == 't-test':
        statistic, p_value = stats.ttest_ind(group1, group2, 
                                            alternative=alternative)
        test_name = "Independent t-test"
        
    elif test == 'welch':
        statistic, p_value = stats.ttest_ind(group1, group2, 
                                            equal_var=False,
                                            alternative=alternative)
        test_name = "Welch's t-test"
        
    elif test == 'paired-t':
        statistic, p_value = stats.ttest_rel(group1, group2,
                                            alternative=alternative)
        test_name = "Paired t-test"
        
    elif test == 'mann-whitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2,
                                               alternative=alternative)
        test_name = "Mann-Whitney U test"
        
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(group1, group2,
                                           alternative=alternative)
        test_name = "Wilcoxon signed-rank test"
        
    else:
        raise InvalidParameterError(f"Unknown test: {test}")
        
    # Calculate effect size
    from .descriptive import calculate_effect_size
    effect_size = calculate_effect_size(group1, group2)
    
    # Determine significance
    is_significant = p_value < alpha
    
    result.update({
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'is_significant': is_significant,
        'test_name': test_name,
        'interpretation': f"{test_name}: {'Significant' if is_significant else 'Not significant'} difference (p={p_value:.4f}, d={effect_size:.3f})"
    })
    
    return result


def compare_multiple_groups(groups: List[np.ndarray],
                          test: str = 'auto',
                          post_hoc: Optional[str] = 'tukey',
                          alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compare multiple groups using ANOVA or Kruskal-Wallis.
    
    Parameters
    ----------
    groups : list of ndarray
        List of group data arrays
    test : str
        Test to use: 'auto', 'anova', 'kruskal'
    post_hoc : str, optional
        Post-hoc test: 'tukey', 'bonferroni', 'dunn'
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Test results including omnibus test and post-hoc comparisons
    """
    # Clean data
    cleaned_groups = []
    for group in groups:
        cleaned = group[~np.isnan(group)]
        if len(cleaned) > 0:
            cleaned_groups.append(cleaned)
            
    if len(cleaned_groups) < 3:
        raise InvalidParameterError("Need at least 3 groups for multiple comparison")
        
    # Auto-select test
    if test == 'auto':
        # Check normality for all groups
        all_normal = all(normality_test(g)['is_normal'] for g in cleaned_groups)
        
        if all_normal:
            # Check variance homogeneity
            _, p_var = stats.levene(*cleaned_groups)
            if p_var > 0.05:
                test = 'anova'
            else:
                test = 'kruskal'  # Use non-parametric if variances unequal
        else:
            test = 'kruskal'
            
        logger.info(f"Auto-selected test: {test}")
        
    result = {
        'test': test,
        'n_groups': len(cleaned_groups),
        'group_sizes': [len(g) for g in cleaned_groups],
        'group_means': [np.mean(g) for g in cleaned_groups],
        'group_medians': [np.median(g) for g in cleaned_groups],
        'alpha': alpha
    }
    
    # Perform omnibus test
    if test == 'anova':
        statistic, p_value = stats.f_oneway(*cleaned_groups)
        test_name = "One-way ANOVA"
        
    elif test == 'kruskal':
        statistic, p_value = stats.kruskal(*cleaned_groups)
        test_name = "Kruskal-Wallis test"
        
    else:
        raise InvalidParameterError(f"Unknown test: {test}")
        
    is_significant = p_value < alpha
    
    result.update({
        'omnibus_statistic': statistic,
        'omnibus_p_value': p_value,
        'omnibus_significant': is_significant,
        'test_name': test_name,
        'interpretation': f"{test_name}: {'Significant' if is_significant else 'No significant'} differences among groups (p={p_value:.4f})"
    })
    
    # Perform post-hoc tests if significant
    if is_significant and post_hoc:
        post_hoc_results = perform_post_hoc(cleaned_groups, method=post_hoc,
                                           parametric=(test == 'anova'),
                                           alpha=alpha)
        result['post_hoc'] = post_hoc_results
        
    return result


def perform_post_hoc(groups: List[np.ndarray],
                    method: str = 'tukey',
                    parametric: bool = True,
                    alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform post-hoc pairwise comparisons.
    
    Parameters
    ----------
    groups : list of ndarray
        Group data arrays
    method : str
        Post-hoc method: 'tukey', 'bonferroni', 'dunn'
    parametric : bool
        Whether to use parametric tests
    alpha : float
        Significance level
        
    Returns
    -------
    post_hoc_results : dict
        Pairwise comparison results
    """
    n_groups = len(groups)
    n_comparisons = n_groups * (n_groups - 1) // 2
    
    results = {
        'method': method,
        'n_comparisons': n_comparisons,
        'comparisons': []
    }
    
    if method == 'tukey' and parametric:
        # Prepare data for Tukey HSD
        all_data = np.concatenate(groups)
        group_labels = np.concatenate([np.full(len(g), i) 
                                     for i, g in enumerate(groups)])
        
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
            
            # Extract results
            for i in range(len(tukey_result.summary().data) - 1):  # Skip header
                row = tukey_result.summary().data[i + 1]
                results['comparisons'].append({
                    'group1': int(row[0]),
                    'group2': int(row[1]),
                    'mean_diff': float(row[2]),
                    'p_value': float(row[5]),
                    'significant': row[6],
                    'lower_ci': float(row[4]),
                    'upper_ci': float(row[5])
                })
                
        except ImportError:
            logger.warning("statsmodels not available, falling back to Bonferroni")
            method = 'bonferroni'
            
    if method == 'bonferroni' or (method == 'tukey' and not parametric):
        # Bonferroni correction with pairwise tests
        adjusted_alpha = alpha / n_comparisons
        
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                if parametric:
                    stat, p = stats.ttest_ind(groups[i], groups[j])
                else:
                    stat, p = stats.mannwhitneyu(groups[i], groups[j])
                    
                results['comparisons'].append({
                    'group1': i,
                    'group2': j,
                    'statistic': stat,
                    'p_value': p,
                    'adjusted_p': min(p * n_comparisons, 1.0),
                    'significant': p < adjusted_alpha,
                    'mean_diff': np.mean(groups[i]) - np.mean(groups[j])
                })
                
    elif method == 'dunn':
        # Dunn's test for non-parametric post-hoc
        try:
            from scipy.stats import rankdata
            
            # Combine all data and rank
            all_data = np.concatenate(groups)
            group_labels = np.concatenate([np.full(len(g), i) 
                                         for i, g in enumerate(groups)])
            ranks = rankdata(all_data)
            
            # Calculate mean ranks for each group
            mean_ranks = []
            for i in range(n_groups):
                group_ranks = ranks[group_labels == i]
                mean_ranks.append(np.mean(group_ranks))
                
            # Pairwise comparisons
            N = len(all_data)
            
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    # Calculate z-statistic
                    diff = mean_ranks[i] - mean_ranks[j]
                    n_i, n_j = len(groups[i]), len(groups[j])
                    
                    se = np.sqrt((N * (N + 1) / 12) * (1/n_i + 1/n_j))
                    z = diff / se
                    
                    # Two-tailed p-value
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                    
                    # Bonferroni adjustment
                    adjusted_p = min(p * n_comparisons, 1.0)
                    
                    results['comparisons'].append({
                        'group1': i,
                        'group2': j,
                        'z_statistic': z,
                        'p_value': p,
                        'adjusted_p': adjusted_p,
                        'significant': adjusted_p < alpha,
                        'mean_rank_diff': diff
                    })
                    
        except Exception as e:
            raise AnalysisError(f"Dunn's test failed: {e}")
            
    return results


def paired_samples_test(before: np.ndarray,
                       after: np.ndarray,
                       test: str = 'auto',
                       alternative: str = 'two-sided',
                       alpha: float = 0.05) -> Dict[str, Union[float, bool, str]]:
    """
    Test for differences in paired samples.
    
    Parameters
    ----------
    before : ndarray
        Pre-treatment measurements
    after : ndarray
        Post-treatment measurements
    test : str
        Test type: 'auto', 'paired-t', 'wilcoxon', 'sign'
    alternative : str
        Alternative hypothesis
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Test results
    """
    # Ensure paired data
    if len(before) != len(after):
        raise InvalidParameterError("Paired samples must have same length")
        
    # Remove pairs with NaN
    mask = ~(np.isnan(before) | np.isnan(after))
    before = before[mask]
    after = after[mask]
    
    if len(before) < 3:
        raise InvalidParameterError("Need at least 3 paired observations")
        
    # Calculate differences
    differences = after - before
    
    # Auto-select test
    if test == 'auto':
        if normality_test(differences)['is_normal']:
            test = 'paired-t'
        else:
            test = 'wilcoxon'
            
    result = {
        'test': test,
        'n_pairs': len(before),
        'mean_before': np.mean(before),
        'mean_after': np.mean(after),
        'mean_difference': np.mean(differences),
        'std_difference': np.std(differences, ddof=1),
        'alternative': alternative,
        'alpha': alpha
    }
    
    # Perform test
    if test == 'paired-t':
        statistic, p_value = stats.ttest_rel(before, after,
                                            alternative=alternative)
        test_name = "Paired t-test"
        
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(before, after,
                                           alternative=alternative)
        test_name = "Wilcoxon signed-rank test"
        
    elif test == 'sign':
        # Sign test
        n_positive = np.sum(differences > 0)
        n_negative = np.sum(differences < 0)
        n_total = n_positive + n_negative  # Exclude zeros
        
        if n_total == 0:
            raise AnalysisError("All differences are zero")
            
        # Binomial test
        p_value = stats.binom_test(n_positive, n_total, 0.5,
                                  alternative=alternative)
        statistic = n_positive
        test_name = "Sign test"
        
    else:
        raise InvalidParameterError(f"Unknown test: {test}")
        
    is_significant = p_value < alpha
    
    # Calculate effect size (Cohen's d for repeated measures)
    effect_size = np.mean(differences) / np.std(differences, ddof=1)
    
    result.update({
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'is_significant': is_significant,
        'test_name': test_name,
        'interpretation': f"{test_name}: {'Significant' if is_significant else 'No significant'} change (p={p_value:.4f}, d={effect_size:.3f})"
    })
    
    return result


def correlation_test(x: np.ndarray,
                    y: np.ndarray,
                    method: str = 'pearson',
                    alternative: str = 'two-sided',
                    alpha: float = 0.05) -> Dict[str, Union[float, bool, str]]:
    """
    Test correlation between two variables.
    
    Parameters
    ----------
    x : ndarray
        First variable
    y : ndarray
        Second variable
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'
    alternative : str
        Alternative hypothesis
    alpha : float
        Significance level
        
    Returns
    -------
    result : dict
        Correlation test results
    """
    # Ensure same length
    if len(x) != len(y):
        raise InvalidParameterError("Variables must have same length")
        
    # Remove pairs with NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        raise InvalidParameterError("Need at least 3 observations")
        
    result = {
        'method': method,
        'n_obs': len(x),
        'alternative': alternative,
        'alpha': alpha
    }
    
    # Calculate correlation
    if method == 'pearson':
        r, p_value = stats.pearsonr(x, y)
        test_name = "Pearson correlation"
        
    elif method == 'spearman':
        r, p_value = stats.spearmanr(x, y)
        test_name = "Spearman rank correlation"
        
    elif method == 'kendall':
        r, p_value = stats.kendalltau(x, y)
        test_name = "Kendall's tau"
        
    else:
        raise InvalidParameterError(f"Unknown correlation method: {method}")
        
    # Adjust p-value for alternative hypothesis
    if alternative != 'two-sided':
        p_value = p_value / 2
        if (alternative == 'greater' and r < 0) or \
           (alternative == 'less' and r > 0):
            p_value = 1 - p_value
            
    is_significant = p_value < alpha
    
    # Calculate confidence interval for correlation
    if method == 'pearson' and len(x) > 3:
        # Fisher z-transformation
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(x) - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        ci_low = np.tanh(z - z_crit * se)
        ci_high = np.tanh(z + z_crit * se)
    else:
        ci_low, ci_high = np.nan, np.nan
        
    # Interpret strength
    abs_r = abs(r)
    if abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
        
    result.update({
        'correlation': r,
        'p_value': p_value,
        'is_significant': is_significant,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'test_name': test_name,
        'interpretation': f"{test_name}: {strength} {'positive' if r > 0 else 'negative'} correlation (r={r:.3f}, p={p_value:.4f})"
    })
    
    return result


def multiple_testing_correction(p_values: np.ndarray,
                              method: str = 'bonferroni',
                              alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Apply multiple testing correction.
    
    Parameters
    ----------
    p_values : ndarray
        Array of p-values
    method : str
        Correction method: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'
    alpha : float
        Family-wise error rate
        
    Returns
    -------
    result : dict
        Corrected p-values and rejection decisions
    """
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        corrected_p = np.minimum(p_values * n_tests, 1.0)
        reject = corrected_p < alpha
        
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        corrected_p = np.zeros_like(p_values)
        reject = np.zeros(n_tests, dtype=bool)
        
        for i, p in enumerate(sorted_p):
            adjusted_p = p * (n_tests - i)
            if adjusted_p < alpha:
                reject[sorted_idx[i]] = True
                corrected_p[sorted_idx[i]] = adjusted_p
            else:
                # Stop testing
                corrected_p[sorted_idx[i:]] = adjusted_p
                break
                
    elif method in ['fdr_bh', 'fdr_by']:
        # Benjamini-Hochberg or Benjamini-Yekutieli
        from statsmodels.stats.multitest import multipletests
        
        reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, 
                                                  method=method)
                                                  
    else:
        raise InvalidParameterError(f"Unknown correction method: {method}")
        
    return {
        'method': method,
        'n_tests': n_tests,
        'alpha': alpha,
        'original_p': p_values,
        'corrected_p': corrected_p,
        'reject': reject,
        'n_significant': np.sum(reject)
    }


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate example data
    np.random.seed(42)
    
    # Two groups for comparison
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(11, 2.5, 50)
    
    # Test normality
    print("Normality Tests:")
    print(f"Group 1: {normality_test(group1)['interpretation']}")
    print(f"Group 2: {normality_test(group2)['interpretation']}")
    
    # Compare two groups
    print("\nTwo-Group Comparison:")
    result = compare_two_groups(group1, group2)
    print(result['interpretation'])
    print(f"Effect size: {result['effect_size']:.3f}")
    
    # Multiple groups
    group3 = np.random.normal(12, 2, 50)
    groups = [group1, group2, group3]
    
    print("\nMultiple Group Comparison:")
    multi_result = compare_multiple_groups(groups)
    print(multi_result['interpretation'])
    
    if multi_result['omnibus_significant']:
        print("\nPost-hoc comparisons:")
        for comp in multi_result['post_hoc']['comparisons']:
            print(f"  Group {comp['group1']} vs {comp['group2']}: "
                  f"p={comp['p_value']:.4f}, "
                  f"{'significant' if comp['significant'] else 'not significant'}")
            
    # Correlation test
    x = np.random.normal(0, 1, 100)
    y = 0.5 * x + np.random.normal(0, 0.5, 100)
    
    print("\nCorrelation Test:")
    corr_result = correlation_test(x, y)
    print(corr_result['interpretation'])
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box plot comparison
    axes[0, 0].boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
    axes[0, 0].set_title('Group Comparison')
    axes[0, 0].set_ylabel('Value')
    
    # Paired data
    before = np.random.normal(10, 2, 30)
    after = before + np.random.normal(1, 1, 30)
    
    axes[0, 1].scatter(before, after, alpha=0.6)
    axes[0, 1].plot([before.min(), before.max()], 
                    [before.min(), before.max()], 'k--')
    axes[0, 1].set_xlabel('Before')
    axes[0, 1].set_ylabel('After')
    axes[0, 1].set_title('Paired Data')
    
    # Correlation
    axes[1, 0].scatter(x, y, alpha=0.6)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title(f'Correlation (r={corr_result["correlation"]:.3f})')
    
    # P-value distribution
    p_values = np.random.uniform(0, 1, 20)
    p_values[:5] = np.random.uniform(0, 0.05, 5)  # Some significant
    
    corrected = multiple_testing_correction(p_values, method='fdr_bh')
    
    axes[1, 1].scatter(np.arange(len(p_values)), p_values, 
                      label='Original', alpha=0.6)
    axes[1, 1].scatter(np.arange(len(p_values)), corrected['corrected_p'], 
                      label='Corrected', alpha=0.6)
    axes[1, 1].axhline(0.05, color='r', linestyle='--', label='α=0.05')
    axes[1, 1].set_xlabel('Test Index')
    axes[1, 1].set_ylabel('P-value')
    axes[1, 1].set_title('Multiple Testing Correction')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()