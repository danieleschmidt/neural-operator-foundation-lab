"""Statistical analysis tools for neural operator research."""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, friedmanchisquare, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    significance_level: float
    is_significant: bool
    interpretation: str


@dataclass
class ComparisonResult:
    """Results from comparing two methods."""
    method1: str
    method2: str
    test_results: Dict[str, StatisticalTest]
    summary: Dict[str, Any]


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for neural operator research."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = []
    
    def compare_methods(self, 
                       results1: List[float], 
                       results2: List[float],
                       method1_name: str = "Method 1",
                       method2_name: str = "Method 2",
                       metrics_name: str = "MSE") -> ComparisonResult:
        """Compare two methods using multiple statistical tests."""
        
        # Convert to numpy arrays
        data1 = np.array(results1)
        data2 = np.array(results2)
        
        # Perform multiple statistical tests
        test_results = {}
        
        # 1. Normality tests
        normality1 = self._test_normality(data1, f"{method1_name} {metrics_name}")
        normality2 = self._test_normality(data2, f"{method2_name} {metrics_name}")
        
        test_results['normality_method1'] = normality1
        test_results['normality_method2'] = normality2
        
        # 2. Choose appropriate test based on normality
        if normality1.is_significant and normality2.is_significant:
            # Both are normal - use t-test
            comparison_test = self._welch_t_test(data1, data2, method1_name, method2_name)
        else:
            # Non-normal data - use non-parametric test
            comparison_test = self._mann_whitney_test(data1, data2, method1_name, method2_name)
        
        test_results['comparison'] = comparison_test
        
        # 3. Effect size
        effect_size_result = self._compute_effect_size(data1, data2, method1_name, method2_name)
        test_results['effect_size'] = effect_size_result
        
        # 4. Confidence interval for difference
        ci_result = self._bootstrap_confidence_interval(data1, data2, method1_name, method2_name)
        test_results['confidence_interval'] = ci_result
        
        # 5. Power analysis
        power_result = self._power_analysis(data1, data2, method1_name, method2_name)
        test_results['power'] = power_result
        
        # Summary statistics
        summary = {
            'method1_stats': {
                'mean': np.mean(data1),
                'std': np.std(data1, ddof=1),
                'median': np.median(data1),
                'n': len(data1)
            },
            'method2_stats': {
                'mean': np.mean(data2),
                'std': np.std(data2, ddof=1),
                'median': np.median(data2),
                'n': len(data2)
            },
            'difference': {
                'mean_diff': np.mean(data1) - np.mean(data2),
                'median_diff': np.median(data1) - np.median(data2),
                'relative_improvement': (np.mean(data2) - np.mean(data1)) / np.mean(data2) * 100
            }
        }
        
        result = ComparisonResult(
            method1=method1_name,
            method2=method2_name,
            test_results=test_results,
            summary=summary
        )
        
        self.results.append(result)
        return result
    
    def _test_normality(self, data: np.ndarray, name: str) -> StatisticalTest:
        """Test for normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return StatisticalTest(
                test_name="Shapiro-Wilk",
                statistic=np.nan,
                p_value=np.nan,
                effect_size=None,
                confidence_interval=None,
                significance_level=self.significance_level,
                is_significant=False,
                interpretation=f"Insufficient data for {name} normality test"
            )
        
        statistic, p_value = stats.shapiro(data)
        is_significant = p_value < self.significance_level
        
        interpretation = (f"Data for {name} is {'not ' if is_significant else ''}normally distributed "
                         f"(W={statistic:.4f}, p={p_value:.4f})")
        
        return StatisticalTest(
            test_name="Shapiro-Wilk",
            statistic=statistic,
            p_value=p_value,
            effect_size=None,
            confidence_interval=None,
            significance_level=self.significance_level,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _welch_t_test(self, data1: np.ndarray, data2: np.ndarray, 
                     name1: str, name2: str) -> StatisticalTest:
        """Perform Welch's t-test (unequal variances)."""
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        is_significant = p_value < self.significance_level
        
        # Degrees of freedom for Welch's t-test
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        # Confidence interval for mean difference
        mean_diff = np.mean(data1) - np.mean(data2)
        se_diff = np.sqrt(s1/n1 + s2/n2)
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        interpretation = (f"{'Significant' if is_significant else 'No significant'} difference "
                         f"between {name1} and {name2} (t={statistic:.4f}, p={p_value:.4f})")
        
        return StatisticalTest(
            test_name="Welch's t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=None,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=self.significance_level,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _mann_whitney_test(self, data1: np.ndarray, data2: np.ndarray,
                          name1: str, name2: str) -> StatisticalTest:
        """Perform Mann-Whitney U test (non-parametric)."""
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        is_significant = p_value < self.significance_level
        
        interpretation = (f"{'Significant' if is_significant else 'No significant'} difference "
                         f"between {name1} and {name2} (U={statistic:.4f}, p={p_value:.4f})")
        
        return StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            effect_size=None,
            confidence_interval=None,
            significance_level=self.significance_level,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _compute_effect_size(self, data1: np.ndarray, data2: np.ndarray,
                           name1: str, name2: str) -> StatisticalTest:
        """Compute Cohen's d effect size."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            magnitude = "negligible"
        elif abs(cohens_d) < 0.5:
            magnitude = "small"
        elif abs(cohens_d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = (f"Effect size between {name1} and {name2} is {magnitude} "
                         f"(Cohen's d={cohens_d:.4f})")
        
        return StatisticalTest(
            test_name="Cohen's d",
            statistic=cohens_d,
            p_value=np.nan,
            effect_size=cohens_d,
            confidence_interval=None,
            significance_level=self.significance_level,
            is_significant=abs(cohens_d) >= 0.2,
            interpretation=interpretation
        )
    
    def _bootstrap_confidence_interval(self, data1: np.ndarray, data2: np.ndarray,
                                     name1: str, name2: str,
                                     n_bootstrap: int = 10000) -> StatisticalTest:
        """Compute bootstrap confidence interval for mean difference."""
        np.random.seed(42)  # For reproducibility
        
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            
            # Mean difference
            diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = self.significance_level
        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        # Check if CI contains zero
        contains_zero = ci_lower <= 0 <= ci_upper
        
        interpretation = (f"Bootstrap 95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]. "
                         f"{'Contains' if contains_zero else 'Does not contain'} zero.")
        
        return StatisticalTest(
            test_name="Bootstrap CI",
            statistic=np.mean(bootstrap_diffs),
            p_value=np.nan,
            effect_size=None,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=self.significance_level,
            is_significant=not contains_zero,
            interpretation=interpretation
        )
    
    def _power_analysis(self, data1: np.ndarray, data2: np.ndarray,
                       name1: str, name2: str) -> StatisticalTest:
        """Estimate statistical power of the test."""
        from scipy.stats import norm
        
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Effect size
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = abs(mean1 - mean2) / pooled_std
        
        # Standard error
        se = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        # Critical value (two-tailed)
        t_critical = stats.t.ppf(1 - self.significance_level/2, n1 + n2 - 2)
        critical_diff = t_critical * se
        
        # Power calculation (simplified)
        true_diff = abs(mean1 - mean2)
        z_beta = (true_diff - critical_diff) / se
        power = 1 - norm.cdf(-z_beta) + norm.cdf(-z_beta - 2*true_diff/se)
        
        interpretation = (f"Estimated statistical power: {power:.3f}. "
                         f"{'Adequate' if power >= 0.8 else 'Low'} power for detecting differences.")
        
        return StatisticalTest(
            test_name="Power Analysis",
            statistic=power,
            p_value=np.nan,
            effect_size=effect_size,
            confidence_interval=None,
            significance_level=self.significance_level,
            is_significant=power >= 0.8,
            interpretation=interpretation
        )
    
    def multiple_method_comparison(self, 
                                 results_dict: Dict[str, List[float]],
                                 metrics_name: str = "MSE") -> Dict[str, Any]:
        """Compare multiple methods using appropriate statistical tests."""
        method_names = list(results_dict.keys())
        all_results = [np.array(results) for results in results_dict.values()]
        
        # Test for overall differences
        if len(method_names) < 3:
            raise ValueError("Need at least 3 methods for multiple comparison")
        
        # Test normality for all methods
        normality_results = {}
        all_normal = True
        
        for name, data in zip(method_names, all_results):
            normality_test = self._test_normality(data, name)
            normality_results[name] = normality_test
            if normality_test.is_significant:  # Rejected normality
                all_normal = False
        
        # Choose appropriate overall test
        if all_normal:
            # Use one-way ANOVA
            f_stat, p_value = stats.f_oneway(*all_results)
            test_name = "One-way ANOVA"
        else:
            # Use Kruskal-Wallis test
            h_stat, p_value = stats.kruskal(*all_results)
            f_stat = h_stat
            test_name = "Kruskal-Wallis"
        
        overall_test = StatisticalTest(
            test_name=test_name,
            statistic=f_stat,
            p_value=p_value,
            effect_size=None,
            confidence_interval=None,
            significance_level=self.significance_level,
            is_significant=p_value < self.significance_level,
            interpretation=f"{'Significant' if p_value < self.significance_level else 'No significant'} "
                          f"differences between methods ({test_name}: statistic={f_stat:.4f}, p={p_value:.4f})"
        )
        
        # Post-hoc pairwise comparisons if overall test is significant
        pairwise_results = {}
        
        if overall_test.is_significant:
            for i in range(len(method_names)):
                for j in range(i+1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    data1, data2 = all_results[i], all_results[j]
                    
                    comparison = self.compare_methods(
                        data1.tolist(), data2.tolist(), method1, method2, metrics_name
                    )
                    pairwise_results[f"{method1}_vs_{method2}"] = comparison
        
        return {
            'overall_test': overall_test,
            'normality_tests': normality_results,
            'pairwise_comparisons': pairwise_results,
            'summary_statistics': self._compute_summary_statistics(results_dict)
        }
    
    def _compute_summary_statistics(self, results_dict: Dict[str, List[float]]) -> Dict[str, Dict]:
        """Compute summary statistics for all methods."""
        summary = {}
        
        for method_name, results in results_dict.items():
            data = np.array(results)
            summary[method_name] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25)
            }
        
        return summary
    
    def generate_statistical_report(self) -> str:
        """Generate a comprehensive statistical analysis report."""
        report = ["Statistical Analysis Report", "=" * 50, ""]
        
        for i, result in enumerate(self.results):
            report.append(f"Comparison {i+1}: {result.method1} vs {result.method2}")
            report.append("-" * 40)
            
            # Summary statistics
            report.append("Summary Statistics:")
            summary = result.summary
            
            for method in ['method1', 'method2']:
                method_name = result.method1 if method == 'method1' else result.method2
                stats = summary[f'{method}_stats']
                report.append(f"  {method_name}:")
                report.append(f"    Mean ± SD: {stats['mean']:.6f} ± {stats['std']:.6f}")
                report.append(f"    Median: {stats['median']:.6f}")
                report.append(f"    N: {stats['n']}")
            
            report.append(f"  Mean difference: {summary['difference']['mean_diff']:.6f}")
            report.append(f"  Relative improvement: {summary['difference']['relative_improvement']:.2f}%")
            report.append("")
            
            # Statistical tests
            report.append("Statistical Tests:")
            for test_name, test_result in result.test_results.items():
                if test_name.startswith('normality'):
                    continue  # Skip normality in main report
                
                report.append(f"  {test_result.test_name}:")
                report.append(f"    {test_result.interpretation}")
                
                if test_result.confidence_interval:
                    ci_lower, ci_upper = test_result.confidence_interval
                    report.append(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                
                report.append("")
            
            report.append("")
        
        return "\n".join(report)


class SignificanceTests:
    """Collection of significance tests for neural operator evaluation."""
    
    @staticmethod
    def mcnemar_test(correct1: List[bool], correct2: List[bool]) -> Dict[str, float]:
        """McNemar's test for comparing binary classifications."""
        correct1 = np.array(correct1, dtype=bool)
        correct2 = np.array(correct2, dtype=bool)
        
        # Contingency table
        both_correct = np.sum(correct1 & correct2)
        only1_correct = np.sum(correct1 & ~correct2)
        only2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if only1_correct + only2_correct == 0:
            # No discordant pairs
            return {'statistic': 0, 'p_value': 1.0}
        
        # Continuity correction
        statistic = (abs(only1_correct - only2_correct) - 1)**2 / (only1_correct + only2_correct)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'both_correct': both_correct,
            'only1_correct': only1_correct,
            'only2_correct': only2_correct,
            'both_wrong': both_wrong
        }
    
    @staticmethod
    def bootstrap_test(data1: List[float], data2: List[float], 
                      n_bootstrap: int = 10000, 
                      statistic_func: callable = np.mean) -> Dict[str, float]:
        """Bootstrap hypothesis test for arbitrary statistics."""
        data1, data2 = np.array(data1), np.array(data2)
        
        # Observed difference
        observed_diff = statistic_func(data1) - statistic_func(data2)
        
        # Combine datasets under null hypothesis
        combined = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)
        
        # Bootstrap under null hypothesis
        bootstrap_diffs = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            # Resample
            resampled = np.random.permutation(combined)
            boot_sample1 = resampled[:n1]
            boot_sample2 = resampled[n1:]
            
            # Compute statistic
            boot_diff = statistic_func(boot_sample1) - statistic_func(boot_sample2)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff),
            np.mean(bootstrap_diffs <= observed_diff)
        )
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'bootstrap_mean': np.mean(bootstrap_diffs),
            'bootstrap_std': np.std(bootstrap_diffs)
        }
    
    @staticmethod
    def permutation_test(data1: List[float], data2: List[float],
                        n_permutations: int = 10000,
                        statistic_func: callable = np.mean) -> Dict[str, float]:
        """Permutation test for comparing two groups."""
        return SignificanceTests.bootstrap_test(data1, data2, n_permutations, statistic_func)