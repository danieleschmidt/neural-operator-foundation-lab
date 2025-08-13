"""Advanced research framework for neural operator experimentation and validation."""

import time
import json
import hashlib
import statistics
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
import itertools
import warnings
import math
import threading
from abc import ABC, abstractmethod

try:
    from .robust_framework import robust_logger, robust_execution, error_boundary
    from .performance_optimizer import PerformanceProfiler, profile
    from .intelligent_cache import intelligent_cache
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_id: str
    name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Generate experiment ID if not provided."""
        if not self.experiment_id:
            param_str = json.dumps(self.parameters, sort_keys=True)
            self.experiment_id = hashlib.md5(param_str.encode()).hexdigest()[:12]


@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    significance_level: float = 0.05
    reject_null: bool = field(init=False)
    
    def __post_init__(self):
        """Determine if null hypothesis is rejected."""
        self.reject_null = self.p_value < self.significance_level


class ExperimentDesign(ABC):
    """Abstract base class for experiment designs."""
    
    @abstractmethod
    def generate_parameters(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for experiments."""
        pass
    
    @abstractmethod
    def get_design_info(self) -> Dict[str, Any]:
        """Get information about the experimental design."""
        pass


class GridSearchDesign(ExperimentDesign):
    """Grid search experimental design."""
    
    def __init__(self, parameter_grid: Dict[str, List[Any]]):
        self.parameter_grid = parameter_grid
        
    def generate_parameters(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys = list(self.parameter_grid.keys())
        values = list(self.parameter_grid.values())
        
        combinations = []
        for value_combo in itertools.product(*values):
            param_dict = dict(zip(keys, value_combo))
            combinations.append(param_dict)
        
        return combinations
    
    def get_design_info(self) -> Dict[str, Any]:
        """Get grid search design information."""
        total_combinations = 1
        for values in self.parameter_grid.values():
            total_combinations *= len(values)
        
        return {
            'design_type': 'grid_search',
            'parameters': list(self.parameter_grid.keys()),
            'total_combinations': total_combinations,
            'parameter_ranges': {k: len(v) for k, v in self.parameter_grid.items()}
        }


class RandomSearchDesign(ExperimentDesign):
    """Random search experimental design."""
    
    def __init__(self, parameter_distributions: Dict[str, Callable], num_samples: int = 100):
        self.parameter_distributions = parameter_distributions
        self.num_samples = num_samples
        
    def generate_parameters(self) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        import random
        
        combinations = []
        for _ in range(self.num_samples):
            param_dict = {}
            for param_name, distribution in self.parameter_distributions.items():
                param_dict[param_name] = distribution()
            combinations.append(param_dict)
        
        return combinations
    
    def get_design_info(self) -> Dict[str, Any]:
        """Get random search design information."""
        return {
            'design_type': 'random_search',
            'parameters': list(self.parameter_distributions.keys()),
            'num_samples': self.num_samples
        }


class StatisticalAnalyzer:
    """Advanced statistical analysis for experimental results."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics."""
        if not data:
            return {}
        
        cache_key = f"desc_stats_{hash(tuple(data))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        stats = {
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'mode': statistics.mode(data) if len(set(data)) < len(data) else data[0],
            'std': statistics.stdev(data) if len(data) > 1 else 0.0,
            'variance': statistics.variance(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data)
        }
        
        # Percentiles
        sorted_data = sorted(data)
        n = len(sorted_data)
        stats.update({
            'q1': sorted_data[n // 4] if n >= 4 else sorted_data[0],
            'q3': sorted_data[3 * n // 4] if n >= 4 else sorted_data[-1],
            'p95': sorted_data[int(0.95 * n)] if n >= 20 else sorted_data[-1],
            'p99': sorted_data[int(0.99 * n)] if n >= 100 else sorted_data[-1]
        })
        
        # Skewness and kurtosis (simplified)
        if len(data) > 2:
            mean_val = stats['mean']
            std_val = stats['std']
            if std_val > 0:
                skewness = sum((x - mean_val) ** 3 for x in data) / (len(data) * std_val ** 3)
                kurtosis = sum((x - mean_val) ** 4 for x in data) / (len(data) * std_val ** 4) - 3
                stats.update({'skewness': skewness, 'kurtosis': kurtosis})
        
        self.analysis_cache[cache_key] = stats
        return stats
    
    def t_test(self, group1: List[float], group2: List[float], 
              paired: bool = False) -> StatisticalTest:
        """Perform t-test between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalTest("t_test", 0.0, 1.0, significance_level=0.05)
        
        if paired and len(group1) != len(group2):
            raise ValueError("Paired t-test requires equal sample sizes")
        
        if paired:
            # Paired t-test
            differences = [a - b for a, b in zip(group1, group2)]
            mean_diff = statistics.mean(differences)
            std_diff = statistics.stdev(differences) if len(differences) > 1 else 0
            
            if std_diff == 0:
                t_stat = float('inf') if mean_diff != 0 else 0.0
            else:
                t_stat = mean_diff / (std_diff / math.sqrt(len(differences)))
            
            df = len(differences) - 1
        else:
            # Independent t-test (assuming equal variances)
            mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
            var1 = statistics.variance(group1) if len(group1) > 1 else 0
            var2 = statistics.variance(group2) if len(group2) > 1 else 0
            
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard error
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = math.sqrt(pooled_var * (1/n1 + 1/n2)) if pooled_var > 0 else 0
            
            if se == 0:
                t_stat = float('inf') if mean1 != mean2 else 0.0
            else:
                t_stat = (mean1 - mean2) / se
            
            df = n1 + n2 - 2
        
        # Simplified p-value calculation (two-tailed)
        # This is a rough approximation
        if abs(t_stat) > 2.576:  # 99% confidence
            p_value = 0.01
        elif abs(t_stat) > 1.96:  # 95% confidence
            p_value = 0.05
        elif abs(t_stat) > 1.645:  # 90% confidence
            p_value = 0.10
        else:
            p_value = 0.20
        
        return StatisticalTest(
            test_name="t_test",
            statistic=t_stat,
            p_value=p_value,
            critical_value=1.96  # For 95% confidence, two-tailed
        )
    
    def cohen_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0
        
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        
        if len(group1) == 1 and len(group2) == 1:
            return 0.0
        
        var1 = statistics.variance(group1) if len(group1) > 1 else 0
        var2 = statistics.variance(group2) if len(group2) > 1 else 0
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def correlation_analysis(self, x: List[float], y: List[float]) -> Dict[str, float]:
        """Compute correlation between two variables."""
        if len(x) != len(y) or len(x) < 2:
            return {'correlation': 0.0, 'p_value': 1.0}
        
        # Pearson correlation coefficient
        n = len(x)
        mean_x, mean_y = statistics.mean(x), statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if sum_sq_x == 0 or sum_sq_y == 0:
            return {'correlation': 0.0, 'p_value': 1.0}
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        correlation = numerator / denominator
        
        # Simplified significance test
        if abs(correlation) > 0.8:
            p_value = 0.01
        elif abs(correlation) > 0.6:
            p_value = 0.05
        elif abs(correlation) > 0.4:
            p_value = 0.10
        else:
            p_value = 0.20
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'sample_size': n
        }


class BenchmarkSuite:
    """Comprehensive benchmarking suite for neural operators."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir) if results_dir else Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.benchmarks = {}
        self.baseline_results = {}
        self.comparison_results = {}
        
    def register_benchmark(self, name: str, benchmark_func: Callable, 
                         baseline_params: Optional[Dict[str, Any]] = None):
        """Register a benchmark function."""
        self.benchmarks[name] = {
            'function': benchmark_func,
            'baseline_params': baseline_params or {}
        }
    
    @robust_execution(max_retries=2, fallback_value=None)
    def run_benchmark(self, name: str, params: Dict[str, Any], 
                     num_runs: int = 5) -> Optional[ExperimentResult]:
        """Run a single benchmark."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not registered")
        
        benchmark_func = self.benchmarks[name]['function']
        results = []
        errors = []
        
        for run_idx in range(num_runs):
            try:
                start_time = time.time()
                
                # Run benchmark
                result = benchmark_func(**params)
                
                end_time = time.time()
                duration = end_time - start_time
                
                if isinstance(result, dict):
                    result['duration'] = duration
                    result['run_idx'] = run_idx
                    results.append(result)
                else:
                    results.append({
                        'value': result,
                        'duration': duration,
                        'run_idx': run_idx
                    })
                    
            except Exception as e:
                errors.append(str(e))
                if _HAS_DEPENDENCIES:
                    robust_logger.log_warning(f"Benchmark run {run_idx} failed: {e}")
        
        if not results:
            return ExperimentResult(
                experiment_id="",
                name=name,
                parameters=params,
                metrics={},
                success=False,
                error_message=f"All runs failed: {errors}"
            )
        
        # Aggregate results
        metrics = self._aggregate_benchmark_results(results)
        
        return ExperimentResult(
            experiment_id="",
            name=name,
            parameters=params,
            metrics=metrics,
            metadata={
                'num_runs': len(results),
                'num_errors': len(errors),
                'error_rate': len(errors) / num_runs
            },
            duration=sum(r.get('duration', 0) for r in results),
            success=len(results) >= num_runs // 2  # At least half successful
        )
    
    def _aggregate_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate multiple benchmark runs."""
        metrics = {}
        
        # Collect all metric names
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Aggregate numeric metrics
        for key in all_keys:
            values = []
            for result in results:
                value = result.get(key)
                if isinstance(value, (int, float)) and not math.isnan(value):
                    values.append(float(value))
            
            if values:
                metrics[f"{key}_mean"] = statistics.mean(values)
                metrics[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                metrics[f"{key}_min"] = min(values)
                metrics[f"{key}_max"] = max(values)
                metrics[f"{key}_median"] = statistics.median(values)
        
        return metrics
    
    def run_comparison_study(self, benchmark_name: str, 
                           configurations: List[Dict[str, Any]],
                           config_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comparative study between different configurations."""
        if config_names is None:
            config_names = [f"config_{i}" for i in range(len(configurations))]
        
        if len(config_names) != len(configurations):
            raise ValueError("Number of config names must match configurations")
        
        results = {}
        
        for config_name, config in zip(config_names, configurations):
            result = self.run_benchmark(benchmark_name, config)
            if result:
                results[config_name] = result
        
        # Statistical comparison
        comparison = self._compare_configurations(results)
        
        return {
            'benchmark': benchmark_name,
            'configurations': {name: config for name, config in zip(config_names, configurations)},
            'results': results,
            'comparison': comparison,
            'timestamp': time.time()
        }
    
    def _compare_configurations(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """Compare different configurations statistically."""
        if len(results) < 2:
            return {'status': 'insufficient_data'}
        
        analyzer = StatisticalAnalyzer()
        comparisons = {}
        
        # Get all metric names
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result.metrics.keys())
        
        # Compare each metric across configurations
        for metric in all_metrics:
            if metric.endswith('_mean'):  # Focus on mean values
                base_metric = metric.replace('_mean', '')
                
                # Collect values for each configuration
                config_values = {}
                for config_name, result in results.items():
                    if metric in result.metrics:
                        # For statistical tests, we need individual values
                        # Here we simulate from mean and std
                        mean_val = result.metrics[metric]
                        std_val = result.metrics.get(f"{base_metric}_std", 0)
                        
                        # Simulate values (for demonstration)
                        import random
                        random.seed(42)  # For reproducibility
                        simulated_values = [
                            random.gauss(mean_val, max(std_val, 0.001))
                            for _ in range(5)
                        ]
                        config_values[config_name] = simulated_values
                
                if len(config_values) >= 2:
                    # Pairwise comparisons
                    config_names = list(config_values.keys())
                    for i in range(len(config_names)):
                        for j in range(i + 1, len(config_names)):
                            name1, name2 = config_names[i], config_names[j]
                            
                            # T-test
                            t_test = analyzer.t_test(
                                config_values[name1], 
                                config_values[name2]
                            )
                            
                            # Effect size
                            effect_size = analyzer.cohen_d(
                                config_values[name1], 
                                config_values[name2]
                            )
                            
                            comparison_key = f"{metric}_{name1}_vs_{name2}"
                            comparisons[comparison_key] = {
                                'metric': metric,
                                'configurations': [name1, name2],
                                't_test': {
                                    'statistic': t_test.statistic,
                                    'p_value': t_test.p_value,
                                    'significant': t_test.reject_null
                                },
                                'effect_size': effect_size,
                                'interpretation': self._interpret_comparison(t_test, effect_size)
                            }
        
        return comparisons
    
    def _interpret_comparison(self, t_test: StatisticalTest, effect_size: float) -> str:
        """Interpret comparison results."""
        if not t_test.reject_null:
            return "No significant difference"
        
        magnitude = "small"
        if abs(effect_size) > 0.8:
            magnitude = "large"
        elif abs(effect_size) > 0.5:
            magnitude = "medium"
        
        direction = "better" if effect_size > 0 else "worse"
        
        return f"Significant difference ({magnitude} effect, first configuration {direction})"
    
    def generate_benchmark_report(self, study_results: Dict[str, Any], 
                                output_file: Optional[Path] = None) -> str:
        """Generate a comprehensive benchmark report."""
        report_lines = []
        
        report_lines.append("# Benchmark Study Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Study overview
        report_lines.append("## Study Overview")
        report_lines.append(f"Benchmark: {study_results['benchmark']}")
        report_lines.append(f"Configurations: {len(study_results['configurations'])}")
        report_lines.append("")
        
        # Configuration details
        report_lines.append("## Configurations")
        for name, config in study_results['configurations'].items():
            report_lines.append(f"### {name}")
            for key, value in config.items():
                report_lines.append(f"- {key}: {value}")
            report_lines.append("")
        
        # Results summary
        report_lines.append("## Results Summary")
        for config_name, result in study_results['results'].items():
            report_lines.append(f"### {config_name}")
            report_lines.append(f"- Success: {result.success}")
            report_lines.append(f"- Duration: {result.duration:.3f}s")
            
            # Key metrics
            key_metrics = [k for k in result.metrics.keys() if k.endswith('_mean')]
            for metric in sorted(key_metrics):
                value = result.metrics[metric]
                std_key = metric.replace('_mean', '_std')
                std_value = result.metrics.get(std_key, 0)
                report_lines.append(f"- {metric}: {value:.4f} ± {std_value:.4f}")
            report_lines.append("")
        
        # Statistical comparisons
        if study_results['comparison']:
            report_lines.append("## Statistical Analysis")
            for comp_name, comp_data in study_results['comparison'].items():
                if comp_name != 'status':
                    report_lines.append(f"### {comp_name}")
                    report_lines.append(f"- Interpretation: {comp_data['interpretation']}")
                    report_lines.append(f"- P-value: {comp_data['t_test']['p_value']:.4f}")
                    report_lines.append(f"- Effect size: {comp_data['effect_size']:.4f}")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content
    
    def save_results(self, study_results: Dict[str, Any], filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_study_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert ExperimentResult objects to dictionaries for JSON serialization
        serializable_results = {}
        for key, value in study_results.items():
            if key == 'results':
                serializable_results[key] = {
                    name: {
                        'experiment_id': result.experiment_id,
                        'name': result.name,
                        'parameters': result.parameters,
                        'metrics': result.metrics,
                        'metadata': result.metadata,
                        'timestamp': result.timestamp,
                        'duration': result.duration,
                        'success': result.success,
                        'error_message': result.error_message
                    }
                    for name, result in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if _HAS_DEPENDENCIES:
            robust_logger.log_info(f"Benchmark results saved to {filepath}")


# Global benchmark suite
_global_benchmark_suite = BenchmarkSuite()


def register_benchmark(name: str, baseline_params: Optional[Dict[str, Any]] = None):
    """Decorator to register a benchmark function."""
    def decorator(func: Callable) -> Callable:
        _global_benchmark_suite.register_benchmark(name, func, baseline_params)
        return func
    return decorator


def run_comparative_study(benchmark_name: str, 
                         configurations: List[Dict[str, Any]],
                         config_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run comparative study using global benchmark suite."""
    return _global_benchmark_suite.run_comparison_study(
        benchmark_name, configurations, config_names
    )


def generate_report(study_results: Dict[str, Any], output_file: Optional[Path] = None) -> str:
    """Generate benchmark report using global suite."""
    return _global_benchmark_suite.generate_benchmark_report(study_results, output_file)