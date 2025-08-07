"""Comprehensive benchmark suite for neural operator research."""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..base import NeuralOperatorBase
from ..utils import compute_spectral_metrics, ConservationLawChecker


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    mse: float
    relative_l2: float
    max_error: float
    spectral_mse: float
    spectral_correlation: float
    energy_conservation: float
    mass_conservation: float
    inference_time: float
    memory_usage: int
    parameters: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    dataset_name: str
    metrics: BenchmarkMetrics
    config: Dict[str, Any]
    timestamp: str
    random_seed: int


class ResearchBenchmarkSuite:
    """Comprehensive benchmark suite for research-grade evaluation."""
    
    def __init__(self, output_dir: str = "benchmark_results", device: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.results = []
        
        # Standard datasets for benchmarking
        self.benchmark_datasets = {
            "navier_stokes_2d": self._create_navier_stokes_2d,
            "darcy_flow": self._create_darcy_flow,
            "burgers_1d": self._create_burgers_1d,
            "heat_equation": self._create_heat_equation,
            "wave_equation": self._create_wave_equation
        }
    
    def run_comprehensive_benchmark(self, 
                                  models: Dict[str, NeuralOperatorBase],
                                  datasets: Optional[List[str]] = None,
                                  num_runs: int = 3,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
        """Run comprehensive benchmark across multiple models and datasets."""
        if datasets is None:
            datasets = list(self.benchmark_datasets.keys())
        
        all_results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Benchmarking model: {model_name}")
            model.to(self.device)
            
            for dataset_name in datasets:
                self.logger.info(f"  Dataset: {dataset_name}")
                
                # Generate dataset
                test_data = self.benchmark_datasets[dataset_name]()
                
                # Multiple runs for statistical significance
                run_results = []
                for run_idx in range(num_runs):
                    seed = 42 + run_idx
                    torch.manual_seed(seed)
                    
                    metrics = self._evaluate_model(model, test_data, seed)
                    
                    result = BenchmarkResult(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        metrics=metrics,
                        config=model.config if hasattr(model, 'config') else {},
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        random_seed=seed
                    )
                    
                    run_results.append(result)
                    all_results.append(result)
                
                # Compute statistics across runs
                self._compute_run_statistics(run_results, confidence_level)
        
        self.results.extend(all_results)
        
        # Generate comprehensive report
        report = self._generate_report(all_results)
        
        # Save results
        self._save_results(all_results, report)
        
        return report
    
    def _evaluate_model(self, model: NeuralOperatorBase, 
                       test_data: Tuple[torch.Tensor, torch.Tensor],
                       seed: int) -> BenchmarkMetrics:
        """Evaluate model on test data."""
        model.eval()
        
        inputs, targets = test_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Timing and memory tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        with torch.no_grad():
            predictions = model(inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_usage = end_memory - start_memory
        else:
            memory_usage = 0
        
        inference_time = end_time - start_time
        
        # Compute metrics
        mse = torch.mean((predictions - targets)**2).item()
        
        # Relative L2 error
        l2_pred = torch.norm(predictions.flatten(), p=2)
        l2_target = torch.norm(targets.flatten(), p=2)
        relative_l2 = (torch.norm((predictions - targets).flatten(), p=2) / l2_target).item()
        
        # Max error
        max_error = torch.max(torch.abs(predictions - targets)).item()
        
        # Spectral metrics
        spectral_metrics = compute_spectral_metrics(predictions, targets)
        
        # Conservation laws
        energy_conservation = ConservationLawChecker.energy_conservation(predictions)
        mass_conservation = ConservationLawChecker.mass_conservation(predictions)
        
        # Model parameters
        num_parameters = model.count_parameters()
        
        return BenchmarkMetrics(
            mse=mse,
            relative_l2=relative_l2,
            max_error=max_error,
            spectral_mse=spectral_metrics['spectral_mse'],
            spectral_correlation=spectral_metrics['spectral_correlation'],
            energy_conservation=energy_conservation,
            mass_conservation=mass_conservation,
            inference_time=inference_time,
            memory_usage=memory_usage,
            parameters=num_parameters
        )
    
    def _create_navier_stokes_2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic 2D Navier-Stokes test data."""
        batch_size, resolution = 16, 64
        
        # Generate random initial conditions
        x = torch.randn(batch_size, resolution, resolution, 3)  # u, v, p
        
        # Simple analytical solution for testing
        X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), indexing='ij')
        
        # Taylor-Green vortex solution
        t = 0.1
        nu = 0.01
        
        u = torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y) * torch.exp(-8*np.pi**2*nu*t)
        v = -torch.cos(2*np.pi*X) * torch.sin(2*np.pi*Y) * torch.exp(-8*np.pi**2*nu*t)
        p = -0.25 * (torch.cos(4*np.pi*X) + torch.cos(4*np.pi*Y)) * torch.exp(-16*np.pi**2*nu*t)
        
        target = torch.stack([u, v, p], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return x, target
    
    def _create_darcy_flow(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic Darcy flow test data."""
        batch_size, resolution = 16, 64
        
        # Permeability field (input)
        x = torch.randn(batch_size, resolution, resolution, 1)
        
        # Pressure field (target) - simplified analytical solution
        X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), indexing='ij')
        
        pressure = torch.sin(np.pi*X) * torch.sin(np.pi*Y)
        target = pressure.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        
        return x, target
    
    def _create_burgers_1d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic 1D Burgers equation test data."""
        batch_size, resolution = 32, 128
        
        # Initial condition
        x_coords = torch.linspace(-1, 1, resolution)
        u0 = torch.sin(np.pi * x_coords)
        
        inputs = u0.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        
        # Analytical solution at t=0.1 (simplified)
        t = 0.1
        nu = 0.01
        
        # Approximate solution
        target = u0 * torch.exp(-nu * np.pi**2 * t)
        target = target.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        
        return inputs, target
    
    def _create_heat_equation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic heat equation test data."""
        batch_size, resolution = 16, 64
        
        # Initial temperature distribution
        X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), indexing='ij')
        
        u0 = torch.sin(np.pi*X) * torch.sin(np.pi*Y)
        inputs = u0.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        
        # Analytical solution
        t = 0.1
        alpha = 0.1
        
        target = u0 * torch.exp(-2*np.pi**2*alpha*t)
        target = target.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        
        return inputs, target
    
    def _create_wave_equation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic wave equation test data."""
        batch_size, resolution = 16, 64
        
        # Initial wave
        X, Y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), indexing='ij')
        
        u0 = torch.sin(np.pi*X) * torch.sin(np.pi*Y)
        inputs = u0.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        
        # Wave solution at t=0.1
        t = 0.1
        c = 1.0  # Wave speed
        
        target = u0 * torch.cos(np.pi*c*t*np.sqrt(2))
        target = target.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        
        return inputs, target
    
    def _compute_run_statistics(self, results: List[BenchmarkResult], 
                               confidence_level: float):
        """Compute statistics across multiple runs."""
        if len(results) < 2:
            return
        
        metrics_list = [result.metrics for result in results]
        
        # Compute mean and confidence intervals
        for metric_name in BenchmarkMetrics.__annotations__:
            values = [getattr(metrics, metric_name) for metrics in metrics_list]
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # T-distribution for confidence intervals
            from scipy.stats import t
            alpha = 1 - confidence_level
            dof = len(values) - 1
            t_val = t.ppf(1 - alpha/2, dof)
            
            margin_error = t_val * std_val / np.sqrt(len(values))
            
            self.logger.info(f"  {metric_name}: {mean_val:.6f} ± {margin_error:.6f}")
    
    def _generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'summary': {},
            'detailed_results': [],
            'statistical_analysis': {},
            'rankings': {}
        }
        
        # Group results by model and dataset
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            grouped_results[result.model_name][result.dataset_name].append(result)
        
        # Compute summary statistics
        for model_name, datasets in grouped_results.items():
            model_summary = {}
            
            for dataset_name, runs in datasets.items():
                metrics_values = defaultdict(list)
                
                for run in runs:
                    for field_name, field_value in asdict(run.metrics).items():
                        metrics_values[field_name].append(field_value)
                
                # Compute statistics
                dataset_stats = {}
                for metric, values in metrics_values.items():
                    dataset_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                
                model_summary[dataset_name] = dataset_stats
            
            report['summary'][model_name] = model_summary
        
        # Store detailed results
        report['detailed_results'] = [asdict(result) for result in results]
        
        # Compute rankings
        report['rankings'] = self._compute_rankings(grouped_results)
        
        return report
    
    def _compute_rankings(self, grouped_results: Dict) -> Dict[str, Dict]:
        """Compute model rankings across different metrics."""
        rankings = defaultdict(dict)
        
        # Define metrics to rank (lower is better except for correlation)
        metrics_to_rank = [
            ('mse', 'lower'),
            ('relative_l2', 'lower'),
            ('spectral_mse', 'lower'),
            ('inference_time', 'lower'),
            ('spectral_correlation', 'higher'),
            ('parameters', 'lower')
        ]
        
        for dataset_name in set().union(*[datasets.keys() for datasets in grouped_results.values()]):
            dataset_rankings = {}
            
            for metric_name, direction in metrics_to_rank:
                model_scores = {}
                
                for model_name, datasets in grouped_results.items():
                    if dataset_name in datasets:
                        runs = datasets[dataset_name]
                        values = [getattr(run.metrics, metric_name) for run in runs]
                        model_scores[model_name] = np.mean(values)
                
                # Sort models
                reverse = (direction == 'higher')
                sorted_models = sorted(model_scores.items(), 
                                     key=lambda x: x[1], reverse=reverse)
                
                dataset_rankings[metric_name] = [
                    {'model': model, 'score': score, 'rank': i+1}
                    for i, (model, score) in enumerate(sorted_models)
                ]
            
            rankings[dataset_name] = dataset_rankings
        
        return dict(rankings)
    
    def _save_results(self, results: List[BenchmarkResult], report: Dict[str, Any]):
        """Save benchmark results and report."""
        # Save raw results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        
        # Save report
        report_file = self.output_dir / "benchmark_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._create_visualizations(results, report)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _create_visualizations(self, results: List[BenchmarkResult], 
                             report: Dict[str, Any]):
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Operator Benchmark Results', fontsize=16)
        
        # Group results for plotting
        grouped_data = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            grouped_data[result.dataset_name][result.model_name].append(result.metrics)
        
        # Plot MSE comparison
        ax = axes[0, 0]
        self._plot_metric_comparison(grouped_data, 'mse', ax, 'MSE Comparison')
        
        # Plot relative L2 error
        ax = axes[0, 1]
        self._plot_metric_comparison(grouped_data, 'relative_l2', ax, 'Relative L2 Error')
        
        # Plot inference time
        ax = axes[1, 0]
        self._plot_metric_comparison(grouped_data, 'inference_time', ax, 'Inference Time (s)')
        
        # Plot spectral correlation
        ax = axes[1, 1]
        self._plot_metric_comparison(grouped_data, 'spectral_correlation', ax, 'Spectral Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create performance vs parameters scatter plot
        self._create_efficiency_plot(results)
    
    def _plot_metric_comparison(self, grouped_data: Dict, metric: str, 
                              ax: plt.Axes, title: str):
        """Plot comparison of a specific metric."""
        datasets = list(grouped_data.keys())
        models = list(set().union(*[models.keys() for models in grouped_data.values()]))
        
        x_pos = np.arange(len(datasets))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            means = []
            stds = []
            
            for dataset in datasets:
                if model in grouped_data[dataset]:
                    values = [getattr(metrics, metric) for metrics in grouped_data[dataset][model]]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x_pos + i * width, means, width, yerr=stds, 
                  label=model, alpha=0.8, capsize=3)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_efficiency_plot(self, results: List[BenchmarkResult]):
        """Create performance vs parameters efficiency plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Group by model
        models = defaultdict(list)
        for result in results:
            models[result.model_name].append(result)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, (model_name, model_results) in enumerate(models.items()):
            mse_values = [result.metrics.mse for result in model_results]
            param_values = [result.metrics.parameters for result in model_results]
            
            ax.scatter(param_values, mse_values, c=[colors[i]], 
                      label=model_name, s=100, alpha=0.7)
        
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('MSE')
        ax.set_title('Model Efficiency: Performance vs Parameters')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_plot.png", dpi=300, bbox_inches='tight')
        plt.close()


class ComparisonStudy:
    """Conduct controlled comparison studies between neural operators."""
    
    def __init__(self, output_dir: str = "comparison_study"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
    
    def conduct_ablation_study(self, base_model: NeuralOperatorBase,
                             ablation_configs: Dict[str, Dict],
                             test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Conduct ablation study to understand component importance."""
        results = {}
        baseline_performance = self._evaluate_model(base_model, test_data)
        results['baseline'] = baseline_performance
        
        for config_name, config_changes in ablation_configs.items():
            self.logger.info(f"Running ablation: {config_name}")
            
            # Create modified model
            modified_model = self._create_modified_model(base_model, config_changes)
            
            # Evaluate
            performance = self._evaluate_model(modified_model, test_data)
            results[config_name] = performance
            
            # Compute relative change
            relative_change = (performance['mse'] - baseline_performance['mse']) / baseline_performance['mse']
            results[config_name]['relative_change'] = relative_change
        
        # Save ablation study results
        self._save_ablation_results(results)
        
        return results
    
    def _evaluate_model(self, model: NeuralOperatorBase, 
                       test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model and return key metrics."""
        model.eval()
        inputs, targets = test_data
        
        with torch.no_grad():
            predictions = model(inputs)
        
        mse = torch.mean((predictions - targets)**2).item()
        relative_l2 = (torch.norm((predictions - targets).flatten()) / 
                      torch.norm(targets.flatten())).item()
        
        return {
            'mse': mse,
            'relative_l2': relative_l2,
            'parameters': model.count_parameters()
        }
    
    def _create_modified_model(self, base_model: NeuralOperatorBase, 
                             config_changes: Dict) -> NeuralOperatorBase:
        """Create modified model based on configuration changes."""
        # This would need to be implemented based on specific model architectures
        # For now, return the base model
        return base_model
    
    def _save_ablation_results(self, results: Dict[str, Any]):
        """Save ablation study results."""
        with open(self.output_dir / "ablation_results.json", 'w') as f:
            json.dump(results, f, indent=2)