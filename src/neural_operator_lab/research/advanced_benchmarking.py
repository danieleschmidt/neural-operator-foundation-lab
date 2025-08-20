# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Advanced Research Benchmarking Pipeline

Comprehensive benchmarking suite for neural operators with statistical analysis,
reproducibility guarantees, and academic-grade reporting.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import hashlib
import logging
from scipy import stats
from abc import ABC, abstractmethod

from neural_operator_lab.base import NeuralOperatorBase
from neural_operator_lab.research.statistical_analysis import StatisticalAnalyzer


@dataclass
class BenchmarkResult:
    """Single benchmark result with metadata."""
    model_name: str
    dataset_name: str
    metric_name: str
    value: float
    std: Optional[float] = None
    num_runs: int = 1
    timestamp: str = field(default_factory=lambda: str(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'metric_name': self.metric_name,
            'value': self.value,
            'std': self.std,
            'num_runs': self.num_runs,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ComparativeStudyResult:
    """Results from a comparative study."""
    baseline_results: List[BenchmarkResult]
    novel_results: List[BenchmarkResult]
    statistical_tests: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendations: List[str]
    
    def is_statistically_significant(self, metric: str, alpha: float = 0.05) -> bool:
        """Check if improvement is statistically significant."""
        p_value = self.statistical_tests.get(metric, {}).get('p_value', 1.0)
        return p_value < alpha


class BenchmarkMetric(ABC):
    """Abstract base class for benchmark metrics."""
    
    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        """Compute the metric."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass
    
    @property
    def higher_is_better(self) -> bool:
        """Whether higher values are better."""
        return False


class MSEMetric(BenchmarkMetric):
    """Mean Squared Error metric."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        return torch.mean((predictions - targets) ** 2).item()
    
    def get_name(self) -> str:
        return "mse"


class RelativeL2Metric(BenchmarkMetric):
    """Relative L2 error metric."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        num = torch.norm(predictions - targets, dim=(-2, -1))
        den = torch.norm(targets, dim=(-2, -1))
        return torch.mean(num / den).item()
    
    def get_name(self) -> str:
        return "relative_l2"


class SpectralMetric(BenchmarkMetric):
    """Spectral error in Fourier domain."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        pred_fft = torch.fft.fftn(predictions, dim=(-2, -1))
        target_fft = torch.fft.fftn(targets, dim=(-2, -1))
        
        spectral_error = torch.mean(torch.abs(pred_fft - target_fft) ** 2)
        return spectral_error.item()
    
    def get_name(self) -> str:
        return "spectral_error"


class ConservationMetric(BenchmarkMetric):
    """Conservation law violation metric."""
    
    def __init__(self, conservation_type: str = "mass"):
        self.conservation_type = conservation_type
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        if self.conservation_type == "mass":
            # Check mass conservation (simplified)
            pred_mass = torch.sum(predictions, dim=(-2, -1))
            target_mass = torch.sum(targets, dim=(-2, -1))
            violation = torch.mean(torch.abs(pred_mass - target_mass) / torch.abs(target_mass))
            return violation.item()
        
        return 0.0
    
    def get_name(self) -> str:
        return f"conservation_{self.conservation_type}"


class AdvancedBenchmarkSuite:
    """Advanced benchmarking suite for neural operators."""
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        random_seed: int = 42,
        num_statistical_runs: int = 5
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_seed = random_seed
        self.num_statistical_runs = num_statistical_runs
        
        # Initialize components
        self.metrics: Dict[str, BenchmarkMetric] = {}
        self.statistical_analyzer = StatisticalAnalyzer()
        self.results: List[BenchmarkResult] = []
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_dir / "benchmark.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Register default metrics
        self.register_metric(MSEMetric())
        self.register_metric(RelativeL2Metric())
        self.register_metric(SpectralMetric())
        self.register_metric(ConservationMetric("mass"))
        
    def register_metric(self, metric: BenchmarkMetric):
        """Register a benchmark metric."""
        self.metrics[metric.get_name()] = metric
        
    def benchmark_model(
        self,
        model: NeuralOperatorBase,
        test_data: torch.utils.data.DataLoader,
        model_name: str,
        dataset_name: str,
        device: str = "cuda",
        compute_uncertainty: bool = False
    ) -> List[BenchmarkResult]:
        """Benchmark a single model."""
        # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
        model = model.to(device)
        
        results = []
        
        # Multiple runs for statistical significance
        all_predictions = []
        all_targets = []
        
        self.logger.info(f"Benchmarking {model_name} on {dataset_name}")
        
        for run in range(self.num_statistical_runs):
            torch.manual_seed(self.random_seed + run)
            np.random.seed(self.random_seed + run)
            
            run_predictions = []
            run_targets = []
            
            with torch.no_grad():
                for batch in test_data:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    else:
                        inputs = batch['input']
                        targets = batch['target']
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    if compute_uncertainty and hasattr(model, 'forward'):
                        try:
                            predictions = model(inputs, return_uncertainty=True)[0]
                        except:
                            predictions = model(inputs)
                    else:
                        predictions = model(inputs)
                    
                    run_predictions.append(predictions.cpu())
                    run_targets.append(targets.cpu())
            
            all_predictions.append(torch.cat(run_predictions, dim=0))
            all_targets.append(torch.cat(run_targets, dim=0))
        
        # Compute metrics across runs
        for metric_name, metric in self.metrics.items():
            metric_values = []
            
            for pred, target in zip(all_predictions, all_targets):
                try:
                    value = metric.compute(pred, target)
                    metric_values.append(value)
                except Exception as e:
                    self.logger.warning(f"Failed to compute {metric_name}: {e}")
                    continue
            
            if metric_values:
                mean_value = np.mean(metric_values)
                std_value = np.std(metric_values) if len(metric_values) > 1 else None
                
                result = BenchmarkResult(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    value=mean_value,
                    std=std_value,
                    num_runs=len(metric_values),
                    metadata={
                        'device': device,
                        'compute_uncertainty': compute_uncertainty,
                        'random_seed': self.random_seed
                    }
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def comparative_study(
        self,
        baseline_models: Dict[str, NeuralOperatorBase],
        novel_models: Dict[str, NeuralOperatorBase],
        test_datasets: Dict[str, torch.utils.data.DataLoader],
        study_name: str,
        device: str = "cuda"
    ) -> ComparativeStudyResult:
        """Run comprehensive comparative study."""
        self.logger.info(f"Starting comparative study: {study_name}")
        
        baseline_results = []
        novel_results = []
        
        # Benchmark baseline models
        for model_name, model in baseline_models.items():
            for dataset_name, dataset in test_datasets.items():
                results = self.benchmark_model(
                    model, dataset, f"baseline_{model_name}", dataset_name, device
                )
                baseline_results.extend(results)
        
        # Benchmark novel models
        for model_name, model in novel_models.items():
            for dataset_name, dataset in test_datasets.items():
                results = self.benchmark_model(
                    model, dataset, f"novel_{model_name}", dataset_name, device
                )
                novel_results.extend(results)
        
        # Statistical analysis
        statistical_tests = {}
        effect_sizes = {}
        confidence_intervals = {}
        recommendations = []
        
        # Group results by metric and dataset
        metrics_datasets = set(
            (r.metric_name, r.dataset_name) 
            for r in baseline_results + novel_results
        )
        
        for metric_name, dataset_name in metrics_datasets:
            baseline_values = [
                r.value for r in baseline_results 
                if r.metric_name == metric_name and r.dataset_name == dataset_name
            ]
            novel_values = [
                r.value for r in novel_results
                if r.metric_name == metric_name and r.dataset_name == dataset_name
            ]
            
            if baseline_values and novel_values:
                # Statistical tests
                key = f"{metric_name}_{dataset_name}"
                
                # T-test for mean comparison
                t_stat, p_value = stats.ttest_ind(baseline_values, novel_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) +
                     (len(novel_values) - 1) * np.var(novel_values, ddof=1)) /
                    (len(baseline_values) + len(novel_values) - 2)
                )
                
                if pooled_std > 0:
                    cohens_d = (np.mean(novel_values) - np.mean(baseline_values)) / pooled_std
                else:
                    cohens_d = 0.0
                
                statistical_tests[key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(baseline_values) + len(novel_values) - 2
                }
                
                effect_sizes[key] = cohens_d
                
                # Confidence intervals
                novel_mean = np.mean(novel_values)
                novel_sem = stats.sem(novel_values)
                ci_lower, ci_upper = stats.t.interval(
                    0.95, len(novel_values) - 1, novel_mean, novel_sem
                )
                confidence_intervals[key] = (ci_lower, ci_upper)
                
                # Generate recommendations
                if p_value < 0.05:
                    improvement = "improvement" if cohens_d > 0 else "degradation"
                    effect_size_desc = self._effect_size_description(abs(cohens_d))
                    recommendations.append(
                        f"{metric_name} on {dataset_name}: Statistically significant "
                        f"{improvement} with {effect_size_desc} effect size (d={cohens_d:.3f})"
                    )
                else:
                    recommendations.append(
                        f"{metric_name} on {dataset_name}: No statistically significant difference"
                    )
        
        study_result = ComparativeStudyResult(
            baseline_results=baseline_results,
            novel_results=novel_results,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations
        )
        
        # Save study results
        self.save_comparative_study(study_result, study_name)
        
        return study_result
    
    def _effect_size_description(self, cohens_d: float) -> str:
        """Describe effect size magnitude."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_research_report(
        self,
        study_result: ComparativeStudyResult,
        study_name: str,
        include_visualizations: bool = True
    ) -> str:
        """Generate comprehensive research report."""
        report_path = self.output_dir / f"{study_name}_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Neural Operator Comparative Study: {study_name}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            significant_results = [
                rec for rec in study_result.recommendations 
                if "significant" in rec and "No statistically" not in rec
            ]
            f.write(f"- **Total Comparisons**: {len(study_result.statistical_tests)}\n")
            f.write(f"- **Statistically Significant Results**: {len(significant_results)}\n")
            f.write(f"- **Baseline Models**: {len(set(r.model_name for r in study_result.baseline_results))}\n")
            f.write(f"- **Novel Models**: {len(set(r.model_name for r in study_result.novel_results))}\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            for rec in study_result.recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Statistical Results Table
            f.write("## Statistical Analysis\n\n")
            f.write("| Metric | Dataset | p-value | Effect Size (Cohen's d) | 95% CI |\n")
            f.write("|--------|---------|---------|-------------------------|--------|\n")
            
            for key in study_result.statistical_tests.keys():
                metric, dataset = key.rsplit('_', 1)
                p_val = study_result.statistical_tests[key]['p_value']
                effect_size = study_result.effect_sizes[key]
                ci_lower, ci_upper = study_result.confidence_intervals[key]
                
                f.write(f"| {metric} | {dataset} | {p_val:.4f} | {effect_size:.3f} | "
                       f"[{ci_lower:.3f}, {ci_upper:.3f}] |\n")
            f.write("\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write(f"- **Statistical Runs**: {self.num_statistical_runs}\n")
            f.write(f"- **Random Seed**: {self.random_seed}\n")
            f.write("- **Statistical Tests**: Independent t-tests\n")
            f.write("- **Effect Size**: Cohen's d\n")
            f.write("- **Significance Level**: α = 0.05\n\n")
            
            # Reproducibility Information
            f.write("## Reproducibility\n\n")
            f.write("### Software Versions\n")
            f.write(f"- PyTorch: {torch.__version__}\n")
            f.write(f"- NumPy: {np.__version__}\n")
            f.write("\n")
            
            # Generate hash for reproducibility
            study_hash = self._generate_study_hash(study_result)
            f.write(f"### Study Hash\n")
            f.write(f"```\n{study_hash}\n```\n\n")
        
        if include_visualizations:
            self._generate_visualizations(study_result, study_name)
        
        self.logger.info(f"Research report generated: {report_path}")
        return str(report_path)
    
    def _generate_study_hash(self, study_result: ComparativeStudyResult) -> str:
        """Generate reproducibility hash for study."""
        # Create deterministic representation
        study_data = {
            'baseline_results': [r.to_dict() for r in study_result.baseline_results],
            'novel_results': [r.to_dict() for r in study_result.novel_results],
            'statistical_tests': study_result.statistical_tests,
            'random_seed': self.random_seed,
            'num_runs': self.num_statistical_runs
        }
        
        study_str = json.dumps(study_data, sort_keys=True)
        return hashlib.sha256(study_str.encode()).hexdigest()
    
    def _generate_visualizations(self, study_result: ComparativeStudyResult, study_name: str):
        """Generate visualization plots."""
        # Performance comparison plot
        plt.figure(figsize=(12, 8))
        
        # Group results by metric
        metrics = set(r.metric_name for r in study_result.baseline_results + study_result.novel_results)
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            
            baseline_values = [r.value for r in study_result.baseline_results if r.metric_name == metric]
            novel_values = [r.value for r in study_result.novel_results if r.metric_name == metric]
            
            if baseline_values and novel_values:
                plt.boxplot([baseline_values, novel_values], labels=['Baseline', 'Novel'])
                plt.title(f'{metric.replace("_", " ").title()}')
                plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{study_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Effect size visualization
        plt.figure(figsize=(10, 6))
        effect_sizes = list(study_result.effect_sizes.values())
        labels = list(study_result.effect_sizes.keys())
        
        colors = ['green' if es > 0 else 'red' for es in effect_sizes]
        plt.barh(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
        plt.yticks(range(len(labels)), [l.replace('_', ' ') for l in labels])
        plt.xlabel("Effect Size (Cohen's d)")
        plt.title("Effect Sizes Across Metrics and Datasets")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{study_name}_effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename: Optional[str] = None):
        """Save all benchmark results."""
        if filename is None:
            filename = f"benchmark_results_{int(time.time())}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'results': [r.to_dict() for r in self.results],
            'metadata': {
                'num_results': len(self.results),
                'random_seed': self.random_seed,
                'num_statistical_runs': self.num_statistical_runs,
                'timestamp': str(time.time())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def save_comparative_study(self, study_result: ComparativeStudyResult, study_name: str):
        """Save comparative study results."""
        filepath = self.output_dir / f"{study_name}_study.json"
        
        study_data = {
            'baseline_results': [r.to_dict() for r in study_result.baseline_results],
            'novel_results': [r.to_dict() for r in study_result.novel_results],
            'statistical_tests': study_result.statistical_tests,
            'effect_sizes': study_result.effect_sizes,
            'confidence_intervals': study_result.confidence_intervals,
            'recommendations': study_result.recommendations,
            'study_hash': self._generate_study_hash(study_result)
        }
        
        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        self.logger.info(f"Comparative study saved to {filepath}")


def create_synthetic_benchmark_data(
    batch_size: int = 8,
    spatial_resolution: int = 64,
    input_dim: int = 3,
    output_dim: int = 1,
    num_samples: int = 100,
    pde_type: str = "navier_stokes"
) -> torch.utils.data.DataLoader:
    """Create synthetic benchmark data for testing."""
    
    class SyntheticPDEDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, spatial_resolution, input_dim, output_dim, pde_type):
            self.num_samples = num_samples
            self.spatial_resolution = spatial_resolution
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.pde_type = pde_type
            
            # Generate synthetic data
            self.data = self._generate_data()
        
        def _generate_data(self):
            data = []
            for i in range(self.num_samples):
                # Create spatial coordinates
                x = torch.linspace(-1, 1, self.spatial_resolution)
                y = torch.linspace(-1, 1, self.spatial_resolution)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                # Generate input based on PDE type
                if self.pde_type == "navier_stokes":
                    # Velocity field + pressure
                    u = torch.sin(np.pi * X) * torch.cos(np.pi * Y)
                    v = -torch.cos(np.pi * X) * torch.sin(np.pi * Y)  
                    p = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
                    input_field = torch.stack([u, v, p], dim=0)  # [3, H, W]
                    
                    # Target (evolved solution)
                    target_field = input_field * 0.9  # Simplified evolution
                    
                elif self.pde_type == "wave":
                    # Wave equation
                    input_field = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
                    input_field = input_field.unsqueeze(0)  # [1, H, W]
                    
                    target_field = input_field * torch.cos(0.1)  # Time evolution
                    
                else:  # Default case
                    input_field = torch.randn(self.input_dim, self.spatial_resolution, self.spatial_resolution)
                    target_field = torch.randn(self.output_dim, self.spatial_resolution, self.spatial_resolution)
                
                data.append((input_field, target_field))
            
            return data
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SyntheticPDEDataset(num_samples, spatial_resolution, input_dim, output_dim, pde_type)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)