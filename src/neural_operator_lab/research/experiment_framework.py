# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Experimental framework for hypothesis-driven neural operator research."""

import torch
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..base import NeuralOperatorBase
from .statistical_analysis import StatisticalAnalyzer


@dataclass
class Hypothesis:
    """Research hypothesis with testable predictions."""
    name: str
    description: str
    predictions: List[str]
    success_metrics: Dict[str, float]  # metric_name -> threshold
    variables: Dict[str, Any]  # Independent variables
    null_hypothesis: str


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    hypothesis: Hypothesis
    model_configs: Dict[str, Dict[str, Any]]
    dataset_configs: Dict[str, Dict[str, Any]]
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    num_runs: int = 3
    random_seeds: Optional[List[int]] = None


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    hypothesis_name: str
    model_name: str
    dataset_name: str
    run_id: int
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    model_size: int
    config: Dict[str, Any]
    timestamp: str
    random_seed: int


class HypothesisTester:
    """Framework for testing research hypotheses."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Store all results
        self.experiment_results = []
        self.hypothesis_results = defaultdict(list)
    
    def test_hypothesis(self, experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """Test a research hypothesis through controlled experiments."""
        hypothesis = experiment_config.hypothesis
        self.logger.info(f"Testing hypothesis: {hypothesis.name}")
        
        # Generate random seeds if not provided
        if experiment_config.random_seeds is None:
            np.random.seed(42)
            experiment_config.random_seeds = np.random.randint(0, 10000, experiment_config.num_runs).tolist()
        
        experiment_results = []
        
        # Run experiments for each model-dataset combination
        for model_name, model_config in experiment_config.model_configs.items():
            for dataset_name, dataset_config in experiment_config.dataset_configs.items():
                
                self.logger.info(f"  Model: {model_name}, Dataset: {dataset_name}")
                
                # Multiple runs for statistical significance
                for run_id, seed in enumerate(experiment_config.random_seeds):
                    self.logger.info(f"    Run {run_id + 1}/{experiment_config.num_runs} (seed={seed})")
                    
                    result = self._run_single_experiment(
                        experiment_config, model_name, model_config,
                        dataset_name, dataset_config, run_id, seed
                    )
                    
                    experiment_results.append(result)
                    self.experiment_results.append(result)
                    self.hypothesis_results[hypothesis.name].append(result)
        
        # Analyze results and test hypothesis
        hypothesis_analysis = self._analyze_hypothesis(hypothesis, experiment_results)
        
        # Save results
        self._save_experiment_results(experiment_config, experiment_results, hypothesis_analysis)
        
        return hypothesis_analysis
    
    def _run_single_experiment(self, 
                              experiment_config: ExperimentConfig,
                              model_name: str,
                              model_config: Dict[str, Any],
                              dataset_name: str,
                              dataset_config: Dict[str, Any],
                              run_id: int,
                              seed: int) -> ExperimentResult:
        """Run a single experiment with specified configuration."""
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        model = self._create_model(model_name, model_config)
        
        # Create dataset
        train_data, val_data, test_data = self._create_dataset(dataset_name, dataset_config)
        
        # Training
        start_time = time.time()
        trained_model = self._train_model(model, train_data, val_data, experiment_config.training_config)
        training_time = time.time() - start_time
        
        # Evaluation
        start_time = time.time()
        metrics = self._evaluate_model(trained_model, test_data, experiment_config.evaluation_metrics)
        inference_time = time.time() - start_time
        
        # Model size
        model_size = trained_model.count_parameters()
        
        return ExperimentResult(
            experiment_name=experiment_config.name,
            hypothesis_name=experiment_config.hypothesis.name,
            model_name=model_name,
            dataset_name=dataset_name,
            run_id=run_id,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            config={
                'model': model_config,
                'dataset': dataset_config,
                'training': experiment_config.training_config
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            random_seed=seed
        )
    
    def _create_model(self, model_name: str, model_config: Dict[str, Any]) -> NeuralOperatorBase:
        """Create model from configuration."""
        # This would be implemented based on available model types
        # For now, create a dummy model
        
        from ..models.fourier import FourierNeuralOperator
        from ..models.transformer import TransformerNeuralOperator
        from ..models.probabilistic import ProbabilisticNeuralOperator
        
        if model_name.lower().startswith('fno'):
            return FourierNeuralOperator(**model_config)
        elif model_name.lower().startswith('tno'):
            return TransformerNeuralOperator(**model_config)
        elif model_name.lower().startswith('pno'):
            return ProbabilisticNeuralOperator(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _create_dataset(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Tuple:
        """Create dataset from configuration."""
        # Create synthetic data for testing
        batch_size = dataset_config.get('batch_size', 32)
        resolution = dataset_config.get('resolution', 64)
        input_dim = dataset_config.get('input_dim', 3)
        output_dim = dataset_config.get('output_dim', 1)
        
        # Generate random data
        if 'navier_stokes' in dataset_name.lower():
            shape = (batch_size, resolution, resolution, input_dim)
        elif 'burgers' in dataset_name.lower():
            shape = (batch_size, resolution, input_dim)
        else:
            shape = (batch_size, resolution, resolution, input_dim)
        
        train_x = torch.randn(shape)
        train_y = torch.randn(shape[:-1] + (output_dim,))
        
        val_x = torch.randn(shape)
        val_y = torch.randn(shape[:-1] + (output_dim,))
        
        test_x = torch.randn(shape)
        test_y = torch.randn(shape[:-1] + (output_dim,))
        
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)
    
    def _train_model(self, model: NeuralOperatorBase, 
                    train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor],
                    training_config: Dict[str, Any]) -> NeuralOperatorBase:
        """Train model with specified configuration."""
        # Simple training loop for testing
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 1e-3))
        
        train_x, train_y = train_data
        epochs = training_config.get('epochs', 10)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(train_x)
            loss = torch.nn.functional.mse_loss(pred, train_y)
            loss.backward()
            optimizer.step()
        
        return model
    
    def _evaluate_model(self, model: NeuralOperatorBase, 
                       test_data: Tuple[torch.Tensor, torch.Tensor],
                       metrics: List[str]) -> Dict[str, float]:
        """Evaluate model on test data."""
        # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
        test_x, test_y = test_data
        
        with torch.no_grad():
            pred = model(test_x)
        
        results = {}
        
        for metric in metrics:
            if metric == 'mse':
                results[metric] = torch.nn.functional.mse_loss(pred, test_y).item()
            elif metric == 'relative_l2':
                results[metric] = (torch.norm((pred - test_y).flatten()) / 
                                 torch.norm(test_y.flatten())).item()
            elif metric == 'max_error':
                results[metric] = torch.max(torch.abs(pred - test_y)).item()
            # Add more metrics as needed
        
        return results
    
    def _analyze_hypothesis(self, hypothesis: Hypothesis, 
                          results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze experiment results to test hypothesis."""
        
        # Group results by model and dataset
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            grouped_results[result.model_name][result.dataset_name].append(result)
        
        # Statistical analysis
        statistical_tests = {}
        
        # Test predictions
        prediction_results = {}
        
        for prediction in hypothesis.predictions:
            prediction_results[prediction] = self._test_prediction(prediction, grouped_results, hypothesis)
        
        # Overall hypothesis test
        hypothesis_supported = all(pred_result['supported'] for pred_result in prediction_results.values())
        
        # Effect sizes and confidence intervals
        effect_sizes = self._compute_effect_sizes(grouped_results)
        
        # Success metrics evaluation
        success_metrics_met = self._evaluate_success_metrics(hypothesis, grouped_results)
        
        analysis = {
            'hypothesis': asdict(hypothesis),
            'hypothesis_supported': hypothesis_supported,
            'prediction_results': prediction_results,
            'success_metrics_met': success_metrics_met,
            'statistical_tests': statistical_tests,
            'effect_sizes': effect_sizes,
            'summary_statistics': self._compute_summary_statistics(grouped_results),
            'recommendations': self._generate_recommendations(hypothesis_supported, prediction_results)
        }
        
        return analysis
    
    def _test_prediction(self, prediction: str, 
                        grouped_results: Dict, 
                        hypothesis: Hypothesis) -> Dict[str, Any]:
        """Test a specific prediction from the hypothesis."""
        
        # Parse prediction (simplified - would need more sophisticated parsing)
        # For now, assume predictions are in format "ModelA > ModelB on DatasetX"
        
        # Extract comparison info
        if ' > ' in prediction:
            parts = prediction.split(' > ')
            model1 = parts[0].strip()
            rest = parts[1].strip()
            
            if ' on ' in rest:
                model2, dataset = rest.split(' on ')
                model2 = model2.strip()
                dataset = dataset.strip()
                
                # Get results for comparison
                if (model1 in grouped_results and dataset in grouped_results[model1] and
                    model2 in grouped_results and dataset in grouped_results[model2]):
                    
                    results1 = grouped_results[model1][dataset]
                    results2 = grouped_results[model2][dataset]
                    
                    # Extract metric values (assuming MSE)
                    values1 = [r.metrics.get('mse', float('inf')) for r in results1]
                    values2 = [r.metrics.get('mse', float('inf')) for r in results2]
                    
                    # Statistical test
                    comparison = self.statistical_analyzer.compare_methods(
                        values1, values2, model1, model2, 'MSE'
                    )
                    
                    # For MSE, lower is better, so model1 > model2 means values1 < values2
                    mean_diff = comparison.summary['difference']['mean_diff']
                    supported = mean_diff < 0 and comparison.test_results['comparison'].is_significant
                    
                    return {
                        'supported': supported,
                        'statistical_test': comparison,
                        'mean_difference': mean_diff,
                        'effect_size': comparison.test_results.get('effect_size', {}).effect_size
                    }
        
        return {'supported': False, 'reason': 'Could not parse or evaluate prediction'}
    
    def _evaluate_success_metrics(self, hypothesis: Hypothesis, 
                                grouped_results: Dict) -> Dict[str, bool]:
        """Evaluate whether success metrics are met."""
        results = {}
        
        for metric_name, threshold in hypothesis.success_metrics.items():
            # Find best performance across all models/datasets
            best_value = float('inf') if 'error' in metric_name or 'mse' in metric_name else float('-inf')
            
            for model_results in grouped_results.values():
                for dataset_results in model_results.values():
                    for result in dataset_results:
                        if metric_name in result.metrics:
                            value = result.metrics[metric_name]
                            
                            if 'error' in metric_name or 'mse' in metric_name:
                                best_value = min(best_value, value)
                            else:
                                best_value = max(best_value, value)
            
            # Check if threshold is met
            if 'error' in metric_name or 'mse' in metric_name:
                results[metric_name] = best_value <= threshold
            else:
                results[metric_name] = best_value >= threshold
        
        return results
    
    def _compute_effect_sizes(self, grouped_results: Dict) -> Dict[str, float]:
        """Compute effect sizes for model comparisons."""
        effect_sizes = {}
        
        # Compare all pairs of models
        models = list(grouped_results.keys())
        
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                
                # Find common datasets
                common_datasets = set(grouped_results[model1].keys()) & set(grouped_results[model2].keys())
                
                for dataset in common_datasets:
                    results1 = grouped_results[model1][dataset]
                    results2 = grouped_results[model2][dataset]
                    
                    # MSE values
                    values1 = [r.metrics.get('mse', float('inf')) for r in results1]
                    values2 = [r.metrics.get('mse', float('inf')) for r in results2]
                    
                    # Cohen's d
                    mean1, mean2 = np.mean(values1), np.mean(values2)
                    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                    pooled_std = np.sqrt(((len(values1)-1)*std1**2 + (len(values2)-1)*std2**2) / 
                                       (len(values1) + len(values2) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (mean1 - mean2) / pooled_std
                        effect_sizes[f"{model1}_vs_{model2}_{dataset}"] = cohens_d
        
        return effect_sizes
    
    def _compute_summary_statistics(self, grouped_results: Dict) -> Dict[str, Dict]:
        """Compute summary statistics for all model-dataset combinations."""
        summary = {}
        
        for model_name, model_results in grouped_results.items():
            summary[model_name] = {}
            
            for dataset_name, results in model_results.items():
                metrics_values = defaultdict(list)
                
                for result in results:
                    for metric_name, metric_value in result.metrics.items():
                        metrics_values[metric_name].append(metric_value)
                
                dataset_summary = {}
                for metric_name, values in metrics_values.items():
                    dataset_summary[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
                
                summary[model_name][dataset_name] = dataset_summary
        
        return summary
    
    def _generate_recommendations(self, hypothesis_supported: bool, 
                                prediction_results: Dict) -> List[str]:
        """Generate research recommendations based on results."""
        recommendations = []
        
        if hypothesis_supported:
            recommendations.append("Hypothesis is supported by the experimental evidence.")
            recommendations.append("Consider conducting follow-up studies to further validate findings.")
            recommendations.append("Explore the mechanisms underlying the observed effects.")
        else:
            recommendations.append("Hypothesis is not supported by the current evidence.")
            recommendations.append("Consider revising the hypothesis or experimental design.")
            recommendations.append("Examine potential confounding factors or methodological issues.")
        
        # Specific recommendations based on predictions
        unsupported_predictions = [pred for pred, result in prediction_results.items() 
                                 if not result['supported']]
        
        if unsupported_predictions:
            recommendations.append("The following predictions were not supported:")
            for pred in unsupported_predictions:
                recommendations.append(f"  - {pred}")
            recommendations.append("Consider investigating why these predictions failed.")
        
        return recommendations
    
    def _save_experiment_results(self, experiment_config: ExperimentConfig,
                               results: List[ExperimentResult],
                               analysis: Dict[str, Any]):
        """Save experiment results and analysis."""
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_config.name
        exp_dir.mkdir(exist_ok=True)
        
        # Save raw results
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        
        # Save analysis
        analysis_file = exp_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save configuration
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(experiment_config), f, indent=2, default=str)
        
        # Generate visualizations
        self._create_experiment_visualizations(results, analysis, exp_dir)
        
        self.logger.info(f"Experiment results saved to {exp_dir}")
    
    def _create_experiment_visualizations(self, results: List[ExperimentResult],
                                        analysis: Dict[str, Any],
                                        output_dir: Path):
        """Create visualizations for experiment results."""
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Experiment Results: {results[0].experiment_name}', fontsize=16)
        
        # Group results for plotting
        model_performance = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            model_performance[result.model_name][result.dataset_name].extend(
                [result.metrics.get('mse', float('inf'))]
            )
        
        # MSE comparison
        self._plot_performance_comparison(model_performance, 'MSE', axes[0, 0])
        
        # Training time comparison
        training_times = defaultdict(lambda: defaultdict(list))
        for result in results:
            training_times[result.model_name][result.dataset_name].append(result.training_time)
        
        self._plot_performance_comparison(training_times, 'Training Time (s)', axes[0, 1])
        
        # Model size comparison
        model_sizes = defaultdict(lambda: defaultdict(list))
        for result in results:
            model_sizes[result.model_name][result.dataset_name].append(result.model_size)
        
        self._plot_performance_comparison(model_sizes, 'Model Size (params)', axes[1, 0])
        
        # Success metrics visualization
        self._plot_success_metrics(analysis.get('success_metrics_met', {}), axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "experiment_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, data: Dict, metric_name: str, ax: plt.Axes):
        """Plot performance comparison across models and datasets."""
        models = list(data.keys())
        datasets = list(set().union(*[d.keys() for d in data.values()]))
        
        x_pos = np.arange(len(datasets))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            means = []
            stds = []
            
            for dataset in datasets:
                if dataset in data[model]:
                    values = data[model][dataset]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x_pos + i * width, means, width, yerr=stds,
                   label=model, alpha=0.8, capsize=3)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_success_metrics(self, success_metrics: Dict[str, bool], ax: plt.Axes):
        """Plot success metrics achievement."""
        if not success_metrics:
            ax.text(0.5, 0.5, 'No success metrics defined', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Metrics')
            return
        
        metrics = list(success_metrics.keys())
        achieved = [success_metrics[m] for m in metrics]
        
        colors = ['green' if a else 'red' for a in achieved]
        
        ax.barh(metrics, [1] * len(metrics), color=colors, alpha=0.7)
        ax.set_xlabel('Achievement')
        ax.set_title('Success Metrics Achievement')
        ax.set_xlim(0, 1)
        
        # Add text labels
        for i, (metric, acheived) in enumerate(zip(metrics, achieved)):
            ax.text(0.5, i, '✓' if acheived else '✗', 
                   ha='center', va='center', fontsize=16, color='white', weight='bold')


class ExperimentManager:
    """Manage multiple experiments and research campaigns."""
    
    def __init__(self, output_dir: str = "research_campaign"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.hypothesis_tester = HypothesisTester(str(self.output_dir / "experiments"))
        self.logger = logging.getLogger(__name__)
        
        self.campaign_results = []
    
    def run_research_campaign(self, experiments: List[ExperimentConfig]) -> Dict[str, Any]:
        """Run a complete research campaign with multiple experiments."""
        
        self.logger.info(f"Starting research campaign with {len(experiments)} experiments")
        
        campaign_results = []
        
        for i, experiment in enumerate(experiments):
            self.logger.info(f"Running experiment {i+1}/{len(experiments)}: {experiment.name}")
            
            result = self.hypothesis_tester.test_hypothesis(experiment)
            campaign_results.append({
                'experiment': experiment,
                'result': result
            })
        
        # Generate campaign summary
        campaign_summary = self._generate_campaign_summary(campaign_results)
        
        # Save campaign results
        self._save_campaign_results(experiments, campaign_results, campaign_summary)
        
        return campaign_summary
    
    def _generate_campaign_summary(self, campaign_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of entire research campaign."""
        
        total_experiments = len(campaign_results)
        supported_hypotheses = sum(1 for r in campaign_results if r['result']['hypothesis_supported'])
        
        # Collect all statistical tests
        all_effect_sizes = []
        for result in campaign_results:
            all_effect_sizes.extend(result['result']['effect_sizes'].values())
        
        summary = {
            'total_experiments': total_experiments,
            'supported_hypotheses': supported_hypotheses,
            'hypothesis_support_rate': supported_hypotheses / total_experiments,
            'overall_effect_sizes': {
                'mean': np.mean(all_effect_sizes) if all_effect_sizes else 0,
                'std': np.std(all_effect_sizes) if all_effect_sizes else 0,
                'distribution': all_effect_sizes
            },
            'experiment_summaries': [
                {
                    'name': r['experiment'].name,
                    'hypothesis_supported': r['result']['hypothesis_supported'],
                    'success_metrics_met': r['result']['success_metrics_met']
                }
                for r in campaign_results
            ]
        }
        
        return summary
    
    def _save_campaign_results(self, experiments: List[ExperimentConfig],
                             campaign_results: List[Dict],
                             campaign_summary: Dict[str, Any]):
        """Save complete campaign results."""
        
        # Save campaign summary
        summary_file = self.output_dir / "campaign_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(campaign_summary, f, indent=2, default=str)
        
        # Generate campaign report
        report = self._generate_campaign_report(campaign_results, campaign_summary)
        
        report_file = self.output_dir / "campaign_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Campaign results saved to {self.output_dir}")
    
    def _generate_campaign_report(self, campaign_results: List[Dict],
                                campaign_summary: Dict[str, Any]) -> str:
        """Generate comprehensive campaign report."""
        
        lines = [
            "# Research Campaign Report",
            "",
            f"**Total Experiments:** {campaign_summary['total_experiments']}",
            f"**Supported Hypotheses:** {campaign_summary['supported_hypotheses']}",
            f"**Support Rate:** {campaign_summary['hypothesis_support_rate']:.1%}",
            "",
            "## Experiment Results",
            ""
        ]
        
        for i, result_data in enumerate(campaign_results):
            experiment = result_data['experiment']
            result = result_data['result']
            
            lines.extend([
                f"### {i+1}. {experiment.name}",
                "",
                f"**Hypothesis:** {experiment.hypothesis.name}",
                f"**Description:** {experiment.hypothesis.description}",
                f"**Supported:** {'✓' if result['hypothesis_supported'] else '✗'}",
                "",
                "**Predictions:**"
            ])
            
            for prediction in experiment.hypothesis.predictions:
                pred_result = result['prediction_results'].get(prediction, {})
                supported = pred_result.get('supported', False)
                lines.append(f"- {prediction} {'✓' if supported else '✗'}")
            
            lines.extend(["", "**Success Metrics:**"])
            
            for metric, met in result['success_metrics_met'].items():
                lines.append(f"- {metric}: {'✓' if met else '✗'}")
            
            lines.extend(["", "**Recommendations:**"])
            
            for rec in result['recommendations']:
                lines.append(f"- {rec}")
            
            lines.extend(["", "---", ""])
        
        lines.extend([
            "## Overall Conclusions",
            "",
            f"This research campaign investigated {campaign_summary['total_experiments']} hypotheses ",
            f"with a {campaign_summary['hypothesis_support_rate']:.1%} support rate. ",
            "",
            "Key findings:",
            f"- Mean effect size: {campaign_summary['overall_effect_sizes']['mean']:.3f}",
            f"- Effect size variability: {campaign_summary['overall_effect_sizes']['std']:.3f}",
        ])
        
        return "\n".join(lines)