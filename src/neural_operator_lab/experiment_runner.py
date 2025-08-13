"""Advanced experiment runner with hyperparameter optimization and reproducibility."""

import time
import json
import hashlib
import random
import os
import math
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Iterator
from dataclasses import dataclass, field, asdict
import itertools
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .robust_framework import robust_logger, robust_execution, error_boundary
    from .performance_optimizer import profile, optimize_parallel_execution
    from .research_framework import ExperimentResult, ExperimentDesign, StatisticalAnalyzer
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    optimization_metric: str = "loss"
    optimization_direction: str = "minimize"  # "minimize" or "maximize"
    max_trials: int = 100
    max_duration: Optional[float] = None  # Maximum duration in seconds
    early_stopping_patience: int = 10
    random_seed: Optional[int] = 42
    parallel_trials: int = 1
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.optimization_direction not in ["minimize", "maximize"]:
            raise ValueError("optimization_direction must be 'minimize' or 'maximize'")


@dataclass
class Trial:
    """Individual trial in an experiment."""
    trial_id: str
    parameters: Dict[str, Any]
    result: Optional[ExperimentResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    
    @property
    def duration(self) -> Optional[float]:
        """Get trial duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_successful(self) -> bool:
        """Check if trial was successful."""
        return self.status == "completed" and self.result and self.result.success


class HyperparameterOptimizer:
    """Base class for hyperparameter optimization algorithms."""
    
    def __init__(self, search_space: Dict[str, Any], 
                 optimization_metric: str = "loss",
                 optimization_direction: str = "minimize"):
        self.search_space = search_space
        self.optimization_metric = optimization_metric
        self.optimization_direction = optimization_direction
        self.trials_history = []
    
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest parameters for next trial."""
        raise NotImplementedError
    
    def update_with_result(self, parameters: Dict[str, Any], result: ExperimentResult):
        """Update optimizer with trial result."""
        self.trials_history.append({
            'parameters': parameters,
            'result': result,
            'trial_number': len(self.trials_history)
        })
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        if not self.trials_history:
            return None
        
        successful_trials = [
            trial for trial in self.trials_history 
            if trial['result'].success and self.optimization_metric in trial['result'].metrics
        ]
        
        if not successful_trials:
            return None
        
        if self.optimization_direction == "minimize":
            best_trial = min(
                successful_trials,
                key=lambda x: x['result'].metrics[self.optimization_metric]
            )
        else:
            best_trial = max(
                successful_trials,
                key=lambda x: x['result'].metrics[self.optimization_metric]
            )
        
        return best_trial['parameters']


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimization."""
    
    def __init__(self, search_space: Dict[str, Any], **kwargs):
        super().__init__(search_space, **kwargs)
        self.random_state = random.Random(42)
    
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest random parameters from search space."""
        parameters = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'uniform')
                
                if param_type == 'uniform':
                    low, high = param_config['low'], param_config['high']
                    parameters[param_name] = self.random_state.uniform(low, high)
                
                elif param_type == 'log_uniform':
                    low, high = param_config['low'], param_config['high']
                    log_low, log_high = math.log(low), math.log(high)
                    parameters[param_name] = math.exp(
                        self.random_state.uniform(log_low, log_high)
                    )
                
                elif param_type == 'choice':
                    choices = param_config['choices']
                    parameters[param_name] = self.random_state.choice(choices)
                
                elif param_type == 'int':
                    low, high = param_config['low'], param_config['high']
                    parameters[param_name] = self.random_state.randint(low, high)
                
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            
            elif isinstance(param_config, list):
                # Simple choice from list
                parameters[param_name] = self.random_state.choice(param_config)
            
            else:
                # Fixed value
                parameters[param_name] = param_config
        
        return parameters


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimization."""
    
    def __init__(self, search_space: Dict[str, Any], **kwargs):
        super().__init__(search_space, **kwargs)
        self.parameter_combinations = self._generate_combinations()
        self.current_index = 0
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        # Convert search space to lists of values
        param_lists = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'choice')
                
                if param_type == 'choice':
                    param_lists[param_name] = param_config['choices']
                elif param_type in ['uniform', 'int']:
                    # Create discrete grid
                    low, high = param_config['low'], param_config['high']
                    num_points = param_config.get('num_points', 10)
                    
                    if param_type == 'int':
                        param_lists[param_name] = list(range(low, high + 1))
                    else:
                        step = (high - low) / (num_points - 1)
                        param_lists[param_name] = [low + i * step for i in range(num_points)]
                
                else:
                    raise ValueError(f"Grid search doesn't support parameter type: {param_type}")
            
            elif isinstance(param_config, list):
                param_lists[param_name] = param_config
            
            else:
                param_lists[param_name] = [param_config]
        
        # Generate all combinations
        keys = list(param_lists.keys())
        combinations = []
        
        for values in itertools.product(*[param_lists[key] for key in keys]):
            combination = dict(zip(keys, values))
            combinations.append(combination)
        
        return combinations
    
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest next parameters from grid."""
        if self.current_index >= len(self.parameter_combinations):
            # Grid exhausted, return last combination
            return self.parameter_combinations[-1]
        
        parameters = self.parameter_combinations[self.current_index]
        self.current_index += 1
        return parameters


class ExperimentRunner:
    """Advanced experiment runner with optimization and tracking."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir) if results_dir else Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.active_experiments = {}
        self.completed_experiments = {}
        self.experiment_functions = {}
        
        # Thread safety
        self._lock = threading.Lock()
    
    def register_experiment(self, name: str, experiment_func: Callable):
        """Register an experiment function."""
        self.experiment_functions[name] = experiment_func
        
        if _HAS_DEPENDENCIES:
            robust_logger.log_info(f"Registered experiment: {name}")
    
    def run_experiment(self, config: ExperimentConfig, 
                      experiment_func: Optional[Callable] = None) -> Dict[str, Any]:
        """Run a complete experiment with optimization."""
        
        # Use registered function if not provided
        if experiment_func is None:
            if config.name not in self.experiment_functions:
                raise ValueError(f"Experiment '{config.name}' not registered")
            experiment_func = self.experiment_functions[config.name]
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            if 'numpy' in globals():
                import numpy as np
                np.random.seed(config.random_seed)
        
        # Initialize optimizer
        if config.search_space:
            if len(config.search_space) <= 3:  # Small search space, use grid
                optimizer = GridSearchOptimizer(
                    config.search_space,
                    optimization_metric=config.optimization_metric,
                    optimization_direction=config.optimization_direction
                )
            else:
                optimizer = RandomSearchOptimizer(
                    config.search_space,
                    optimization_metric=config.optimization_metric,
                    optimization_direction=config.optimization_direction
                )
        else:
            optimizer = None
        
        # Initialize experiment tracking
        experiment_id = self._generate_experiment_id(config)
        
        with self._lock:
            self.active_experiments[experiment_id] = {
                'config': config,
                'trials': [],
                'start_time': time.time(),
                'status': 'running'
            }
        
        try:
            # Run trials
            trials = self._run_trials(
                experiment_id, config, experiment_func, optimizer
            )
            
            # Analyze results
            analysis = self._analyze_experiment(trials, config)
            
            # Save results
            experiment_results = {
                'experiment_id': experiment_id,
                'config': asdict(config),
                'trials': [asdict(trial) for trial in trials],
                'analysis': analysis,
                'end_time': time.time()
            }
            
            self._save_experiment_results(experiment_id, experiment_results)
            
            with self._lock:
                self.completed_experiments[experiment_id] = experiment_results
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]
            
            return experiment_results
            
        except Exception as e:
            with self._lock:
                if experiment_id in self.active_experiments:
                    self.active_experiments[experiment_id]['status'] = 'failed'
                    self.active_experiments[experiment_id]['error'] = str(e)
            
            if _HAS_DEPENDENCIES:
                robust_logger.log_error(f"Experiment {experiment_id} failed: {e}")
            
            raise
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        content = f"{config.name}_{time.time()}_{config.random_seed}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _run_trials(self, experiment_id: str, config: ExperimentConfig,
                   experiment_func: Callable, 
                   optimizer: Optional[HyperparameterOptimizer]) -> List[Trial]:
        """Run all trials for an experiment."""
        trials = []
        start_time = time.time()
        best_metric = None
        patience_counter = 0
        
        if config.parallel_trials > 1:
            # Parallel execution
            trials = self._run_trials_parallel(
                experiment_id, config, experiment_func, optimizer
            )
        else:
            # Sequential execution
            for trial_num in range(config.max_trials):
                # Check stopping conditions
                if config.max_duration:
                    elapsed = time.time() - start_time
                    if elapsed > config.max_duration:
                        if _HAS_DEPENDENCIES:
                            robust_logger.log_info(f"Experiment {experiment_id} stopped due to time limit")
                        break
                
                # Generate parameters
                if optimizer:
                    parameters = optimizer.suggest_parameters(trial_num)
                else:
                    parameters = config.parameters.copy()
                
                # Add fixed parameters
                parameters.update(config.fixed_params)
                
                # Create trial
                trial = Trial(
                    trial_id=f"{experiment_id}_trial_{trial_num}",
                    parameters=parameters
                )
                
                # Run trial
                trial = self._run_single_trial(trial, experiment_func, config)
                trials.append(trial)
                
                # Update optimizer
                if optimizer and trial.result:
                    optimizer.update_with_result(parameters, trial.result)
                
                # Check early stopping
                if trial.is_successful and config.optimization_metric in trial.result.metrics:
                    current_metric = trial.result.metrics[config.optimization_metric]
                    
                    if best_metric is None:
                        best_metric = current_metric
                        patience_counter = 0
                    else:
                        if config.optimization_direction == "minimize":
                            improved = current_metric < best_metric
                        else:
                            improved = current_metric > best_metric
                        
                        if improved:
                            best_metric = current_metric
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= config.early_stopping_patience:
                            if _HAS_DEPENDENCIES:
                                robust_logger.log_info(f"Early stopping after {trial_num + 1} trials")
                            break
                
                # Save intermediate results
                if config.save_intermediate:
                    self._save_trial_result(experiment_id, trial)
        
        return trials
    
    def _run_trials_parallel(self, experiment_id: str, config: ExperimentConfig,
                           experiment_func: Callable,
                           optimizer: Optional[HyperparameterOptimizer]) -> List[Trial]:
        """Run trials in parallel."""
        trials = []
        
        def run_trial_batch(batch_start: int, batch_size: int) -> List[Trial]:
            batch_trials = []
            
            for i in range(batch_size):
                trial_num = batch_start + i
                if trial_num >= config.max_trials:
                    break
                
                # Generate parameters
                if optimizer:
                    # Note: This is not thread-safe for optimizers that learn from results
                    # For now, we use random parameters for parallel execution
                    parameters = optimizer.suggest_parameters(trial_num)
                else:
                    parameters = config.parameters.copy()
                
                parameters.update(config.fixed_params)
                
                trial = Trial(
                    trial_id=f"{experiment_id}_trial_{trial_num}",
                    parameters=parameters
                )
                
                trial = self._run_single_trial(trial, experiment_func, config)
                batch_trials.append(trial)
            
            return batch_trials
        
        # Run trials in batches
        batch_size = config.parallel_trials
        with ThreadPoolExecutor(max_workers=config.parallel_trials) as executor:
            futures = []
            
            for batch_start in range(0, config.max_trials, batch_size):
                future = executor.submit(run_trial_batch, batch_start, batch_size)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_trials = future.result()
                    trials.extend(batch_trials)
                except Exception as e:
                    if _HAS_DEPENDENCIES:
                        robust_logger.log_error(f"Parallel trial batch failed: {e}")
        
        return trials
    
    @robust_execution(max_retries=2, fallback_value=None)
    def _run_single_trial(self, trial: Trial, experiment_func: Callable,
                         config: ExperimentConfig) -> Trial:
        """Run a single trial."""
        trial.start_time = time.time()
        trial.status = "running"
        
        try:
            # Run experiment function
            result = experiment_func(**trial.parameters)
            
            trial.end_time = time.time()
            
            # Convert result to ExperimentResult if needed
            if isinstance(result, dict):
                trial.result = ExperimentResult(
                    experiment_id=trial.trial_id,
                    name=config.name,
                    parameters=trial.parameters,
                    metrics=result,
                    duration=trial.duration or 0,
                    success=True
                )
            elif isinstance(result, ExperimentResult):
                trial.result = result
            else:
                # Single metric result
                trial.result = ExperimentResult(
                    experiment_id=trial.trial_id,
                    name=config.name,
                    parameters=trial.parameters,
                    metrics={config.optimization_metric: float(result)},
                    duration=trial.duration or 0,
                    success=True
                )
            
            trial.status = "completed"
            
        except Exception as e:
            trial.end_time = time.time()
            trial.status = "failed"
            trial.result = ExperimentResult(
                experiment_id=trial.trial_id,
                name=config.name,
                parameters=trial.parameters,
                metrics={},
                duration=trial.duration or 0,
                success=False,
                error_message=str(e)
            )
            
            if _HAS_DEPENDENCIES:
                robust_logger.log_warning(f"Trial {trial.trial_id} failed: {e}")
        
        return trial
    
    def _analyze_experiment(self, trials: List[Trial], 
                          config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze experiment results."""
        analysis = {
            'total_trials': len(trials),
            'successful_trials': sum(1 for t in trials if t.is_successful),
            'failed_trials': sum(1 for t in trials if not t.is_successful),
            'total_duration': sum(t.duration or 0 for t in trials),
            'metrics_analysis': {}
        }
        
        # Extract successful results
        successful_trials = [t for t in trials if t.is_successful]
        
        if successful_trials:
            # Analyze metrics
            analyzer = StatisticalAnalyzer()
            
            # Get all metric names
            all_metrics = set()
            for trial in successful_trials:
                all_metrics.update(trial.result.metrics.keys())
            
            for metric in all_metrics:
                values = []
                for trial in successful_trials:
                    if metric in trial.result.metrics:
                        values.append(trial.result.metrics[metric])
                
                if values:
                    stats = analyzer.descriptive_statistics(values)
                    analysis['metrics_analysis'][metric] = stats
            
            # Best trial
            if config.optimization_metric in all_metrics:
                if config.optimization_direction == "minimize":
                    best_trial = min(
                        successful_trials,
                        key=lambda t: t.result.metrics[config.optimization_metric]
                    )
                else:
                    best_trial = max(
                        successful_trials,
                        key=lambda t: t.result.metrics[config.optimization_metric]
                    )
                
                analysis['best_trial'] = {
                    'trial_id': best_trial.trial_id,
                    'parameters': best_trial.parameters,
                    'metrics': best_trial.result.metrics,
                    'metric_value': best_trial.result.metrics[config.optimization_metric]
                }
        
        return analysis
    
    def _save_experiment_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results to file."""
        filepath = self.results_dir / f"{experiment_id}.json"
        
        # Convert nested objects to dictionaries for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if _HAS_DEPENDENCIES:
            robust_logger.log_info(f"Experiment results saved to {filepath}")
    
    def _save_trial_result(self, experiment_id: str, trial: Trial):
        """Save individual trial result."""
        trial_dir = self.results_dir / experiment_id / "trials"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = trial_dir / f"{trial.trial_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self._make_serializable(asdict(trial)), f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
        else:
            return obj
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an experiment."""
        with self._lock:
            if experiment_id in self.active_experiments:
                return self.active_experiments[experiment_id]
            elif experiment_id in self.completed_experiments:
                return self.completed_experiments[experiment_id]
        
        return None
    
    def list_experiments(self) -> Dict[str, List[str]]:
        """List all experiments."""
        with self._lock:
            return {
                'active': list(self.active_experiments.keys()),
                'completed': list(self.completed_experiments.keys())
            }


# Global experiment runner
_global_runner = ExperimentRunner()


def register_experiment(name: str):
    """Decorator to register an experiment function."""
    def decorator(func: Callable) -> Callable:
        _global_runner.register_experiment(name, func)
        return func
    return decorator


def run_optimization_study(name: str, search_space: Dict[str, Any],
                         optimization_metric: str = "loss",
                         max_trials: int = 50,
                         **kwargs) -> Dict[str, Any]:
    """Run hyperparameter optimization study."""
    config = ExperimentConfig(
        name=name,
        search_space=search_space,
        optimization_metric=optimization_metric,
        max_trials=max_trials,
        **kwargs
    )
    
    return _global_runner.run_experiment(config)