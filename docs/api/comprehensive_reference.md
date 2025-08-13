# Comprehensive API Reference

Complete API reference for the Neural Operator Foundation Lab.

## 🚀 Quick Navigation

- [Core Models](#core-models)
- [Training & Optimization](#training--optimization)
- [Data Handling](#data-handling)
- [Performance & Monitoring](#performance--monitoring)
- [Security & Compliance](#security--compliance)
- [Research & Experiments](#research--experiments)
- [Utilities & Configuration](#utilities--configuration)

## Core Models

### Base Classes

#### `NeuralOperatorBase`
Abstract base class for all neural operators.

```python
class NeuralOperatorBase(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int, **kwargs)
    
    # Abstract methods
    def forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor
    def forward_operator(self, features: torch.Tensor) -> torch.Tensor
    
    # Implemented methods
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor, inputs: Optional[torch.Tensor] = None) -> torch.Tensor
    def count_parameters(self) -> int
    def get_model_summary(self) -> Dict[str, Any]
```

### Foundation Models

#### `UniversalPhysicsTransformer`
Pre-trained universal model for multiple physics domains.

```python
class UniversalPhysicsTransformer(TransformerNeuralOperator):
    def __init__(self, config: UniversalPhysicsConfig)
    
    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: Optional[str] = None) -> 'UniversalPhysicsTransformer'
    
    def freeze_backbone(self)
    def add_pde_head(self, pde_type: str, output_channels: int, **kwargs)
    def adapt_few_shot(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None
    def cross_physics_attention(self, x: torch.Tensor, physics_types: List[str]) -> torch.Tensor
```

#### `UniversalPhysicsConfig`
Configuration for Universal Physics Transformer.

```python
@dataclass
class UniversalPhysicsConfig:
    input_dim: int = 64
    output_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 16
    physics_types: List[str] = field(default_factory=list)
    cross_physics_attention: bool = True
    few_shot_learning: bool = True
    conservation_loss_weight: float = 0.1
    boundary_loss_weight: float = 0.05
    symmetry_loss_weight: float = 0.02
```

### Specialized Models

#### `FourierNeuralOperator`
Fourier Neural Operator implementation.

```python
class FourierNeuralOperator(NeuralOperatorBase):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 modes: List[int],
                 width: int = 64,
                 num_layers: int = 4,
                 activation: str = 'gelu',
                 padding: int = 0)
```

#### `TransformerNeuralOperator`
Transformer-based neural operator.

```python
class TransformerNeuralOperator(NeuralOperatorBase):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 1024,
                 use_positional_encoding: bool = True,
                 use_physics_embedding: bool = False)
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

#### `ProbabilisticNeuralOperator`
Neural operator with uncertainty quantification.

```python
class ProbabilisticNeuralOperator(NeuralOperatorBase):
    def __init__(self,
                 trunk_layers: List[int],
                 branch_layers: List[int],
                 num_basis: int = 64,
                 uncertainty_type: str = 'epistemic',
                 dropout_rate: float = 0.1,
                 num_samples: int = 50)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    def sample_predictions(self, x: torch.Tensor, num_samples: int) -> torch.Tensor
```

## Training & Optimization

### Training Framework

#### `AdaptiveTrainer`
Advanced trainer with adaptive features.

```python
class AdaptiveTrainer:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 use_mixed_precision: bool = False,
                 gradient_clipping: float = 1.0,
                 weight_decay: float = 1e-4)
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None, verbose: bool = True) -> Dict[str, Any]
    def evaluate(self, dataset: Dataset) -> Dict[str, float]
    def add_callback(self, name: str, callback: Callable) -> None
    def save_checkpoint(self, filepath: str) -> None
    def load_checkpoint(self, filepath: str) -> None
```

### Performance Optimization

#### `PerformanceProfiler`
Advanced performance profiling and optimization.

```python
class PerformanceProfiler:
    def __init__(self)
    
    @contextmanager
    def profile(self, operation_name: str)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]
    def identify_bottlenecks(self, threshold_seconds: float = 1.0) -> List[Dict[str, Any]]
    def get_summary(self) -> Dict[str, Any]
```

#### `AdvancedCache`
High-performance caching system.

```python
class AdvancedCache:
    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None)
    
    def get(self, key: str) -> Tuple[bool, Any]
    def put(self, key: str, value: Any) -> None
    def clear(self) -> None
    def get_stats(self) -> Dict[str, Any]
```

#### `@cached` Decorator
Caching decorator for functions.

```python
def cached(maxsize: int = 128, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Advanced caching decorator."""
    
# Usage
@cached(maxsize=100, ttl=3600)
def expensive_computation(data):
    return result
```

### Parallel Processing

#### `ParallelProcessor`
Advanced parallel processing with automatic scaling.

```python
class ParallelProcessor:
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False)
    
    def parallel_map(self, func: Callable, items: List[Any], chunk_size: Optional[int] = None) -> List[Any]
    def get_stats(self) -> Dict[str, Any]
```

#### `AutoScaler`
Automatic resource scaling based on load.

```python
class AutoScaler:
    def __init__(self, min_resources: int = 1, max_resources: int = 16)
    
    def record_load(self, current_load: float, max_capacity: float) -> None
    def auto_scale(self) -> int
    def get_scaling_recommendation(self) -> Dict[str, Any]
```

## Data Handling

### Datasets

#### `PhysicsDataset`
Dataset for physics problems.

```python
class PhysicsDataset(Dataset):
    def __init__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 physics_labels: Optional[torch.Tensor] = None,
                 transform: Optional[Callable] = None)
    
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]
    def add_physics_context(self, context: Dict[str, Any]) -> None
```

### Data Loaders

#### `AdaptiveDataLoader`
Data loader with adaptive batching.

```python
class AdaptiveDataLoader:
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 adaptive_batching: bool = True,
                 memory_efficient: bool = True)
    
    def __iter__(self) -> Iterator[Any]
    def get_batch_stats(self) -> Dict[str, Any]
```

## Performance & Monitoring

### Intelligent Caching

#### `AdaptiveCache`
ML-inspired caching with intelligent eviction.

```python
class AdaptiveCache:
    def __init__(self,
                 max_size_mb: float = 100.0,
                 compression_enabled: bool = True,
                 ml_eviction: bool = True,
                 access_pattern_learning: bool = True)
    
    def get(self, key: str, default: Any = None) -> Any
    def put(self, key: str, value: Any) -> None
    def warm(self, key: str, compute_func: Callable) -> None
    def get_statistics(self) -> Dict[str, Any]
```

#### `IntelligentEvictionStrategy`
ML-based cache eviction strategy.

```python
class IntelligentEvictionStrategy:
    def __init__(self)
    
    def set_weights(self, weights: Dict[str, float]) -> None
    def predict_eviction_order(self, cache_entries: List[CacheEntry]) -> List[str]
    def update_access_patterns(self, access_log: List[CacheAccess]) -> None
```

### Monitoring

#### `MetricsCollector`
Comprehensive metrics collection.

```python
class MetricsCollector:
    def __init__(self, collection_interval: float = 30.0)
    
    def start(self) -> None
    def stop(self) -> None
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None
    def get_current_metrics(self) -> Dict[str, Any]
    def get_metric_history(self, name: str, duration: Optional[float] = None) -> List[MetricPoint]
```

#### `HealthChecker`
System health monitoring.

```python
class HealthChecker:
    def __init__(self, check_interval: float = 60.0)
    
    def add_check(self, name: str, check_func: Callable[[], bool]) -> None
    def remove_check(self, name: str) -> None
    def get_health_status(self) -> Dict[str, Any]
    def start_monitoring(self) -> None
    def stop_monitoring(self) -> None
```

## Security & Compliance

### Security Framework

#### `SecurityConfig`
Security configuration management.

```python
@dataclass
class SecurityConfig:
    enable_input_validation: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    max_input_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = field(default_factory=lambda: ['.pt', '.pth', '.json'])
    secure_model_loading: bool = True
```

#### `InputValidator`
Input validation and sanitization.

```python
class InputValidator:
    def __init__(self, config: SecurityConfig)
    
    def validate_tensor(self, tensor: torch.Tensor, expected_shape: Optional[Tuple[int, ...]] = None) -> bool
    def validate_file(self, filepath: str) -> bool
    def sanitize_input(self, data: Any) -> Any
    def check_adversarial_patterns(self, tensor: torch.Tensor) -> Dict[str, Any]
```

### Compliance

#### `GDPRCompliance`
GDPR compliance manager.

```python
class GDPRCompliance:
    def __init__(self, data_manager: Optional[DataProtectionManager] = None)
    
    def process_subject_access_request(self, data_subject_id: str) -> Dict[str, Any]
    def process_erasure_request(self, data_subject_id: str, reason: str = 'withdrawal_of_consent') -> Dict[str, Any]
    def validate_consent(self, consent_data: Dict[str, Any]) -> Dict[str, Any]
    def conduct_dpia(self, processing_description: Dict[str, Any]) -> Dict[str, Any]
```

#### `DataProtectionManager`
Comprehensive data protection.

```python
class DataProtectionManager:
    def __init__(self)
    
    def record_data_processing(self, data_type: str, purpose: str, legal_basis: str, data_subject_id: Optional[str] = None) -> str
    def right_to_be_forgotten(self, data_subject_id: str) -> Dict[str, Any]
    def data_portability_export(self, data_subject_id: str) -> Dict[str, Any]
    def get_processing_records(self, data_subject_id: Optional[str] = None) -> List[ProcessingRecord]
```

## Research & Experiments

### Research Framework

#### `StatisticalAnalyzer`
Advanced statistical analysis.

```python
class StatisticalAnalyzer:
    def __init__(self)
    
    def descriptive_statistics(self, data: List[float]) -> Dict[str, float]
    def hypothesis_test(self, data1: List[float], data2: List[float], test_type: str = 'ttest') -> Dict[str, Any]
    def correlation_analysis(self, variables: Dict[str, List[float]]) -> Dict[str, Any]
    def distribution_fitting(self, data: List[float]) -> Dict[str, Any]
```

#### `BenchmarkSuite`
Comprehensive benchmarking framework.

```python
class BenchmarkSuite:
    def __init__(self)
    
    def register_benchmark(self, name: str, benchmark_func: Callable) -> None
    def run_benchmark(self, name: str, config: Dict[str, Any] = {}, num_runs: int = 5) -> BenchmarkResult
    def run_all_benchmarks(self, config: Dict[str, Any] = {}) -> Dict[str, BenchmarkResult]
    def compare_benchmarks(self, results1: Dict[str, BenchmarkResult], results2: Dict[str, BenchmarkResult]) -> Dict[str, Any]
```

### Experiment Management

#### `ExperimentRunner`
Advanced experiment management.

```python
class ExperimentRunner:
    def __init__(self, base_config: Optional[Dict[str, Any]] = None, results_dir: str = "experiment_results")
    
    def create_experiment(self, name: str, description: str, config: Dict[str, Any]) -> str
    def run_experiment(self, experiment_id: str) -> ExperimentResult
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]
```

#### `GridSearchOptimizer`
Grid search hyperparameter optimization.

```python
class GridSearchOptimizer:
    def __init__(self, 
                 search_space: Dict[str, List[Any]],
                 optimization_metric: str = 'val_loss',
                 optimization_direction: str = 'minimize')
    
    def optimize(self, objective_function: Callable, max_trials: int = 50) -> OptimizationResult
    def get_best_params(self) -> Dict[str, Any]
    def get_optimization_history(self) -> List[Dict[str, Any]]
```

## Utilities & Configuration

### Configuration Management

#### `ConfigManager`
Advanced configuration management.

```python
class ConfigManager:
    def __init__(self, config_dir: str = "config", environment: Optional[str] = None)
    
    def load_config(self, config_name: str) -> Dict[str, Any]
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> None
    def validate_config(self, config_data: Dict[str, Any], schema: Dict[str, Any]) -> bool
    def get_environment_config(self) -> Dict[str, Any]
```

### Robust Framework

#### `@robust_execution` Decorator
Robust execution with retries and fallbacks.

```python
def robust_execution(max_retries: int = 3, 
                    retry_delay: float = 1.0, 
                    backoff_factor: float = 2.0,
                    fallback_value: Any = None,
                    exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Robust execution decorator with automatic retries."""
    
# Usage
@robust_execution(max_retries=3, retry_delay=1.0)
def unreliable_function():
    return result
```

#### `CircuitBreaker`
Circuit breaker pattern implementation.

```python
class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 recovery_threshold: int = 3)
    
    def call(self, func: Callable, *args, **kwargs) -> Any
    def get_state(self) -> str  # 'closed', 'open', 'half_open'
    def get_stats(self) -> Dict[str, Any]
```

### Global Deployment

#### `GlobalDeploymentManager`
Multi-region deployment management.

```python
class GlobalDeploymentManager:
    def __init__(self, base_config: Dict[str, Any])
    
    def generate_deployment_artifacts(self, target_regions: List[str]) -> Dict[str, Any]
    def create_docker_artifacts(self, region_config: Dict[str, Any]) -> Dict[str, str]
    def create_kubernetes_artifacts(self, region_config: Dict[str, Any]) -> Dict[str, str]
    def create_terraform_artifacts(self, region_config: Dict[str, Any]) -> Dict[str, str]
    def validate_compliance(self, region: str, artifacts: Dict[str, Any]) -> Dict[str, Any]
```

## Function Utilities

### Model Creation

```python
def create_model(model_type: str, config: Dict[str, Any]) -> NeuralOperatorBase:
    """Factory function for creating models from configuration."""

def load_checkpoint(checkpoint_path: str, model: Optional[NeuralOperatorBase] = None) -> NeuralOperatorBase:
    """Load model from checkpoint file."""

def save_checkpoint(model: NeuralOperatorBase, checkpoint_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint with metadata."""
```

### Performance Utilities

```python
def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""

def optimize_parallel_execution(func: Callable, items: List[Any], **kwargs) -> List[Any]:
    """Optimized parallel execution with automatic scaling."""

def record_system_load(current_load: float, max_capacity: float) -> None:
    """Record system load for autoscaling."""

def auto_scale_resources() -> int:
    """Trigger automatic resource scaling."""
```

### Decorators

```python
@profile(operation_name: str)
def decorated_function():
    """Profile function performance."""

@cached(maxsize: int = 128, ttl: Optional[float] = None)
def cached_function():
    """Cache function results."""

@robust_execution(max_retries: int = 3)
def robust_function():
    """Execute function with automatic retries."""
```

## Configuration Examples

### Model Configurations

```python
# Navier-Stokes Configuration
navier_stokes_config = {
    'model_type': 'fourier_neural_operator',
    'input_dim': 4,  # u, v, w, p
    'output_dim': 4,
    'modes': [32, 32, 32],
    'width': 256,
    'num_layers': 6,
    'activation': 'gelu',
    'physics_informed': True,
    'conservation_loss': True
}

# Multi-Physics Configuration
multi_physics_config = {
    'model_type': 'universal_physics_transformer',
    'input_dim': 64,
    'output_dim': 64,
    'hidden_dim': 512,
    'num_layers': 8,
    'physics_types': ['fluid_dynamics', 'heat_transfer', 'electromagnetics'],
    'cross_physics_attention': True,
    'few_shot_learning': True
}
```

### Training Configurations

```python
# Advanced Training Configuration
training_config = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'use_mixed_precision': True,
    'gradient_clipping': 1.0,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine_annealing',
    'warmup_epochs': 5
}

# Performance Configuration
performance_config = {
    'caching': {
        'enabled': True,
        'max_size_mb': 1000,
        'compression': True,
        'ml_eviction': True
    },
    'parallel_processing': {
        'max_workers': 'auto',
        'use_processes': False,
        'chunk_size': 'auto'
    },
    'profiling': {
        'enabled': True,
        'detailed_memory': True
    }
}
```

---

**This comprehensive API reference covers all major components of the Neural Operator Foundation Lab.** 🚀