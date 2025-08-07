"""Configuration and security settings for Neural Operator Foundation Lab."""

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

# Optional yaml import
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment settings."""
    
    name: str = "neural_operator_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume_from: Optional[str] = None


@dataclass 
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer settings
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Loss function settings
    loss_function: str = "mse"
    physics_loss_weight: float = 0.1
    conservation_loss_weight: float = 0.05
    
    # Validation settings
    val_frequency: int = 1
    save_frequency: int = 10
    early_stopping_patience: int = 20
    
    # Mixed precision and optimization
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Checkpoint settings
    save_best_only: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Model type and basic settings
    model_type: str = "transformer_neural_operator"
    input_dim: int = 3
    output_dim: int = 1
    
    # Architecture-specific settings
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Operator-specific settings
    modes: List[int] = field(default_factory=lambda: [32, 32])
    spectral_layers: int = 4
    activation: str = "gelu"
    normalization: str = "layer_norm"
    
    # Advanced features
    use_positional_encoding: bool = True
    use_physics_embedding: bool = True
    use_adaptive_activation: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    
    # Data format and processing
    data_format: str = "h5"
    variables: List[str] = field(default_factory=list)
    resolution: List[int] = field(default_factory=lambda: [64, 64])
    
    # Preprocessing
    normalize: bool = True
    augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data loader settings
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Streaming and caching
    cache_data: bool = False
    streaming: bool = False
    max_cache_size: str = "1GB"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Basic distributed settings
    enabled: bool = False
    backend: str = "nccl"
    init_method: str = "env://"
    
    # Parallelism strategies
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    
    # Advanced distributed features
    zero_optimization: bool = False
    zero_stage: int = 2
    cpu_offloading: bool = False
    activation_checkpointing: bool = True
    
    # Communication settings
    gradient_compression: bool = False
    bucket_size: int = 25 * 1024 * 1024  # 25MB
    find_unused_parameters: bool = True


class SecurityConfig:
    """Security configuration for safe model loading and execution."""
    
    def __init__(self):
        self.validate_models = True
        self.max_model_size = "1GB"
        self.allowed_file_types = [".pt", ".pth", ".h5", ".npz"]
        self.sandbox_execution = True
        self.secure_mode = True
    
    def update(self, **kwargs):
        """Update security configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown security config: {key}")


# Global security configuration
_security_config = SecurityConfig()


def configure_security(
    validate_models: Optional[bool] = None,
    max_model_size: Optional[str] = None,
    allowed_file_types: Optional[List[str]] = None,
    sandbox_execution: Optional[bool] = None,
    secure_mode: Optional[bool] = None,
) -> None:
    """Configure security settings for the library.
    
    Args:
        validate_models: Whether to validate model files before loading
        max_model_size: Maximum allowed model file size (e.g., "1GB", "500MB")
        allowed_file_types: List of allowed file extensions
        sandbox_execution: Whether to sandbox user code execution
        secure_mode: Enable comprehensive security measures
    """
    config_updates = {}
    
    if validate_models is not None:
        config_updates["validate_models"] = validate_models
    if max_model_size is not None:
        config_updates["max_model_size"] = max_model_size
    if allowed_file_types is not None:
        config_updates["allowed_file_types"] = allowed_file_types
    if sandbox_execution is not None:
        config_updates["sandbox_execution"] = sandbox_execution
    if secure_mode is not None:
        config_updates["secure_mode"] = secure_mode
    
    _security_config.update(**config_updates)


def get_security_config() -> SecurityConfig:
    """Get current security configuration."""
    return _security_config


@dataclass
class Config:
    """Main configuration container."""
    
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            distributed=DistributedConfig(**config_dict.get('distributed', {}))
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def merge(self, other: 'Config') -> 'Config':
        """Merge with another configuration."""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def deep_merge(dict1, dict2):
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(self_dict, other_dict)
        return self.__class__.from_dict(merged_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate experiment config
        if not self.experiment.name:
            errors.append("Experiment name cannot be empty")
        
        # Validate training config
        if self.training.epochs <= 0:
            errors.append("Training epochs must be positive")
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        # Validate model config
        if self.model.input_dim <= 0:
            errors.append("Model input dimension must be positive")
        if self.model.output_dim <= 0:
            errors.append("Model output dimension must be positive")
        if self.model.hidden_dim <= 0:
            errors.append("Hidden dimension must be positive")
        
        # Validate data config
        if not self.data.variables:
            errors.append("Data variables list cannot be empty")
        if not self.data.resolution or any(r <= 0 for r in self.data.resolution):
            errors.append("Data resolution must contain positive values")
        
        return errors
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Experiment settings
        if os.getenv("NEURAL_OP_EXPERIMENT_NAME"):
            self.experiment.name = os.getenv("NEURAL_OP_EXPERIMENT_NAME")
        
        if os.getenv("NEURAL_OP_SEED"):
            self.experiment.seed = int(os.getenv("NEURAL_OP_SEED"))
        
        # Training settings
        if os.getenv("NEURAL_OP_EPOCHS"):
            self.training.epochs = int(os.getenv("NEURAL_OP_EPOCHS"))
        
        if os.getenv("NEURAL_OP_BATCH_SIZE"):
            self.training.batch_size = int(os.getenv("NEURAL_OP_BATCH_SIZE"))
        
        if os.getenv("NEURAL_OP_LEARNING_RATE"):
            self.training.learning_rate = float(os.getenv("NEURAL_OP_LEARNING_RATE"))
        
        # Distributed settings
        if os.getenv("NEURAL_OP_DISTRIBUTED"):
            self.distributed.enabled = os.getenv("NEURAL_OP_DISTRIBUTED").lower() == "true"
        
        if os.getenv("NEURAL_OP_ZERO_OPTIMIZATION"):
            self.distributed.zero_optimization = os.getenv("NEURAL_OP_ZERO_OPTIMIZATION").lower() == "true"


def create_config_template(config_type: str = "default") -> Config:
    """Create configuration template for different use cases."""
    
    if config_type == "default":
        return Config()
    
    elif config_type == "small":
        config = Config()
        config.model.hidden_dim = 128
        config.model.num_layers = 4
        config.model.num_heads = 4
        config.training.batch_size = 16
        return config
    
    elif config_type == "large":
        config = Config()
        config.model.hidden_dim = 512
        config.model.num_layers = 12
        config.model.num_heads = 16
        config.training.batch_size = 8
        config.training.gradient_accumulation_steps = 4
        return config
    
    elif config_type == "distributed":
        config = Config()
        config.distributed.enabled = True
        config.distributed.zero_optimization = True
        config.distributed.zero_stage = 2
        config.training.batch_size = 16  # Per GPU
        return config
    
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def load_config_with_overrides(base_config_path: Optional[str] = None,
                              override_config_path: Optional[str] = None,
                              env_overrides: bool = True) -> Config:
    """Load configuration with optional overrides."""
    
    # Start with default config
    config = Config()
    
    # Load base config if provided
    if base_config_path:
        base_config = Config.load(base_config_path)
        config = config.merge(base_config)
    
    # Apply override config if provided
    if override_config_path:
        override_config = Config.load(override_config_path)
        config = config.merge(override_config)
    
    # Apply environment variable overrides
    if env_overrides:
        config.update_from_env()
    
    # Validate final configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


# Initialize from environment variables
if os.getenv("NEURAL_OP_SECURE_MODE", "").lower() == "true":
    configure_security(secure_mode=True)

if os.getenv("NEURAL_OP_VALIDATE_MODELS", "").lower() == "true":
    configure_security(validate_models=True)

if os.getenv("NEURAL_OP_SANDBOX_EXECUTION", "").lower() == "true":
    configure_security(sandbox_execution=True)