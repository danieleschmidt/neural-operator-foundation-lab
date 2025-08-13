"""Advanced configuration management with validation and security."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints
from dataclasses import dataclass, field, fields, asdict
import warnings
from copy import deepcopy
import logging

try:
    from .robust_framework import robust_logger, ValidationError, InputValidator
    from .security.enhanced_security import SecurityConfig, InputSanitizer
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    
    class ValidationError(Exception):
        pass


@dataclass
class ModelConfig:
    """Configuration for neural operator models."""
    model_type: str = "fourier"
    input_dim: int = 4
    output_dim: int = 4
    hidden_dim: int = 256
    num_layers: int = 6
    activation: str = "gelu"
    dropout: float = 0.1
    use_bias: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValidationError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValidationError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValidationError("dropout must be between 0 and 1")


@dataclass 
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    mixed_precision: bool = False
    accumulate_grad_batches: int = 1
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValidationError("validation_split must be between 0 and 1")


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_path: str = "data/"
    resolution: int = 64
    variables: List[str] = field(default_factory=lambda: ["u", "v", "p"])
    normalize: bool = True
    augmentation: bool = False
    cache_data: bool = True
    num_workers: int = 4
    
    def __post_init__(self):
        """Validate data configuration."""
        if self.resolution <= 0:
            raise ValidationError("resolution must be positive")
        if self.num_workers < 0:
            raise ValidationError("num_workers must be non-negative")


@dataclass
class SystemConfig:
    """System-level configuration."""
    device: str = "auto"
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints/"
    enable_profiling: bool = False
    max_memory_gb: Optional[float] = None
    distributed: bool = False
    
    def __post_init__(self):
        """Validate system configuration."""
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValidationError(f"device must be one of {valid_devices} or 'cuda:N'")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValidationError(f"log_level must be one of {valid_log_levels}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "neural_operator_experiment"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    security: Optional[SecurityConfig] = None
    
    def __post_init__(self):
        """Initialize security config if not provided."""
        if self.security is None and _HAS_DEPENDENCIES:
            self.security = SecurityConfig()


class ConfigManager:
    """Advanced configuration manager with validation and security."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration registry
        self.registered_configs = {}
        self.config_templates = {}
        self.validation_rules = {}
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register default configurations
        self._register_default_configs()
    
    def _register_default_configs(self):
        """Register default configuration classes."""
        self.register_config_type("model", ModelConfig)
        self.register_config_type("training", TrainingConfig)
        self.register_config_type("data", DataConfig)
        self.register_config_type("system", SystemConfig)
        self.register_config_type("experiment", ExperimentConfig)
    
    def register_config_type(self, name: str, config_class: Type, 
                           validation_func: Optional[callable] = None):
        """Register a configuration type."""
        self.registered_configs[name] = config_class
        if validation_func:
            self.validation_rules[name] = validation_func
    
    def create_template(self, name: str, config: Any) -> None:
        """Create a configuration template."""
        self.config_templates[name] = deepcopy(config)
        self.logger.info(f"Created template '{name}'")
    
    def load_from_template(self, template_name: str, **overrides) -> Any:
        """Load configuration from template with overrides."""
        if template_name not in self.config_templates:
            raise ValidationError(f"Template '{template_name}' not found")
        
        config = deepcopy(self.config_templates[template_name])
        
        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        return config
    
    def save_config(self, config: Any, filename: str, format: str = "auto") -> Path:
        """Save configuration to file."""
        file_path = self.config_dir / filename
        
        # Auto-detect format from extension
        if format == "auto":
            suffix = file_path.suffix.lower()
            if suffix == ".yaml" or suffix == ".yml":
                format = "yaml"
            elif suffix == ".json":
                format = "json"
            else:
                format = "yaml"  # Default
                file_path = file_path.with_suffix(".yaml")
        
        # Convert to dictionary
        if hasattr(config, '__dict__'):
            config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
        else:
            config_dict = config
        
        # Sanitize configuration
        if _HAS_DEPENDENCIES:
            sanitizer = InputSanitizer(SecurityConfig())
            config_dict = sanitizer.sanitize_config_input(config_dict)
        
        # Save based on format
        try:
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif format == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                raise ValidationError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved configuration to {file_path}")
            return file_path
            
        except Exception as e:
            raise ValidationError(f"Failed to save configuration: {e}")
    
    def load_config(self, filename: Union[str, Path], 
                   config_type: Optional[str] = None) -> Any:
        """Load configuration from file."""
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path
        
        if not file_path.exists():
            raise ValidationError(f"Configuration file not found: {file_path}")
        
        # Load based on file extension
        try:
            suffix = file_path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                with open(file_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif suffix == ".json":
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValidationError(f"Unsupported file format: {suffix}")
            
        except Exception as e:
            raise ValidationError(f"Failed to load configuration: {e}")
        
        # Convert to appropriate config object
        if config_type and config_type in self.registered_configs:
            config_class = self.registered_configs[config_type]
            config = self._dict_to_config(config_dict, config_class)
        else:
            config = config_dict
        
        # Validate if validation rules exist
        if config_type and config_type in self.validation_rules:
            self.validation_rules[config_type](config)
        
        self.logger.info(f"Loaded configuration from {file_path}")
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any], 
                       config_class: Type) -> Any:
        """Convert dictionary to configuration object."""
        try:
            # Handle nested configurations
            if hasattr(config_class, '__dataclass_fields__'):
                field_types = get_type_hints(config_class)
                converted_dict = {}
                
                for key, value in config_dict.items():
                    if key in field_types:
                        field_type = field_types[key]
                        
                        # Handle nested dataclasses
                        if hasattr(field_type, '__dataclass_fields__'):
                            if isinstance(value, dict):
                                converted_dict[key] = self._dict_to_config(value, field_type)
                            else:
                                converted_dict[key] = value
                        else:
                            converted_dict[key] = value
                    else:
                        converted_dict[key] = value
                
                return config_class(**converted_dict)
            else:
                return config_class(**config_dict)
                
        except Exception as e:
            raise ValidationError(f"Failed to convert dict to {config_class.__name__}: {e}")
    
    def _apply_overrides(self, config: Any, overrides: Dict[str, Any]) -> Any:
        """Apply overrides to configuration."""
        if hasattr(config, '__dict__'):
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    warnings.warn(f"Override key '{key}' not found in configuration")
        elif isinstance(config, dict):
            config.update(overrides)
        
        return config
    
    def validate_config(self, config: Any, config_type: Optional[str] = None) -> bool:
        """Validate configuration."""
        try:
            # Basic validation for dataclasses (post_init will be called)
            if hasattr(config, '__post_init__'):
                config.__post_init__()
            
            # Custom validation rules
            if config_type and config_type in self.validation_rules:
                self.validation_rules[config_type](config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def merge_configs(self, base_config: Any, override_config: Any) -> Any:
        """Merge two configurations."""
        if type(base_config) != type(override_config):
            raise ValidationError("Cannot merge configs of different types")
        
        merged = deepcopy(base_config)
        
        if hasattr(override_config, '__dict__'):
            for key, value in override_config.__dict__.items():
                if value is not None:  # Only override non-None values
                    setattr(merged, key, value)
        elif isinstance(override_config, dict):
            if isinstance(merged, dict):
                merged.update(override_config)
        
        return merged
    
    def get_environment_overrides(self, prefix: str = "NEURAL_OP_") -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    overrides[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    overrides[config_key] = int(value)
                else:
                    try:
                        overrides[config_key] = float(value)
                    except ValueError:
                        overrides[config_key] = value
        
        return overrides
    
    def create_experiment_config(self, name: str, **kwargs) -> ExperimentConfig:
        """Create a complete experiment configuration."""
        config = ExperimentConfig(name=name)
        
        # Apply any provided overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Apply environment overrides
        env_overrides = self.get_environment_overrides()
        config = self._apply_overrides(config, env_overrides)
        
        # Validate the complete configuration
        if not self.validate_config(config, "experiment"):
            raise ValidationError("Failed to create valid experiment configuration")
        
        return config
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        config_files = []
        for ext in [".yaml", ".yml", ".json"]:
            config_files.extend([f.name for f in self.config_dir.glob(f"*{ext}")])
        return sorted(config_files)
    
    def list_templates(self) -> List[str]:
        """List available configuration templates."""
        return list(self.config_templates.keys())


# Global configuration manager
default_config_manager = ConfigManager()


def create_config(config_type: str, **kwargs) -> Any:
    """Create a configuration of specified type with overrides."""
    return default_config_manager.create_experiment_config("default", **kwargs)


def load_config_file(filename: str, config_type: Optional[str] = None) -> Any:
    """Load configuration from file using default manager."""
    return default_config_manager.load_config(filename, config_type)


def save_config_file(config: Any, filename: str, format: str = "auto") -> Path:
    """Save configuration to file using default manager."""
    return default_config_manager.save_config(config, filename, format)