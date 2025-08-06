"""Tests for configuration module."""

import pytest
import json
import yaml
from pathlib import Path
from neural_operator_lab.config import (
    configure_security, get_security_config, Config, 
    ExperimentConfig, TrainingConfig, ModelConfig,
    DataConfig, DistributedConfig, create_config_template,
    load_config_with_overrides
)


def test_default_security_config():
    """Test default security configuration."""
    config = get_security_config()
    assert config.validate_models is True
    assert config.max_model_size == "1GB"
    assert ".pt" in config.allowed_file_types
    assert config.sandbox_execution is True
    assert config.secure_mode is True


def test_configure_security():
    """Test security configuration updates."""
    original_config = get_security_config()
    
    # Test updating configuration
    configure_security(
        validate_models=False,
        max_model_size="2GB",
        allowed_file_types=[".pt", ".h5"],
        sandbox_execution=False,
        secure_mode=False
    )
    
    config = get_security_config()
    assert config.validate_models is False
    assert config.max_model_size == "2GB"
    assert config.allowed_file_types == [".pt", ".h5"]
    assert config.sandbox_execution is False
    assert config.secure_mode is False
    
    # Restore original configuration
    configure_security(
        validate_models=original_config.validate_models,
        max_model_size=original_config.max_model_size,
        allowed_file_types=original_config.allowed_file_types,
        sandbox_execution=original_config.sandbox_execution,
        secure_mode=original_config.secure_mode
    )


def test_partial_configuration_update():
    """Test partial configuration updates."""
    original_config = get_security_config()
    
    # Update only one setting
    configure_security(validate_models=False)
    
    config = get_security_config()
    assert config.validate_models is False
    # Other settings should remain unchanged
    assert config.max_model_size == original_config.max_model_size
    assert config.allowed_file_types == original_config.allowed_file_types
    
    # Restore
    configure_security(validate_models=original_config.validate_models)


class TestConfigDataclasses:
    """Test configuration dataclass functionality."""
    
    def test_experiment_config_defaults(self):
        """Test ExperimentConfig default values."""
        config = ExperimentConfig()
        assert config.name == "neural_operator_experiment"
        assert config.seed == 42
        assert config.log_level == "INFO"
        assert isinstance(config.tags, list)
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adamw"
        assert config.mixed_precision is True
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.model_type == "transformer_neural_operator"
        assert config.input_dim == 3
        assert config.output_dim == 1
        assert config.hidden_dim == 256
        assert config.modes == [32, 32]
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.data_format == "h5"
        assert config.resolution == [64, 64]
        assert config.normalize is True
        assert config.num_workers == 4
    
    def test_distributed_config_defaults(self):
        """Test DistributedConfig default values."""
        config = DistributedConfig()
        assert config.enabled is False
        assert config.backend == "nccl"
        assert config.data_parallel is True
        assert config.zero_stage == 2


class TestMainConfig:
    """Test main Config class functionality."""
    
    def test_config_initialization(self):
        """Test Config class initialization."""
        config = Config()
        
        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.distributed, DistributedConfig)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "experiment" in config_dict
        assert "training" in config_dict
        assert "model" in config_dict
        assert "data" in config_dict
        assert "distributed" in config_dict
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "experiment": {"name": "test_experiment", "seed": 123},
            "training": {"epochs": 50, "batch_size": 16},
            "model": {"hidden_dim": 128}
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.experiment.name == "test_experiment"
        assert config.experiment.seed == 123
        assert config.training.epochs == 50
        assert config.training.batch_size == 16
        assert config.model.hidden_dim == 128
    
    def test_config_validation(self):
        """Test config validation."""
        config = Config()
        errors = config.validate()
        
        # Default config should have some validation errors (empty data variables)
        assert len(errors) > 0
        assert any("variables" in error for error in errors)
    
    def test_config_validation_with_valid_config(self):
        """Test validation with valid configuration."""
        config = Config()
        config.data.variables = ["u", "v", "p"]
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_config_merge(self):
        """Test config merging."""
        base_config = Config()
        base_config.training.epochs = 100
        base_config.model.hidden_dim = 256
        
        override_config = Config()
        override_config.training.epochs = 50  # Override
        override_config.training.learning_rate = 2e-4  # New value
        
        merged = base_config.merge(override_config)
        
        assert merged.training.epochs == 50  # Overridden
        assert merged.training.learning_rate == 2e-4  # New value
        assert merged.model.hidden_dim == 256  # Preserved


class TestConfigFileOperations:
    """Test config file save/load operations."""
    
    def test_save_load_json(self, tmp_path):
        """Test saving and loading JSON config."""
        config = Config()
        config.experiment.name = "test_json"
        config.training.epochs = 42
        
        config_path = tmp_path / "config.json"
        config.save(config_path)
        
        assert config_path.exists()
        
        loaded_config = Config.load(config_path)
        assert loaded_config.experiment.name == "test_json"
        assert loaded_config.training.epochs == 42
    
    def test_save_load_yaml(self, tmp_path):
        """Test saving and loading YAML config."""
        config = Config()
        config.experiment.name = "test_yaml"
        config.training.batch_size = 64
        
        config_path = tmp_path / "config.yaml"
        config.save(config_path)
        
        assert config_path.exists()
        
        loaded_config = Config.load(config_path)
        assert loaded_config.experiment.name == "test_yaml"
        assert loaded_config.training.batch_size == 64
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent config file."""
        config_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            Config.load(config_path)
    
    def test_unsupported_format(self, tmp_path):
        """Test unsupported file format."""
        config = Config()
        config_path = tmp_path / "config.txt"
        
        with pytest.raises(ValueError, match="Unsupported config format"):
            config.save(config_path)


class TestConfigTemplates:
    """Test config template creation."""
    
    def test_default_template(self):
        """Test default config template."""
        config = create_config_template("default")
        
        assert isinstance(config, Config)
        assert config.model.hidden_dim == 256  # Default value
    
    def test_small_template(self):
        """Test small config template."""
        config = create_config_template("small")
        
        assert config.model.hidden_dim == 128
        assert config.model.num_layers == 4
        assert config.training.batch_size == 16
    
    def test_large_template(self):
        """Test large config template."""
        config = create_config_template("large")
        
        assert config.model.hidden_dim == 512
        assert config.model.num_layers == 12
        assert config.training.gradient_accumulation_steps == 4
    
    def test_distributed_template(self):
        """Test distributed config template."""
        config = create_config_template("distributed")
        
        assert config.distributed.enabled is True
        assert config.distributed.zero_optimization is True
    
    def test_unknown_template(self):
        """Test unknown template type."""
        with pytest.raises(ValueError, match="Unknown config type"):
            create_config_template("unknown")


class TestConfigWithOverrides:
    """Test config loading with overrides."""
    
    def test_load_with_base_config(self, tmp_path):
        """Test loading with base config file."""
        # Create base config
        base_config = Config()
        base_config.training.epochs = 200
        base_config.model.hidden_dim = 512
        
        base_path = tmp_path / "base.json"
        base_config.save(base_path)
        
        # Load with base config
        loaded_config = load_config_with_overrides(
            base_config_path=str(base_path),
            env_overrides=False
        )
        
        assert loaded_config.training.epochs == 200
        assert loaded_config.model.hidden_dim == 512
    
    def test_load_with_override_config(self, tmp_path):
        """Test loading with override config."""
        # Create base config
        base_config = Config()
        base_config.training.epochs = 100
        base_config.model.hidden_dim = 256
        
        base_path = tmp_path / "base.json"
        base_config.save(base_path)
        
        # Create override config
        override_config = Config()
        override_config.training.epochs = 50  # Override
        
        override_path = tmp_path / "override.json"
        override_config.save(override_path)
        
        # Load with both configs
        loaded_config = load_config_with_overrides(
            base_config_path=str(base_path),
            override_config_path=str(override_path),
            env_overrides=False
        )
        
        assert loaded_config.training.epochs == 50  # Overridden
        assert loaded_config.model.hidden_dim == 256  # From base
    
    def test_config_validation_failure(self):
        """Test config validation failure."""
        with pytest.raises(ValueError, match="Configuration validation failed"):
            # Empty variables list should fail validation
            load_config_with_overrides(env_overrides=False)