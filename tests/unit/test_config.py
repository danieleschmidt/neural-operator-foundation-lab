"""Tests for configuration module."""

import pytest
from neural_operator_lab.config import configure_security, get_security_config


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