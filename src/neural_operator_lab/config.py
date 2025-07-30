"""Configuration and security settings for Neural Operator Foundation Lab."""

import os
from typing import Dict, List, Optional, Union


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


# Initialize from environment variables
if os.getenv("NEURAL_OP_SECURE_MODE", "").lower() == "true":
    configure_security(secure_mode=True)

if os.getenv("NEURAL_OP_VALIDATE_MODELS", "").lower() == "true":
    configure_security(validate_models=True)

if os.getenv("NEURAL_OP_SANDBOX_EXECUTION", "").lower() == "true":
    configure_security(sandbox_execution=True)