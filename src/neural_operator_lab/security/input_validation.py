"""Advanced input validation and secure model loading."""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import hashlib
import os
import pickle
from pathlib import Path
import logging
import json
from dataclasses import dataclass
import warnings


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    max_tensor_size: int = 1e9  # Max elements in tensor
    max_batch_size: int = 1024  # Max batch size
    allowed_dtypes: List[str] = None  # Allowed tensor dtypes
    max_memory_mb: int = 16384  # Max memory usage in MB
    check_nan_inf: bool = True  # Check for NaN/Inf values
    validate_ranges: bool = True  # Validate value ranges
    
    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = ['float32', 'float64', 'int32', 'int64']


class InputValidator:
    """Comprehensive input validation for neural operators."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate_tensor_input(self, tensor: torch.Tensor, name: str = "input") -> torch.Tensor:
        """Validate tensor input with comprehensive security checks."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        # Size validation
        if tensor.numel() > self.config.max_tensor_size:
            raise ValueError(
                f"{name} tensor too large: {tensor.numel()} elements "
                f"(max: {self.config.max_tensor_size})"
            )
        
        # Batch size validation
        if tensor.dim() > 0 and tensor.size(0) > self.config.max_batch_size:
            raise ValueError(
                f"{name} batch size too large: {tensor.size(0)} "
                f"(max: {self.config.max_batch_size})"
            )
        
        # Data type validation
        if str(tensor.dtype).replace('torch.', '') not in self.config.allowed_dtypes:
            raise ValueError(
                f"{name} dtype {tensor.dtype} not allowed. "
                f"Allowed: {self.config.allowed_dtypes}"
            )
        
        # Memory usage validation
        memory_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
        if memory_mb > self.config.max_memory_mb:
            raise ValueError(
                f"{name} memory usage too high: {memory_mb:.1f}MB "
                f"(max: {self.config.max_memory_mb}MB)"
            )
        
        # NaN/Inf validation
        if self.config.check_nan_inf:
            if torch.isnan(tensor).any():
                raise ValueError(f"{name} contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError(f"{name} contains infinite values")
        
        # Range validation for common physics variables
        if self.config.validate_ranges and tensor.dim() > 0:
            self._validate_physics_ranges(tensor, name)
        
        self.logger.debug(f"Validated {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        return tensor
    
    def _validate_physics_ranges(self, tensor: torch.Tensor, name: str):
        """Validate ranges for physics variables."""
        # Common physics variable ranges
        range_checks = {
            'velocity': (-1000, 1000),  # m/s
            'pressure': (-1e6, 1e8),    # Pa
            'temperature': (0, 5000),   # K
            'density': (0, 20000),      # kg/m³
            'energy': (0, 1e15),        # J
        }
        
        name_lower = name.lower()
        for var_name, (min_val, max_val) in range_checks.items():
            if var_name in name_lower:
                if tensor.min() < min_val or tensor.max() > max_val:
                    warnings.warn(
                        f"{name} values outside typical {var_name} range "
                        f"[{min_val}, {max_val}]: [{tensor.min():.2e}, {tensor.max():.2e}]"
                    )
                break
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration parameters."""
        validated_config = {}
        
        # Required fields
        required_fields = ['input_dim', 'output_dim']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required config field missing: {field}")
            validated_config[field] = config[field]
        
        # Dimension validation
        if not isinstance(config['input_dim'], int) or config['input_dim'] <= 0:
            raise ValueError("input_dim must be positive integer")
        if not isinstance(config['output_dim'], int) or config['output_dim'] <= 0:
            raise ValueError("output_dim must be positive integer")
        
        # Hidden dimension validation
        if 'hidden_dim' in config:
            hidden_dim = config['hidden_dim']
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive integer")
            if hidden_dim > 8192:  # Reasonable upper limit
                warnings.warn(f"Very large hidden_dim: {hidden_dim}")
            validated_config['hidden_dim'] = hidden_dim
        
        # Layer count validation
        if 'num_layers' in config:
            num_layers = config['num_layers']
            if not isinstance(num_layers, int) or num_layers <= 0:
                raise ValueError("num_layers must be positive integer")
            if num_layers > 100:  # Reasonable upper limit
                warnings.warn(f"Very deep model: {num_layers} layers")
            validated_config['num_layers'] = num_layers
        
        # Learning rate validation
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError("learning_rate must be positive number")
            if lr > 1.0:
                warnings.warn(f"Very high learning rate: {lr}")
            validated_config['learning_rate'] = lr
        
        # Copy other safe parameters
        safe_params = {
            'batch_size', 'max_epochs', 'patience', 'weight_decay',
            'dropout_rate', 'activation', 'normalization'
        }
        for param in safe_params:
            if param in config:
                validated_config[param] = config[param]
        
        return validated_config
    
    def sanitize_file_path(self, path: str) -> Path:
        """Sanitize and validate file paths."""
        path_obj = Path(path).resolve()
        
        # Check for path traversal attempts
        if '..' in str(path_obj):
            raise ValueError("Path traversal detected")
        
        # Restrict to allowed directories (configurable)
        allowed_dirs = [
            Path.cwd(),
            Path.home() / 'data',
            Path('/tmp'),
            Path('/var/tmp')
        ]
        
        is_allowed = any(
            str(path_obj).startswith(str(allowed_dir))
            for allowed_dir in allowed_dirs
        )
        
        if not is_allowed:
            raise ValueError(f"Path not in allowed directories: {path_obj}")
        
        return path_obj


class SecureModelLoader:
    """Secure model loading with validation and integrity checks."""
    
    def __init__(self, validator: Optional[InputValidator] = None):
        self.validator = validator or InputValidator()
        self.logger = logging.getLogger(__name__)
        
        # Track loaded models for integrity
        self.loaded_models: Dict[str, str] = {}  # path -> hash
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def validate_model_file(self, file_path: str) -> Path:
        """Validate model file before loading."""
        path = self.validator.sanitize_file_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # File size check (prevent loading huge files)
        max_size_mb = 5000  # 5GB limit
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"Model file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
        
        # File extension check
        allowed_extensions = {'.pt', '.pth', '.ckpt', '.safetensors'}
        if path.suffix not in allowed_extensions:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        
        return path
    
    def safe_load_checkpoint(self, file_path: str, 
                           expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Safely load model checkpoint with validation."""
        path = self.validate_model_file(file_path)
        
        # Calculate and store hash
        file_hash = self.calculate_file_hash(path)
        self.loaded_models[str(path)] = file_hash
        
        try:
            # Load with restricted pickle protocols
            checkpoint = torch.load(
                path, 
                map_location='cpu',
                weights_only=True  # PyTorch 1.13+ security feature
            )
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise ValueError(f"Invalid checkpoint file: {e}")
        
        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary")
        
        # Check expected keys
        if expected_keys:
            missing_keys = set(expected_keys) - set(checkpoint.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Validate state dictionaries
        if 'model_state_dict' in checkpoint:
            self._validate_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self._validate_optimizer_state(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Successfully loaded checkpoint: {path}")
        return checkpoint
    
    def _validate_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Validate model state dictionary."""
        if not isinstance(state_dict, dict):
            raise ValueError("State dict must be a dictionary")
        
        total_params = 0
        for name, tensor in state_dict.items():
            if not isinstance(name, str):
                raise ValueError(f"Parameter name must be string: {name}")
            
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Parameter {name} must be tensor")
            
            # Validate parameter tensor
            self.validator.validate_tensor_input(tensor, f"parameter_{name}")
            total_params += tensor.numel()
        
        # Check total parameter count
        max_params = 20e9  # 20B parameters max
        if total_params > max_params:
            raise ValueError(f"Too many parameters: {total_params} (max: {max_params})")
        
        self.logger.debug(f"Validated state dict with {total_params:,} parameters")
    
    def _validate_optimizer_state(self, optimizer_state: Dict[str, Any]):
        """Validate optimizer state dictionary."""
        if not isinstance(optimizer_state, dict):
            raise ValueError("Optimizer state must be a dictionary")
        
        # Validate state components
        if 'state' in optimizer_state:
            for param_id, param_state in optimizer_state['state'].items():
                if isinstance(param_state, dict):
                    for key, value in param_state.items():
                        if isinstance(value, torch.Tensor):
                            self.validator.validate_tensor_input(
                                value, f"optimizer_state_{param_id}_{key}"
                            )
    
    def verify_model_integrity(self, file_path: str) -> bool:
        """Verify model file integrity using stored hash."""
        path = Path(file_path).resolve()
        path_str = str(path)
        
        if path_str not in self.loaded_models:
            self.logger.warning(f"No integrity hash found for: {path}")
            return False
        
        current_hash = self.calculate_file_hash(path)
        stored_hash = self.loaded_models[path_str]
        
        if current_hash != stored_hash:
            self.logger.error(f"Integrity check failed for: {path}")
            return False
        
        return True
    
    def create_secure_checkpoint(self, 
                               checkpoint: Dict[str, Any],
                               file_path: str,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create checkpoint with security metadata."""
        path = self.validator.sanitize_file_path(file_path)
        
        # Add security metadata
        secure_checkpoint = checkpoint.copy()
        secure_checkpoint['_security_metadata'] = {
            'created_timestamp': torch.tensor(torch.get_rng_state().sum().item()),
            'pytorch_version': torch.__version__,
            'validator_version': '1.0.0',
            'integrity_hash': None,  # Will be filled after saving
            'custom_metadata': metadata or {}
        }
        
        # Save checkpoint
        torch.save(secure_checkpoint, path)
        
        # Calculate and update integrity hash
        file_hash = self.calculate_file_hash(path)
        self.loaded_models[str(path)] = file_hash
        
        # Update checkpoint with hash
        secure_checkpoint['_security_metadata']['integrity_hash'] = file_hash
        torch.save(secure_checkpoint, path)
        
        self.logger.info(f"Created secure checkpoint: {path}")
        return file_hash