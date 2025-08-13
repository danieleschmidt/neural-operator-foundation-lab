"""Enhanced security module for neural operators."""

import hashlib
import hmac
import secrets
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import warnings
import tempfile
import functools
from contextlib import contextmanager

try:
    from ..robust_framework import robust_logger, ValidationError, SecurityError, InputValidator
    _HAS_ROBUST_FRAMEWORK = True
except ImportError:
    _HAS_ROBUST_FRAMEWORK = False
    
    class ValidationError(Exception):
        pass
    
    class SecurityError(Exception):
        pass


@dataclass
class SecurityConfig:
    """Security configuration for neural operators."""
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_model_integrity_checks: bool = True
    enable_audit_logging: bool = True
    max_model_size_mb: int = 1000
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.pt', '.pth', '.onnx', '.h5'])
    forbidden_paths: List[str] = field(default_factory=lambda: ['/etc', '/proc', '/sys'])
    enable_encryption: bool = False
    audit_log_retention_days: int = 90


class SecureModelLoader:
    """Secure model loading with integrity checks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.trusted_checksums = {}
        self.audit_log = []
        
    def register_trusted_model(self, model_path: str, checksum: str):
        """Register a trusted model with its checksum."""
        self.trusted_checksums[model_path] = checksum
        
    def compute_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Compute hash of a file."""
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except IOError as e:
            raise SecurityError(f"Failed to read file for hashing: {e}")
    
    def validate_model_file(self, model_path: Union[str, Path]) -> bool:
        """Validate model file security."""
        model_path = Path(model_path)
        
        # Check if file exists
        if not model_path.exists():
            raise SecurityError(f"Model file does not exist: {model_path}")
        
        # Check file extension
        if self.config.allowed_file_extensions:
            if model_path.suffix not in self.config.allowed_file_extensions:
                raise SecurityError(
                    f"File extension {model_path.suffix} not in allowed list: "
                    f"{self.config.allowed_file_extensions}"
                )
        
        # Check file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_model_size_mb:
            raise SecurityError(
                f"Model file size ({file_size_mb:.1f}MB) exceeds limit "
                f"({self.config.max_model_size_mb}MB)"
            )
        
        # Check forbidden paths
        for forbidden in self.config.forbidden_paths:
            if str(model_path).startswith(forbidden):
                raise SecurityError(f"Model path {model_path} is in forbidden location")
        
        # Check integrity if we have a trusted checksum
        model_str = str(model_path)
        if model_str in self.trusted_checksums:
            computed_hash = self.compute_file_hash(model_path)
            expected_hash = self.trusted_checksums[model_str]
            
            if computed_hash != expected_hash:
                raise SecurityError(
                    f"Model integrity check failed. Expected: {expected_hash}, "
                    f"Got: {computed_hash}"
                )
        
        return True
    
    def secure_load_model(self, model_path: Union[str, Path], 
                         loader_func: Callable,
                         **loader_kwargs) -> Any:
        """Securely load a model with validation."""
        model_path = Path(model_path)
        
        # Validate model file
        self.validate_model_file(model_path)
        
        # Log the loading attempt
        if self.config.enable_audit_logging:
            self._log_model_access(model_path, "load_attempt")
        
        try:
            # Load the model
            model = loader_func(str(model_path), **loader_kwargs)
            
            # Log successful loading
            if self.config.enable_audit_logging:
                self._log_model_access(model_path, "load_success")
            
            return model
            
        except Exception as e:
            # Log failed loading
            if self.config.enable_audit_logging:
                self._log_model_access(model_path, "load_failure", error=str(e))
            raise SecurityError(f"Failed to load model: {e}")
    
    def _log_model_access(self, model_path: Path, action: str, **kwargs):
        """Log model access for audit purposes."""
        log_entry = {
            'timestamp': time.time(),
            'action': action,
            'model_path': str(model_path),
            'file_size': model_path.stat().st_size if model_path.exists() else 0,
            'checksum': self.compute_file_hash(model_path) if model_path.exists() else None,
            **kwargs
        }
        
        self.audit_log.append(log_entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]


class InputSanitizer:
    """Sanitize and validate inputs for neural operators."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def sanitize_tensor_input(self, tensor: Any, name: str = "input") -> Any:
        """Sanitize tensor input."""
        if not self.config.enable_input_validation:
            return tensor
        
        # Basic validation
        if not hasattr(tensor, 'shape'):
            raise ValidationError(f"{name} must be a tensor-like object")
        
        # Check for suspicious values
        try:
            # Check for NaN/Inf
            if hasattr(tensor, 'isnan') and tensor.isnan().any():
                raise ValidationError(f"{name} contains NaN values")
            
            if hasattr(tensor, 'isinf') and tensor.isinf().any():
                raise ValidationError(f"{name} contains infinite values")
            
            # Check for extremely large values that might cause overflow
            if hasattr(tensor, 'abs'):
                max_val = tensor.abs().max()
                if hasattr(max_val, 'item'):
                    max_val = max_val.item()
                    if max_val > 1e10:
                        warnings.warn(f"{name} contains very large values (max: {max_val})")
                        
        except (AttributeError, RuntimeError):
            # Skip checks if tensor doesn't support these operations
            pass
        
        return tensor
    
    def sanitize_config_input(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration input."""
        if not isinstance(config, dict):
            raise ValidationError("Config must be a dictionary")
        
        # Remove potentially dangerous keys
        dangerous_keys = ['__', 'exec', 'eval', 'import', 'open', 'file']
        sanitized_config = {}
        
        for key, value in config.items():
            # Check for dangerous key patterns
            if any(dangerous in str(key).lower() for dangerous in dangerous_keys):
                warnings.warn(f"Skipping potentially dangerous config key: {key}")
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                # Remove potential code injection patterns
                dangerous_patterns = ['__import__', 'exec(', 'eval(', 'open(']
                if any(pattern in value for pattern in dangerous_patterns):
                    warnings.warn(f"Sanitizing potentially dangerous string value for key: {key}")
                    continue
            
            sanitized_config[key] = value
        
        return sanitized_config


class OutputSanitizer:
    """Sanitize outputs from neural operators."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def sanitize_tensor_output(self, tensor: Any, name: str = "output") -> Any:
        """Sanitize tensor output."""
        if not self.config.enable_output_sanitization:
            return tensor
        
        try:
            # Clamp extreme values
            if hasattr(tensor, 'clamp'):
                # Clamp to reasonable range
                tensor = tensor.clamp(-1e8, 1e8)
            
            # Replace NaN/Inf with zeros
            if hasattr(tensor, 'nan_to_num'):
                tensor = tensor.nan_to_num(nan=0.0, posinf=1e8, neginf=-1e8)
            
        except (AttributeError, RuntimeError):
            # Skip if tensor doesn't support these operations
            pass
        
        return tensor


class SecureTemporaryFiles:
    """Secure temporary file management."""
    
    def __init__(self):
        self.temp_files = []
        
    @contextmanager
    def create_temp_file(self, suffix: str = "", prefix: str = "neural_op_"):
        """Create a secure temporary file."""
        try:
            # Create temporary file with secure permissions
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(fd)  # Close the file descriptor
            
            # Set secure permissions (owner read/write only)
            os.chmod(temp_path, 0o600)
            
            self.temp_files.append(temp_path)
            yield Path(temp_path)
            
        finally:
            # Clean up
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if temp_path in self.temp_files:
                    self.temp_files.remove(temp_path)
            except OSError:
                pass
    
    def cleanup_all(self):
        """Clean up all temporary files."""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                self.temp_files.remove(temp_file)
            except OSError:
                pass


def secure_execution(security_config: Optional[SecurityConfig] = None):
    """Decorator for secure execution of neural operator functions."""
    if security_config is None:
        security_config = SecurityConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Input sanitization
            if security_config.enable_input_validation:
                sanitizer = InputSanitizer(security_config)
                
                # Sanitize tensor arguments
                sanitized_args = []
                for i, arg in enumerate(args):
                    if hasattr(arg, 'shape'):  # Likely a tensor
                        sanitized_args.append(
                            sanitizer.sanitize_tensor_input(arg, f"arg_{i}")
                        )
                    else:
                        sanitized_args.append(arg)
                
                # Sanitize keyword arguments
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if hasattr(value, 'shape'):  # Likely a tensor
                        sanitized_kwargs[key] = sanitizer.sanitize_tensor_input(value, key)
                    elif isinstance(value, dict):
                        sanitized_kwargs[key] = sanitizer.sanitize_config_input(value)
                    else:
                        sanitized_kwargs[key] = value
                
                args = tuple(sanitized_args)
                kwargs = sanitized_kwargs
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Output sanitization
            if security_config.enable_output_sanitization:
                output_sanitizer = OutputSanitizer(security_config)
                if hasattr(result, 'shape'):  # Likely a tensor
                    result = output_sanitizer.sanitize_tensor_output(result)
                elif isinstance(result, (list, tuple)):
                    # Sanitize tensor elements in collections
                    sanitized_result = []
                    for item in result:
                        if hasattr(item, 'shape'):
                            sanitized_result.append(
                                output_sanitizer.sanitize_tensor_output(item)
                            )
                        else:
                            sanitized_result.append(item)
                    result = type(result)(sanitized_result)
            
            return result
        
        return wrapper
    return decorator


class SecurityAuditor:
    """Security auditing and compliance checking."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_results = []
    
    def audit_model_security(self, model: Any) -> Dict[str, Any]:
        """Audit model security."""
        audit_result = {
            'timestamp': time.time(),
            'model_type': type(model).__name__,
            'issues': [],
            'recommendations': [],
            'score': 100  # Start with perfect score
        }
        
        # Check for potentially dangerous attributes
        dangerous_attrs = ['exec', 'eval', 'compile', '__import__']
        for attr in dangerous_attrs:
            if hasattr(model, attr):
                audit_result['issues'].append(f"Model has dangerous attribute: {attr}")
                audit_result['score'] -= 20
        
        # Check model size
        try:
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                if param_count > 1e9:  # 1B parameters
                    audit_result['issues'].append(f"Very large model ({param_count:,} parameters)")
                    audit_result['recommendations'].append("Consider model compression")
        except Exception:
            pass
        
        # Check for serialization safety
        if hasattr(model, '__reduce__') or hasattr(model, '__reduce_ex__'):
            audit_result['issues'].append("Model has custom serialization methods")
            audit_result['recommendations'].append("Verify serialization methods are safe")
            audit_result['score'] -= 10
        
        self.audit_results.append(audit_result)
        return audit_result
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.audit_results:
            return {'status': 'no_audits_performed'}
        
        total_audits = len(self.audit_results)
        total_issues = sum(len(result['issues']) for result in self.audit_results)
        avg_score = sum(result['score'] for result in self.audit_results) / total_audits
        
        return {
            'summary': {
                'total_audits': total_audits,
                'total_issues': total_issues,
                'average_security_score': avg_score,
                'overall_status': 'good' if avg_score >= 80 else 'needs_attention'
            },
            'recent_audits': self.audit_results[-10:],
            'common_issues': self._get_common_issues(),
            'recommendations': self._get_common_recommendations()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get most common security issues."""
        issue_counts = {}
        for result in self.audit_results:
            for issue in result['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return sorted(issue_counts.keys(), key=issue_counts.get, reverse=True)[:5]
    
    def _get_common_recommendations(self) -> List[str]:
        """Get most common recommendations."""
        rec_counts = {}
        for result in self.audit_results:
            for rec in result['recommendations']:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        return sorted(rec_counts.keys(), key=rec_counts.get, reverse=True)[:5]


# Global instances
_default_security_config = SecurityConfig()
_default_model_loader = SecureModelLoader(_default_security_config)
_default_security_auditor = SecurityAuditor(_default_security_config)


def get_default_security_config() -> SecurityConfig:
    """Get default security configuration."""
    return _default_security_config


def get_secure_model_loader() -> SecureModelLoader:
    """Get default secure model loader."""
    return _default_model_loader


def get_security_auditor() -> SecurityAuditor:
    """Get default security auditor."""
    return _default_security_auditor