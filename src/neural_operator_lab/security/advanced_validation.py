"""Advanced security validation for neural operator frameworks."""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import pickle
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
import warnings
import tempfile
import subprocess
import ast


@dataclass
class SecurityConfig:
    """Security configuration for neural operators."""
    max_model_size_mb: int = 500
    max_tensor_size_gb: int = 8
    max_computation_time_s: int = 300
    allowed_file_extensions: List[str] = None
    scan_for_malicious_code: bool = True
    validate_computational_graph: bool = True
    check_memory_usage: bool = True
    enable_secure_serialization: bool = True
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.pt', '.pth', '.h5', '.hdf5', '.npz']


class SecurityScanner:
    """Comprehensive security scanner for models and data."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Known dangerous patterns
        self.dangerous_patterns = [
            # Python execution patterns
            'exec(', 'eval(', '__import__', 'compile(',
            # System command patterns  
            'os.system', 'subprocess.', 'popen(', 'shell=True',
            # File system patterns
            'open(', 'file(', '__file__', 'pickle.loads',
            # Network patterns
            'urllib', 'requests.', 'socket.', 'http',
            # Dangerous builtins
            'globals()', 'locals()', 'vars()', 'dir()'
        ]
    
    def scan_model_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive security scan of model files."""
        scan_results = {
            'file_path': file_path,
            'safe': True,
            'warnings': [],
            'errors': [],
            'file_info': {}
        }
        
        try:
            # Basic file validation
            if not os.path.exists(file_path):
                scan_results['errors'].append(f"File does not exist: {file_path}")
                scan_results['safe'] = False
                return scan_results
            
            # File size check
            file_size = os.path.getsize(file_path)
            scan_results['file_info']['size_mb'] = file_size / (1024 * 1024)
            
            if file_size > self.config.max_model_size_mb * 1024 * 1024:
                scan_results['errors'].append(f"File too large: {file_size / (1024*1024):.1f}MB")
                scan_results['safe'] = False
            
            # Extension check
            _, ext = os.path.splitext(file_path)
            if ext not in self.config.allowed_file_extensions:
                scan_results['warnings'].append(f"Unusual file extension: {ext}")
            
            # Content scanning
            if self.config.scan_for_malicious_code:
                content_scan = self._scan_file_content(file_path)
                scan_results.update(content_scan)
            
            # PyTorch model specific checks
            if ext in ['.pt', '.pth']:
                model_scan = self._scan_pytorch_model(file_path)
                scan_results.update(model_scan)
            
        except Exception as e:
            scan_results['errors'].append(f"Scan error: {e}")
            scan_results['safe'] = False
        
        return scan_results
    
    def _scan_file_content(self, file_path: str) -> Dict[str, Any]:
        """Scan file content for malicious patterns."""
        results = {'content_warnings': [], 'content_errors': []}
        
        try:
            # Try to read as text first
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # First 10KB
                
                for pattern in self.dangerous_patterns:
                    if pattern in content:
                        results['content_warnings'].append(f"Suspicious pattern found: {pattern}")
        
        except UnicodeDecodeError:
            # Binary file, limited scanning
            with open(file_path, 'rb') as f:
                content = f.read(10000)
                
                # Check for embedded Python code in binary
                if b'exec(' in content or b'eval(' in content:
                    results['content_errors'].append("Embedded Python execution code in binary")
        
        except Exception as e:
            results['content_errors'].append(f"Content scan error: {e}")
        
        return results
    
    def _scan_pytorch_model(self, file_path: str) -> Dict[str, Any]:
        """Scan PyTorch model for security issues."""
        results = {'model_warnings': [], 'model_errors': []}
        
        try:
            # Use pickle protocol to inspect without loading
            with open(file_path, 'rb') as f:
                # Check if it's a valid pickle file
                try:
                    pickler = pickle.Unpickler(f)
                    # Don't actually load, just check structure
                    f.seek(0)
                    
                    # Read first few bytes to check pickle protocol
                    header = f.read(100)
                    if b'pytorch' not in header.lower() and b'torch' not in header.lower():
                        results['model_warnings'].append("File doesn't appear to be a PyTorch model")
                
                except pickle.UnpicklingError:
                    results['model_errors'].append("Invalid pickle format")
        
        except Exception as e:
            results['model_errors'].append(f"PyTorch scan error: {e}")
        
        return results
    
    def validate_tensor_safety(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Validate tensor for security and safety issues."""
        results = {
            'tensor_name': name,
            'safe': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            # Size validation
            total_elements = tensor.numel()
            memory_usage_gb = total_elements * tensor.element_size() / (1024**3)
            
            results['stats'] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'total_elements': total_elements,
                'memory_gb': memory_usage_gb,
                'device': str(tensor.device)
            }
            
            if memory_usage_gb > self.config.max_tensor_size_gb:
                results['errors'].append(f"Tensor too large: {memory_usage_gb:.1f}GB")
                results['safe'] = False
            
            # Value validation
            if torch.isnan(tensor).any():
                results['warnings'].append("Tensor contains NaN values")
            
            if torch.isinf(tensor).any():
                results['warnings'].append("Tensor contains infinite values")
            
            # Range validation
            if tensor.dtype in [torch.float32, torch.float64]:
                max_val = torch.max(torch.abs(tensor)).item()
                if max_val > 1e6:
                    results['warnings'].append(f"Large values detected: max={max_val:.2e}")
                
                min_val = torch.min(torch.abs(tensor[tensor != 0])).item() if (tensor != 0).any() else 0
                if min_val < 1e-10 and max_val > 1e-10:
                    results['warnings'].append("Extreme dynamic range detected")
        
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
            results['safe'] = False
        
        return results


class SecureModelLoader:
    """Secure model loading with validation and sandboxing."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.scanner = SecurityScanner(config)
        self.logger = logging.getLogger(__name__)
    
    def load_model_safely(self, file_path: str, map_location: str = 'cpu') -> Tuple[Any, Dict[str, Any]]:
        """Load model with comprehensive security validation."""
        # Security scan first
        scan_results = self.scanner.scan_model_file(file_path)
        
        if not scan_results['safe']:
            raise SecurityError(f"Model failed security scan: {scan_results['errors']}")
        
        if scan_results['warnings']:
            for warning in scan_results['warnings']:
                self.logger.warning(f"Model security warning: {warning}")
        
        try:
            # Load in sandboxed environment
            if self.config.enable_secure_serialization:
                model = self._load_with_sandbox(file_path, map_location)
            else:
                model = torch.load(file_path, map_location=map_location)
            
            # Validate loaded model
            model_validation = self._validate_loaded_model(model)
            scan_results.update(model_validation)
            
            return model, scan_results
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load model safely: {e}")
    
    def _load_with_sandbox(self, file_path: str, map_location: str) -> Any:
        """Load model in sandboxed environment."""
        # Create restricted unpickler
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Allow only specific safe modules
                safe_modules = {
                    'torch': ['Tensor', 'Size', 'dtype', 'device'],
                    'torch.nn': ['Module', 'Parameter'],
                    'torch.nn.modules': ['*'],  # Allow all nn modules
                    'collections': ['OrderedDict'],
                    'numpy': ['ndarray', 'dtype'],
                    '__builtin__': ['object'],
                    'builtins': ['object']
                }
                
                if module in safe_modules:
                    allowed = safe_modules[module]
                    if '*' in allowed or name in allowed:
                        return super().find_class(module, name)
                
                raise pickle.UnpicklingError(f"Blocked unsafe class: {module}.{name}")
        
        with open(file_path, 'rb') as f:
            unpickler = RestrictedUnpickler(f)
            return unpickler.load()
    
    def _validate_loaded_model(self, model: Any) -> Dict[str, Any]:
        """Validate loaded model structure and properties."""
        results = {'model_validation': []}
        
        try:
            if isinstance(model, nn.Module):
                # Check for suspicious operations
                suspicious_ops = []
                for name, module in model.named_modules():
                    if hasattr(module, '__class__'):
                        class_name = module.__class__.__name__
                        if 'Exec' in class_name or 'Eval' in class_name:
                            suspicious_ops.append(f"Suspicious module: {name} ({class_name})")
                
                if suspicious_ops:
                    results['model_validation'].extend(suspicious_ops)
                
                # Memory usage check
                param_count = sum(p.numel() for p in model.parameters())
                memory_mb = param_count * 4 / (1024 * 1024)  # Assume float32
                
                if memory_mb > self.config.max_model_size_mb:
                    results['model_validation'].append(f"Model too large: {memory_mb:.1f}MB")
        
        except Exception as e:
            results['model_validation'].append(f"Validation error: {e}")
        
        return results


class ComputationMonitor:
    """Monitor computation resources and detect potential attacks."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        """Start monitoring computation."""
        import time
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Check computation limits on exit."""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            
            if elapsed > self.config.max_computation_time_s:
                self.logger.warning(f"Computation exceeded time limit: {elapsed:.1f}s")
        
        if self.start_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - self.start_memory) / (1024**3)
            
            if memory_used > self.config.max_tensor_size_gb:
                self.logger.warning(f"Computation exceeded memory limit: {memory_used:.1f}GB")


class SecurityError(Exception):
    """Security-related error."""
    pass


class ModelLoadError(Exception):
    """Model loading error."""
    pass


def secure_tensor_operation(operation: Callable) -> Callable:
    """Decorator to secure tensor operations."""
    def wrapper(*args, **kwargs):
        # Pre-operation validation
        scanner = SecurityScanner()
        
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                validation = scanner.validate_tensor_safety(arg, f"arg_{i}")
                if not validation['safe']:
                    raise SecurityError(f"Unsafe tensor in argument {i}: {validation['errors']}")
        
        # Execute with monitoring
        with ComputationMonitor():
            result = operation(*args, **kwargs)
        
        # Post-operation validation
        if isinstance(result, torch.Tensor):
            validation = scanner.validate_tensor_safety(result, "result")
            if not validation['safe']:
                raise SecurityError(f"Unsafe result tensor: {validation['errors']}")
        
        return result
    
    return wrapper