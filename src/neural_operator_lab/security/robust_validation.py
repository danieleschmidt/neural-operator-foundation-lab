"""Enhanced security validation for neural operator models and data."""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import hashlib
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.warnings:
            self.warnings = []
        if not self.errors:
            self.errors = []
        if not self.metadata:
            self.metadata = {}


class RobustInputValidator:
    """Enhanced input validation with security checks."""
    
    def __init__(self, 
                 max_tensor_size: int = 1e9,
                 allowed_dtypes: List[torch.dtype] = None,
                 max_batch_size: int = 1024,
                 check_numerical_stability: bool = True):
        """Initialize robust input validator.
        
        Args:
            max_tensor_size: Maximum allowed tensor size in bytes
            allowed_dtypes: List of allowed tensor data types
            max_batch_size: Maximum allowed batch size
            check_numerical_stability: Whether to check for numerical issues
        """
        self.max_tensor_size = max_tensor_size
        self.allowed_dtypes = allowed_dtypes or [
            torch.float32, torch.float64, torch.float16, torch.int32, torch.int64
        ]
        self.max_batch_size = max_batch_size
        self.check_numerical_stability = check_numerical_stability
        
        # Statistics for monitoring
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'last_validation': None
        }
    
    def validate_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> ValidationResult:
        """Validate a single tensor."""
        warnings_list = []
        errors = []
        metadata = {}
        
        try:
            # Basic type and shape validation
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"{name} is not a torch.Tensor (type: {type(tensor)})")
                return ValidationResult(False, warnings_list, errors, metadata)
            
            # Data type validation
            if tensor.dtype not in self.allowed_dtypes:
                errors.append(f"{name} has disallowed dtype: {tensor.dtype}")
            
            # Size validation
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size > self.max_tensor_size:
                errors.append(f"{name} exceeds size limit: {tensor_size} > {self.max_tensor_size}")
            
            # Batch size validation
            if len(tensor.shape) > 0 and tensor.shape[0] > self.max_batch_size:
                warnings_list.append(f"{name} has large batch size: {tensor.shape[0]} > {self.max_batch_size}")
            
            # Shape validation
            if len(tensor.shape) > 6:
                warnings_list.append(f"{name} has unusually high dimensionality: {len(tensor.shape)}")
            
            # Numerical stability checks
            if self.check_numerical_stability:
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                
                if nan_count > 0:
                    errors.append(f"{name} contains {nan_count} NaN values")
                
                if inf_count > 0:
                    errors.append(f"{name} contains {inf_count} infinite values")
                
                # Check for very large values that might cause overflow
                if tensor.dtype in [torch.float32, torch.float64]:
                    max_abs = torch.max(torch.abs(tensor)).item()
                    if max_abs > 1e10:
                        warnings_list.append(f"{name} contains very large values (max: {max_abs:.2e})")
                    
                    # Check for very small values that might cause underflow
                    min_nonzero = torch.min(torch.abs(tensor[tensor != 0])).item() if (tensor != 0).any() else 0
                    if min_nonzero > 0 and min_nonzero < 1e-10:
                        warnings_list.append(f"{name} contains very small values (min: {min_nonzero:.2e})")
            
            # Metadata collection
            metadata.update({
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'size_bytes': tensor_size,
                'min_value': float(torch.min(tensor).item()) if tensor.numel() > 0 else None,
                'max_value': float(torch.max(tensor).item()) if tensor.numel() > 0 else None,
                'mean_value': float(torch.mean(tensor.float()).item()) if tensor.numel() > 0 else None
            })
            
        except Exception as e:
            errors.append(f"Validation failed for {name}: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, warnings_list, errors, metadata)
    
    def validate_model_input(self, x: torch.Tensor, 
                           expected_shape: Optional[Tuple[int, ...]] = None,
                           grid: Optional[torch.Tensor] = None) -> ValidationResult:
        """Validate model input with optional grid."""
        all_warnings = []
        all_errors = []
        all_metadata = {}
        
        # Validate main input
        input_result = self.validate_tensor(x, "input")
        all_warnings.extend(input_result.warnings)
        all_errors.extend(input_result.errors)
        all_metadata['input'] = input_result.metadata
        
        # Shape validation
        if expected_shape is not None:
            if x.shape != expected_shape:
                all_errors.append(f"Input shape mismatch: expected {expected_shape}, got {x.shape}")
        
        # Validate grid if provided
        if grid is not None:
            grid_result = self.validate_tensor(grid, "grid")
            all_warnings.extend(grid_result.warnings)
            all_errors.extend(grid_result.errors)
            all_metadata['grid'] = grid_result.metadata
            
            # Check grid-input compatibility
            if x.shape[0] != grid.shape[0]:
                all_errors.append(f"Batch size mismatch: input={x.shape[0]}, grid={grid.shape[0]}")
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        self.validation_stats['warnings_count'] += len(all_warnings)
        self.validation_stats['errors_count'] += len(all_errors)
        self.validation_stats['last_validation'] = time.time()
        
        if len(all_errors) == 0:
            self.validation_stats['successful_validations'] += 1
        
        is_valid = len(all_errors) == 0
        return ValidationResult(is_valid, all_warnings, all_errors, all_metadata)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()


class ModelSecurityChecker:
    """Security checker for neural operator models."""
    
    def __init__(self):
        self.known_safe_hashes = set()
        self.security_warnings = []
    
    def compute_model_hash(self, model: torch.nn.Module) -> str:
        """Compute secure hash of model parameters."""
        hasher = hashlib.sha256()
        
        for name, param in model.named_parameters():
            if param is not None:
                param_bytes = param.detach().cpu().numpy().tobytes()
                hasher.update(name.encode('utf-8'))
                hasher.update(param_bytes)
        
        return hasher.hexdigest()
    
    def validate_model_architecture(self, model: torch.nn.Module) -> ValidationResult:
        """Validate model architecture for security issues."""
        warnings_list = []
        errors = []
        metadata = {}
        
        try:
            # Check parameter count
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > 1e9:  # More than 1B parameters
                warnings_list.append(f"Model has very large number of parameters: {total_params:,}")
            
            # Check for suspicious layer types
            suspicious_layers = []
            for name, module in model.named_modules():
                module_type = type(module).__name__
                if 'Exec' in module_type or 'Eval' in module_type:
                    suspicious_layers.append((name, module_type))
            
            if suspicious_layers:
                errors.append(f"Potentially dangerous layers found: {suspicious_layers}")
            
            # Check for unusual activations or operations
            unusual_modules = []
            for name, module in model.named_modules():
                if hasattr(module, 'forward') and hasattr(module.forward, '__code__'):
                    # This is a basic check - in practice, more sophisticated analysis needed
                    if 'exec' in str(module.forward.__code__.co_names):
                        unusual_modules.append(name)
            
            if unusual_modules:
                warnings_list.append(f"Modules with potentially unsafe operations: {unusual_modules}")
            
            # Model metadata
            metadata.update({
                'total_parameters': total_params,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_hash': self.compute_model_hash(model),
                'layer_count': len(list(model.modules())),
                'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
            })
            
        except Exception as e:
            errors.append(f"Model validation failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, warnings_list, errors, metadata)
    
    def validate_model_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> ValidationResult:
        """Validate model state dictionary."""
        warnings_list = []
        errors = []
        metadata = {}
        
        try:
            total_params = 0
            suspicious_keys = []
            
            for key, tensor in state_dict.items():
                # Validate tensor
                tensor_result = RobustInputValidator().validate_tensor(tensor, f"state_dict[{key}]")
                if not tensor_result.is_valid:
                    errors.extend(tensor_result.errors)
                warnings_list.extend(tensor_result.warnings)
                
                total_params += tensor.numel()
                
                # Check for suspicious parameter names
                if any(suspicious in key.lower() for suspicious in ['exec', 'eval', 'import', '__']):
                    suspicious_keys.append(key)
            
            if suspicious_keys:
                warnings_list.append(f"Suspicious parameter names found: {suspicious_keys}")
            
            metadata.update({
                'parameter_count': total_params,
                'state_dict_keys': list(state_dict.keys()),
                'total_size_mb': sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 ** 2)
            })
            
        except Exception as e:
            errors.append(f"State dict validation failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, warnings_list, errors, metadata)
    
    def add_safe_hash(self, model_hash: str):
        """Add a model hash to the known safe list."""
        self.known_safe_hashes.add(model_hash)
    
    def is_known_safe_model(self, model: torch.nn.Module) -> bool:
        """Check if model is in the known safe list."""
        model_hash = self.compute_model_hash(model)
        return model_hash in self.known_safe_hashes


class DataIntegrityChecker:
    """Check data integrity and detect potential attacks."""
    
    def __init__(self):
        self.baseline_stats = {}
        self.anomaly_threshold = 3.0  # Standard deviations
    
    def establish_baseline(self, dataset_name: str, data: torch.Tensor):
        """Establish baseline statistics for a dataset."""
        stats = {
            'mean': float(torch.mean(data).item()),
            'std': float(torch.std(data).item()),
            'min': float(torch.min(data).item()),
            'max': float(torch.max(data).item()),
            'shape': list(data.shape),
            'dtype': str(data.dtype)
        }
        self.baseline_stats[dataset_name] = stats
        logger.info(f"Established baseline for {dataset_name}: {stats}")
    
    def check_data_anomalies(self, dataset_name: str, data: torch.Tensor) -> ValidationResult:
        """Check for data anomalies compared to baseline."""
        warnings_list = []
        errors = []
        metadata = {}
        
        if dataset_name not in self.baseline_stats:
            warnings_list.append(f"No baseline established for dataset {dataset_name}")
            return ValidationResult(True, warnings_list, errors, metadata)
        
        baseline = self.baseline_stats[dataset_name]
        
        try:
            current_stats = {
                'mean': float(torch.mean(data).item()),
                'std': float(torch.std(data).item()),
                'min': float(torch.min(data).item()),
                'max': float(torch.max(data).item()),
            }
            
            # Check for statistical anomalies
            for stat_name, current_value in current_stats.items():
                if stat_name in baseline:
                    baseline_value = baseline[stat_name]
                    baseline_std = baseline.get('std', 1.0)
                    
                    if baseline_std > 0:
                        z_score = abs(current_value - baseline_value) / baseline_std
                        if z_score > self.anomaly_threshold:
                            warnings_list.append(
                                f"Statistical anomaly in {stat_name}: "
                                f"current={current_value:.6f}, baseline={baseline_value:.6f}, "
                                f"z_score={z_score:.2f}"
                            )
            
            # Check shape consistency
            if list(data.shape[1:]) != baseline['shape'][1:]:  # Ignore batch dimension
                warnings_list.append(f"Shape mismatch: expected {baseline['shape']}, got {list(data.shape)}")
            
            # Check dtype consistency
            if str(data.dtype) != baseline['dtype']:
                warnings_list.append(f"Data type changed: expected {baseline['dtype']}, got {data.dtype}")
            
            metadata.update({
                'current_stats': current_stats,
                'baseline_stats': baseline,
                'anomaly_scores': {
                    stat: abs(current_stats[stat] - baseline.get(stat, 0)) / baseline.get('std', 1)
                    for stat in current_stats
                }
            })
            
        except Exception as e:
            errors.append(f"Anomaly detection failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, warnings_list, errors, metadata)


class ComprehensiveSecurityValidator:
    """Comprehensive security validation combining all checkers."""
    
    def __init__(self, 
                 strict_mode: bool = False,
                 log_all_validations: bool = True):
        """Initialize comprehensive validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
            log_all_validations: Whether to log all validation results
        """
        self.input_validator = RobustInputValidator()
        self.model_checker = ModelSecurityChecker()
        self.data_checker = DataIntegrityChecker()
        self.strict_mode = strict_mode
        self.log_all_validations = log_all_validations
        
        self.validation_history = []
    
    def validate_training_batch(self, 
                              x: torch.Tensor, 
                              y: torch.Tensor,
                              model: torch.nn.Module,
                              dataset_name: str = "unknown") -> ValidationResult:
        """Comprehensive validation of a training batch."""
        all_warnings = []
        all_errors = []
        all_metadata = {}
        
        # Validate inputs
        input_result = self.input_validator.validate_model_input(x)
        all_warnings.extend(input_result.warnings)
        all_errors.extend(input_result.errors)
        all_metadata['input_validation'] = input_result.metadata
        
        # Validate targets
        target_result = self.input_validator.validate_tensor(y, "target")
        all_warnings.extend(target_result.warnings)
        all_errors.extend(target_result.errors)
        all_metadata['target_validation'] = target_result.metadata
        
        # Check for data anomalies
        data_anomaly_result = self.data_checker.check_data_anomalies(dataset_name, x)
        all_warnings.extend(data_anomaly_result.warnings)
        all_errors.extend(data_anomaly_result.errors)
        all_metadata['data_anomalies'] = data_anomaly_result.metadata
        
        # Validate model (periodically to avoid overhead)
        if len(self.validation_history) % 100 == 0:  # Every 100 validations
            model_result = self.model_checker.validate_model_architecture(model)
            all_warnings.extend(model_result.warnings)
            all_errors.extend(model_result.errors)
            all_metadata['model_validation'] = model_result.metadata
        
        # Apply strict mode
        if self.strict_mode and all_warnings:
            all_errors.extend(all_warnings)
            all_warnings = []
        
        # Log if requested
        if self.log_all_validations:
            if all_errors:
                logger.error(f"Validation errors: {all_errors}")
            if all_warnings:
                logger.warning(f"Validation warnings: {all_warnings}")
        
        # Record validation
        validation_record = {
            'timestamp': time.time(),
            'dataset': dataset_name,
            'warnings_count': len(all_warnings),
            'errors_count': len(all_errors),
            'batch_size': x.shape[0] if x.numel() > 0 else 0
        }
        self.validation_history.append(validation_record)
        
        # Keep history manageable
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        is_valid = len(all_errors) == 0
        all_metadata['validation_summary'] = validation_record
        
        return ValidationResult(is_valid, all_warnings, all_errors, all_metadata)
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            'validation_statistics': self.input_validator.get_validation_statistics(),
            'security_warnings': self.model_checker.security_warnings,
            'known_safe_models': len(self.model_checker.known_safe_hashes),
            'data_baselines': list(self.data_checker.baseline_stats.keys()),
            'recent_validations': len(self.validation_history),
            'strict_mode': self.strict_mode
        }
        
        # Recent validation summary
        if self.validation_history:
            recent = self.validation_history[-100:]  # Last 100 validations
            report['recent_summary'] = {
                'total_warnings': sum(v['warnings_count'] for v in recent),
                'total_errors': sum(v['errors_count'] for v in recent),
                'validation_rate': len(recent) / max(1, recent[-1]['timestamp'] - recent[0]['timestamp']),
                'datasets_seen': list(set(v['dataset'] for v in recent))
            }
        
        return report


# Global validator instance
_global_validator = None

def get_security_validator(strict_mode: bool = False) -> ComprehensiveSecurityValidator:
    """Get global security validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ComprehensiveSecurityValidator(strict_mode=strict_mode)
    return _global_validator