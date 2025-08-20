"""Robust error handling for neural operator models."""

import torch
import torch.nn as nn
import logging
import warnings
from typing import Optional, Dict, Any, Callable, Type, List
from functools import wraps
from contextlib import contextmanager
import traceback


class ModelError(Exception):
    """Base exception class for model errors."""
    pass


class InvalidInputError(ModelError):
    """Raised when model receives invalid input."""
    pass


class ComputationError(ModelError):
    """Raised when computation fails during forward pass."""
    pass


class MemoryError(ModelError):
    """Raised when model runs out of memory."""
    pass


class ConvergenceError(ModelError):
    """Raised when iterative methods fail to converge."""
    pass


def handle_model_errors(error_callback: Optional[Callable] = None):
    """Decorator for robust error handling in model methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"GPU out of memory in {func.__name__}: {e}"
                logging.error(error_msg)
                if error_callback:
                    error_callback(MemoryError(error_msg))
                raise MemoryError(error_msg) from e
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    error_msg = f"NaN/Inf detected in {func.__name__}: {e}"
                    logging.error(error_msg)
                    if error_callback:
                        error_callback(ComputationError(error_msg))
                    raise ComputationError(error_msg) from e
                else:
                    error_msg = f"Runtime error in {func.__name__}: {e}"
                    logging.error(error_msg)
                    if error_callback:
                        error_callback(ComputationError(error_msg))
                    raise ComputationError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error in {func.__name__}: {e}"
                logging.error(error_msg)
                if error_callback:
                    error_callback(ModelError(error_msg))
                raise ModelError(error_msg) from e
        return wrapper
    return decorator


class RobustNeuralOperator(nn.Module):
    """Base class with robust error handling and recovery mechanisms."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_count = 0
        self.max_errors = 10
        self.fallback_enabled = True
        
    @handle_model_errors()
    def validate_input(self, x: torch.Tensor, **kwargs) -> None:
        """Validate input tensor for common issues."""
        if not isinstance(x, torch.Tensor):
            raise InvalidInputError(f"Expected torch.Tensor, got {type(x)}")
        
        if torch.isnan(x).any():
            raise InvalidInputError("Input contains NaN values")
        
        if torch.isinf(x).any():
            raise InvalidInputError("Input contains infinite values")
        
        if x.numel() == 0:
            raise InvalidInputError("Input tensor is empty")
        
        # Check for reasonable value ranges
        if torch.abs(x).max() > 1e6:
            warnings.warn("Input contains very large values, may cause numerical instability")
        
        if torch.abs(x).min() < 1e-10 and torch.abs(x).max() > 1e-10:
            warnings.warn("Input has extreme dynamic range, consider normalization")
    
    @handle_model_errors()
    def validate_output(self, output: torch.Tensor) -> torch.Tensor:
        """Validate and sanitize output tensor."""
        if torch.isnan(output).any():
            self.logger.warning("Output contains NaN values, attempting correction")
            output = torch.nan_to_num(output, nan=0.0)
        
        if torch.isinf(output).any():
            self.logger.warning("Output contains infinite values, attempting correction")
            output = torch.nan_to_num(output, posinf=1e6, neginf=-1e6)
        
        return output
    
    @contextmanager
    def error_recovery_context(self):
        """Context manager for error recovery during forward pass."""
        try:
            yield
        except ComputationError as e:
            self.error_count += 1
            self.logger.warning(f"Computation error #{self.error_count}: {e}")
            
            if self.error_count > self.max_errors:
                self.logger.error("Too many errors, disabling error recovery")
                self.fallback_enabled = False
                raise
            
            # Attempt simple fallback
            if self.fallback_enabled:
                self.logger.info("Attempting fallback computation")
                # Subclasses should override this behavior
                raise
        except MemoryError as e:
            self.logger.error(f"Memory error: {e}")
            # Clear cache and try again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def safe_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Safe forward pass with comprehensive error handling."""
        # Input validation
        self.validate_input(x, **kwargs)
        
        with self.error_recovery_context():
            # Perform forward pass
            output = self.forward(x, **kwargs)
            
            # Output validation
            output = self.validate_output(output)
            
            return output
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'error_count': self.error_count,
            'max_errors': self.max_errors,
            'fallback_enabled': self.fallback_enabled
        }


class MemoryAwareForward:
    """Memory-aware forward pass with automatic batching for large inputs."""
    
    def __init__(self, model: nn.Module, max_batch_size: int = 32, chunk_size: Optional[int] = None):
        self.model = model
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Memory-aware forward pass."""
        batch_size = x.size(0)
        
        # If batch is small enough, process normally
        if batch_size <= self.max_batch_size:
            return self.model(x, **kwargs)
        
        # Split into chunks
        chunks = torch.split(x, self.max_batch_size, dim=0)
        outputs = []
        
        for i, chunk in enumerate(chunks):
            try:
                with torch.cuda.device(x.device) if torch.cuda.is_available() else torch.no_grad():
                    chunk_output = self.model(chunk, **kwargs)
                    outputs.append(chunk_output)
                    
                    # Clear intermediate memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"Memory error on chunk {i}, reducing batch size")
                    # Recursively reduce batch size
                    smaller_forward = MemoryAwareForward(
                        self.model, 
                        max_batch_size=max(1, self.max_batch_size // 2)
                    )
                    chunk_output = smaller_forward(chunk, **kwargs)
                    outputs.append(chunk_output)
                else:
                    raise
        
        return torch.cat(outputs, dim=0)


class GradientHealthChecker:
    """Monitor and fix gradient health during training."""
    
    def __init__(self, model: nn.Module, clip_norm: float = 1.0):
        self.model = model
        self.clip_norm = clip_norm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gradient_stats = []
    
    def check_gradients(self) -> Dict[str, float]:
        """Check gradient health and return statistics."""
        total_norm = 0.0
        num_params = 0
        nan_count = 0
        inf_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1
                
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    self.logger.warning(f"NaN gradient in {name}")
                    param.grad.data.zero_()  # Zero out NaN gradients
                
                if torch.isinf(param.grad).any():
                    inf_count += 1
                    self.logger.warning(f"Inf gradient in {name}")
                    param.grad.data = torch.clamp(param.grad.data, -1e6, 1e6)
        
        total_norm = total_norm ** 0.5
        
        # Clip gradients if necessary
        if total_norm > self.clip_norm:
            clip_coef = self.clip_norm / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        stats = {
            'total_norm': total_norm,
            'num_params': num_params,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'clipped': total_norm > self.clip_norm
        }
        
        self.gradient_stats.append(stats)
        
        return stats
    
    def get_gradient_history(self) -> List[Dict[str, float]]:
        """Get history of gradient statistics."""
        return self.gradient_stats


class NumericalStabilizer:
    """Utilities for numerical stabilization."""
    
    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable logarithm."""
        return torch.log(torch.clamp(x, min=eps))
    
    @staticmethod
    def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable square root."""
        return torch.sqrt(torch.clamp(x, min=eps))
    
    @staticmethod
    def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable division."""
        return a / (b + eps)
    
    @staticmethod
    def safe_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """Numerically stable softmax."""
        x = x / temperature
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_shifted = x - x_max
        return torch.softmax(x_shifted, dim=dim)
    
    @staticmethod
    def clamp_preserve_gradient(x: torch.Tensor, min_val: float = -1e6, max_val: float = 1e6) -> torch.Tensor:
        """Clamp values while preserving gradients."""
        return x + (torch.clamp(x, min_val, max_val) - x).detach()


def make_model_robust(model_class: Type[nn.Module]) -> Type[nn.Module]:
    """Class decorator to add robust error handling to any model."""
    
    class RobustModel(model_class, RobustNeuralOperator):
        def __init__(self, *args, **kwargs):
            model_class.__init__(self, *args, **kwargs)
            RobustNeuralOperator.__init__(self)
            
            # Wrap forward method
            original_forward = self.forward
            
            @handle_model_errors()
            def robust_forward(x, **forward_kwargs):
                return self.safe_forward_impl(x, original_forward, **forward_kwargs)
            
            self.forward = robust_forward
        
        def safe_forward_impl(self, x: torch.Tensor, original_forward: Callable, **kwargs) -> torch.Tensor:
            """Implementation of safe forward pass."""
            self.validate_input(x, **kwargs)
            
            with self.error_recovery_context():
                output = original_forward(x, **kwargs)
                output = self.validate_output(output)
                return output
    
    RobustModel.__name__ = f"Robust{model_class.__name__}"
    RobustModel.__qualname__ = f"Robust{model_class.__qualname__}"
    
    return RobustModel