# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Advanced Error Handling and Recovery System

Comprehensive error handling with automatic recovery, detailed logging,
and graceful degradation strategies for neural operator training.
"""

import torch
import torch.nn as nn
import traceback
import logging
import time
import sys
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import threading
from contextlib import contextmanager
from functools import wraps
from collections import deque
import warnings

from neural_operator_lab.base import NeuralOperatorBase


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    timestamp: float
    function_name: str
    module_name: str
    traceback_str: str
    model_state: Optional[Dict[str, Any]] = None
    input_shapes: Optional[Dict[str, Tuple]] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'traceback_str': self.traceback_str,
            'model_state': self.model_state,
            'input_shapes': self.input_shapes,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        self.attempt_count = 0
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the error."""
        return True
    
    def recover(self, error_context: ErrorContext, *args, **kwargs) -> Tuple[bool, Any]:
        """Attempt to recover from the error."""
        self.attempt_count += 1
        if self.attempt_count > self.max_attempts:
            return False, None
        
        return self._do_recovery(error_context, *args, **kwargs)
    
    def _do_recovery(self, error_context: ErrorContext, *args, **kwargs) -> Tuple[bool, Any]:
        """Implementation-specific recovery logic."""
        return False, None
    
    def reset(self):
        """Reset attempt counter."""
        self.attempt_count = 0


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for CUDA out-of-memory errors."""
    
    def __init__(self):
        super().__init__("memory_recovery", max_attempts=3)
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return "out of memory" in error_context.error_message.lower() or \
               "cuda" in error_context.error_message.lower()
    
    def _do_recovery(self, error_context: ErrorContext, *args, **kwargs) -> Tuple[bool, Any]:
        """Attempt memory recovery."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reduce batch size if possible
            if 'reduce_batch_size' in kwargs and kwargs['reduce_batch_size']:
                original_batch_size = kwargs.get('original_batch_size', 32)
                new_batch_size = max(1, original_batch_size // 2)
                
                return True, {'new_batch_size': new_batch_size}
            
            return True, None
            
        except Exception as e:
            return False, str(e)


class GradientRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for gradient-related errors."""
    
    def __init__(self):
        super().__init__("gradient_recovery", max_attempts=2)
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        gradient_errors = [
            "gradient", "backward", "autograd", "nan", "inf"
        ]
        return any(err in error_context.error_message.lower() for err in gradient_errors)
    
    def _do_recovery(self, error_context: ErrorContext, *args, **kwargs) -> Tuple[bool, Any]:
        """Attempt gradient recovery."""
        try:
            recovery_actions = {}
            
            # Gradient clipping
            model = kwargs.get('model')
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                recovery_actions['gradient_clipped'] = True
            
            # Reduce learning rate
            optimizer = kwargs.get('optimizer')
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                recovery_actions['learning_rate_reduced'] = True
            
            return True, recovery_actions
            
        except Exception as e:
            return False, str(e)


class ModelRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for model-related errors."""
    
    def __init__(self):
        super().__init__("model_recovery", max_attempts=2)
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        model_errors = [
            "size mismatch", "shape", "dimension", "runtime error"
        ]
        return any(err in error_context.error_message.lower() for err in model_errors)
    
    def _do_recovery(self, error_context: ErrorContext, *args, **kwargs) -> Tuple[bool, Any]:
        """Attempt model recovery."""
        try:
            model = kwargs.get('model')
            if model is None:
                return False, "No model provided for recovery"
            
            # Reset model to eval mode and back
            was_training = model.training
            # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
            if was_training:
                model.train()
            
            # Clear any cached computations
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            
            return True, {'model_reset': True}
            
        except Exception as e:
            return False, str(e)


class AdvancedErrorHandler:
    """Advanced error handler with recovery strategies and detailed logging."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_error_history: int = 1000,
        enable_recovery: bool = True
    ):
        self.max_error_history = max_error_history
        self.enable_recovery = enable_recovery
        self.error_history: deque = deque(maxlen=max_error_history)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = [
            MemoryRecoveryStrategy(),
            GradientRecoveryStrategy(), 
            ModelRecoveryStrategy()
        ]
        
        # Error statistics
        self.error_stats: Dict[str, int] = {}
        self.recovery_stats: Dict[str, int] = {}
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def create_error_context(
        self,
        exception: Exception,
        function_name: str,
        module_name: str,
        **kwargs
    ) -> ErrorContext:
        """Create comprehensive error context."""
        
        # Get memory usage
        memory_usage = None
        gpu_usage = None
        
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
        except ImportError:
            pass
        
        try:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        except:
            pass
        
        # Extract input shapes if provided
        input_shapes = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                input_shapes[key] = tuple(value.shape)
            elif hasattr(value, 'shape'):
                input_shapes[key] = tuple(value.shape)
        
        return ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            timestamp=time.time(),
            function_name=function_name,
            module_name=module_name,
            traceback_str=traceback.format_exc(),
            input_shapes=input_shapes if input_shapes else None,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage
        )
    
    def handle_error(
        self,
        exception: Exception,
        function_name: str,
        module_name: str,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Handle error with recovery attempts."""
        
        with self._lock:
            # Create error context
            error_context = self.create_error_context(
                exception, function_name, module_name, **kwargs
            )
            
            # Update statistics
            error_type = error_context.error_type
            self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
            
            # Log error
            self.logger.error(
                f"Error in {module_name}.{function_name}: {error_context.error_message}"
            )
            self.logger.debug(f"Error context: {error_context.to_dict()}")
            
            # Add to history
            self.error_history.append(error_context)
            
            # Attempt recovery if enabled
            if self.enable_recovery:
                return self._attempt_recovery(error_context, **kwargs)
            
            return False, None
    
    def _attempt_recovery(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, Any]:
        """Attempt recovery using available strategies."""
        
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error_context):
                self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
                
                try:
                    success, result = strategy.recover(error_context, **kwargs)
                    
                    # Update recovery statistics
                    strategy_name = strategy.name
                    self.recovery_stats[strategy_name] = self.recovery_stats.get(strategy_name, 0) + 1
                    
                    if success:
                        error_context.recovery_attempted = True
                        error_context.recovery_successful = True
                        
                        self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                        return True, result
                    
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    continue
        
        # No recovery was successful
        error_context.recovery_attempted = True
        error_context.recovery_successful = False
        
        return False, None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            'total_errors': len(self.error_history),
            'error_types': self.error_stats.copy(),
            'recovery_attempts': self.recovery_stats.copy(),
            'recent_errors': [
                ctx.to_dict() for ctx in list(self.error_history)[-10:]
            ]
        }
    
    def export_error_log(self, filepath: str):
        """Export error history to file."""
        error_data = {
            'error_history': [ctx.to_dict() for ctx in self.error_history],
            'statistics': self.get_error_statistics(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        self.logger.info(f"Error log exported to: {filepath}")
    
    def clear_history(self):
        """Clear error history."""
        with self._lock:
            self.error_history.clear()
            self.error_stats.clear()
            self.recovery_stats.clear()


def robust_function(
    error_handler: Optional[AdvancedErrorHandler] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """Decorator to make functions robust with error handling and retries."""
    
    if error_handler is None:
        error_handler = AdvancedErrorHandler()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Handle error
                    success, recovery_data = error_handler.handle_error(
                        e, func.__name__, func.__module__, **kwargs
                    )
                    
                    if success and recovery_data:
                        # Apply recovery data to kwargs if possible
                        if isinstance(recovery_data, dict):
                            kwargs.update(recovery_data)
                    
                    if attempt < max_retries:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed
                        break
            
            # If we reach here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def error_monitoring_context(error_handler: AdvancedErrorHandler, **context_kwargs):
    """Context manager for error monitoring."""
    try:
        yield
    except Exception as e:
        # Get caller information
        frame = sys._getframe(1)
        function_name = frame.f_code.co_name
        module_name = frame.f_globals.get('__name__', 'unknown')
        
        # Handle the error
        success, recovery_data = error_handler.handle_error(
            e, function_name, module_name, **context_kwargs
        )
        
        if not success:
            raise


class RobustTrainer:
    """Training wrapper with advanced error handling."""
    
    def __init__(
        self,
        model: NeuralOperatorBase,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        error_handler: Optional[AdvancedErrorHandler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.error_handler = error_handler or AdvancedErrorHandler()
        
        # Training state for recovery
        self.training_state = {
            'epoch': 0,
            'best_loss': float('inf'),
            'model_checkpoint': None
        }
    
    @robust_function()
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Robust training step with error handling."""
        
        with error_monitoring_context(
            self.error_handler,
            model=self.model,
            optimizer=self.optimizer,
            inputs=inputs,
            targets=targets
        ):
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (defensive)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return {
                'loss': loss.item(),
                'grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf')).item()
            }
    
    def save_training_checkpoint(self, filepath: str):
        """Save training state for recovery."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.training_state,
            'error_statistics': self.error_handler.get_error_statistics()
        }
        
        torch.save(checkpoint, filepath)
    
    def load_training_checkpoint(self, filepath: str):
        """Load training state for recovery."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_state = checkpoint['training_state']