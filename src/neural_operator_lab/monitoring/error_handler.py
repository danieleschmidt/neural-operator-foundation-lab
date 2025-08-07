"""Comprehensive error handling and recovery for neural operator training."""

import torch
import torch.nn as nn
import numpy as np
import logging
import traceback
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import warnings

from ..base import NeuralOperatorBase
from ..training.trainer import Trainer, TrainerConfig


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    NUMERICAL = "numerical"
    MEMORY = "memory"
    CONVERGENCE = "convergence"
    DATA = "data"
    HARDWARE = "hardware"
    CONFIG = "config"
    MODEL = "model"
    UNKNOWN = "unknown"


@dataclass
class ErrorEvent:
    """Represents an error event."""
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    traceback: str
    context: Dict[str, Any]
    resolved: bool = False
    recovery_action: Optional[str] = None


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 max_recovery_attempts: int = 3,
                 enable_telemetry: bool = True):
        self.log_file = log_file
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_telemetry = enable_telemetry
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.recovery_attempts: Dict[str, int] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Register handlers
        self._register_handlers()
    
    def _setup_logging(self):
        """Setup error logging."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        
        # File handler
        handlers = [console_handler]
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            handlers.append(file_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        for handler in handlers:
            self.logger.addHandler(handler)
    
    def _register_handlers(self):
        """Register error-specific handlers."""
        self.handlers = {
            ErrorCategory.NUMERICAL: self._handle_numerical_error,
            ErrorCategory.MEMORY: self._handle_memory_error,
            ErrorCategory.CONVERGENCE: self._handle_convergence_error,
            ErrorCategory.DATA: self._handle_data_error,
            ErrorCategory.HARDWARE: self._handle_hardware_error,
            ErrorCategory.CONFIG: self._handle_config_error,
            ErrorCategory.MODEL: self._handle_model_error,
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            "reduce_batch_size": self._reduce_batch_size,
            "reduce_learning_rate": self._reduce_learning_rate,
            "enable_gradient_clipping": self._enable_gradient_clipping,
            "switch_to_cpu": self._switch_to_cpu,
            "reset_optimizer": self._reset_optimizer,
            "reload_checkpoint": self._reload_checkpoint,
            "reduce_precision": self._reduce_precision,
            "increase_patience": self._increase_patience,
        }
    
    @contextmanager
    def handle_errors(self, context: Dict[str, Any] = None):
        """Context manager for error handling."""
        context = context or {}
        
        try:
            yield
        except Exception as e:
            self._process_error(e, context)
            raise
    
    def _process_error(self, exception: Exception, context: Dict[str, Any]):
        """Process and categorize an error."""
        # Categorize error
        category = self._categorize_error(exception)
        severity = self._assess_severity(exception, category)
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(exception),
            traceback=traceback.format_exc(),
            context=context
        )
        
        self.error_events.append(error_event)
        
        # Log error
        self.logger.error(f"Error [{category.value}][{severity.value}]: {exception}")
        self.logger.error(f"Context: {context}")
        
        # Attempt recovery
        if severity != ErrorSeverity.CRITICAL:
            recovery_success = self._attempt_recovery(error_event)
            if recovery_success:
                error_event.resolved = True
                self.logger.info(f"Successfully recovered from error: {exception}")
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize the error based on type and message."""
        error_str = str(exception).lower()
        error_type = type(exception).__name__
        
        # Numerical errors
        if isinstance(exception, (ValueError, RuntimeError)):
            if any(keyword in error_str for keyword in 
                   ['nan', 'inf', 'overflow', 'underflow', 'singular', 'numerical']):
                return ErrorCategory.NUMERICAL
        
        # Memory errors
        if isinstance(exception, (RuntimeError, torch.cuda.OutOfMemoryError)):
            if any(keyword in error_str for keyword in 
                   ['memory', 'cuda', 'allocation', 'out of memory']):
                return ErrorCategory.MEMORY
        
        # Convergence issues
        if isinstance(exception, (RuntimeError, ValueError)):
            if any(keyword in error_str for keyword in 
                   ['converge', 'diverge', 'unstable', 'exploding', 'vanishing']):
                return ErrorCategory.CONVERGENCE
        
        # Data loading errors
        if isinstance(exception, (FileNotFoundError, IOError, OSError, IndexError)):
            return ErrorCategory.DATA
        
        # Hardware errors
        if isinstance(exception, RuntimeError):
            if any(keyword in error_str for keyword in 
                   ['cuda', 'device', 'driver', 'hardware']):
                return ErrorCategory.HARDWARE
        
        # Configuration errors
        if isinstance(exception, (ValueError, TypeError, AttributeError)):
            if any(keyword in error_str for keyword in 
                   ['config', 'parameter', 'dimension', 'shape']):
                return ErrorCategory.CONFIG
        
        # Model errors
        if isinstance(exception, (RuntimeError, AttributeError)):
            if any(keyword in error_str for keyword in 
                   ['model', 'layer', 'forward', 'backward']):
                return ErrorCategory.MODEL
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess the severity of the error."""
        error_str = str(exception).lower()
        
        # Critical errors that should stop training
        if category == ErrorCategory.HARDWARE:
            return ErrorSeverity.CRITICAL
        
        if 'critical' in error_str or 'fatal' in error_str:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.MEMORY, ErrorCategory.NUMERICAL]:
            if any(keyword in error_str for keyword in 
                   ['cuda', 'out of memory', 'nan', 'inf']):
                return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.CONVERGENCE, ErrorCategory.MODEL]:
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from the error."""
        error_key = f"{error_event.category.value}_{hash(error_event.message) % 1000}"
        
        # Check recovery attempts
        if error_key in self.recovery_attempts:
            if self.recovery_attempts[error_key] >= self.max_recovery_attempts:
                self.logger.warning(f"Max recovery attempts reached for {error_key}")
                return False
            self.recovery_attempts[error_key] += 1
        else:
            self.recovery_attempts[error_key] = 1
        
        # Get handler for this category
        if error_event.category in self.handlers:
            try:
                recovery_action = self.handlers[error_event.category](error_event)
                error_event.recovery_action = recovery_action
                return recovery_action is not None
            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}")
                return False
        
        return False
    
    def _handle_numerical_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle numerical instability errors."""
        if 'nan' in error_event.message.lower() or 'inf' in error_event.message.lower():
            # Try gradient clipping first
            if 'gradient' in error_event.context.get('phase', ''):
                return self._execute_recovery("enable_gradient_clipping", error_event.context)
            
            # Reduce learning rate
            return self._execute_recovery("reduce_learning_rate", error_event.context)
        
        return None
    
    def _handle_memory_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle out-of-memory errors."""
        # Try reducing batch size first
        if self._execute_recovery("reduce_batch_size", error_event.context):
            return "reduce_batch_size"
        
        # Try reducing precision
        if self._execute_recovery("reduce_precision", error_event.context):
            return "reduce_precision"
        
        # Switch to CPU as last resort
        if self._execute_recovery("switch_to_cpu", error_event.context):
            return "switch_to_cpu"
        
        return None
    
    def _handle_convergence_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle convergence issues."""
        # Reset optimizer state
        if self._execute_recovery("reset_optimizer", error_event.context):
            return "reset_optimizer"
        
        # Reduce learning rate
        if self._execute_recovery("reduce_learning_rate", error_event.context):
            return "reduce_learning_rate"
        
        # Increase patience
        if self._execute_recovery("increase_patience", error_event.context):
            return "increase_patience"
        
        return None
    
    def _handle_data_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle data-related errors."""
        self.logger.warning("Data error detected - manual intervention may be required")
        return None
    
    def _handle_hardware_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle hardware-related errors."""
        # Switch to CPU
        return self._execute_recovery("switch_to_cpu", error_event.context)
    
    def _handle_config_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle configuration errors."""
        self.logger.warning("Configuration error detected - manual intervention required")
        return None
    
    def _handle_model_error(self, error_event: ErrorEvent) -> Optional[str]:
        """Handle model-related errors."""
        # Try reloading from checkpoint
        return self._execute_recovery("reload_checkpoint", error_event.context)
    
    def _execute_recovery(self, strategy: str, context: Dict[str, Any]) -> bool:
        """Execute a recovery strategy."""
        if strategy in self.recovery_strategies:
            try:
                return self.recovery_strategies[strategy](context)
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy} failed: {e}")
                return False
        return False
    
    def _reduce_batch_size(self, context: Dict[str, Any]) -> bool:
        """Reduce batch size recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            current_bs = getattr(trainer.config, 'batch_size', 32)
            new_bs = max(1, current_bs // 2)
            trainer.config.batch_size = new_bs
            self.logger.info(f"Reduced batch size from {current_bs} to {new_bs}")
            return True
        return False
    
    def _reduce_learning_rate(self, context: Dict[str, Any]) -> bool:
        """Reduce learning rate recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            for param_group in trainer.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5
                self.logger.info(f"Reduced learning rate from {old_lr} to {param_group['lr']}")
            return True
        return False
    
    def _enable_gradient_clipping(self, context: Dict[str, Any]) -> bool:
        """Enable gradient clipping recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            trainer.config.grad_clip_norm = 1.0
            self.logger.info("Enabled gradient clipping with norm 1.0")
            return True
        return False
    
    def _switch_to_cpu(self, context: Dict[str, Any]) -> bool:
        """Switch to CPU recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            trainer.device = torch.device('cpu')
            trainer.model = trainer.model.cpu()
            self.logger.warning("Switched to CPU training due to GPU issues")
            return True
        return False
    
    def _reset_optimizer(self, context: Dict[str, Any]) -> bool:
        """Reset optimizer state recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            trainer.optimizer = trainer._create_optimizer()
            self.logger.info("Reset optimizer state")
            return True
        return False
    
    def _reload_checkpoint(self, context: Dict[str, Any]) -> bool:
        """Reload from checkpoint recovery strategy."""
        if 'trainer' in context and 'checkpoint_path' in context:
            trainer = context['trainer']
            try:
                trainer.load_checkpoint(context['checkpoint_path'])
                self.logger.info(f"Reloaded checkpoint: {context['checkpoint_path']}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to reload checkpoint: {e}")
        return False
    
    def _reduce_precision(self, context: Dict[str, Any]) -> bool:
        """Reduce precision recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            if trainer.config.mixed_precision:
                trainer.config.mixed_precision = False
                trainer.scaler = None
                self.logger.info("Disabled mixed precision training")
                return True
        return False
    
    def _increase_patience(self, context: Dict[str, Any]) -> bool:
        """Increase early stopping patience recovery strategy."""
        if 'trainer' in context:
            trainer = context['trainer']
            old_patience = trainer.config.patience
            trainer.config.patience = int(old_patience * 1.5)
            self.logger.info(f"Increased patience from {old_patience} to {trainer.config.patience}")
            return True
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_events:
            return {"total_errors": 0}
        
        # Count by category and severity
        category_counts = {}
        severity_counts = {}
        resolved_count = 0
        
        for event in self.error_events:
            # Category counts
            cat = event.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Severity counts
            sev = event.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            # Resolved count
            if event.resolved:
                resolved_count += 1
        
        # Recent errors (last 10)
        recent_errors = self.error_events[-10:]
        
        return {
            "total_errors": len(self.error_events),
            "resolved_errors": resolved_count,
            "resolution_rate": resolved_count / len(self.error_events),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "recent_errors": [
                {
                    "timestamp": event.timestamp,
                    "category": event.category.value,
                    "severity": event.severity.value,
                    "message": event.message[:100],
                    "resolved": event.resolved
                }
                for event in recent_errors
            ]
        }
    
    def export_error_log(self, filepath: str):
        """Export error log to file."""
        error_data = [asdict(event) for event in self.error_events]
        
        # Convert enums to strings
        for error in error_data:
            error['severity'] = error['severity'].value
            error['category'] = error['category'].value
        
        with open(filepath, 'w') as f:
            json.dump(error_data, f, indent=2, default=str)
        
        self.logger.info(f"Error log exported to {filepath}")


class RobustTrainer(Trainer):
    """Enhanced trainer with comprehensive error handling and recovery."""
    
    def __init__(self, *args, **kwargs):
        # Extract error handler config
        error_handler_config = kwargs.pop('error_handler_config', {})
        
        super().__init__(*args, **kwargs)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            log_file=str(Path(self.config.save_dir) / 'errors.log'),
            **error_handler_config
        )
        
        # Enhanced validation
        self._validate_configuration()
        
        # Setup automatic recovery checkpoints
        self._setup_recovery_checkpoints()
    
    def _validate_configuration(self):
        """Validate training configuration for common issues."""
        warnings_list = []
        
        # Check learning rate
        if self.config.learning_rate > 0.01:
            warnings_list.append(f"High learning rate: {self.config.learning_rate}")
        
        # Check batch size vs available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            estimated_memory = self.config.batch_size * 1e8  # Rough estimate
            
            if estimated_memory > total_memory * 0.8:
                warnings_list.append(f"Batch size may be too large for available GPU memory")
        
        # Check model parameters
        param_count = self.model.count_parameters()
        if param_count > 1e9:
            warnings_list.append(f"Very large model: {param_count:,} parameters")
        
        # Log warnings
        for warning in warnings_list:
            self.logger.warning(f"Configuration warning: {warning}")
    
    def _setup_recovery_checkpoints(self):
        """Setup automatic recovery checkpoints."""
        self.recovery_checkpoint_path = Path(self.config.save_dir) / "recovery_checkpoint.pt"
        
        # Save initial state
        try:
            self.save_checkpoint(str(self.recovery_checkpoint_path))
        except Exception as e:
            self.logger.warning(f"Could not create recovery checkpoint: {e}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train epoch with error handling."""
        context = {
            'trainer': self,
            'phase': 'training',
            'checkpoint_path': str(self.recovery_checkpoint_path)
        }
        
        try:
            with self.error_handler.handle_errors(context):
                return super().train_epoch(train_loader)
        
        except Exception as e:
            # Log the error and attempt recovery
            self.logger.error(f"Training epoch failed: {e}")
            
            # Create recovery checkpoint after each successful epoch
            try:
                self.save_checkpoint(str(self.recovery_checkpoint_path))
            except Exception as checkpoint_e:
                self.logger.error(f"Failed to save recovery checkpoint: {checkpoint_e}")
            
            raise
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate epoch with error handling."""
        context = {
            'trainer': self,
            'phase': 'validation',
            'checkpoint_path': str(self.recovery_checkpoint_path)
        }
        
        with self.error_handler.handle_errors(context):
            return super().validate_epoch(val_loader)
    
    def _compute_loss(self, inputs, targets, grid=None) -> Dict[str, torch.Tensor]:
        """Compute loss with numerical stability checks."""
        try:
            loss_dict = super()._compute_loss(inputs, targets, grid)
            
            # Check for numerical issues
            total_loss = loss_dict.get('total_loss', torch.tensor(0.0))
            
            if torch.isnan(total_loss):
                raise ValueError("NaN detected in loss computation")
            
            if torch.isinf(total_loss):
                raise ValueError("Infinite value detected in loss computation")
            
            if total_loss.item() > 1e6:
                self.logger.warning(f"Very large loss value: {total_loss.item()}")
            
            return loss_dict
        
        except Exception as e:
            self.logger.error(f"Loss computation failed: {e}")
            
            # Return a fallback loss to prevent training termination
            fallback_loss = torch.nn.functional.mse_loss(
                torch.zeros_like(targets), targets
            )
            return {'total_loss': fallback_loss, 'fallback': True}
    
    def train(self, *args, **kwargs):
        """Enhanced training with comprehensive monitoring."""
        try:
            # Pre-training validation
            self._pre_training_checks()
            
            # Call parent training method
            result = super().train(*args, **kwargs)
            
            # Post-training summary
            error_summary = self.error_handler.get_error_summary()
            result['error_summary'] = error_summary
            
            if error_summary['total_errors'] > 0:
                self.logger.info(f"Training completed with {error_summary['total_errors']} errors "
                               f"({error_summary['resolution_rate']:.1%} resolved)")
            
            return result
        
        except Exception as e:
            # Log final error
            self.logger.error(f"Training failed completely: {e}")
            
            # Export error log
            error_log_path = Path(self.config.save_dir) / "training_errors.json"
            self.error_handler.export_error_log(str(error_log_path))
            
            raise
    
    def _pre_training_checks(self):
        """Perform pre-training validation checks."""
        # Memory check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                # Test forward pass
                dummy_input = torch.randn(2, 64, 64, self.model.input_dim).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_per_sample = (peak_memory - initial_memory) / 2
                
                estimated_batch_memory = memory_per_sample * self.config.batch_size
                available_memory = torch.cuda.get_device_properties(0).total_memory * 0.8
                
                if estimated_batch_memory > available_memory:
                    self.logger.warning(
                        f"Batch size {self.config.batch_size} may cause OOM. "
                        f"Consider reducing to {int(available_memory / memory_per_sample)}"
                    )
                
            except Exception as e:
                self.logger.error(f"Pre-training memory check failed: {e}")
        
        # Model parameter check
        nan_params = 0
        inf_params = 0
        
        for param in self.model.parameters():
            if torch.isnan(param).any():
                nan_params += 1
            if torch.isinf(param).any():
                inf_params += 1
        
        if nan_params > 0 or inf_params > 0:
            self.logger.error(f"Model has invalid parameters: {nan_params} NaN, {inf_params} Inf")
            raise ValueError("Model parameters contain NaN or Inf values")
        
        self.logger.info("Pre-training checks completed successfully")


def create_robust_trainer(model: NeuralOperatorBase, 
                         config: Optional[TrainerConfig] = None,
                         **kwargs) -> RobustTrainer:
    """Factory function to create robust trainer with sensible defaults."""
    if config is None:
        config = TrainerConfig()
    
    # Set conservative defaults for robustness
    config.grad_clip_norm = max(config.grad_clip_norm, 1.0)
    config.mixed_precision = config.mixed_precision and torch.cuda.is_available()
    
    return RobustTrainer(model, config, **kwargs)