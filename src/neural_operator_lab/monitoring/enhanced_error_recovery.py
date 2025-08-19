"""Enhanced error recovery and resilience for neural operator training."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import logging
import time
import traceback
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Error event information."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_str: str = ""
    recovery_action: Optional[str] = None
    recovery_successful: bool = False


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        self.attempt_count = 0
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from the error."""
        self.attempt_count += 1
        if self.attempt_count > self.max_attempts:
            return False
        return self._attempt_recovery(error, context)
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Actual recovery implementation."""
        raise NotImplementedError
    
    def reset(self):
        """Reset strategy state."""
        self.attempt_count = 0


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for out-of-memory errors."""
    
    def __init__(self):
        super().__init__("memory_recovery", max_attempts=3)
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            'out of memory', 'cuda out of memory', 'memory error'
        ])
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt memory recovery."""
        try:
            logger.info(f"Attempting memory recovery (attempt {self.attempt_count}/{self.max_attempts})")
            
            # Clear Python garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reduce batch size if possible
            if 'batch_size' in context and context['batch_size'] > 1:
                new_batch_size = max(1, context['batch_size'] // 2)
                context['batch_size'] = new_batch_size
                logger.info(f"Reduced batch size to {new_batch_size}")
            
            # Enable gradient checkpointing if model supports it
            if 'model' in context:
                model = context['model']
                if hasattr(model, 'gradient_checkpointing') and not model.gradient_checkpointing:
                    model.gradient_checkpointing = True
                    logger.info("Enabled gradient checkpointing")
            
            time.sleep(1)  # Brief pause to allow memory cleanup
            return True
            
        except Exception as recovery_error:
            logger.error(f"Memory recovery failed: {recovery_error}")
            return False


class NumericalInstabilityRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for numerical instability."""
    
    def __init__(self):
        super().__init__("numerical_recovery", max_attempts=2)
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            'nan', 'inf', 'numerical', 'overflow', 'underflow'
        ])
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt numerical stability recovery."""
        try:
            logger.info(f"Attempting numerical recovery (attempt {self.attempt_count}/{self.max_attempts})")
            
            # Reduce learning rate
            if 'optimizer' in context:
                optimizer = context['optimizer']
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * 0.1
                    logger.info(f"Reduced learning rate from {old_lr} to {param_group['lr']}")
            
            # Add gradient clipping
            if 'model' in context:
                model = context['model']
                context['grad_clip_norm'] = 0.5
                logger.info("Added gradient clipping with norm 0.5")
            
            # Reset model parameters if they contain NaN/Inf
            if 'model' in context:
                model = context['model']
                for name, param in model.named_parameters():
                    if param is not None and (torch.isnan(param).any() or torch.isinf(param).any()):
                        logger.warning(f"Resetting parameter {name} due to NaN/Inf values")
                        torch.nn.init.xavier_uniform_(param.data)
            
            return True
            
        except Exception as recovery_error:
            logger.error(f"Numerical recovery failed: {recovery_error}")
            return False


class ModelCheckpointRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy using model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        super().__init__("checkpoint_recovery", max_attempts=1)
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Can always attempt checkpoint recovery
        return 'model' in context and self.checkpoint_dir.exists()
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery from checkpoint."""
        try:
            logger.info("Attempting recovery from checkpoint")
            
            # Find most recent checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
            if not checkpoint_files:
                logger.warning("No checkpoints found")
                return False
            
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            
            if 'model' in context:
                model = context['model']
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Model state restored from checkpoint")
            
            if 'optimizer' in context and 'optimizer_state_dict' in checkpoint:
                optimizer = context['optimizer']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state restored from checkpoint")
            
            # Update context with checkpoint info
            context['recovered_epoch'] = checkpoint.get('epoch', 0)
            context['recovered_loss'] = checkpoint.get('loss', float('inf'))
            
            return True
            
        except Exception as recovery_error:
            logger.error(f"Checkpoint recovery failed: {recovery_error}")
            return False


class ErrorRecoveryManager:
    """Manages error recovery strategies and error history."""
    
    def __init__(self, 
                 strategies: Optional[List[RecoveryStrategy]] = None,
                 max_error_history: int = 1000,
                 auto_save_checkpoints: bool = True):
        """Initialize error recovery manager.
        
        Args:
            strategies: List of recovery strategies
            max_error_history: Maximum number of errors to keep in history
            auto_save_checkpoints: Whether to automatically save checkpoints
        """
        self.strategies = strategies or self._get_default_strategies()
        self.error_history: List[ErrorEvent] = []
        self.max_error_history = max_error_history
        self.auto_save_checkpoints = auto_save_checkpoints
        
        # Statistics
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_by_strategy': {},
            'error_by_type': {}
        }
        
        # Thread-safe error queue for async handling
        self.error_queue = queue.Queue()
        self._start_error_handler_thread()
    
    def _get_default_strategies(self) -> List[RecoveryStrategy]:
        """Get default recovery strategies."""
        return [
            MemoryRecoveryStrategy(),
            NumericalInstabilityRecoveryStrategy(),
            ModelCheckpointRecoveryStrategy()
        ]
    
    def _start_error_handler_thread(self):
        """Start background thread for error handling."""
        def error_handler():
            while True:
                try:
                    error_event = self.error_queue.get(timeout=1.0)
                    if error_event is None:  # Shutdown signal
                        break
                    self._process_error_async(error_event)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in background error handler: {e}")
        
        self.error_thread = threading.Thread(target=error_handler, daemon=True)
        self.error_thread.start()
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any],
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> bool:
        """Handle error with recovery attempt.
        
        Args:
            error: Exception that occurred
            context: Context information (model, optimizer, etc.)
            severity: Error severity level
            
        Returns:
            True if recovery was successful, False otherwise
        """
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            context=self._sanitize_context(context),
            traceback_str=traceback.format_exc()
        )
        
        # Update statistics
        self.recovery_stats['total_errors'] += 1
        error_type = error_event.error_type
        self.recovery_stats['error_by_type'][error_type] = \
            self.recovery_stats['error_by_type'].get(error_type, 0) + 1
        
        # Log error
        logger.error(f"Error occurred: {error_event.error_type}: {error_event.error_message}")
        
        # Attempt recovery
        recovery_successful = False
        
        for strategy in self.strategies:
            if strategy.can_recover(error, context):
                logger.info(f"Attempting recovery with strategy: {strategy.name}")
                
                try:
                    if strategy.recover(error, context):
                        recovery_successful = True
                        error_event.recovery_action = strategy.name
                        error_event.recovery_successful = True
                        
                        # Update statistics
                        self.recovery_stats['successful_recoveries'] += 1
                        self.recovery_stats['recovery_by_strategy'][strategy.name] = \
                            self.recovery_stats['recovery_by_strategy'].get(strategy.name, 0) + 1
                        
                        logger.info(f"Recovery successful with strategy: {strategy.name}")
                        break
                        
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    continue
        
        if not recovery_successful:
            self.recovery_stats['failed_recoveries'] += 1
            logger.error("All recovery strategies failed")
        
        # Add to history
        self._add_to_history(error_event)
        
        # Auto-save checkpoint if enabled and recovery was successful
        if recovery_successful and self.auto_save_checkpoints:
            self._save_emergency_checkpoint(context)
        
        # Queue for async processing
        self.error_queue.put(error_event)
        
        return recovery_successful
    
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context for logging (remove large objects)."""
        sanitized = {}
        for key, value in context.items():
            if key in ['model', 'optimizer']:
                # Just store type info, not the actual objects
                sanitized[key + '_type'] = type(value).__name__
            elif isinstance(value, torch.Tensor):
                sanitized[key + '_shape'] = list(value.shape)
                sanitized[key + '_dtype'] = str(value.dtype)
            elif isinstance(value, (int, float, str, bool, list, dict)):
                sanitized[key] = value
            else:
                sanitized[key + '_type'] = type(value).__name__
        return sanitized
    
    def _add_to_history(self, error_event: ErrorEvent):
        """Add error event to history."""
        self.error_history.append(error_event)
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _save_emergency_checkpoint(self, context: Dict[str, Any]):
        """Save emergency checkpoint after successful recovery."""
        try:
            if 'model' in context:
                checkpoint_dir = Path("./emergency_checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time())
                checkpoint_path = checkpoint_dir / f"emergency_checkpoint_{timestamp}.pt"
                
                checkpoint = {
                    'model_state_dict': context['model'].state_dict(),
                    'timestamp': timestamp,
                    'recovery_event': True
                }
                
                if 'optimizer' in context:
                    checkpoint['optimizer_state_dict'] = context['optimizer'].state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def _process_error_async(self, error_event: ErrorEvent):
        """Process error event asynchronously."""
        # This could be extended to send notifications, update dashboards, etc.
        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error detected: {error_event.error_message}")
            # Could send alerts, shutdown training, etc.
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        stats = self.recovery_stats.copy()
        stats.update({
            'total_errors_in_history': len(self.error_history),
            'recent_errors_count': len(recent_errors),
            'recovery_success_rate': (
                self.recovery_stats['successful_recoveries'] / 
                max(1, self.recovery_stats['total_errors'])
            ),
            'recent_error_types': {
                event.error_type: sum(1 for e in recent_errors if e.error_type == event.error_type)
                for event in recent_errors
            }
        })
        
        return stats
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorEvent]:
        """Get recent error events."""
        cutoff_time = time.time() - (hours * 3600)
        return [e for e in self.error_history if e.timestamp >= cutoff_time]
    
    def reset_strategies(self):
        """Reset all recovery strategies."""
        for strategy in self.strategies:
            strategy.reset()
    
    def shutdown(self):
        """Shutdown error recovery manager."""
        # Signal error handler thread to stop
        self.error_queue.put(None)
        if hasattr(self, 'error_thread'):
            self.error_thread.join(timeout=5.0)


@contextmanager
def robust_training_context(model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          error_manager: Optional[ErrorRecoveryManager] = None,
                          max_retries: int = 3):
    """Context manager for robust training with automatic error recovery."""
    
    if error_manager is None:
        error_manager = ErrorRecoveryManager()
    
    context = {
        'model': model,
        'optimizer': optimizer,
        'batch_size': getattr(optimizer, 'batch_size', 32)  # Default batch size
    }
    
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            yield context
            break  # Success, exit retry loop
            
        except Exception as e:
            retry_count += 1
            
            if retry_count > max_retries:
                logger.error(f"Maximum retries ({max_retries}) exceeded")
                raise
            
            logger.warning(f"Training error (attempt {retry_count}/{max_retries + 1}): {e}")
            
            # Attempt recovery
            recovery_successful = error_manager.handle_error(e, context)
            
            if not recovery_successful:
                logger.error(f"Recovery failed for attempt {retry_count}")
                if retry_count == max_retries:
                    raise
            else:
                logger.info(f"Recovery successful, retrying (attempt {retry_count + 1})")


# Global error recovery manager
_global_error_manager = None

def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorRecoveryManager()
    return _global_error_manager