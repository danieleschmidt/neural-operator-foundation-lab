"""Training callbacks for neural operators."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer, history: Dict[str, Any]):
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: List[TrainingCallback]):
        self.callbacks = callbacks or []
    
    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_train_end(trainer, history)
    
    def on_epoch_begin(self, trainer, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, train_losses, val_losses)
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, losses)


class EarlyStopping(TrainingCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda x, y: x < y - self.min_delta
        else:
            self.best_value = -float('inf')
            self.monitor_op = lambda x, y: x > y + self.min_delta
    
    def on_train_begin(self, trainer):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = -float('inf')
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        if self.stopped_epoch > 0:
            trainer.logger.info(f"Early stopping at epoch {self.stopped_epoch}")
        
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            trainer.logger.info("Restored best model weights")
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        # Get current value
        current_value = val_losses.get(self.monitor.replace('val_', ''), 
                                      train_losses.get(self.monitor.replace('train_', ''), 0))
        
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        pass


class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints during training."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 period: int = 1):
        """Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' 
            save_best_only: Only save when monitored metric improves
            save_weights_only: Only save model weights
            period: Save every N epochs
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        
        self.epochs_since_last_save = 0
        
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_value = -float('inf')
            self.monitor_op = lambda x, y: x > y
    
    def on_train_begin(self, trainer):
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = -float('inf')
        
        self.epochs_since_last_save = 0
        
        # Create directory if needed
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            # Get current value
            current_value = val_losses.get(self.monitor.replace('val_', ''), 
                                          train_losses.get(self.monitor.replace('train_', ''), 0))
            
            should_save = True
            if self.save_best_only:
                should_save = self.monitor_op(current_value, self.best_value)
                if should_save:
                    self.best_value = current_value
            
            if should_save:
                filepath = self.filepath.format(epoch=epoch, **train_losses, **val_losses)
                
                if self.save_weights_only:
                    torch.save(trainer.model.state_dict(), filepath)
                else:
                    trainer.save_checkpoint(filepath)
                
                trainer.logger.info(f"Checkpoint saved: {filepath}")
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        pass


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduling callback."""
    
    def __init__(self, scheduler, monitor: str = 'val_loss'):
        """Initialize LR scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Get monitored metric
            metric = val_losses.get(self.monitor.replace('val_', ''), 
                                   train_losses.get(self.monitor.replace('train_', ''), 0))
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        pass


class ProgressLogger(TrainingCallback):
    """Log training progress."""
    
    def __init__(self, log_frequency: int = 10, log_level: int = logging.INFO):
        """Initialize progress logger.
        
        Args:
            log_frequency: Log every N batches
            log_level: Logging level
        """
        self.log_frequency = log_frequency
        self.log_level = log_level
        self.batch_times = []
        self.epoch_start_time = None
    
    def on_train_begin(self, trainer):
        trainer.logger.log(self.log_level, "Starting training...")
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        total_time = sum(history.get('epoch_time', []))
        trainer.logger.log(self.log_level, f"Training completed in {total_time:.2f}s")
    
    def on_epoch_begin(self, trainer, epoch: int):
        self.epoch_start_time = time.time()
        self.batch_times = []
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            # Format losses
            train_str = ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()])
            val_str = ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]) if val_losses else 'N/A'
            
            log_msg = (f"Epoch {epoch}: train_losses=[{train_str}], "
                      f"val_losses=[{val_str}], time={epoch_time:.2f}s")
            
            trainer.logger.log(self.log_level, log_msg)
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        if batch_idx % self.log_frequency == 0:
            loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
            trainer.logger.log(self.log_level, f"Batch {batch_idx}: {loss_str}")


class MetricsTracker(TrainingCallback):
    """Track and store training metrics."""
    
    def __init__(self, metrics_file: Optional[str] = None):
        """Initialize metrics tracker.
        
        Args:
            metrics_file: Optional file to save metrics to
        """
        self.metrics_file = metrics_file
        self.metrics_history = {
            'epoch': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': []
        }
    
    def on_train_begin(self, trainer):
        self.metrics_history = {
            'epoch': [],
            'train_losses': [],
            'val_losses': [], 
            'learning_rates': [],
            'epoch_times': []
        }
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        if self.metrics_file:
            import json
            with open(self.metrics_file, 'w') as f:
                # Convert tensors to float for JSON serialization
                serializable_history = {}
                for key, values in self.metrics_history.items():
                    serializable_history[key] = [
                        float(v) if torch.is_tensor(v) else v for v in values
                    ]
                json.dump(serializable_history, f, indent=2)
    
    def on_epoch_begin(self, trainer, epoch: int):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        epoch_time = time.time() - self.epoch_start_time
        
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_losses'].append(train_losses.copy())
        self.metrics_history['val_losses'].append(val_losses.copy())
        self.metrics_history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
        self.metrics_history['epoch_times'].append(epoch_time)
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        pass
    
    def get_metrics(self) -> Dict[str, List]:
        """Get stored metrics."""
        return self.metrics_history.copy()


class GradientClipping(TrainingCallback):
    """Gradient clipping callback."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """Initialize gradient clipping.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer, history: Dict[str, Any]):
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_losses: Dict[str, float], 
                     val_losses: Dict[str, float]):
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, losses: Dict[str, torch.Tensor]):
        # Apply gradient clipping after backward pass
        if hasattr(trainer.model, 'parameters'):
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), 
                max_norm=self.max_norm,
                norm_type=self.norm_type
            )