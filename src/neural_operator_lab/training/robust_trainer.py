# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Robust training infrastructure with advanced error handling and recovery."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
import json
import logging
import numpy as np
import warnings
from contextmanager import contextmanager

from ..base import NeuralOperatorBase, TrainingMetrics
from ..utils import setup_device, compute_spectral_metrics, gradient_clipping
from ..security.robust_validation import get_security_validator
from ..monitoring.enhanced_error_recovery import get_error_recovery_manager, robust_training_context
from .losses import PhysicsInformedLoss, SpectralLoss, ConservationLoss

logger = logging.getLogger(__name__)


@dataclass
class RobustTrainingConfig:
    """Configuration for robust neural operator training."""
    
    # Model and data
    device: str = 'auto'
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    
    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    
    # Loss configuration
    use_physics_loss: bool = True
    use_spectral_loss: bool = False
    physics_weight: float = 0.1
    spectral_weight: float = 0.05
    
    # Robustness features
    enable_error_recovery: bool = True
    enable_security_validation: bool = True
    strict_validation: bool = False
    max_recovery_attempts: int = 3
    auto_checkpoint_frequency: int = 50
    
    # Logging and monitoring
    save_dir: str = './checkpoints'
    log_every: int = 10
    save_every: int = 50
    evaluate_every: int = 25
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    detect_anomaly: bool = False  # PyTorch autograd anomaly detection
    
    # Memory optimization
    empty_cache_frequency: int = 10  # Clear CUDA cache every N steps
    max_memory_usage: float = 0.9  # Maximum GPU memory usage
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Enable anomaly detection if requested
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)


class RobustTrainer:
    """Robust trainer with comprehensive error handling and recovery."""
    
    def __init__(self,
                 model: NeuralOperatorBase,
                 config: RobustTrainingConfig = None,
                 loss_fn: Optional[nn.Module] = None):
        """Initialize robust trainer.
        
        Args:
            model: Neural operator model to train
            config: Training configuration
            loss_fn: Loss function (will create default if None)
        """
        self.config = config or RobustTrainingConfig()
        self.device = setup_device(self.config.device)
        self.model = model.to(self.device)
        
        # Initialize components
        self.loss_fn = loss_fn or self._create_default_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        
        # Metrics and monitoring
        self.metrics = TrainingMetrics()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0
        
        # Security and error handling
        if self.config.enable_security_validation:
            self.security_validator = get_security_validator(
                strict_mode=self.config.strict_validation
            )
        else:
            self.security_validator = None
            
        if self.config.enable_error_recovery:
            self.error_manager = get_error_recovery_manager()
        else:
            self.error_manager = None
        
        # Memory monitoring
        self.memory_usage_history = []
        self.performance_history = []
        
        # Model setup
        self._setup_model()
        
        logger.info(f"Robust trainer initialized with {self._count_parameters():,} parameters")
    
    def _create_default_loss(self) -> nn.Module:
        """Create default loss function."""
        losses = [nn.MSELoss()]
        weights = [1.0]
        
        if self.config.use_physics_loss:
            losses.append(PhysicsInformedLoss())
            weights.append(self.config.physics_weight)
        
        if self.config.use_spectral_loss:
            losses.append(SpectralLoss())
            weights.append(self.config.spectral_weight)
        
        if len(losses) == 1:
            return losses[0]
        else:
            return CombinedLoss(losses, weights)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.max_epochs
            )
        elif self.config.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=self.config.patience // 2
            )
        elif self.config.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_model(self):
        """Setup model for training."""
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _validate_batch(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Validate training batch."""
        if self.security_validator is None:
            return True
        
        try:
            result = self.security_validator.validate_training_batch(
                x, y, self.model, dataset_name="training"
            )
            return result.is_valid
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            return not self.config.strict_validation  # Allow training if not strict
    
    def _memory_cleanup(self):
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _monitor_memory_usage(self):
        """Monitor and log memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
            
            usage_ratio = memory_reserved / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
            
            self.memory_usage_history.append({
                'step': self.global_step,
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'free_gb': memory_free,
                'usage_ratio': usage_ratio
            })
            
            # Trigger cleanup if memory usage is high
            if usage_ratio > self.config.max_memory_usage:
                logger.warning(f"High memory usage detected: {usage_ratio:.2%}")
                self._memory_cleanup()
    
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Single training step with error handling."""
        
        context = {
            'model': self.model,
            'optimizer': self.optimizer,
            'batch_size': batch[0].size(0) if batch[0].numel() > 0 else self.config.batch_size,
            'global_step': self.global_step,
            'epoch': self.current_epoch
        }
        
        if self.config.enable_error_recovery:
            with robust_training_context(
                self.model, 
                self.optimizer, 
                self.error_manager,
                max_retries=self.config.max_recovery_attempts
            ) as training_context:
                return self._train_step_impl(batch, training_context)
        else:
            return self._train_step_impl(batch, context)
    
    def _train_step_impl(self, batch: Tuple[torch.Tensor, ...], context: Dict[str, Any]) -> Dict[str, float]:
        """Implementation of training step."""
        x, y = batch[:2]
        grid = batch[2] if len(batch) > 2 else None
        
        # Move to device
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if grid is not None:
            grid = grid.to(self.device, non_blocking=True)
        
        # Validate batch
        if not self._validate_batch(x, y):
            logger.warning("Skipping batch due to validation failure")
            return {'loss': 0.0, 'skipped': True}
        
        # Forward pass
        self.optimizer.zero_grad()
        
        step_start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Model forward pass
            if grid is not None:
                pred = self.model(x, grid)
            else:
                pred = self.model(x)
            
            # Compute loss
            if hasattr(self.loss_fn, 'forward'):
                loss = self.loss_fn(pred, y, x)  # Pass input for physics-informed loss
            else:
                loss = self.loss_fn(pred, y)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = gradient_clipping(
                    self.model.parameters(), 
                    self.config.grad_clip_norm
                )
            else:
                grad_norm = 0.0
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        step_time = time.time() - step_start_time
        
        # Compute metrics
        with torch.no_grad():
            mse_loss = nn.functional.mse_loss(pred, y)
            relative_error = torch.mean((pred - y)**2) / torch.mean(y**2)
            
            # Spectral metrics if enabled
            spectral_metrics = {}
            if self.config.use_spectral_loss:
                try:
                    spectral_metrics = compute_spectral_metrics(pred, y)
                except Exception as e:
                    logger.warning(f"Spectral metrics computation failed: {e}")
        
        # Memory monitoring
        if self.global_step % self.config.empty_cache_frequency == 0:
            self._monitor_memory_usage()
        
        # Performance tracking
        self.performance_history.append({
            'step': self.global_step,
            'step_time': step_time,
            'loss': loss.item(),
            'mse_loss': mse_loss.item(),
            'relative_error': relative_error.item(),
            'batch_size': x.size(0),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Keep history manageable
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-10000:]
        
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'mse_loss': mse_loss.item(),
            'relative_error': relative_error.item(),
            'grad_norm': grad_norm if 'grad_norm' in locals() else 0.0,
            'step_time': step_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        metrics.update(spectral_metrics)
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            self.global_step += 1
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Skip aggregation if batch was skipped
            if step_metrics.get('skipped', False):
                continue
            
            # Aggregate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                self._log_step_metrics(batch_idx, step_metrics)
            
            # Memory cleanup
            if batch_idx % self.config.empty_cache_frequency == 0:
                self._memory_cleanup()
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_metrics.get('loss', float('inf')))
            else:
                self.scheduler.step()
        
        return epoch_metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        # SECURITY_DISABLED: self.model.eval(...)  # eval() disabled for security
        val_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[:2]
                grid = batch[2] if len(batch) > 2 else None
                
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if grid is not None:
                    grid = grid.to(self.device, non_blocking=True)
                
                # Skip validation if batch is invalid
                if not self._validate_batch(x, y):
                    continue
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    if grid is not None:
                        pred = self.model(x, grid)
                    else:
                        pred = self.model(x)
                    
                    if hasattr(self.loss_fn, 'forward'):
                        loss = self.loss_fn(pred, y, x)
                    else:
                        loss = self.loss_fn(pred, y)
                
                # Compute metrics
                mse_loss = nn.functional.mse_loss(pred, y)
                relative_error = torch.mean((pred - y)**2) / torch.mean(y**2)
                
                batch_metrics = {
                    'val_loss': loss.item(),
                    'val_mse_loss': mse_loss.item(),
                    'val_relative_error': relative_error.item()
                }
                
                # Aggregate metrics
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
                
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= num_batches
        
        return val_metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, List[float]]:
        """Full training loop."""
        
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = {}
            if val_loader is not None and epoch % self.config.evaluate_every == 0:
                val_metrics = self.evaluate(val_loader)
            
            # Update metrics
            self.metrics.update(**train_metrics, **val_metrics)
            
            # Update history
            training_history['train_loss'].append(train_metrics.get('loss', 0.0))
            if val_metrics:
                training_history['val_loss'].append(val_metrics.get('val_loss', 0.0))
            training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Early stopping
            current_val_loss = val_metrics.get('val_loss', train_metrics.get('loss', float('inf')))
            if current_val_loss < self.best_loss:
                self.best_loss = current_val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(
                    Path(self.config.save_dir) / "best_model.pt",
                    is_best=True
                )
            else:
                self.patience_counter += 1
            
            # Auto-checkpoint
            if epoch % self.config.auto_checkpoint_frequency == 0:
                self.save_checkpoint(
                    Path(self.config.save_dir) / f"checkpoint_epoch_{epoch}.pt"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        logger.info("Training completed!")
        return training_history
    
    def save_checkpoint(self, path: Union[str, Path], is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,
            'metrics': self.metrics.get_summary(),
            'is_best': is_best
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def _log_step_metrics(self, step: int, metrics: Dict[str, float]):
        """Log step metrics."""
        log_msg = f"Step {step}: "
        for key, value in metrics.items():
            if key != 'skipped':
                log_msg += f"{key}={value:.6f} "
        logger.info(log_msg)
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch metrics."""
        log_msg = f"Epoch {epoch}: "
        log_msg += f"train_loss={train_metrics.get('loss', 0):.6f} "
        if val_metrics:
            log_msg += f"val_loss={val_metrics.get('val_loss', 0):.6f} "
        log_msg += f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
        log_msg += f"time={epoch_time:.1f}s"
        logger.info(log_msg)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'total_parameters': self._count_parameters(),
            'memory_usage_history': self.memory_usage_history[-100:],  # Last 100 records
            'performance_history': self.performance_history[-100:],    # Last 100 records
        }
        
        if self.security_validator:
            stats['security_report'] = self.security_validator.get_security_report()
        
        if self.error_manager:
            stats['error_statistics'] = self.error_manager.get_error_statistics()
        
        return stats


class CombinedLoss(nn.Module):
    """Combined loss function with multiple loss terms."""
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            if hasattr(loss_fn, 'forward') and 'inputs' in loss_fn.forward.__code__.co_varnames:
                loss_value = loss_fn(pred, target, inputs)
            else:
                loss_value = loss_fn(pred, target)
            total_loss += weight * loss_value
        return total_loss