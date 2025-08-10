"""Main training infrastructure for neural operators."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
import json
import logging
import warnings
import gc
from contextlib import contextmanager

from ..base import NeuralOperatorBase, TrainingMetrics
from ..utils import PerformanceProfiler, setup_device, compute_spectral_metrics
from .losses import PhysicsInformedLoss, SpectralLoss, ConservationLoss
from .callbacks import TrainingCallback, CallbackList
from ..monitoring import ComprehensiveLogger, AdvancedErrorHandler
from ..security import InputValidator


@dataclass
class TrainerConfig:
    """Configuration for neural operator training."""
    
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
    
    # Logging and checkpointing
    save_dir: str = './checkpoints'
    log_every: int = 10
    save_every: int = 50
    evaluate_every: int = 25
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    
    # Validation
    val_fraction: float = 0.1
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


class Trainer:
    """Main trainer class for neural operators."""
    
    def __init__(self,
                 model: NeuralOperatorBase,
                 config: TrainerConfig = None,
                 loss_fn: Optional[nn.Module] = None,
                 callbacks: Optional[List[TrainingCallback]] = None):
        """Initialize trainer.
        
        Args:
            model: Neural operator model to train
            config: Training configuration
            loss_fn: Loss function (will create default if None)
            callbacks: List of training callbacks
        """
        self.config = config or TrainerConfig()
        self.device = setup_device(self.config.device)
        
        # Setup robust monitoring and error handling
        self.logger = ComprehensiveLogger(
            log_dir=Path(self.config.save_dir) / "logs",
            experiment_name="neural_operator_training"
        )
        self.error_handler = AdvancedErrorHandler(
            logger=self.logger,
            max_retries=3,
            recovery_strategies=['checkpoint_restore', 'lr_reduction', 'batch_reduction']
        )
        
        # Setup security validation
        self.input_validator = InputValidator()
        
        # Setup model with error handling
        try:
            self.model = model.to(self.device)
            self.logger.log_info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.error_handler.handle_error(e, context="model_initialization")
            raise
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = self._create_default_loss()
        else:
            self.loss_fn = loss_fn.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.log_info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self._training_interrupted = False
        
        # Metrics and logging
        self.metrics = TrainingMetrics()
        self.profiler = PerformanceProfiler()
        
        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        # Setup logging
        self._setup_logging()
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        # Memory optimization
        if hasattr(torch.backends, 'cudnn') and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _create_default_loss(self) -> nn.Module:
        """Create default loss function."""
        losses = []
        weights = []
        
        # Always include MSE loss
        losses.append(nn.MSELoss())
        weights.append(1.0)
        
        # Add physics loss if enabled
        if self.config.use_physics_loss:
            physics_loss = PhysicsInformedLoss(
                data_weight=1.0,
                pde_weight=self.config.physics_weight
            )
            losses.append(physics_loss)
            weights.append(1.0)
        
        # Add spectral loss if enabled
        if self.config.use_spectral_loss:
            spectral_loss = SpectralLoss(spectral_weight=self.config.spectral_weight)
            losses.append(spectral_loss)
            weights.append(1.0)
        
        if len(losses) == 1:
            return losses[0]
        else:
            return CombinedLoss(losses, weights)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2
            )
        elif self.config.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.save_dir) / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # Manual implementation for models without built-in support
            for module in self.model.modules():
                if hasattr(module, '_gradient_checkpointing'):
                    module._gradient_checkpointing = True
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        with self.profiler.profile('epoch_training'):
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                if len(batch) == 3:
                    inputs, targets, grid = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = grid.to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = None
                
                # Forward pass
                with self.profiler.profile('forward_pass'):
                    if self.config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            loss_dict = self._compute_loss(inputs, targets, grid)
                    else:
                        loss_dict = self._compute_loss(inputs, targets, grid)
                
                # Backward pass
                with self.profiler.profile('backward_pass'):
                    self._backward_step(loss_dict['total_loss'])
                
                # Update metrics
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()
                
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if batch_idx % self.config.log_every == 0:
                    self._log_batch(batch_idx, len(train_loader), loss_dict)
                
                # Callbacks
                self.callbacks.on_batch_end(self, batch_idx, loss_dict)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                     grid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch."""
        # Forward pass
        predictions = self.model(inputs, grid)
        
        # Compute loss
        if isinstance(self.loss_fn, PhysicsInformedLoss):
            loss_dict = self.loss_fn(
                pred=predictions,
                target=targets,
                coords=grid,
                inputs=inputs
            )
        elif isinstance(self.loss_fn, CombinedLoss):
            loss_dict = self.loss_fn(predictions, targets, grid, inputs)
        else:
            # Simple loss function
            loss = self.loss_fn(predictions, targets)
            loss_dict = {'total_loss': loss}
        
        return loss_dict
    
    def _backward_step(self, loss: torch.Tensor):
        """Perform backward pass and optimization step."""
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate gradients
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            # Optimizer step
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if len(batch) == 3:
                    inputs, targets, grid = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = grid.to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = None
                
                # Forward pass
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss_dict = self._compute_loss(inputs, targets, grid)
                else:
                    loss_dict = self._compute_loss(inputs, targets, grid)
                
                # Update metrics
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = 0
                    val_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None) -> Dict[str, Any]:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
            
        Returns:
            Training history and final metrics
        """
        epochs = epochs or self.config.max_epochs
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        self.logger.info(f"Device: {self.device}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Callbacks
        self.callbacks.on_train_begin(self)
        
        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Callbacks
                self.callbacks.on_epoch_begin(self, epoch)
                
                # Training
                train_losses = self.train_epoch(train_loader)
                
                # Validation
                val_losses = {}
                if val_loader is not None:
                    val_losses = self.validate_epoch(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_losses.get('total_loss', train_losses['total_loss']))
                    else:
                        self.scheduler.step()
                
                # Update history
                epoch_time = time.time() - epoch_start_time
                history['train_loss'].append(train_losses.get('total_loss', 0))
                history['val_loss'].append(val_losses.get('total_loss', 0))
                history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                history['epoch_time'].append(epoch_time)
                
                # Update metrics
                self.metrics.update(
                    train_loss=train_losses.get('total_loss', 0),
                    val_loss=val_losses.get('total_loss', 0),
                    epoch_time=epoch_time
                )
                
                # Logging
                self._log_epoch(epoch, train_losses, val_losses, epoch_time)
                
                # Checkpointing
                if val_losses:
                    val_loss = val_losses.get('total_loss', float('inf'))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model.pt')
                
                if epoch % self.config.save_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                
                # Callbacks
                self.callbacks.on_epoch_end(self, epoch, train_losses, val_losses)
                
                # Early stopping check
                if self._should_early_stop(history):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            # Callbacks
            self.callbacks.on_train_end(self, history)
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Performance summary
        performance_summary = self.profiler.get_summary()
        self.logger.info(f"Training completed in {sum(history['epoch_time']):.2f}s")
        
        return {
            'history': history,
            'final_metrics': self.metrics.get_summary(),
            'performance': performance_summary,
            'best_val_loss': self.best_val_loss
        }
    
    def _should_early_stop(self, history: Dict[str, List]) -> bool:
        """Check if training should stop early."""
        if len(history['val_loss']) < self.config.patience:
            return False
        
        # Check if validation loss hasn't improved
        recent_losses = history['val_loss'][-self.config.patience:]
        best_recent = min(recent_losses)
        current_best = min(history['val_loss'])
        
        return best_recent > current_best
    
    def _log_batch(self, batch_idx: int, total_batches: int, loss_dict: Dict[str, torch.Tensor]):
        """Log batch information."""
        if batch_idx % self.config.log_every == 0:
            losses_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            self.logger.info(
                f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches} - {losses_str}"
            )
    
    def _log_epoch(self, epoch: int, train_losses: Dict[str, float],
                   val_losses: Dict[str, float], epoch_time: float):
        """Log epoch information."""
        train_loss_str = ", ".join([f"train_{k}: {v:.4f}" for k, v in train_losses.items()])
        
        if val_losses:
            val_loss_str = ", ".join([f"val_{k}: {v:.4f}" for k, v in val_losses.items()])
            log_str = f"Epoch {epoch}: {train_loss_str}, {val_loss_str}"
        else:
            log_str = f"Epoch {epoch}: {train_loss_str}"
        
        log_str += f", lr: {self.optimizer.param_groups[0]['lr']:.2e}, time: {epoch_time:.2f}s"
        
        self.logger.info(log_str)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.save_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': self.metrics.get_summary()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def evaluate_rollout(self,
                        test_loader: DataLoader,
                        timesteps: int = 100,
                        compute_conservation: bool = True) -> Dict[str, Any]:
        """Evaluate rollout performance over multiple timesteps."""
        self.model.eval()
        
        rollout_errors = []
        conservation_violations = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inputs, targets, grid = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = grid.to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    grid = None
                
                # Initial prediction
                current_state = inputs
                
                # Rollout
                errors = []
                for t in range(min(timesteps, targets.size(1) if targets.dim() > 3 else 1)):
                    # Predict next state
                    pred = self.model(current_state, grid)
                    
                    # Compute error against ground truth
                    if t < targets.size(1):
                        target_t = targets[:, t] if targets.dim() > 3 else targets
                        error = torch.mean((pred - target_t)**2).item()
                        errors.append(error)
                    
                    # Update current state for next prediction
                    current_state = pred
                
                rollout_errors.extend(errors)
                
                # Conservation laws
                if compute_conservation:
                    energy_violation = ConservationLoss().conservation_checker.energy_conservation(current_state)
                    conservation_violations.append(energy_violation)
        
        return {
            'mean_error': np.mean(rollout_errors) if rollout_errors else 0,
            'error_growth': rollout_errors,
            'energy_conservation': np.mean(conservation_violations) if conservation_violations else 0,
            'rollout_stability': len([e for e in rollout_errors if e < 1.0]) / len(rollout_errors) if rollout_errors else 0
        }


class CombinedLoss(nn.Module):
    """Combine multiple loss functions with weights."""
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                grid: Optional[torch.Tensor] = None,
                inputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        total_loss = 0
        loss_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            if isinstance(loss_fn, PhysicsInformedLoss):
                losses = loss_fn(pred, target, grid, inputs)
                for key, value in losses.items():
                    loss_dict[f'loss_{i}_{key}'] = value
                total_loss += weight * losses['total_loss']
            else:
                loss = loss_fn(pred, target)
                loss_dict[f'loss_{i}'] = loss
                total_loss += weight * loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict