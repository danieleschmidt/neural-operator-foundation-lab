"""Fault-tolerant training system with automatic recovery."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
import logging
import pickle
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import gc
import psutil
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass 
class FaultToleranceConfig:
    """Configuration for fault-tolerant training."""
    checkpoint_interval: int = 100  # steps
    max_retries: int = 3
    memory_threshold: float = 0.9  # 90% of available memory
    gradient_clip_threshold: float = 1.0
    nan_detection_enabled: bool = True
    auto_recovery_enabled: bool = True
    backup_frequency: int = 500  # steps
    health_check_interval: int = 50  # steps
    
    # Error handling strategies
    oom_handling: str = "reduce_batch_size"  # or "gradient_accumulation"
    nan_handling: str = "restart_from_checkpoint"  # or "skip_batch"
    slow_convergence_handling: str = "adaptive_learning_rate"
    
    # Recovery strategies
    checkpoint_keep_last: int = 5
    automatic_model_surgery: bool = True
    dynamic_memory_management: bool = True


class HealthMonitor:
    """Monitor training health and detect anomalies."""
    
    def __init__(self, config: FaultToleranceConfig):
        self.config = config
        self.metrics_history = []
        self.anomaly_scores = []
        self.baseline_established = False
        self.baseline_metrics = {}
        
    def check_health(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check training health and return status."""
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': [],
            'anomaly_score': 0.0,
            'recommendations': []
        }
        
        # Check for NaN/Inf values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                health_status['healthy'] = False
                health_status['errors'].append(f"Invalid value in {key}: {value}")
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent / 100.0
        if memory_percent > self.config.memory_threshold:
            health_status['warnings'].append(
                f"High memory usage: {memory_percent:.1%}"
            )
            health_status['recommendations'].append("reduce_batch_size")
        
        # Check gradient health
        if 'gradient_norm' in metrics:
            grad_norm = metrics['gradient_norm']
            if grad_norm > self.config.gradient_clip_threshold * 10:
                health_status['warnings'].append(
                    f"Large gradient norm: {grad_norm:.2f}"
                )
                health_status['recommendations'].append("gradient_clipping")
        
        # Anomaly detection
        if self.baseline_established:
            anomaly_score = self._compute_anomaly_score(metrics)
            health_status['anomaly_score'] = anomaly_score
            
            if anomaly_score > 0.8:
                health_status['warnings'].append(
                    f"High anomaly score: {anomaly_score:.2f}"
                )
                health_status['recommendations'].append("investigate_anomaly")
        
        # Update history
        self.metrics_history.append(metrics.copy())
        if len(self.metrics_history) > 1000:  # Keep last 1000 entries
            self.metrics_history.pop(0)
        
        # Establish baseline if enough data
        if not self.baseline_established and len(self.metrics_history) >= 50:
            self._establish_baseline()
        
        return health_status
    
    def _compute_anomaly_score(self, current_metrics: Dict[str, float]) -> float:
        """Compute anomaly score compared to baseline."""
        if not self.baseline_established:
            return 0.0
        
        anomaly_score = 0.0
        count = 0
        
        for key, value in current_metrics.items():
            if key in self.baseline_metrics:
                baseline_mean = self.baseline_metrics[key]['mean']
                baseline_std = self.baseline_metrics[key]['std']
                
                if baseline_std > 0:
                    z_score = abs(value - baseline_mean) / baseline_std
                    anomaly_score += min(z_score / 3.0, 1.0)  # Normalize to [0,1]
                    count += 1
        
        return anomaly_score / count if count > 0 else 0.0
    
    def _establish_baseline(self):
        """Establish baseline metrics from history."""
        if len(self.metrics_history) < 10:
            return
        
        # Compute baseline statistics
        keys = self.metrics_history[0].keys()
        for key in keys:
            values = [m[key] for m in self.metrics_history if key in m and not (np.isnan(m[key]) or np.isinf(m[key]))]
            if values:
                self.baseline_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        self.baseline_established = True
        logger.info("Training health baseline established")


class CheckpointManager:
    """Manage training checkpoints with fault tolerance."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], config: FaultToleranceConfig):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        self.latest_checkpoint = None
        self.checkpoint_history = []
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], step: int, metrics: Dict[str, float],
                       additional_state: Optional[Dict[str, Any]] = None) -> Path:
        """Save training checkpoint with metadata."""
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            
            # Prepare checkpoint data
            checkpoint_data = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics.copy(),
                'timestamp': time.time(),
                'model_config': getattr(model, 'config', None),
            }
            
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            if additional_state is not None:
                checkpoint_data['additional_state'] = additional_state
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update tracking
            self.latest_checkpoint = checkpoint_path
            self.checkpoint_history.append({
                'path': checkpoint_path,
                'step': step,
                'timestamp': time.time(),
                'metrics': metrics.copy()
            })
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load training checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint
        
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning("No checkpoint found for loading")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def find_best_checkpoint(self, metric_key: str = 'val_accuracy', 
                           maximize: bool = True) -> Optional[Path]:
        """Find best checkpoint based on specified metric."""
        if not self.checkpoint_history:
            return None
        
        best_checkpoint = None
        best_value = float('-inf') if maximize else float('inf')
        
        for checkpoint_info in self.checkpoint_history:
            metrics = checkpoint_info['metrics']
            if metric_key in metrics:
                value = metrics[metric_key]
                if (maximize and value > best_value) or (not maximize and value < best_value):
                    best_value = value
                    best_checkpoint = checkpoint_info['path']
        
        return best_checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) <= self.config.checkpoint_keep_last:
            return
        
        # Sort by step number
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['step'])
        
        # Remove oldest checkpoints
        to_remove = len(sorted_checkpoints) - self.config.checkpoint_keep_last
        for i in range(to_remove):
            checkpoint_info = sorted_checkpoints[i]
            checkpoint_path = checkpoint_info['path']
            
            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                self.checkpoint_history.remove(checkpoint_info)
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint_path}: {e}")


class FaultRecoverySystem:
    """System for recovering from training faults."""
    
    def __init__(self, config: FaultToleranceConfig, checkpoint_manager: CheckpointManager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.recovery_attempts = {}
        
    def recover_from_oom(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        batch_size: int) -> Tuple[bool, int, str]:
        """Recover from out-of-memory error."""
        logger.warning("Attempting OOM recovery")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.config.oom_handling == "reduce_batch_size":
            new_batch_size = max(1, batch_size // 2)
            logger.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
            return True, new_batch_size, "batch_size_reduced"
        
        elif self.config.oom_handling == "gradient_accumulation":
            # Implement gradient accumulation logic
            logger.info("Switching to gradient accumulation mode")
            return True, batch_size, "gradient_accumulation_enabled"
        
        return False, batch_size, "oom_recovery_failed"
    
    def recover_from_nan(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """Recover from NaN gradients/losses."""
        logger.warning("Attempting NaN recovery")
        
        if self.config.nan_handling == "restart_from_checkpoint":
            # Load last good checkpoint
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data is not None:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Recovered from checkpoint after NaN")
                return True
        
        elif self.config.nan_handling == "skip_batch":
            # Reset optimizer state
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.zero_()
            logger.info("Skipped batch with NaN")
            return True
        
        return False
    
    def handle_slow_convergence(self, optimizer: torch.optim.Optimizer, 
                              metrics_history: List[Dict[str, float]]) -> bool:
        """Handle slow convergence by adjusting training parameters."""
        if len(metrics_history) < 10:
            return False
        
        # Check if loss has plateaued
        recent_losses = [m.get('loss', float('inf')) for m in metrics_history[-10:]]
        if len(set(recent_losses)) == 1:  # All losses are the same
            return False
        
        loss_trend = np.diff(recent_losses)
        if np.mean(np.abs(loss_trend)) < 1e-6:  # Very small changes
            if self.config.slow_convergence_handling == "adaptive_learning_rate":
                # Increase learning rate
                for group in optimizer.param_groups:
                    old_lr = group['lr']
                    group['lr'] = min(old_lr * 1.5, 1e-2)
                    logger.info(f"Increased learning rate from {old_lr:.2e} to {group['lr']:.2e}")
                return True
        
        return False
    
    def perform_model_surgery(self, model: nn.Module) -> bool:
        """Perform automatic model surgery to fix issues."""
        if not self.config.automatic_model_surgery:
            return False
        
        logger.info("Performing automatic model surgery")
        surgery_performed = False
        
        # Check for dead neurons
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Check for dead weights (all zeros or very small)
                    weight_norms = torch.norm(module.weight, dim=1)
                    dead_neurons = (weight_norms < 1e-6).sum().item()
                    
                    if dead_neurons > 0:
                        # Reinitialize dead neurons
                        for i in torch.where(weight_norms < 1e-6)[0]:
                            nn.init.xavier_uniform_(module.weight[i:i+1])
                            if module.bias is not None:
                                nn.init.zeros_(module.bias[i:i+1])
                        
                        logger.info(f"Reinitialized {dead_neurons} dead neurons in {name}")
                        surgery_performed = True
        
        return surgery_performed


@contextmanager
def fault_tolerant_training_step(fault_recovery: FaultRecoverySystem, 
                                model: nn.Module, optimizer: torch.optim.Optimizer,
                                health_monitor: HealthMonitor):
    """Context manager for fault-tolerant training step."""
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM Error: {e}")
            # This would typically require batch size adjustment at higher level
            raise
        elif "nan" in str(e).lower() or "inf" in str(e).lower():
            logger.error(f"NaN/Inf Error: {e}")
            fault_recovery.recover_from_nan(model, optimizer)
            raise
        else:
            logger.error(f"Unexpected error: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")
        logger.error(traceback.format_exc())
        raise


class FaultTolerantTrainer:
    """Fault-tolerant training wrapper."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 config: FaultToleranceConfig, checkpoint_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, config)
        self.health_monitor = HealthMonitor(config)
        self.fault_recovery = FaultRecoverySystem(config, self.checkpoint_manager)
        
        # Training state
        self.current_step = 0
        self.metrics_history = []
        self.recovery_count = 0
        
    def train_step_with_fault_tolerance(self, batch: Any, loss_fn: Callable) -> Dict[str, float]:
        """Execute a single training step with fault tolerance."""
        metrics = {}
        retry_count = 0
        
        while retry_count < self.config.max_retries:
            try:
                with fault_tolerant_training_step(
                    self.fault_recovery, self.model, self.optimizer, self.health_monitor
                ):
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch[0])
                    loss = loss_fn(outputs, batch[1])
                    
                    # Check for NaN/Inf
                    if self.config.nan_detection_enabled:
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"NaN/Inf loss detected: {loss}")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_threshold > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip_threshold
                        )
                    
                    # Check gradient health
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    gradient_norm = total_norm ** 0.5
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Collect metrics
                    metrics = {
                        'loss': loss.item(),
                        'gradient_norm': gradient_norm,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'step': self.current_step
                    }
                    
                    # Health check
                    if self.current_step % self.config.health_check_interval == 0:
                        health_status = self.health_monitor.check_health(metrics)
                        if not health_status['healthy']:
                            logger.warning(f"Health issues detected: {health_status}")
                    
                    # Checkpoint saving
                    if self.current_step % self.config.checkpoint_interval == 0:
                        self.checkpoint_manager.save_checkpoint(
                            self.model, self.optimizer, None, 
                            self.current_step, metrics
                        )
                    
                    self.current_step += 1
                    self.metrics_history.append(metrics)
                    
                    # Handle slow convergence
                    if len(self.metrics_history) > 20:
                        self.fault_recovery.handle_slow_convergence(
                            self.optimizer, self.metrics_history[-20:]
                        )
                    
                    break  # Success, exit retry loop
                    
            except RuntimeError as e:
                retry_count += 1
                logger.error(f"Training step failed (attempt {retry_count}/{self.config.max_retries}): {e}")
                
                if "out of memory" in str(e).lower():
                    # This would require batch size adjustment at higher level
                    raise
                elif retry_count >= self.config.max_retries:
                    logger.error("Max retries exceeded, giving up")
                    raise
                else:
                    # Try recovery
                    self.recovery_count += 1
                    if self.config.auto_recovery_enabled:
                        self.fault_recovery.perform_model_surgery(self.model)
                    
                    time.sleep(1)  # Brief pause before retry
        
        return metrics
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'current_step': self.current_step,
            'total_recovery_attempts': self.recovery_count,
            'metrics_history_length': len(self.metrics_history),
            'health_baseline_established': self.health_monitor.baseline_established,
            'checkpoint_count': len(self.checkpoint_manager.checkpoint_history),
            'latest_checkpoint': str(self.checkpoint_manager.latest_checkpoint) if self.checkpoint_manager.latest_checkpoint else None
        }


def create_fault_tolerant_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_interval: int = 100,
    max_retries: int = 3,
    auto_recovery: bool = True
) -> FaultTolerantTrainer:
    """Create a fault-tolerant trainer with default configuration."""
    config = FaultToleranceConfig(
        checkpoint_interval=checkpoint_interval,
        max_retries=max_retries,
        auto_recovery_enabled=auto_recovery
    )
    
    return FaultTolerantTrainer(model, optimizer, config, checkpoint_dir)