"""Distributed training for neural operators with advanced scaling."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import os
import time
import logging
from pathlib import Path

from ..base import NeuralOperatorBase
from ..training import Trainer, TrainerConfig
from .utils import setup_distributed, cleanup_distributed, reduce_tensor


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Distribution settings
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Advanced distributed features
    gradient_compression: bool = False
    gradient_clipping_distributed: bool = True
    bucket_size_mb: int = 25
    
    # Mixed precision with distributed
    fp16: bool = False
    loss_scale: float = 65536.0
    dynamic_loss_scaling: bool = True
    
    # ZeRO optimization levels
    zero_stage: int = 0  # 0: disabled, 1: optimizer states, 2: + gradients, 3: + parameters
    
    # Communication optimization
    overlap_comm_compute: bool = True
    find_unused_parameters: bool = True
    broadcast_buffers: bool = True
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    
    def __post_init__(self):
        # Set device based on local rank
        if torch.cuda.is_available():
            self.device = f'cuda:{self.local_rank}'
        else:
            self.device = 'cpu'


class DistributedTrainer(Trainer):
    """Distributed neural operator trainer with advanced optimizations."""
    
    def __init__(self,
                 model: NeuralOperatorBase,
                 config: TrainerConfig,
                 distributed_config: DistributedConfig,
                 **kwargs):
        """Initialize distributed trainer.
        
        Args:
            model: Neural operator model
            config: Training configuration
            distributed_config: Distributed training configuration
            **kwargs: Additional arguments for base trainer
        """
        self.dist_config = distributed_config
        
        # Initialize distributed backend
        if self.dist_config.world_size > 1:
            setup_distributed(
                backend=self.dist_config.backend,
                rank=self.dist_config.rank,
                world_size=self.dist_config.world_size
            )
        
        # Setup device
        self.device = torch.device(self.dist_config.device)
        torch.cuda.set_device(self.device)
        
        # Initialize base trainer
        config.device = str(self.device)
        super().__init__(model, config, **kwargs)
        
        # Wrap model for distributed training
        self.model = self._setup_distributed_model(model)
        
        # Setup distributed optimizer
        self.optimizer = self._setup_distributed_optimizer()
        
        # Advanced features
        self._setup_zero_optimization()
        self._setup_gradient_compression()
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.comm_time = 0.0
        self.compute_time = 0.0
    
    def _setup_distributed_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        model = model.to(self.device)
        
        if self.dist_config.world_size > 1:
            # Synchronize model parameters across ranks
            self._broadcast_model_parameters(model)
            
            # Wrap with DistributedDataParallel
            model = DDP(
                model,
                device_ids=[self.dist_config.local_rank],
                output_device=self.dist_config.local_rank,
                find_unused_parameters=self.dist_config.find_unused_parameters,
                broadcast_buffers=self.dist_config.broadcast_buffers,
                bucket_cap_mb=self.dist_config.bucket_size_mb,
                gradient_as_bucket_view=True  # Memory optimization
            )
        
        return model
    
    def _broadcast_model_parameters(self, model: nn.Module):
        """Broadcast model parameters from rank 0."""
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        # Also broadcast buffers
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    
    def _setup_distributed_optimizer(self):
        """Setup optimizer with distributed considerations."""
        optimizer = super()._create_optimizer()
        
        # Apply distributed-specific optimizations
        if self.dist_config.gradient_compression:
            optimizer = self._wrap_gradient_compression(optimizer)
        
        return optimizer
    
    def _setup_zero_optimization(self):
        """Setup ZeRO optimization."""
        if self.dist_config.zero_stage > 0:
            try:
                from deepspeed.ops.adam import FusedAdam
                from deepspeed import initialize
                
                # DeepSpeed ZeRO configuration
                ds_config = {
                    "train_batch_size": self.config.batch_size * self.dist_config.world_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "optimizer": {
                        "type": "Adam",
                        "params": {
                            "lr": self.config.learning_rate,
                            "weight_decay": self.config.weight_decay
                        }
                    },
                    "zero_optimization": {
                        "stage": self.dist_config.zero_stage,
                        "allgather_partitions": True,
                        "allgather_bucket_size": self.dist_config.bucket_size_mb * 1024 * 1024,
                        "overlap_comm": self.dist_config.overlap_comm_compute,
                        "reduce_scatter": True,
                        "reduce_bucket_size": self.dist_config.bucket_size_mb * 1024 * 1024,
                        "contiguous_gradients": True
                    },
                    "fp16": {
                        "enabled": self.dist_config.fp16,
                        "loss_scale": self.dist_config.loss_scale,
                        "initial_scale_power": 16,
                        "loss_scale_window": 1000,
                        "hysteresis": 2,
                        "min_loss_scale": 1
                    }
                }
                
                # Initialize DeepSpeed
                self.model, self.optimizer, _, _ = initialize(
                    model=self.model,
                    config=ds_config
                )
                
                self.use_deepspeed = True
                self.logger.info(f"Initialized DeepSpeed ZeRO stage {self.dist_config.zero_stage}")
                
            except ImportError:
                self.logger.warning("DeepSpeed not available. ZeRO optimization disabled.")
                self.use_deepspeed = False
        else:
            self.use_deepspeed = False
    
    def _setup_gradient_compression(self):
        """Setup gradient compression for communication efficiency."""
        if self.dist_config.gradient_compression:
            # Implement gradient compression (simplified)
            self.gradient_compressor = GradientCompressor()
    
    def _wrap_gradient_compression(self, optimizer):
        """Wrap optimizer with gradient compression."""
        original_step = optimizer.step
        
        def compressed_step(closure=None):
            if hasattr(self, 'gradient_compressor'):
                self.gradient_compressor.compress_gradients(self.model)
            return original_step(closure)
        
        optimizer.step = compressed_step
        return optimizer
    
    def create_distributed_loader(self, dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create distributed data loader."""
        sampler = None
        if self.dist_config.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist_config.world_size,
                rank=self.dist_config.rank,
                shuffle=shuffle,
                drop_last=True
            )
            shuffle = False  # Handled by sampler
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True
        )
    
    def train_epoch_distributed(self, train_loader: DataLoader) -> Dict[str, float]:
        """Distributed training epoch with performance monitoring."""
        self.model.train()
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)
        
        epoch_losses = {}
        num_batches = 0
        
        comm_time_total = 0.0
        compute_time_total = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            compute_start = time.time()
            
            # Move batch to device
            if len(batch) == 3:
                inputs, targets, grid = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                grid = grid.to(self.device, non_blocking=True)
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                grid = None
            
            # Forward pass with timing
            if self.use_deepspeed:
                loss_dict = self._deepspeed_forward_backward(inputs, targets, grid)
            else:
                loss_dict = self._standard_forward_backward(inputs, targets, grid)
            
            compute_time = time.time() - compute_start
            
            # Communication timing (DDP allreduce)
            comm_start = time.time()
            if not self.use_deepspeed and self.dist_config.world_size > 1:
                # Gradients are automatically synchronized by DDP
                pass
            comm_time = time.time() - comm_start
            
            comm_time_total += comm_time
            compute_time_total += compute_time
            
            # Update metrics
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item() if hasattr(value, 'item') else value
            
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0 and self.dist_config.rank == 0:
                self._log_distributed_batch(batch_idx, len(train_loader), loss_dict, 
                                           compute_time, comm_time)
        
        # Reduce metrics across all ranks
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            if self.dist_config.world_size > 1:
                epoch_losses[key] = reduce_tensor(
                    torch.tensor(epoch_losses[key]).to(self.device)
                ).item()
        
        # Store timing information
        self.comm_time = comm_time_total
        self.compute_time = compute_time_total
        
        return epoch_losses
    
    def _deepspeed_forward_backward(self, inputs, targets, grid) -> Dict[str, torch.Tensor]:
        """DeepSpeed forward and backward pass."""
        loss_dict = self._compute_loss(inputs, targets, grid)
        
        # DeepSpeed handles scaling and backward pass
        self.model.backward(loss_dict['total_loss'])
        self.model.step()
        
        return loss_dict
    
    def _standard_forward_backward(self, inputs, targets, grid) -> Dict[str, torch.Tensor]:
        """Standard distributed forward and backward pass."""
        # Mixed precision training
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                loss_dict = self._compute_loss(inputs, targets, grid)
            
            # Scaled backward pass
            self.scaler.scale(loss_dict['total_loss']).backward()
            
            # Gradient clipping with scaling
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict = self._compute_loss(inputs, targets, grid)
            loss_dict['total_loss'].backward()
            
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        return loss_dict
    
    def _log_distributed_batch(self, batch_idx: int, total_batches: int, 
                              loss_dict: Dict[str, torch.Tensor],
                              compute_time: float, comm_time: float):
        """Log distributed training progress."""
        if self.dist_config.rank == 0:
            losses_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            
            efficiency = compute_time / (compute_time + comm_time + 1e-6) * 100
            
            self.logger.info(
                f"Rank {self.dist_config.rank} | "
                f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches} | "
                f"{losses_str} | "
                f"Compute: {compute_time:.3f}s, Comm: {comm_time:.3f}s | "
                f"Efficiency: {efficiency:.1f}%"
            )
    
    def save_distributed_checkpoint(self, filename: str):
        """Save checkpoint in distributed setting."""
        if self.dist_config.rank == 0:
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'distributed_config': self.dist_config,
                'performance_stats': {
                    'comm_time': self.comm_time,
                    'compute_time': self.compute_time
                }
            }
            
            if self.use_deepspeed:
                # DeepSpeed handles model saving
                self.model.save_checkpoint(str(Path(self.config.save_dir) / filename))
            else:
                # Extract model from DDP wrapper
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                checkpoint['model_state_dict'] = model_state
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                
                torch.save(checkpoint, Path(self.config.save_dir) / filename)
            
            self.logger.info(f"Saved distributed checkpoint: {filename}")
    
    def get_distributed_metrics(self) -> Dict[str, Any]:
        """Get distributed training metrics."""
        base_metrics = super().get_metric_summary() if hasattr(super(), 'get_metric_summary') else {}
        
        distributed_metrics = {
            'world_size': self.dist_config.world_size,
            'rank': self.dist_config.rank,
            'communication_efficiency': self.compute_time / (self.compute_time + self.comm_time + 1e-6),
            'communication_overhead': self.comm_time / (self.compute_time + 1e-6),
            'using_deepspeed': self.use_deepspeed,
            'zero_stage': self.dist_config.zero_stage if self.use_deepspeed else 0
        }
        
        return {**base_metrics, **distributed_metrics}
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if self.dist_config.world_size > 1:
            cleanup_distributed()
        
        if hasattr(super(), 'cleanup'):
            super().cleanup()


class GradientCompressor:
    """Gradient compression for efficient communication."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
    
    def compress_gradients(self, model: nn.Module):
        """Compress gradients using top-k sparsification."""
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Top-k compression
                k = max(1, int(grad.numel() * self.compression_ratio))
                
                # Flatten gradient
                flat_grad = grad.flatten()
                
                # Get top-k values
                _, indices = torch.topk(torch.abs(flat_grad), k)
                
                # Create sparse gradient
                compressed_grad = torch.zeros_like(flat_grad)
                compressed_grad[indices] = flat_grad[indices]
                
                # Reshape back
                param.grad.data = compressed_grad.reshape(grad.shape)


def launch_distributed_training(
    train_fn: callable,
    world_size: int,
    backend: str = 'nccl',
    **kwargs
):
    """Launch distributed training across multiple processes."""
    
    if world_size == 1:
        # Single process training
        dist_config = DistributedConfig(world_size=1, rank=0, local_rank=0)
        return train_fn(dist_config, **kwargs)
    
    else:
        # Multi-process training
        import torch.multiprocessing as mp
        
        def worker(rank, world_size, backend, train_fn, kwargs):
            dist_config = DistributedConfig(
                world_size=world_size,
                rank=rank,
                local_rank=rank,
                backend=backend
            )
            return train_fn(dist_config, **kwargs)
        
        mp.spawn(
            worker,
            args=(world_size, backend, train_fn, kwargs),
            nprocs=world_size,
            join=True
        )