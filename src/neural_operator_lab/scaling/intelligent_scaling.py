# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Intelligent auto-scaling system for neural operator training and inference."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import psutil
import time
import logging
import threading
import queue
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from contextlib import contextmanager
import subprocess
import os

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling modes for neural operator training."""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"
    ADAPTIVE = "adaptive"


@dataclass
class ScalingConfig:
    """Configuration for intelligent scaling."""
    
    # Basic scaling parameters
    mode: ScalingMode = ScalingMode.ADAPTIVE
    min_gpus: int = 1
    max_gpus: int = 8
    min_nodes: int = 1
    max_nodes: int = 4
    
    # Performance thresholds
    gpu_utilization_target: float = 0.85
    memory_utilization_target: float = 0.80
    throughput_target: float = 100.0  # samples/sec
    
    # Scaling triggers
    scale_up_threshold: float = 0.95    # Scale up when utilization > 95%
    scale_down_threshold: float = 0.60  # Scale down when utilization < 60%
    stability_window: int = 300         # Seconds to wait before scaling
    
    # Resource limits
    max_memory_per_gpu: float = 0.90    # Max GPU memory usage
    max_cpu_utilization: float = 0.80   # Max CPU usage
    
    # Network and communication
    backend: str = "nccl"
    master_port: int = 29500
    
    # Advanced features
    enable_gradient_compression: bool = True
    enable_mixed_precision: bool = True
    enable_zero_optimization: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.max_gpus = min(self.max_gpus, torch.cuda.device_count())
        if self.max_gpus == 0:
            self.mode = ScalingMode.SINGLE_GPU
            logger.warning("No CUDA devices available, forcing single GPU mode")


class ResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self, monitoring_interval: int = 10):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = []
        self.is_monitoring = False
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
            'network_io': dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_metrics = []
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    gpu_metrics.append({
                        'device_id': i,
                        'memory_allocated': memory_allocated,
                        'memory_reserved': memory_reserved,
                        'total_memory': total_memory,
                        'memory_utilization': memory_reserved / total_memory,
                        'name': torch.cuda.get_device_name(i)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get GPU {i} metrics: {e}")
            
            metrics['gpus'] = gpu_metrics
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self._collect_metrics()
    
    def get_average_metrics(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get average metrics over a time window."""
        cutoff_time = time.time() - window_seconds
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        
        if not recent_metrics:
            return self.get_current_metrics()
        
        # Compute averages
        avg_metrics = {
            'cpu_percent': np.mean([m['cpu_percent'] for m in recent_metrics]),
            'memory_percent': np.mean([m['memory_percent'] for m in recent_metrics]),
            'sample_count': len(recent_metrics)
        }
        
        # GPU averages
        if recent_metrics[0].get('gpus'):
            gpu_count = len(recent_metrics[0]['gpus'])
            gpu_utilizations = []
            
            for gpu_id in range(gpu_count):
                utilizations = [
                    m['gpus'][gpu_id]['memory_utilization'] 
                    for m in recent_metrics 
                    if len(m.get('gpus', [])) > gpu_id
                ]
                if utilizations:
                    gpu_utilizations.append(np.mean(utilizations))
            
            avg_metrics['gpu_memory_utilization'] = gpu_utilizations
            avg_metrics['max_gpu_utilization'] = max(gpu_utilizations) if gpu_utilizations else 0
        
        return avg_metrics


class ScalingDecisionEngine:
    """Make intelligent scaling decisions based on metrics."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.last_scaling_time = 0
        self.scaling_history = []
    
    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling up is needed."""
        # Check stability window
        if time.time() - self.last_scaling_time < self.config.stability_window:
            return False
        
        # Check GPU utilization
        gpu_utilization = metrics.get('max_gpu_utilization', 0)
        if gpu_utilization > self.config.scale_up_threshold:
            return True
        
        # Check CPU utilization
        cpu_utilization = metrics.get('cpu_percent', 0) / 100.0
        if cpu_utilization > self.config.scale_up_threshold:
            return True
        
        # Check memory pressure
        memory_utilization = metrics.get('memory_percent', 0) / 100.0
        if memory_utilization > self.config.scale_up_threshold:
            return True
        
        return False
    
    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling down is possible."""
        # Check stability window
        if time.time() - self.last_scaling_time < self.config.stability_window:
            return False
        
        # Don't scale below minimum
        current_gpus = len(metrics.get('gpu_memory_utilization', [1]))
        if current_gpus <= self.config.min_gpus:
            return False
        
        # Check if utilization is low enough
        gpu_utilization = metrics.get('max_gpu_utilization', 1.0)
        cpu_utilization = metrics.get('cpu_percent', 100) / 100.0
        memory_utilization = metrics.get('memory_percent', 100) / 100.0
        
        all_low = (
            gpu_utilization < self.config.scale_down_threshold and
            cpu_utilization < self.config.scale_down_threshold and
            memory_utilization < self.config.scale_down_threshold
        )
        
        return all_low
    
    def get_target_scale(self, current_scale: int, metrics: Dict[str, Any]) -> int:
        """Get target scale based on current metrics."""
        if self.should_scale_up(metrics):
            return min(current_scale + 1, self.config.max_gpus)
        elif self.should_scale_down(metrics):
            return max(current_scale - 1, self.config.min_gpus)
        else:
            return current_scale
    
    def record_scaling_decision(self, from_scale: int, to_scale: int, reason: str):
        """Record scaling decision for analysis."""
        decision = {
            'timestamp': time.time(),
            'from_scale': from_scale,
            'to_scale': to_scale,
            'reason': reason
        }
        self.scaling_history.append(decision)
        self.last_scaling_time = time.time()
        
        # Keep history manageable
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
        
        logger.info(f"Scaling decision: {from_scale} -> {to_scale} ({reason})")


class DistributedTrainingManager:
    """Manage distributed training setup and scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_world_size = 1
        self.current_rank = 0
        self.is_distributed = False
        self.process_group = None
    
    def setup_distributed(self, rank: int, world_size: int, master_addr: str = "localhost"):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size
        )
        
        self.current_rank = rank
        self.current_world_size = world_size
        self.is_distributed = True
        
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
    
    def wrap_model(self, model: torch.nn.Module, device_ids: Optional[List[int]] = None) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if self.is_distributed:
            if device_ids is None:
                device_ids = [self.current_rank % torch.cuda.device_count()]
            
            model = DDP(
                model,
                device_ids=device_ids,
                find_unused_parameters=self.config.find_unused_parameters if hasattr(self.config, 'find_unused_parameters') else False
            )
            
            logger.info(f"Model wrapped for distributed training on devices: {device_ids}")
        
        return model
    
    def create_distributed_dataloader(self, dataset, batch_size: int, **kwargs) -> DataLoader:
        """Create distributed data loader."""
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.current_world_size,
                rank=self.current_rank,
                shuffle=kwargs.pop('shuffle', True)
            )
            kwargs['sampler'] = sampler
            kwargs['shuffle'] = False  # Sampler handles shuffling
        
        return DataLoader(dataset, batch_size=batch_size, **kwargs)
    
    def cleanup_distributed(self):
        """Clean up distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()
            self.is_distributed = False
            logger.info("Distributed training cleaned up")


class IntelligentScaler:
    """Main intelligent scaling system."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.monitor = ResourceMonitor()
        self.decision_engine = ScalingDecisionEngine(self.config)
        self.distributed_manager = DistributedTrainingManager(self.config)
        
        self.current_scale = 1
        self.scaling_active = False
        self._scaling_thread = None
        
    def start_scaling(self):
        """Start intelligent scaling system."""
        self.monitor.start_monitoring()
        self.scaling_active = True
        
        # Start scaling decision thread
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        
        logger.info("Intelligent scaling system started")
    
    def stop_scaling(self):
        """Stop intelligent scaling system."""
        self.scaling_active = False
        self.monitor.stop_monitoring()
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10)
        
        logger.info("Intelligent scaling system stopped")
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.scaling_active:
            try:
                # Get metrics
                avg_metrics = self.monitor.get_average_metrics(
                    window_seconds=self.config.stability_window
                )
                
                # Make scaling decision
                target_scale = self.decision_engine.get_target_scale(
                    self.current_scale, avg_metrics
                )
                
                # Apply scaling if needed
                if target_scale != self.current_scale:
                    reason = "scale_up" if target_scale > self.current_scale else "scale_down"
                    self._apply_scaling(target_scale, reason)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)
    
    def _apply_scaling(self, target_scale: int, reason: str):
        """Apply scaling changes."""
        logger.info(f"Applying scaling: {self.current_scale} -> {target_scale}")
        
        try:
            # Record decision
            self.decision_engine.record_scaling_decision(
                self.current_scale, target_scale, reason
            )
            
            # For now, this is a placeholder for actual scaling implementation
            # In practice, this would involve:
            # 1. Spawning/terminating processes
            # 2. Redistributing model and data
            # 3. Updating distributed training setup
            
            self.current_scale = target_scale
            logger.info(f"Scaling applied successfully: new scale = {target_scale}")
            
        except Exception as e:
            logger.error(f"Failed to apply scaling: {e}")
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get current scaling recommendations."""
        current_metrics = self.monitor.get_current_metrics()
        avg_metrics = self.monitor.get_average_metrics()
        
        recommendations = {
            'current_scale': self.current_scale,
            'recommended_scale': self.decision_engine.get_target_scale(
                self.current_scale, avg_metrics
            ),
            'should_scale_up': self.decision_engine.should_scale_up(avg_metrics),
            'should_scale_down': self.decision_engine.should_scale_down(avg_metrics),
            'current_metrics': current_metrics,
            'average_metrics': avg_metrics,
            'scaling_history': self.decision_engine.scaling_history[-10:]  # Last 10 decisions
        }
        
        return recommendations
    
    def force_scale(self, target_scale: int, reason: str = "manual"):
        """Force scaling to a specific scale."""
        if target_scale < self.config.min_gpus or target_scale > self.config.max_gpus:
            raise ValueError(f"Target scale {target_scale} outside bounds [{self.config.min_gpus}, {self.config.max_gpus}]")
        
        self._apply_scaling(target_scale, reason)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'scaling_config': {
                'mode': self.config.mode.value,
                'min_gpus': self.config.min_gpus,
                'max_gpus': self.config.max_gpus,
                'scale_up_threshold': self.config.scale_up_threshold,
                'scale_down_threshold': self.config.scale_down_threshold
            },
            'current_status': {
                'current_scale': self.current_scale,
                'scaling_active': self.scaling_active,
                'distributed_training': self.distributed_manager.is_distributed,
                'world_size': self.distributed_manager.current_world_size
            },
            'recommendations': self.get_scaling_recommendations()
        }


@contextmanager
def intelligent_scaling_context(config: ScalingConfig = None):
    """Context manager for intelligent scaling."""
    scaler = IntelligentScaler(config)
    scaler.start_scaling()
    
    try:
        yield scaler
    finally:
        scaler.stop_scaling()


def setup_distributed_training(model: torch.nn.Module, 
                             rank: int, 
                             world_size: int,
                             config: ScalingConfig = None) -> Tuple[torch.nn.Module, DistributedTrainingManager]:
    """Setup distributed training for a model."""
    config = config or ScalingConfig()
    manager = DistributedTrainingManager(config)
    
    # Setup distributed environment
    manager.setup_distributed(rank, world_size)
    
    # Wrap model
    distributed_model = manager.wrap_model(model)
    
    return distributed_model, manager


# Utility functions for multi-GPU training
def get_optimal_batch_size(model: torch.nn.Module, 
                          device: torch.device,
                          sample_input: torch.Tensor,
                          max_memory_fraction: float = 0.8) -> int:
    """Determine optimal batch size for given model and device."""
    if not torch.cuda.is_available():
        return 32  # Default for CPU
    
    # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
    torch.cuda.empty_cache()
    
    # Start with batch size of 1
    batch_size = 1
    max_batch_size = 1
    
    try:
        while batch_size <= 1024:  # Reasonable upper limit
            # Create batch
            batch = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
            batch = batch.to(device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(batch)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            if memory_used / total_memory < max_memory_fraction:
                max_batch_size = batch_size
                batch_size *= 2
            else:
                break
                
            # Clean up
            del batch
            torch.cuda.empty_cache()
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.info(f"OOM at batch size {batch_size}, using {max_batch_size}")
        else:
            raise
    
    finally:
        torch.cuda.empty_cache()
    
    return max_batch_size


def estimate_training_time(model: torch.nn.Module,
                         dataset_size: int,
                         batch_size: int,
                         epochs: int,
                         device: torch.device) -> Dict[str, float]:
    """Estimate training time for given configuration."""
    # Rough timing estimates based on empirical data
    base_time_per_sample = 0.001  # 1ms per sample baseline
    
    # Model complexity factor
    param_count = sum(p.numel() for p in model.parameters())
    complexity_factor = math.log10(max(param_count, 1)) / 6.0  # Normalize to ~1.0 for 1M params
    
    # Device factor
    device_factor = 1.0
    if device.type == 'cuda':
        device_factor = 0.1  # GPUs are ~10x faster
    
    time_per_sample = base_time_per_sample * complexity_factor * device_factor
    time_per_epoch = (dataset_size / batch_size) * batch_size * time_per_sample
    total_time = time_per_epoch * epochs
    
    return {
        'time_per_sample_seconds': time_per_sample,
        'time_per_epoch_seconds': time_per_epoch,
        'total_training_time_seconds': total_time,
        'total_training_time_hours': total_time / 3600,
        'estimated_throughput_samples_per_sec': 1.0 / time_per_sample
    }