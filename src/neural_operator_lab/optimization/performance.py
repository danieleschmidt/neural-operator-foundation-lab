"""Performance optimization utilities for neural operators."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, OrderedDict
import functools
import weakref

from ..base import NeuralOperatorBase


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Memory optimizations
    enable_memory_efficient: bool = True
    gradient_checkpointing: bool = True
    pin_memory: bool = True
    empty_cache_frequency: int = 100  # Every N batches
    
    # Computation optimizations
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile
    fusion_enabled: bool = True
    
    # Data loading optimizations
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 1000
    cache_policy: str = 'lru'  # lru, lfu, fifo
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_schedule: str = 'wait=1,warmup=1,active=3,repeat=1'


class AdaptiveCache:
    """Adaptive caching system with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, policy: str = 'lru'):
        self.max_size = max_size
        self.policy = policy.lower()
        self.cache: OrderedDict = OrderedDict()
        self.access_counts: Dict = defaultdict(int)
        self.access_times: Dict = {}
        self._lock = threading.RLock()
        
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Update access patterns
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                
                # Move to end for LRU
                if self.policy == 'lru':
                    self.cache.move_to_end(key)
                
                self.hit_count += 1
                return self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                if self.policy == 'lru':
                    self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    self._evict()
                
                self.cache[key] = value
                self.access_counts[key] = 1
                self.access_times[key] = time.time()
    
    def _evict(self):
        """Evict item based on policy."""
        if not self.cache:
            return
        
        if self.policy == 'lru':
            # Remove oldest (first item)
            key = next(iter(self.cache))
        elif self.policy == 'lfu':
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.access_counts[k])
        elif self.policy == 'fifo':
            # Remove first in
            key = next(iter(self.cache))
        else:
            # Default to LRU
            key = next(iter(self.cache))
        
        del self.cache[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'policy': self.policy
        }


class ModelCache:
    """Cache for model computations and intermediate results."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self.cache: Dict[str, Any] = {}
        self.memory_sizes: Dict[str, int] = {}
        self.access_order: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _get_tensor_memory(self, tensor: torch.Tensor) -> int:
        """Get memory usage of tensor in bytes."""
        return tensor.element_size() * tensor.numel()
    
    def _get_key_memory(self, key: str) -> int:
        """Get memory usage of cached item."""
        if key not in self.cache:
            return 0
        
        item = self.cache[key]
        if torch.is_tensor(item):
            return self._get_tensor_memory(item)
        elif isinstance(item, (list, tuple)):
            return sum(self._get_tensor_memory(t) for t in item if torch.is_tensor(t))
        elif isinstance(item, dict):
            return sum(self._get_tensor_memory(v) for v in item.values() if torch.is_tensor(v))
        else:
            # Rough estimate for other objects
            return 1024  # 1KB default
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                self.access_order.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with memory management."""
        with self._lock:
            # Calculate memory requirement
            item_memory = 0
            if torch.is_tensor(value):
                item_memory = self._get_tensor_memory(value)
            elif isinstance(value, (list, tuple)):
                item_memory = sum(self._get_tensor_memory(t) for t in value if torch.is_tensor(t))
            elif isinstance(value, dict):
                item_memory = sum(self._get_tensor_memory(v) for v in value.values() if torch.is_tensor(v))
            else:
                item_memory = 1024  # Default estimate
            
            # Skip if item is too large
            if item_memory > self.max_memory_bytes:
                return False
            
            # Evict if necessary
            while (self.current_memory + item_memory > self.max_memory_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Remove existing item if updating
            if key in self.cache:
                old_memory = self.memory_sizes[key]
                self.current_memory -= old_memory
            
            # Add new item
            self.cache[key] = value
            self.memory_sizes[key] = item_memory
            self.current_memory += item_memory
            self.access_order[key] = True
            
            return True
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_order:
            return
        
        lru_key = next(iter(self.access_order))
        memory_freed = self.memory_sizes.get(lru_key, 0)
        
        del self.cache[lru_key]
        del self.memory_sizes[lru_key]
        del self.access_order[lru_key]
        
        self.current_memory -= memory_freed
        self.evictions += 1
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self.cache.clear()
            self.memory_sizes.clear()
            self.access_order.clear()
            self.current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'items': len(self.cache),
            'memory_mb': self.current_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_usage': self.current_memory / self.max_memory_bytes,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Caches
        self.adaptive_cache = AdaptiveCache(
            max_size=self.config.cache_size,
            policy=self.config.cache_policy
        ) if self.config.enable_cache else None
        
        self.model_cache = ModelCache(max_memory_mb=512)
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.optimization_history = []
        
        # Auto-scaling parameters
        self.batch_size_history = []
        self.memory_usage_history = []
        self.throughput_history = []
        
        # Optimization state
        self.current_optimizations = set()
        
    def optimize_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Apply model-level optimizations."""
        optimized_model = model
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            optimized_model = self._enable_gradient_checkpointing(optimized_model)
        
        # Apply torch.compile if available and enabled
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                optimized_model = torch.compile(optimized_model)
                self.logger.info("Applied torch.compile optimization")
                self.current_optimizations.add('torch_compile')
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
        
        # Enable tensor fusion optimizations
        if self.config.fusion_enabled:
            optimized_model = self._enable_fusion_optimizations(optimized_model)
        
        return optimized_model
    
    def _enable_gradient_checkpointing(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Enable gradient checkpointing to save memory."""
        def checkpoint_wrapper(module):
            if hasattr(module, '_checkpoint_wrapper'):
                return module
            
            class CheckpointWrapper(nn.Module):
                def __init__(self, wrapped_module):
                    super().__init__()
                    self.wrapped_module = wrapped_module
                    self._checkpoint_wrapper = True
                
                def forward(self, *args, **kwargs):
                    if self.training:
                        return torch.utils.checkpoint.checkpoint(
                            self.wrapped_module, *args, **kwargs
                        )
                    else:
                        return self.wrapped_module(*args, **kwargs)
            
            return CheckpointWrapper(module)
        
        # Apply checkpointing to large modules
        for name, module in model.named_modules():
            if any(layer_type in str(type(module)) for layer_type in 
                   ['Linear', 'Conv', 'MultiHeadAttention', 'FNOBlock']):
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, module_name, checkpoint_wrapper(module))
                else:
                    setattr(model, module_name, checkpoint_wrapper(module))
        
        self.logger.info("Enabled gradient checkpointing")
        self.current_optimizations.add('gradient_checkpointing')
        return model
    
    def _enable_fusion_optimizations(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Enable tensor fusion optimizations."""
        # Enable cudnn benchmarking for consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.logger.info("Enabled cuDNN benchmarking")
        
        # Enable tensor cores if available
        if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.logger.info("Enabled TensorFloat-32 (TF32) optimizations")
        
        self.current_optimizations.add('tensor_fusion')
        return model
    
    def optimize_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Create optimized dataloader."""
        # Extract current parameters
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        shuffle = dataloader.dataset_kind == 'Iterable' or getattr(dataloader, 'shuffle', False)
        
        # Create optimized dataloader
        optimized_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else 2,
            drop_last=True  # For consistent batch sizes
        )
        
        self.logger.info(f"Optimized dataloader: workers={self.config.num_workers}, "
                        f"pin_memory={self.config.pin_memory}, "
                        f"persistent_workers={self.config.persistent_workers}")
        
        return optimized_loader
    
    def adaptive_batch_size_scaling(self, 
                                  trainer,
                                  target_memory_usage: float = 0.85,
                                  target_throughput_increase: float = 1.1) -> int:
        """Adaptively scale batch size based on performance metrics."""
        current_batch_size = trainer.config.batch_size
        
        # Collect performance metrics
        if torch.cuda.is_available():
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_used = torch.cuda.memory_allocated()
            memory_usage_ratio = memory_used / memory_total
        else:
            memory_usage_ratio = 0.5  # Conservative estimate for CPU
        
        # Calculate throughput (samples per second)
        if hasattr(trainer, 'performance_stats'):
            recent_times = trainer.performance_stats.get('batch_time', [])[-10:]
            if recent_times:
                avg_batch_time = sum(recent_times) / len(recent_times)
                throughput = current_batch_size / avg_batch_time
            else:
                throughput = 0
        else:
            throughput = 0
        
        # Store history
        self.batch_size_history.append(current_batch_size)
        self.memory_usage_history.append(memory_usage_ratio)
        self.throughput_history.append(throughput)
        
        # Keep history limited
        max_history = 100
        if len(self.batch_size_history) > max_history:
            self.batch_size_history = self.batch_size_history[-max_history:]
            self.memory_usage_history = self.memory_usage_history[-max_history:]
            self.throughput_history = self.throughput_history[-max_history:]
        
        # Decision making
        new_batch_size = current_batch_size
        
        if len(self.memory_usage_history) >= 5:
            avg_memory_usage = sum(self.memory_usage_history[-5:]) / 5
            
            # Increase batch size if memory usage is low and we want more throughput
            if avg_memory_usage < target_memory_usage * 0.7 and throughput > 0:
                if len(self.throughput_history) >= 5:
                    recent_throughput = sum(self.throughput_history[-3:]) / 3
                    older_throughput = sum(self.throughput_history[-6:-3]) / 3 if len(self.throughput_history) >= 6 else recent_throughput
                    
                    if recent_throughput >= older_throughput * target_throughput_increase:
                        new_batch_size = int(current_batch_size * 1.2)
                        self.logger.info(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
            
            # Decrease batch size if memory usage is too high
            elif avg_memory_usage > target_memory_usage:
                new_batch_size = max(1, int(current_batch_size * 0.8))
                self.logger.info(f"Decreasing batch size from {current_batch_size} to {new_batch_size}")
        
        # Update trainer config
        trainer.config.batch_size = new_batch_size
        
        return new_batch_size
    
    def memory_efficient_forward(self, model: NeuralOperatorBase, 
                                inputs: torch.Tensor,
                                grid: Optional[torch.Tensor] = None,
                                chunk_size: Optional[int] = None) -> torch.Tensor:
        """Memory-efficient forward pass with chunking."""
        if chunk_size is None:
            batch_size = inputs.size(0)
            # Auto-determine chunk size based on available memory
            if torch.cuda.is_available():
                memory_total = torch.cuda.get_device_properties(0).total_memory
                memory_free = memory_total - torch.cuda.memory_allocated()
                # Use rough heuristic: 1GB per 8 samples
                chunk_size = max(1, min(batch_size, int(memory_free / (1024**3) * 8)))
            else:
                chunk_size = min(batch_size, 16)  # Conservative for CPU
        
        batch_size = inputs.size(0)
        if batch_size <= chunk_size:
            # No chunking needed
            return model(inputs, grid)
        
        # Process in chunks
        outputs = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_inputs = inputs[i:end_idx]
            chunk_grid = grid[i:end_idx] if grid is not None else None
            
            with torch.cuda.amp.autocast() if self.config.mixed_precision else torch.no_grad():
                chunk_output = model(chunk_inputs, chunk_grid)
                outputs.append(chunk_output.cpu() if chunk_output.is_cuda else chunk_output)
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % self.config.empty_cache_frequency == 0:
                torch.cuda.empty_cache()
        
        # Concatenate results
        final_output = torch.cat(outputs, dim=0)
        return final_output.to(inputs.device)
    
    def cached_inference(self, model: NeuralOperatorBase,
                        inputs: torch.Tensor,
                        grid: Optional[torch.Tensor] = None,
                        cache_key: Optional[str] = None) -> torch.Tensor:
        """Cached inference with automatic key generation."""
        if not self.config.enable_cache or self.model_cache is None:
            return model(inputs, grid)
        
        # Generate cache key if not provided
        if cache_key is None:
            input_hash = hash(inputs.data_ptr()) if inputs.data_ptr() != 0 else hash(inputs.cpu().numpy().tobytes())
            grid_hash = hash(grid.data_ptr()) if grid is not None and grid.data_ptr() != 0 else 0
            model_hash = hash(str(model.state_dict().keys()))
            cache_key = f"inference_{input_hash}_{grid_hash}_{model_hash}"
        
        # Check cache
        cached_result = self.model_cache.get(cache_key)
        if cached_result is not None:
            return cached_result.to(inputs.device)
        
        # Compute and cache
        with torch.no_grad():
            result = model(inputs, grid)
            self.model_cache.put(cache_key, result.cpu().clone())
        
        return result
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            'active_optimizations': list(self.current_optimizations),
            'config': {
                'mixed_precision': self.config.mixed_precision,
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'num_workers': self.config.num_workers,
                'pin_memory': self.config.pin_memory,
                'cache_enabled': self.config.enable_cache
            },
            'performance_stats': dict(self.performance_stats),
            'memory_usage_trend': self.memory_usage_history[-20:] if self.memory_usage_history else [],
            'throughput_trend': self.throughput_history[-20:] if self.throughput_history else [],
            'batch_size_trend': self.batch_size_history[-20:] if self.batch_size_history else []
        }
        
        # Add cache statistics
        if self.adaptive_cache:
            report['adaptive_cache_stats'] = self.adaptive_cache.stats()
        
        if self.model_cache:
            report['model_cache_stats'] = self.model_cache.stats()
        
        return report
    
    def profile_training_step(self, 
                            trainer,
                            train_loader: DataLoader,
                            num_steps: int = 10) -> Dict[str, Any]:
        """Profile training step performance."""
        if not self.config.enable_profiling:
            return {}
        
        device = trainer.device
        model = trainer.model
        
        # Warmup
        model.train()
        warmup_steps = 3
        
        data_iter = iter(train_loader)
        
        # Warmup phase
        for _ in range(warmup_steps):
            try:
                batch = next(data_iter)
                if len(batch) == 3:
                    inputs, targets, grid = batch
                    inputs, targets, grid = inputs.to(device), targets.to(device), grid.to(device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    grid = None
                
                trainer._compute_loss(inputs, targets, grid)
                
            except StopIteration:
                data_iter = iter(train_loader)
                continue
        
        # Profiling phase
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        step_times = []
        memory_usage = []
        
        for step in range(num_steps):
            step_start = time.time()
            
            try:
                batch = next(data_iter)
                if len(batch) == 3:
                    inputs, targets, grid = batch
                    inputs, targets, grid = inputs.to(device), targets.to(device), grid.to(device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    grid = None
                
                # Forward pass
                loss_dict = trainer._compute_loss(inputs, targets, grid)
                
                # Record metrics
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
                
                step_times.append(time.time() - step_start)
                
            except StopIteration:
                break
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        throughput = (trainer.config.batch_size * len(step_times)) / total_time if total_time > 0 else 0
        
        profile_results = {
            'avg_step_time': avg_step_time,
            'total_time': total_time,
            'throughput_samples_per_sec': throughput,
            'steps_profiled': len(step_times),
            'step_times': step_times,
        }
        
        if memory_usage:
            profile_results.update({
                'avg_memory_gb': sum(memory_usage) / len(memory_usage),
                'max_memory_gb': max(memory_usage),
                'min_memory_gb': min(memory_usage),
                'memory_usage_trend': memory_usage
            })
        
        return profile_results


def auto_optimize_training(trainer, 
                          dataloader: DataLoader,
                          performance_config: PerformanceConfig = None) -> Tuple[Any, DataLoader]:
    """Automatically optimize trainer and dataloader for performance."""
    optimizer = PerformanceOptimizer(performance_config)
    
    # Optimize model
    trainer.model = optimizer.optimize_model(trainer.model)
    
    # Optimize dataloader
    optimized_dataloader = optimizer.optimize_dataloader(dataloader)
    
    # Set up adaptive batch size scaling
    def batch_size_callback():
        if hasattr(trainer, 'current_epoch') and trainer.current_epoch > 0:
            if trainer.current_epoch % 10 == 0:  # Check every 10 epochs
                optimizer.adaptive_batch_size_scaling(trainer)
    
    # Add performance monitoring callback
    if hasattr(trainer, 'callbacks'):
        from ..training.callbacks import TrainingCallback
        
        class PerformanceCallback(TrainingCallback):
            def __init__(self, performance_optimizer):
                self.optimizer = performance_optimizer
            
            def on_epoch_end(self, trainer, epoch, train_losses, val_losses):
                if epoch > 0 and epoch % 10 == 0:
                    self.optimizer.adaptive_batch_size_scaling(trainer)
                    
                    # Log performance report
                    report = self.optimizer.get_optimization_report()
                    trainer.logger.info(f"Performance report: {report}")
        
        trainer.callbacks.callbacks.append(PerformanceCallback(optimizer))
    
    trainer.logger.info("Applied automatic performance optimizations")
    
    return trainer, optimized_dataloader


class ConcurrentInference:
    """Concurrent inference processor for batch predictions."""
    
    def __init__(self, 
                 model: NeuralOperatorBase,
                 num_workers: int = 4,
                 batch_size: int = 32,
                 device: str = 'auto'):
        self.model = model.eval()
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Thread pool for concurrent processing
        self.executor = None
        
    def __enter__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def predict_batch(self, inputs: torch.Tensor, 
                     grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict single batch."""
        with torch.no_grad():
            inputs = inputs.to(self.device)
            if grid is not None:
                grid = grid.to(self.device)
            
            return self.model(inputs, grid).cpu()
    
    def predict_concurrent(self, 
                          inputs_list: List[torch.Tensor],
                          grids_list: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """Predict multiple batches concurrently."""
        if self.executor is None:
            raise RuntimeError("ConcurrentInference must be used as context manager")
        
        if grids_list is None:
            grids_list = [None] * len(inputs_list)
        
        # Submit all tasks
        futures = []
        for inputs, grid in zip(inputs_list, grids_list):
            future = self.executor.submit(self.predict_batch, inputs, grid)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout
                results.append(result)
            except Exception as e:
                logging.error(f"Concurrent prediction failed: {e}")
                results.append(None)
        
        return results