"""Advanced memory and compute optimization for neural operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import gc
import psutil
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import weakref
import numpy as np


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""
    enable_gradient_checkpointing: bool = True
    enable_activation_checkpointing: bool = True
    enable_memory_mapping: bool = True
    enable_cpu_offloading: bool = True
    enable_smart_caching: bool = True
    max_memory_usage_gb: float = 16.0
    memory_warning_threshold: float = 0.85
    activation_cache_size: int = 1000
    gradient_accumulation_steps: int = 1
    offload_threshold_mb: int = 1000


class MemoryTracker:
    """Track and optimize memory usage throughout training."""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.memory_log = []
        self.peak_memory = 0
        self.current_allocations = {}
        self.allocation_history = defaultdict(list)
        self.optimization_stats = defaultdict(int)
        
    def track_allocation(self, name: str, size_bytes: int):
        """Track memory allocation."""
        self.current_allocations[name] = size_bytes
        self.allocation_history[name].append({
            'timestamp': time.time(),
            'size_bytes': size_bytes,
            'action': 'allocate'
        })
        
        # Update peak memory
        total_memory = sum(self.current_allocations.values())
        if total_memory > self.peak_memory:
            self.peak_memory = total_memory
    
    def track_deallocation(self, name: str):
        """Track memory deallocation."""
        if name in self.current_allocations:
            size_bytes = self.current_allocations[name]
            del self.current_allocations[name]
            
            self.allocation_history[name].append({
                'timestamp': time.time(),
                'size_bytes': size_bytes,
                'action': 'deallocate'
            })
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_total = sum(self.current_allocations.values())
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        
        return {
            'current_allocations_bytes': current_total,
            'current_allocations_gb': current_total / (1024**3),
            'peak_memory_bytes': self.peak_memory,
            'peak_memory_gb': self.peak_memory / (1024**3),
            'system_memory_percent': system_memory.percent,
            'gpu_memory': gpu_memory,
            'active_allocations': len(self.current_allocations),
            'optimization_stats': dict(self.optimization_stats)
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on tracking data."""
        suggestions = []
        stats = self.get_memory_stats()
        
        if stats['current_allocations_gb'] > self.config.max_memory_usage_gb * 0.8:
            suggestions.append("Enable gradient checkpointing to reduce memory usage")
            suggestions.append("Consider reducing batch size")
            suggestions.append("Enable CPU offloading for large tensors")
        
        if torch.cuda.is_available():
            gpu_usage = stats['gpu_memory'].get('allocated', 0) / (1024**3)
            if gpu_usage > 8:  # Assuming 16GB GPU
                suggestions.append("Enable mixed precision training")
                suggestions.append("Use gradient accumulation instead of large batches")
        
        # Check for memory leaks
        large_allocations = [name for name, size in self.current_allocations.items() 
                           if size > 100 * 1024 * 1024]  # > 100MB
        if len(large_allocations) > 10:
            suggestions.append("Potential memory leak detected - review large allocations")
        
        return suggestions


class SmartTensorCache:
    """Intelligent tensor caching with memory pressure awareness."""
    
    def __init__(self, max_size_mb: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.last_access = {}
        self.current_size = 0
        
    def _compute_cache_key(self, tensor: torch.Tensor, operation: str) -> str:
        """Compute cache key for tensor."""
        return f"{operation}_{tensor.shape}_{tensor.dtype}_{tensor.device}_{hash(tensor.data_ptr())}"
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        if key in self.cache:
            self.access_counts[key] += 1
            self.last_access[key] = time.time()
            
            # Move to end (LRU)
            tensor = self.cache[key]
            del self.cache[key]
            self.cache[key] = tensor
            
            return tensor
        return None
    
    def put(self, key: str, tensor: torch.Tensor):
        """Put tensor in cache with intelligent eviction."""
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Check if tensor is too large
        if tensor_size > self.max_size_bytes * 0.5:
            return  # Don't cache very large tensors
        
        # Evict if necessary
        while self.current_size + tensor_size > self.max_size_bytes and self.cache:
            self._evict_least_valuable()
        
        # Add to cache
        self.cache[key] = tensor.clone().detach()
        self.current_size += tensor_size
        self.access_counts[key] = 1
        self.last_access[key] = time.time()
    
    def _evict_least_valuable(self):
        """Evict least valuable tensor based on access patterns."""
        if not self.cache:
            return
        
        # Score based on recency and frequency
        scores = {}
        current_time = time.time()
        
        for key in self.cache:
            recency = current_time - self.last_access.get(key, current_time)
            frequency = self.access_counts.get(key, 1)
            
            # Lower score = less valuable
            scores[key] = frequency / (1 + recency)
        
        # Evict lowest scoring tensor
        least_valuable = min(scores, key=scores.get)
        tensor = self.cache[least_valuable]
        tensor_size = tensor.numel() * tensor.element_size()
        
        del self.cache[least_valuable]
        self.current_size -= tensor_size
        
        if least_valuable in self.access_counts:
            del self.access_counts[least_valuable]
        if least_valuable in self.last_access:
            del self.last_access[least_valuable]


class CPUOffloadManager:
    """Manage CPU offloading for large tensors."""
    
    def __init__(self, offload_threshold_mb: int = 1000):
        self.offload_threshold_bytes = offload_threshold_mb * 1024 * 1024
        self.offloaded_tensors = {}
        self.offload_threads = {}
        self.stats = {'offloaded': 0, 'restored': 0, 'bytes_saved': 0}
    
    def should_offload(self, tensor: torch.Tensor) -> bool:
        """Determine if tensor should be offloaded to CPU."""
        if not torch.cuda.is_available() or tensor.device == torch.device('cpu'):
            return False
        
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Offload large tensors
        if tensor_size > self.offload_threshold_bytes:
            return True
        
        # Offload when GPU memory is high
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if gpu_memory_usage > 0.8 and tensor_size > 50 * 1024 * 1024:  # > 50MB
                return True
        
        return False
    
    def offload_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Offload tensor to CPU and return placeholder."""
        if not self.should_offload(tensor):
            return tensor
        
        # Create CPU copy
        cpu_tensor = tensor.cpu()
        
        # Store metadata for restoration
        self.offloaded_tensors[name] = {
            'tensor': cpu_tensor,
            'original_device': tensor.device,
            'shape': tensor.shape,
            'dtype': tensor.dtype
        }
        
        # Update stats
        self.stats['offloaded'] += 1
        self.stats['bytes_saved'] += tensor.numel() * tensor.element_size()
        
        # Return a placeholder with minimal GPU memory
        placeholder = torch.zeros(1, device=tensor.device, dtype=tensor.dtype)
        placeholder._offloaded_name = name
        
        return placeholder
    
    def restore_tensor(self, placeholder: torch.Tensor) -> torch.Tensor:
        """Restore offloaded tensor from CPU."""
        if not hasattr(placeholder, '_offloaded_name'):
            return placeholder
        
        name = placeholder._offloaded_name
        if name not in self.offloaded_tensors:
            return placeholder
        
        # Restore from CPU
        offload_info = self.offloaded_tensors[name]
        restored = offload_info['tensor'].to(offload_info['original_device'])
        
        # Cleanup
        del self.offloaded_tensors[name]
        self.stats['restored'] += 1
        
        return restored
    
    async def async_offload(self, tensor: torch.Tensor, name: str):
        """Asynchronously offload tensor to CPU."""
        def offload_worker():
            self.offload_tensor(tensor, name)
        
        thread = threading.Thread(target=offload_worker, daemon=True)
        self.offload_threads[name] = thread
        thread.start()


class ActivationCheckpointing:
    """Advanced activation checkpointing for memory efficiency."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = SmartTensorCache(cache_size)
        self.checkpoint_functions = {}
        self.recompute_stats = {'hits': 0, 'misses': 0, 'recomputes': 0}
    
    def checkpoint_function(self, func: Callable, *args, **kwargs):
        """Checkpoint a function and its activations."""
        # Create unique key for this computation
        key = self._create_checkpoint_key(func, args, kwargs)
        
        # Check cache first
        cached_result = self.cache.get(key)
        if cached_result is not None:
            self.recompute_stats['hits'] += 1
            return cached_result
        
        # Compute and cache result
        self.recompute_stats['misses'] += 1
        
        # Use torch checkpointing for gradient computation
        if torch.is_grad_enabled():
            result = checkpoint(func, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        # Cache the result
        if isinstance(result, torch.Tensor):
            self.cache.put(key, result)
        
        return result
    
    def _create_checkpoint_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Create unique key for checkpointing."""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Hash input tensors
        input_hashes = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_hashes.append(f"{arg.shape}_{arg.dtype}_{hash(arg.data_ptr())}")
            else:
                input_hashes.append(str(hash(str(arg))))
        
        return f"{func_name}_{'_'.join(input_hashes)}"


class GradientCompressionManager:
    """Manage gradient compression for memory efficiency."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.compressed_gradients = {}
        self.compression_stats = {'original_size': 0, 'compressed_size': 0}
    
    def compress_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Compress model gradients."""
        compressed = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Store original size
                original_size = grad.numel() * grad.element_size()
                self.compression_stats['original_size'] += original_size
                
                # Apply top-k sparsification
                k = max(1, int(grad.numel() * self.compression_ratio))
                
                # Get top-k absolute values
                grad_flat = grad.view(-1)
                _, indices = torch.topk(torch.abs(grad_flat), k)
                
                # Store compressed representation
                compressed[name] = {
                    'values': grad_flat[indices],
                    'indices': indices,
                    'shape': grad.shape,
                    'dtype': grad.dtype
                }
                
                # Update compression stats
                compressed_size = (len(indices) * grad.element_size() + 
                                 len(indices) * 8)  # 8 bytes per index
                self.compression_stats['compressed_size'] += compressed_size
        
        return compressed
    
    def decompress_gradients(self, compressed: Dict[str, Any], model: nn.Module):
        """Decompress and apply gradients to model."""
        for name, param in model.named_parameters():
            if name in compressed and param.grad is not None:
                comp_data = compressed[name]
                
                # Reconstruct sparse gradient
                grad_flat = torch.zeros(param.numel(), 
                                      dtype=comp_data['dtype'], 
                                      device=param.device)
                grad_flat[comp_data['indices']] = comp_data['values']
                
                # Reshape and apply
                param.grad.data = grad_flat.view(comp_data['shape'])
    
    def get_compression_ratio(self) -> float:
        """Get actual compression ratio achieved."""
        if self.compression_stats['original_size'] == 0:
            return 0.0
        
        return (self.compression_stats['compressed_size'] / 
                self.compression_stats['original_size'])


class AdvancedMemoryOptimizer:
    """Comprehensive memory optimization for neural operators."""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.memory_tracker = MemoryTracker(config)
        self.tensor_cache = SmartTensorCache(config.activation_cache_size)
        self.cpu_offloader = CPUOffloadManager(config.offload_threshold_mb)
        self.activation_checkpointing = ActivationCheckpointing(config.activation_cache_size)
        self.gradient_compressor = GradientCompressionManager()
        
        # Monitoring
        self.optimization_active = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        self.monitor_thread.start()
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive memory optimizations to model."""
        if self.config.enable_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        if self.config.enable_cpu_offloading:
            model = self._apply_cpu_offloading(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Wrap transformer layers with checkpointing
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return self.activation_checkpointing.checkpoint_function(
                        original_forward, *args, **kwargs
                    )
                
                module.forward = checkpointed_forward
        
        return model
    
    def _apply_cpu_offloading(self, model: nn.Module) -> nn.Module:
        """Apply CPU offloading to large model parameters."""
        for name, param in model.named_parameters():
            if self.cpu_offloader.should_offload(param):
                # Create offloaded parameter
                offloaded_param = self.cpu_offloader.offload_tensor(param, name)
                
                # Replace parameter with placeholder
                # Note: This is a simplified version - production would need 
                # more sophisticated parameter replacement
                setattr(model, name.replace('.', '_'), offloaded_param)
        
        return model
    
    def optimize_forward_pass(self, forward_func: Callable, *args, **kwargs):
        """Optimize forward pass with memory management."""
        # Pre-forward memory cleanup
        self._cleanup_memory()
        
        # Track memory before forward pass
        memory_before = self._get_current_memory()
        
        try:
            # Execute forward pass with optimizations
            if self.config.enable_activation_checkpointing:
                result = self.activation_checkpointing.checkpoint_function(
                    forward_func, *args, **kwargs
                )
            else:
                result = forward_func(*args, **kwargs)
            
            # Track memory after forward pass
            memory_after = self._get_current_memory()
            memory_used = memory_after - memory_before
            
            self.memory_tracker.track_allocation('forward_pass', memory_used)
            
            return result
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Handle OOM with recovery strategies
                return self._handle_oom_recovery(forward_func, *args, **kwargs)
            else:
                raise
    
    def _handle_oom_recovery(self, forward_func: Callable, *args, **kwargs):
        """Handle out-of-memory errors with recovery strategies."""
        print("Out of memory detected - applying recovery strategies...")
        
        # Strategy 1: Clear caches
        self._cleanup_memory()
        torch.cuda.empty_cache()
        
        try:
            return forward_func(*args, **kwargs)
        except RuntimeError:
            pass
        
        # Strategy 2: Enable gradient checkpointing
        print("Enabling emergency gradient checkpointing...")
        return checkpoint(forward_func, *args, **kwargs)
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear tensor cache if memory pressure is high
        memory_stats = self.memory_tracker.get_memory_stats()
        if memory_stats['current_allocations_gb'] > self.config.max_memory_usage_gb * 0.8:
            self.tensor_cache.cache.clear()
            self.tensor_cache.current_size = 0
    
    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss
    
    def _memory_monitor(self):
        """Background memory monitoring."""
        while self.optimization_active:
            try:
                memory_stats = self.memory_tracker.get_memory_stats()
                
                # Check for memory pressure
                if memory_stats['current_allocations_gb'] > self.config.max_memory_usage_gb * self.config.memory_warning_threshold:
                    print(f"Memory warning: {memory_stats['current_allocations_gb']:.1f}GB used")
                    suggestions = self.memory_tracker.suggest_optimizations()
                    for suggestion in suggestions:
                        print(f"  - {suggestion}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Memory monitor error: {e}")
                time.sleep(10)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'memory_stats': self.memory_tracker.get_memory_stats(),
            'cache_stats': {
                'size_mb': self.tensor_cache.current_size / (1024 * 1024),
                'entries': len(self.tensor_cache.cache),
                'hit_rate': len(self.tensor_cache.access_counts) / max(1, sum(self.tensor_cache.access_counts.values()))
            },
            'offload_stats': self.cpu_offloader.stats,
            'checkpoint_stats': self.activation_checkpointing.recompute_stats,
            'compression_stats': {
                'ratio': self.gradient_compressor.get_compression_ratio(),
                'original_size_mb': self.gradient_compressor.compression_stats['original_size'] / (1024 * 1024),
                'compressed_size_mb': self.gradient_compressor.compression_stats['compressed_size'] / (1024 * 1024)
            }
        }
    
    def shutdown(self):
        """Shutdown memory optimizer."""
        self.optimization_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)