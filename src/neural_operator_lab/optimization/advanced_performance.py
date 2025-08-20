# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.


# SECURITY_DISABLED: # Safe operations mapping to replace eval(...)  # eval() disabled for security
SAFE_OPERATIONS = {
    'torch.cuda.is_available()': lambda: torch.cuda.is_available(),
    'torch.cuda.device_count()': lambda: torch.cuda.device_count(),
    'psutil.cpu_count()': lambda: psutil.cpu_count(),
    'psutil.virtual_memory().available': lambda: psutil.virtual_memory().available,
}
"""Advanced Performance Optimization Suite

Comprehensive performance optimization including GPU kernels, memory management,
mixed precision, and automatic performance tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue
from functools import wraps
import math
import warnings

# Try to import advanced optimizations
try:
    import torch.utils.cpp_extension
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False

try:
    from torch.jit import script, trace
    HAS_JIT = True
except ImportError:
    HAS_JIT = False

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True
    kernel_fusion: bool = True
    async_data_loading: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    compile_model: bool = True
    optimization_level: str = "O2"  # O0, O1, O2, O3
    cache_size_mb: int = 1024
    
    # Auto-tuning parameters
    auto_tune_batch_size: bool = True
    auto_tune_learning_rate: bool = False
    performance_target_ms: float = 100.0  # Target inference time
    
    # Memory management
    memory_fraction: float = 0.9  # GPU memory fraction to use
    empty_cache_frequency: int = 100  # Empty cache every N steps


class MemoryPool:
    """Advanced memory pool for efficient tensor allocation."""
    
    def __init__(self, max_size_mb: int = 1024, device: str = "cuda"):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.device = device
        self.pools: Dict[Tuple[torch.dtype, Tuple[int, ...]], List[torch.Tensor]] = {}
        self.allocated_bytes = 0
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one."""
        key = (dtype, shape)
        
        with self._lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()  # Clear previous data
                return tensor
        
        # Allocate new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        tensor_bytes = tensor.element_size() * tensor.numel()
        
        with self._lock:
            self.allocated_bytes += tensor_bytes
        
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to the pool."""
        if tensor.device.type != self.device.split(':')[0]:
            return  # Don't pool tensors from different devices
        
        key = (tensor.dtype, tuple(tensor.shape))
        tensor_bytes = tensor.element_size() * tensor.numel()
        
        with self._lock:
            # Check if we have space
            if self.allocated_bytes + tensor_bytes > self.max_size_bytes:
                return  # Don't pool if we're at capacity
            
            if key not in self.pools:
                self.pools[key] = []
            
            self.pools[key].append(tensor.detach())
    
    def clear(self):
        """Clear all pooled tensors."""
        with self._lock:
            self.pools.clear()
            self.allocated_bytes = 0
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FusedOperations:
    """Collection of fused operations for better performance."""
    
    @staticmethod
    def fused_layer_norm_activation(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str = "gelu",
        eps: float = 1e-5
    ) -> torch.Tensor:
        """Fused layer normalization + activation."""
        # Layer norm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + eps)
        scaled = normalized * weight + bias
        
        # Activation
        if activation == "gelu":
            return F.gelu(scaled)
        elif activation == "relu":
            return F.relu(scaled)
        elif activation == "swish":
            return scaled * torch.sigmoid(scaled)
        else:
            return scaled
    
    @staticmethod
    def fused_linear_bias_activation(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str = "gelu"
    ) -> torch.Tensor:
        """Fused linear + bias + activation."""
        linear_out = F.linear(x, weight, bias)
        
        if activation == "gelu":
            return F.gelu(linear_out)
        elif activation == "relu":
            return F.relu(linear_out)
        elif activation == "swish":
            return linear_out * torch.sigmoid(linear_out)
        else:
            return linear_out
    
    @staticmethod
    def fused_attention_computation(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """Memory-efficient fused attention computation."""
        batch_size, seq_len, head_dim = query.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # Use Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                scale=scale
            )
        
        # Fallback to standard attention with memory optimization
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        if dropout_p > 0.0:
            attention_probs = F.dropout(attention_probs, p=dropout_p, training=True)
        
        return torch.matmul(attention_probs, value)


class AdaptiveOptimizer:
    """Adaptive optimizer that automatically tunes hyperparameters."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_steps: int = 1000,
        target_loss: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.target_loss = target_loss
        
        # Performance tracking
        self.loss_history = []
        self.lr_history = []
        self.step_count = 0
        
        # Auto-tuning state
        self.best_lr = None
        self.best_loss = float('inf')
        self.lr_search_active = False
    
    def step(self, loss: Optional[float] = None):
        """Optimizer step with adaptive learning rate."""
        self.step_count += 1
        
        if loss is not None:
            self.loss_history.append(loss)
            
            # Update best metrics
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_lr = self.optimizer.param_groups[0]['lr']
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            warmup_lr = self.best_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        self.optimizer.step()
        
        # Schedule learning rate
        if self.scheduler is not None and self.step_count > self.warmup_steps:
            self.scheduler.step()
        
        # Record current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
    
    def auto_tune_lr(self, model: nn.Module, data_loader, num_steps: int = 100):
        """Automatically find optimal learning rate."""
        original_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # Learning rate range test
        lrs = torch.logspace(-6, -1, num_steps)
        losses = []
        
        model.train()
        for i, lr in enumerate(lrs):
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr.item()
            
            # Training step
            try:
                batch = next(iter(data_loader))
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch['input']
                    targets = batch['target']
                
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
            except Exception as e:
                # Stop if we hit numerical instabilities
                break
        
        # Find optimal learning rate (steepest descent point)
        if len(losses) > 10:
            gradients = np.gradient(losses)
            optimal_idx = np.argmin(gradients[:len(gradients)//2])  # Look in first half
            optimal_lr = lrs[optimal_idx].item()
            
            # Set optimal learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = optimal_lr
            
            self.best_lr = optimal_lr
            return optimal_lr
        
        # Restore original learning rates if auto-tuning failed
        for param_group, original_lr in zip(self.optimizer.param_groups, original_lrs):
            param_group['lr'] = original_lr
        
        return original_lrs[0]


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.memory_pool = MemoryPool(
            max_size_mb=self.config.cache_size_mb,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.fused_ops = FusedOperations()
        
        # Mixed precision setup
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Performance metrics
        self.metrics = {
            'forward_times': [],
            'backward_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
        # Step counter for cache management
        self.step_counter = 0
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive model optimizations."""
        
        # Convert to optimized data types
        if self.config.mixed_precision:
            model = model.half()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing(model)
        
        # Compile model if requested
        if self.config.compile_model and HAS_JIT:
            try:
                # Try to script the model
                model = torch.jit.script(model)
            except Exception:
                try:
                    # Fallback to tracing (requires sample input)
                    dummy_input = torch.randn(1, 64, 64, device=next(model.parameters()).device)
                    model = torch.jit.trace(model, dummy_input)
                except Exception:
                    warnings.warn("Model compilation failed, using eager mode")
        
        return model
    
    def _enable_gradient_checkpointing(self, model: nn.Module):
        """Enable gradient checkpointing for memory efficiency."""
        def checkpoint_wrapper(module):
            def forward_wrapper(*args, **kwargs):
                if self.training:
                    return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
                else:
                    return module(*args, **kwargs)
            return forward_wrapper
        
        # Apply to attention and MLP layers
        for name, module in model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['attention', 'mlp', 'feed_forward']):
                module.forward = checkpoint_wrapper(module.forward)
    
    def optimize_data_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        num_workers: int = 4,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """Create optimized data loader."""
        
        # Auto-tune batch size if requested
        if self.config.auto_tune_batch_size:
            batch_size = self._find_optimal_batch_size(dataset, batch_size)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
            **kwargs
        )
    
    def _find_optimal_batch_size(self, dataset, initial_batch_size: int) -> int:
        """Find optimal batch size through binary search."""
        min_batch_size = 1
        max_batch_size = initial_batch_size * 4
        
        # Binary search for largest working batch size
        while min_batch_size < max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size + 1) // 2
            
            try:
                # Test if this batch size works
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=mid_batch_size)
                batch = next(iter(test_loader))
                
                # Try to allocate memory for this batch
                if isinstance(batch, (list, tuple)):
                    test_tensor = batch[0]
                else:
                    test_tensor = batch['input']
                
                # Simple memory test
                _ = test_tensor.to(self.memory_pool.device)
                
                min_batch_size = mid_batch_size
                
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                max_batch_size = mid_batch_size - 1
            
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return max(1, min_batch_size)
    
    @contextmanager
    def optimized_forward(self, enable_autocast: bool = True):
        """Context manager for optimized forward pass."""
        
        start_time = time.perf_counter()
        
        # Memory management
        if self.step_counter % self.config.empty_cache_frequency == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        try:
            if enable_autocast and self.config.mixed_precision:
                with autocast():
                    yield
            else:
                yield
        finally:
            forward_time = time.perf_counter() - start_time
            self.metrics['forward_times'].append(forward_time)
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.metrics['memory_usage'].append(memory_mb)
            
            self.step_counter += 1
    
    def optimized_backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Optimized backward pass with gradient scaling."""
        
        start_time = time.perf_counter()
        
        if self.scaler is not None:
            # Mixed precision backward
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                max_norm=1.0
            )
            
            # Optimizer step with scaling
            self.scaler.step(optimizer)
            self.scaler.update()
            
        else:
            # Standard backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']],
                max_norm=1.0
            )
            
            optimizer.step()
        
        backward_time = time.perf_counter() - start_time
        self.metrics['backward_times'].append(backward_time)
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """Comprehensive model benchmarking."""
        
        device = next(model.parameters()).device
        # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark forward pass
        forward_times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with self.optimized_forward():
                    output = model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                forward_time = time.perf_counter() - start_time
                forward_times.append(forward_time)
                
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    memory_usage.append(memory_mb)
        
        # Calculate statistics
        forward_times = np.array(forward_times)
        
        results = {
            'mean_forward_time_ms': np.mean(forward_times) * 1000,
            'std_forward_time_ms': np.std(forward_times) * 1000,
            'min_forward_time_ms': np.min(forward_times) * 1000,
            'max_forward_time_ms': np.max(forward_times) * 1000,
            'throughput_samples_per_sec': input_shape[0] / np.mean(forward_times),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        if memory_usage:
            results.update({
                'mean_memory_usage_mb': np.mean(memory_usage),
                'peak_memory_usage_mb': np.max(memory_usage)
            })
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        report = {
            'optimization_config': {
                'mixed_precision': self.config.mixed_precision,
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'memory_efficient_attention': self.config.memory_efficient_attention,
                'kernel_fusion': self.config.kernel_fusion
            },
            'performance_metrics': {}
        }
        
        # Calculate statistics for collected metrics
        for metric_name, values in self.metrics.items():
            if values:
                report['performance_metrics'][metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        # Memory pool statistics
        report['memory_pool'] = {
            'allocated_bytes': self.memory_pool.allocated_bytes,
            'pools_count': len(self.memory_pool.pools),
            'total_tensors_pooled': sum(len(pool) for pool in self.memory_pool.pools.values())
        }
        
        return report
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        for metric_list in self.metrics.values():
            metric_list.clear()
        
        self.memory_pool.clear()
        self.step_counter = 0


# Decorator for automatic performance optimization
def optimize_performance(config: Optional[PerformanceConfig] = None):
    """Decorator to automatically apply performance optimizations."""
    
    def decorator(func):
        optimizer = PerformanceOptimizer(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply optimizations if model is provided
            for i, arg in enumerate(args):
                if isinstance(arg, nn.Module):
                    args = list(args)
                    args[i] = optimizer.optimize_model(arg)
                    args = tuple(args)
                    break
            
            # Execute function with performance monitoring
            with optimizer.optimized_forward():
                return func(*args, **kwargs)
        
        # Attach optimizer to function for access to metrics
        wrapper.performance_optimizer = optimizer
        return wrapper
    
    return decorator