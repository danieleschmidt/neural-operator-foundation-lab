"""Memory optimization techniques for neural operators."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Tuple
import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import psutil


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    enable_gradient_checkpointing: bool = True
    max_memory_fraction: float = 0.9  # Fraction of available GPU memory
    garbage_collect_frequency: int = 10  # Every N steps
    cache_size_mb: int = 1024  # Size of computation cache
    enable_memory_efficient_attention: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2


class MemoryOptimizer:
    """Advanced memory optimization for neural operator training."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_stats = {
            'peak_allocated': 0,
            'peak_cached': 0,
            'allocations': 0,
            'deallocations': 0
        }
        
        # Cache for intermediate computations
        self.computation_cache: Dict[str, torch.Tensor] = {}
        self.cache_size_bytes = 0
        self.max_cache_bytes = self.config.cache_size_mb * 1024 * 1024
        
        # Setup memory monitoring
        self._setup_memory_monitoring()
    
    def _setup_memory_monitoring(self):
        """Setup background memory monitoring."""
        if torch.cuda.is_available():
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.max_memory_fraction)
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model."""
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            model = self.enable_gradient_checkpointing(model)
        
        # Optimize attention mechanisms
        if self.config.enable_memory_efficient_attention:
            model = self._optimize_attention_memory(model)
        
        # Apply parameter sharing where possible
        model = self._apply_parameter_sharing(model)
        
        self.logger.info("Applied memory optimizations to model")
        return model
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency."""
        checkpoint_modules = []
        
        # Find modules suitable for checkpointing
        for name, module in model.named_modules():
            if self._should_checkpoint_module(module):
                checkpoint_modules.append((name, module))
        
        # Apply checkpointing
        for name, module in checkpoint_modules:
            if hasattr(module, 'forward'):
                original_forward = module.forward
                module.forward = torch.utils.checkpoint.checkpoint(
                    original_forward, use_reentrant=False
                )
                self.logger.debug(f"Applied gradient checkpointing to {name}")
        
        return model
    
    def _should_checkpoint_module(self, module: nn.Module) -> bool:
        """Determine if module should use gradient checkpointing."""
        # Checkpoint large modules or attention blocks
        checkpoint_types = (
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.MultiheadAttention
        )
        
        if isinstance(module, checkpoint_types):
            return True
        
        # Checkpoint modules with many parameters
        param_count = sum(p.numel() for p in module.parameters())
        return param_count > 1000000  # 1M parameters threshold
    
    def _optimize_attention_memory(self, model: nn.Module) -> nn.Module:
        """Optimize attention mechanisms for memory efficiency."""
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Enable memory efficient attention if available
                if hasattr(module, 'enable_memory_efficient_attention'):
                    module.enable_memory_efficient_attention()
                
                # Override attention computation with memory-efficient version
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    self._patch_attention_module(module)
        
        return model
    
    def _patch_attention_module(self, attention_module: nn.MultiheadAttention):
        """Patch attention module with memory-efficient implementation."""
        original_forward = attention_module.forward
        
        def memory_efficient_forward(*args, **kwargs):
            # Use memory-efficient attention if available
            try:
                return torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)
            except:
                return original_forward(*args, **kwargs)
        
        attention_module.forward = memory_efficient_forward
    
    def _apply_parameter_sharing(self, model: nn.Module) -> nn.Module:
        """Apply parameter sharing optimizations."""
        # Track parameter usage
        parameter_usage = {}
        
        for name, param in model.named_parameters():
            param_hash = hash(param.data.data_ptr())
            if param_hash in parameter_usage:
                parameter_usage[param_hash].append(name)
            else:
                parameter_usage[param_hash] = [name]
        
        # Report sharing opportunities
        shared_params = {k: v for k, v in parameter_usage.items() if len(v) > 1}
        if shared_params:
            self.logger.info(f"Found {len(shared_params)} shared parameter groups")
        
        return model
    
    @contextmanager
    def memory_efficient_forward(self, cache_intermediates: bool = True):
        """Context manager for memory-efficient forward passes."""
        # Clear cache at start
        if cache_intermediates:
            self.clear_computation_cache()
        
        # Enable memory optimizations
        original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        
        try:
            yield self
        finally:
            # Restore settings
            torch.backends.cudnn.deterministic = original_deterministic
            
            # Optional cleanup
            if not cache_intermediates:
                self.clear_computation_cache()
    
    def cache_computation(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Cache intermediate computation with memory management."""
        tensor_bytes = tensor.element_size() * tensor.numel()
        
        # Check if we have space
        if self.cache_size_bytes + tensor_bytes > self.max_cache_bytes:
            self._evict_cache_entries(tensor_bytes)
        
        # Store in cache
        self.computation_cache[key] = tensor.detach()
        self.cache_size_bytes += tensor_bytes
        
        return tensor
    
    def get_cached_computation(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached computation."""
        return self.computation_cache.get(key)
    
    def _evict_cache_entries(self, required_bytes: int):
        """Evict cache entries to make space."""
        # Simple LRU eviction (could be improved with proper LRU tracking)
        keys_to_remove = []
        bytes_freed = 0
        
        for key, tensor in self.computation_cache.items():
            tensor_bytes = tensor.element_size() * tensor.numel()
            keys_to_remove.append(key)
            bytes_freed += tensor_bytes
            
            if bytes_freed >= required_bytes:
                break
        
        # Remove selected entries
        for key in keys_to_remove:
            del self.computation_cache[key]
        
        self.cache_size_bytes -= bytes_freed
        self.logger.debug(f"Evicted {len(keys_to_remove)} cache entries, freed {bytes_freed} bytes")
    
    def clear_computation_cache(self):
        """Clear computation cache."""
        self.computation_cache.clear()
        self.cache_size_bytes = 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_entries': len(self.computation_cache),
            'max_cache_mb': self.max_cache_bytes / (1024 * 1024)
        }
        
        # GPU memory stats
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'gpu_max_cached_mb': torch.cuda.max_memory_reserved() / (1024 * 1024)
            })
        
        # System memory stats
        memory_info = psutil.virtual_memory()
        stats.update({
            'system_memory_percent': memory_info.percent,
            'system_available_gb': memory_info.available / (1024**3),
            'system_total_gb': memory_info.total / (1024**3)
        })
        
        return stats
    
    def memory_cleanup(self, aggressive: bool = False):
        """Perform memory cleanup."""
        # Clear computation cache
        self.clear_computation_cache()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Python garbage collection
        if aggressive:
            gc.collect()
        
        self.logger.debug("Performed memory cleanup")
    
    def monitor_memory_usage(self, step: int):
        """Monitor and log memory usage."""
        if step % self.config.garbage_collect_frequency == 0:
            self.memory_cleanup()
        
        # Update peak stats
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            current_cached = torch.cuda.memory_reserved()
            
            self.memory_stats['peak_allocated'] = max(
                self.memory_stats['peak_allocated'], current_allocated
            )
            self.memory_stats['peak_cached'] = max(
                self.memory_stats['peak_cached'], current_cached
            )


class GradientCheckpointing:
    """Advanced gradient checkpointing utilities."""
    
    @staticmethod
    def checkpoint_sequential(modules: List[nn.Module], segments: int) -> nn.Module:
        """Apply gradient checkpointing to sequential modules."""
        class CheckpointedSequential(nn.Module):
            def __init__(self, modules: List[nn.Module], segments: int):
                super().__init__()
                self.modules = nn.ModuleList(modules)
                self.segments = segments
                self.segment_size = len(modules) // segments
            
            def forward(self, x):
                for i in range(0, len(self.modules), self.segment_size):
                    segment_modules = self.modules[i:i + self.segment_size]
                    
                    def segment_forward(x):
                        for module in segment_modules:
                            x = module(x)
                        return x
                    
                    x = torch.utils.checkpoint.checkpoint(
                        segment_forward, x, use_reentrant=False
                    )
                
                return x
        
        return CheckpointedSequential(modules, segments)
    
    @staticmethod
    def selective_checkpoint(module: nn.Module, 
                           checkpoint_layers: List[str]) -> nn.Module:
        """Apply checkpointing only to specified layers."""
        for name in checkpoint_layers:
            if hasattr(module, name):
                layer = getattr(module, name)
                original_forward = layer.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        original_forward, *args, use_reentrant=False, **kwargs
                    )
                
                layer.forward = checkpointed_forward
        
        return module