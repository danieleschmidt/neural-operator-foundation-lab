"""Neural Operator-specific performance optimizations.

This module provides specialized optimizations for neural operators including
mixed precision training, gradient checkpointing, kernel fusion, and memory management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import math
import numpy as np
from contextlib import contextmanager
import time

try:
    from torch.cuda.amp import autocast, GradScaler
    _HAS_AMP = True
except ImportError:
    _HAS_AMP = False


class NeuralOperatorOptimizer:
    """Advanced optimizer for neural operators with specialized features."""
    
    def __init__(self, 
                 model: nn.Module,
                 mixed_precision: bool = True,
                 gradient_checkpointing: bool = True,
                 kernel_fusion: bool = True,
                 memory_efficient: bool = True):
        self.model = model
        self.mixed_precision = mixed_precision and _HAS_AMP
        self.gradient_checkpointing = gradient_checkpointing
        self.kernel_fusion = kernel_fusion
        self.memory_efficient = memory_efficient
        
        # Initialize optimizers
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Optimization statistics
        self.stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'memory_allocated': 0.0,
            'memory_cached': 0.0,
            'gradient_scale': 1.0
        }
        
        # Apply optimizations
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup model optimizations."""
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        if self.kernel_fusion:
            self._fuse_kernels()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        def checkpoint_wrapper(module):
            if hasattr(module, '__call__'):
                def forward_with_checkpoint(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        module, *args, **kwargs, use_reentrant=False
                    )
                return forward_with_checkpoint
            return module
        
        # Apply to transformer layers and large conv layers
        for name, module in self.model.named_modules():
            if any(layer_type in name.lower() for layer_type in 
                   ['transformer', 'attention', 'conv2d', 'conv3d']):
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    module.forward = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                        original_forward, *args, **kwargs, use_reentrant=False
                    )
    
    def _fuse_kernels(self):
        """Fuse compatible kernels for better performance."""
        # Fuse Conv-BatchNorm-ReLU patterns
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Sequential):
                fused_modules = []
                i = 0
                while i < len(module):
                    current = module[i]
                    
                    # Check for Conv-BN-ReLU pattern
                    if (i + 2 < len(module) and
                        isinstance(current, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and
                        isinstance(module[i+1], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and
                        isinstance(module[i+2], (nn.ReLU, nn.GELU))):
                        
                        # Create fused module
                        fused = FusedConvBNActivation(current, module[i+1], module[i+2])
                        fused_modules.append(fused)
                        i += 3
                    else:
                        fused_modules.append(current)
                        i += 1
                
                # Replace sequential with fused version
                if len(fused_modules) != len(module):
                    setattr(self.model, name.split('.')[-1], nn.Sequential(*fused_modules))
    
    @contextmanager
    def optimization_context(self):
        """Context manager for optimized forward/backward pass."""
        start_time = time.time()
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        try:
            if self.mixed_precision:
                with autocast():
                    yield
            else:
                yield
        finally:
            # Update statistics
            self.stats['forward_time'] = time.time() - start_time
            
            if torch.cuda.is_available():
                self.stats['memory_allocated'] = torch.cuda.memory_allocated() - initial_memory
                self.stats['memory_cached'] = torch.cuda.memory_reserved()
            
            if self.scaler:
                self.stats['gradient_scale'] = self.scaler.get_scale()
    
    def optimize_step(self, optimizer, loss):
        """Optimized training step."""
        if self.mixed_precision and self.scaler:
            # Scale loss and backward
            self.scaler.scale(loss).backward()
            
            # Unscale gradients and step
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        optimizer.zero_grad()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.stats.copy()


class FusedConvBNActivation(nn.Module):
    """Fused convolution + batch normalization + activation."""
    
    def __init__(self, conv, bn, activation):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation
        
        # Fuse conv and bn weights during inference
        self.fused_conv = None
        self._fuse_weights()
    
    def _fuse_weights(self):
        """Fuse conv and batch norm weights."""
        if not self.training:
            # Create fused convolution
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
                bias=True
            )
            
            # Fuse weights
            bn_weight = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
            fused_conv.weight.data = self.conv.weight * bn_weight.view(-1, 1, 1, 1)
            
            if self.conv.bias is not None:
                fused_conv.bias.data = (self.conv.bias - self.bn.running_mean) * bn_weight + self.bn.bias
            else:
                fused_conv.bias.data = (0 - self.bn.running_mean) * bn_weight + self.bn.bias
            
            self.fused_conv = fused_conv
    
    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
        else:
            if self.fused_conv is None:
                self._fuse_weights()
            x = self.fused_conv(x)
            x = self.activation(x)
        return x


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention for large neural operators."""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 chunk_size: int = 1024,
                 sparse_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.sparse_attention = sparse_attention
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention forward pass."""
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.sparse_attention:
            attn_output = self._sparse_attention(q, k, v, mask)
        else:
            attn_output = self._chunked_attention(q, k, v, mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output
    
    def _chunked_attention(self, q, k, v, mask=None):
        """Compute attention in chunks to save memory."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Process in chunks
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores += mask[:, :, i:end_i, :]
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            output[:, :, i:end_i] = attn_output
        
        return output
    
    def _sparse_attention(self, q, k, v, mask=None):
        """Compute sparse attention for very long sequences."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Simple local attention pattern
        window_size = min(self.chunk_size, seq_len // 4)
        
        output = torch.zeros_like(q)
        
        for i in range(0, seq_len, window_size):
            end_i = min(i + window_size, seq_len)
            
            # Local window
            start_k = max(0, i - window_size // 2)
            end_k = min(seq_len, end_i + window_size // 2)
            
            q_local = q[:, :, i:end_i]
            k_local = k[:, :, start_k:end_k]
            v_local = v[:, :, start_k:end_k]
            
            scores = torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores += mask[:, :, i:end_i, start_k:end_k]
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_local)
            
            output[:, :, i:end_i] = attn_output
        
        return output


class AdaptiveBatchSampler:
    """Adaptive batch sampler based on input complexity."""
    
    def __init__(self, 
                 dataset,
                 base_batch_size: int = 32,
                 max_batch_size: int = 128,
                 complexity_fn: Optional[callable] = None):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.complexity_fn = complexity_fn or self._default_complexity
        
        # Compute complexity for all samples
        self.complexities = self._compute_complexities()
        
    def _default_complexity(self, sample):
        """Default complexity function based on input size."""
        if isinstance(sample, torch.Tensor):
            return sample.numel()
        elif isinstance(sample, (list, tuple)):
            return sum(self._default_complexity(s) for s in sample)
        else:
            return 1
    
    def _compute_complexities(self):
        """Compute complexity for all dataset samples."""
        complexities = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            complexity = self.complexity_fn(sample)
            complexities.append(complexity)
        return complexities
    
    def get_adaptive_batch_size(self, indices: List[int]) -> int:
        """Get adaptive batch size based on sample complexities."""
        if not indices:
            return self.base_batch_size
        
        # Get complexities for these indices
        batch_complexities = [self.complexities[i] for i in indices]
        avg_complexity = sum(batch_complexities) / len(batch_complexities)
        
        # Normalize complexity (assuming max complexity is 10x base)
        complexity_ratio = avg_complexity / (max(self.complexities) + 1e-8)
        
        # Adaptive batch size (inverse relationship with complexity)
        adaptive_size = int(self.base_batch_size / (complexity_ratio + 0.1))
        
        return min(max(adaptive_size, 1), self.max_batch_size)
    
    def create_batches(self) -> List[List[int]]:
        """Create batches with adaptive sizing."""
        indices = list(range(len(self.dataset)))
        
        # Sort by complexity for better batching
        sorted_indices = sorted(indices, key=lambda i: self.complexities[i])
        
        batches = []
        current_batch = []
        
        for idx in sorted_indices:
            current_batch.append(idx)
            
            # Check if we should create a batch
            if len(current_batch) >= self.get_adaptive_batch_size(current_batch):
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining samples
        if current_batch:
            batches.append(current_batch)
        
        return batches


class NeuralOperatorCache:
    """Intelligent caching system for neural operator computations."""
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 ttl_seconds: float = 3600):
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache = {}
        self.timestamps = {}
        self.access_counts = {}
        
    def _get_cache_key(self, x: torch.Tensor, model_id: str) -> str:
        """Generate cache key from input tensor and model."""
        # Use tensor hash and shape as key
        tensor_hash = hash(x.data_ptr())
        shape_str = "_".join(map(str, x.shape))
        return f"{model_id}_{tensor_hash}_{shape_str}"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used items."""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort by access count and timestamp
        items = [(key, self.access_counts.get(key, 0), self.timestamps.get(key, 0)) 
                for key in self.cache.keys()]
        items.sort(key=lambda x: (x[1], x[2]))  # Sort by access count, then timestamp
        
        # Remove least used items
        for key, _, _ in items[:len(items) - self.max_cache_size]:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get(self, x: torch.Tensor, model_id: str) -> Optional[torch.Tensor]:
        """Get cached result if available."""
        key = self._get_cache_key(x, model_id)
        
        if key in self.cache and not self._is_expired(key):
            # Update access count and timestamp
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.timestamps[key] = time.time()
            
            return self.cache[key]
        
        return None
    
    def put(self, x: torch.Tensor, model_id: str, result: torch.Tensor):
        """Cache computation result."""
        key = self._get_cache_key(x, model_id)
        
        # Evict if necessary
        self._evict_lru()
        
        # Store result
        self.cache[key] = result.detach().clone()
        self.timestamps[key] = time.time()
        self.access_counts[key] = 1
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        active_entries = sum(1 for ts in self.timestamps.values() 
                           if current_time - ts <= self.ttl_seconds)
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'cache_size_mb': sum(tensor.numel() * tensor.element_size() 
                               for tensor in self.cache.values()) / (1024 * 1024),
            'hit_rate': sum(self.access_counts.values()) / max(len(self.cache), 1)
        }


class DistributedNeuralOperatorTrainer:
    """Distributed training utilities for neural operators."""
    
    def __init__(self, 
                 model: nn.Module,
                 device_ids: List[int] = None,
                 sync_batchnorm: bool = True):
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.sync_batchnorm = sync_batchnorm
        
        self._setup_distributed_model()
    
    def _setup_distributed_model(self):
        """Setup model for distributed training."""
        if len(self.device_ids) > 1:
            # Convert BatchNorm to SyncBatchNorm
            if self.sync_batchnorm:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            
            # Wrap with DataParallel or DistributedDataParallel
            if torch.distributed.is_initialized():
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=self.device_ids,
                    find_unused_parameters=True
                )
            else:
                self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
    
    def all_reduce_gradients(self):
        """Manually reduce gradients across devices."""
        if not torch.distributed.is_initialized():
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad /= torch.distributed.get_world_size()
    
    def broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        if not torch.distributed.is_initialized():
            return
        
        for param in self.model.parameters():
            torch.distributed.broadcast(param.data, src=0)


def optimize_neural_operator_model(model: nn.Module, 
                                 optimization_config: Dict[str, Any]) -> nn.Module:
    """Apply comprehensive optimizations to a neural operator model."""
    
    # Create optimizer
    optimizer = NeuralOperatorOptimizer(
        model=model,
        mixed_precision=optimization_config.get('mixed_precision', True),
        gradient_checkpointing=optimization_config.get('gradient_checkpointing', True),
        kernel_fusion=optimization_config.get('kernel_fusion', True),
        memory_efficient=optimization_config.get('memory_efficient', True)
    )
    
    # Compile model if PyTorch 2.0+
    if hasattr(torch, 'compile') and optimization_config.get('compile_model', False):
        model = torch.compile(model, mode='max-autotune')
    
    return model, optimizer


def create_optimized_dataloader(dataset, 
                               batch_config: Dict[str, Any],
                               optimization_config: Dict[str, Any]):
    """Create optimized data loader for neural operators."""
    
    # Adaptive batch sampler
    if optimization_config.get('adaptive_batching', False):
        batch_sampler = AdaptiveBatchSampler(
            dataset=dataset,
            base_batch_size=batch_config.get('batch_size', 32),
            max_batch_size=batch_config.get('max_batch_size', 128)
        )
        
        batches = batch_sampler.create_batches()
        
        # Create custom dataloader with adaptive batches
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=batch_config.get('num_workers', 4),
            pin_memory=optimization_config.get('pin_memory', True),
            persistent_workers=optimization_config.get('persistent_workers', True)
        )
    else:
        # Standard dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_config.get('batch_size', 32),
            shuffle=batch_config.get('shuffle', True),
            num_workers=batch_config.get('num_workers', 4),
            pin_memory=optimization_config.get('pin_memory', True),
            persistent_workers=optimization_config.get('persistent_workers', True)
        )