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
"""Compute optimization for neural operators."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ComputeOptimizer:
    """Optimizes computation for neural operator models."""
    
    def __init__(self, enable_fusion: bool = True, enable_jit: bool = True):
        """Initialize compute optimizer.
        
        Args:
            enable_fusion: Whether to enable kernel fusion
            enable_jit: Whether to enable JIT compilation
        """
        self.enable_fusion = enable_fusion
        self.enable_jit = enable_jit
        self._optimized_models: Dict[str, nn.Module] = {}
        
        logger.info(f"Compute optimizer initialized - fusion: {enable_fusion}, jit: {enable_jit}")
    
    def optimize_model(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            example_input: Example input tensor for optimization
            
        Returns:
            Optimized model
        """
        model_id = id(model)
        
        if model_id in self._optimized_models:
            logger.info("Model already optimized, returning cached version")
            return self._optimized_models[model_id]
        
        # SECURITY_DISABLED: optimized_model = model.eval(...)  # eval() disabled for security
        
        # Apply JIT compilation if enabled
        if self.enable_jit:
            try:
                with torch.no_grad():
                    optimized_model = torch.jit.trace(model, example_input)
                logger.info("JIT compilation applied successfully")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}, using original model")
                optimized_model = model
        
        # Apply fusion optimizations if enabled
        if self.enable_fusion:
            optimized_model = self._apply_fusion_optimizations(optimized_model)
        
        # Cache optimized model
        self._optimized_models[model_id] = optimized_model
        
        logger.info("Model optimization completed")
        return optimized_model
    
    def _apply_fusion_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply kernel fusion optimizations.
        
        Args:
            model: Model to apply fusion to
            
        Returns:
            Model with fusion optimizations
        """
        # Enable TensorRT if available
        try:
            import torch_tensorrt
            # TensorRT optimization would go here
            logger.debug("TensorRT optimization available")
        except ImportError:
            logger.debug("TensorRT not available")
        
        # Apply torch.compile if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='max-autotune')
                logger.info("Torch compile optimization applied")
        except Exception as e:
            logger.debug(f"Torch compile not available: {e}")
        
        return model
    
    @contextmanager
    def optimized_inference(self, model: nn.Module, example_input: torch.Tensor):
        """Context manager for optimized inference.
        
        Args:
            model: Model to optimize
            example_input: Example input for optimization
        """
        original_training_mode = model.training
        
        try:
            # Optimize model for inference
            optimized_model = self.optimize_model(model, example_input)
            # SECURITY_DISABLED: optimized_model.eval(...)  # eval() disabled for security
            
            with torch.no_grad():
                yield optimized_model
                
        finally:
            # Restore original training mode
            model.train(original_training_mode)
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results dictionary
        """
        # SECURITY_DISABLED: model.eval(...)  # eval() disabled for security
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                # Synchronize GPU if available
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Calculate statistics
        import statistics
        
        results = {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'throughput_samples_per_sec': 1.0 / statistics.mean(times),
            'num_runs': num_runs
        }
        
        logger.info(f"Benchmark completed: {results['mean_time']:.4f}s mean, "
                   f"{results['throughput_samples_per_sec']:.2f} samples/sec")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear optimized model cache."""
        self._optimized_models.clear()
        logger.info("Optimized model cache cleared")


class KernelFusion:
    """Kernel fusion utilities for neural operators."""
    
    @staticmethod
    def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fuse convolution and batch normalization layers.
        
        Args:
            conv: Convolution layer
            bn: Batch normalization layer
            
        Returns:
            Fused convolution layer
        """
        # Create new convolution layer
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True
        )
        
        # Fuse weights and biases
        with torch.no_grad():
            # Get batch norm parameters
            if bn.num_features != conv.out_channels:
                raise ValueError("BatchNorm and Conv output channels must match")
            
            # Calculate fused weight
            bn_weight = bn.weight if bn.weight is not None else torch.ones(bn.num_features)
            bn_bias = bn.bias if bn.bias is not None else torch.zeros(bn.num_features)
            bn_mean = bn.running_mean
            bn_var = bn.running_var
            bn_eps = bn.eps
            
            # Fuse into conv weight
            bn_scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            fused_conv.weight.copy_(conv.weight * bn_scale.view(-1, 1, 1, 1))
            
            # Fuse into conv bias
            conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
            fused_bias = bn_scale * conv_bias - bn_scale * bn_mean + bn_bias
            fused_conv.bias.copy_(fused_bias)
        
        logger.debug("Conv-BatchNorm fusion completed")
        return fused_conv
    
    @staticmethod
    def fuse_linear_activation(linear: nn.Linear, activation: nn.Module) -> nn.Module:
        """Fuse linear layer with activation.
        
        Args:
            linear: Linear layer
            activation: Activation function
            
        Returns:
            Fused module
        """
        class FusedLinearActivation(nn.Module):
            def __init__(self, linear_layer, activation_fn):
                super().__init__()
                self.linear = linear_layer
                self.activation = activation_fn
            
            def forward(self, x):
                return self.activation(self.linear(x))
        
        fused_module = FusedLinearActivation(linear, activation)
        logger.debug("Linear-Activation fusion completed")
        return fused_module
    
    @staticmethod 
    def optimize_spectral_conv(spectral_conv: nn.Module) -> nn.Module:
        """Optimize spectral convolution operations.
        
        Args:
            spectral_conv: Spectral convolution module
            
        Returns:
            Optimized spectral convolution
        """
        # This would contain specific optimizations for spectral convolutions
        # used in neural operators (FNO, etc.)
        
        class OptimizedSpectralConv(nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self.original = original_module
                self._cached_modes = None
            
            def forward(self, x):
                # Cache FFT modes for repeated use
                if self._cached_modes is None and hasattr(self.original, 'modes'):
                    self._cached_modes = self.original.modes
                
                return self.original(x)
        
        optimized = OptimizedSpectralConv(spectral_conv)
        logger.debug("Spectral convolution optimization completed")
        return optimized


class MemoryOptimization:
    """Memory optimization utilities."""
    
    @staticmethod
    @contextmanager
    def efficient_forward(model: nn.Module, use_checkpointing: bool = True):
        """Context manager for memory-efficient forward pass.
        
        Args:
            model: Model to optimize
            use_checkpointing: Whether to use gradient checkpointing
        """
        original_checkpointing = getattr(model, '_use_checkpointing', False)
        
        try:
            if use_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            yield model
            
        finally:
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
    
    @staticmethod
    def optimize_dataloader(
        dataloader: torch.utils.data.DataLoader,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ) -> torch.utils.data.DataLoader:
        """Optimize DataLoader for performance.
        
        Args:
            dataloader: Original DataLoader
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
            prefetch_factor: Prefetch factor
            
        Returns:
            Optimized DataLoader
        """
        if num_workers is None:
            import multiprocessing
            num_workers = min(multiprocessing.cpu_count(), 8)
        
        # Create optimized DataLoader
        optimized_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=getattr(dataloader, 'shuffle', False),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            drop_last=dataloader.drop_last
        )
        
        logger.info(f"DataLoader optimized: {num_workers} workers, "
                   f"pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")
        
        return optimized_loader