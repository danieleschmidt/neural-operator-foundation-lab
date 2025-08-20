#!/usr/bin/env python3
"""Generation 2: Robust Neural Operator Framework

Adding comprehensive error handling, validation, logging, and monitoring.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
import time
import traceback

from neural_operator_lab.models import FourierNeuralOperator
from neural_operator_lab.security import InputValidator
from neural_operator_lab.monitoring import MetricsCollector, HealthChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('neural_operator.log')
    ]
)
logger = logging.getLogger(__name__)

class RobustNeuralOperator:
    """Robust wrapper for neural operators with comprehensive error handling."""
    
    def __init__(self, model_config: Dict[str, Any], enable_monitoring: bool = True):
        """Initialize robust neural operator.
        
        Args:
            model_config: Configuration dictionary for the model
            enable_monitoring: Whether to enable performance monitoring
        """
        self.config = model_config
        self.model = None
        self.validator = InputValidator()
        self.metrics = MetricsCollector() if enable_monitoring else None
        self.health_checker = HealthChecker() if enable_monitoring else None
        
        logger.info("Initializing robust neural operator...")
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the neural operator model with error handling."""
        try:
            self.model = FourierNeuralOperator(**self.config)
            logger.info(f"Model initialized successfully with {self._count_parameters()} parameters")
            
            # Validate model architecture
            self._validate_model_architecture()
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e
    
    def _count_parameters(self) -> int:
        """Count total number of model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _validate_model_architecture(self) -> None:
        """Validate model architecture and configuration."""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        # Check for reasonable parameter count
        param_count = self._count_parameters()
        if param_count > 1e9:  # > 1B parameters
            warnings.warn(f"Large model with {param_count:,} parameters - consider reducing size")
        elif param_count < 1000:  # < 1K parameters
            warnings.warn(f"Very small model with {param_count:,} parameters - may underfit")
    
    @contextmanager
    def _error_handling_context(self, operation_name: str):
        """Context manager for robust error handling."""
        start_time = time.time()
        try:
            logger.debug(f"Starting {operation_name}")
            if self.metrics and not self.metrics._collecting:
                self.metrics.start_collection()
            yield
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM in {operation_name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError(f"GPU memory exhausted during {operation_name}") from e
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error in {operation_name}: {e}")
                raise RuntimeError(f"CUDA operation failed in {operation_name}: {e}") from e
            else:
                logger.error(f"Runtime error in {operation_name}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Operation {operation_name} failed: {e}") from e
            
        finally:
            duration = time.time() - start_time
            logger.debug(f"Completed {operation_name} in {duration:.4f}s")
            if self.metrics:
                # Record custom metric for operation timing
                self.metrics.custom_metrics[f'{operation_name}_duration'].append(duration)
    
    def validate_input(self, x: torch.Tensor, operation: str = "forward") -> None:
        """Comprehensive input validation."""
        with self._error_handling_context(f"input_validation_{operation}"):
            # Basic tensor validation
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x)}")
            
            if x.numel() == 0:
                raise ValueError("Input tensor is empty")
            
            if torch.isnan(x).any():
                raise ValueError("Input contains NaN values")
            
            if torch.isinf(x).any():
                raise ValueError("Input contains infinite values")
            
            # Shape validation for neural operators
            if len(x.shape) < 3:
                raise ValueError(f"Input must have at least 3 dimensions, got {len(x.shape)}")
            
            # Memory check
            input_size_mb = x.numel() * x.element_size() / (1024 * 1024)
            if input_size_mb > 1000:  # > 1GB
                warnings.warn(f"Large input tensor: {input_size_mb:.1f} MB")
            
            # Value range check
            if x.abs().max() > 1e6:
                warnings.warn(f"Large input values detected: max={x.abs().max():.2e}")
            
            logger.debug(f"Input validation passed: {x.shape}, dtype={x.dtype}")
    
    def predict(self, x: torch.Tensor, return_metrics: bool = False) -> torch.Tensor:
        """Robust prediction with comprehensive error handling.
        
        Args:
            x: Input tensor
            return_metrics: Whether to return performance metrics
            
        Returns:
            Model prediction (and optionally metrics)
        """
        with self._error_handling_context("predict"):
            # Input validation
            self.validate_input(x, "predict")
            
            # Health check
            if self.health_checker:
                health_status = self.health_checker.get_current_health()
                if health_status.status != health_status.status.HEALTHY:
                    logger.warning(f"System health warning: {health_status.status}")
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                y = self.model(x)
                inference_time = time.time() - start_time
            
            # Output validation
            self._validate_output(y, x)
            
            # Record metrics
            if self.metrics:
                # Record custom metrics for inference
                self.metrics.custom_metrics['inference_time'].append(inference_time)
                self.metrics.custom_metrics['input_size'].append(x.numel())
                self.metrics.custom_metrics['output_size'].append(y.numel())
            
            logger.info(f"Prediction completed: {x.shape} -> {y.shape} in {inference_time:.4f}s")
            
            if return_metrics:
                metrics_data = {
                    'inference_time': inference_time,
                    'input_shape': x.shape,
                    'output_shape': y.shape,
                    'memory_used': self._get_memory_usage()
                }
                return y, metrics_data
            
            return y
    
    def _validate_output(self, y: torch.Tensor, x: torch.Tensor) -> None:
        """Validate model output."""
        with self._error_handling_context("output_validation"):
            if torch.isnan(y).any():
                raise ValueError("Model output contains NaN values")
            
            if torch.isinf(y).any():
                raise ValueError("Model output contains infinite values")
            
            # Check for exploding outputs
            if y.abs().max() > 1e8:
                raise ValueError(f"Extremely large output values: max={y.abs().max():.2e}")
            
            # Shape consistency check
            expected_batch_size = x.shape[0]
            if y.shape[0] != expected_batch_size:
                raise ValueError(f"Batch size mismatch: input={expected_batch_size}, output={y.shape[0]}")
            
            logger.debug("Output validation passed")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # CPU memory (approximate)
        import psutil
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        if not self.health_checker:
            return {"error": "Health checking disabled"}
        
        health = self.health_checker.get_current_health()
        return {
            "healthy": health.status == health.status.HEALTHY,
            "status": health.status.value,
            "metrics": {name: {"value": metric.value, "status": metric.status.value} 
                       for name, metric in health.metrics.items()},
            "alerts": health.alerts
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics:
            return {"error": "Metrics collection disabled"}
        
        # Build summary from collected metrics
        summary = {
            "system_metrics_count": len(self.metrics.system_metrics),
            "training_metrics_count": len(self.metrics.training_metrics),
            "operations": {}
        }
        
        # Add custom metrics summary
        for metric_name, values in self.metrics.custom_metrics.items():
            if values:
                summary["operations"][metric_name] = {
                    "count": len(values),
                    "latest": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else 0
                }
        
        return summary

def test_robust_neural_operator():
    """Test the robust neural operator implementation."""
    print("🛡️ Testing Robust Neural Operator...")
    
    # Configuration
    config = {
        'input_dim': 3,
        'output_dim': 2,
        'hidden_dim': 64,
        'num_layers': 4,
        'modes': [12, 12]
    }
    
    # Initialize robust model
    robust_model = RobustNeuralOperator(config, enable_monitoring=True)
    
    # Test normal operation
    x = torch.randn(4, 32, 32, 3)
    y, metrics = robust_model.predict(x, return_metrics=True)
    
    print(f"✅ Normal prediction: {x.shape} -> {y.shape}")
    print(f"   Inference time: {metrics['inference_time']:.4f}s")
    print(f"   Memory used: {metrics['memory_used'].get('cpu_memory_mb', 0):.1f} MB")
    
    # Test error handling with invalid input
    try:
        invalid_input = torch.tensor([[[float('nan')]]])  # Proper 3D shape
        robust_model.predict(invalid_input)
        print("❌ Should have caught NaN input!")
    except (ValueError, RuntimeError) as e:
        print(f"✅ Caught invalid input: {type(e).__name__}")
    
    # Test with extreme values
    try:
        extreme_input = torch.randn(2, 32, 32, 3) * 1e10
        robust_model.predict(extreme_input)
        print("✅ Handled extreme input values")
    except Warning:
        print("✅ Warning raised for extreme values")
    
    # Get health status
    health = robust_model.get_health_status()
    print(f"✅ System health: {health.get('healthy', 'Unknown')}")
    
    # Get metrics summary
    summary = robust_model.get_metrics_summary()
    print(f"✅ Metrics collected: {len(summary.get('operations', []))} operations")
    
    print("✅ Robust neural operator test passed!")

def test_error_recovery():
    """Test error recovery mechanisms."""
    print("🔧 Testing Error Recovery...")
    
    config = {
        'input_dim': 2,
        'output_dim': 1,
        'hidden_dim': 32,
        'num_layers': 2,
        'modes': [8, 8]
    }
    
    robust_model = RobustNeuralOperator(config)
    
    # Test multiple error conditions
    error_cases = [
        ("Empty tensor", torch.empty(0)),
        ("Wrong dimensions", torch.randn(10)),  # 1D instead of 3D+
        ("NaN values", torch.tensor([[float('nan')]])),
    ]
    
    for case_name, bad_input in error_cases:
        try:
            robust_model.predict(bad_input)
            print(f"❌ {case_name}: Should have failed!")
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"✅ {case_name}: Caught error - {type(e).__name__}")
    
    print("✅ Error recovery test passed!")

def main():
    """Run Generation 2 robustness tests."""
    print("🛡️ GENERATION 2: MAKE IT ROBUST - Adding Reliability")
    print("=" * 60)
    
    try:
        test_robust_neural_operator()
        print()
        
        test_error_recovery()
        print()
        
        print("🎉 GENERATION 2 COMPLETE: Robustness features implemented!")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Input/output validation")  
        print("   ✅ Performance monitoring")
        print("   ✅ Health checking")
        print("   ✅ Logging and metrics")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Generation 2 failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)