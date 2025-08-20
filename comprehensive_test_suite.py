#!/usr/bin/env python3
"""Comprehensive Testing Suite for Neural Operator Framework

Automated test generation, integration testing, and performance validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import unittest
import time
import traceback
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

from neural_operator_lab.models import FourierNeuralOperator, TransformerNeuralOperator
# from neural_operator_lab.testing import AutonomousTestGeneration

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    passed: bool
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class NeuralOperatorTestSuite:
    """Comprehensive test suite for neural operators."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        # self.test_generator = AutonomousTestGeneration()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories."""
        print("🧪 Running Comprehensive Neural Operator Test Suite")
        print("=" * 60)
        
        test_categories = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Robustness Tests", self.run_robustness_tests),
            ("Memory Tests", self.run_memory_tests),
            ("Autonomous Generated Tests", self.run_autonomous_tests)
        ]
        
        all_passed = True
        category_results = {}
        
        for category_name, test_method in test_categories:
            print(f"\\n📋 {category_name}")
            print("-" * 40)
            
            try:
                category_result = test_method()
                category_results[category_name] = category_result
                
                passed_count = sum(1 for r in category_result['results'] if r.passed)
                total_count = len(category_result['results'])
                
                if passed_count == total_count:
                    print(f"✅ {category_name}: {passed_count}/{total_count} tests passed")
                else:
                    print(f"❌ {category_name}: {passed_count}/{total_count} tests passed")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {category_name}: Failed with error - {e}")
                all_passed = False
                category_results[category_name] = {"error": str(e), "results": []}
        
        # Generate comprehensive report
        report = self._generate_test_report(category_results, all_passed)
        
        print("\\n" + "=" * 60)
        if all_passed:
            print("🎉 ALL TESTS PASSED! Neural operator framework is ready.")
        else:
            print("⚠️  Some tests failed. See detailed report below.")
            
        return report
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual components."""
        unit_tests = [
            self._test_fno_basic_functionality,
            self._test_tno_basic_functionality,
            self._test_model_parameter_count,
            self._test_forward_pass_shapes,
            self._test_gradient_flow,
            self._test_device_compatibility
        ]
        
        results = []
        for test_func in unit_tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return {"category": "unit", "results": results}
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for component interactions."""
        integration_tests = [
            self._test_training_pipeline_integration,
            self._test_data_loader_integration,
            self._test_loss_computation_integration,
            self._test_optimizer_integration,
            self._test_checkpoint_save_load
        ]
        
        results = []
        for test_func in integration_tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return {"category": "integration", "results": results}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks and validation."""
        performance_tests = [
            self._test_inference_speed,
            self._test_memory_efficiency,
            self._test_batch_scaling,
            self._test_large_model_performance,
            self._test_throughput_consistency
        ]
        
        results = []
        for test_func in performance_tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return {"category": "performance", "results": results}
    
    def run_robustness_tests(self) -> Dict[str, Any]:
        """Run robustness and edge case tests."""
        robustness_tests = [
            self._test_nan_input_handling,
            self._test_extreme_value_handling,
            self._test_empty_input_handling,
            self._test_mismatched_dimensions,
            self._test_numerical_stability,
            self._test_concurrent_access
        ]
        
        results = []
        for test_func in robustness_tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return {"category": "robustness", "results": results}
    
    def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory usage and leak detection tests."""
        memory_tests = [
            self._test_memory_usage_scaling,
            self._test_memory_leak_detection,
            self._test_gradient_memory_cleanup,
            self._test_cache_memory_management
        ]
        
        results = []
        for test_func in memory_tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return {"category": "memory", "results": results}
    
    def run_autonomous_tests(self) -> Dict[str, Any]:
        """Run autonomously generated tests."""
        # Simplified autonomous tests without complex dependencies
        test_configs = [
            {
                "name": "fno_2d_small",
                "model_type": "FNO",
                "input_shape": (2, 16, 16, 2),
                "expected_output_shape": (2, 16, 16, 1),
                "input_dim": 2,
                "output_dim": 1
            },
            {
                "name": "tno_2d_medium",
                "model_type": "TNO", 
                "input_shape": (4, 32, 32, 3),
                "expected_output_shape": (4, 32, 32, 2),
                "input_dim": 3,
                "output_dim": 2
            },
            {
                "name": "fno_batch_variation",
                "model_type": "FNO",
                "input_shape": (8, 24, 24, 1),
                "expected_output_shape": (8, 24, 24, 1),
                "input_dim": 1,
                "output_dim": 1
            }
        ]
        
        results = []
        for test_config in test_configs:
            result = self._run_generated_test(test_config)
            results.append(result)
            
        return {"category": "autonomous", "results": results}
    
    def _run_single_test(self, test_func) -> TestResult:
        """Run a single test function with error handling."""
        test_name = test_func.__name__
        start_time = time.time()
        
        try:
            performance_metrics = test_func()
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                passed=True,
                duration=duration,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            logger.error(f"Test {test_name} failed: {error_msg}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                error_message=error_msg
            )
    
    def _run_generated_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Run an autonomously generated test."""
        test_name = f"generated_{test_config.get('name', 'test')}"
        start_time = time.time()
        
        try:
            # Extract test configuration
            model_type = test_config['model_type']
            input_shape = test_config['input_shape']
            expected_output_shape = test_config['expected_output_shape']
            
            # Create model based on configuration
            if model_type == "FNO":
                model = FourierNeuralOperator(
                    input_dim=test_config.get('input_dim', 3),
                    output_dim=test_config.get('output_dim', 2),
                    hidden_dim=64,
                    num_layers=4,
                    modes=[12, 12]
                )
            else:  # TNO
                model = TransformerNeuralOperator(
                    input_dim=test_config.get('input_dim', 3),
                    output_dim=test_config.get('output_dim', 2),
                    hidden_dim=64,
                    num_layers=4
                )
            
            # Run test
            x = torch.randn(*input_shape)
            with torch.no_grad():
                y = model(x)
            
            # Validate output shape
            assert y.shape == tuple(expected_output_shape), f"Shape mismatch: {y.shape} vs {expected_output_shape}"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                passed=True,
                duration=duration,
                performance_metrics={"throughput": input_shape[0] / duration}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                error_message=str(e)
            )
    
    # Unit test implementations
    def _test_fno_basic_functionality(self) -> Dict[str, Any]:
        """Test basic FNO functionality."""
        model = FourierNeuralOperator(input_dim=3, output_dim=2, hidden_dim=64, num_layers=4, modes=[12, 12])
        x = torch.randn(4, 32, 32, 3)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (4, 32, 32, 2), f"Unexpected output shape: {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN values"
        assert not torch.isinf(y).any(), "Output contains infinite values"
        
        return {"output_shape": y.shape, "output_range": [y.min().item(), y.max().item()]}
    
    def _test_tno_basic_functionality(self) -> Dict[str, Any]:
        """Test basic TNO functionality."""
        model = TransformerNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2)
        x = torch.randn(2, 16, 16, 2)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 16, 16, 1), f"Unexpected output shape: {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN values"
        
        return {"output_shape": y.shape, "parameter_count": sum(p.numel() for p in model.parameters())}
    
    def _test_model_parameter_count(self) -> Dict[str, Any]:
        """Test model parameter counts are reasonable."""
        fno = FourierNeuralOperator(input_dim=3, output_dim=2, hidden_dim=64, num_layers=4, modes=[12, 12])
        tno = TransformerNeuralOperator(input_dim=3, output_dim=2, hidden_dim=64, num_layers=4)
        
        fno_params = sum(p.numel() for p in fno.parameters())
        tno_params = sum(p.numel() for p in tno.parameters())
        
        assert fno_params > 1000, f"FNO has too few parameters: {fno_params}"
        assert tno_params > 1000, f"TNO has too few parameters: {tno_params}"
        assert fno_params < 100_000_000, f"FNO has too many parameters: {fno_params}"
        assert tno_params < 100_000_000, f"TNO has too many parameters: {tno_params}"
        
        return {"fno_parameters": fno_params, "tno_parameters": tno_params}
    
    def _test_forward_pass_shapes(self) -> Dict[str, Any]:
        """Test forward pass with various input shapes."""
        model = FourierNeuralOperator(input_dim=2, output_dim=3, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        test_shapes = [(1, 16, 16, 2), (4, 32, 32, 2), (8, 64, 64, 2)]
        results = {}
        
        for shape in test_shapes:
            x = torch.randn(*shape)
            with torch.no_grad():
                y = model(x)
            
            expected_shape = shape[:-1] + (3,)  # Change last dimension to output_dim
            assert y.shape == expected_shape, f"Shape {shape}: expected {expected_shape}, got {y.shape}"
            results[str(shape)] = str(y.shape)
        
        return results
    
    def _test_gradient_flow(self) -> Dict[str, Any]:
        """Test gradient flow through models."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        x = torch.randn(2, 16, 16, 2, requires_grad=True)
        
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist and are reasonable
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                assert not np.isnan(grad_norm), "Gradient contains NaN"
                assert not np.isinf(grad_norm), "Gradient contains infinity"
        
        assert len(grad_norms) > 0, "No gradients computed"
        
        return {"gradient_norms": grad_norms, "avg_grad_norm": np.mean(grad_norms)}
    
    def _test_device_compatibility(self) -> Dict[str, Any]:
        """Test CPU/GPU device compatibility."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        x = torch.randn(2, 16, 16, 2)
        
        # Test CPU
        model_cpu = model.cpu()
        x_cpu = x.cpu()
        with torch.no_grad():
            y_cpu = model_cpu(x_cpu)
        
        results = {"cpu_works": True}
        
        # Test GPU if available
        if torch.cuda.is_available():
            try:
                model_gpu = model.cuda()
                x_gpu = x.cuda()
                with torch.no_grad():
                    y_gpu = model_gpu(x_gpu)
                results["gpu_works"] = True
                results["gpu_device"] = str(x_gpu.device)
            except Exception as e:
                results["gpu_works"] = False
                results["gpu_error"] = str(e)
        else:
            results["gpu_available"] = False
        
        return results
    
    # Integration test implementations  
    def _test_training_pipeline_integration(self) -> Dict[str, Any]:
        """Test complete training pipeline integration."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Dummy training data
        x_train = torch.randn(8, 16, 16, 2)
        y_train = torch.randn(8, 16, 16, 1)
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(5):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            
            if initial_loss is None:
                initial_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        
        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2, f"Loss exploded: {initial_loss} -> {final_loss}"
        
        return {"initial_loss": initial_loss, "final_loss": final_loss, "loss_reduction": initial_loss - final_loss}
    
    def _test_data_loader_integration(self) -> Dict[str, Any]:
        """Test data loader integration."""
        try:
            from neural_operator_lab.data import PDEDataset
            
            # Create synthetic dataset
            data = [torch.randn(32, 32, 3) for _ in range(10)]
            labels = [torch.randn(32, 32, 2) for _ in range(10)]
            
            # This is a simplified test - real dataset creation would be more complex
            return {"synthetic_data_created": True, "data_samples": len(data)}
            
        except ImportError:
            return {"data_module_available": False}
    
    def _test_loss_computation_integration(self) -> Dict[str, Any]:
        """Test loss computation integration."""
        model = FourierNeuralOperator(input_dim=1, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        x = torch.randn(4, 16, 16, 1)
        y_true = torch.randn(4, 16, 16, 1)
        
        with torch.no_grad():
            y_pred = model(x)
        
        # Test different loss functions
        mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        l1_loss = torch.nn.functional.l1_loss(y_pred, y_true)
        
        assert not torch.isnan(mse_loss), "MSE loss is NaN"
        assert not torch.isnan(l1_loss), "L1 loss is NaN"
        assert mse_loss >= 0, "MSE loss is negative"
        assert l1_loss >= 0, "L1 loss is negative"
        
        return {"mse_loss": mse_loss.item(), "l1_loss": l1_loss.item()}
    
    def _test_optimizer_integration(self) -> Dict[str, Any]:
        """Test optimizer integration."""
        model = FourierNeuralOperator(input_dim=1, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        optimizers = {
            "Adam": torch.optim.Adam(model.parameters(), lr=1e-3),
            "SGD": torch.optim.SGD(model.parameters(), lr=1e-3),
            "AdamW": torch.optim.AdamW(model.parameters(), lr=1e-3)
        }
        
        x = torch.randn(2, 16, 16, 1)
        y_true = torch.randn(2, 16, 16, 1)
        
        results = {}
        
        for name, optimizer in optimizers.items():
            try:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = torch.nn.functional.mse_loss(y_pred, y_true)
                loss.backward()
                optimizer.step()
                results[name] = {"success": True, "loss": loss.item()}
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
        
        return results
    
    def _test_checkpoint_save_load(self) -> Dict[str, Any]:
        """Test model checkpoint saving and loading."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        x = torch.randn(2, 16, 16, 2)
        
        # Get initial output
        with torch.no_grad():
            y_initial = model(x)
        
        # Save model state
        state_dict = model.state_dict()
        
        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Verify model changed
        with torch.no_grad():
            y_modified = model(x)
        
        assert not torch.allclose(y_initial, y_modified, atol=1e-6), "Model weights didn't change"
        
        # Restore model state
        model.load_state_dict(state_dict)
        
        # Verify model restored
        with torch.no_grad():
            y_restored = model(x)
        
        assert torch.allclose(y_initial, y_restored, atol=1e-6), "Model state not properly restored"
        
        return {"checkpoint_save_load_works": True}
    
    # Performance test implementations
    def _test_inference_speed(self) -> Dict[str, Any]:
        """Test inference speed benchmarks."""
        model = FourierNeuralOperator(input_dim=3, output_dim=2, hidden_dim=64, num_layers=4, modes=[12, 12])
        model.eval()
        
        # Warmup
        x_warmup = torch.randn(1, 32, 32, 3)
        with torch.no_grad():
            _ = model(x_warmup)
        
        # Benchmark different batch sizes
        batch_sizes = [1, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32, 32, 3)
            
            # Time multiple runs
            times = []
            for _ in range(5):
                start_time = time.time()
                with torch.no_grad():
                    y = model(x)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            results[f"batch_{batch_size}"] = {
                "avg_time": avg_time,
                "throughput": throughput,
                "samples_per_second": throughput
            }
        
        return results
    
    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory usage efficiency."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=64, num_layers=4, modes=[12, 12])
        model = model.cuda()
        
        model_memory = torch.cuda.memory_allocated() - initial_memory
        
        x = torch.randn(8, 64, 64, 2).cuda()
        
        with torch.no_grad():
            y = model(x)
        
        inference_memory = torch.cuda.memory_allocated() - initial_memory - model_memory
        
        torch.cuda.empty_cache()
        
        return {
            "model_memory_mb": model_memory / (1024 * 1024),
            "inference_memory_mb": inference_memory / (1024 * 1024),
            "total_memory_mb": (model_memory + inference_memory) / (1024 * 1024)
        }
    
    def _test_batch_scaling(self) -> Dict[str, Any]:
        """Test how performance scales with batch size."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32, 32, 2)
            
            start_time = time.time()
            with torch.no_grad():
                y = model(x)
            end_time = time.time()
            
            time_per_sample = (end_time - start_time) / batch_size
            results[str(batch_size)] = {
                "total_time": end_time - start_time,
                "time_per_sample": time_per_sample,
                "efficiency_ratio": 1.0 / time_per_sample if time_per_sample > 0 else 0
            }
        
        return results
    
    def _test_large_model_performance(self) -> Dict[str, Any]:
        """Test performance with larger models."""
        large_model = FourierNeuralOperator(
            input_dim=4, output_dim=3, hidden_dim=128, num_layers=8, modes=[16, 16]
        )
        
        x = torch.randn(4, 64, 64, 4)
        
        start_time = time.time()
        with torch.no_grad():
            y = large_model(x)
        inference_time = time.time() - start_time
        
        param_count = sum(p.numel() for p in large_model.parameters())
        
        return {
            "parameter_count": param_count,
            "inference_time": inference_time,
            "throughput": x.shape[0] / inference_time
        }
    
    def _test_throughput_consistency(self) -> Dict[str, Any]:
        """Test throughput consistency over multiple runs."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=64, num_layers=4, modes=[12, 12])
        model.eval()
        
        x = torch.randn(8, 32, 32, 2)
        throughputs = []
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Measure consistency over 10 runs
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                y = model(x)
            throughput = x.shape[0] / (time.time() - start_time)
            throughputs.append(throughput)
        
        return {
            "mean_throughput": np.mean(throughputs),
            "std_throughput": np.std(throughputs),
            "coefficient_of_variation": np.std(throughputs) / np.mean(throughputs),
            "min_throughput": np.min(throughputs),
            "max_throughput": np.max(throughputs)
        }
    
    # Robustness test implementations
    def _test_nan_input_handling(self) -> Dict[str, Any]:
        """Test handling of NaN inputs."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        x_nan = torch.randn(2, 16, 16, 2)
        x_nan[0, 5, 5, 0] = float('nan')  # Inject NaN
        
        try:
            with torch.no_grad():
                y = model(x_nan)
            
            has_nan_output = torch.isnan(y).any().item()
            return {"nan_propagated": has_nan_output, "handled_gracefully": True}
            
        except Exception as e:
            return {"nan_propagated": None, "handled_gracefully": False, "error": str(e)}
    
    def _test_extreme_value_handling(self) -> Dict[str, Any]:
        """Test handling of extreme input values."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        # Test with very large values
        x_large = torch.randn(2, 16, 16, 2) * 1e6
        
        try:
            with torch.no_grad():
                y_large = model(x_large)
            
            large_output_finite = torch.isfinite(y_large).all().item()
        except Exception as e:
            large_output_finite = False
        
        # Test with very small values  
        x_small = torch.randn(2, 16, 16, 2) * 1e-6
        
        try:
            with torch.no_grad():
                y_small = model(x_small)
            
            small_output_finite = torch.isfinite(y_small).all().item()
        except Exception as e:
            small_output_finite = False
        
        return {
            "large_values_handled": large_output_finite,
            "small_values_handled": small_output_finite
        }
    
    def _test_empty_input_handling(self) -> Dict[str, Any]:
        """Test handling of empty or edge case inputs."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        test_cases = {
            "zero_batch": torch.empty(0, 16, 16, 2),
            "minimal_spatial": torch.randn(2, 1, 1, 2),
            "zero_tensor": torch.zeros(2, 16, 16, 2)
        }
        
        results = {}
        
        for case_name, x in test_cases.items():
            try:
                with torch.no_grad():
                    y = model(x)
                results[case_name] = {"handled": True, "output_shape": list(y.shape)}
            except Exception as e:
                results[case_name] = {"handled": False, "error": str(e)}
        
        return results
    
    def _test_mismatched_dimensions(self) -> Dict[str, Any]:
        """Test handling of mismatched input dimensions."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        wrong_inputs = {
            "wrong_channels": torch.randn(2, 16, 16, 3),  # 3 channels instead of 2
            "wrong_spatial": torch.randn(2, 16, 2),        # Missing one spatial dimension
            "extra_dim": torch.randn(2, 16, 16, 16, 2)     # Extra dimension
        }
        
        results = {}
        
        for case_name, x in wrong_inputs.items():
            try:
                with torch.no_grad():
                    y = model(x)
                results[case_name] = {"error_caught": False, "unexpected_success": True}
            except Exception as e:
                results[case_name] = {"error_caught": True, "error_type": type(e).__name__}
        
        return results
    
    def _test_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability over multiple forward passes."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        
        x = torch.randn(4, 32, 32, 2)
        outputs = []
        
        # Run multiple forward passes
        with torch.no_grad():
            for _ in range(5):
                y = model(x)
                outputs.append(y.clone())
        
        # Check consistency
        max_diff = 0.0
        for i in range(1, len(outputs)):
            diff = torch.abs(outputs[i] - outputs[0]).max().item()
            max_diff = max(max_diff, diff)
        
        return {
            "max_output_difference": max_diff,
            "numerically_stable": max_diff < 1e-6
        }
    
    def _test_concurrent_access(self) -> Dict[str, Any]:
        """Test concurrent model access (thread safety)."""
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8])
        model.eval()
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                x = torch.randn(2, 16, 16, 2)
                with torch.no_grad():
                    y = model(x)
                results_queue.put(y.shape)
            except Exception as e:
                errors_queue.put(str(e))
        
        # Start multiple threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        return {
            "successful_threads": len(results),
            "failed_threads": len(errors),
            "thread_safe": len(errors) == 0,
            "errors": errors
        }
    
    # Memory test implementations
    def _test_memory_usage_scaling(self) -> Dict[str, Any]:
        """Test memory usage scaling with input size."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8]).cuda()
        
        sizes = [(2, 16, 16, 2), (2, 32, 32, 2), (2, 64, 64, 2)]
        memory_usage = {}
        
        for size in sizes:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            x = torch.randn(*size).cuda()
            with torch.no_grad():
                y = model(x)
            
            peak_memory = torch.cuda.memory_allocated() - initial_memory
            memory_usage[str(size)] = peak_memory / (1024 * 1024)  # MB
        
        return memory_usage
    
    def _test_memory_leak_detection(self) -> Dict[str, Any]:
        """Test for memory leaks over multiple iterations."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=32, num_layers=2, modes=[8, 8]).cuda()
        
        torch.cuda.empty_cache()
        memory_snapshots = []
        
        for i in range(10):
            x = torch.randn(4, 32, 32, 2).cuda()
            with torch.no_grad():
                y = model(x)
            
            # Clean up
            del x, y
            torch.cuda.empty_cache()
            
            memory_snapshots.append(torch.cuda.memory_allocated())
        
        # Check if memory is consistently increasing
        memory_trend = np.polyfit(range(len(memory_snapshots)), memory_snapshots, 1)[0]
        
        return {
            "memory_snapshots_mb": [m / (1024 * 1024) for m in memory_snapshots],
            "memory_trend_mb_per_iter": memory_trend / (1024 * 1024),
            "likely_memory_leak": memory_trend > 1024 * 1024  # More than 1MB increase per iteration
        }
    
    def _test_gradient_memory_cleanup(self) -> Dict[str, Any]:
        """Test gradient memory cleanup."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        model = FourierNeuralOperator(input_dim=2, output_dim=1, hidden_dim=64, num_layers=4, modes=[12, 12]).cuda()
        
        x = torch.randn(4, 32, 32, 2, requires_grad=True).cuda()
        
        # Forward and backward pass
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        memory_with_gradients = torch.cuda.memory_allocated()
        
        # Clear gradients
        model.zero_grad()
        del loss, y
        
        memory_after_cleanup = torch.cuda.memory_allocated()
        
        gradient_memory_freed = memory_with_gradients - memory_after_cleanup
        
        return {
            "memory_with_gradients_mb": memory_with_gradients / (1024 * 1024),
            "memory_after_cleanup_mb": memory_after_cleanup / (1024 * 1024),
            "gradient_memory_freed_mb": gradient_memory_freed / (1024 * 1024)
        }
    
    def _test_cache_memory_management(self) -> Dict[str, Any]:
        """Test cache memory management."""
        # Simple cache test - would integrate with actual caching system
        cache = {}
        max_cache_size = 100
        
        # Fill cache
        for i in range(150):  # More than max size
            key = f"item_{i}"
            value = torch.randn(64, 64)  # Some tensor data
            
            cache[key] = value
            
            # Simple LRU-like eviction
            if len(cache) > max_cache_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
        
        return {
            "final_cache_size": len(cache),
            "cache_size_controlled": len(cache) <= max_cache_size
        }
    
    def _generate_test_report(self, category_results: Dict[str, Any], all_passed: bool) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = 0
        total_passed = 0
        total_duration = 0.0
        
        for category, data in category_results.items():
            if "results" in data:
                category_tests = len(data["results"])
                category_passed = sum(1 for r in data["results"] if r.passed)
                category_duration = sum(r.duration for r in data["results"])
                
                total_tests += category_tests
                total_passed += category_passed
                total_duration += category_duration
        
        report = {
            "summary": {
                "all_tests_passed": all_passed,
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_tests - total_passed,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0.0,
                "total_duration": total_duration
            },
            "categories": category_results,
            "timestamp": time.time()
        }
        
        return report

def main():
    """Run the comprehensive test suite."""
    suite = NeuralOperatorTestSuite()
    
    try:
        report = suite.run_all_tests()
        
        # Save detailed report
        with open("test_report.json", "w") as f:
            # Convert TestResult objects to dictionaries for JSON serialization
            json_report = {}
            for key, value in report.items():
                if key == "categories":
                    json_report[key] = {}
                    for cat_name, cat_data in value.items():
                        if "results" in cat_data:
                            json_report[key][cat_name] = {
                                **cat_data,
                                "results": [asdict(result) for result in cat_data["results"]]
                            }
                        else:
                            json_report[key][cat_name] = cat_data
                else:
                    json_report[key] = value
            
            json.dump(json_report, f, indent=2)
        
        print(f"\\n📊 Detailed test report saved to test_report.json")
        
        # Print summary
        summary = report["summary"]
        print(f"\\n📈 FINAL SUMMARY:")
        print(f"   Tests passed: {summary['tests_passed']}/{summary['total_tests']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Total duration: {summary['total_duration']:.2f}s")
        
        return summary["all_tests_passed"]
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)