"""Performance benchmarks and tests for Neural Operator Foundation Lab."""

import pytest
import torch
import time
import psutil
import numpy as np
from unittest.mock import Mock, patch
import gc


class PerformanceBenchmark:
    """Base class for performance benchmarking."""
    
    def __init__(self):
        self.results = {}
        self.baseline_times = {
            'fno_forward': 0.1,    # 100ms baseline
            'tno_forward': 0.2,    # 200ms baseline  
            'pno_forward': 0.15,   # 150ms baseline
            'training_step': 0.5,  # 500ms baseline
        }
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        # Warmup runs
        for _ in range(3):
            func(*args, **kwargs)
        
        # Actual timing
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result': result
        }
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        # Clear cache and collect garbage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            final_gpu_memory = 0
        
        return {
            'cpu_memory_delta': final_memory - initial_memory,
            'gpu_memory_delta': final_gpu_memory - initial_gpu_memory,
            'peak_cpu_memory': final_memory,
            'peak_gpu_memory': final_gpu_memory,
            'result': result
        }
    
    def benchmark_throughput(self, func, batch_sizes, *args, **kwargs):
        """Benchmark throughput across different batch sizes."""
        throughput_results = {}
        
        for batch_size in batch_sizes:
            # Create batch data
            if len(args) > 0 and hasattr(args[0], 'shape'):
                # Assume first arg is input tensor
                input_data = args[0]
                original_batch_size = input_data.shape[0]
                
                # Scale to desired batch size
                scale_factor = batch_size / original_batch_size
                if scale_factor != 1.0:
                    repeats = max(1, int(scale_factor))
                    scaled_input = input_data.repeat(repeats, *([1] * (len(input_data.shape) - 1)))
                    scaled_input = scaled_input[:batch_size]  # Trim to exact size
                    scaled_args = (scaled_input,) + args[1:]
                else:
                    scaled_args = args
            else:
                scaled_args = args
            
            # Time execution
            timing_result = self.time_function(func, *scaled_args, **kwargs)
            
            # Calculate throughput (samples per second)
            throughput = batch_size / timing_result['mean_time']
            
            throughput_results[batch_size] = {
                'throughput': throughput,
                'mean_time': timing_result['mean_time'],
                'samples_per_second': throughput
            }
        
        return throughput_results


# Import mock models for testing
from ..unit.test_neural_operators import (
    MockFourierNeuralOperator, 
    MockTransformerNeuralOperator,
    MockProbabilisticNeuralOperator
)


@pytest.mark.slow
@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for neural operator models."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.fixture
    def performance_models(self):
        """Create models for performance testing."""
        return {
            'fno_small': MockFourierNeuralOperator(3, 1, [8, 8], 32),
            'fno_medium': MockFourierNeuralOperator(3, 1, [16, 16], 64),
            'fno_large': MockFourierNeuralOperator(3, 1, [32, 32], 128),
            'tno_small': MockTransformerNeuralOperator(3, 1, 64, 2, 2),
            'tno_medium': MockTransformerNeuralOperator(3, 1, 128, 4, 4),
            'tno_large': MockTransformerNeuralOperator(3, 1, 256, 6, 8),
            'pno_small': MockProbabilisticNeuralOperator(3, 1, [32, 32], [32, 32]),
            'pno_medium': MockProbabilisticNeuralOperator(3, 1, [64, 64], [64, 64]),
        }
    
    @pytest.fixture
    def performance_data(self):
        """Create data for performance testing."""
        return {
            'small': torch.randn(4, 3, 32, 32),
            'medium': torch.randn(8, 3, 64, 64),
            'large': torch.randn(16, 3, 128, 128),
            'xlarge': torch.randn(32, 3, 256, 256)
        }
    
    def test_forward_pass_performance(self, benchmark, performance_models, performance_data):
        """Test forward pass performance across different models and data sizes."""
        results = {}
        
        for model_name, model in performance_models.items():
            model.eval()
            model_results = {}
            
            for data_name, data in performance_data.items():
                try:
                    # Skip incompatible combinations
                    if 'xlarge' in data_name and 'large' in model_name:
                        continue
                    
                    with torch.no_grad():
                        timing_result = benchmark.time_function(model, data)
                    
                    model_results[data_name] = timing_result
                    
                    # Performance assertion (model should complete within reasonable time)
                    assert timing_result['mean_time'] < 2.0, f"Model {model_name} too slow on {data_name}"
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Expected for large combinations
                        model_results[data_name] = {'error': 'OOM'}
                    else:
                        raise
            
            results[model_name] = model_results
        
        # Store results for analysis
        benchmark.results['forward_pass'] = results
    
    def test_memory_usage(self, benchmark, performance_models, performance_data):
        """Test memory usage during model inference."""
        memory_results = {}
        
        for model_name, model in performance_models.items():
            model.eval()
            
            # Test with medium data size
            data = performance_data['medium']
            
            try:
                with torch.no_grad():
                    memory_result = benchmark.measure_memory_usage(model, data)
                
                memory_results[model_name] = memory_result
                
                # Memory usage should be reasonable
                assert memory_result['cpu_memory_delta'] < 1000, f"Model {model_name} uses too much CPU memory"
                
                if torch.cuda.is_available():
                    assert memory_result['gpu_memory_delta'] < 2000, f"Model {model_name} uses too much GPU memory"
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_results[model_name] = {'error': 'OOM'}
                else:
                    raise
        
        benchmark.results['memory_usage'] = memory_results
    
    def test_batch_size_scaling(self, benchmark, performance_models, performance_data):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 2, 4, 8, 16]
        scaling_results = {}
        
        # Test with small model and medium resolution
        model = performance_models['fno_small']
        model.eval()
        
        base_data = performance_data['medium'][:1]  # Single sample
        
        with torch.no_grad():
            throughput_results = benchmark.benchmark_throughput(
                model, batch_sizes, base_data
            )
        
        scaling_results['fno_small'] = throughput_results
        
        # Verify throughput generally increases with batch size (up to a point)
        throughputs = [result['throughput'] for result in throughput_results.values()]
        
        # At least some increase in throughput should be observed
        assert max(throughputs) > min(throughputs) * 1.5, "Batch processing should improve throughput"
        
        benchmark.results['batch_scaling'] = scaling_results
    
    @pytest.mark.parametrize("precision", ["float32", "float16"])
    def test_mixed_precision_performance(self, benchmark, performance_models, precision):
        """Test performance with different precisions."""
        if precision == "float16" and not torch.cuda.is_available():
            pytest.skip("Float16 requires CUDA")
        
        model = performance_models['fno_medium']
        data = performance_data['medium']
        
        # Convert to specified precision
        if precision == "float16":
            model = model.half().cuda()
            data = data.half().cuda()
        else:
            model = model.float()
            data = data.float()
        
        model.eval()
        
        with torch.no_grad():
            timing_result = benchmark.time_function(model, data)
        
        # Store precision-specific results
        if 'precision' not in benchmark.results:
            benchmark.results['precision'] = {}
        
        benchmark.results['precision'][precision] = timing_result
        
        # Performance should be reasonable
        assert timing_result['mean_time'] < 1.0, f"Model too slow with {precision}"
    
    def test_training_step_performance(self, benchmark, performance_models, performance_data):
        """Test performance of training steps."""
        model = performance_models['fno_medium']
        data = performance_data['medium']
        target = torch.randn_like(data[:, :1])  # Single output channel
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        def training_step():
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        timing_result = benchmark.time_function(training_step)
        
        # Training step should complete within reasonable time
        assert timing_result['mean_time'] < 2.0, "Training step too slow"
        
        # Loss should be computed (not NaN)
        assert not np.isnan(timing_result['result']), "Training produced NaN loss"
        
        benchmark.results['training_step'] = timing_result


@pytest.mark.slow
@pytest.mark.performance
class TestDataLoadingPerformance:
    """Performance tests for data loading and preprocessing."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    def test_dataloader_performance(self, benchmark, mock_dataset):
        """Test data loading performance."""
        batch_sizes = [1, 4, 8, 16, 32]
        num_workers_list = [0, 1, 2, 4]
        
        results = {}
        
        for batch_size in batch_sizes:
            batch_results = {}
            
            for num_workers in num_workers_list:
                try:
                    dataloader = torch.utils.data.DataLoader(
                        mock_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available()
                    )
                    
                    def load_batches():
                        data_list = []
                        for i, batch in enumerate(dataloader):
                            data_list.append(batch)
                            if i >= 10:  # Load first 10 batches
                                break
                        return len(data_list)
                    
                    timing_result = benchmark.time_function(load_batches)
                    batch_results[f'workers_{num_workers}'] = timing_result
                    
                except Exception as e:
                    batch_results[f'workers_{num_workers}'] = {'error': str(e)}
            
            results[f'batch_{batch_size}'] = batch_results
        
        benchmark.results['dataloader'] = results
        
        # Verify that data loading completed successfully
        for batch_size, batch_results in results.items():
            successful_configs = [k for k, v in batch_results.items() if 'error' not in v]
            assert len(successful_configs) > 0, f"No successful configurations for {batch_size}"
    
    def test_data_preprocessing_performance(self, benchmark):
        """Test data preprocessing performance."""
        # Create large dataset for preprocessing testing
        large_data = torch.randn(100, 3, 128, 128)
        
        def normalize_data(data):
            return (data - data.mean()) / data.std()
        
        def augment_data(data):
            # Simple augmentation: random flip
            return torch.flip(data, dims=[-1])
        
        def resize_data(data):
            return torch.nn.functional.interpolate(
                data, size=(64, 64), mode='bilinear', align_corners=False
            )
        
        # Test individual operations
        operations = {
            'normalize': normalize_data,
            'augment': augment_data,
            'resize': resize_data
        }
        
        preprocessing_results = {}
        
        for op_name, op_func in operations.items():
            timing_result = benchmark.time_function(op_func, large_data)
            preprocessing_results[op_name] = timing_result
            
            # Operations should be reasonably fast
            assert timing_result['mean_time'] < 1.0, f"Preprocessing operation {op_name} too slow"
        
        benchmark.results['preprocessing'] = preprocessing_results
    
    def test_memory_mapped_loading(self, benchmark, tmp_path):
        """Test memory-mapped file loading performance."""
        # Create a large numpy array file
        large_array = np.random.randn(1000, 3, 64, 64).astype(np.float32)
        array_path = tmp_path / "large_array.npy"
        np.save(array_path, large_array)
        
        def load_full_array():
            return np.load(array_path)
        
        def load_memory_mapped():
            return np.load(array_path, mmap_mode='r')
        
        # Test both loading methods
        full_load_result = benchmark.time_function(load_full_array)
        mmap_load_result = benchmark.time_function(load_memory_mapped)
        
        # Memory mapping should be faster for large files
        # (though this may vary depending on system and file size)
        benchmark.results['file_loading'] = {
            'full_load': full_load_result,
            'memory_mapped': mmap_load_result
        }
        
        # Both methods should complete successfully
        assert full_load_result['result'].shape == (1000, 3, 64, 64)
        assert mmap_load_result['result'].shape == (1000, 3, 64, 64)


@pytest.mark.slow
@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Scalability benchmarks for large-scale scenarios."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    def test_model_parameter_scaling(self, benchmark):
        """Test how performance scales with model parameters."""
        model_configs = [
            {'width': 32, 'modes': [8, 8]},     # ~50K parameters
            {'width': 64, 'modes': [16, 16]},   # ~200K parameters  
            {'width': 128, 'modes': [32, 32]},  # ~800K parameters
        ]
        
        test_data = torch.randn(4, 3, 64, 64)
        scaling_results = {}
        
        for i, config in enumerate(model_configs):
            model = MockFourierNeuralOperator(
                input_dim=3, output_dim=1, 
                modes=config['modes'], width=config['width']
            )
            model.eval()
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            with torch.no_grad():
                timing_result = benchmark.time_function(model, test_data)
            
            scaling_results[f'config_{i}'] = {
                'param_count': param_count,
                'timing': timing_result,
                'throughput': test_data.shape[0] / timing_result['mean_time']
            }
        
        benchmark.results['parameter_scaling'] = scaling_results
        
        # Verify that all configurations work
        for config_name, result in scaling_results.items():
            assert result['timing']['mean_time'] > 0, f"Invalid timing for {config_name}"
            assert result['param_count'] > 0, f"Invalid parameter count for {config_name}"
    
    def test_resolution_scaling(self, benchmark):
        """Test how performance scales with input resolution."""
        resolutions = [32, 64, 128, 256]
        model = MockFourierNeuralOperator(3, 1, [16, 16], 64)
        model.eval()
        
        resolution_results = {}
        
        for resolution in resolutions:
            test_data = torch.randn(2, 3, resolution, resolution)
            
            try:
                with torch.no_grad():
                    timing_result = benchmark.time_function(model, test_data)
                    memory_result = benchmark.measure_memory_usage(model, test_data)
                
                resolution_results[resolution] = {
                    'timing': timing_result,
                    'memory': memory_result,
                    'pixels_per_second': (resolution * resolution * 2) / timing_result['mean_time']
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    resolution_results[resolution] = {'error': 'OOM'}
                else:
                    raise
        
        benchmark.results['resolution_scaling'] = resolution_results
        
        # At least smaller resolutions should work
        successful_resolutions = [r for r, result in resolution_results.items() 
                                if 'error' not in result]
        assert len(successful_resolutions) >= 2, "Too few resolutions completed successfully"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_utilization(self, benchmark):
        """Test GPU utilization and performance."""
        model = MockFourierNeuralOperator(3, 1, [16, 16], 128)
        model = model.cuda()
        model.eval()
        
        test_data = torch.randn(8, 3, 128, 128).cuda()
        
        # Measure GPU performance
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(100):
                output = model(test_data)
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100
        throughput = test_data.shape[0] / avg_time
        
        gpu_results = {
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'gpu_memory_used': torch.cuda.memory_allocated() / 1024 / 1024  # MB
        }
        
        benchmark.results['gpu_utilization'] = gpu_results
        
        # GPU should provide reasonable performance
        assert avg_time < 0.5, "GPU inference too slow"
        assert throughput > 10, "GPU throughput too low"


@pytest.mark.slow
@pytest.mark.performance
class TestRegressionBenchmarks:
    """Performance regression benchmarks."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    def test_baseline_performance_regression(self, benchmark):
        """Test against baseline performance metrics."""
        # Standard test configuration
        model = MockFourierNeuralOperator(3, 1, [16, 16], 64)
        model.eval()
        test_data = torch.randn(8, 3, 64, 64)
        
        with torch.no_grad():
            timing_result = benchmark.time_function(model, test_data)
        
        # Check against baseline
        baseline_time = benchmark.baseline_times['fno_forward']
        current_time = timing_result['mean_time']
        
        # Allow 50% performance degradation before failing
        max_allowed_time = baseline_time * 1.5
        
        assert current_time < max_allowed_time, \
            f"Performance regression detected: {current_time:.3f}s > {max_allowed_time:.3f}s"
        
        # Store current performance as potential new baseline
        benchmark.results['baseline_check'] = {
            'current_time': current_time,
            'baseline_time': baseline_time,
            'performance_ratio': current_time / baseline_time
        }
    
    def test_memory_regression(self, benchmark):
        """Test for memory usage regression."""
        model = MockTransformerNeuralOperator(3, 1, 128, 4, 4)
        model.eval()
        test_data = torch.randn(4, 3, 64, 64)
        
        with torch.no_grad():
            memory_result = benchmark.measure_memory_usage(model, test_data)
        
        # Define reasonable memory limits
        max_cpu_memory = 500  # MB
        max_gpu_memory = 1000  # MB
        
        assert memory_result['cpu_memory_delta'] < max_cpu_memory, \
            f"CPU memory usage too high: {memory_result['cpu_memory_delta']:.1f}MB"
        
        if torch.cuda.is_available():
            assert memory_result['gpu_memory_delta'] < max_gpu_memory, \
                f"GPU memory usage too high: {memory_result['gpu_memory_delta']:.1f}MB"
        
        benchmark.results['memory_regression'] = memory_result


@pytest.fixture
def performance_report():
    """Generate performance report after all benchmarks."""
    yield
    # This would typically generate a performance report
    # For now, we just ensure benchmarks completed