"""Comprehensive test suite for advanced neural operator implementations.

Tests all new architectures, performance optimizations, and research capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import time
import tempfile
import json
from pathlib import Path
import warnings

# Import new implementations
from neural_operator_lab.models.geometric import GeometricNeuralOperator
from neural_operator_lab.models.physics_informed import (
    PhysicsInformedUniversalNeuralOperator,
    NavierStokesConstraint,
    WaveEquationConstraint
)
from neural_operator_lab.models.adaptive import MultiResolutionAdaptiveNeuralOperator
from neural_operator_lab.research.advanced_benchmarking import (
    AdvancedBenchmarkSuite,
    create_synthetic_benchmark_data
)
from neural_operator_lab.optimization.advanced_performance import (
    PerformanceOptimizer,
    PerformanceConfig,
    optimize_performance
)
from neural_operator_lab.scaling.intelligent_auto_scaler import (
    IntelligentAutoScaler,
    ResourceMonitor,
    WorkloadMetrics
)
from neural_operator_lab.monitoring.comprehensive_logging import (
    ComprehensiveLogger,
    setup_logging
)
from neural_operator_lab.monitoring.advanced_error_handler import (
    AdvancedErrorHandler,
    robust_function
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestAdvancedArchitectures:
    """Test suite for advanced neural operator architectures."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def sample_data(self, device):
        """Create sample PDE data for testing."""
        batch_size = 4
        num_points = 64
        spatial_dim = 3
        input_dim = 3
        output_dim = 1
        
        x = torch.randn(batch_size, num_points, input_dim, device=device)
        coords = torch.randn(batch_size, num_points, spatial_dim, device=device)
        targets = torch.randn(batch_size, num_points, output_dim, device=device)
        
        return x, coords, targets
    
    def test_geometric_neural_operator(self, device, sample_data):
        """Test Geometric Neural Operator implementation."""
        x, coords, targets = sample_data
        
        model = GeometricNeuralOperator(
            input_dim=3,
            output_dim=1,
            hidden_dim=64,
            num_layers=2,
            coordinate_dim=3
        ).to(device)
        
        # Test forward pass
        output = model(x, coords)
        assert output.shape == targets.shape
        
        # Test with geometric loss computation
        coords.requires_grad_(True)
        output, geom_losses = model(x, coords, return_geometric_loss=True)
        
        assert isinstance(geom_losses, dict)
        assert len(geom_losses) > 0
        
        # Test loss computation
        loss = model.get_loss(output, targets, {'coords': coords})
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_physics_informed_universal_operator(self, device, sample_data):
        """Test Physics-Informed Universal Neural Operator."""
        x, coords, targets = sample_data
        
        # Test with Navier-Stokes constraint
        navier_stokes = NavierStokesConstraint(viscosity=0.01)
        
        model = PhysicsInformedUniversalNeuralOperator(
            input_dim=3,
            output_dim=4,  # u, v, w, p
            hidden_dim=64,
            num_layers=2,
            physics_constraints=[navier_stokes],
            conserved_quantities=["mass", "momentum"]
        ).to(device)
        
        # Adjust targets for 4 outputs
        targets_4d = torch.randn(x.shape[0], x.shape[1], 4, device=device)
        
        # Test forward pass without physics
        output = model(x, coords)
        assert output.shape == targets_4d.shape
        
        # Test forward pass with physics
        coords.requires_grad_(True)
        output, physics_losses = model(x, coords, pde_type="navier_stokes", return_physics_loss=True)
        
        assert isinstance(physics_losses, dict)
        
        # Test universal capability
        model.transfer_to_pde("custom_pde", output_dim=2)
        assert "custom_pde" in model.pde_embeddings
    
    def test_adaptive_neural_operator(self, device, sample_data):
        """Test Multi-Resolution Adaptive Neural Operator."""
        x, coords, targets = sample_data
        
        model = MultiResolutionAdaptiveNeuralOperator(
            input_dim=3,
            output_dim=1,
            hidden_dim=64,
            num_layers=2,
            num_scales=2,
            refinement_threshold=0.1,
            max_refinement_levels=2
        ).to(device)
        
        # Test forward pass with adaptive refinement
        output = model(x, coords, adaptive_refinement=True)
        assert output.shape[0] == x.shape[0]  # Same batch size
        assert output.shape[2] == targets.shape[2]  # Same feature dim
        
        # Test with refinement info
        output, refinement_info = model(x, coords, return_refinement_info=True)
        
        assert isinstance(refinement_info, dict)
        
        # Test cost estimation
        cost_estimate = model.estimate_computational_cost(x.shape[1])
        assert isinstance(cost_estimate, dict)
        assert 'total_operations' in cost_estimate


class TestResearchBenchmarking:
    """Test suite for research benchmarking capabilities."""
    
    def test_advanced_benchmark_suite(self):
        """Test advanced benchmarking suite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_suite = AdvancedBenchmarkSuite(
                output_dir=temp_dir,
                num_statistical_runs=2  # Reduced for testing
            )
            
            # Create simple test model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(64, 1)
                
                def forward(self, x):
                    # Reshape to 2D for linear layer
                    batch_size, seq_len, features = x.shape
                    x = x.view(-1, features)
                    out = self.linear(x)
                    return out.view(batch_size, seq_len, 1)
            
            model = SimpleModel()
            
            # Create test dataset
            test_data = create_synthetic_benchmark_data(
                batch_size=2,
                spatial_resolution=8,  # Small for testing
                num_samples=4
            )
            
            # Run benchmark
            results = benchmark_suite.benchmark_model(
                model=model,
                test_data=test_data,
                model_name="simple_test",
                dataset_name="synthetic",
                device="cpu"
            )
            
            assert len(results) > 0
            assert all(isinstance(r.value, (int, float)) for r in results)
            
            # Test statistics
            stats = benchmark_suite.get_error_statistics()
            assert isinstance(stats, dict)
    
    def test_comparative_study(self):
        """Test comparative study functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_suite = AdvancedBenchmarkSuite(
                output_dir=temp_dir,
                num_statistical_runs=2
            )
            
            # Create baseline and novel models
            class BaselineModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    batch_size, seq_len, features = x.shape
                    x = x.view(-1, features)
                    out = self.net(x)
                    return out.view(batch_size, seq_len, 1)
            
            class NovelModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.GELU(),
                        nn.Linear(64, 1)
                    )
                
                def forward(self, x):
                    batch_size, seq_len, features = x.shape
                    x = x.view(-1, features)
                    out = self.net(x)
                    return out.view(batch_size, seq_len, 1)
            
            baseline_models = {"baseline": BaselineModel()}
            novel_models = {"novel": NovelModel()}
            
            test_datasets = {
                "synthetic": create_synthetic_benchmark_data(
                    batch_size=2, spatial_resolution=8, num_samples=4
                )
            }
            
            # Run comparative study
            study_result = benchmark_suite.comparative_study(
                baseline_models=baseline_models,
                novel_models=novel_models,
                test_datasets=test_datasets,
                study_name="test_study",
                device="cpu"
            )
            
            assert len(study_result.baseline_results) > 0
            assert len(study_result.novel_results) > 0
            assert len(study_result.recommendations) > 0


class TestPerformanceOptimization:
    """Test suite for performance optimization."""
    
    def test_performance_optimizer(self):
        """Test performance optimizer."""
        config = PerformanceConfig(
            mixed_precision=False,  # Disable for CPU testing
            gradient_checkpointing=True,
            compile_model=False
        )
        
        optimizer = PerformanceOptimizer(config)
        
        # Test model optimization
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimized_model = optimizer.optimize_model(model)
        assert optimized_model is not None
        
        # Test benchmarking
        benchmark_results = optimizer.benchmark_model(
            model=optimized_model,
            input_shape=(4, 64),
            num_iterations=5,
            warmup_iterations=2
        )
        
        assert 'mean_forward_time_ms' in benchmark_results
        assert 'throughput_samples_per_sec' in benchmark_results
        assert benchmark_results['mean_forward_time_ms'] > 0
    
    def test_performance_decorator(self):
        """Test performance optimization decorator."""
        
        @optimize_performance()
        def sample_training_function(model, data):
            output = model(data)
            loss = output.mean()
            return loss
        
        model = nn.Linear(10, 1)
        data = torch.randn(4, 10)
        
        loss = sample_training_function(model, data)
        assert isinstance(loss, torch.Tensor)
        
        # Check that performance optimizer was attached
        assert hasattr(sample_training_function, 'performance_optimizer')


class TestAutoScaling:
    """Test suite for auto-scaling capabilities."""
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        monitor = ResourceMonitor(monitoring_interval=0.1)
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect a few samples
        monitor.stop_monitoring()
        
        # Check metrics collection
        recent_metrics = monitor.get_recent_metrics(10.0)
        assert len(recent_metrics) > 0
        
        avg_metrics = monitor.get_average_metrics(10.0)
        assert avg_metrics is not None
        assert avg_metrics.cpu_percent >= 0
        assert avg_metrics.memory_percent >= 0
    
    def test_intelligent_auto_scaler(self):
        """Test intelligent auto-scaler."""
        auto_scaler = IntelligentAutoScaler(scaling_interval=0.5)
        
        # Set performance targets
        targets = {
            'throughput_samples_per_sec': 100.0,
            'latency_ms': 50.0
        }
        auto_scaler.set_performance_targets(targets)
        
        # Test workload metrics update
        workload_metrics = WorkloadMetrics(
            timestamp=time.time(),
            batch_size=32,
            throughput_samples_per_sec=80.0,
            loss_value=0.5,
            gradient_norm=0.1,
            learning_rate=0.001,
            epoch=1,
            step=100,
            estimated_remaining_time=3600,
            queue_size=0
        )
        
        auto_scaler.update_workload_metrics(workload_metrics)
        
        # Test status retrieval
        status = auto_scaler.get_scaling_status()
        assert isinstance(status, dict)
        assert 'performance_targets' in status
        assert 'scaling_active' in status


class TestLoggingAndErrorHandling:
    """Test suite for logging and error handling."""
    
    def test_comprehensive_logger(self):
        """Test comprehensive logging system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ComprehensiveLogger(
                log_dir=temp_dir,
                enable_wandb=False,
                enable_tensorboard=False
            )
            
            # Test basic logging
            logger.info("Test info message", extra_param="test_value")
            logger.error("Test error message")
            
            # Test training metrics logging
            metrics = {
                'loss': 0.5,
                'accuracy': 0.85,
                'learning_rate': 0.001
            }
            logger.log_training_metrics(epoch=1, metrics=metrics)
            
            # Test model info logging
            model = nn.Linear(10, 1)
            logger.log_model_info(model, input_shape=(4, 10))
            
            # Check log files were created
            log_files = list(Path(temp_dir).glob("*.log"))
            assert len(log_files) > 0
    
    def test_advanced_error_handler(self):
        """Test advanced error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "errors.log"
            error_handler = AdvancedErrorHandler(log_file=str(log_file))
            
            # Test error handling
            try:
                raise ValueError("Test error")
            except Exception as e:
                success, result = error_handler.handle_error(
                    e, "test_function", "test_module"
                )
            
            # Check error was logged
            stats = error_handler.get_error_statistics()
            assert stats['total_errors'] > 0
            assert 'ValueError' in stats['error_types']
    
    def test_robust_function_decorator(self):
        """Test robust function decorator."""
        error_handler = AdvancedErrorHandler()
        
        @robust_function(error_handler=error_handler, max_retries=2)
        def flaky_function(fail_count=0):
            if hasattr(flaky_function, '_call_count'):
                flaky_function._call_count += 1
            else:
                flaky_function._call_count = 1
            
            if flaky_function._call_count <= fail_count:
                raise RuntimeError(f"Simulated failure {flaky_function._call_count}")
            
            return "Success"
        
        # Test successful execution after retries
        result = flaky_function(fail_count=1)
        assert result == "Success"
        assert flaky_function._call_count == 2


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_training_workflow(self):
        """Test complete training workflow with all components."""
        device = "cpu"  # Use CPU for reliable testing
        
        # Create model with advanced architecture
        model = GeometricNeuralOperator(
            input_dim=3,
            output_dim=1,
            hidden_dim=32,  # Smaller for testing
            num_layers=2,
            coordinate_dim=3
        ).to(device)
        
        # Setup performance optimization
        perf_config = PerformanceConfig(
            mixed_precision=False,
            compile_model=False
        )
        perf_optimizer = PerformanceOptimizer(perf_config)
        model = perf_optimizer.optimize_model(model)
        
        # Create sample data
        batch_size = 2
        num_points = 16
        x = torch.randn(batch_size, num_points, 3, device=device)
        coords = torch.randn(batch_size, num_points, 3, device=device)
        targets = torch.randn(batch_size, num_points, 1, device=device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop with performance monitoring
        model.train()
        for step in range(3):  # Just a few steps for testing
            with perf_optimizer.optimized_forward():
                coords.requires_grad_(True)
                output = model(x, coords)
                loss = nn.MSELoss()(output, targets)
            
            # Backward pass
            perf_optimizer.optimized_backward(loss, optimizer)
            optimizer.zero_grad()
        
        # Get performance report
        perf_report = perf_optimizer.get_performance_report()
        assert 'performance_metrics' in perf_report
        assert 'forward_times' in perf_report['performance_metrics']
    
    def test_research_pipeline_integration(self):
        """Test research pipeline with benchmarking and analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup benchmark suite
            benchmark_suite = AdvancedBenchmarkSuite(
                output_dir=temp_dir,
                num_statistical_runs=2
            )
            
            # Create models to compare
            class BaselineModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Linear(64, 1)
                
                def forward(self, x):
                    return self.net(x.view(x.shape[0], -1)).unsqueeze(-1)
            
            baseline_model = BaselineModel()
            
            # Advanced model
            novel_model = GeometricNeuralOperator(
                input_dim=64,
                output_dim=1,
                hidden_dim=32,
                num_layers=1,
                coordinate_dim=3
            )
            
            # Create test data
            test_data = create_synthetic_benchmark_data(
                batch_size=2,
                spatial_resolution=4,  # Very small for testing
                input_dim=64,
                output_dim=1,
                num_samples=4
            )
            
            # Run comparative study
            study_result = benchmark_suite.comparative_study(
                baseline_models={"baseline": baseline_model},
                novel_models={"novel": novel_model},
                test_datasets={"test": test_data},
                study_name="integration_test",
                device="cpu"
            )
            
            # Generate research report
            report_path = benchmark_suite.generate_research_report(
                study_result,
                "integration_test",
                include_visualizations=False  # Skip plots for testing
            )
            
            assert Path(report_path).exists()
            
            # Verify report content
            with open(report_path, 'r') as f:
                report_content = f.read()
                assert "Neural Operator Comparative Study" in report_content
                assert "Statistical Analysis" in report_content


def run_comprehensive_tests():
    """Run all tests and provide summary."""
    print("🧪 Running Comprehensive Test Suite for Advanced Neural Operator Framework")
    print("=" * 80)
    
    # Run tests with pytest
    test_files = [
        "TestAdvancedArchitectures",
        "TestResearchBenchmarking", 
        "TestPerformanceOptimization",
        "TestAutoScaling",
        "TestLoggingAndErrorHandling",
        "TestIntegration"
    ]
    
    results = {}
    
    for test_class in test_files:
        print(f"\n📋 Running {test_class}...")
        try:
            # In a real implementation, we would use pytest.main()
            # For now, we'll simulate test execution
            
            if test_class == "TestAdvancedArchitectures":
                test_instance = TestAdvancedArchitectures()
                device = "cpu"
                sample_data = (
                    torch.randn(2, 32, 3),
                    torch.randn(2, 32, 3), 
                    torch.randn(2, 32, 1)
                )
                
                test_instance.test_geometric_neural_operator(device, sample_data)
                test_instance.test_physics_informed_universal_operator(device, sample_data)
                test_instance.test_adaptive_neural_operator(device, sample_data)
                
            elif test_class == "TestResearchBenchmarking":
                test_instance = TestResearchBenchmarking()
                test_instance.test_advanced_benchmark_suite()
                test_instance.test_comparative_study()
                
            elif test_class == "TestPerformanceOptimization":
                test_instance = TestPerformanceOptimization()
                test_instance.test_performance_optimizer()
                test_instance.test_performance_decorator()
                
            elif test_class == "TestAutoScaling":
                test_instance = TestAutoScaling()
                test_instance.test_resource_monitor()
                test_instance.test_intelligent_auto_scaler()
                
            elif test_class == "TestLoggingAndErrorHandling":
                test_instance = TestLoggingAndErrorHandling()
                test_instance.test_comprehensive_logger()
                test_instance.test_advanced_error_handler()
                test_instance.test_robust_function_decorator()
                
            elif test_class == "TestIntegration":
                test_instance = TestIntegration()
                test_instance.test_end_to_end_training_workflow()
                test_instance.test_research_pipeline_integration()
            
            results[test_class] = "✅ PASSED"
            print(f"   ✅ {test_class} - All tests passed")
            
        except Exception as e:
            results[test_class] = f"❌ FAILED: {str(e)}"
            print(f"   ❌ {test_class} - Tests failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    for test_class, result in results.items():
        print(f"{result:<50} {test_class}")
    
    print(f"\n🎯 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🏆 ALL TESTS PASSED - Advanced Neural Operator Framework is ready!")
        return True
    else:
        print("⚠️  Some tests failed - Review implementation before deployment")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)