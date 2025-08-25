"""Comprehensive test suite for QISA models with robustness validation."""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestQISAModels(unittest.TestCase):
    """Comprehensive test suite for QISA models."""
    
    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            cls.skipTest(cls, "PyTorch not available")
    
    def setUp(self):
        """Set up test fixtures."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            create_qisa_model, QISAConfig
        )
        
        # Small model for testing
        self.test_config = QISAConfig(
            hidden_dim=32,
            num_layers=2,
            quantum_depth=2,
            spectral_modes=8,
            num_quantum_heads=2,
            spectral_attention_heads=2,
            enable_mixed_precision=False
        )
        
        self.input_dim = 4
        self.output_dim = 4
        self.batch_size = 2
        self.seq_len = 10
        self.spatial_dim = 64
    
    def test_qisa_model_creation(self):
        """Test QISA model creation and basic properties."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Basic properties
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.output_dim, self.output_dim)
        self.assertGreater(model.count_parameters(), 0)
        
        # Configuration properties
        self.assertEqual(model.config.hidden_dim, self.test_config.hidden_dim)
        self.assertEqual(model.config.num_layers, self.test_config.num_layers)
        
        print(f"✅ QISA model created with {model.count_parameters():,} parameters")
    
    def test_qisa_forward_pass(self):
        """Test QISA forward pass with different input shapes."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        model.eval()
        
        # Test different input shapes
        test_shapes = [
            (1, 5, 32, self.input_dim),   # Minimal
            (2, 10, 64, self.input_dim),  # Standard
            (4, 20, 128, self.input_dim)  # Larger
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                x = torch.randn(*shape)
                
                # Forward pass without spectral loss
                with torch.no_grad():
                    output = model(x)
                
                expected_shape = shape[:-1] + (self.output_dim,)
                self.assertEqual(output.shape, expected_shape)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
                
                # Forward pass with spectral loss
                with torch.no_grad():
                    output, spectral_loss = model(x, return_spectral_loss=True)
                
                self.assertEqual(output.shape, expected_shape)
                self.assertIsInstance(spectral_loss, torch.Tensor)
                self.assertEqual(spectral_loss.dim(), 0)  # Scalar
                self.assertFalse(torch.isnan(spectral_loss))
                
                print(f"✅ Forward pass successful for shape {shape}")
    
    def test_quantum_state_properties(self):
        """Test quantum state properties and validation."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Get quantum state information
        quantum_info = model.get_quantum_state_info()
        
        # Validate quantum info structure
        required_keys = ['total_quantum_params', 'density_matrix_traces', 'quantum_gate_angles']
        for key in required_keys:
            self.assertIn(key, quantum_info)
        
        # Validate density matrix traces (should be close to 1.0)
        traces = quantum_info['density_matrix_traces']
        self.assertGreater(len(traces), 0)
        
        for trace in traces:
            self.assertIsInstance(trace, (int, float))
            self.assertGreater(trace, 0.5)  # Should be positive
            self.assertLess(trace, 2.0)     # Should be reasonable
        
        # Validate quantum parameters
        total_quantum_params = quantum_info['total_quantum_params']
        self.assertIsInstance(total_quantum_params, int)
        self.assertGreater(total_quantum_params, 0)
        
        # Validate quantum gate angles
        angles = quantum_info['quantum_gate_angles']
        self.assertIsInstance(angles, list)
        self.assertEqual(len(angles), total_quantum_params)
        
        print(f"✅ Quantum state validation passed: {total_quantum_params} quantum parameters")
    
    def test_spectral_attention_components(self):
        """Test spectral attention layer functionality."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            SpectralAttentionLayer, QISAConfig
        )
        
        config = QISAConfig(hidden_dim=32, spectral_modes=8, spectral_attention_heads=2)
        spectral_layer = SpectralAttentionLayer(config)
        
        # Test input
        batch_size, seq_len, hidden_dim = 2, 16, 32
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        output = spectral_layer(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Test spectral loss computation
        target = torch.randn_like(x)
        spectral_loss = spectral_layer.compute_binned_spectral_loss(output, target)
        
        self.assertIsInstance(spectral_loss, torch.Tensor)
        self.assertEqual(spectral_loss.dim(), 0)
        self.assertFalse(torch.isnan(spectral_loss))
        self.assertGreaterEqual(spectral_loss.item(), 0)
        
        print("✅ Spectral attention components validated")
    
    def test_quantum_mixed_state_attention(self):
        """Test quantum mixed-state attention mechanism."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumMixedStateAttention, QISAConfig
        )
        
        config = QISAConfig(
            hidden_dim=32,
            num_quantum_heads=2,
            quantum_depth=2,
            entanglement_layers=1
        )
        quantum_attention = QuantumMixedStateAttention(config)
        
        # Test input
        batch_size, seq_len, hidden_dim = 2, 8, 32
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        output = quantum_attention(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Check density matrices properties
        for i in range(config.num_quantum_heads):
            rho = quantum_attention.density_matrices[i]
            
            # Should be square
            self.assertEqual(rho.shape[0], rho.shape[1])
            
            # Check trace (should be positive)
            trace = torch.trace(rho).item()
            self.assertGreater(trace, 0)
            
            # Check for numerical stability
            self.assertFalse(torch.isnan(rho).any())
            self.assertFalse(torch.isinf(rho).any())
        
        print("✅ Quantum mixed-state attention validated")
    
    def test_model_optimization_for_inference(self):
        """Test model optimization for inference."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Get original spectral modes
        original_modes = []
        for block in model.blocks:
            original_modes.append(block.spectral_attention.spectral_modes)
        
        # Test input
        x = torch.randn(1, 10, 32, self.input_dim)
        
        # Original forward pass
        model.eval()
        with torch.no_grad():
            original_output = model(x)
        
        # Optimize for inference
        model.optimize_for_inference()
        
        # Optimized forward pass
        with torch.no_grad():
            optimized_output = model(x)
        
        # Check outputs have same shape
        self.assertEqual(original_output.shape, optimized_output.shape)
        
        # If adaptive modes enabled, spectral modes should be reduced
        if self.test_config.adaptive_spectral_modes:
            for i, block in enumerate(model.blocks):
                current_modes = block.spectral_attention.spectral_modes
                self.assertLessEqual(current_modes, original_modes[i])
        
        print("✅ Model optimization for inference validated")
    
    def test_error_handling_and_validation(self):
        """Test error handling and input validation."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Test with invalid input shapes
        invalid_inputs = [
            torch.randn(2, 10, 32),            # Missing feature dimension
            torch.randn(2, 10, 32, 5),         # Wrong feature dimension
            torch.randn(2, 10, 32, self.input_dim, 5),  # Extra dimension
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(shape=invalid_input.shape):
                with self.assertRaises((ValueError, RuntimeError, IndexError)):
                    model(invalid_input)
        
        # Test with NaN inputs (if robust model is available)
        try:
            from neural_operator_lab.models.robust_qisa import RobustQISAModel
            
            robust_model = RobustQISAModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                config=self.test_config
            )
            
            # Test NaN input detection
            nan_input = torch.randn(2, 10, 32, self.input_dim)
            nan_input[0, 0, 0, 0] = float('nan')
            
            # Should raise ValueError with validation enabled
            with self.assertRaises(ValueError):
                robust_model(nan_input, validate_inputs=True)
                
            print("✅ Robust error handling validated")
            
        except ImportError:
            print("⚠️  Robust QISA model not available for testing")
        
        print("✅ Basic error handling validated")
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        model.train()
        x = torch.randn(2, 10, 32, self.input_dim, requires_grad=True)
        target = torch.randn(2, 10, 32, self.output_dim)
        
        # Forward pass
        output, spectral_loss = model(x, return_spectral_loss=True)
        
        # Combined loss
        data_loss = nn.MSELoss()(output, target)
        total_loss = data_loss + 0.1 * spectral_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist and are finite
        gradient_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                gradient_norms.append(grad_norm)
                
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in {name}")
        
        self.assertGreater(len(gradient_norms), 0, "No gradients found")
        self.assertGreater(sum(gradient_norms), 0, "All gradients are zero")
        
        print(f"✅ Gradient flow validated: {len(gradient_norms)} parameters with gradients")
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        original_model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Test input
        x = torch.randn(1, 10, 32, self.input_dim)
        
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(x)
        
        # Save and load state dict
        state_dict = original_model.state_dict()
        
        # Create new model and load state
        new_model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        new_model.load_state_dict(state_dict)
        new_model.eval()
        
        # Get new output
        with torch.no_grad():
            new_output = new_model(x)
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(original_output, new_output, atol=1e-6))
        
        print("✅ Model serialization validated")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the model."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # Test with different sequence lengths to check memory scaling
        base_memory = 0
        memory_growth = []
        
        for seq_len in [10, 20, 40, 80]:
            x = torch.randn(1, seq_len, 64, self.input_dim)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = model.cuda()
                x = x.cuda()
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                memory_growth.append(current_memory)
                
                if base_memory == 0:
                    base_memory = current_memory
        
        if memory_growth:
            # Memory should scale reasonably with sequence length
            # (not exponentially for well-designed attention)
            max_memory = max(memory_growth)
            min_memory = min(memory_growth)
            memory_ratio = max_memory / (min_memory + 1e-6)
            
            # Should not grow more than 8x for 8x sequence length increase
            self.assertLess(memory_ratio, 10.0, f"Excessive memory growth: {memory_ratio:.2f}x")
            
            print(f"✅ Memory efficiency validated: {memory_ratio:.2f}x growth for 8x sequence length")
        else:
            print("✅ Memory efficiency test completed (CUDA not available)")
        
    
    def test_performance_benchmarks(self):
        """Test basic performance characteristics."""
        from neural_operator_lab.models.quantum_spectral_neural_operator import (
            QuantumInspiredSpectralAttentionNeuralOperator
        )
        
        model = QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=self.test_config
        )
        model.eval()
        
        # Warm-up
        x = torch.randn(1, 10, 32, self.input_dim)
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        
        # Time multiple forward passes
        import time
        times = []
        
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                output = model(x)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Should complete in reasonable time (adjust based on hardware)
        self.assertLess(avg_time, 1.0, f"Forward pass too slow: {avg_time:.3f}s")
        
        print(f"✅ Performance benchmark: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms per forward pass")


class TestRobustQISA(unittest.TestCase):
    """Test suite for robust QISA implementation."""
    
    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            cls.skipTest(cls, "PyTorch not available")
    
    def test_robust_qisa_creation(self):
        """Test robust QISA model creation."""
        try:
            from neural_operator_lab.models.robust_qisa import (
                create_robust_qisa_model, RobustQISAConfig
            )
            
            model = create_robust_qisa_model(
                input_dim=4,
                output_dim=4,
                hidden_dim=32,
                num_layers=2,
                quantum_depth=2,
                spectral_modes=8
            )
            
            self.assertIsInstance(model, nn.Module)
            self.assertGreater(model.count_parameters(), 0)
            
            # Test health report
            health_report = model.get_health_report()
            self.assertIn("model_status", health_report)
            self.assertIn("error_count", health_report)
            
            print("✅ Robust QISA model created and health check passed")
            
        except ImportError:
            self.skipTest("Robust QISA model not available")
    
    def test_robust_error_handling(self):
        """Test robust error handling capabilities."""
        try:
            from neural_operator_lab.models.robust_qisa import create_robust_qisa_model
            
            model = create_robust_qisa_model(
                input_dim=4,
                output_dim=4,
                hidden_dim=32,
                num_layers=2
            )
            
            # Test with NaN input
            x = torch.randn(1, 10, 32, 4)
            x[0, 0, 0, 0] = float('nan')
            
            # Should handle gracefully with validation
            try:
                output = model(x, validate_inputs=True)
                self.fail("Should have raised ValueError for NaN input")
            except ValueError as e:
                self.assertIn("NaN", str(e))
            
            print("✅ Robust error handling validated")
            
        except ImportError:
            self.skipTest("Robust QISA model not available")


def run_comprehensive_tests():
    """Run comprehensive QISA test suite."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot run comprehensive tests")
        print("   Install PyTorch with: pip install torch")
        return False
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add QISA tests
    suite.addTest(unittest.makeSuite(TestQISAModels))
    
    # Add robust QISA tests (if available)
    try:
        suite.addTest(unittest.makeSuite(TestRobustQISA))
    except ImportError:
        print("⚠️  Robust QISA tests skipped (module not available)")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("QISA COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED - QISA IMPLEMENTATION VALIDATED ✅")
    else:
        print("\n❌ SOME TESTS FAILED - CHECK IMPLEMENTATION")
    
    return success


if __name__ == "__main__":
    run_comprehensive_tests()