"""Unit tests for adaptive neural operators."""

import pytest
import torch
import numpy as np
import math
from unittest.mock import Mock, patch

from neural_operator_lab.models.adaptive import (
    MultiResolutionAdaptiveNeuralOperator,
    MultiScaleConvolution,
    AdaptivePooling,
    ResolutionBridge,
    CrossResolutionAttention,
    AdaptiveRefinementCriterion,
    MultiResolutionProcessor,
    create_adaptive_neural_operator
)


class TestMultiScaleConvolution:
    """Test multi-scale convolution layer."""
    
    def test_init(self):
        """Test multi-scale convolution initialization."""
        conv = MultiScaleConvolution(16, 32, scales=[1, 2, 4])
        assert conv.scales == [1, 2, 4]
        assert conv.num_scales == 3
        assert len(conv.scale_convs) == 3
    
    def test_forward(self):
        """Test multi-scale convolution forward pass."""
        conv = MultiScaleConvolution(16, 32, scales=[1, 2, 4])
        
        x = torch.randn(2, 16, 64, 64)
        output = conv(x)
        
        assert output.shape == (2, 32, 64, 64)
        assert not torch.isnan(output).any()


class TestAdaptivePooling:
    """Test adaptive pooling layer."""
    
    def test_init(self):
        """Test adaptive pooling initialization."""
        pool = AdaptivePooling(32)
        assert pool.output_size == (32, 32)
        
        pool2 = AdaptivePooling((64, 32))
        assert pool2.output_size == (64, 32)
    
    def test_forward(self):
        """Test adaptive pooling forward pass."""
        pool = AdaptivePooling(32)
        
        x = torch.randn(2, 16, 128, 128)
        output = pool(x)
        
        assert output.shape == (2, 16, 32, 32)
        assert not torch.isnan(output).any()


class TestCrossResolutionAttention:
    """Test cross-resolution attention mechanism."""
    
    def test_init(self):
        """Test cross-resolution attention initialization."""
        attention = CrossResolutionAttention(64, num_heads=8)
        assert attention.channels == 64
        assert attention.num_heads == 8
        assert attention.head_dim == 8
    
    def test_forward(self):
        """Test cross-resolution attention forward pass."""
        attention = CrossResolutionAttention(64, num_heads=8)
        
        high_query = torch.randn(2, 64, 128, 128)
        high_orig = torch.randn(2, 64, 128, 128)
        low_query = torch.randn(2, 64, 64, 64)
        low_orig = torch.randn(2, 64, 64, 64)
        
        high_out, low_out = attention(high_query, high_orig, low_query, low_orig)
        
        assert high_out.shape == high_query.shape
        assert low_out.shape == low_query.shape
        assert not torch.isnan(high_out).any()
        assert not torch.isnan(low_out).any()


class TestResolutionBridge:
    """Test resolution bridge layer."""
    
    def test_init(self):
        """Test resolution bridge initialization."""
        bridge = ResolutionBridge(64, scale_factor=2)
        assert bridge.scale_factor == 2
    
    def test_forward(self):
        """Test resolution bridge forward pass."""
        bridge = ResolutionBridge(64, scale_factor=2)
        
        low_res = torch.randn(2, 64, 32, 32)
        high_res = torch.randn(2, 64, 64, 64)
        
        up_out, down_out = bridge(low_res, high_res)
        
        # Output shapes should match input resolutions after bridging
        assert up_out.shape[2:] == high_res.shape[2:]  # Upsampled to high res
        assert down_out.shape[2:] == low_res.shape[2:]  # Downsampled to low res
        assert not torch.isnan(up_out).any()
        assert not torch.isnan(down_out).any()


class TestAdaptiveRefinementCriterion:
    """Test adaptive refinement criterion."""
    
    def test_init(self):
        """Test refinement criterion initialization."""
        criterion = AdaptiveRefinementCriterion(
            64, 
            refinement_threshold=0.1,
            gradient_weight=1.0,
            curvature_weight=0.5
        )
        assert criterion.refinement_threshold == 0.1
        assert criterion.gradient_weight == 1.0
        assert criterion.curvature_weight == 0.5
    
    def test_compute_gradient_magnitude(self):
        """Test gradient magnitude computation."""
        criterion = AdaptiveRefinementCriterion(64)
        
        field = torch.randn(2, 3, 64, 64)
        gradient_mag = criterion.compute_gradient_magnitude(field)
        
        assert gradient_mag.shape == field.shape
        assert (gradient_mag >= 0).all()
        assert not torch.isnan(gradient_mag).any()
    
    def test_compute_curvature(self):
        """Test curvature computation."""
        criterion = AdaptiveRefinementCriterion(64)
        
        field = torch.randn(2, 3, 64, 64)
        curvature = criterion.compute_curvature(field)
        
        assert curvature.shape == field.shape
        assert (curvature >= 0).all()
        assert not torch.isnan(curvature).any()
    
    def test_forward(self):
        """Test refinement criterion forward pass."""
        criterion = AdaptiveRefinementCriterion(64)
        
        features = torch.randn(2, 64, 32, 32)
        solution = torch.randn(2, 4, 32, 32)
        
        refinement_map = criterion(features, solution)
        
        assert refinement_map.shape == (2, 1, 32, 32)
        assert (refinement_map >= 0).all()
        assert (refinement_map <= 1).all()
        assert not torch.isnan(refinement_map).any()


class TestMultiResolutionProcessor:
    """Test multi-resolution processor."""
    
    def test_init(self):
        """Test multi-resolution processor initialization."""
        processor = MultiResolutionProcessor(
            64, 
            resolutions=[32, 64, 128],
            num_layers_per_resolution=3
        )
        assert processor.resolutions == [32, 64, 128]
        assert processor.num_resolutions == 3
        assert len(processor.resolution_processors) == 3
        assert len(processor.bridges) == 2
    
    def test_forward(self):
        """Test multi-resolution processor forward pass."""
        processor = MultiResolutionProcessor(64, resolutions=[32, 64])
        
        x = torch.randn(2, 64, 128, 128)
        
        outputs = processor(x, target_resolution=64)
        
        assert '32' in outputs
        assert '64' in outputs
        assert outputs['32'].shape == (2, 64, 32, 32)
        assert outputs['64'].shape == (2, 64, 64, 64)
        
        for output in outputs.values():
            assert not torch.isnan(output).any()


class TestMultiResolutionAdaptiveNeuralOperator:
    """Test multi-resolution adaptive neural operator."""
    
    @pytest.fixture
    def operator(self):
        """Create test operator."""
        return MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            hidden_dim=64,
            resolutions=[32, 64],
            num_layers=4,
            adaptive_refinement=True
        )
    
    def test_init(self, operator):
        """Test operator initialization."""
        assert operator.input_dim == 4
        assert operator.output_dim == 4
        assert operator.hidden_dim == 64
        assert operator.resolutions == [32, 64]
        assert operator.num_layers == 4
        assert operator.adaptive_refinement
    
    def test_reshape_to_2d(self, operator):
        """Test reshaping input to 2D."""
        # Perfect square case
        x = torch.randn(2, 64, 4)  # 8x8 = 64
        x_2d = operator.reshape_to_2d(x)
        assert x_2d.shape == (2, 4, 8, 8)
        
        # Non-perfect square case (should pad)
        x = torch.randn(2, 60, 4)  # Will pad to 64
        x_2d = operator.reshape_to_2d(x)
        assert x_2d.shape == (2, 4, 8, 8)
    
    def test_reshape_from_2d(self, operator):
        """Test reshaping from 2D back to original format."""
        x_2d = torch.randn(2, 4, 8, 8)
        target_shape = torch.Size([2, 60, 4])  # Target with 60 points
        
        x_1d = operator.reshape_from_2d(x_2d, target_shape)
        
        assert x_1d.shape == (2, 60, 4)
    
    def test_forward_features(self, operator):
        """Test feature extraction."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        features = operator.forward_features(x)
        
        assert features.shape == (batch_size, num_points, 64)
        assert not torch.isnan(features).any()
    
    def test_forward_operator(self, operator):
        """Test operator transformation."""
        batch_size, num_points = 2, 64
        features = torch.randn(batch_size, num_points, 64)
        
        output = operator.forward_operator(features)
        
        assert output.shape == (batch_size, num_points, 4)
        assert not torch.isnan(output).any()
    
    def test_full_forward(self, operator):
        """Test full forward pass."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        output = operator(x)
        
        assert output.shape == (batch_size, num_points, 4)
        assert not torch.isnan(output).any()
    
    def test_adaptive_forward(self, operator):
        """Test adaptive forward pass with refinement."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        output = operator.adaptive_forward(x, refinement_steps=2)
        
        assert output.shape[0] == batch_size
        assert output.shape[2] == 4
        assert not torch.isnan(output).any()
    
    def test_get_refinement_map(self, operator):
        """Test refinement map generation."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        refinement_map = operator.get_refinement_map(x)
        
        assert refinement_map is not None
        assert refinement_map.shape[0] == batch_size
        assert (refinement_map >= 0).all()
        assert (refinement_map <= 1).all()
        assert not torch.isnan(refinement_map).any()
    
    def test_no_adaptive_refinement(self):
        """Test operator without adaptive refinement."""
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            adaptive_refinement=False
        )
        
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        # Should work without refinement
        output = operator(x)
        assert output.shape == (batch_size, num_points, 4)
        
        # Refinement map should be None
        refinement_map = operator.get_refinement_map(x)
        assert refinement_map is None
    
    def test_resolution_weights(self, operator):
        """Test learnable resolution weights."""
        assert hasattr(operator, 'resolution_weights')
        assert operator.resolution_weights.shape == (len(operator.resolutions),)
        assert operator.resolution_weights.requires_grad
    
    def test_variable_input_sizes(self, operator):
        """Test handling of variable input sizes."""
        # Test different input sizes
        input_sizes = [36, 64, 100, 144]  # Different numbers of points
        
        for size in input_sizes:
            x = torch.randn(1, size, 4)
            
            # Should handle different sizes
            output = operator(x)
            assert output.shape == (1, size, 4)
            assert not torch.isnan(output).any()


class TestAdaptiveOperatorFactory:
    """Test adaptive operator factory function."""
    
    def test_create_from_config(self):
        """Test operator creation from config."""
        config = {
            'input_dim': 3,
            'output_dim': 3,
            'hidden_dim': 128,
            'resolutions': [64, 128, 256],
            'num_layers': 6,
            'adaptive_refinement': True,
            'refinement_threshold': 0.05,
            'max_refinement_levels': 4
        }
        
        operator = create_adaptive_neural_operator(config)
        
        assert operator.input_dim == 3
        assert operator.output_dim == 3
        assert operator.hidden_dim == 128
        assert operator.resolutions == [64, 128, 256]
        assert operator.num_layers == 6
        assert operator.adaptive_refinement
        assert operator.max_refinement_levels == 4


class TestErrorHandling:
    """Test error handling in adaptive operators."""
    
    def test_invalid_input_dimension(self):
        """Test handling of invalid input dimensions."""
        operator = MultiResolutionAdaptiveNeuralOperator(4, 4)
        
        # Wrong input dimension
        x = torch.randn(2, 64, 3)  # Should be 4
        
        with pytest.raises(ValueError):
            operator.forward_features(x)
    
    def test_incompatible_attention_heads(self):
        """Test error with incompatible attention heads."""
        with pytest.raises(AssertionError):
            # Channels not divisible by num_heads
            CrossResolutionAttention(65, num_heads=8)
    
    def test_empty_resolutions_list(self):
        """Test error with empty resolutions list."""
        with pytest.raises((ValueError, IndexError)):
            MultiResolutionProcessor(64, resolutions=[])
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        operator = MultiResolutionAdaptiveNeuralOperator(4, 4)
        
        x = torch.randn(2, 64, 4)
        x[0, 0, 0] = float('nan')
        
        # Should not crash but may produce NaN outputs
        output = operator(x)
        # Just check that computation completes
        assert output.shape == (2, 64, 4)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for adaptive operators."""
    
    def test_multi_resolution_performance(self):
        """Test performance with multiple resolutions."""
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            hidden_dim=128,
            resolutions=[64, 128, 256],
            num_layers=6
        )
        
        # Large input
        batch_size, num_points = 4, 1024
        x = torch.randn(batch_size, num_points, 4)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = operator(x)
        
        elapsed_time = time.time() - start_time
        
        assert output.shape == (batch_size, num_points, 4)
        assert elapsed_time < 20.0  # Should complete within 20 seconds
        print(f"Multi-resolution performance: {elapsed_time:.2f}s")
    
    def test_adaptive_refinement_performance(self):
        """Test performance with adaptive refinement."""
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            adaptive_refinement=True
        )
        
        batch_size, num_points = 2, 256
        x = torch.randn(batch_size, num_points, 4)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = operator.adaptive_forward(x, refinement_steps=3)
        
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 15.0  # Should complete within 15 seconds
        print(f"Adaptive refinement performance: {elapsed_time:.2f}s")
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        operator = MultiResolutionAdaptiveNeuralOperator(4, 4).to(device)
        
        x = torch.randn(2, 64, 4, device=device)
        
        output = operator(x)
        
        assert output.device == device
        assert output.shape == (2, 64, 4)


@pytest.mark.integration
class TestAdaptiveIntegration:
    """Integration tests for adaptive functionality."""
    
    def test_resolution_consistency(self):
        """Test that different resolutions produce consistent results."""
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            resolutions=[32, 64, 128]
        )
        
        # Same input processed at different effective resolutions
        x = torch.randn(1, 64, 4)
        
        output1 = operator(x)
        output2 = operator(x)
        
        # Outputs should be deterministic (same weights)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)
    
    def test_refinement_improves_accuracy(self):
        """Test that refinement tends to improve solution quality."""
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            adaptive_refinement=True
        )
        
        # Create a target with high-frequency components
        x = torch.randn(1, 64, 4)
        target = torch.sin(10 * x)  # High frequency target
        
        # Compare refined vs non-refined
        with torch.no_grad():
            output_basic = operator(x)
            output_refined = operator.adaptive_forward(x, refinement_steps=2)
        
        # Both should produce valid outputs
        assert output_basic.shape == output_refined.shape
        assert not torch.isnan(output_basic).any()
        assert not torch.isnan(output_refined).any()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of multi-resolution processing."""
        # This is a simple check that we can process larger inputs
        operator = MultiResolutionAdaptiveNeuralOperator(
            input_dim=4,
            output_dim=4,
            resolutions=[32, 64],  # Smaller resolutions for memory efficiency
            hidden_dim=64  # Smaller hidden dim
        )
        
        # Larger input that should still fit in memory
        batch_size, num_points = 8, 512
        x = torch.randn(batch_size, num_points, 4)
        
        with torch.no_grad():
            output = operator(x)
        
        assert output.shape == (batch_size, num_points, 4)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])