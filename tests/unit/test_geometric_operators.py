"""Unit tests for geometric neural operators."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from neural_operator_lab.models.geometric import (
    GeometricNeuralOperator,
    GeometricConvolution,
    ManifoldEmbedding,
    PositionalEncoding,
    AdaptiveMeshRefinement,
    create_geometric_neural_operator
)


class TestGeometricConvolution:
    """Test geometric convolution layer."""
    
    def test_init(self):
        """Test geometric convolution initialization."""
        conv = GeometricConvolution(16, 32, max_neighbors=8)
        assert conv.in_channels == 16
        assert conv.out_channels == 32
        assert conv.max_neighbors == 8
    
    def test_forward(self):
        """Test geometric convolution forward pass."""
        conv = GeometricConvolution(16, 32)
        
        # Mock data
        x = torch.randn(100, 16)  # 100 nodes, 16 features
        edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        edge_attr = torch.randn(200, 3)  # Edge attributes
        
        output = conv(x, edge_index, edge_attr)
        
        assert output.shape == (100, 32)
        assert not torch.isnan(output).any()


class TestManifoldEmbedding:
    """Test manifold embedding layer."""
    
    def test_init(self):
        """Test manifold embedding initialization."""
        embedding = ManifoldEmbedding(3, 64)
        assert embedding.manifold_dim == 3
        assert embedding.embed_dim == 64
    
    def test_forward(self):
        """Test manifold embedding forward pass."""
        embedding = ManifoldEmbedding(3, 64)
        coords = torch.randn(100, 3)
        
        output = embedding(coords)
        
        assert output.shape == (100, 64)
        assert not torch.isnan(output).any()


class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_init(self):
        """Test positional encoding initialization."""
        encoding = PositionalEncoding(3, 64)
        assert encoding.coord_dim == 3
        assert encoding.embed_dim == 64
    
    def test_forward(self):
        """Test positional encoding forward pass."""
        encoding = PositionalEncoding(3, 64)
        coords = torch.randn(100, 3)
        
        output = encoding(coords)
        
        assert output.shape == (100, 64)
        assert not torch.isnan(output).any()


class TestAdaptiveMeshRefinement:
    """Test adaptive mesh refinement."""
    
    def test_init(self):
        """Test adaptive mesh refinement initialization."""
        refiner = AdaptiveMeshRefinement(threshold=0.1)
        assert refiner.threshold == 0.1
    
    def test_compute_refinement_indicator(self):
        """Test refinement indicator computation."""
        refiner = AdaptiveMeshRefinement()
        
        solution = torch.randn(100, 4)
        coords = torch.rand(100, 3)
        
        indicator = refiner.compute_refinement_indicator(solution, coords)
        
        assert indicator.shape == (100,)
        assert (indicator >= 0).all()


class TestGeometricNeuralOperator:
    """Test geometric neural operator."""
    
    @pytest.fixture
    def operator(self):
        """Create test operator."""
        return GeometricNeuralOperator(
            input_dim=4,
            output_dim=4,
            hidden_dim=64,
            num_layers=3,
            manifold_dim=3
        )
    
    def test_init(self, operator):
        """Test operator initialization."""
        assert operator.input_dim == 4
        assert operator.output_dim == 4
        assert operator.hidden_dim == 64
        assert operator.num_layers == 3
        assert operator.manifold_dim == 3
    
    def test_forward_features(self, operator):
        """Test feature extraction."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        grid = torch.rand(batch_size, num_points, 3)
        
        features = operator.forward_features(x, grid)
        
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
        grid = torch.rand(batch_size, num_points, 3)
        
        output = operator(x, grid)
        
        assert output.shape == (batch_size, num_points, 4)
        assert not torch.isnan(output).any()
    
    def test_adaptive_forward(self, operator):
        """Test adaptive forward pass."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        grid = torch.rand(batch_size, num_points, 3)
        
        output = operator.adaptive_forward(x, grid, num_refinements=2)
        
        assert output.shape[0] == batch_size
        assert output.shape[2] == 4
        assert not torch.isnan(output).any()
    
    def test_compute_geometric_loss(self, operator):
        """Test geometric loss computation."""
        batch_size, num_points = 2, 64
        pred = torch.randn(batch_size, num_points, 4)
        target = torch.randn(batch_size, num_points, 4)
        coords = torch.rand(batch_size, num_points, 3)
        
        loss = operator.compute_geometric_loss(pred, target, coords)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_build_edge_index(self, operator):
        """Test edge index construction."""
        coords = torch.rand(100, 3)
        
        edge_index = operator.build_edge_index(coords)
        
        assert edge_index.shape[0] == 2
        assert edge_index.max() < 100
        assert edge_index.min() >= 0
    
    def test_compute_edge_features(self, operator):
        """Test edge feature computation."""
        coords = torch.rand(100, 3)
        edge_index = torch.randint(0, 100, (2, 200))
        
        edge_features = operator.compute_edge_features(coords, edge_index)
        
        assert edge_features.shape == (200, 3)  # distance + direction + curvature
        assert not torch.isnan(edge_features).any()
    
    def test_parameter_count(self, operator):
        """Test parameter counting."""
        param_count = operator.count_parameters()
        assert param_count > 0
    
    def test_model_summary(self, operator):
        """Test model summary."""
        summary = operator.get_model_summary()
        
        assert 'parameters' in summary
        assert 'input_dim' in summary
        assert 'output_dim' in summary
        assert 'model_type' in summary
        assert summary['input_dim'] == 4
        assert summary['output_dim'] == 4


class TestGeometricOperatorFactory:
    """Test geometric operator factory function."""
    
    def test_create_from_config(self):
        """Test operator creation from config."""
        config = {
            'input_dim': 3,
            'output_dim': 3,
            'hidden_dim': 128,
            'num_layers': 4,
            'manifold_dim': 2,
            'max_neighbors': 8,
            'use_adaptive_refinement': False
        }
        
        operator = create_geometric_neural_operator(config)
        
        assert operator.input_dim == 3
        assert operator.output_dim == 3
        assert operator.hidden_dim == 128
        assert operator.num_layers == 4
        assert operator.manifold_dim == 2
        assert operator.max_neighbors == 8
        assert not operator.use_adaptive_refinement


class TestErrorHandling:
    """Test error handling in geometric operators."""
    
    def test_invalid_input_dimension(self):
        """Test handling of invalid input dimensions."""
        operator = GeometricNeuralOperator(4, 4)
        
        # Wrong input dimension
        x = torch.randn(2, 64, 3)  # Should be 4
        
        with pytest.raises(ValueError):
            operator.forward_features(x)
    
    def test_mismatched_grid_shape(self):
        """Test handling of mismatched grid shapes."""
        operator = GeometricNeuralOperator(4, 4, manifold_dim=3)
        
        x = torch.randn(2, 64, 4)
        grid = torch.rand(2, 32, 3)  # Wrong number of points
        
        with pytest.raises(ValueError):
            operator.forward_features(x, grid)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        operator = GeometricNeuralOperator(4, 4)
        
        x = torch.randn(2, 64, 4)
        x[0, 0, 0] = float('nan')
        
        # Should not crash but may produce NaN outputs
        output = operator(x)
        # Just check that computation completes
        assert output.shape == (2, 64, 4)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for geometric operators."""
    
    def test_large_input_performance(self):
        """Test performance with large inputs."""
        operator = GeometricNeuralOperator(4, 4, hidden_dim=256, num_layers=6)
        
        # Large input
        batch_size, num_points = 4, 1024
        x = torch.randn(batch_size, num_points, 4)
        grid = torch.rand(batch_size, num_points, 3)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = operator(x, grid)
        
        elapsed_time = time.time() - start_time
        
        assert output.shape == (batch_size, num_points, 4)
        assert elapsed_time < 10.0  # Should complete within 10 seconds
        print(f"Large input performance: {elapsed_time:.2f}s")
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        operator = GeometricNeuralOperator(4, 4).to(device)
        
        x = torch.randn(2, 64, 4, device=device)
        grid = torch.rand(2, 64, 3, device=device)
        
        output = operator(x, grid)
        
        assert output.device == device
        assert output.shape == (2, 64, 4)


if __name__ == "__main__":
    pytest.main([__file__])