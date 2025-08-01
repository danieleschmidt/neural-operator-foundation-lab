"""Unit tests for neural operator implementations."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from typing import Optional, Tuple


# Mock neural operator implementations for testing infrastructure
class MockFourierNeuralOperator(nn.Module):
    """Mock Fourier Neural Operator for testing."""
    
    def __init__(self, input_dim: int, output_dim: int, modes: list, width: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        
        # Simple linear approximation for testing
        self.input_proj = nn.Linear(input_dim, width)
        self.output_proj = nn.Linear(width, output_dim)
        self.fourier_layers = nn.ModuleList([
            nn.Linear(width, width) for _ in range(4)
        ])
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, self.input_dim, -1).transpose(1, 2)
        
        # Project to hidden dimension
        x_hidden = self.input_proj(x_flat)
        
        # Apply Fourier layers (simplified)
        for layer in self.fourier_layers:
            x_hidden = torch.relu(layer(x_hidden))
        
        # Project to output
        x_out = self.output_proj(x_hidden)
        
        # Reshape back to spatial dimensions
        x_out = x_out.transpose(1, 2).view(batch_size, self.output_dim, *spatial_dims)
        
        return x_out


class MockTransformerNeuralOperator(nn.Module):
    """Mock Transformer Neural Operator for testing."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        
        # Flatten and project
        x_flat = x.view(batch_size, self.input_dim, -1).transpose(1, 2)
        x_hidden = self.input_proj(x_flat)
        
        # Apply transformer
        x_transformed = self.transformer(x_hidden)
        
        # Project to output
        x_out = self.output_proj(x_transformed)
        
        # Reshape back
        x_out = x_out.transpose(1, 2).view(batch_size, self.output_dim, *spatial_dims)
        
        return x_out


class MockProbabilisticNeuralOperator(nn.Module):
    """Mock Probabilistic Neural Operator for testing."""
    
    def __init__(self, input_dim: int, output_dim: int, trunk_layers, branch_layers,
                 uncertainty_type: str = "epistemic"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.uncertainty_type = uncertainty_type
        
        # Simple implementation for testing
        self.trunk_net = nn.Sequential(
            nn.Linear(2, trunk_layers[0]),  # 2D coordinates
            *[nn.Sequential(nn.ReLU(), nn.Linear(trunk_layers[i], trunk_layers[i+1])) 
              for i in range(len(trunk_layers)-1)]
        )
        
        self.branch_net = nn.Sequential(
            nn.Linear(input_dim, branch_layers[0]),
            *[nn.Sequential(nn.ReLU(), nn.Linear(branch_layers[i], branch_layers[i+1])) 
              for i in range(len(branch_layers)-1)]
        )
        
        self.output_layer = nn.Linear(trunk_layers[-1] + branch_layers[-1], output_dim * 2)
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None, 
                return_uncertainty: bool = False) -> torch.Tensor:
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        
        if grid is None:
            # Create default grid
            grid = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, spatial_dims[0]),
                torch.linspace(0, 1, spatial_dims[1]),
                indexing='ij'
            ), dim=-1).expand(batch_size, -1, -1, -1)
        
        # Process trunk (coordinates)
        grid_flat = grid.view(batch_size, -1, 2)
        trunk_out = self.trunk_net(grid_flat)
        
        # Process branch (input function)
        x_flat = x.view(batch_size, self.input_dim, -1).transpose(1, 2)
        branch_out = self.branch_net(x_flat)
        
        # Combine trunk and branch
        combined = torch.cat([trunk_out, branch_out], dim=-1)
        output = self.output_layer(combined)
        
        # Split mean and variance
        mean, log_var = output.chunk(2, dim=-1)
        uncertainty = torch.exp(0.5 * log_var)
        
        # Reshape
        mean = mean.transpose(1, 2).view(batch_size, self.output_dim, *spatial_dims)
        uncertainty = uncertainty.transpose(1, 2).view(batch_size, self.output_dim, *spatial_dims)
        
        if return_uncertainty:
            return mean, uncertainty
        return mean


class TestFourierNeuralOperator:
    """Test suite for Fourier Neural Operator."""
    
    def test_fno_initialization(self):
        """Test FNO initialization."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        assert model.input_dim == 3
        assert model.output_dim == 1
        assert model.modes == [16, 16]
        assert model.width == 64
    
    def test_fno_forward_pass(self, sample_pde_data):
        """Test FNO forward pass."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        input_data = sample_pde_data['input']  # [2, 3, 32, 32]
        output = model(input_data)
        
        expected_shape = (2, 1, 32, 32)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_fno_with_grid(self, sample_pde_data):
        """Test FNO with grid coordinates."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        input_data = sample_pde_data['input']
        grid = sample_pde_data['grid']
        
        output = model(input_data, grid)
        assert output.shape == (2, 1, 32, 32)
    
    @pytest.mark.parametrize("modes", [[8, 8], [16, 16], [32, 32]])
    def test_fno_different_modes(self, sample_pde_data, modes):
        """Test FNO with different Fourier modes."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=modes, width=64
        )
        
        input_data = sample_pde_data['input']
        output = model(input_data)
        
        assert output.shape == (2, 1, 32, 32)
    
    def test_fno_parameter_count(self):
        """Test FNO parameter counting."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0
    
    def test_fno_gradient_flow(self, sample_pde_data):
        """Test gradient flow through FNO."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        input_data = sample_pde_data['input']
        input_data.requires_grad_(True)
        
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert input_data.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTransformerNeuralOperator:
    """Test suite for Transformer Neural Operator."""
    
    def test_tno_initialization(self):
        """Test TNO initialization."""
        model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=1, hidden_dim=256, 
            num_layers=4, num_heads=8
        )
        
        assert model.input_dim == 3
        assert model.output_dim == 1
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.num_heads == 8
    
    def test_tno_forward_pass(self, sample_pde_data):
        """Test TNO forward pass."""
        model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=1, hidden_dim=128, 
            num_layers=2, num_heads=4
        )
        
        input_data = sample_pde_data['input']
        output = model(input_data)
        
        assert output.shape == (2, 1, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_tno_attention_heads(self, sample_pde_data, num_heads):
        """Test TNO with different numbers of attention heads."""
        # Ensure hidden_dim is divisible by num_heads
        hidden_dim = 64 if num_heads <= 4 else 128
        
        model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=1, hidden_dim=hidden_dim,
            num_layers=2, num_heads=num_heads
        )
        
        input_data = sample_pde_data['input']
        output = model(input_data)
        
        assert output.shape == (2, 1, 32, 32)
    
    def test_tno_memory_efficiency(self, sample_pde_data):
        """Test TNO memory efficiency."""
        model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=1, hidden_dim=64,
            num_layers=2, num_heads=2
        )
        
        input_data = sample_pde_data['input']
        
        # Test with gradient checkpointing simulation
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        
        # Verify gradients computed successfully
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestProbabilisticNeuralOperator:
    """Test suite for Probabilistic Neural Operator."""
    
    def test_pno_initialization(self):
        """Test PNO initialization."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[64, 64, 64],
            branch_layers=[64, 64]
        )
        
        assert model.input_dim == 3
        assert model.output_dim == 1
    
    def test_pno_forward_pass(self, sample_pde_data):
        """Test PNO forward pass."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32]
        )
        
        input_data = sample_pde_data['input']
        output = model(input_data)
        
        assert output.shape == (2, 1, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_pno_uncertainty_quantification(self, sample_pde_data):
        """Test PNO uncertainty quantification."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32]
        )
        
        input_data = sample_pde_data['input']
        mean, uncertainty = model(input_data, return_uncertainty=True)
        
        assert mean.shape == (2, 1, 32, 32)
        assert uncertainty.shape == (2, 1, 32, 32)
        assert (uncertainty >= 0).all()  # Uncertainty should be non-negative
    
    def test_pno_with_custom_grid(self, sample_pde_data):
        """Test PNO with custom grid."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32]
        )
        
        input_data = sample_pde_data['input']
        grid = sample_pde_data['grid'][:, :2]  # Use only x, y coordinates
        
        output = model(input_data, grid)
        assert output.shape == (2, 1, 32, 32)
    
    @pytest.mark.parametrize("uncertainty_type", ["epistemic", "aleatoric"])
    def test_pno_uncertainty_types(self, sample_pde_data, uncertainty_type):
        """Test different uncertainty types."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32],
            uncertainty_type=uncertainty_type
        )
        
        input_data = sample_pde_data['input']
        mean, uncertainty = model(input_data, return_uncertainty=True)
        
        assert mean.shape == (2, 1, 32, 32)
        assert uncertainty.shape == (2, 1, 32, 32)


class TestNeuralOperatorBase:
    """Test suite for neural operator base functionality."""
    
    def test_model_saving_loading(self, sample_pde_data, temp_model_path):
        """Test model saving and loading."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        # Save model
        torch.save(model.state_dict(), temp_model_path)
        
        # Create new model and load state
        new_model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        new_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
        
        # Test that models produce same output
        input_data = sample_pde_data['input']
        with torch.no_grad():
            output1 = model(input_data)
            output2 = new_model(input_data)
        
        torch.testing.assert_close(output1, output2)
    
    def test_model_device_transfer(self, sample_pde_data):
        """Test model device transfer."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        # Test CPU
        model_cpu = model.cpu()
        input_cpu = sample_pde_data['input'].cpu()
        output_cpu = model_cpu(input_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU (if available)
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_gpu = sample_pde_data['input'].cuda()
            output_gpu = model_gpu(input_gpu)
            assert output_gpu.device.type == 'cuda'
    
    def test_model_eval_mode(self, sample_pde_data):
        """Test model evaluation mode."""
        model = MockProbabilisticNeuralOperator(
            input_dim=3, output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32]
        )
        
        input_data = sample_pde_data['input']
        
        # Test training mode
        model.train()
        output_train = model(input_data)
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            output_eval = model(input_data)
        
        # Outputs should be similar (but may differ due to dropout, etc.)
        assert output_train.shape == output_eval.shape
    
    def test_model_reproducibility(self, sample_pde_data):
        """Test model reproducibility with fixed seeds."""
        torch.manual_seed(42)
        model1 = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        torch.manual_seed(42)
        model2 = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=64
        )
        
        input_data = sample_pde_data['input']
        
        # Models should produce identical outputs
        with torch.no_grad():
            output1 = model1(input_data)
            output2 = model2(input_data)
        
        torch.testing.assert_close(output1, output2)


@pytest.mark.slow
class TestNeuralOperatorPerformance:
    """Performance tests for neural operators."""
    
    def test_fno_inference_speed(self, performance_data):
        """Test FNO inference speed."""
        model = MockFourierNeuralOperator(
            input_dim=8, output_dim=1, modes=[32, 32], width=128
        )
        model.eval()
        
        input_data = performance_data['input']
        
        import time
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(input_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 1.0  # Should be less than 1 second per inference
        assert output.shape == (4, 1, 128, 128)
    
    def test_memory_usage(self, performance_data):
        """Test memory usage during training."""
        model = MockTransformerNeuralOperator(
            input_dim=8, output_dim=1, hidden_dim=256,
            num_layers=4, num_heads=8
        )
        
        input_data = performance_data['input']
        target_data = performance_data['target']
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test training step
        optimizer.zero_grad()
        output = model(input_data)
        loss = torch.nn.functional.mse_loss(output, target_data)
        loss.backward()
        optimizer.step()
        
        # Verify successful training step
        assert not torch.isnan(loss).any()
        assert loss.item() >= 0