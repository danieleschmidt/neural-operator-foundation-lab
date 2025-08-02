"""Model tests for Neural Operator Foundation Lab."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


class MockNeuralOperator(nn.Module):
    """Mock neural operator for testing."""
    
    def __init__(self, input_dim: int = 3, output_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Simple linear layers for testing
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through mock operator."""
        # Flatten spatial dimensions for linear layers
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Simple encoding-decoding
        encoded = self.activation(self.encoder(x_flat))
        decoded = self.decoder(encoded)
        
        # Reshape back to original spatial dimensions
        output_shape = (batch_size, self.output_dim) + x.shape[2:]
        return decoded.view(output_shape)


class MockTransformerNeuralOperator(nn.Module):
    """Mock transformer neural operator for testing."""
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Simple projection layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Mock transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through mock transformer operator."""
        batch_size, channels, height, width = x.shape
        
        # Reshape to sequence format (batch, sequence, features)
        x_seq = x.view(batch_size, channels, -1).transpose(1, 2)  # (B, H*W, C)
        
        # Project to hidden dimension
        x_proj = self.input_proj(x_seq)
        
        # Apply transformer
        x_transformed = self.transformer(x_proj)
        
        # Project back to output dimension
        x_out = self.output_proj(x_transformed)
        
        # Reshape back to original format
        x_out = x_out.transpose(1, 2).view(batch_size, channels, height, width)
        
        return x_out


class TestModelInitialization:
    """Test model initialization and configuration."""
    
    def test_mock_neural_operator_init(self):
        """Test mock neural operator initialization."""
        model = MockNeuralOperator(input_dim=3, output_dim=3, hidden_dim=64)
        
        assert model.input_dim == 3
        assert model.output_dim == 3
        assert model.hidden_dim == 64
        assert isinstance(model.encoder, nn.Linear)
        assert isinstance(model.decoder, nn.Linear)
        assert isinstance(model.activation, nn.ReLU)
    
    def test_transformer_operator_init(self):
        """Test transformer neural operator initialization."""
        model = MockTransformerNeuralOperator(
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )
        
        assert model.input_dim == 4
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.num_heads == 4
        assert isinstance(model.input_proj, nn.Linear)
        assert isinstance(model.output_proj, nn.Linear)
        assert isinstance(model.transformer, nn.TransformerEncoder)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = MockNeuralOperator(input_dim=3, output_dim=3, hidden_dim=64)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable by default
    
    def test_model_device_placement(self, device):
        """Test model device placement."""
        model = MockNeuralOperator()
        model = model.to(device)
        
        # Check that model parameters are on the correct device
        for param in model.parameters():
            assert param.device == device


class TestModelForwardPass:
    """Test model forward pass functionality."""
    
    def test_mock_operator_forward(self, sample_pde_data, device):
        """Test mock neural operator forward pass."""
        model = MockNeuralOperator(input_dim=3, output_dim=3)
        model = model.to(device)
        
        input_tensor = sample_pde_data["input"].to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        assert output.shape == input_tensor.shape
        assert output.device == device
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_transformer_operator_forward(self, sample_pde_data, device):
        """Test transformer neural operator forward pass."""
        model = MockTransformerNeuralOperator(input_dim=3, hidden_dim=64)
        model = model.to(device)
        
        input_tensor = sample_pde_data["input"].to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape and properties
        assert output.shape == input_tensor.shape
        assert output.device == device
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_batch_size_handling(self, device):
        """Test handling of different batch sizes."""
        model = MockNeuralOperator()
        model = model.to(device)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        input_shape = (3, 32, 32)  # channels, height, width
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, *input_shape).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            expected_shape = (batch_size,) + input_shape
            assert output.shape == expected_shape
    
    def test_gradient_flow(self, sample_pde_data, device):
        """Test gradient flow through model."""
        model = MockNeuralOperator()
        model = model.to(device)
        model.train()
        
        input_tensor = sample_pde_data["input"].to(device)
        target_tensor = sample_pde_data["target"].to(device)
        
        # Forward pass
        output = model(input_tensor)
        
        # Compute loss
        loss = nn.MSELoss()(output, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for parameter {name}"


class TestModelSaving:
    """Test model saving and loading functionality."""
    
    def test_model_state_dict_save_load(self, temp_model_path, device):
        """Test saving and loading model state dict."""
        model = MockNeuralOperator(input_dim=3, output_dim=3, hidden_dim=64)
        model = model.to(device)
        
        # Save model state dict
        torch.save(model.state_dict(), temp_model_path)
        
        # Create new model and load state dict
        new_model = MockNeuralOperator(input_dim=3, output_dim=3, hidden_dim=64)
        new_model = new_model.to(device)
        new_model.load_state_dict(torch.load(temp_model_path, map_location=device))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    def test_full_model_save_load(self, temp_model_path, device):
        """Test saving and loading full model."""
        model = MockNeuralOperator()
        model = model.to(device)
        
        # Save full model
        torch.save(model, temp_model_path)
        
        # Load model
        loaded_model = torch.load(temp_model_path, map_location=device)
        
        # Test that loaded model works
        test_input = torch.randn(1, 3, 32, 32).to(device)
        
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        
        assert torch.allclose(original_output, loaded_output)
    
    @pytest.mark.security
    def test_secure_model_loading(self, temp_model_path, secure_config):
        """Test secure model loading with validation."""
        from neural_operator_lab.config import get_security_config
        
        model = MockNeuralOperator()
        torch.save(model.state_dict(), temp_model_path)
        
        config = get_security_config()
        
        # Verify security settings
        assert config.validate_models is True
        assert temp_model_path.suffix in config.allowed_file_types
        
        # In actual implementation, would validate file before loading
        assert temp_model_path.exists()
        assert temp_model_path.stat().st_size > 0
        
        # Load with security validation (mocked)
        loaded_state = torch.load(temp_model_path, map_location="cpu")
        assert isinstance(loaded_state, dict)


class TestModelTraining:
    """Test model training functionality."""
    
    def test_training_mode_switch(self):
        """Test switching between training and evaluation modes."""
        model = MockNeuralOperator()
        
        # Test initial mode
        assert model.training is True
        
        # Switch to eval mode
        model.eval()
        assert model.training is False
        
        # Switch back to train mode
        model.train()
        assert model.training is True
    
    def test_simple_training_step(self, sample_pde_data, device):
        """Test a simple training step."""
        model = MockNeuralOperator()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        input_tensor = sample_pde_data["input"].to(device)
        target_tensor = sample_pde_data["target"].to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0.0
    
    def test_optimizer_state(self):
        """Test optimizer state management."""
        model = MockNeuralOperator()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Initial state
        assert len(optimizer.state) == 0
        
        # After one step, optimizer should have state
        dummy_loss = sum(p.sum() for p in model.parameters())
        dummy_loss.backward()
        optimizer.step()
        
        assert len(optimizer.state) > 0


class TestModelValidation:
    """Test model validation and testing functionality."""
    
    def test_evaluation_mode(self, sample_pde_data, device):
        """Test model in evaluation mode."""
        model = MockNeuralOperator()
        model = model.to(device)
        model.eval()
        
        input_tensor = sample_pde_data["input"].to(device)
        
        # Multiple forward passes should give same result in eval mode
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        assert torch.allclose(output1, output2)
    
    def test_model_consistency(self, device):
        """Test model output consistency."""
        model = MockNeuralOperator()
        model = model.to(device)
        model.eval()
        
        # Same input should always produce same output
        test_input = torch.randn(2, 3, 32, 32).to(device)
        
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_model_determinism(self, device):
        """Test model determinism with fixed seed."""
        torch.manual_seed(42)
        model1 = MockNeuralOperator()
        
        torch.manual_seed(42)
        model2 = MockNeuralOperator()
        
        # Models initialized with same seed should have same parameters
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


@pytest.mark.slow
class TestModelPerformance:
    """Test model performance and benchmarking."""
    
    def test_inference_speed(self, performance_data, device):
        """Test model inference speed."""
        import time
        
        model = MockNeuralOperator()
        model = model.to(device)
        model.eval()
        
        input_tensor = performance_data["input"].to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        
        # Measure inference time
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Check that inference completed and was reasonably fast
        assert output is not None
        assert inference_time < 5.0  # Should complete within 5 seconds
    
    def test_memory_usage(self, performance_data, device):
        """Test model memory usage."""
        model = MockNeuralOperator()
        model = model.to(device)
        
        input_tensor = performance_data["input"].to(device)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            
            # Memory usage should be reasonable
            assert memory_used > 0
            assert memory_used < 1024**3  # Less than 1GB for this test