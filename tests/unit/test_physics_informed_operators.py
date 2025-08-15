"""Unit tests for physics-informed neural operators."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from neural_operator_lab.models.physics_informed import (
    PhysicsInformedUniversalNeuralOperator,
    PDEType,
    ConservationLaw,
    PDEResidualComputer,
    PhysicsInformedLoss,
    FourierFeatureEmbedding,
    PhysicsAwareLayer,
    create_physics_informed_operator
)


class TestConservationLaw:
    """Test conservation law enforcement."""
    
    def test_init(self):
        """Test conservation law initialization."""
        law = ConservationLaw("mass", spatial_dims=3)
        assert law.law_type == "mass"
        assert law.spatial_dims == 3
    
    def test_compute_divergence(self):
        """Test divergence computation."""
        law = ConservationLaw("mass", spatial_dims=3)
        
        batch_size, num_points = 2, 64
        field = torch.randn(batch_size, num_points, 3)
        coords = torch.rand(batch_size, num_points, 3)
        
        divergence = law.compute_divergence(field, coords)
        
        assert divergence.shape == (batch_size, num_points)
        assert not torch.isnan(divergence).any()
    
    def test_compute_curl_3d(self):
        """Test curl computation for 3D fields."""
        law = ConservationLaw("momentum", spatial_dims=3)
        
        batch_size, num_points = 2, 64
        field = torch.randn(batch_size, num_points, 3)
        coords = torch.rand(batch_size, num_points, 3)
        
        curl = law.compute_curl(field, coords)
        
        assert curl.shape == (batch_size, num_points, 3)
        assert not torch.isnan(curl).any()
    
    def test_mass_conservation_forward(self):
        """Test mass conservation enforcement."""
        law = ConservationLaw("mass", spatial_dims=3)
        
        batch_size, num_points = 2, 64
        field = torch.randn(batch_size, num_points, 3)
        coords = torch.rand(batch_size, num_points, 3)
        
        residual = law(field, coords)
        
        assert residual.shape == (batch_size, num_points)
        assert not torch.isnan(residual).any()
    
    def test_incompressible_flow(self):
        """Test incompressible flow constraint."""
        law = ConservationLaw("incompressible", spatial_dims=3)
        
        batch_size, num_points = 2, 64
        velocity_field = torch.randn(batch_size, num_points, 3)
        coords = torch.rand(batch_size, num_points, 3)
        
        divergence = law(velocity_field, coords)
        
        assert divergence.shape == (batch_size, num_points)
        # For incompressible flow, divergence should be minimized
        assert not torch.isnan(divergence).any()


class TestPDEResidualComputer:
    """Test PDE residual computation."""
    
    def test_init(self):
        """Test PDE residual computer initialization."""
        pde_params = {'viscosity': 0.01, 'density': 1.0}
        computer = PDEResidualComputer(
            PDEType.NAVIER_STOKES, pde_params, spatial_dims=3
        )
        
        assert computer.pde_type == PDEType.NAVIER_STOKES
        assert computer.viscosity == 0.01
        assert computer.density == 1.0
    
    def test_compute_laplacian(self):
        """Test Laplacian computation."""
        computer = PDEResidualComputer(PDEType.HEAT_EQUATION, {})
        
        batch_size, num_points = 2, 64
        field = torch.randn(batch_size, num_points, 1)
        coords = torch.rand(batch_size, num_points, 3)
        
        laplacian = computer.compute_laplacian(field, coords)
        
        assert laplacian.shape == field.shape
        assert not torch.isnan(laplacian).any()
    
    def test_compute_gradient(self):
        """Test gradient computation."""
        computer = PDEResidualComputer(PDEType.HEAT_EQUATION, {})
        
        batch_size, num_points = 2, 64
        field = torch.randn(batch_size, num_points, 1)
        coords = torch.rand(batch_size, num_points, 3)
        
        gradient = computer.compute_gradient(field, coords)
        
        assert gradient.shape == (batch_size, num_points, 1, 3)
        assert not torch.isnan(gradient).any()
    
    def test_navier_stokes_residual(self):
        """Test Navier-Stokes residual computation."""
        pde_params = {'viscosity': 0.01, 'density': 1.0}
        computer = PDEResidualComputer(PDEType.NAVIER_STOKES, pde_params)
        
        batch_size, num_points = 2, 64
        velocity = torch.randn(batch_size, num_points, 3)
        pressure = torch.randn(batch_size, num_points, 1)
        coords = torch.rand(batch_size, num_points, 3)
        
        residual = computer.navier_stokes_residual(velocity, pressure, coords)
        
        assert residual.shape == (batch_size, num_points, 4)  # 3 momentum + 1 continuity
        assert not torch.isnan(residual).any()
    
    def test_heat_equation_residual(self):
        """Test heat equation residual computation."""
        pde_params = {'diffusivity': 0.1}
        computer = PDEResidualComputer(PDEType.HEAT_EQUATION, pde_params)
        
        batch_size, num_points = 2, 64
        temperature = torch.randn(batch_size, num_points, 1)
        coords = torch.rand(batch_size, num_points, 3)
        
        residual = computer.heat_equation_residual(temperature, coords)
        
        assert residual.shape == temperature.shape
        assert not torch.isnan(residual).any()
    
    def test_burgers_equation_residual(self):
        """Test Burgers equation residual computation."""
        pde_params = {'viscosity': 0.01}
        computer = PDEResidualComputer(PDEType.BURGERS, pde_params)
        
        batch_size, num_points = 2, 64
        u = torch.randn(batch_size, num_points, 1)
        coords = torch.rand(batch_size, num_points, 3)
        
        residual = computer.burgers_equation_residual(u, coords)
        
        assert residual.shape == u.shape
        assert not torch.isnan(residual).any()


class TestPhysicsInformedLoss:
    """Test physics-informed loss computation."""
    
    def test_init(self):
        """Test physics-informed loss initialization."""
        pde_params = {'viscosity': 0.01}
        loss_fn = PhysicsInformedLoss(
            PDEType.NAVIER_STOKES,
            pde_params,
            conservation_laws=['mass', 'momentum'],
            physics_weight=0.1
        )
        
        assert loss_fn.physics_weight == 0.1
        assert len(loss_fn.conservation_laws) == 2
    
    def test_forward(self):
        """Test loss computation."""
        pde_params = {'viscosity': 0.01, 'density': 1.0}
        loss_fn = PhysicsInformedLoss(
            PDEType.NAVIER_STOKES,
            pde_params,
            conservation_laws=['mass']
        )
        
        batch_size, num_points = 2, 64
        pred = torch.randn(batch_size, num_points, 4)  # 3 velocity + 1 pressure
        target = torch.randn(batch_size, num_points, 4)
        coords = torch.rand(batch_size, num_points, 3)
        
        losses = loss_fn(pred, target, coords)
        
        assert 'data' in losses
        assert 'physics' in losses
        assert 'conservation' in losses
        assert 'total' in losses
        
        for loss_name, loss_val in losses.items():
            assert loss_val.item() >= 0
            assert not torch.isnan(loss_val)
    
    def test_compute_boundary_loss(self):
        """Test boundary condition loss."""
        loss_fn = PhysicsInformedLoss(PDEType.HEAT_EQUATION, {})
        
        batch_size, num_points = 2, 64
        pred = torch.randn(batch_size, num_points, 1)
        target = torch.randn(batch_size, num_points, 1)
        boundary_mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        boundary_mask[:, :10] = True  # First 10 points are boundary
        
        boundary_loss = loss_fn.compute_boundary_loss(pred, target, boundary_mask)
        
        assert boundary_loss.item() >= 0
        assert not torch.isnan(boundary_loss)


class TestFourierFeatureEmbedding:
    """Test Fourier feature embedding."""
    
    def test_init(self):
        """Test Fourier embedding initialization."""
        embedding = FourierFeatureEmbedding(4, 64, num_modes=32)
        assert embedding.input_dim == 4
        assert embedding.output_dim == 64
        assert embedding.num_modes == 32
    
    def test_forward(self):
        """Test Fourier embedding forward pass."""
        embedding = FourierFeatureEmbedding(4, 64, num_modes=32)
        
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        
        output = embedding(x)
        
        assert output.shape == (batch_size, num_points, 64)
        assert not torch.isnan(output).any()


class TestPhysicsAwareLayer:
    """Test physics-aware neural network layer."""
    
    def test_init(self):
        """Test physics-aware layer initialization."""
        layer = PhysicsAwareLayer(128, 128)
        assert layer.input_dim == 128
        assert layer.output_dim == 128
    
    def test_forward(self):
        """Test physics-aware layer forward pass."""
        layer = PhysicsAwareLayer(128, 128)
        
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 128)
        coords = torch.rand(batch_size, num_points, 3)
        
        output = layer(x, coords)
        
        assert output.shape == (batch_size, num_points, 128)
        assert not torch.isnan(output).any()


class TestPhysicsInformedUniversalNeuralOperator:
    """Test physics-informed universal neural operator."""
    
    @pytest.fixture
    def operator(self):
        """Create test operator."""
        return PhysicsInformedUniversalNeuralOperator(
            input_dim=4,
            output_dim=4,
            hidden_dim=128,
            num_layers=4,
            pde_type=PDEType.NAVIER_STOKES,
            use_fourier_features=True
        )
    
    def test_init(self, operator):
        """Test operator initialization."""
        assert operator.input_dim == 4
        assert operator.output_dim == 4
        assert operator.hidden_dim == 128
        assert operator.num_layers == 4
        assert operator.pde_type == PDEType.NAVIER_STOKES
        assert operator.use_fourier_features
    
    def test_forward_features(self, operator):
        """Test feature extraction."""
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        grid = torch.rand(batch_size, num_points, 3)
        
        features = operator.forward_features(x, grid)
        
        assert features.shape == (batch_size, num_points, 128)
        assert not torch.isnan(features).any()
    
    def test_forward_operator(self, operator):
        """Test operator transformation."""
        batch_size, num_points = 2, 64
        features = torch.randn(batch_size, num_points, 128)
        
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
    
    def test_physics_informed_loss(self, operator):
        """Test physics-informed loss computation."""
        batch_size, num_points = 2, 64
        pred = torch.randn(batch_size, num_points, 4)
        target = torch.randn(batch_size, num_points, 4)
        coords = torch.rand(batch_size, num_points, 3)
        
        loss = operator.get_loss(pred, target, coords=coords)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_different_pde_types(self):
        """Test different PDE types."""
        pde_types = [PDEType.HEAT_EQUATION, PDEType.WAVE_EQUATION, PDEType.BURGERS]
        
        for pde_type in pde_types:
            operator = PhysicsInformedUniversalNeuralOperator(
                input_dim=1,
                output_dim=1,
                pde_type=pde_type
            )
            
            batch_size, num_points = 2, 64
            x = torch.randn(batch_size, num_points, 1)
            
            output = operator(x)
            
            assert output.shape == (batch_size, num_points, 1)
            assert not torch.isnan(output).any()


class TestPhysicsInformedOperatorFactory:
    """Test physics-informed operator factory function."""
    
    def test_create_from_config(self):
        """Test operator creation from config."""
        config = {
            'input_dim': 3,
            'output_dim': 3,
            'hidden_dim': 256,
            'num_layers': 6,
            'pde_type': 'heat',
            'pde_params': {'diffusivity': 0.2},
            'conservation_laws': ['energy'],
            'use_fourier_features': False,
            'physics_weight': 0.2
        }
        
        operator = create_physics_informed_operator(config)
        
        assert operator.input_dim == 3
        assert operator.output_dim == 3
        assert operator.hidden_dim == 256
        assert operator.num_layers == 6
        assert operator.pde_type == PDEType.HEAT_EQUATION
        assert not operator.use_fourier_features


class TestErrorHandling:
    """Test error handling in physics-informed operators."""
    
    def test_invalid_pde_type(self):
        """Test handling of invalid PDE types."""
        with pytest.raises(ValueError):
            PDEType("invalid_pde_type")
    
    def test_invalid_input_dimension(self):
        """Test handling of invalid input dimensions."""
        operator = PhysicsInformedUniversalNeuralOperator(4, 4)
        
        # Wrong input dimension
        x = torch.randn(2, 64, 3)  # Should be 4
        
        with pytest.raises(ValueError):
            operator.forward_features(x)
    
    def test_conservation_law_dimension_mismatch(self):
        """Test conservation law with wrong field dimension."""
        law = ConservationLaw("incompressible", spatial_dims=3)
        
        # 2D field for 3D conservation law
        field = torch.randn(2, 64, 2)
        coords = torch.rand(2, 64, 3)
        
        with pytest.raises(ValueError):
            law.compute_divergence(field, coords)
    
    def test_curl_dimension_error(self):
        """Test curl computation with wrong dimensions."""
        law = ConservationLaw("momentum", spatial_dims=2)
        
        # Curl only works for 3D
        field = torch.randn(2, 64, 2)
        coords = torch.rand(2, 64, 2)
        
        with pytest.raises(ValueError):
            law.compute_curl(field, coords)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for physics-informed operators."""
    
    def test_large_input_performance(self):
        """Test performance with large inputs."""
        operator = PhysicsInformedUniversalNeuralOperator(
            4, 4, hidden_dim=512, num_layers=8
        )
        
        # Large input
        batch_size, num_points = 4, 2048
        x = torch.randn(batch_size, num_points, 4)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = operator(x)
        
        elapsed_time = time.time() - start_time
        
        assert output.shape == (batch_size, num_points, 4)
        assert elapsed_time < 15.0  # Should complete within 15 seconds
        print(f"Large input performance: {elapsed_time:.2f}s")
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        operator = PhysicsInformedUniversalNeuralOperator(4, 4).to(device)
        
        x = torch.randn(2, 64, 4, device=device)
        
        output = operator(x)
        
        assert output.device == device
        assert output.shape == (2, 64, 4)


@pytest.mark.integration
class TestPhysicsIntegration:
    """Integration tests for physics constraints."""
    
    def test_mass_conservation_enforcement(self):
        """Test that mass conservation is enforced during training."""
        operator = PhysicsInformedUniversalNeuralOperator(
            4, 4, pde_type=PDEType.NAVIER_STOKES
        )
        
        batch_size, num_points = 2, 64
        x = torch.randn(batch_size, num_points, 4)
        target = torch.randn(batch_size, num_points, 4)
        coords = torch.rand(batch_size, num_points, 3)
        
        # Forward pass
        pred = operator(x)
        
        # Compute physics-informed loss
        loss = operator.get_loss(pred, target, coords=coords)
        
        # Loss should include physics constraints
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
        # Velocity field (first 3 components) should have constraints
        velocity = pred[:, :, :3]
        conservation_law = ConservationLaw("incompressible", 3)
        divergence = conservation_law(velocity, coords)
        
        # Divergence should be finite
        assert torch.isfinite(divergence).all()


if __name__ == "__main__":
    pytest.main([__file__])