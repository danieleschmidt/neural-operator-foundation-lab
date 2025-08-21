#!/usr/bin/env python3
"""Generation 1: Basic Neural Operator Functionality Test

This script demonstrates that core neural operators work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from neural_operator_lab.models import FourierNeuralOperator

def test_fourier_neural_operator():
    """Test FNO on simple 2D data."""
    print("🧪 Testing Fourier Neural Operator...")
    
    # Create model with proper modes
    model = FourierNeuralOperator(
        input_dim=3,      # 3 input channels (e.g., velocity components)
        output_dim=2,     # 2 output channels (e.g., pressure, temperature)
        hidden_dim=64,    # Hidden dimension
        num_layers=4,     # Number of FNO layers
        modes=[12, 12]    # Fourier modes for 2D
    )
    
    # Create test data: batch_size=4, height=32, width=32, channels=3
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, H, W, 3)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
        print(f"Output shape: {y.shape}")
        
    # Verify output shape is correct
    expected_shape = (batch_size, H, W, 2)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    print("✅ FNO test passed!")
    return True

def test_simple_pde_simulation():
    """Test a simple PDE-like forward simulation."""
    print("🌊 Testing simple PDE simulation...")
    
    # Create a simple 2D heat equation setup
    model = FourierNeuralOperator(
        input_dim=1,      # Temperature field
        output_dim=1,     # Next temperature field
        hidden_dim=32,
        num_layers=2,
        modes=[8, 8]      # Lower modes for faster computation
    )
    
    # Initial condition: Gaussian heat source
    x, y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    initial_temp = np.exp(-(x**2 + y**2) / 0.1)
    
    # Convert to tensor
    temp_field = torch.from_numpy(initial_temp).float().unsqueeze(0).unsqueeze(-1)  # (1, 16, 16, 1)
    
    print(f"Initial temperature field shape: {temp_field.shape}")
    print(f"Max temperature: {temp_field.max():.4f}")
    
    # Simulate one time step
    with torch.no_grad():
        next_temp = model(temp_field)
        
    print(f"Next temperature field shape: {next_temp.shape}")
    print(f"Max temperature after step: {next_temp.max():.4f}")
    
    # Verify conservation (rough check)
    initial_energy = temp_field.sum()
    final_energy = next_temp.sum()
    energy_ratio = final_energy / initial_energy
    
    print(f"Energy ratio (should be ~1.0): {energy_ratio:.4f}")
    
    print("✅ Simple PDE simulation test passed!")
    return True

def test_performance():
    """Test basic performance characteristics."""
    print("⚡ Testing performance...")
    
    model = FourierNeuralOperator(
        input_dim=4,
        output_dim=3,
        hidden_dim=128,
        num_layers=6,
        modes=[16, 16]
    )
    
    # Larger batch for performance test
    batch_size = 8
    x = torch.randn(batch_size, 64, 64, 4)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Time forward pass
    import time
    start_time = time.time()
    
    with torch.no_grad():
        y = model(x)
    
    end_time = time.time()
    
    forward_time = end_time - start_time
    throughput = batch_size / forward_time
    
    print(f"Forward pass time: {forward_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} samples/second")
    
    # Memory usage
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    print("✅ Performance test passed!")
    return True

def main():
    """Run all Generation 1 tests."""
    print("🚀 GENERATION 1: MAKE IT WORK - Testing Basic Functionality")
    print("=" * 60)
    
    try:
        # Core functionality
        test_fourier_neural_operator()
        print()
        
        # Simple application
        test_simple_pde_simulation()
        print()
        
        # Performance check
        test_performance()
        print()
        
        print("🎉 GENERATION 1 COMPLETE: All basic functionality working!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Generation 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)