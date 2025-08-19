#!/usr/bin/env python3
"""Comprehensive test for Generation 1 implementation."""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all core imports work."""
    print("🔍 Testing core imports...")
    
    try:
        from neural_operator_lab import FourierNeuralOperator, TransformerNeuralOperator, ProbabilisticNeuralOperator
        print("✅ Core models imported successfully")
    except Exception as e:
        print(f"❌ Core model imports failed: {e}")
        return False
    
    try:
        from neural_operator_lab.data import create_synthetic_dataset
        print("✅ Data utilities imported successfully")
    except Exception as e:
        print(f"❌ Data imports failed: {e}")
        return False
    
    try:
        from neural_operator_lab.base import create_coordinate_grid
        print("✅ Base utilities imported successfully")
    except Exception as e:
        print(f"❌ Base imports failed: {e}")
        return False
    
    return True


def test_fno():
    """Test FNO functionality."""
    print("\n🧠 Testing Fourier Neural Operator...")
    
    try:
        from neural_operator_lab import FourierNeuralOperator
        from neural_operator_lab.data import create_synthetic_dataset
        
        # Create model
        model = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=[8, 8],
            width=32,
            num_layers=2
        )
        
        # Test forward pass
        x, y = create_synthetic_dataset('darcy', resolution=32, n_samples=4)
        pred = model(x)
        
        assert pred.shape == y.shape, f"Shape mismatch: {pred.shape} vs {y.shape}"
        assert not torch.isnan(pred).any(), "FNO output contains NaN"
        assert torch.isfinite(pred).all(), "FNO output contains Inf"
        
        # Test parameter count
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "Model has no parameters"
        
        print(f"✅ FNO test passed (parameters: {params:,})")
        return True
        
    except Exception as e:
        print(f"❌ FNO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tno():
    """Test Transformer Neural Operator."""
    print("\n🎯 Testing Transformer Neural Operator...")
    
    try:
        from neural_operator_lab import TransformerNeuralOperator
        from neural_operator_lab.data import create_synthetic_dataset
        from neural_operator_lab.base import create_coordinate_grid
        
        # Create model
        model = TransformerNeuralOperator(
            input_dim=1,
            output_dim=1,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_resolution=32
        )
        
        # Test forward pass
        x, y = create_synthetic_dataset('darcy', resolution=16, n_samples=2)
        
        # Create coordinate grid
        grid = create_coordinate_grid((16, 16))
        grid = grid.unsqueeze(0).expand(2, -1, -1, -1)
        
        pred = model(x, grid)
        
        assert pred.shape == y.shape, f"Shape mismatch: {pred.shape} vs {y.shape}"
        assert not torch.isnan(pred).any(), "TNO output contains NaN"
        assert torch.isfinite(pred).all(), "TNO output contains Inf"
        
        # Test attention visualization
        pred_with_attn, attention_maps = model.forward_with_attention_maps(x[:1], grid[:1])
        assert len(attention_maps) == model.num_layers, f"Wrong number of attention maps: {len(attention_maps)} vs {model.num_layers}"
        
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ TNO test passed (parameters: {params:,}, attention layers: {len(attention_maps)})")
        return True
        
    except Exception as e:
        print(f"❌ TNO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pno():
    """Test Probabilistic Neural Operator."""
    print("\n🎲 Testing Probabilistic Neural Operator...")
    
    try:
        from neural_operator_lab import ProbabilisticNeuralOperator
        from neural_operator_lab.data import create_synthetic_dataset
        
        # Test epistemic uncertainty
        model = ProbabilisticNeuralOperator(
            input_dim=1,
            output_dim=1,
            trunk_layers=[32, 32],
            branch_layers=[32, 32],
            num_basis=16,
            uncertainty_type='epistemic',
            modes=[8, 8]
        )
        
        x, y = create_synthetic_dataset('darcy', resolution=16, n_samples=2)
        
        # Test standard forward pass
        pred = model(x)
        assert pred.shape == y.shape, f"Shape mismatch: {pred.shape} vs {y.shape}"
        
        # Test uncertainty quantification
        mean, uncertainty = model.forward_with_uncertainty(x, num_samples=5)
        assert mean.shape == y.shape, f"Mean shape mismatch: {mean.shape} vs {y.shape}"
        assert uncertainty.shape == y.shape, f"Uncertainty shape mismatch: {uncertainty.shape} vs {y.shape}"
        assert (uncertainty >= 0).all(), "Uncertainty should be non-negative"
        
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ PNO test passed (parameters: {params:,}, uncertainty type: {model.uncertainty_type})")
        return True
        
    except Exception as e:
        print(f"❌ PNO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spectral_convolutions():
    """Test spectral convolution layers."""
    print("\n🌊 Testing Spectral Convolutions...")
    
    try:
        from neural_operator_lab.utils import SpectralConv1d, SpectralConv2d, SpectralConv3d
        
        # Test 1D
        conv1d = SpectralConv1d(in_channels=4, out_channels=8, modes=16)
        x1d = torch.randn(2, 4, 64)  # (batch, channels, spatial)
        out1d = conv1d(x1d)
        assert out1d.shape == (2, 8, 64), f"1D conv shape mismatch: {out1d.shape}"
        
        # Test 2D  
        conv2d = SpectralConv2d(in_channels=4, out_channels=8, modes1=8, modes2=8)
        x2d = torch.randn(2, 4, 32, 32)
        out2d = conv2d(x2d)
        assert out2d.shape == (2, 8, 32, 32), f"2D conv shape mismatch: {out2d.shape}"
        
        # Test 3D
        conv3d = SpectralConv3d(in_channels=4, out_channels=8, modes=[4, 4, 4])
        x3d = torch.randn(2, 4, 16, 16, 16)
        out3d = conv3d(x3d)
        assert out3d.shape == (2, 8, 16, 16, 16), f"3D conv shape mismatch: {out3d.shape}"
        
        print("✅ All spectral convolutions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Spectral convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\n📊 Testing Data Generation...")
    
    try:
        from neural_operator_lab.data import create_synthetic_dataset
        
        # Test Darcy flow
        x_darcy, y_darcy = create_synthetic_dataset('darcy', resolution=32, n_samples=10)
        assert x_darcy.shape == (10, 32, 32, 1), f"Darcy input shape: {x_darcy.shape}"
        assert y_darcy.shape == (10, 32, 32, 1), f"Darcy output shape: {y_darcy.shape}"
        
        # Test Burgers equation
        x_burgers, y_burgers = create_synthetic_dataset('burgers', resolution=64, n_samples=5)
        assert x_burgers.shape == (5, 64, 1), f"Burgers input shape: {x_burgers.shape}"
        assert y_burgers.shape == (5, 64, 1), f"Burgers output shape: {y_burgers.shape}"
        
        # Test data properties
        assert torch.isfinite(x_darcy).all(), "Darcy input contains non-finite values"
        assert torch.isfinite(y_darcy).all(), "Darcy output contains non-finite values"
        assert torch.isfinite(x_burgers).all(), "Burgers input contains non-finite values"
        assert torch.isfinite(y_burgers).all(), "Burgers output contains non-finite values"
        
        print("✅ Data generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False


def test_coordinate_grid():
    """Test coordinate grid generation."""
    print("\n📐 Testing Coordinate Grid Generation...")
    
    try:
        from neural_operator_lab.base import create_coordinate_grid
        
        # Test 1D grid
        grid_1d = create_coordinate_grid(64)
        assert grid_1d.shape == (64, 1), f"1D grid shape: {grid_1d.shape}"
        
        # Test 2D grid
        grid_2d = create_coordinate_grid((32, 32))
        assert grid_2d.shape == (32, 32, 2), f"2D grid shape: {grid_2d.shape}"
        
        # Test 3D grid
        grid_3d = create_coordinate_grid((16, 16, 16))
        assert grid_3d.shape == (16, 16, 16, 3), f"3D grid shape: {grid_3d.shape}"
        
        # Test value ranges
        assert (grid_2d >= 0).all() and (grid_2d <= 1).all(), "Grid values outside [0,1]"
        
        print("✅ Coordinate grid generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Coordinate grid test failed: {e}")
        return False


def run_all_tests():
    """Run all Generation 1 tests."""
    print("🚀 Neural Operator Foundation Lab - Generation 1 Test Suite")
    print("=" * 65)
    
    tests = [
        ("Imports", test_imports),
        ("FNO", test_fno),
        ("TNO", test_tno),
        ("PNO", test_pno),
        ("Spectral Convolutions", test_spectral_convolutions),
        ("Data Generation", test_data_generation),
        ("Coordinate Grids", test_coordinate_grid),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            if results[test_name]:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 65)
    print("📋 Test Summary:")
    print(f"✅ Passed: {passed}/{len(tests)} tests")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    if passed == len(tests):
        print("\n🎉 All Generation 1 tests passed! Ready for Generation 2.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please fix before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)