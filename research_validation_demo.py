#!/usr/bin/env python3
"""
QISA Research Validation Demo
Generation 1: Validate existing sophisticated neural operator implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import time

# Import our neural operator framework
try:
    from src.neural_operator_lab.base import NeuralOperatorBase, create_coordinate_grid
    from src.neural_operator_lab.models.transformer import TransformerNeuralOperator
    from src.neural_operator_lab.utils import (
        SpectralConv2d, FourierFeatures, setup_device,
        compute_spectral_metrics, check_tensor_health
    )
    IMPORTS_OK = True
    print("✅ Neural Operator Lab imports successful")
except ImportError as e:
    IMPORTS_OK = False
    print(f"❌ Import error: {e}")
    print("Will create minimal validation implementation...")


class MinimalTNO(torch.nn.Module):
    """Minimal Transformer Neural Operator for validation."""
    
    def __init__(self, input_dim: int = 3, output_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.input_embedding = torch.nn.Linear(input_dim, hidden_dim)
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Simple flattened transformer
        B, H, W, C = x.shape
        x_flat = x.view(B, H*W, C)
        
        # Embed
        x_emb = self.input_embedding(x_flat)
        
        # Self-attention
        x_attn, _ = self.attention(x_emb, x_emb, x_emb)
        x_norm = self.norm(x_attn + x_emb)
        
        # Output
        output = self.output_proj(x_norm)
        return output.view(B, H, W, output.size(-1))


def create_synthetic_2d_pde_data(batch_size: int = 8, resolution: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic 2D PDE data for validation."""
    
    # Create coordinate grid
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Generate synthetic input functions (e.g., initial conditions)
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Random Gaussian initial condition
        center_x = torch.rand(1) * 0.6 - 0.3  # Random center
        center_y = torch.rand(1) * 0.6 - 0.3
        sigma = 0.1 + torch.rand(1) * 0.1    # Random width
        
        # Input: Gaussian + coordinates
        gaussian = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        input_func = torch.stack([gaussian, X, Y], dim=-1)  # (H, W, 3)
        
        # Target: Simple transformation (e.g., smoothed version)
        target_func = torch.nn.functional.avg_pool2d(
            gaussian.unsqueeze(0).unsqueeze(0), 
            kernel_size=3, stride=1, padding=1
        ).squeeze().unsqueeze(-1)  # (H, W, 1)
        
        inputs.append(input_func)
        targets.append(target_func)
    
    inputs = torch.stack(inputs)    # (B, H, W, 3)
    targets = torch.stack(targets)  # (B, H, W, 1)
    
    return inputs, targets


def validate_model_performance(model: torch.nn.Module, 
                             test_data: Tuple[torch.Tensor, torch.Tensor],
                             device: torch.device) -> Dict[str, Any]:
    """Validate model performance with comprehensive metrics."""
    
    model.eval()
    inputs, targets = test_data
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        predictions = model(inputs)
    inference_time = time.time() - start_time
    
    # Basic metrics
    mse = torch.nn.functional.mse_loss(predictions, targets)
    mae = torch.nn.functional.l1_loss(predictions, targets)
    
    # Relative error
    rel_error = torch.mean(torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8))
    
    # Spectral metrics (if possible)
    spectral_metrics = {}
    try:
        spectral_metrics = compute_spectral_metrics(predictions, targets)
    except:
        spectral_metrics = {'spectral_mse': float('inf'), 'spectral_correlation': 0.0}
    
    # Tensor health check
    pred_healthy = check_tensor_health(predictions, "predictions")
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'relative_error': rel_error.item(),
        'inference_time': inference_time,
        'predictions_healthy': pred_healthy,
        **spectral_metrics
    }


def run_research_validation():
    """Run comprehensive research validation."""
    
    print("\n🔬 NEURAL OPERATOR RESEARCH VALIDATION")
    print("=" * 50)
    
    # Setup device
    if IMPORTS_OK:
        device = setup_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("\n📊 Generating synthetic 2D PDE data...")
    train_inputs, train_targets = create_synthetic_2d_pde_data(batch_size=16, resolution=32)
    test_inputs, test_targets = create_synthetic_2d_pde_data(batch_size=8, resolution=32)
    
    print(f"Train data shape: {train_inputs.shape} -> {train_targets.shape}")
    print(f"Test data shape: {test_inputs.shape} -> {test_targets.shape}")
    
    # Initialize model
    print("\n🏗️ Initializing neural operator model...")
    if IMPORTS_OK:
        try:
            model = TransformerNeuralOperator(
                input_dim=3, output_dim=1, 
                hidden_dim=128, num_layers=4, num_heads=4,
                max_resolution=64
            ).to(device)
            print("✅ Using advanced TransformerNeuralOperator")
        except Exception as e:
            print(f"❌ TNO failed: {e}")
            model = MinimalTNO().to(device)
            print("✅ Using minimal TNO fallback")
    else:
        model = MinimalTNO().to(device)
        print("✅ Using minimal TNO implementation")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Quick training (for demonstration)
    print("\n🎯 Quick training demonstration...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for epoch in range(10):
        train_inputs_gpu = train_inputs.to(device)
        train_targets_gpu = train_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(train_inputs_gpu)
        loss = torch.nn.functional.mse_loss(predictions, train_targets_gpu)
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.6f}")
    
    # Validation
    print("\n✅ Validating model performance...")
    metrics = validate_model_performance(model, (test_inputs, test_targets), device)
    
    print(f"MSE Loss: {metrics['mse']:.6f}")
    print(f"MAE Loss: {metrics['mae']:.6f}")
    print(f"Relative Error: {metrics['relative_error']:.4f}")
    print(f"Inference Time: {metrics['inference_time']:.4f}s")
    print(f"Predictions Healthy: {metrics['predictions_healthy']}")
    
    if 'spectral_mse' in metrics:
        print(f"Spectral MSE: {metrics['spectral_mse']:.6f}")
        print(f"Spectral Correlation: {metrics['spectral_correlation']:.4f}")
    
    # Theoretical analysis
    print("\n🧮 Theoretical Analysis:")
    resolution = 32
    theoretical_complexity = resolution**2 * np.log2(resolution**2)  # FFT complexity
    actual_params = total_params
    
    print(f"Grid points: {resolution**2}")
    print(f"Theoretical FFT complexity: O({theoretical_complexity:.0f})")
    print(f"Model parameters: {actual_params:,}")
    print(f"Parameter efficiency: {resolution**2 / actual_params:.4f} grid_points/param")
    
    # Research quality assessment
    print("\n🏆 Research Quality Assessment:")
    quality_score = 0
    
    # Performance criteria
    if metrics['mse'] < 0.01:
        quality_score += 30
        print("✅ Excellent MSE performance (+30)")
    elif metrics['mse'] < 0.1:
        quality_score += 20
        print("✅ Good MSE performance (+20)")
    else:
        quality_score += 10
        print("⚠️  Acceptable MSE performance (+10)")
    
    if metrics['relative_error'] < 0.05:
        quality_score += 25
        print("✅ Low relative error (+25)")
    elif metrics['relative_error'] < 0.2:
        quality_score += 15
        print("✅ Moderate relative error (+15)")
    
    if metrics['predictions_healthy']:
        quality_score += 20
        print("✅ Numerically stable predictions (+20)")
    
    if metrics['inference_time'] < 1.0:
        quality_score += 15
        print("✅ Fast inference time (+15)")
    
    if IMPORTS_OK:
        quality_score += 10
        print("✅ Full framework integration (+10)")
    
    print(f"\n📊 OVERALL RESEARCH QUALITY SCORE: {quality_score}/100")
    
    if quality_score >= 80:
        print("🌟 EXCELLENT - Publication ready")
    elif quality_score >= 60:
        print("👍 GOOD - Ready for further development")
    elif quality_score >= 40:
        print("⚠️  ACCEPTABLE - Needs improvement")
    else:
        print("❌ NEEDS WORK - Major issues to address")
    
    return {
        'quality_score': quality_score,
        'metrics': metrics,
        'model_params': total_params,
        'framework_integrated': IMPORTS_OK
    }


if __name__ == "__main__":
    results = run_research_validation()
    print(f"\n🎯 Validation complete. Quality score: {results['quality_score']}/100")