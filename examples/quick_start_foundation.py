#!/usr/bin/env python3
"""Quick start example for Universal Physics Transformer."""

import torch
import numpy as np
from neural_operator_lab.models import UniversalPhysicsTransformer, UniversalPhysicsConfig
from neural_operator_lab.validation import ModelValidator, ValidationMetrics
from neural_operator_lab.training import Trainer, TrainerConfig
from neural_operator_lab.data import create_pde_dataset


def generate_synthetic_navier_stokes(batch_size: int = 4, resolution: int = 64) -> tuple:
    """Generate synthetic Navier-Stokes data for testing."""
    # Create velocity and pressure fields
    x = torch.randn(batch_size, resolution, resolution, 4)  # u, v, w, p
    
    # Make velocity divergence-free (approximately)
    x[..., 0] = torch.sin(2 * np.pi * torch.linspace(0, 1, resolution)) * torch.ones(resolution).unsqueeze(0)
    x[..., 1] = torch.cos(2 * np.pi * torch.linspace(0, 1, resolution)) * torch.ones(resolution).unsqueeze(1)
    x[..., 2] = 0.1 * torch.randn(batch_size, resolution, resolution)  # w component
    x[..., 3] = 0.1 * torch.randn(batch_size, resolution, resolution)  # pressure
    
    # Target is evolved field (simplified)
    y = x + 0.01 * torch.randn_like(x)
    
    # Generate coordinate grid
    coords_x = torch.linspace(0, 1, resolution)
    coords_y = torch.linspace(0, 1, resolution)
    grid_x, grid_y = torch.meshgrid(coords_x, coords_y, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    return x, y, grid


def main():
    """Main demonstration of Universal Physics Transformer."""
    print("🚀 Universal Physics Transformer - Quick Start")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create model configuration
    config = UniversalPhysicsConfig(
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        max_resolution=128,
        fourier_features=128,
        enable_adaptive_resolution=True,
        enable_cross_attention=True
    )
    
    # 2. Initialize Universal Physics Transformer
    print("\n📊 Initializing Universal Physics Transformer...")
    model = UniversalPhysicsTransformer(
        config=config,
        input_dim=4,
        output_dim=4
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model summary: {model.get_model_summary()}")
    
    # 3. Generate synthetic data
    print("\n🔬 Generating synthetic Navier-Stokes data...")
    x_train, y_train, grid_train = generate_synthetic_navier_stokes(batch_size=8, resolution=64)
    x_test, y_test, grid_test = generate_synthetic_navier_stokes(batch_size=2, resolution=64)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # 4. Add PDE-specific head
    print("\n🧠 Adding Navier-Stokes head...")
    model.add_pde_head('navier_stokes_3d', output_channels=4)
    
    # 5. Test forward pass
    print("\n⚡ Testing forward pass...")
    model.eval()
    with torch.no_grad():
        pred = model(x_test[:1], grid_test[:1], pde_type='navier_stokes_3d')
        print(f"Prediction shape: {pred.shape}")
        print(f"Prediction range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    
    # 6. Validation
    print("\n🔍 Running model validation...")
    validator = ModelValidator(model, device='cpu')
    
    # Single prediction validation
    metrics = validator.validate_single_prediction(
        x_test[:1], y_test[:1], grid_test[:1]
    )
    print(f"Validation metrics: {metrics.to_dict()}")
    
    # 7. Basic training setup (demonstration)
    print("\n🏋️ Setting up training...")
    
    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, x, y, grid):
            self.x = x
            self.y = y
            self.grid = grid
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.grid[idx]
    
    train_dataset = SimpleDataset(x_train, y_train, grid_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Configure trainer
    trainer_config = TrainerConfig(
        learning_rate=1e-4,
        batch_size=2,
        max_epochs=5,
        validation_split=0.2,
        early_stopping_patience=3,
        mixed_precision=False  # Keep simple for demo
    )
    
    trainer = Trainer(
        model=model,
        config=trainer_config,
        device=device
    )
    
    # Quick training run
    print("\n🚀 Running quick training (5 epochs)...")
    trainer.train(train_loader, val_loader=None)
    
    # 8. Test few-shot adaptation
    print("\n🎯 Testing few-shot adaptation...")
    
    # Generate different physics data (simplified wave equation)
    x_wave = torch.randn(4, 32, 32, 2)  # Wave field
    y_wave = x_wave + 0.1 * torch.randn_like(x_wave)  # Evolved wave
    support_data = [(x_wave[i:i+1], y_wave[i:i+1]) for i in range(4)]
    
    # Few-shot adaptation
    model.few_shot_adaptation(support_data, 'wave_2d', num_steps=5)
    
    # Test adapted model
    with torch.no_grad():
        wave_pred = model(x_wave[:1], pde_type='wave_2d')
        print(f"Wave prediction shape: {wave_pred.shape}")
    
    # 9. Rollout validation
    print("\n🔄 Testing rollout stability...")
    rollout_results = validator.validate_rollout(
        initial_condition=x_test[:1],
        true_trajectory=torch.stack([y_test[:1] for _ in range(10)]),
        num_steps=10,
        dt=0.01
    )
    
    print(f"Rollout mean error: {rollout_results['mean_error']:.6f}")
    print(f"Rollout stable: {rollout_results['is_stable']}")
    print(f"Conservation drift: {rollout_results['conservation_drift']:.6f}")
    
    # 10. Model inspection
    print("\n🔍 Model inspection...")
    print("Available PDE heads:", list(model.pde_heads.keys()))
    print("Physics registry:", model.physics_embedding.physics_registry)
    print("Backbone frozen:", model.frozen_backbone)
    
    print("\n✅ Universal Physics Transformer demonstration complete!")
    print("=" * 50)
    
    return model, validator, trainer


if __name__ == "__main__":
    # Run demonstration
    model, validator, trainer = main()
    
    # Additional analysis
    print("\n📈 Additional Analysis:")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test model serialization
    print("\n💾 Testing model serialization...")
    torch.save(model.state_dict(), '/tmp/upt_demo.pt')
    print("Model saved successfully!")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")