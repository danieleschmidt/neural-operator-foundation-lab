#!/usr/bin/env python3
"""Demonstration of Transformer Neural Operator on synthetic data."""

import torch
import torch.nn.functional as F
from neural_operator_lab import TransformerNeuralOperator
from neural_operator_lab.data import create_synthetic_dataset
from neural_operator_lab.base import create_coordinate_grid
import numpy as np
import matplotlib.pyplot as plt
import os


def demo_transformer_neural_operator():
    """Demonstrate TNO capabilities."""
    print("🌟 Neural Operator Foundation Lab - Transformer Neural Operator Demo")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("\n📊 Generating synthetic dataset...")
    resolution = 32  # Smaller for TNO demo
    n_samples = 500
    
    x_data, y_data = create_synthetic_dataset('darcy', resolution, n_samples)
    
    # Split data
    n_train = int(0.8 * n_samples)
    x_train, y_train = x_data[:n_train], y_data[:n_train]
    x_test, y_test = x_data[n_train:], y_data[n_train:]
    
    print(f"Training: {x_train.shape} -> {y_train.shape}")
    print(f"Testing: {x_test.shape} -> {y_test.shape}")
    
    # Create coordinate grid
    grid = create_coordinate_grid((resolution, resolution), domain=(0.0, 1.0), device=device)
    grid = grid.unsqueeze(0).expand(x_train.size(0), -1, -1, -1)
    grid_test = grid[:x_test.size(0)]
    
    # Move to device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    # Create TNO model
    print("\n🎯 Creating Transformer Neural Operator...")
    model = TransformerNeuralOperator(
        input_dim=1,
        output_dim=1,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        fourier_features=True,
        max_resolution=64
    ).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TNO parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, total_steps=50, pct_start=0.3
    )
    
    # Training loop
    print("\n🔄 Training TNO...")
    batch_size = 16
    epochs = 50
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(x_train), batch_size):
            end_idx = min(i + batch_size, len(x_train))
            batch_x = x_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            batch_grid = grid[:end_idx-i]
            
            optimizer.zero_grad()
            
            # Forward pass with coordinate grid
            pred = model(batch_x, batch_grid)
            loss = F.mse_loss(pred, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
    
    # Evaluation with attention visualization
    print("\n📈 Evaluating with attention maps...")
    model.eval()
    
    with torch.no_grad():
        # Get prediction and attention for first test sample
        sample_x = x_test[:1]  # First sample
        sample_grid = grid_test[:1]
        
        pred, attention_maps = model.forward_with_attention_maps(sample_x, sample_grid)
        test_loss = F.mse_loss(pred, y_test[:1]).item()
        
        print(f"Test sample MSE: {test_loss:.6f}")
        print(f"Attention maps collected from {len(attention_maps)} layers")
    
    # Visualization
    print("\n🎨 Creating visualizations...")
    
    # Convert to numpy for plotting
    input_field = sample_x[0, :, :, 0].cpu().numpy()
    true_field = y_test[0, :, :, 0].cpu().numpy()
    pred_field = pred[0, :, :, 0].cpu().numpy()
    
    # Plot results and attention
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input, True, Prediction
    im1 = axes[0, 0].imshow(input_field, cmap='viridis')
    axes[0, 0].set_title('Input (Permeability)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(true_field, cmap='RdBu_r')
    axes[0, 1].set_title('True Solution')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(pred_field, cmap='RdBu_r')
    axes[0, 2].set_title('TNO Prediction')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Attention visualizations (first, middle, last layer)
    layer_indices = [0, len(attention_maps)//2, -1]
    layer_names = ['First Layer', 'Middle Layer', 'Last Layer']
    
    for i, (layer_idx, name) in enumerate(zip(layer_indices, layer_names)):
        # Average attention over heads and select center token
        attn = attention_maps[layer_idx][0].mean(0)  # Average over heads
        center_token = attn.size(0) // 2
        attn_map = attn[center_token].view(resolution, resolution).cpu().numpy()
        
        im = axes[1, i].imshow(attn_map, cmap='Blues')
        axes[1, i].set_title(f'{name} Attention')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/tno_demo_results.png', dpi=150, bbox_inches='tight')
    print("📁 Results saved to: outputs/tno_demo_results.png")
    
    # Performance summary
    print(f"\n📊 TNO Performance Summary:")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    relative_error = torch.mean((pred - y_test[:1])**2) / torch.mean(y_test[:1]**2)
    print(f"Relative error: {relative_error.item():.6f}")
    
    return model, attention_maps


if __name__ == "__main__":
    try:
        model, attention_maps = demo_transformer_neural_operator()
        print("\n✅ TNO demo completed successfully!")
        print(f"🎯 Attention patterns captured from {len(attention_maps)} transformer layers")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()