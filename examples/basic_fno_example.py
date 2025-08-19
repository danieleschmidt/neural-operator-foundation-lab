#!/usr/bin/env python3
"""Basic example demonstrating FNO on synthetic data."""

import torch
import torch.nn.functional as F
from neural_operator_lab import FourierNeuralOperator
from neural_operator_lab.data import create_synthetic_dataset
import numpy as np
import matplotlib.pyplot as plt
import os


def train_basic_fno():
    """Train a basic FNO on synthetic Darcy flow data."""
    print("🚀 Neural Operator Foundation Lab - Basic FNO Example")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("\n📊 Generating synthetic Darcy flow dataset...")
    resolution = 64
    n_train = 800
    n_test = 200
    
    # Training data
    x_train, y_train = create_synthetic_dataset('darcy', resolution, n_train)
    x_test, y_test = create_synthetic_dataset('darcy', resolution, n_test)
    
    print(f"Training data shape: {x_train.shape} -> {y_train.shape}")
    print(f"Test data shape: {x_test.shape} -> {y_test.shape}")
    
    # Move to device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    # Create FNO model
    print("\n🧠 Creating Fourier Neural Operator...")
    model = FourierNeuralOperator(
        input_dim=1,      # Permeability field
        output_dim=1,     # Pressure field  
        modes=[16, 16],   # Fourier modes
        width=64,         # Hidden width
        num_layers=4      # Number of FNO blocks
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    print("\n🔄 Training FNO...")
    batch_size = 32
    epochs = 100
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Log progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Evaluation
    print("\n📈 Evaluating model on test set...")
    model.eval()
    test_loss = 0
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            test_loss += loss.item()
            
            predictions.append(pred)
    
    avg_test_loss = test_loss / ((len(x_test) + batch_size - 1) // batch_size)
    print(f"Final test MSE: {avg_test_loss:.6f}")
    
    # Visualize results
    print("\n🎨 Creating visualization...")
    predictions = torch.cat(predictions, dim=0)
    
    # Plot training curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Show example predictions
    idx = 0  # First test sample
    input_field = x_test[idx, :, :, 0].cpu().numpy()
    true_field = y_test[idx, :, :, 0].cpu().numpy() 
    pred_field = predictions[idx, :, :, 0].cpu().numpy()
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_field, cmap='RdBu_r')
    plt.colorbar()
    plt.title('True Solution')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_field, cmap='RdBu_r')
    plt.colorbar()
    plt.title('FNO Prediction')
    
    plt.tight_layout()
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/fno_basic_results.png', dpi=150, bbox_inches='tight')
    print("📁 Results saved to: outputs/fno_basic_results.png")
    
    # Compute error metrics
    relative_error = torch.mean((predictions - y_test)**2) / torch.mean(y_test**2)
    print(f"\n📊 Final Metrics:")
    print(f"Test MSE: {avg_test_loss:.6f}")
    print(f"Relative Error: {relative_error:.6f}")
    
    return model, (x_test, y_test, predictions)


if __name__ == "__main__":
    try:
        model, results = train_basic_fno()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()