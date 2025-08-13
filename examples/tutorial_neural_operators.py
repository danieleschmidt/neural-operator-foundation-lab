#!/usr/bin/env python3
"""
Neural Operators Tutorial

A comprehensive tutorial covering the fundamentals and advanced usage
of neural operators in the Neural Operator Foundation Lab.

This tutorial covers:
1. Basic neural operator concepts
2. Setting up and training models
3. Physics-informed learning
4. Cross-domain transfer
5. Performance optimization
6. Production deployment
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neural operator lab components
try:
    from neural_operator_lab.models.foundation import (
        UniversalPhysicsTransformer, 
        UniversalPhysicsConfig
    )
    from neural_operator_lab.models.fourier import FourierNeuralOperator
    from neural_operator_lab.models.transformer import TransformerNeuralOperator
    from neural_operator_lab.models.probabilistic import ProbabilisticNeuralOperator
    from neural_operator_lab.data.datasets import PhysicsDataset
    from neural_operator_lab.training.trainer import AdaptiveTrainer
    from neural_operator_lab.performance_optimizer import cached, profile
    from neural_operator_lab.robust_framework import robust_execution
    print("✅ All imports successful!")
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("❌ Import failed. Please install the neural_operator_lab package.")
    raise


class NeuralOperatorTutorial:
    """
    Comprehensive tutorial for neural operators.
    """
    
    def __init__(self):
        """Initialize the tutorial."""
        self.models = {}
        self.datasets = {}
        
        print("🎓 Neural Operator Tutorial initialized!")
        print("=" * 60)
    
    # ========================================================================
    # TUTORIAL 1: BASIC CONCEPTS AND SETUP
    # ========================================================================
    
    def tutorial_1_basic_concepts(self):
        """
        Tutorial 1: Understanding Neural Operators
        
        Neural operators learn mappings between function spaces,
        making them ideal for solving PDEs and physics simulations.
        """
        print("\\n📚 TUTORIAL 1: Basic Neural Operator Concepts")
        print("-" * 50)
        
        print("""
        Neural operators are a class of neural networks designed to learn 
        mappings between infinite-dimensional function spaces. Unlike traditional
        neural networks that work with finite-dimensional vectors, neural operators
        can handle functions as inputs and outputs.
        
        Key advantages:
        1. Resolution independence: Works with different grid resolutions
        2. Generalization: Can extrapolate to unseen domains
        3. Efficiency: Faster than traditional numerical solvers
        4. Physics awareness: Can incorporate physical constraints
        """)
        
        # Create a simple 1D function mapping example
        print("\\n🔍 Example: 1D Function Mapping")
        print("Input: u(x) = sin(kx)")
        print("Output: v(x) = d²u/dx² (second derivative)")
        
        # Generate sample data
        x = torch.linspace(0, 2*np.pi, 64)
        k_values = [1, 2, 3, 4]
        
        inputs = []
        targets = []
        
        for k in k_values:
            u = torch.sin(k * x)  # Input function
            v = -k**2 * torch.sin(k * x)  # Analytical second derivative
            
            inputs.append(u.unsqueeze(0))  # Add channel dimension
            targets.append(v.unsqueeze(0))
        
        # Stack into batches
        input_batch = torch.stack(inputs)
        target_batch = torch.stack(targets)
        
        print(f"Input shape: {input_batch.shape}")  # [batch, channels, spatial]
        print(f"Target shape: {target_batch.shape}")
        
        # Visualize the mapping
        self._plot_function_mapping(x, input_batch, target_batch)
        
        print("✅ Tutorial 1 completed!")
        return input_batch, target_batch
    
    def _plot_function_mapping(self, x, inputs, targets):
        """Plot the function mapping example."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(4):
            axes[i].plot(x, inputs[i, 0], label=f'Input u(x), k={i+1}', linewidth=2)
            axes[i].plot(x, targets[i, 0], label=f'Target d²u/dx², k={i+1}', linewidth=2, linestyle='--')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'k = {i+1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Neural Operator Function Mapping Example')
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TUTORIAL 2: FOURIER NEURAL OPERATORS
    # ========================================================================
    
    def tutorial_2_fourier_operators(self):
        """
        Tutorial 2: Fourier Neural Operators (FNO)
        
        FNOs use the Fourier transform to handle global dependencies
        efficiently in the frequency domain.
        """
        print("\\n📚 TUTORIAL 2: Fourier Neural Operators")
        print("-" * 50)
        
        print("""
        Fourier Neural Operators (FNO) leverage the Fourier transform to:
        1. Capture global dependencies efficiently
        2. Handle periodic boundary conditions naturally  
        3. Achieve resolution independence
        4. Scale well to high-dimensional problems
        """)
        
        # Create 2D Fourier Neural Operator
        print("\\n🏗️ Creating 2D Fourier Neural Operator")
        
        fno = FourierNeuralOperator(
            input_dim=3,      # 3 input channels (u, v, p for fluid dynamics)
            output_dim=3,     # 3 output channels
            modes=[16, 16],   # Fourier modes in x and y directions
            width=32,         # Hidden channel width
            num_layers=4,     # Number of Fourier layers
            activation='gelu'
        )
        
        self.models['fno_2d'] = fno
        
        print(f"Model parameters: {fno.count_parameters():,}")
        print(f"Model summary: {fno.get_model_summary()}")
        
        # Generate 2D fluid dynamics data
        print("\\n🌊 Generating 2D Fluid Dynamics Data")
        
        resolution = 32
        num_samples = 100
        
        input_data, target_data = self._generate_2d_fluid_data(resolution, num_samples)
        
        print(f"Generated data shapes:")
        print(f"  Input: {input_data.shape}")
        print(f"  Target: {target_data.shape}")
        
        # Test forward pass
        print("\\n⚡ Testing Forward Pass")
        with torch.no_grad():
            sample_input = input_data[:4]  # First 4 samples
            output = fno(sample_input)
            
            print(f"Input shape: {sample_input.shape}")
            print(f"Output shape: {output.shape}")
            
            # Compute error
            sample_target = target_data[:4]
            mse = torch.mean((output - sample_target) ** 2)
            print(f"Initial MSE (untrained): {mse:.6f}")
        
        # Visualize one sample
        self._visualize_2d_sample(input_data[0], target_data[0], output[0])
        
        print("✅ Tutorial 2 completed!")
        return input_data, target_data
    
    def _generate_2d_fluid_data(self, resolution: int, num_samples: int):
        """Generate simplified 2D fluid dynamics data."""
        
        # Create coordinate grid
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Random wave numbers
            k1, k2 = np.random.randint(1, 4, 2)
            A, B = np.random.uniform(0.5, 2.0, 2)
            
            # Velocity components
            u = A * torch.sin(k1 * X) * torch.cos(k2 * Y)
            v = -B * torch.cos(k1 * X) * torch.sin(k2 * Y)
            
            # Pressure
            p = torch.sin(X) * torch.cos(Y)
            
            # Stack channels
            input_field = torch.stack([u, v, p], dim=0)
            
            # Simple evolution (time step)
            dt = 0.01
            u_new = u * (1 - dt) + dt * torch.sin(2 * X)
            v_new = v * (1 - dt) + dt * torch.cos(2 * Y)
            p_new = p * (1 - dt * 0.1)
            
            target_field = torch.stack([u_new, v_new, p_new], dim=0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _visualize_2d_sample(self, input_sample, target_sample, prediction_sample):
        """Visualize a 2D fluid dynamics sample."""
        
        channels = ['u', 'v', 'p']
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        
        for i, channel in enumerate(channels):
            # Input
            im1 = axes[i, 0].imshow(input_sample[i].cpu().numpy(), cmap='RdBu_r', origin='lower')
            axes[i, 0].set_title(f'Input {channel}')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Target
            im2 = axes[i, 1].imshow(target_sample[i].cpu().numpy(), cmap='RdBu_r', origin='lower')
            axes[i, 1].set_title(f'Target {channel}')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Prediction
            im3 = axes[i, 2].imshow(prediction_sample[i].detach().cpu().numpy(), cmap='RdBu_r', origin='lower')
            axes[i, 2].set_title(f'Prediction {channel}')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.suptitle('2D Fluid Dynamics: Input → Target → Prediction')
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TUTORIAL 3: TRANSFORMER NEURAL OPERATORS
    # ========================================================================
    
    def tutorial_3_transformer_operators(self):
        """
        Tutorial 3: Transformer Neural Operators (TNO)
        
        TNOs use attention mechanisms to capture long-range dependencies
        and enable cross-domain learning.
        """
        print("\\n📚 TUTORIAL 3: Transformer Neural Operators")
        print("-" * 50)
        
        print("""
        Transformer Neural Operators (TNO) combine the power of:
        1. Self-attention for long-range dependencies
        2. Cross-attention for multi-modal data
        3. Positional encoding for spatial awareness
        4. Physics-informed embeddings
        """)
        
        # Create Transformer Neural Operator
        print("\\n🏗️ Creating Transformer Neural Operator")
        
        tno = TransformerNeuralOperator(
            input_dim=4,           # Input channels
            output_dim=3,          # Output channels
            hidden_dim=128,        # Hidden dimension
            num_layers=4,          # Number of transformer layers
            num_heads=8,           # Number of attention heads
            dropout=0.1,
            max_seq_length=1024,   # Max sequence length
            use_positional_encoding=True,
            use_physics_embedding=True
        )
        
        self.models['tno'] = tno
        
        print(f"Model parameters: {tno.count_parameters():,}")
        
        # Generate multi-physics data
        print("\\n🔬 Generating Multi-Physics Data")
        
        # Create data with different physics types
        fluid_data = self._generate_physics_sequence('fluid', 64, 50)
        heat_data = self._generate_physics_sequence('heat', 64, 50)
        
        # Combine different physics types
        combined_input = torch.cat([fluid_data[0], heat_data[0]], dim=0)
        combined_target = torch.cat([fluid_data[1], heat_data[1]], dim=0)
        
        # Create physics labels for cross-attention
        physics_labels = torch.cat([
            torch.zeros(50, dtype=torch.long),  # Fluid
            torch.ones(50, dtype=torch.long)    # Heat
        ])
        
        print(f"Combined input shape: {combined_input.shape}")
        print(f"Physics labels: {physics_labels.shape}")
        
        # Test with attention visualization
        print("\\n👁️ Testing Attention Mechanisms")
        
        with torch.no_grad():
            sample_batch = combined_input[:4]
            sample_labels = physics_labels[:4]
            
            # Forward pass with attention weights
            output, attention_weights = tno.forward_with_attention(sample_batch)
            
            print(f"Output shape: {output.shape}")
            print(f"Attention weights shape: {attention_weights.shape}")
            
            # Visualize attention patterns
            self._visualize_attention_patterns(attention_weights[0])  # First sample
        
        print("✅ Tutorial 3 completed!")
        return combined_input, combined_target, physics_labels
    
    def _generate_physics_sequence(self, physics_type: str, seq_len: int, num_samples: int):
        """Generate sequence data for specific physics type."""
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            if physics_type == 'fluid':
                # Fluid dynamics sequence
                t = torch.linspace(0, 2*np.pi, seq_len)
                u = torch.sin(2 * t + i * 0.1)
                v = torch.cos(2 * t + i * 0.1) 
                p = torch.sin(t) * torch.cos(t)
                rho = torch.ones_like(t)
                
                input_seq = torch.stack([u, v, p, rho], dim=1)  # [seq_len, channels]
                
                # Target: next time step
                target_u = torch.sin(2 * t + i * 0.1 + 0.1)
                target_v = torch.cos(2 * t + i * 0.1 + 0.1)
                target_p = torch.sin(t + 0.1) * torch.cos(t + 0.1)
                
                target_seq = torch.stack([target_u, target_v, target_p], dim=1)
                
            elif physics_type == 'heat':
                # Heat transfer sequence
                t = torch.linspace(0, 1, seq_len)
                T = 100 * torch.exp(-5 * t) + 20 + i * torch.sin(10 * t)
                q = -torch.gradient(T, spacing=t[1] - t[0])[0]
                k = torch.ones_like(t)  # Thermal conductivity
                cp = torch.ones_like(t) * 1000  # Specific heat
                
                input_seq = torch.stack([T, q, k, cp], dim=1)
                
                # Target: evolved temperature
                target_T = T * 0.9 + torch.sin(5 * t) * 2
                target_q = -torch.gradient(target_T, spacing=t[1] - t[0])[0]
                target_flux = target_q * k
                
                target_seq = torch.stack([target_T, target_q, target_flux], dim=1)
            
            input_data.append(input_seq)
            target_data.append(target_seq)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _visualize_attention_patterns(self, attention_weights):
        """Visualize attention patterns."""
        
        # attention_weights shape: [num_heads, seq_len, seq_len]
        num_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for head in range(min(8, num_heads)):
            im = axes[head].imshow(attention_weights[head].cpu().numpy(), cmap='Blues')
            axes[head].set_title(f'Attention Head {head + 1}')
            axes[head].set_xlabel('Key Position')
            axes[head].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[head])
        
        plt.suptitle('Transformer Neural Operator Attention Patterns')
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TUTORIAL 4: PHYSICS-INFORMED LEARNING
    # ========================================================================
    
    def tutorial_4_physics_informed(self):
        """
        Tutorial 4: Physics-Informed Neural Operators
        
        Incorporate physical laws and constraints directly into training.
        """
        print("\\n📚 TUTORIAL 4: Physics-Informed Learning")
        print("-" * 50)
        
        print("""
        Physics-Informed Neural Operators enforce physical laws by:
        1. Adding physics-based loss terms
        2. Enforcing conservation laws
        3. Respecting boundary conditions
        4. Incorporating symmetries and invariances
        """)
        
        # Create Universal Physics Transformer
        print("\\n🌍 Creating Universal Physics Transformer")
        
        config = UniversalPhysicsConfig(
            input_dim=32,
            output_dim=32,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            physics_types=['fluid_dynamics', 'heat_transfer'],
            cross_physics_attention=True,
            conservation_loss_weight=0.1,
            boundary_loss_weight=0.05,
            symmetry_loss_weight=0.02
        )
        
        upt = UniversalPhysicsTransformer(config)
        self.models['universal_physics'] = upt
        
        print(f"Model parameters: {upt.count_parameters():,}")
        
        # Generate physics-informed data
        print("\\n⚖️ Generating Physics-Informed Data")
        
        data = self._generate_conservation_law_data(64, 100)
        
        # Demonstrate physics-informed loss
        print("\\n🧮 Computing Physics-Informed Loss")
        
        sample_input = data['input'][:8]
        sample_target = data['target'][:8]
        
        with torch.no_grad():
            prediction = upt(sample_input)
            
            # Standard data loss
            data_loss = nn.MSELoss()(prediction, sample_target)
            
            # Physics-informed losses
            conservation_loss = self._compute_conservation_loss(sample_input, prediction)
            boundary_loss = self._compute_boundary_loss(prediction)
            
            # Total physics-informed loss
            total_loss = (data_loss + 
                         config.conservation_loss_weight * conservation_loss +
                         config.boundary_loss_weight * boundary_loss)
            
            print(f"Data Loss: {data_loss:.6f}")
            print(f"Conservation Loss: {conservation_loss:.6f}")
            print(f"Boundary Loss: {boundary_loss:.6f}")
            print(f"Total Physics-Informed Loss: {total_loss:.6f}")
        
        # Visualize conservation properties
        self._visualize_conservation(sample_input[0], prediction[0])
        
        print("✅ Tutorial 4 completed!")
        return data
    
    def _generate_conservation_law_data(self, resolution: int, num_samples: int):
        """Generate data satisfying conservation laws."""
        
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Create a divergence-free velocity field (incompressible flow)
            psi = torch.sin(2*X) * torch.cos(2*Y)  # Stream function
            
            # Velocity from stream function (ensures div(u) = 0)
            u = torch.gradient(psi, dim=1)[0]   # ∂ψ/∂y
            v = -torch.gradient(psi, dim=0)[0]  # -∂ψ/∂x
            
            # Density (conserved quantity)
            rho = 1 + 0.1 * torch.sin(X + Y)
            
            # Combine into field
            input_field = torch.stack([u, v, rho], dim=0)
            
            # Evolution preserving conservation (simplified)
            dt = 0.01
            u_new = u + dt * torch.sin(X)
            v_new = v + dt * torch.cos(Y)
            rho_new = rho  # Mass conservation
            
            target_field = torch.stack([u_new, v_new, rho_new], dim=0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return {
            'input': torch.stack(input_data),
            'target': torch.stack(target_data)
        }
    
    def _compute_conservation_loss(self, inputs, predictions):
        """Compute conservation law violation."""
        
        # Extract velocity components
        u = predictions[:, 0]  # u-velocity
        v = predictions[:, 1]  # v-velocity
        
        # Compute divergence ∇ · u (should be zero for incompressible flow)
        du_dx = torch.gradient(u, dim=2)[0]
        dv_dy = torch.gradient(v, dim=1)[0] 
        
        divergence = du_dx + dv_dy
        
        # Conservation loss: minimize divergence
        conservation_loss = torch.mean(divergence ** 2)
        
        return conservation_loss
    
    def _compute_boundary_loss(self, predictions):
        """Compute boundary condition violation."""
        
        # For this example, enforce zero velocity at boundaries
        boundary_loss = (
            torch.mean(predictions[:, :2, 0, :] ** 2) +  # Top boundary
            torch.mean(predictions[:, :2, -1, :] ** 2) +  # Bottom boundary
            torch.mean(predictions[:, :2, :, 0] ** 2) +   # Left boundary
            torch.mean(predictions[:, :2, :, -1] ** 2)    # Right boundary
        )
        
        return boundary_loss
    
    def _visualize_conservation(self, input_field, prediction):
        """Visualize conservation properties."""
        
        u = prediction[0].detach().cpu().numpy()
        v = prediction[1].detach().cpu().numpy()
        rho = prediction[2].detach().cpu().numpy()
        
        # Compute divergence
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        divergence = du_dx + dv_dy
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity field
        im1 = axes[0, 0].streamplot(np.arange(u.shape[1]), np.arange(u.shape[0]), u, v)
        axes[0, 0].set_title('Velocity Field')
        
        # Density
        im2 = axes[0, 1].imshow(rho, cmap='viridis', origin='lower')
        axes[0, 1].set_title('Density')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Divergence (should be close to zero)
        im3 = axes[1, 0].imshow(divergence, cmap='RdBu_r', origin='lower')
        axes[1, 0].set_title(f'Divergence (max: {np.max(np.abs(divergence)):.6f})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Divergence histogram
        axes[1, 1].hist(divergence.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Divergence')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Divergence Distribution')
        axes[1, 1].axvline(0, color='red', linestyle='--', label='Perfect Conservation')
        axes[1, 1].legend()
        
        plt.suptitle('Physics-Informed Predictions: Conservation Properties')
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TUTORIAL 5: TRAINING AND OPTIMIZATION
    # ========================================================================
    
    @profile('training_tutorial')
    def tutorial_5_training(self):
        """
        Tutorial 5: Training Neural Operators
        
        Best practices for training, optimization, and performance tuning.
        """
        print("\\n📚 TUTORIAL 5: Training and Optimization")
        print("-" * 50)
        
        print("""
        Effective training of neural operators requires:
        1. Proper data preprocessing and augmentation
        2. Adaptive learning rate schedules
        3. Physics-informed loss functions
        4. Regularization techniques
        5. Performance optimization
        """)
        
        # Use the FNO model from tutorial 2
        if 'fno_2d' not in self.models:
            self.tutorial_2_fourier_operators()
        
        model = self.models['fno_2d']
        
        # Generate training data
        print("\\n📊 Preparing Training Data")
        
        train_input, train_target = self._generate_2d_fluid_data(32, 500)
        val_input, val_target = self._generate_2d_fluid_data(32, 100)
        
        # Create datasets
        train_dataset = PhysicsDataset(
            inputs=train_input,
            targets=train_target,
            transform=self._data_augmentation
        )
        
        val_dataset = PhysicsDataset(
            inputs=val_input,
            targets=val_target
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Initialize trainer with advanced features
        print("\\n🏋️ Setting up Advanced Trainer")
        
        trainer = AdaptiveTrainer(
            model=model,
            learning_rate=1e-3,
            batch_size=16,
            num_epochs=20,
            early_stopping_patience=5,
            use_mixed_precision=True,
            gradient_clipping=1.0,
            weight_decay=1e-4
        )
        
        # Add custom callbacks
        trainer.add_callback('lr_scheduler', self._get_lr_scheduler)
        trainer.add_callback('physics_validation', self._physics_validation_callback)
        
        # Train the model
        print("\\n🚀 Starting Training")
        
        training_metrics = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            verbose=True
        )
        
        print(f"Training completed!")
        print(f"Best validation loss: {training_metrics['best_val_loss']:.6f}")
        print(f"Final training loss: {training_metrics['final_train_loss']:.6f}")
        
        # Analyze training dynamics
        self._plot_training_curves(training_metrics['train_losses'], 
                                 training_metrics['val_losses'])
        
        # Performance optimization demonstration
        self.demo_performance_optimization(model)
        
        print("✅ Tutorial 5 completed!")
        return training_metrics
    
    def _data_augmentation(self, sample):
        """Data augmentation for physics data."""
        input_data, target_data = sample
        
        # Random rotation (90 degree increments to preserve physics)
        if torch.rand(1) > 0.5:
            k = torch.randint(0, 4, (1,)).item()
            input_data = torch.rot90(input_data, k, dims=[-2, -1])
            target_data = torch.rot90(target_data, k, dims=[-2, -1])
        
        # Random flipping
        if torch.rand(1) > 0.5:
            input_data = torch.flip(input_data, dims=[-1])
            target_data = torch.flip(target_data, dims=[-1])
        
        # Small random noise
        if torch.rand(1) > 0.7:
            noise_std = 0.01
            input_data = input_data + torch.randn_like(input_data) * noise_std
        
        return input_data, target_data
    
    def _get_lr_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    def _physics_validation_callback(self, model, val_data, epoch):
        """Custom physics validation callback."""
        with torch.no_grad():
            sample_input = val_data[:4]
            prediction = model(sample_input)
            
            # Compute physics-based metrics
            conservation_error = self._compute_conservation_loss(sample_input, prediction)
            
            return {'conservation_error': conservation_error.item()}
    
    def _plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation curves."""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Loss difference
        loss_diff = np.array(val_losses) - np.array(train_losses)
        ax2.plot(epochs, loss_diff, label='Val - Train Loss', color='red', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.set_title('Overfitting Indicator')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    @cached(maxsize=10, ttl=300)
    def demo_performance_optimization(self, model):
        """Demonstrate performance optimization techniques."""
        print("\\n⚡ Performance Optimization Demo")
        
        # Create test data
        test_data = torch.randn(32, 3, 32, 32)
        
        # Baseline inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_data)
        baseline_time = time.time() - start_time
        
        print(f"Baseline inference time (100 runs): {baseline_time:.3f}s")
        
        # Optimized model (compilation, if available)
        if hasattr(torch, 'compile'):
            optimized_model = torch.compile(model, mode='reduce-overhead')
            
            # Warm up
            with torch.no_grad():
                _ = optimized_model(test_data)
            
            # Optimized inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = optimized_model(test_data)
            optimized_time = time.time() - start_time
            
            speedup = baseline_time / optimized_time
            print(f"Optimized inference time (100 runs): {optimized_time:.3f}s")
            print(f"Speedup: {speedup:.2f}x")
        
        # Memory usage optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Run inference
            with torch.no_grad():
                _ = model(test_data.cuda())
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
            
            print(f"GPU memory usage: {memory_usage:.2f}MB")
    
    # ========================================================================
    # MAIN TUTORIAL RUNNER
    # ========================================================================
    
    def run_all_tutorials(self):
        """Run all tutorials in sequence."""
        print("🎓 Starting Complete Neural Operator Tutorial")
        print("=" * 60)
        
        try:
            # Tutorial 1: Basic concepts
            self.tutorial_1_basic_concepts()
            
            # Tutorial 2: Fourier Neural Operators
            self.tutorial_2_fourier_operators()
            
            # Tutorial 3: Transformer Neural Operators  
            self.tutorial_3_transformer_operators()
            
            # Tutorial 4: Physics-informed learning
            self.tutorial_4_physics_informed()
            
            # Tutorial 5: Training and optimization
            self.tutorial_5_training()
            
            print("\\n🎉 ALL TUTORIALS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Summary
            print("\\n📋 Tutorial Summary:")
            print(f"✅ Models created: {len(self.models)}")
            print(f"✅ Datasets generated: {len(self.datasets)}")
            print("✅ Key concepts covered:")
            print("   - Neural operator fundamentals")
            print("   - Fourier Neural Operators (FNO)")
            print("   - Transformer Neural Operators (TNO)")
            print("   - Physics-informed learning")
            print("   - Training optimization")
            print("   - Performance tuning")
            
            return True
            
        except Exception as e:
            logger.error(f"Tutorial failed: {e}")
            return False


def main():
    """Main tutorial execution."""
    
    # Initialize tutorial
    tutorial = NeuralOperatorTutorial()
    
    # Run all tutorials
    success = tutorial.run_all_tutorials()
    
    if success:
        print("\\n🏆 Congratulations! You've completed the Neural Operator Tutorial!")
        print("You're now ready to apply neural operators to your physics problems.")
    else:
        print("\\n❌ Tutorial encountered errors. Please check the logs.")
    
    return success


if __name__ == "__main__":
    main()