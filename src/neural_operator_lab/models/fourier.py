"""Fourier Neural Operator (FNO) implementation."""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import numpy as np

from ..base import NeuralOperatorBase
from ..utils import SpectralConv1d, SpectralConv2d, SpectralConv3d


class FNOBlock(nn.Module):
    """Single FNO block with spectral and local convolutions."""
    
    def __init__(self, modes: List[int], width: int, activation: str = 'gelu'):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Spectral convolution based on dimensionality
        if len(modes) == 1:
            self.spectral_conv = SpectralConv1d(width, width, modes[0])
        elif len(modes) == 2:
            self.spectral_conv = SpectralConv2d(width, width, modes[0], modes[1])
        elif len(modes) == 3:
            self.spectral_conv = SpectralConv3d(width, width, modes)
        else:
            raise ValueError(f"Unsupported number of modes: {len(modes)}")
        
        # Local convolution (pointwise)
        self.local_conv = nn.Conv1d(width, width, 1) if len(modes) == 1 else \
                         nn.Conv2d(width, width, 1) if len(modes) == 2 else \
                         nn.Conv3d(width, width, 1)
        
        # Activation
        self.activation = getattr(torch.nn.functional, activation)
        
        # Normalization
        self.norm = nn.LayerNorm(width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO block."""
        batch_size = x.size(0)
        
        # Move channel dimension for convolution
        if len(self.modes) == 1:
            x_conv = x.permute(0, 2, 1)  # (B, C, N)
        elif len(self.modes) == 2:
            x_conv = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        elif len(self.modes) == 3:
            x_conv = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        
        # Spectral convolution
        x1 = self.spectral_conv(x_conv)
        
        # Local convolution
        x2 = self.local_conv(x_conv)
        
        # Combine and move channel dimension back
        out = x1 + x2
        
        if len(self.modes) == 1:
            out = out.permute(0, 2, 1)  # (B, N, C)
        elif len(self.modes) == 2:
            out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        elif len(self.modes) == 3:
            out = out.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        
        # Residual connection and activation
        out = x + out
        out = self.activation(out)
        out = self.norm(out)
        
        return out


class FourierNeuralOperator(NeuralOperatorBase):
    """Fourier Neural Operator for learning operators between function spaces."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 modes: List[int],
                 width: int = 64,
                 num_layers: int = 4,
                 activation: str = 'gelu',
                 padding: int = 8,
                 **kwargs):
        """Initialize FNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension  
            modes: Number of Fourier modes to keep in each dimension
            width: Width of hidden layers
            num_layers: Number of FNO blocks
            activation: Activation function
            padding: Padding for handling different resolutions
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, width)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(modes, width, activation) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(width, 128),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        # Learnable padding
        if padding > 0:
            self.pad_layer = nn.ReflectionPad2d(padding) if len(modes) == 2 else \
                           nn.ReflectionPad3d(padding) if len(modes) == 3 else \
                           nn.ReflectionPad1d(padding)
    
    def forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from input function."""
        # Add positional encoding if grid provided
        if grid is not None:
            # Concatenate input with grid coordinates
            if x.dim() == grid.dim():
                x = torch.cat([x, grid], dim=-1)
            else:
                # Broadcast grid to match input batch size
                grid_expanded = grid.unsqueeze(0).expand(x.size(0), *[-1]*grid.dim())
                x = torch.cat([x, grid_expanded], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply padding if specified
        if hasattr(self, 'pad_layer') and self.padding > 0:
            # Move channel dimension for padding
            original_shape = x.shape
            if len(self.modes) == 2:
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = self.pad_layer(x)
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            elif len(self.modes) == 3:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
                x = self.pad_layer(x)
                x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        
        return x
    
    def forward_operator(self, features: torch.Tensor) -> torch.Tensor:
        """Apply FNO transformation."""
        x = features
        
        # Apply FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Remove padding if it was added
        if hasattr(self, 'pad_layer') and self.padding > 0:
            if len(self.modes) == 2:
                x = x[:, self.padding:-self.padding, self.padding:-self.padding, :]
            elif len(self.modes) == 3:
                x = x[:, self.padding:-self.padding, self.padding:-self.padding, 
                      self.padding:-self.padding, :]
            elif len(self.modes) == 1:
                x = x[:, self.padding:-self.padding, :]
        
        # Project to output dimension
        output = self.output_proj(x)
        
        return output
    
    def compute_pde_residual(self, pred: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss for common PDEs."""
        # This is a placeholder - specific implementations should override
        # For demonstration, compute a simple physics constraint
        
        # Example: penalize non-smooth solutions
        if pred.dim() >= 3:  # Spatial dimensions present
            # Compute total variation as physics penalty
            tv_loss = 0
            for dim in range(1, pred.dim() - 1):  # Skip batch and channel dims
                diff = torch.diff(pred, dim=dim)
                tv_loss += torch.sum(torch.abs(diff))
            
            return tv_loss / pred.numel()
        
        return torch.tensor(0.0, device=pred.device)
    
    def get_fourier_modes(self) -> List[int]:
        """Get the Fourier modes used by this model."""
        return self.modes
    
    def set_fourier_modes(self, new_modes: List[int]):
        """Update Fourier modes (requires retraining)."""
        if len(new_modes) != len(self.modes):
            raise ValueError("New modes must have same dimensionality")
        
        self.modes = new_modes
        
        # Reinitialize spectral convolutions with new modes
        for block in self.fno_blocks:
            if len(new_modes) == 1:
                block.spectral_conv = SpectralConv1d(self.width, self.width, new_modes[0])
            elif len(new_modes) == 2:
                block.spectral_conv = SpectralConv2d(self.width, self.width, new_modes[0], new_modes[1])
            elif len(new_modes) == 3:
                block.spectral_conv = SpectralConv3d(self.width, self.width, new_modes)


class AdaptiveFNO(FourierNeuralOperator):
    """Adaptive FNO that can adjust modes during training."""
    
    def __init__(self, *args, adaptive_threshold: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_threshold = adaptive_threshold
        self.mode_importance = {}
    
    def forward_operator(self, features: torch.Tensor) -> torch.Tensor:
        """Apply FNO with mode importance tracking."""
        x = features
        
        # Track mode importance during forward pass
        for i, block in enumerate(self.fno_blocks):
            x_prev = x.clone()
            x = block(x)
            
            # Compute mode importance (simplified)
            importance = torch.mean(torch.abs(x - x_prev))
            self.mode_importance[f'layer_{i}'] = importance.item()
        
        # Remove padding and project output
        if hasattr(self, 'pad_layer') and self.padding > 0:
            if len(self.modes) == 2:
                x = x[:, self.padding:-self.padding, self.padding:-self.padding, :]
            elif len(self.modes) == 3:
                x = x[:, self.padding:-self.padding, self.padding:-self.padding, 
                      self.padding:-self.padding, :]
        
        output = self.output_proj(x)
        return output
    
    def suggest_mode_adaptation(self) -> List[int]:
        """Suggest new modes based on importance scores."""
        avg_importance = np.mean(list(self.mode_importance.values()))
        
        if avg_importance < self.adaptive_threshold:
            # Reduce modes
            new_modes = [max(1, int(m * 0.8)) for m in self.modes]
        else:
            # Increase modes
            new_modes = [min(64, int(m * 1.2)) for m in self.modes]
        
        return new_modes


def create_fno_1d(input_dim: int = 1, output_dim: int = 1, 
                  modes: int = 16, **kwargs) -> FourierNeuralOperator:
    """Create 1D FNO for problems like Burgers equation."""
    return FourierNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        modes=[modes],
        **kwargs
    )


def create_fno_2d(input_dim: int = 3, output_dim: int = 1,
                  modes: Tuple[int, int] = (12, 12), **kwargs) -> FourierNeuralOperator:
    """Create 2D FNO for problems like Darcy flow."""
    return FourierNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        modes=list(modes),
        **kwargs
    )


def create_fno_3d(input_dim: int = 4, output_dim: int = 4,
                  modes: Tuple[int, int, int] = (8, 8, 8), **kwargs) -> FourierNeuralOperator:
    """Create 3D FNO for problems like Navier-Stokes."""
    return FourierNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        modes=list(modes),
        **kwargs
    )