"""Base classes for neural operators and datasets."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class NeuralOperatorBase(nn.Module, ABC):
    """Abstract base class for all neural operators."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = kwargs
        
    @abstractmethod
    def forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from input function."""
        pass
    
    @abstractmethod
    def forward_operator(self, features: torch.Tensor) -> torch.Tensor:
        """Apply neural operator transformation."""
        pass
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the neural operator."""
        features = self.forward_features(x, grid)
        return self.forward_operator(features)
    
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                 inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss with optional physics constraints."""
        data_loss = nn.functional.mse_loss(pred, target)
        
        # Add physics-informed loss if implemented
        if hasattr(self, 'compute_pde_residual') and inputs is not None:
            physics_loss = self.compute_pde_residual(pred, inputs)
            return data_loss + 0.1 * physics_loss
        
        return data_loss
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary statistics."""
        return {
            'parameters': self.count_parameters(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'model_type': self.__class__.__name__
        }


class PDEDatasetBase(torch.utils.data.Dataset, ABC):
    """Abstract base class for PDE datasets."""
    
    def __init__(self, data_path: str, resolution: int, variables: list, **kwargs):
        self.data_path = data_path
        self.resolution = resolution
        self.variables = variables
        self.config = kwargs
        self._validate_config()
        
    @abstractmethod
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return input/output data tensors."""
        pass
    
    @abstractmethod
    def get_grid(self) -> torch.Tensor:
        """Generate coordinate grid for the problem domain."""
        pass
    
    def _validate_config(self):
        """Validate dataset configuration."""
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if not self.variables:
            raise ValueError("Variables list cannot be empty")
    
    def normalize_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Normalize data and return normalization statistics."""
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        
        stats = {
            'mean': mean.item() if mean.numel() == 1 else mean.tolist(),
            'std': std.item() if std.numel() == 1 else std.tolist()
        }
        
        return normalized, stats
    
    def denormalize_data(self, normalized_data: torch.Tensor, 
                        stats: Dict[str, float]) -> torch.Tensor:
        """Denormalize data using provided statistics."""
        mean = torch.tensor(stats['mean'])
        std = torch.tensor(stats['std'])
        return normalized_data * std + mean


class TrainingMetrics:
    """Container for training metrics and statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.train_loss = []
        self.val_loss = []
        self.physics_loss = []
        self.convergence_rate = []
        self.epoch_times = []
        
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key).append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        summary = {}
        for attr in ['train_loss', 'val_loss', 'physics_loss']:
            values = getattr(self, attr)
            if values:
                summary[attr] = {
                    'current': values[-1],
                    'best': min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return summary


class ModelValidator:
    """Validator for neural operator models and inputs."""
    
    @staticmethod
    def validate_model_inputs(x: torch.Tensor, expected_dim: int):
        """Validate model input tensors."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if x.dim() < 2:
            raise ValueError("Input must have at least 2 dimensions")
        
        if x.size(-1) != expected_dim:
            raise ValueError(f"Expected input dimension {expected_dim}, got {x.size(-1)}")
    
    @staticmethod
    def validate_grid(grid: torch.Tensor, batch_size: int, spatial_dims: int):
        """Validate coordinate grid tensor."""
        if grid is not None:
            expected_shape = (batch_size, -1, spatial_dims)
            if grid.dim() != 3 or grid.size(0) != batch_size or grid.size(-1) != spatial_dims:
                raise ValueError(f"Grid shape mismatch. Expected {expected_shape}, got {grid.shape}")
    
    @staticmethod
    def validate_pde_output(output: torch.Tensor, target: torch.Tensor):
        """Validate PDE solver output against target."""
        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} doesn't match target {target.shape}")
        
        if torch.isnan(output).any():
            raise ValueError("Model output contains NaN values")
        
        if torch.isinf(output).any():
            raise ValueError("Model output contains infinite values")


def create_coordinate_grid(resolution: Union[int, Tuple[int, ...]], 
                          domain: Tuple[float, float] = (0.0, 1.0),
                          device: str = 'cpu') -> torch.Tensor:
    """Create coordinate grid for neural operators.
    
    Args:
        resolution: Grid resolution (int for 1D, tuple for multi-D)
        domain: Domain bounds (start, end)
        device: Device to place tensor on
        
    Returns:
        Coordinate grid tensor of shape (*resolution, spatial_dims)
    """
    if isinstance(resolution, int):
        resolution = (resolution,)
    
    spatial_dims = len(resolution)
    coordinates = []
    
    for i, res in enumerate(resolution):
        coord = torch.linspace(domain[0], domain[1], res, device=device)
        coordinates.append(coord)
    
    # Create meshgrid
    grid = torch.meshgrid(*coordinates, indexing='ij')
    
    # Stack and reshape
    grid = torch.stack(grid, dim=-1)
    
    return grid