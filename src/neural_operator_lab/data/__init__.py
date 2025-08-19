"""Data loading and preprocessing for PDE datasets."""

# Safe imports with fallbacks
__all__ = []

try:
    from .datasets import PDEDataset, TurbulenceDataset, DarcyFlowDataset
    __all__.extend(['PDEDataset', 'TurbulenceDataset', 'DarcyFlowDataset'])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import datasets: {e}")

try:
    from .loaders import create_pde_dataset, DataLoaderConfig
    __all__.extend(['create_pde_dataset', 'DataLoaderConfig'])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import loaders: {e}")

try:
    from .preprocessing import PDEPreprocessor, normalize_fields
    __all__.extend(['PDEPreprocessor', 'normalize_fields'])
except ImportError:
    pass

try:
    from .augmentation import PDEAugmentation
    __all__.append('PDEAugmentation')
except ImportError:
    pass

# Basic dataset creation function for simple cases
def create_synthetic_dataset(equation_type='darcy', resolution=64, n_samples=1000):
    """Create synthetic PDE dataset for testing."""
    import torch
    import numpy as np
    
    if equation_type == 'darcy':
        # Simple Darcy flow dataset
        x = torch.randn(n_samples, resolution, resolution, 1)  # Permeability
        # Simple solution approximation
        y = torch.tanh(x * 2.0)  # Pressure field
        return x, y
    
    elif equation_type == 'burgers':
        # 1D Burgers equation
        x = torch.randn(n_samples, resolution, 1)  # Initial condition
        # Simple propagation
        y = x * torch.exp(-0.1 * torch.arange(resolution).float().unsqueeze(-1))
        return x, y
    
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")

__all__.append('create_synthetic_dataset')