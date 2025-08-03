"""Utility functions and helper classes for neural operators."""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize Fourier weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) / 
            (in_channels * out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution."""
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Multiply by weights in Fourier space
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), 
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", 
                                                 x_ft[:, :, :self.modes], 
                                                 self.weights)
        
        # Inverse FFT
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Initialize Fourier weights
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) / 
            (in_channels * out_channels)
        )
        self.weights2 = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) / 
            (in_channels * out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2D spectral convolution."""
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Multiply by weights in Fourier space
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-2), x_ft.size(-1),
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], 
            self.weights1
        )
        
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], 
            self.weights2
        )
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer for volumetric data."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes
        
        # Initialize Fourier weights for different mode combinations
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 
                       dtype=torch.cfloat) / (in_channels * out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D spectral convolution."""
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Multiply by weights in Fourier space
        out_ft = torch.zeros(batch_size, self.out_channels, 
                            x_ft.size(-3), x_ft.size(-2), x_ft.size(-1),
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz", 
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], 
            self.weights1
        )
        
        # Inverse FFT
        return torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=(-3, -2, -1))


class FourierFeatures(nn.Module):
    """Fourier feature encoding for positional information."""
    
    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.scale = scale
        
        # Random Fourier features
        self.B = nn.Parameter(
            torch.randn(num_features // 2, 3) * scale,
            requires_grad=False
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with Fourier features."""
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # Add batch dimension
        
        # Project coordinates
        proj = 2 * np.pi * coords @ self.B.T
        
        # Compute sine and cosine features
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        return features


class PDEResidualComputer:
    """Compute PDE residuals for physics-informed training."""
    
    @staticmethod
    def gradient(u: torch.Tensor, x: torch.Tensor, 
                order: int = 1) -> torch.Tensor:
        """Compute gradient using automatic differentiation."""
        u.requires_grad_(True)
        x.requires_grad_(True)
        
        grad = u
        for _ in range(order):
            grad = torch.autograd.grad(
                grad.sum(), x, 
                create_graph=True, 
                retain_graph=True
            )[0]
        
        return grad
    
    @staticmethod
    def divergence(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence of vector field."""
        batch_size, *spatial_dims, num_components = u.shape
        
        div = torch.zeros(batch_size, *spatial_dims, device=u.device)
        
        for i in range(num_components):
            if i < len(spatial_dims):
                grad_component = PDEResidualComputer.gradient(u[..., i], coords)
                div += grad_component[..., i]
        
        return div
    
    @staticmethod
    def laplacian(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using second derivatives."""
        batch_size, *spatial_dims = u.shape
        num_dims = len(spatial_dims)
        
        laplacian = torch.zeros_like(u)
        
        for i in range(num_dims):
            second_deriv = PDEResidualComputer.gradient(u, coords, order=2)
            if second_deriv.dim() > u.dim():
                # Select the i-th diagonal component
                laplacian += second_deriv[..., i, i]
            else:
                laplacian += second_deriv
        
        return laplacian


class ConservationLawChecker:
    """Check conservation laws in PDE solutions."""
    
    @staticmethod
    def energy_conservation(u: torch.Tensor, dt: float = 1.0) -> float:
        """Check energy conservation in solution."""
        energy = torch.sum(u**2, dim=(-2, -1))  # Spatial integration
        
        if energy.dim() > 1:  # Time series
            energy_change = torch.abs(energy[-1] - energy[0])
            initial_energy = torch.abs(energy[0])
            return (energy_change / (initial_energy + 1e-8)).item()
        
        return 0.0
    
    @staticmethod
    def mass_conservation(u: torch.Tensor, dt: float = 1.0) -> float:
        """Check mass conservation in solution."""
        mass = torch.sum(u, dim=(-2, -1))  # Spatial integration
        
        if mass.dim() > 1:  # Time series
            mass_change = torch.abs(mass[-1] - mass[0])
            initial_mass = torch.abs(mass[0])
            return (mass_change / (initial_mass + 1e-8)).item()
        
        return 0.0


class DataLoader:
    """Efficient data loader for large PDE datasets."""
    
    @staticmethod
    def load_hdf5(file_path: str, variables: List[str]) -> Dict[str, np.ndarray]:
        """Load data from HDF5 file."""
        data = {}
        
        with h5py.File(file_path, 'r') as f:
            for var in variables:
                if var in f:
                    data[var] = f[var][:]
                else:
                    logger.warning(f"Variable {var} not found in {file_path}")
        
        return data
    
    @staticmethod
    def save_hdf5(file_path: str, data: Dict[str, np.ndarray]):
        """Save data to HDF5 file."""
        with h5py.File(file_path, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression='gzip')
    
    @staticmethod
    def create_batches(data: torch.Tensor, batch_size: int, 
                      shuffle: bool = True) -> List[torch.Tensor]:
        """Create batches from data tensor."""
        num_samples = data.size(0)
        indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
        
        batches = []
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(data[batch_indices])
        
        return batches


class PerformanceProfiler:
    """Profile performance of neural operator training."""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                self.memory_usage[operation_name] = end_memory - start_memory
            
            end_time = time.time()
            self.timings[operation_name] = end_time - start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'timings': self.timings,
            'memory_usage': self.memory_usage
        }


def setup_device(device: Optional[str] = None) -> torch.device:
    """Setup computation device."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU")
    
    return device


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_file_safety(file_path: str, max_size: str = "1GB") -> bool:
    """Validate file safety for loading."""
    from .config import get_security_config
    
    config = get_security_config()
    
    if not config.validate_models:
        return True
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext not in config.allowed_file_types:
        raise ValueError(f"File type {ext} not allowed")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    max_bytes = parse_size_string(config.max_model_size)
    
    if file_size > max_bytes:
        raise ValueError(f"File size {file_size} exceeds limit {config.max_model_size}")
    
    return True


def parse_size_string(size_str: str) -> int:
    """Parse size string like '1GB' to bytes."""
    size_str = size_str.upper()
    
    if size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1e9)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1e6)
    elif size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1e3)
    else:
        return int(size_str)


def compute_spectral_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute spectral domain metrics."""
    # FFT of predictions and targets
    pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
    target_fft = torch.fft.fftn(target, dim=(-2, -1))
    
    # Power spectral density
    pred_psd = torch.abs(pred_fft)**2
    target_psd = torch.abs(target_fft)**2
    
    # Spectral error
    spectral_error = torch.mean((pred_psd - target_psd)**2) / torch.mean(target_psd**2)
    
    return {
        'spectral_mse': spectral_error.item(),
        'spectral_correlation': torch.corrcoef(
            torch.stack([pred_psd.flatten(), target_psd.flatten()])
        )[0, 1].item()
    }