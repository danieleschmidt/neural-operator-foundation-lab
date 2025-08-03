"""PDE dataset implementations for different physics problems."""

import torch
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import glob

from ..base import PDEDatasetBase, create_coordinate_grid
from ..utils import validate_file_safety


class PDEDataset(PDEDatasetBase):
    """General PDE dataset for loading simulation data."""
    
    def __init__(self, 
                 data_path: str,
                 resolution: int,
                 variables: List[str],
                 time_steps: Optional[int] = None,
                 train_split: float = 0.8,
                 normalize: bool = True,
                 subsample_space: int = 1,
                 subsample_time: int = 1,
                 **kwargs):
        """Initialize PDE dataset.
        
        Args:
            data_path: Path to data files (supports glob patterns)
            resolution: Spatial resolution
            variables: List of variable names to load
            time_steps: Number of time steps to use (None for all)
            train_split: Fraction of data for training
            normalize: Whether to normalize the data
            subsample_space: Spatial subsampling factor
            subsample_time: Temporal subsampling factor
        """
        super().__init__(data_path, resolution, variables, **kwargs)
        
        self.time_steps = time_steps
        self.train_split = train_split
        self.normalize = normalize
        self.subsample_space = subsample_space
        self.subsample_time = subsample_time
        
        # Load and process data
        self.data_files = self._find_data_files()
        self.input_data, self.output_data = self.load_data()
        self.grid = self.get_grid()
        
        # Normalization statistics
        self.norm_stats = {}
        if self.normalize:
            self.input_data, self.norm_stats['input'] = self.normalize_data(self.input_data)
            self.output_data, self.norm_stats['output'] = self.normalize_data(self.output_data)
    
    def _find_data_files(self) -> List[str]:
        """Find all data files matching the pattern."""
        if os.path.isfile(self.data_path):
            return [self.data_path]
        
        # Support glob patterns
        files = glob.glob(self.data_path)
        if not files:
            # Try common extensions
            for ext in ['.h5', '.hdf5', '.nc', '.npz']:
                pattern = self.data_path + ext if not self.data_path.endswith(ext) else self.data_path
                files = glob.glob(pattern)
                if files:
                    break
        
        if not files:
            raise FileNotFoundError(f"No data files found matching: {self.data_path}")
        
        # Validate files
        for file_path in files:
            validate_file_safety(file_path)
        
        return sorted(files)
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load PDE data from files."""
        all_inputs = []
        all_outputs = []
        
        for file_path in self.data_files:
            try:
                file_data = self._load_single_file(file_path)
                
                # Process variables
                input_fields = []
                output_fields = []
                
                for var in self.variables:
                    if var in file_data:
                        field = file_data[var]
                        
                        # Apply subsampling
                        if self.subsample_space > 1:
                            field = field[..., ::self.subsample_space, ::self.subsample_space]
                        
                        if field.ndim > 3 and self.subsample_time > 1:
                            field = field[::self.subsample_time]
                        
                        # Limit time steps if specified
                        if self.time_steps is not None and field.ndim > 3:
                            field = field[:self.time_steps]
                        
                        input_fields.append(field)
                        
                        # For now, use next time step as output (can be customized)
                        if field.ndim > 3 and field.shape[0] > 1:
                            output_fields.append(field[1:])
                        else:
                            output_fields.append(field)
                
                # Stack variables
                if input_fields:
                    inputs = np.stack(input_fields, axis=-1)
                    outputs = np.stack(output_fields, axis=-1)
                    
                    all_inputs.append(inputs)
                    all_outputs.append(outputs)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid data could be loaded")
        
        # Concatenate all data
        input_data = np.concatenate(all_inputs, axis=0)
        output_data = np.concatenate(all_outputs, axis=0)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        return input_tensor, output_tensor
    
    def _load_single_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from a single file."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.h5', '.hdf5']:
            return self._load_hdf5(file_path)
        elif file_ext == '.npz':
            return self._load_npz(file_path)
        elif file_ext == '.nc':
            return self._load_netcdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_hdf5(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from HDF5 file."""
        data = {}
        with h5py.File(file_path, 'r') as f:
            for var in self.variables:
                if var in f:
                    data[var] = f[var][:]
                else:
                    # Try common naming conventions
                    for possible_name in [var.lower(), var.upper(), f'{var}_field']:
                        if possible_name in f:
                            data[var] = f[possible_name][:]
                            break
        return data
    
    def _load_npz(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from NPZ file."""
        data = {}
        with np.load(file_path) as f:
            for var in self.variables:
                if var in f:
                    data[var] = f[var]
        return data
    
    def _load_netcdf(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from NetCDF file."""
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 package required for .nc files")
        
        data = {}
        with nc.Dataset(file_path, 'r') as f:
            for var in self.variables:
                if var in f.variables:
                    data[var] = f.variables[var][:]
        return data
    
    def get_grid(self) -> torch.Tensor:
        """Generate coordinate grid for the dataset."""
        if self.input_data.dim() >= 3:
            spatial_shape = self.input_data.shape[-3:-1]  # Exclude channel dimension
            return create_coordinate_grid(
                spatial_shape,
                domain=(0.0, 1.0),
                device=self.input_data.device
            )
        else:
            # 1D case
            grid_size = self.input_data.shape[-2]
            return torch.linspace(0, 1, grid_size, device=self.input_data.device).unsqueeze(-1)
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        
        # Return input, output, and grid
        return input_sample, output_sample, self.grid
    
    def get_train_test_split(self) -> Tuple['PDEDataset', 'PDEDataset']:
        """Split dataset into train and test sets."""
        num_samples = len(self)
        num_train = int(num_samples * self.train_split)
        
        # Create indices
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        # Create subset datasets
        train_dataset = SubsetPDEDataset(self, train_indices)
        test_dataset = SubsetPDEDataset(self, test_indices)
        
        return train_dataset, test_dataset


class SubsetPDEDataset(torch.utils.data.Dataset):
    """Subset of a PDE dataset."""
    
    def __init__(self, dataset: PDEDataset, indices: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[idx]]


class TurbulenceDataset(PDEDataset):
    """Dataset for turbulent flow simulations."""
    
    def __init__(self, data_path: str, resolution: int = 256, **kwargs):
        """Initialize turbulence dataset.
        
        Args:
            data_path: Path to turbulence simulation data
            resolution: Spatial resolution
        """
        # Default variables for turbulence
        default_variables = ['velocity_x', 'velocity_y', 'velocity_z', 'pressure']
        variables = kwargs.pop('variables', default_variables)
        
        super().__init__(
            data_path=data_path,
            resolution=resolution,
            variables=variables,
            **kwargs
        )
    
    def compute_vorticity(self) -> torch.Tensor:
        """Compute vorticity from velocity field."""
        if self.input_data.size(-1) >= 3:  # 3D velocity
            u = self.input_data[..., 0]  # x-component
            v = self.input_data[..., 1]  # y-component
            w = self.input_data[..., 2]  # z-component
            
            # Compute curl (simplified finite difference)
            # This is a basic implementation - could be improved
            vorticity_x = torch.diff(w, dim=-2) - torch.diff(v, dim=-1)
            vorticity_y = torch.diff(u, dim=-1) - torch.diff(w, dim=-3)
            vorticity_z = torch.diff(v, dim=-3) - torch.diff(u, dim=-2)
            
            # Handle dimension mismatch with padding
            min_shape = [min(vort.shape[i] for vort in [vorticity_x, vorticity_y, vorticity_z]) 
                        for i in range(len(vorticity_x.shape))]
            
            vorticity_x = vorticity_x[..., :min_shape[-3], :min_shape[-2], :min_shape[-1]]
            vorticity_y = vorticity_y[..., :min_shape[-3], :min_shape[-2], :min_shape[-1]]
            vorticity_z = vorticity_z[..., :min_shape[-3], :min_shape[-2], :min_shape[-1]]
            
            vorticity = torch.stack([vorticity_x, vorticity_y, vorticity_z], dim=-1)
            return vorticity
        
        return torch.zeros_like(self.input_data[..., :3])
    
    def compute_energy_spectrum(self) -> torch.Tensor:
        """Compute energy spectrum of turbulent flow."""
        if self.input_data.size(-1) >= 3:
            # Velocity field
            velocity = self.input_data[..., :3]
            
            # FFT of velocity field
            velocity_fft = torch.fft.fftn(velocity, dim=(-3, -2, -1))
            
            # Energy spectrum
            energy = torch.sum(torch.abs(velocity_fft)**2, dim=-1)
            
            return energy
        
        return torch.zeros(self.input_data.shape[:-1])


class DarcyFlowDataset(PDEDataset):
    """Dataset for Darcy flow problems."""
    
    def __init__(self, data_path: str, resolution: int = 85, **kwargs):
        """Initialize Darcy flow dataset.
        
        Args:
            data_path: Path to Darcy flow data
            resolution: Spatial resolution (typically 85x85)
        """
        # Default variables for Darcy flow
        default_variables = ['permeability', 'pressure']
        variables = kwargs.pop('variables', default_variables)
        
        super().__init__(
            data_path=data_path,
            resolution=resolution,
            variables=variables,
            **kwargs
        )
    
    def get_permeability_statistics(self) -> Dict[str, float]:
        """Get statistics of permeability field."""
        if 'permeability' in self.variables:
            perm_idx = self.variables.index('permeability')
            permeability = self.input_data[..., perm_idx]
            
            return {
                'mean': permeability.mean().item(),
                'std': permeability.std().item(),
                'min': permeability.min().item(),
                'max': permeability.max().item(),
                'contrast_ratio': (permeability.max() / permeability.min()).item()
            }
        
        return {}


class SyntheticPDEDataset(PDEDataset):
    """Generate synthetic PDE data for testing."""
    
    def __init__(self, 
                 pde_type: str,
                 num_samples: int = 1000,
                 resolution: int = 64,
                 **kwargs):
        """Generate synthetic PDE dataset.
        
        Args:
            pde_type: Type of PDE ('burgers', 'wave', 'heat')
            num_samples: Number of samples to generate
            resolution: Spatial resolution
        """
        self.pde_type = pde_type
        self.num_samples = num_samples
        
        # Generate synthetic data
        input_data, output_data = self._generate_synthetic_data()
        
        # Initialize without calling parent __init__
        self.data_path = f"synthetic_{pde_type}"
        self.resolution = resolution
        self.variables = ['field']
        self.config = kwargs
        
        self.input_data = input_data
        self.output_data = output_data
        self.grid = self.get_grid()
        self.norm_stats = {}
    
    def _generate_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic PDE data."""
        if self.pde_type == 'burgers':
            return self._generate_burgers_data()
        elif self.pde_type == 'wave':
            return self._generate_wave_data()
        elif self.pde_type == 'heat':
            return self._generate_heat_data()
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
    
    def _generate_burgers_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Burgers equation data."""
        x = torch.linspace(0, 1, self.resolution)
        
        inputs = []
        outputs = []
        
        for _ in range(self.num_samples):
            # Random initial condition
            amplitude = torch.rand(1) * 2 + 0.5
            frequency = torch.randint(1, 5, (1,)).float()
            phase = torch.rand(1) * 2 * np.pi
            
            u0 = amplitude * torch.sin(frequency * 2 * np.pi * x + phase)
            
            # Simple evolution (not exact solution)
            dt = 0.01
            u1 = u0 - dt * u0 * torch.gradient(u0, spacing=1.0/self.resolution)[0]
            
            inputs.append(u0.unsqueeze(-1))
            outputs.append(u1.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def _generate_wave_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate wave equation data."""
        x = torch.linspace(0, 1, self.resolution)
        
        inputs = []
        outputs = []
        
        for _ in range(self.num_samples):
            # Random wave parameters
            amplitude = torch.rand(1) * 2 + 0.5
            frequency = torch.randint(1, 5, (1,)).float()
            velocity = torch.rand(1) * 2 + 0.5
            
            # Initial condition
            u0 = amplitude * torch.sin(frequency * 2 * np.pi * x)
            
            # Wave propagation
            dt = 0.01
            u1 = amplitude * torch.sin(frequency * 2 * np.pi * (x - velocity * dt))
            
            inputs.append(u0.unsqueeze(-1))
            outputs.append(u1.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def _generate_heat_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate heat equation data."""
        x = torch.linspace(0, 1, self.resolution)
        
        inputs = []
        outputs = []
        
        for _ in range(self.num_samples):
            # Random initial condition
            amplitude = torch.rand(1) * 2 + 0.5
            frequency = torch.randint(1, 5, (1,)).float()
            
            u0 = amplitude * torch.sin(frequency * 2 * np.pi * x)
            
            # Heat diffusion (simplified)
            dt = 0.01
            diffusivity = 0.1
            dx = 1.0 / self.resolution
            
            # Simple finite difference
            u1 = u0.clone()
            u1[1:-1] = u0[1:-1] + diffusivity * dt / dx**2 * (u0[2:] - 2*u0[1:-1] + u0[:-2])
            
            inputs.append(u0.unsqueeze(-1))
            outputs.append(u1.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pre-generated data."""
        return self.input_data, self.output_data