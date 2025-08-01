"""Test data generators for Neural Operator Foundation Lab tests."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import h5py
import json


class SyntheticPDEDataGenerator:
    """Generate synthetic PDE data for testing purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def generate_burgers_1d(
        self, 
        num_samples: int = 100,
        spatial_resolution: int = 256,
        time_steps: int = 50,
        viscosity: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Generate 1D Burgers' equation data."""
        x = torch.linspace(0, 2*np.pi, spatial_resolution)
        t = torch.linspace(0, 1, time_steps)
        
        inputs = []
        outputs = []
        
        for i in range(num_samples):
            # Random initial condition
            freq1, freq2 = torch.randint(1, 5, (2,))
            amp1, amp2 = torch.rand(2) * 0.5 + 0.5
            phase1, phase2 = torch.rand(2) * 2 * np.pi
            
            u0 = amp1 * torch.sin(freq1 * x + phase1) + amp2 * torch.cos(freq2 * x + phase2)
            
            # Simplified evolution (not exact Burgers solution)
            u_evolved = u0.clone()
            for _ in range(10):  # Simple diffusion steps
                # Diffusion term approximation
                u_laplace = torch.cat([
                    (u_evolved[1:2] - u_evolved[0:1]),
                    u_evolved[2:] - 2*u_evolved[1:-1] + u_evolved[:-2],
                    (u_evolved[-2:-1] - u_evolved[-1:])
                ])
                u_evolved = u_evolved + 0.01 * viscosity * u_laplace
            
            inputs.append(u0.unsqueeze(0))  # Add channel dimension
            outputs.append(u_evolved.unsqueeze(0))
        
        return {
            'inputs': torch.stack(inputs),
            'outputs': torch.stack(outputs),
            'grid': x.unsqueeze(0).expand(num_samples, -1),
            'metadata': {
                'pde_type': 'burgers_1d',
                'viscosity': viscosity,
                'spatial_resolution': spatial_resolution,
                'time_steps': time_steps
            }
        }
    
    def generate_darcy_flow_2d(
        self,
        num_samples: int = 100,
        resolution: int = 64,
        permeability_range: Tuple[float, float] = (0.1, 1.0)
    ) -> Dict[str, torch.Tensor]:
        """Generate 2D Darcy flow data."""
        # Create spatial grid
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        inputs = []
        outputs = []
        
        for i in range(num_samples):
            # Generate random permeability field
            # Use smooth random field
            k_field = torch.randn(resolution//4, resolution//4)
            k_field = torch.nn.functional.interpolate(
                k_field.unsqueeze(0).unsqueeze(0),
                size=(resolution, resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Scale to desired range
            k_min, k_max = permeability_range
            k_field = k_min + (k_max - k_min) * torch.sigmoid(k_field)
            
            # Generate forcing term (right-hand side)
            # Simple source/sink pattern
            center_x, center_y = torch.rand(2) * 0.6 + 0.2
            source_strength = torch.rand(1) * 2 - 1  # Random strength
            forcing = source_strength * torch.exp(
                -((X - center_x)**2 + (Y - center_y)**2) / 0.1
            )
            
            # Approximate solution using simple relaxation
            # (Not exact, but physically reasonable for testing)
            pressure = torch.zeros_like(X)
            for _ in range(100):  # Relaxation iterations
                pressure_new = pressure.clone()
                
                # Interior points (simplified finite difference)
                pressure_new[1:-1, 1:-1] = 0.25 * (
                    pressure[2:, 1:-1] + pressure[:-2, 1:-1] +
                    pressure[1:-1, 2:] + pressure[1:-1, :-2] -
                    forcing[1:-1, 1:-1] / k_field[1:-1, 1:-1]
                )
                
                pressure = 0.8 * pressure_new + 0.2 * pressure  # Damping
            
            # Stack input channels: permeability + forcing
            input_tensor = torch.stack([k_field, forcing])
            output_tensor = pressure.unsqueeze(0)  # Pressure field
            
            inputs.append(input_tensor)
            outputs.append(output_tensor)
        
        return {
            'inputs': torch.stack(inputs),
            'outputs': torch.stack(outputs),
            'grid': torch.stack([X, Y]).unsqueeze(0).expand(num_samples, -1, -1, -1),
            'metadata': {
                'pde_type': 'darcy_flow_2d',
                'resolution': resolution,
                'permeability_range': permeability_range
            }
        }
    
    def generate_wave_equation_2d(
        self,
        num_samples: int = 100,
        resolution: int = 64,
        wave_speed: float = 1.0,
        time_steps: int = 20
    ) -> Dict[str, torch.Tensor]:
        """Generate 2D wave equation data."""
        # Spatial grid
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        inputs = []
        outputs = []
        
        for i in range(num_samples):
            # Random initial displacement
            # Gaussian pulse at random location
            center_x = torch.rand(1) * 1.4 - 0.7  # [-0.7, 0.7]
            center_y = torch.rand(1) * 1.4 - 0.7
            width = torch.rand(1) * 0.2 + 0.1  # [0.1, 0.3]
            amplitude = torch.rand(1) * 0.8 + 0.2  # [0.2, 1.0]
            
            u0 = amplitude * torch.exp(
                -((X - center_x)**2 + (Y - center_y)**2) / width**2
            )
            
            # Initial velocity (zero for simplicity)
            v0 = torch.zeros_like(u0)
            
            # Simple wave evolution (not exact solution, but wave-like)
            u_current = u0.clone()
            u_previous = u0.clone()
            dt = 0.05
            dx = 2.0 / resolution
            
            for t in range(time_steps):
                # Wave equation finite difference (simplified)
                c_dt_dx = (wave_speed * dt / dx) ** 2
                
                u_new = torch.zeros_like(u_current)
                u_new[1:-1, 1:-1] = (
                    2 * u_current[1:-1, 1:-1] - u_previous[1:-1, 1:-1] +
                    c_dt_dx * (
                        u_current[2:, 1:-1] + u_current[:-2, 1:-1] +
                        u_current[1:-1, 2:] + u_current[1:-1, :-2] -
                        4 * u_current[1:-1, 1:-1]
                    )
                )
                
                u_previous = u_current.clone()
                u_current = u_new
            
            # Stack input channels: initial displacement + initial velocity
            input_tensor = torch.stack([u0, v0])
            output_tensor = u_current.unsqueeze(0)  # Final displacement
            
            inputs.append(input_tensor)
            outputs.append(output_tensor)
        
        return {
            'inputs': torch.stack(inputs),
            'outputs': torch.stack(outputs),
            'grid': torch.stack([X, Y]).unsqueeze(0).expand(num_samples, -1, -1, -1),
            'metadata': {
                'pde_type': 'wave_2d',
                'wave_speed': wave_speed,
                'time_steps': time_steps,
                'resolution': resolution
            }
        }
    
    def generate_navier_stokes_2d(
        self,
        num_samples: int = 50,
        resolution: int = 64,
        reynolds_number: float = 100.0,
        time_steps: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Generate simplified 2D Navier-Stokes data."""
        # This is a highly simplified version for testing
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        inputs = []
        outputs = []
        
        for i in range(num_samples):
            # Random initial velocity field
            # Create divergence-free initial condition (approximately)
            psi = torch.randn(resolution, resolution)
            psi = torch.nn.functional.conv2d(
                psi.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 5, 5) / 25,
                padding=2
            ).squeeze()
            
            # Velocity from stream function (ensures div-free)
            u = torch.gradient(psi, dim=0)[0]  # du/dy
            v = -torch.gradient(psi, dim=1)[0]  # -du/dx
            
            # Pressure field (random smooth field)
            p = torch.randn(resolution, resolution)
            p = torch.nn.functional.conv2d(
                p.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 3, 3) / 9,
                padding=1
            ).squeeze()
            
            # Simple evolution (not accurate NS, but maintains structure)
            nu = 1.0 / reynolds_number
            dt = 0.01
            
            for _ in range(time_steps):
                # Simplified diffusion step
                u_diff = torch.nn.functional.conv2d(
                    u.unsqueeze(0).unsqueeze(0),
                    torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]) * dt * nu,
                    padding=1
                ).squeeze()
                v_diff = torch.nn.functional.conv2d(
                    v.unsqueeze(0).unsqueeze(0),
                    torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]) * dt * nu,
                    padding=1
                ).squeeze()
                
                u = u + u_diff
                v = v + v_diff
            
            # Stack input channels: u, v, p
            input_tensor = torch.stack([u, v, p])
            
            # For output, return evolved velocity field
            output_tensor = torch.stack([u, v])
            
            inputs.append(input_tensor)
            outputs.append(output_tensor)
        
        return {
            'inputs': torch.stack(inputs),
            'outputs': torch.stack(outputs),
            'grid': torch.stack([X, Y]).unsqueeze(0).expand(num_samples, -1, -1, -1),
            'metadata': {
                'pde_type': 'navier_stokes_2d',
                'reynolds_number': reynolds_number,
                'time_steps': time_steps,
                'resolution': resolution
            }
        }


class TestDatasetCreator:
    """Create complete test datasets with proper structure."""
    
    def __init__(self, output_dir: Path, generator: Optional[SyntheticPDEDataGenerator] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator or SyntheticPDEDataGenerator()
    
    def create_dataset(
        self,
        pde_type: str,
        num_samples: int = 1000,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        **kwargs
    ) -> Dict[str, Path]:
        """Create a complete dataset with train/val/test splits."""
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Generate data based on PDE type
        if pde_type == 'burgers_1d':
            data = self.generator.generate_burgers_1d(num_samples, **kwargs)
        elif pde_type == 'darcy_flow_2d':
            data = self.generator.generate_darcy_flow_2d(num_samples, **kwargs)
        elif pde_type == 'wave_2d':
            data = self.generator.generate_wave_equation_2d(num_samples, **kwargs)
        elif pde_type == 'navier_stokes_2d':
            data = self.generator.generate_navier_stokes_2d(num_samples, **kwargs)
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")
        
        # Calculate split indices
        train_size = int(num_samples * train_split)
        val_size = int(num_samples * val_split)
        test_size = num_samples - train_size - val_size
        
        # Split data
        indices = torch.randperm(num_samples)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        # Save data splits
        dataset_dir = self.output_dir / f"{pde_type}_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for split_name, split_idx in splits.items():
            split_data = {
                'inputs': data['inputs'][split_idx],
                'outputs': data['outputs'][split_idx],
                'grid': data['grid'][split_idx] if len(data['grid'].shape) > 2 else data['grid'],
            }
            
            # Save as HDF5
            h5_path = dataset_dir / f"{split_name}.h5"
            with h5py.File(h5_path, 'w') as f:
                for key, tensor in split_data.items():
                    f.create_dataset(key, data=tensor.numpy())
            
            saved_files[split_name] = h5_path
        
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            metadata = data['metadata'].copy()
            metadata.update({
                'num_samples': num_samples,
                'splits': {
                    'train': train_size,
                    'val': val_size,  
                    'test': test_size
                },
                'created_by': 'TestDatasetCreator'
            })
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = metadata_path
        
        return saved_files
    
    def create_multi_pde_dataset(
        self,
        pde_configs: List[Dict[str, Any]],
        samples_per_pde: int = 200
    ) -> Dict[str, Dict[str, Path]]:
        """Create datasets for multiple PDE types."""
        all_datasets = {}
        
        for config in pde_configs:
            pde_type = config.pop('pde_type')
            dataset_files = self.create_dataset(
                pde_type, 
                num_samples=samples_per_pde,
                **config
            )
            all_datasets[pde_type] = dataset_files
        
        return all_datasets


class TestDataValidator:
    """Validate test data for correctness and consistency."""
    
    @staticmethod 
    def validate_tensor_data(data: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """Validate tensor data structure and properties."""
        results = {}
        
        # Check required keys
        required_keys = ['inputs', 'outputs']
        for key in required_keys:
            results[f'has_{key}'] = key in data
        
        if not all(results[f'has_{key}'] for key in required_keys):
            return results
        
        inputs = data['inputs']
        outputs = data['outputs']
        
        # Check shapes
        results['same_batch_size'] = inputs.shape[0] == outputs.shape[0]
        results['valid_input_dims'] = len(inputs.shape) >= 3  # At least [B, C, ...]
        results['valid_output_dims'] = len(outputs.shape) >= 3
        
        # Check for NaN/Inf
        results['inputs_finite'] = torch.isfinite(inputs).all().item()
        results['outputs_finite'] = torch.isfinite(outputs).all().item()
        
        # Check value ranges (should be reasonable)
        results['inputs_reasonable'] = (inputs.abs().max() < 1000).item()
        results['outputs_reasonable'] = (outputs.abs().max() < 1000).item()
        
        # Check grid if present
        if 'grid' in data:
            grid = data['grid']
            results['has_grid'] = True
            results['grid_finite'] = torch.isfinite(grid).all().item()
        else:
            results['has_grid'] = False
        
        return results
    
    @staticmethod
    def validate_dataset_files(dataset_dir: Path) -> Dict[str, bool]:
        """Validate dataset file structure."""
        results = {}
        
        # Check directory exists
        results['directory_exists'] = dataset_dir.exists()
        if not results['directory_exists']:
            return results
        
        # Check for required files
        required_files = ['train.h5', 'val.h5', 'test.h5', 'metadata.json']
        for filename in required_files:
            file_path = dataset_dir / filename
            results[f'has_{filename}'] = file_path.exists()
        
        # Validate HDF5 files
        for split in ['train', 'val', 'test']:
            h5_path = dataset_dir / f"{split}.h5"
            if h5_path.exists():
                try:
                    with h5py.File(h5_path, 'r') as f:
                        required_datasets = ['inputs', 'outputs']
                        for dataset_name in required_datasets:
                            results[f'{split}_has_{dataset_name}'] = dataset_name in f
                        
                        # Check shapes are consistent
                        if all(dataset_name in f for dataset_name in required_datasets):
                            inputs_shape = f['inputs'].shape
                            outputs_shape = f['outputs'].shape
                            results[f'{split}_consistent_batch'] = inputs_shape[0] == outputs_shape[0]
                        
                except Exception as e:
                    results[f'{split}_valid_h5'] = False
                    results[f'{split}_error'] = str(e)
                else:
                    results[f'{split}_valid_h5'] = True
        
        # Validate metadata
        metadata_path = dataset_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                required_metadata = ['pde_type', 'num_samples', 'splits']
                for key in required_metadata:
                    results[f'metadata_has_{key}'] = key in metadata
                
            except Exception as e:
                results['metadata_valid'] = False
                results['metadata_error'] = str(e)
            else:
                results['metadata_valid'] = True
        
        return results