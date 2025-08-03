"""Data loading utilities and configuration."""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import multiprocessing

from .datasets import PDEDataset, TurbulenceDataset, DarcyFlowDataset, SyntheticPDEDataset


@dataclass
class DataLoaderConfig:
    """Configuration for PDE data loaders."""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    def __post_init__(self):
        # Adjust num_workers based on available CPUs
        max_workers = multiprocessing.cpu_count()
        self.num_workers = min(self.num_workers, max_workers)
        
        # Disable persistent workers if num_workers is 0
        if self.num_workers == 0:
            self.persistent_workers = False


def create_pde_dataset(
    data_path: str = None,
    pde_type: str = 'custom',
    resolution: int = 64,
    variables: List[str] = None,
    normalize: bool = True,
    train_split: float = 0.8,
    subsample_space: int = 1,
    subsample_time: int = 1,
    synthetic_samples: int = 1000,
    **kwargs
) -> Union[PDEDataset, TurbulenceDataset, DarcyFlowDataset]:
    """Create PDE dataset based on type and parameters.
    
    Args:
        data_path: Path to data files (None for synthetic)
        pde_type: Type of PDE dataset
        resolution: Spatial resolution
        variables: List of variables to load
        normalize: Whether to normalize data
        train_split: Training data fraction
        subsample_space: Spatial subsampling factor
        subsample_time: Temporal subsampling factor
        synthetic_samples: Number of synthetic samples
        
    Returns:
        Configured PDE dataset
    """
    
    common_args = {
        'resolution': resolution,
        'normalize': normalize,
        'train_split': train_split,
        'subsample_space': subsample_space,
        'subsample_time': subsample_time,
        **kwargs
    }
    
    if data_path is None:
        # Create synthetic dataset
        if pde_type in ['burgers', 'wave', 'heat']:
            return SyntheticPDEDataset(
                pde_type=pde_type,
                num_samples=synthetic_samples,
                **common_args
            )
        else:
            raise ValueError(f"Synthetic data not available for PDE type: {pde_type}")
    
    # Add data path to common args
    common_args['data_path'] = data_path
    
    if variables is not None:
        common_args['variables'] = variables
    
    # Create specific dataset types
    if pde_type == 'turbulence' or pde_type == 'navier_stokes':
        return TurbulenceDataset(**common_args)
    
    elif pde_type == 'darcy' or pde_type == 'darcy_flow':
        return DarcyFlowDataset(**common_args)
    
    elif pde_type == 'custom':
        if variables is None:
            raise ValueError("Variables must be specified for custom PDE dataset")
        return PDEDataset(**common_args)
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def create_data_loaders(
    dataset: PDEDataset,
    config: DataLoaderConfig = None,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        dataset: PDE dataset
        config: Data loader configuration
        val_split: Validation split fraction
        test_split: Test split fraction
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = DataLoaderConfig()
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffling for test
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2
    )
    
    return train_loader, val_loader, test_loader


def create_streaming_loader(
    data_path: str,
    pde_type: str,
    batch_size: int = 32,
    chunk_size: int = 1000,
    **kwargs
) -> 'StreamingDataLoader':
    """Create streaming data loader for very large datasets.
    
    Args:
        data_path: Path to data files
        pde_type: Type of PDE dataset
        batch_size: Batch size
        chunk_size: Number of samples to load at once
        
    Returns:
        Streaming data loader
    """
    return StreamingDataLoader(
        data_path=data_path,
        pde_type=pde_type,
        batch_size=batch_size,
        chunk_size=chunk_size,
        **kwargs
    )


class StreamingDataLoader:
    """Streaming data loader for large PDE datasets that don't fit in memory."""
    
    def __init__(self,
                 data_path: str,
                 pde_type: str,
                 batch_size: int = 32,
                 chunk_size: int = 1000,
                 **kwargs):
        """Initialize streaming loader.
        
        Args:
            data_path: Path to data files
            pde_type: Type of PDE dataset
            batch_size: Batch size
            chunk_size: Number of samples to load at once
        """
        self.data_path = data_path
        self.pde_type = pde_type
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.kwargs = kwargs
        
        # Get dataset metadata without loading data
        self._get_metadata()
        
        self.current_chunk = 0
        self.current_batch = 0
        self.current_data = None
    
    def _get_metadata(self):
        """Get dataset metadata."""
        # Create a small dataset to get metadata
        small_dataset = create_pde_dataset(
            data_path=self.data_path,
            pde_type=self.pde_type,
            **{k: v for k, v in self.kwargs.items() if k != 'num_samples'}
        )
        
        self.resolution = small_dataset.resolution
        self.variables = small_dataset.variables
        self.input_shape = small_dataset.input_data.shape[1:]
        self.output_shape = small_dataset.output_data.shape[1:]
        self.total_samples = len(small_dataset)
        self.num_chunks = (self.total_samples + self.chunk_size - 1) // self.chunk_size
        
        del small_dataset  # Free memory
    
    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of data."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        # Load chunk data (this is simplified - would need actual chunked loading)
        dataset = create_pde_dataset(
            data_path=self.data_path,
            pde_type=self.pde_type,
            **self.kwargs
        )
        
        # Extract chunk
        self.current_data = {
            'input': dataset.input_data[start_idx:end_idx],
            'output': dataset.output_data[start_idx:end_idx],
            'grid': dataset.grid
        }
        
        del dataset  # Free memory
    
    def __iter__(self):
        """Iterate over batches."""
        self.current_chunk = 0
        self.current_batch = 0
        return self
    
    def __next__(self):
        """Get next batch."""
        # Load new chunk if needed
        if self.current_data is None or self.current_batch >= len(self.current_data['input']) // self.batch_size:
            if self.current_chunk >= self.num_chunks:
                raise StopIteration
            
            self._load_chunk(self.current_chunk)
            self.current_chunk += 1
            self.current_batch = 0
        
        # Get batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.current_data['input']))
        
        batch = (
            self.current_data['input'][start_idx:end_idx],
            self.current_data['output'][start_idx:end_idx],
            self.current_data['grid']
        )
        
        self.current_batch += 1
        
        return batch
    
    def __len__(self):
        """Get number of batches."""
        return (self.total_samples + self.batch_size - 1) // self.batch_size


class MultiResolutionDataLoader:
    """Data loader that provides multiple resolutions for progressive training."""
    
    def __init__(self,
                 base_dataset: PDEDataset,
                 resolutions: List[int],
                 batch_size: int = 32):
        """Initialize multi-resolution loader.
        
        Args:
            base_dataset: Base high-resolution dataset
            resolutions: List of resolutions to provide
            batch_size: Batch size
        """
        self.base_dataset = base_dataset
        self.resolutions = sorted(resolutions)
        self.batch_size = batch_size
        self.current_resolution_idx = 0
    
    def get_current_loader(self) -> DataLoader:
        """Get data loader for current resolution."""
        current_res = self.resolutions[self.current_resolution_idx]
        
        # Downsample dataset to current resolution
        downsampled_dataset = self._downsample_dataset(current_res)
        
        return DataLoader(
            downsampled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def _downsample_dataset(self, target_resolution: int) -> PDEDataset:
        """Downsample dataset to target resolution."""
        # This is a simplified implementation
        # In practice, you'd want proper interpolation
        
        original_res = self.base_dataset.resolution
        if target_resolution >= original_res:
            return self.base_dataset
        
        # Calculate downsampling factor
        factor = original_res // target_resolution
        
        # Downsample data
        downsampled_input = self.base_dataset.input_data[..., ::factor, ::factor, :]
        downsampled_output = self.base_dataset.output_data[..., ::factor, ::factor, :]
        
        # Create new dataset
        downsampled_dataset = PDEDataset.__new__(PDEDataset)
        downsampled_dataset.input_data = downsampled_input
        downsampled_dataset.output_data = downsampled_output
        downsampled_dataset.resolution = target_resolution
        downsampled_dataset.variables = self.base_dataset.variables
        downsampled_dataset.grid = self.base_dataset.get_grid()
        
        return downsampled_dataset
    
    def step_resolution(self):
        """Move to next resolution."""
        if self.current_resolution_idx < len(self.resolutions) - 1:
            self.current_resolution_idx += 1
    
    def get_current_resolution(self) -> int:
        """Get current resolution."""
        return self.resolutions[self.current_resolution_idx]


def collate_pde_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for PDE data batches."""
    inputs, outputs, grids = zip(*batch)
    
    # Stack inputs and outputs
    batched_inputs = torch.stack(inputs, dim=0)
    batched_outputs = torch.stack(outputs, dim=0)
    
    # Grid is typically the same for all samples
    batched_grid = grids[0]
    
    return batched_inputs, batched_outputs, batched_grid


def get_dataset_statistics(dataset: PDEDataset) -> Dict[str, Any]:
    """Compute comprehensive statistics for a PDE dataset."""
    stats = {
        'num_samples': len(dataset),
        'resolution': dataset.resolution,
        'variables': dataset.variables,
        'input_shape': list(dataset.input_data.shape),
        'output_shape': list(dataset.output_data.shape),
    }
    
    # Per-variable statistics
    for i, var in enumerate(dataset.variables):
        if i < dataset.input_data.size(-1):
            var_data = dataset.input_data[..., i]
            stats[f'{var}_stats'] = {
                'mean': var_data.mean().item(),
                'std': var_data.std().item(),
                'min': var_data.min().item(),
                'max': var_data.max().item(),
                'shape': list(var_data.shape)
            }
    
    # Memory usage
    input_memory = dataset.input_data.element_size() * dataset.input_data.nelement()
    output_memory = dataset.output_data.element_size() * dataset.output_data.nelement()
    stats['memory_usage_mb'] = (input_memory + output_memory) / (1024 * 1024)
    
    return stats