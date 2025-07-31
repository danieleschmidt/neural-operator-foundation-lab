"""Pytest configuration and shared fixtures for Neural Operator Foundation Lab."""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any

# Test configuration
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def sample_pde_data() -> Dict[str, torch.Tensor]:
    """Generate sample PDE data for testing.
    
    Returns:
        Dictionary containing sample input and target tensors
    """
    batch_size, channels, height, width = 2, 3, 32, 32
    
    return {
        "input": torch.randn(batch_size, channels, height, width),
        "target": torch.randn(batch_size, channels, height, width),
        "grid": torch.randn(batch_size, 2, height, width)  # x, y coordinates
    }


@pytest.fixture
def sample_3d_data() -> Dict[str, torch.Tensor]:
    """Generate sample 3D PDE data for testing.
    
    Returns:
        Dictionary containing sample 3D tensors
    """
    batch_size, channels, depth, height, width = 1, 4, 16, 32, 32
    
    return {
        "input": torch.randn(batch_size, channels, depth, height, width),
        "target": torch.randn(batch_size, channels, depth, height, width),
        "grid": torch.randn(batch_size, 3, depth, height, width)  # x, y, z coordinates
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Generate sample configuration for testing.
    
    Returns:
        Dictionary containing test configuration
    """
    return {
        "model": {
            "type": "transformer",
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "epochs": 5
        },
        "data": {
            "resolution": 32,
            "variables": ["u", "v", "p"]
        }
    }


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """Create temporary path for model saving/loading tests.
    
    Args:
        tmp_path: Pytest temporary directory fixture
        
    Returns:
        Path to temporary model file
    """
    return tmp_path / "test_model.pt"


@pytest.fixture
def mock_dataset_path(tmp_path: Path) -> Path:
    """Create mock dataset files for testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture
        
    Returns:
        Path to temporary dataset directory
    """
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()
    
    # Create mock data files
    for i in range(3):
        data = np.random.randn(10, 32, 32, 3)  # time, height, width, channels
        np.save(dataset_dir / f"sample_{i}.npy", data)
    
    return dataset_dir


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get appropriate device for testing.
    
    Returns:
        Torch device (CPU for CI/CD compatibility)
    """
    return torch.device("cpu")  # Use CPU for reproducible CI tests


class MockPDEDataset:
    """Mock PDE dataset for testing without large files."""
    
    def __init__(self, size: int = 10, resolution: int = 32):
        self.size = size
        self.resolution = resolution
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate deterministic data based on index
        torch.manual_seed(idx)
        input_data = torch.randn(3, self.resolution, self.resolution)
        target_data = torch.randn(3, self.resolution, self.resolution)
        return input_data, target_data


@pytest.fixture
def mock_dataset() -> MockPDEDataset:
    """Create mock dataset for testing.
    
    Returns:
        MockPDEDataset instance
    """
    return MockPDEDataset(size=20, resolution=32)


# Security testing fixtures
@pytest.fixture
def secure_config() -> Dict[str, Any]:
    """Generate secure configuration for security tests.
    
    Returns:
        Dictionary with security settings enabled
    """
    return {
        "security": {
            "validate_models": True,
            "max_model_size": "10MB",
            "allowed_file_types": [".pt", ".pth"],
            "sandbox_execution": True,
            "secure_mode": True
        }
    }


@pytest.fixture
def malicious_model_path(tmp_path: Path) -> Path:
    """Create mock malicious model file for security testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture
        
    Returns:
        Path to mock malicious file
    """
    malicious_path = tmp_path / "malicious.pkl"
    # Create a file that would be caught by security scanning
    with open(malicious_path, "wb") as f:
        f.write(b"malicious content that should be blocked")
    return malicious_path


# Performance testing fixtures
@pytest.fixture
def performance_data() -> Dict[str, torch.Tensor]:
    """Generate larger data for performance testing.
    
    Returns:
        Dictionary with larger tensors for performance tests
    """
    batch_size, channels, height, width = 4, 8, 128, 128
    
    return {
        "input": torch.randn(batch_size, channels, height, width),
        "target": torch.randn(batch_size, channels, height, width)
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "security: marks security-related tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker by default
        if not any(marker.name in ["integration", "security", "slow"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add security marker for security tests
        if "security" in item.name or "malicious" in item.name:
            item.add_marker(pytest.mark.security)
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.name:
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))