# Neural Operator Foundation Lab - Developer Guide

## Table of Contents
1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Adding New Features](#adding-new-features)
7. [Performance Optimization](#performance-optimization)
8. [Release Process](#release-process)

---

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- CUDA 11.0+ (for GPU development)
- Docker (for containerized development)

### Local Development Setup

#### 1. Clone and Setup Environment
```bash
git clone https://github.com/yourusername/neural-operator-foundation-lab.git
cd neural-operator-foundation-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev,test,docs]"
```

#### 2. Setup Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files  # Test all hooks
```

#### 3. Verify Installation
```bash
# Run tests
pytest tests/

# Check code quality
make lint
make typecheck

# Build documentation
make docs
```

### Docker Development
```bash
# Build development container
docker build -f Dockerfile.dev -t neural-ops-dev .

# Run interactive development
docker run -it --gpus all -v $(pwd):/workspace neural-ops-dev bash
```

### VSCode Development
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.pylint
code --install-extension ms-python.black-formatter

# Open in container
code --folder-uri vscode-remote://dev-container+$(pwd)/workspace
```

---

## Architecture Overview

### Project Structure
```
neural-operator-foundation-lab/
├── src/neural_operator_lab/          # Main package
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── base.py                       # Abstract base classes
│   ├── models/                       # Neural operator implementations
│   │   ├── fourier.py               # FNO implementation
│   │   ├── probabilistic.py         # PNO implementation
│   │   ├── transformer.py           # TNO implementation
│   │   └── graph.py                 # Graph neural operators
│   ├── data/                        # Data handling
│   │   ├── datasets.py              # Dataset classes
│   │   ├── loaders.py               # Data loaders
│   │   └── preprocessing.py         # Data preprocessing
│   ├── training/                    # Training infrastructure
│   │   ├── trainer.py               # Main trainer class
│   │   ├── losses.py                # Loss functions
│   │   └── callbacks.py             # Training callbacks
│   ├── benchmarks/                  # Benchmarking suite
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── suite.py                 # Benchmark suite
│   └── utils/                       # Utilities
│       ├── visualization.py         # Plotting utilities
│       └── security.py              # Security utilities
├── tests/                           # Test suite
├── docs/                           # Documentation
├── examples/                       # Example scripts
├── scripts/                        # Utility scripts
└── configs/                        # Configuration files
```

### Key Design Patterns

#### 1. Abstract Base Classes
```python
# All neural operators inherit from NeuralOperatorBase
class NeuralOperatorBase(nn.Module):
    """Base class for all neural operators."""
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the operator."""
        raise NotImplementedError
    
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                 inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss for training."""
        return F.mse_loss(pred, target)
    
    @property
    def parameter_count(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

#### 2. Configuration System
```python
# Hierarchical configuration with validation
@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 256
    
    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    
    @classmethod
    def from_file(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

#### 3. Plugin System
```python
# Register new operators
@register_operator("custom_fno")
class CustomFNO(NeuralOperatorBase):
    def __init__(self, config):
        super().__init__()
        # Implementation
        
    def forward(self, x, grid=None):
        # Custom forward pass
        return x

# Use registered operators
model = create_operator("custom_fno", config)
```

---

## Contributing Guidelines

### Git Workflow
```bash
# 1. Create feature branch
git checkout -b feature/new-operator

# 2. Make changes and commit
git add .
git commit -m "feat: add new operator implementation"

# 3. Push and create PR
git push origin feature/new-operator
# Create PR on GitHub
```

### Commit Message Format
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Examples:
```bash
git commit -m "feat(models): add transformer neural operator"
git commit -m "fix(training): resolve gradient clipping issue"
git commit -m "docs: update installation instructions"
```

### Pull Request Process
1. **Ensure CI passes**: All tests, linting, and type checking
2. **Add tests**: New features require comprehensive tests
3. **Update documentation**: API changes need doc updates
4. **Request review**: Tag appropriate reviewers
5. **Address feedback**: Respond to all review comments

---

## Code Standards

### Python Style Guide
We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints for all public functions
def process_data(data: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Process input data with optional normalization.
    
    Args:
        data: Input tensor of shape [batch, channels, height, width]
        normalize: Whether to apply normalization
        
    Returns:
        Processed tensor with same shape as input
        
    Raises:
        ValueError: If data has wrong number of dimensions
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {data.ndim}D")
    
    if normalize:
        data = (data - data.mean()) / data.std()
    
    return data
```

### Code Formatting
We use automated formatting tools:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Type Checking
All code must pass type checking:

```bash
# Run type checker
mypy src/neural_operator_lab/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
```

### Documentation Standards
```python
class TransformerNeuralOperator(NeuralOperatorBase):
    """Transformer-based neural operator for PDEs.
    
    This implementation uses multi-head attention to capture long-range
    dependencies in the solution space, making it particularly effective
    for complex, multi-scale problems.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels  
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        
    Example:
        >>> model = TransformerNeuralOperator(
        ...     input_dim=3, output_dim=1,
        ...     hidden_dim=512, num_layers=8
        ... )
        >>> output = model(input_tensor, grid)
    """
```

---

## Testing

### Testing Philosophy
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Property-based tests**: Test invariants and edge cases
- **Performance tests**: Benchmark critical paths

### Test Structure
```
tests/
├── unit/                   # Unit tests
│   ├── test_models.py
│   ├── test_training.py
│   └── test_data.py
├── integration/            # Integration tests
│   ├── test_training_pipeline.py
│   └── test_benchmarks.py
├── end_to_end/            # E2E tests
│   └── test_complete_workflow.py
├── performance/           # Performance tests
│   └── test_benchmarks.py
├── fixtures/              # Test data and fixtures
└── conftest.py           # Pytest configuration
```

### Writing Tests
```python
import pytest
import torch
from neural_operator_lab.models import FourierNeuralOperator

class TestFourierNeuralOperator:
    """Test suite for Fourier Neural Operator."""
    
    @pytest.fixture
    def model(self):
        """Create FNO instance for testing."""
        return FourierNeuralOperator(
            input_dim=2, output_dim=1,
            modes=[16, 16], width=64
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, height, width = 4, 64, 64
        input_data = torch.randn(batch_size, 2, height, width)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing='ij'
        ), dim=-1)
        return input_data, grid
    
    def test_forward_pass(self, model, sample_data):
        """Test forward pass produces correct output shape."""
        input_data, grid = sample_data
        output = model(input_data, grid)
        
        expected_shape = (4, 1, 64, 64)  # [batch, output_dim, height, width]
        assert output.shape == expected_shape
    
    def test_gradient_flow(self, model, sample_data):
        """Test gradients flow properly through model."""
        input_data, grid = sample_data
        output = model(input_data, grid)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model works with different batch sizes."""
        input_data = torch.randn(batch_size, 2, 64, 64)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 64),
            torch.linspace(0, 1, 64),
            indexing='ij'
        ), dim=-1)
        
        output = model(input_data, grid)
        assert output.shape[0] == batch_size
    
    def test_memory_efficiency(self, model):
        """Test model doesn't leak memory."""
        initial_memory = torch.cuda.memory_allocated()
        
        for _ in range(10):
            input_data = torch.randn(4, 2, 64, 64, device='cuda')
            output = model.cuda()(input_data)
            del output, input_data
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory + 1024**2  # Allow 1MB tolerance

# Property-based testing with hypothesis
from hypothesis import given, strategies as st

@given(
    batch_size=st.integers(min_value=1, max_value=16),
    height=st.integers(min_value=32, max_value=128),
    width=st.integers(min_value=32, max_value=128)
)
def test_fno_invariants(batch_size, height, width):
    """Test FNO satisfies basic invariants."""
    model = FourierNeuralOperator(input_dim=1, output_dim=1, modes=[8, 8])
    input_data = torch.randn(batch_size, 1, height, width)
    
    output = model(input_data)
    
    # Output should have same spatial dimensions
    assert output.shape[2:] == input_data.shape[2:]
    # Output should have correct batch size
    assert output.shape[0] == batch_size
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=neural_operator_lab --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run tests in parallel
pytest -n auto  # Requires pytest-xdist
```

---

## Adding New Features

### Adding a New Neural Operator

#### 1. Create Model Class
```python
# src/neural_operator_lab/models/my_operator.py
from typing import Optional
import torch
import torch.nn as nn
from ..base import NeuralOperatorBase

class MyNeuralOperator(NeuralOperatorBase):
    """My custom neural operator implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define your layers here
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the operator."""
        # Implement your forward logic
        return self.layers(x)
```

#### 2. Register the Operator
```python
# src/neural_operator_lab/models/__init__.py
from .my_operator import MyNeuralOperator

# Add to __all__
__all__ = [
    "FourierNeuralOperator",
    "ProbabilisticNeuralOperator", 
    "TransformerNeuralOperator",
    "MyNeuralOperator",  # Add your operator
]

# Register in factory
from ..registry import register_operator
register_operator("my_operator", MyNeuralOperator)
```

#### 3. Add Configuration
```python
# src/neural_operator_lab/configs/my_operator.py
from dataclasses import dataclass
from ..base import ConfigBase

@dataclass
class MyOperatorConfig(ConfigBase):
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    activation: str = "relu"
    
    def validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
```

#### 4. Write Tests
```python
# tests/unit/test_my_operator.py
import pytest
import torch
from neural_operator_lab.models import MyNeuralOperator

class TestMyNeuralOperator:
    def test_initialization(self):
        model = MyNeuralOperator(input_dim=3, output_dim=1)
        assert model.input_dim == 3
        assert model.output_dim == 1
    
    def test_forward_pass(self):
        model = MyNeuralOperator(input_dim=3, output_dim=1)
        x = torch.randn(4, 3, 64, 64)
        output = model(x)
        assert output.shape == (4, 1, 64, 64)
```

#### 5. Add Documentation
```python
# docs/operators/my_operator.md
# My Neural Operator

## Overview
Description of your operator...

## Usage
```python
from neural_operator_lab import MyNeuralOperator

model = MyNeuralOperator(input_dim=3, output_dim=1)
```

## Theory
Mathematical background...
```

### Adding a New Dataset

#### 1. Create Dataset Class
```python
# src/neural_operator_lab/data/my_dataset.py
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from ..base import PDEDatasetBase

class MyPDEDataset(PDEDatasetBase):
    """Dataset for my specific PDE problem."""
    
    def __init__(self, data_path: str, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self):
        """Load data from disk."""
        # Implement data loading logic
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample."""
        input_data = self.data[idx]['input']
        output_data = self.data[idx]['output']
        return input_data, output_data
```

---

## Performance Optimization

### Profiling
```python
# Profile training loop
from neural_operator_lab.utils.profiling import profile_training

@profile_training
def train_step(model, batch):
    # Training step implementation
    pass

# Profile memory usage
from neural_operator_lab.utils.profiling import MemoryProfiler

with MemoryProfiler() as profiler:
    model = create_model()
    train_model(model)

profiler.print_memory_stats()
```

### Optimization Techniques

#### 1. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch.input)
        loss = criterion(output, batch.target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2. Gradient Checkpointing
```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Use gradient checkpointing for memory-intensive layers
        x = checkpoint.checkpoint(self.expensive_layer, x)
        return x
```

#### 3. Model Compilation (PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```

### CUDA Optimizations
```python
# Optimize CUDA settings
torch.backends.cudnn.benchmark = True  # For fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # For A100 GPUs

# Use fused optimizers
from torch._C._distributed_c10d import ProcessGroup
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3, 
    fused=True  # Fused kernel for speed
)
```

---

## Release Process

### Versioning Strategy
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

#### Pre-Release
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `__init__.py`
- [ ] Security scan completed
- [ ] Performance benchmarks run

#### Release Process
```bash
# 1. Create release branch
git checkout -b release/v1.2.0

# 2. Update version
bump2version minor  # Updates version strings

# 3. Update changelog
git add CHANGELOG.md
git commit -m "docs: update changelog for v1.2.0"

# 4. Create release PR
git push origin release/v1.2.0
# Create PR to main branch

# 5. After PR approval, tag release
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 6. GitHub Actions will handle PyPI upload
```

#### Post-Release
- [ ] Verify PyPI upload
- [ ] Update documentation
- [ ] Announce on social media
- [ ] Close GitHub milestone
- [ ] Plan next release

### Continuous Integration
Our CI pipeline includes:

1. **Code Quality**: Linting, formatting, type checking
2. **Testing**: Unit, integration, and end-to-end tests
3. **Security**: Vulnerability scanning, dependency checks
4. **Performance**: Benchmark regression testing
5. **Documentation**: Build and deploy docs
6. **Packaging**: Build and test distributions

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev,test]"
      - name: Run tests
        run: pytest --cov=neural_operator_lab
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Getting Help

### Development Resources
- **Architecture docs**: [docs/ARCHITECTURE.md](../ARCHITECTURE.md)
- **API reference**: [https://neural-operator-lab.readthedocs.io](https://neural-operator-lab.readthedocs.io)
- **Example implementations**: [examples/](../../examples/)
- **Development discussions**: GitHub Discussions

### Community Support
- **Discord**: Real-time developer chat
- **GitHub Issues**: Bug reports and feature requests
- **Code reviews**: Learn from experienced contributors
- **Office hours**: Monthly developer Q&A sessions

### Mentorship Program
We offer mentorship for new contributors:
- Pair programming sessions
- Code review guidance
- Project planning assistance
- Career development advice

---

*This guide is maintained by the core development team. Last updated: 2025-08-01*