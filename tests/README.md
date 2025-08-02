# Neural Operator Foundation Lab - Test Suite

## Overview

This directory contains the comprehensive test suite for the Neural Operator Foundation Lab. The tests are organized into different categories to ensure thorough coverage of all functionality.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini              # Pytest configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_models.py
│   ├── test_data_loaders.py
│   ├── test_neural_operators.py
│   ├── test_config.py
│   └── test_security.py
├── integration/             # Integration tests
│   └── test_training_pipeline.py
├── end_to_end/             # End-to-end tests
│   └── test_complete_workflow.py
├── performance/            # Performance benchmarks
│   └── test_benchmarks.py
└── fixtures/               # Test data and utilities
    └── test_data_generator.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Single function or class
- **Dependencies**: Minimal, use mocks for external dependencies

**Run unit tests:**
```bash
pytest tests/unit/ -m unit
```

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between components
- **Speed**: Medium (1-10 seconds each)
- **Scope**: Multiple components working together
- **Dependencies**: Real components, minimal external services

**Run integration tests:**
```bash
pytest tests/integration/ -m integration
```

### End-to-End Tests (`tests/end_to_end/`)
- **Purpose**: Test complete workflows from start to finish
- **Speed**: Slow (10+ seconds each)
- **Scope**: Full system functionality
- **Dependencies**: Complete system setup

**Run end-to-end tests:**
```bash
pytest tests/end_to_end/ -m end_to_end
```

### Performance Tests (`tests/performance/`)
- **Purpose**: Benchmark performance and detect regressions
- **Speed**: Variable (can be very slow)
- **Scope**: Performance-critical code paths
- **Dependencies**: Performance monitoring tools

**Run performance tests:**
```bash
pytest tests/performance/ -m performance
```

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest

# Run only fast tests
pytest -m "not slow"

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "unit or integration"

# Run tests with coverage
pytest --cov=neural_operator_lab

# Run tests in parallel (faster)
pytest -n auto

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test function
pytest tests/unit/test_models.py::TestFourierNeuralOperator::test_forward_pass

# Run tests matching pattern
pytest -k "test_training"
```

### Test Environment Setup

#### Local Development
```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

#### CI/CD Environment
```bash
# Install all dependencies
pip install -e ".[dev,test]"

# Run comprehensive test suite
pytest --cov=neural_operator_lab --cov-fail-under=80
```

#### GPU Testing
```bash
# Run GPU-specific tests (requires CUDA)
pytest -m gpu

# Skip GPU tests
pytest -m "not gpu"
```

## Test Markers

Use pytest markers to categorize and select tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.end_to_end` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.security` - Security tests
- `@pytest.mark.parametrize` - Parameterized tests

### Example Usage

```python
import pytest

@pytest.mark.unit
def test_model_initialization():
    """Unit test for model initialization."""
    pass

@pytest.mark.integration
def test_training_pipeline():
    """Integration test for training pipeline."""
    pass

@pytest.mark.slow
@pytest.mark.performance
def test_large_model_benchmark():
    """Performance test for large models."""
    pass

@pytest.mark.gpu
def test_cuda_operations():
    """Test CUDA-specific operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
```

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

Common fixtures available to all tests:

- `sample_pde_data` - Standard 2D PDE data for testing
- `sample_3d_data` - 3D PDE data for testing
- `mock_dataset` - Mock dataset for training tests
- `temp_model_path` - Temporary file path for model saving/loading
- `device` - Appropriate device for testing (CPU/GPU)

### Test Data Generation

Use the `TestDataGenerator` classes in `tests/fixtures/` to create synthetic test data:

```python
from tests.fixtures.test_data_generator import SyntheticPDEDataGenerator

generator = SyntheticPDEDataGenerator(seed=42)
data = generator.generate_burgers_1d(num_samples=100)
```

## Writing New Tests

### Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Use descriptive test function names
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Performance Considerations**: Mark slow tests appropriately

### Example Test Structure

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
            input_dim=3, output_dim=1,
            modes=[16, 16], width=64
        )
    
    @pytest.mark.unit
    def test_initialization(self, model):
        """Test model initializes correctly."""
        assert model.input_dim == 3
        assert model.output_dim == 1
    
    @pytest.mark.unit
    def test_forward_pass(self, model, sample_pde_data):
        """Test forward pass produces correct output shape."""
        input_data = sample_pde_data['input']
        output = model(input_data)
        
        expected_shape = (input_data.shape[0], 1, 
                         input_data.shape[2], input_data.shape[3])
        assert output.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model handles different batch sizes."""
        input_data = torch.randn(batch_size, 3, 64, 64)
        output = model(input_data)
        assert output.shape[0] == batch_size
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_inference_speed(self, model, performance_data):
        """Test model inference speed."""
        import time
        
        model.eval()
        input_data = performance_data['large']
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 1.0  # Should be fast
```

## Continuous Integration

### GitHub Actions Integration

Tests are automatically run in CI/CD pipelines with different configurations:

- Python versions: 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- Hardware: CPU and GPU (when available)

### Coverage Requirements

- Minimum coverage: 80%
- Coverage reports generated in HTML and XML formats
- Uncovered lines reported in CI output

### Performance Monitoring

- Performance tests run on every PR
- Benchmark results compared against baselines
- Performance regressions trigger warnings

## Debugging Tests

### Common Issues and Solutions

#### Test Failures
```bash
# Run specific failing test with verbose output
pytest tests/unit/test_models.py::test_forward_pass -v -s

# Debug with pdb
pytest --pdb tests/unit/test_models.py::test_forward_pass

# Show local variables on failure
pytest --tb=long tests/unit/test_models.py::test_forward_pass
```

#### Memory Issues
```bash
# Run with memory profiling
pytest --memory-profile tests/performance/

# Monitor GPU memory
nvidia-smi -l 1  # In separate terminal
```

#### Slow Tests
```bash
# Profile test execution time
pytest --durations=10

# Run only fast tests during development
pytest -m "not slow"
```

## Contributing

When adding new functionality:

1. **Write tests first** (Test-Driven Development)
2. **Maintain coverage** (aim for >90% on new code)
3. **Include integration tests** for user-facing features
4. **Add performance tests** for performance-critical code
5. **Update documentation** including test descriptions

### Test Review Checklist

- [ ] Tests cover happy path and edge cases
- [ ] Tests are properly categorized with markers
- [ ] Test names are descriptive
- [ ] Tests are isolated and don't depend on each other
- [ ] Mock external dependencies appropriately
- [ ] Performance tests have reasonable baselines
- [ ] Tests pass in CI/CD environment

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [PyTorch testing best practices](https://pytorch.org/docs/stable/testing.html)