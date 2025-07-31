# Testing Guide

Comprehensive testing strategy and guidelines for Neural Operator Foundation Lab.

## 🧪 Testing Philosophy

Our testing approach follows a multi-layered strategy:

1. **Unit Tests**: Fast, isolated tests for individual functions
2. **Integration Tests**: Test component interactions
3. **Security Tests**: Validate security measures and policies
4. **Performance Tests**: Benchmark and profiling
5. **End-to-End Tests**: Complete workflow validation

## 🏃‍♂️ Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=neural_operator_lab --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m security           # Security tests only
pytest -m integration        # Integration tests only
```

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_config.py           # Configuration tests
├── test_security.py         # Security validation tests
├── test_models.py           # Model functionality tests
├── unit/                    # Unit tests
│   ├── test_operators.py
│   ├── test_data_loaders.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── test_training_pipeline.py
│   ├── test_model_io.py
│   └── test_benchmarks.py
├── security/                # Security-focused tests
│   ├── test_input_validation.py
│   ├── test_file_security.py
│   └── test_model_security.py
└── performance/             # Performance benchmarks
    ├── test_inference_speed.py
    ├── test_memory_usage.py
    └── test_scalability.py
```

## 🔧 Test Configuration

### Pytest Configuration

The `pyproject.toml` includes comprehensive pytest settings:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=neural_operator_lab --cov-report=term-missing --cov-report=html"
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests that require GPU",
    "security: marks security-related tests",
    "integration: marks integration tests",
    "unit: marks unit tests"
]
```

### Test Markers

Use markers to categorize and selectively run tests:

```python
import pytest

@pytest.mark.unit
def test_basic_functionality():
    """Unit test example."""
    pass

@pytest.mark.slow
def test_performance_benchmark():
    """Performance test that takes time."""
    pass

@pytest.mark.security
def test_input_validation():
    """Security-related test."""
    pass

@pytest.mark.gpu
def test_gpu_training():
    """Test requiring GPU hardware."""
    pass
```

## 🧬 Unit Testing

### Model Tests

```python
# tests/unit/test_models.py
import torch
import pytest
from neural_operator_lab.models import TransformerNeuralOperator

class TestTransformerNeuralOperator:
    
    def test_initialization(self):
        """Test model initialization."""
        model = TransformerNeuralOperator(
            input_dim=4,
            hidden_dim=64,
            num_layers=2
        )
        assert model.input_dim == 4
        assert model.hidden_dim == 64
        assert model.num_layers == 2
    
    def test_forward_pass(self, sample_pde_data):
        """Test forward pass with sample data."""
        model = TransformerNeuralOperator(input_dim=3, hidden_dim=64)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_pde_data["input"])
            
        assert output.shape == sample_pde_data["input"].shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self, sample_pde_data):
        """Test model works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        device = torch.device("cuda")
        model = TransformerNeuralOperator().to(device)
        input_data = sample_pde_data["input"].to(device)
        
        output = model(input_data)
        assert output.device == device
```

### Data Pipeline Tests

```python
# tests/unit/test_data_loaders.py
import pytest
from neural_operator_lab.data import PDEDataset

class TestPDEDataset:
    
    def test_dataset_creation(self, mock_dataset_path):
        """Test dataset creation from files."""
        dataset = PDEDataset(mock_dataset_path)
        assert len(dataset) > 0
    
    def test_data_normalization(self, mock_dataset):
        """Test data normalization."""
        normalized_data = mock_dataset.normalize()
        # Test that mean is close to 0 and std close to 1
        assert abs(normalized_data.mean()) < 0.1
        assert abs(normalized_data.std() - 1.0) < 0.1
    
    def test_data_augmentation(self, mock_dataset):
        """Test data augmentation."""
        augmented = mock_dataset.augment(rotation=True, noise=0.01)
        assert len(augmented) >= len(mock_dataset)
```

## 🔗 Integration Testing

### Training Pipeline Tests

```python
# tests/integration/test_training_pipeline.py
import pytest
import torch
from neural_operator_lab.training import Trainer
from neural_operator_lab.models import TransformerNeuralOperator

class TestTrainingPipeline:
    
    def test_complete_training_workflow(self, mock_dataset, temp_model_path):
        """Test complete training workflow."""
        model = TransformerNeuralOperator(input_dim=3, hidden_dim=32)
        trainer = Trainer(
            model=model,
            dataset=mock_dataset,
            batch_size=2,
            learning_rate=1e-3
        )
        
        # Run short training
        history = trainer.train(epochs=2, save_path=temp_model_path)
        
        # Verify training completed
        assert len(history["loss"]) == 2
        assert temp_model_path.exists()
        
        # Test model loading
        loaded_model = torch.load(temp_model_path)
        assert isinstance(loaded_model, TransformerNeuralOperator)
    
    def test_distributed_training(self, mock_dataset):
        """Test distributed training setup."""
        # Mock distributed environment
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
            model = TransformerNeuralOperator()
            trainer = Trainer(model, mock_dataset, distributed=True)
            
            # Verify distributed setup
            assert trainer.distributed
            assert trainer.world_size == 2
```

### Model I/O Tests

```python
# tests/integration/test_model_io.py
class TestModelIO:
    
    def test_model_save_load_consistency(self, temp_model_path):
        """Test model save/load preserves functionality."""
        original_model = TransformerNeuralOperator()
        test_input = torch.randn(1, 3, 32, 32)
        
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(test_input)
        
        # Save and load model
        torch.save(original_model.state_dict(), temp_model_path)
        loaded_model = TransformerNeuralOperator()
        loaded_model.load_state_dict(torch.load(temp_model_path))
        
        # Compare outputs
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(test_input)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
```

## 🔒 Security Testing

### Input Validation Tests

```python
# tests/security/test_input_validation.py
import pytest
from neural_operator_lab.security import validate_input, SecurityError

class TestInputValidation:
    
    def test_malicious_file_rejection(self, malicious_model_path):
        """Test rejection of malicious files."""
        with pytest.raises(SecurityError):
            validate_model_file(malicious_model_path)
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32"
        ]
        
        for path in malicious_paths:
            with pytest.raises(SecurityError):
                validate_file_path(path)
    
    def test_injection_prevention(self):
        """Test prevention of code injection."""
        malicious_inputs = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')"
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(SecurityError):
                validate_user_code(malicious_input)
```

### Model Security Tests

```python
# tests/security/test_model_security.py
class TestModelSecurity:
    
    @pytest.mark.security
    def test_secure_model_loading(self, secure_config):
        """Test secure model loading configuration."""
        from neural_operator_lab.config import get_security_config
        
        config = get_security_config()
        assert config.validate_models
        assert config.secure_mode
        assert config.sandbox_execution
    
    def test_file_size_limits(self, tmp_path):
        """Test file size validation."""
        # Create file exceeding limits
        large_file = tmp_path / "large_model.pt"
        large_file.write_bytes(b"0" * (2 * 1024 * 1024 * 1024))  # 2GB
        
        with pytest.raises(SecurityError, match="File too large"):
            validate_model_file(large_file)
    
    def test_allowed_file_extensions(self):
        """Test file extension validation."""
        allowed_files = ["model.pt", "weights.pth", "data.h5"]
        forbidden_files = ["script.py", "malware.exe", "payload.pkl"]
        
        for file_path in allowed_files:
            # Should not raise
            validate_file_extension(file_path)
        
        for file_path in forbidden_files:
            with pytest.raises(SecurityError):
                validate_file_extension(file_path)
```

## ⚡ Performance Testing

### Inference Speed Tests

```python
# tests/performance/test_inference_speed.py
import time
import pytest
from neural_operator_lab.models import TransformerNeuralOperator

@pytest.mark.slow
class TestInferenceSpeed:
    
    def test_single_inference_time(self, performance_data):
        """Test single inference execution time."""
        model = TransformerNeuralOperator()
        model.eval()
        
        input_data = performance_data["input"]
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Assert reasonable performance
        assert inference_time < 1.0  # Less than 1 second
        assert output is not None
    
    def test_batch_inference_scaling(self):
        """Test inference time scaling with batch size."""
        model = TransformerNeuralOperator()
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 3, 64, 64)
            
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Time should scale sub-linearly with batch size
        efficiency = times[-1] / (times[0] * batch_sizes[-1])
        assert efficiency < 1.0  # More efficient than linear scaling
```

### Memory Usage Tests

```python
# tests/performance/test_memory_usage.py
@pytest.mark.gpu
class TestMemoryUsage:
    
    def test_gpu_memory_efficiency(self, performance_data):
        """Test GPU memory usage efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device("cuda")
        model = TransformerNeuralOperator().to(device)
        input_data = performance_data["input"].to(device)
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            output = model(input_data)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_used < 2 * 1024**3  # Less than 2GB
        
        # Cleanup
        del output, input_data, model
        torch.cuda.empty_cache()
```

## 🎯 Test Automation

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run unit tests
      run: pytest tests/unit/ -v
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run security tests
      run: pytest tests/security/ -v
    
    - name: Generate coverage report
      run: pytest --cov=neural_operator_lab --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
        
      - id: pytest-security
        name: pytest-security
        entry: pytest tests/security/
        language: system
        pass_filenames: false
```

## 📊 Test Coverage

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/neural_operator_lab"]
omit = ["*/tests/*", "*/test_*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
show_missing = true
```

### Coverage Targets

- **Overall Coverage**: Minimum 85%
- **Critical Modules**: Minimum 95%
- **Security Code**: 100% coverage required
- **New Code**: Must maintain or improve coverage

## 🐛 Test Data Management

### Fixtures and Mocks

```python
# tests/conftest.py
@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    return create_synthetic_dataset(
        size=1000,
        resolution=256,
        timesteps=100
    )

@pytest.fixture
def gpu_model():
    """Create GPU-enabled model if available."""
    if torch.cuda.is_available():
        return TransformerNeuralOperator().cuda()
    else:
        pytest.skip("GPU not available")
```

### Test Data Isolation

- Use temporary directories for file operations
- Mock external dependencies
- Clean up resources after tests
- Seed random number generators for reproducibility

## 🚀 Running Tests

### Local Development

```bash
# Quick smoke test
pytest tests/unit/ -x

# Run with coverage
pytest --cov=neural_operator_lab --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest -k "test_security" -v

# Run tests by marker
pytest -m "not slow" -v
```

### Production Testing

```bash
# Full test suite
tox

# Specific environment
tox -e py39

# Security tests only
tox -e security

# Performance benchmarks
tox -e performance
```

### Docker Testing

```bash
# Run tests in container
docker-compose run --rm tests

# Run specific test category
docker-compose run --rm neural-operator-lab pytest -m security
```

## 📈 Test Metrics and Reporting

### Metrics to Track

- Test execution time
- Coverage percentage
- Test failure rate
- Security test results
- Performance benchmarks

### Reporting Tools

- **Coverage**: HTML reports, Codecov integration
- **Performance**: pytest-benchmark for timing
- **Security**: Bandit reports, Safety checks
- **Quality**: SonarQube integration

This comprehensive testing guide ensures high-quality, secure, and performant code through systematic validation at multiple levels.