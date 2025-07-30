# Development Guide

This guide covers the development setup and workflow for Neural Operator Foundation Lab.

## 🚀 Quick Setup

### Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/terragon-labs/neural-operator-foundation-lab.git
cd neural-operator-foundation-lab

# Run automated setup
./scripts/setup_dev.sh
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## 📁 Project Structure

```
neural-operator-foundation-lab/
├── src/neural_operator_lab/     # Main package source
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration and security
│   ├── models/                 # Neural operator architectures
│   ├── data/                   # Data loading utilities
│   ├── training/               # Training loops and optimizers
│   ├── benchmarks/             # Evaluation and benchmarking
│   └── visualization/          # Plotting and visualization
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_config.py          # Configuration tests
│   ├── test_models/            # Model tests
│   ├── test_data/              # Data loading tests
│   └── fixtures/               # Test data and fixtures
├── docs/                       # Documentation
│   ├── tutorials/              # Step-by-step guides
│   ├── api/                    # API reference
│   └── examples/               # Code examples
├── examples/                   # Example scripts and notebooks
├── scripts/                    # Utility scripts
├── .github/                    # GitHub configuration
└── Configuration files...
```

## 🛠️ Development Workflow

### 1. Code Style
We enforce consistent code style using:
- **Black**: Code formatting (88 character line limit)
- **isort**: Import sorting
- **flake8**: Linting and style checks
- **mypy**: Type checking

Run formatting:
```bash
make format  # Format all code
make lint    # Check code style
make type-check  # Run type checking
```

### 2. Testing
We use pytest for testing with the following structure:
- Unit tests for individual components
- Integration tests for workflows
- Performance benchmarks for critical paths

Run tests:
```bash
make test        # Run all tests with coverage
make test-fast   # Run tests without coverage (faster)
pytest tests/test_specific.py -v  # Run specific test file
```

### 3. Pre-commit Hooks
Pre-commit hooks run automatically before each commit:
```bash
pre-commit run --all-files  # Run manually on all files
```

Hooks include:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security checks
- Large file detection

### 4. Documentation
Documentation is built with Sphinx:
```bash
make docs       # Build HTML documentation
make docs-serve # Serve documentation locally (localhost:8000)
```

### 5. Git Workflow
We follow a standard Git workflow:
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "feat: your feature"`
3. Push branch: `git push origin feature/your-feature`
4. Create pull request
5. Address review feedback
6. Merge when approved

## 🧪 Testing Guidelines

### Test Structure
```python
# tests/test_example.py
"""Tests for example module."""

import pytest
from neural_operator_lab.example import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    def test_initialization(self):
        """Test proper initialization."""
        instance = ExampleClass()
        assert instance is not None
    
    def test_method_with_params(self):
        """Test method with different parameters."""
        instance = ExampleClass()
        result = instance.method(param="test")
        assert result == "expected_value"
    
    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple parameter sets."""
        instance = ExampleClass()
        result = instance.process(input)
        assert result == expected
```

### Test Categories
- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical operations
- **Security tests**: Validate security measures

### Coverage Requirements
- Minimum 80% test coverage
- All public APIs must be tested
- Critical paths require 100% coverage

## 🔒 Security Practices

### Code Security
- Never commit secrets or API keys
- Use secure randomness (`secrets` module)
- Validate all user inputs
- Follow principle of least privilege

### Dependency Security
- Pin dependency versions
- Regular security updates
- Vulnerability scanning with Dependabot

### Model Security
- Validate model files before loading
- Use safe deserialization methods
- Implement resource limits

## 📊 Performance Considerations

### Profiling
```python
# Profile CPU usage
python -m cProfile -o profile.stats script.py

# Profile memory usage
from memory_profiler import profile

@profile
def my_function():
    # Your code here
    pass
```

### Optimization Guidelines
- Use vectorized operations (NumPy/PyTorch)
- Minimize memory allocations
- Profile before optimizing
- Consider GPU acceleration

## 🚀 Release Process

### Version Management
We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Steps
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. CI automatically publishes to PyPI

## 🐛 Debugging

### Common Issues
```bash
# Import errors
python -c "import neural_operator_lab; print('✅ Import successful')"

# Test failures
pytest tests/test_failing.py -v -s --tb=long

# Memory issues
python -X dev your_script.py  # Enable development mode
```

### Debugging Tools
- **pdb**: Interactive debugger
- **ipdb**: Enhanced debugger with IPython
- **pytest --pdb**: Drop into debugger on test failures

### Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

## 📞 Getting Help

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bug reports and feature requests
- **Development Discord**: Real-time chat
- **Code Review**: Use draft PRs for early feedback

## 🎯 Development Tips

### Efficient Development
```bash
# Run subset of tests during development
pytest tests/test_current_feature.py -k "test_specific_case"

# Watch for file changes and re-run tests
pytest-watch tests/

# Format code on save (VS Code)
# Add to settings.json:
{
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### IDE Configuration
- **VS Code**: Extensions for Python, Black, mypy
- **PyCharm**: Configure Black and isort
- **Vim/Neovim**: Use ALE or similar for linting

Happy developing! 🚀