# Contributing to Neural Operator Foundation Lab

We welcome contributions to the Neural Operator Foundation Lab! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/neural-operator-foundation-lab.git
   cd neural-operator-foundation-lab
   ```
3. **Set up development environment**:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🧪 Development Workflow

### Code Style
- Follow [Black](https://black.readthedocs.io/) formatting (88 character line limit)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Pass [flake8](https://flake8.pycqa.org/) linting
- Include type hints where possible ([mypy](https://mypy.readthedocs.io/))

### Testing
- Write tests for new functionality in `tests/`
- Ensure all tests pass: `pytest`
- Maintain test coverage above 80%
- Test with multiple Python versions (3.9-3.11) if possible

### Pre-commit Hooks
Pre-commit hooks run automatically before each commit:
```bash
pre-commit run --all-files  # Run manually
```

## 📝 Pull Request Process

1. **Ensure tests pass** and code follows style guidelines
2. **Update documentation** if you've changed APIs
3. **Add tests** for new functionality
4. **Write clear commit messages** following [Conventional Commits](https://www.conventionalcommits.org/)
5. **Submit pull request** with:
   - Clear description of changes
   - Link to relevant issues
   - Screenshots/examples if applicable

### PR Review Criteria
- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered

## 🎯 Priority Contribution Areas

### High Priority
- **New neural operator architectures** (FNO variants, graph operators)
- **Additional PDE benchmarks** (climate, materials science)
- **Performance optimizations** (GPU kernels, memory efficiency)
- **Documentation improvements** (tutorials, API docs)

### Medium Priority
- **Visualization tools** for 3D flow fields
- **Data loading utilities** for common PDE formats
- **Integration with HPC frameworks** (Horovod, Ray)
- **Mobile/edge deployment** optimization

### Welcome for Beginners
- **Bug fixes** with clear reproduction steps
- **Documentation typos** and clarity improvements
- **Test coverage** improvements
- **Example notebooks** and tutorials

## 🏗️ Code Organization

```
neural-operator-foundation-lab/
├── src/neural_operator_lab/     # Main package
│   ├── models/                  # Neural operator architectures
│   ├── data/                    # Data loading and preprocessing
│   ├── training/                # Training loops and utilities
│   ├── benchmarks/              # Evaluation and benchmarking
│   └── visualization/           # Plotting and animation tools
├── tests/                       # Test suites
├── docs/                        # Documentation
├── examples/                    # Example scripts and notebooks
└── scripts/                     # Utility scripts
```

## 🐛 Bug Reports

When reporting bugs, please include:
- **Python version** and OS
- **Package versions**: `pip freeze`
- **Minimal reproduction example**
- **Expected vs actual behavior**
- **Error messages** (full traceback)

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

## 💡 Feature Requests

For new features, please:
- **Check existing issues** to avoid duplicates
- **Describe the use case** clearly
- **Provide examples** of desired API/behavior
- **Consider implementation complexity**

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

## 📚 Documentation

- **Docstrings**: Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Type hints**: Include for all public APIs
- **Examples**: Provide usage examples in docstrings
- **Tutorials**: Add to `docs/tutorials/` for complex features

## 🤝 Community Guidelines

- **Be respectful** and inclusive to all contributors
- **Provide constructive feedback** in reviews
- **Help newcomers** get started
- **Credit others** for their contributions
- **Follow our [Code of Conduct](CODE_OF_CONDUCT.md)**

## ⚡ Development Tips

### Local Testing
```bash
# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=neural_operator_lab

# Run performance tests
pytest tests/test_performance.py -v --benchmark-only
```

### Documentation Building
```bash
cd docs/
make html
open _build/html/index.html
```

### Debugging
- Use `ipdb` for interactive debugging
- Enable verbose logging: `NEURAL_OP_LOG_LEVEL=DEBUG`
- Profile with `cProfile` for performance issues

## 🏷️ Release Process

For maintainers:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. GitHub Actions handles PyPI upload

## 📞 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat (link in README)
- **Email**: maintainers@neural-operator-lab.org

Thank you for contributing to Neural Operator Foundation Lab! 🙏