# GitHub Actions Workflows Setup

⚠️ **Note**: Per Terragon Labs policy, GitHub Actions workflows must be manually created by repository maintainers for security reasons.

## Required Manual Setup

The following GitHub Actions workflows should be created manually in `.github/workflows/`:

### 1. Continuous Integration (`ci.yml`)
```yaml
name: CI
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
          pip install -r requirements-dev.txt
          pip install -e .
      - name: Run tests
        run: pytest tests/ --cov=neural_operator_lab
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 2. Code Quality (`quality.yml`)
```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run black
        run: black --check src/ tests/
      - name: Run isort
        run: isort --check-only src/ tests/
      - name: Run flake8
        run: flake8 src/ tests/
      - name: Run mypy
        run: mypy src/neural_operator_lab
```

### 3. Security Scanning (`security.yml`)
```yaml
name: Security
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r src/
      - name: Run Safety
        run: |
          pip install safety
          safety check -r requirements.txt
```

## Setup Instructions

1. **Create workflows manually** in the GitHub web interface
2. **Configure secrets** in repository settings:
   - `CODECOV_TOKEN` for test coverage
   - `PYPI_TOKEN` for package publishing (if needed)
3. **Enable branch protection** rules requiring status checks
4. **Configure Dependabot** for dependency updates

For detailed workflow configurations, see [DEVELOPMENT.md](DEVELOPMENT.md#github-actions).