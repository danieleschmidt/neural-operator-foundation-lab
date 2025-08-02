# CI/CD Setup Guide

## Required GitHub Actions Workflows

Due to GitHub App permission limitations, the following workflow files need to be manually created in `.github/workflows/` directory by repository maintainers.

## 1. Continuous Integration (`ci.yml`)

**Location**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, '3.10', 3.11]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
        
    - name: Lint with ruff
      run: |
        ruff check src/ tests/
        
    - name: Format check with black
      run: |
        black --check src/ tests/
        
    - name: Type check with mypy
      run: |
        mypy src/neural_operator_lab/
        
    - name: Security check with bandit
      run: |
        bandit -r src/
        
    - name: Test with pytest
      run: |
        pytest tests/ --cov=neural_operator_lab --cov-report=xml --cov-fail-under=80
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## 2. Security Scanning (`security.yml`)

**Location**: `.github/workflows/security.yml`

```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] safety
        
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Run Safety dependency scan
      run: |
        safety check --json --output safety-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

## 3. Performance Benchmarks (`benchmarks.yml`)

**Location**: `.github/workflows/benchmarks.yml`

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Need full history for comparison
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
        
    - name: Run performance tests
      run: |
        pytest tests/performance/ -m performance --benchmark-json=benchmark.json
        
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '150%'
```

## 4. Documentation Build (`docs.yml`)

**Location**: `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
        
    - name: Build documentation
      run: |
        cd docs
        make html
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## 5. Release Automation (`release.yml`)

**Location**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build distribution
      run: |
        python -m build
        
    - name: Check distribution
      run: |
        twine check dist/*
        
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
        
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
```

## Required Secrets

Add these secrets in GitHub repository settings:

1. `CODECOV_TOKEN` - For coverage reporting
2. `PYPI_API_TOKEN` - For PyPI package publishing

## Branch Protection Rules

Configure branch protection for `main` branch:

1. Require pull request reviews before merging
2. Require status checks to pass before merging:
   - CI tests for all Python versions
   - Security scans
   - Code coverage threshold
3. Require branches to be up to date before merging
4. Restrict pushes that create files to the `main` branch

## Setup Instructions

1. Copy the workflow files from `docs/workflows/examples/` to `.github/workflows/`
2. Add required secrets to repository settings
3. Configure branch protection rules
4. Enable GitHub Pages for documentation
5. Test workflows by creating a pull request

## Monitoring and Maintenance

- Review workflow runs regularly
- Update action versions quarterly
- Monitor security alerts
- Adjust performance thresholds based on benchmarks
- Update Python versions as needed