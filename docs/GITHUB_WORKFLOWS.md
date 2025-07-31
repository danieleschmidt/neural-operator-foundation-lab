# GitHub Actions Workflows

This document contains the recommended GitHub Actions workflows for Neural Operator Foundation Lab. These workflows need to be manually created by a repository administrator with appropriate permissions.

## ⚠️ Setup Requirements

To implement these workflows, a repository administrator must:

1. Have `workflows` permission on the repository
2. Create the `.github/workflows/` directory
3. Add the workflow files shown below
4. Configure required secrets (see [Secrets Configuration](#secrets-configuration))

## 🔄 CI/CD Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  PYTHON_VERSION: "3.9"
  POETRY_VERSION: "1.6.1"

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Run tests with coverage
      run: |
        pytest --cov=neural_operator_lab --cov-report=xml --cov-report=term-missing
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        files: ./coverage.xml
        fail_ci_if_error: true

  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Run black
      run: black --check --diff src/ tests/
      
    - name: Run isort
      run: isort --check-only --diff src/ tests/
      
    - name: Run flake8
      run: flake8 src/ tests/
      
    - name: Run mypy
      run: mypy src/neural_operator_lab/

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Bandit security scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        args: '-r src/ -f json -o bandit-report.json'
        
    - name: Run Safety check
      run: |
        python -m pip install --upgrade pip safety
        safety check --json --output safety-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high --file=requirements.txt
        
    - name: Upload Snyk report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: snyk-report
        path: snyk.sarif

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, lint, security]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip build twine
        
    - name: Build package
      run: python -m build
      
    - name: Check package
      run: twine check dist/*
      
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

## 🔒 Security Workflow

Create `.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: moderate

  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Run TruffleHog
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
        extra_args: --debug --only-verified

  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install cyclonedx-bom
        
    - name: Generate SBOM
      run: cyclonedx-py -o sbom.json
      
    - name: Upload SBOM
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.json

  vulnerability-scan:
    name: Container Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Build Docker image
      run: docker build -t neural-operator-lab:latest .
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'neural-operator-lab:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
```

## 🚀 Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

env:
  PYTHON_VERSION: "3.9"

jobs:
  test:
    name: Test Before Release
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Run tests
      run: pytest --cov=neural_operator_lab
      
    - name: Run security scan
      run: |
        bandit -r src/ -f json -o bandit-report.json
        safety check

  build-and-publish:
    name: Build and Publish
    runs-on: ubuntu-latest
    needs: test
    permissions:
      contents: write
      id-token: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip build twine
        
    - name: Build package
      run: python -m build
      
    - name: Check package
      run: twine check dist/*
      
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json
        
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          dist/*
          sbom.json
        generate_release_notes: true
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        skip-existing: true

  docker-build:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: test
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=tag
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  notify:
    name: Notify Release
    runs-on: ubuntu-latest
    needs: [build-and-publish, docker-build]
    if: always()
    
    steps:
    - name: Notify success
      if: needs.build-and-publish.result == 'success' && needs.docker-build.result == 'success'
      run: |
        echo "Release ${{ github.ref_name }} published successfully!"
        
    - name: Notify failure
      if: needs.build-and-publish.result == 'failure' || needs.docker-build.result == 'failure'
      run: |
        echo "Release ${{ github.ref_name }} failed to publish!"
        exit 1
```

## 🔐 Secrets Configuration

Configure these secrets in your GitHub repository settings:

### Required Secrets

| Secret Name | Description | Required For |
|------------|-------------|--------------|
| `CODECOV_TOKEN` | Codecov integration token | Coverage reporting |
| `SNYK_TOKEN` | Snyk vulnerability scanning token | Security scanning |
| `PYPI_API_TOKEN` | PyPI publishing token | Package publishing |

### Optional Secrets

| Secret Name | Description | Use Case |
|------------|-------------|----------|
| `SLACK_WEBHOOK` | Slack notification webhook | Build notifications |
| `DISCORD_WEBHOOK` | Discord notification webhook | Release notifications |

### Setting Up Secrets

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each required secret with its corresponding value

## 📋 Workflow Setup Checklist

### Pre-setup
- [ ] Repository administrator access confirmed
- [ ] `workflows` permission available
- [ ] Required external service accounts created (Codecov, Snyk, etc.)

### Setup Steps
- [ ] Create `.github/workflows/` directory
- [ ] Add `ci.yml` workflow file
- [ ] Add `security.yml` workflow file  
- [ ] Add `release.yml` workflow file
- [ ] Configure all required secrets
- [ ] Test workflows with a test push/PR

### Post-setup Validation
- [ ] CI workflow runs successfully on push
- [ ] Security scans complete without blocking issues
- [ ] Test release workflow (create a test tag)
- [ ] Verify artifact uploads and reporting
- [ ] Check that branch protection rules work with required checks

## 🔧 Customization Options

### Adjusting Python Versions
Modify the matrix strategy in CI workflow:

```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]  # Add/remove versions
```

### Security Scan Frequency
Modify the cron schedule in security workflow:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
    # - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
```

### Branch Protection
Configure branch protection rules that require:
- CI workflow to pass
- Security scans to pass  
- Code review approval
- Up-to-date branches

## 🚨 Troubleshooting

### Common Issues

**Workflow not triggering:**
- Check branch name matches trigger conditions
- Verify workflow file syntax with GitHub Actions validator
- Ensure repository has Actions enabled

**Permission errors:**
- Verify `GITHUB_TOKEN` has required permissions
- Check if organization has workflow restrictions
- Confirm secrets are properly configured

**Build failures:**
- Check Python version compatibility
- Verify all dependencies are in requirements files
- Review test execution logs for specific failures

### Getting Help

1. Check GitHub Actions documentation
2. Review workflow run logs in the Actions tab
3. Validate YAML syntax using online validators
4. Test workflows in a fork first if unsure

This comprehensive workflow setup provides enterprise-grade CI/CD automation while maintaining security and reliability standards.