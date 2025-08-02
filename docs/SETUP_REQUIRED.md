# Manual Setup Required

Due to GitHub App permission limitations, the following items require manual setup by repository maintainers.

## 🔧 Required GitHub Actions Workflows

Copy the workflow files from `docs/workflows/examples/` to `.github/workflows/`:

### Core Workflows
- [ ] `ci.yml` - Continuous Integration with testing across Python versions
- [ ] `security.yml` - Comprehensive security scanning
- [ ] `performance.yml` - Performance benchmarking and regression detection
- [ ] `docs.yml` - Documentation building and deployment
- [ ] `release.yml` - Automated release and PyPI publishing

### Command to Copy Workflows
```bash
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

## 🔑 Required GitHub Secrets

Add these secrets in GitHub repository settings (`Settings > Secrets and variables > Actions`):

### Essential Secrets
- [ ] `CODECOV_TOKEN` - For code coverage reporting
- [ ] `PYPI_API_TOKEN` - For automated PyPI publishing

### Optional Secrets (for enhanced features)
- [ ] `DOCKER_HUB_USERNAME` - For Docker image publishing
- [ ] `DOCKER_HUB_ACCESS_TOKEN` - For Docker Hub authentication
- [ ] `SLACK_WEBHOOK_URL` - For Slack notifications
- [ ] `TEAMS_WEBHOOK_URL` - For Microsoft Teams notifications

## 🛡️ Branch Protection Rules

Configure branch protection for the `main` branch (`Settings > Branches`):

### Required Settings
- [ ] **Require pull request reviews before merging**
  - Required approving reviews: 1
  - Dismiss stale reviews when new commits are pushed
  - Require review from code owners

- [ ] **Require status checks to pass before merging**
  - Require branches to be up to date before merging
  - Required checks:
    - [ ] `test (ubuntu-latest, 3.9)`
    - [ ] `test (ubuntu-latest, 3.10)`
    - [ ] `test (ubuntu-latest, 3.11)`
    - [ ] `integration-test`
    - [ ] `security / Dependency Security Scan`
    - [ ] `security / Code Security Scan`

- [ ] **Restrict pushes that create files**
  - Only allow merge commits
  - Restrict pushes to matching branches

## 📚 GitHub Pages Setup

Enable GitHub Pages for documentation hosting (`Settings > Pages`):

- [ ] **Source**: Deploy from a branch
- [ ] **Branch**: `gh-pages` (will be created by workflow)
- [ ] **Folder**: `/ (root)`

## 🏷️ Repository Settings

Update repository settings (`Settings > General`):

### Basic Information
- [ ] **Description**: "A training & benchmarking suite for foundation neural operators that tackle high-dimensional PDEs"
- [ ] **Website**: `https://neural-operator-lab.readthedocs.io`
- [ ] **Topics**: `neural-operators`, `pde`, `machine-learning`, `physics`, `deep-learning`, `scientific-computing`

### Features
- [ ] **Issues**: Enabled
- [ ] **Wiki**: Disabled (using docs/ instead)
- [ ] **Discussions**: Enabled (for community Q&A)
- [ ] **Projects**: Enabled

### Pull Requests
- [ ] **Allow merge commits**: Enabled
- [ ] **Allow squash merging**: Enabled
- [ ] **Allow rebase merging**: Disabled
- [ ] **Always suggest updating pull request branches**: Enabled
- [ ] **Automatically delete head branches**: Enabled

## 🔒 Security Settings

Configure security settings (`Settings > Security`):

### Code Security and Analysis
- [ ] **Dependency graph**: Enabled
- [ ] **Dependabot alerts**: Enabled
- [ ] **Dependabot security updates**: Enabled
- [ ] **Code scanning**: Enabled (CodeQL will be set up by security workflow)
- [ ] **Secret scanning**: Enabled

### Dependabot Version Updates
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 5
```

## 📋 Issue Templates

GitHub issue templates are provided in `.github/ISSUE_TEMPLATE/`. Ensure they are properly configured:

- [ ] Bug report template
- [ ] Feature request template
- [ ] Documentation improvement template
- [ ] Performance issue template

## 👥 Team and Collaborator Setup

### Required Teams (if using GitHub organization)
- [ ] **Core maintainers**: Full access to repository
- [ ] **Contributors**: Write access for regular contributors
- [ ] **Triagers**: Triage access for issue management

### CODEOWNERS Configuration
The `CODEOWNERS` file should be created to automatically request reviews:

```
# Global owners
* @core-maintainer-1 @core-maintainer-2

# Documentation
/docs/ @docs-team
/README.md @docs-team

# Security-sensitive files
/pyproject.toml @security-team
/.github/ @devops-team
/Dockerfile @devops-team

# Core neural operator code
/src/neural_operator_lab/models/ @ml-team
/src/neural_operator_lab/training/ @ml-team

# Tests
/tests/ @qa-team
```

## 🚀 First-Time Setup Commands

After completing the above manual steps, run these commands to verify setup:

```bash
# 1. Test workflow syntax
gh workflow list

# 2. Trigger initial CI run
git checkout -b test/setup-verification
git commit --allow-empty -m "test: verify CI/CD setup"
git push origin test/setup-verification

# 3. Create test PR to verify all checks
gh pr create --title "Test: CI/CD Setup Verification" --body "Testing automated workflows"

# 4. Check security scanning
gh api repos/:owner/:repo/code-scanning/alerts

# 5. Verify branch protection is working
git checkout main
git push origin main  # Should fail if protection is working
```

## ✅ Verification Checklist

After manual setup, verify the following:

### Workflows
- [ ] CI workflow runs successfully on PR
- [ ] Security scans complete without blocking errors
- [ ] Documentation builds and deploys
- [ ] Performance benchmarks execute (may be slow first time)

### Branch Protection
- [ ] Cannot push directly to main branch
- [ ] PR requires review and passing checks
- [ ] Status checks are enforced

### Security
- [ ] Dependabot creates security update PRs
- [ ] Code scanning alerts appear in Security tab
- [ ] Secret scanning is active

### Documentation
- [ ] GitHub Pages site is accessible
- [ ] Documentation auto-updates on main branch changes

## 🆘 Troubleshooting

### Common Issues

**Workflow Permission Errors**
- Ensure `GITHUB_TOKEN` has sufficient permissions
- Check repository settings for action permissions

**Secret Not Found Errors**
- Verify secrets are added with exact names
- Check secret scope (repository vs organization)

**Branch Protection Bypass**
- Ensure protection rules apply to administrators
- Check for bypass permissions in organization settings

**Documentation Not Deploying**
- Verify GitHub Pages source is set correctly
- Check workflow logs for build errors

### Getting Help

If you encounter issues during setup:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow run logs in the Actions tab
3. Create an issue using the provided templates
4. Contact the development team via discussions

## 📅 Maintenance Schedule

After initial setup, maintain the following schedule:

### Weekly
- [ ] Review Dependabot PRs
- [ ] Check security alerts

### Monthly  
- [ ] Update action versions
- [ ] Review performance benchmarks
- [ ] Audit access permissions

### Quarterly
- [ ] Review and update branch protection rules
- [ ] Audit team permissions
- [ ] Update documentation

---

**Setup Status**: ⏳ Manual setup required  
**Priority**: High  
**Estimated Time**: 30-60 minutes  
**Required Permissions**: Repository admin  

*This document will be updated as GitHub App permissions are expanded or setup requirements change.*