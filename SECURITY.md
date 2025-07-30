# Security Policy

## Supported Versions

We actively support the following versions of Neural Operator Foundation Lab:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 Private Reporting (Preferred)

1. **Email us directly**: security@terragon.ai
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes (optional)

### 📋 Information to Include

- **Type of vulnerability** (code execution, data exposure, etc.)
- **Location** (file path, function name, line number)
- **Affected versions**
- **Prerequisites** for exploitation
- **Proof of concept** (if safe to share)

### ⏱️ Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Resolution target**: 30 days for critical issues

### 🛡️ Security Measures

#### Data Protection
- **No sensitive data logging** in production
- **Secure model storage** with encryption at rest
- **Input validation** for all user-provided data
- **Sandboxed execution** for user code/models

#### Dependency Security
- **Automated vulnerability scanning** via GitHub Dependabot
- **Regular dependency updates**
- **Pinned versions** for reproducibility
- **Security-focused dependency selection**

#### Infrastructure
- **Secure defaults** in all configurations
- **Principle of least privilege**
- **Regular security updates**
- **Monitoring and alerting**

## 🔍 Known Security Considerations

### Model Loading
- **Pickle security**: We use safe loading mechanisms
- **Model validation**: All models are validated before execution
- **Resource limits**: Memory and computation limits enforced

### Data Processing
- **Input sanitization**: All data inputs are validated
- **File type restrictions**: Only approved file formats accepted
- **Size limits**: Protection against resource exhaustion

### Dependencies
- **PyTorch**: Keep updated for security patches
- **NumPy/SciPy**: Regular updates for numerical stability
- **H5PY**: Secure file format handling

## 🚨 Security Best Practices

### For Users
```python
# ✅ Good: Use official model loading
from neural_operator_lab import load_model
model = load_model("path/to/model.pt", safe=True)

# ❌ Avoid: Direct pickle loading
import pickle
model = pickle.load(open("model.pkl", "rb"))  # Potential security risk
```

### For Contributors
- **Never commit secrets** (API keys, passwords)
- **Use secure randomness** (`secrets` module, not `random`)
- **Validate all inputs** before processing
- **Follow secure coding practices**

### For Deployments
```bash
# Use security-focused installation
pip install neural-operator-foundation-lab[security]

# Enable security features
export NEURAL_OP_SECURE_MODE=true
export NEURAL_OP_VALIDATE_MODELS=true
```

## 🔧 Security Configuration

### Environment Variables
```bash
# Security settings
NEURAL_OP_SECURE_MODE=true          # Enable security mode
NEURAL_OP_VALIDATE_MODELS=true      # Validate all model files
NEURAL_OP_MAX_MODEL_SIZE=1GB        # Limit model file sizes
NEURAL_OP_SANDBOX_EXECUTION=true    # Sandbox user code
```

### Secure Configuration Example
```python
import neural_operator_lab as nol

# Configure security settings
nol.configure_security(
    validate_models=True,
    max_model_size="1GB",
    allowed_file_types=[".pt", ".pth", ".h5"],
    sandbox_execution=True
)
```

## 📚 Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/security.html)
- [Secure AI Development](https://owasp.org/www-project-top-10-for-llm-applications/)

## 🏆 Acknowledgments

We thank the security research community for responsible disclosure of vulnerabilities. Contributors to our security will be acknowledged (with permission) in our release notes.

## 📞 Contact

- **Security Team**: security@terragon.ai
- **General Issues**: Use GitHub Issues (for non-security bugs)
- **Emergency Contact**: security+urgent@terragon.ai

---

**Note**: For non-security related bugs, please use our standard [GitHub Issues](https://github.com/terragon-labs/neural-operator-foundation-lab/issues).