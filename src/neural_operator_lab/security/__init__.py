"""Security utilities and enhancements."""

# Core security that doesn't depend on external libraries
try:
    from .enhanced_security import (
        SecurityConfig, SecureModelLoader, InputSanitizer,
        OutputSanitizer, SecureTemporaryFiles, SecurityAuditor
    )
    _HAS_ENHANCED_SECURITY = True
    __all__ = [
        'SecurityConfig', 'SecureModelLoader', 'InputSanitizer',
        'OutputSanitizer', 'SecureTemporaryFiles', 'SecurityAuditor'
    ]
except ImportError:
    _HAS_ENHANCED_SECURITY = False
    __all__ = []

# Traditional security modules (require torch)
try:
    from .input_validation import InputValidator, SecureModelLoader as TorchSecureModelLoader
    from .encryption import EncryptionManager, ModelEncryption
    from .audit import AuditLogger, SecurityAuditor as TorchSecurityAuditor
    
    __all__.extend([
        'InputValidator', 'TorchSecureModelLoader', 
        'EncryptionManager', 'ModelEncryption',
        'AuditLogger', 'TorchSecurityAuditor'
    ])
except ImportError:
    pass