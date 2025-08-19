"""Advanced security framework for neural operator training and inference."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Set
import hashlib
import hmac
import secrets
import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Comprehensive security configuration."""
    # Model security
    enable_model_encryption: bool = True
    enable_weight_integrity_checks: bool = True
    enable_gradient_privacy: bool = True
    differential_privacy_epsilon: float = 1.0
    
    # Input validation
    enable_input_validation: bool = True
    max_input_size: int = 1024 * 1024 * 100  # 100MB
    allowed_input_dtypes: Set[str] = field(default_factory=lambda: {'float32', 'float64'})
    
    # Access control
    enable_rbac: bool = True
    require_api_keys: bool = True
    session_timeout: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    
    # Audit and monitoring
    enable_audit_logging: bool = True
    enable_anomaly_detection: bool = True
    log_all_operations: bool = True
    
    # Secure communication
    require_https: bool = True
    enable_mutual_tls: bool = False
    min_tls_version: str = "1.2"
    
    # Model serving security
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_request_signing: bool = True


class ModelEncryption:
    """Encrypt and decrypt model weights for secure storage and transmission."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or secrets.token_urlsafe(32)
        self._key = None
        
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        if self._key is None:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self._key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        return self._key
    
    def encrypt_model(self, model: nn.Module) -> bytes:
        """Encrypt model state dict."""
        try:
            # Serialize model state
            model_bytes = torch.save(model.state_dict(), buffer=True)
            
            # Generate salt and derive key
            salt = secrets.token_bytes(32)
            key = self._derive_key(salt)
            
            # Encrypt
            f = Fernet(key)
            encrypted_data = f.encrypt(model_bytes)
            
            # Prepend salt to encrypted data
            return salt + encrypted_data
            
        except Exception as e:
            logger.error(f"Model encryption failed: {e}")
            raise SecurityException(f"Model encryption failed: {e}")
    
    def decrypt_model(self, encrypted_data: bytes, model: nn.Module) -> nn.Module:
        """Decrypt and load model state dict."""
        try:
            # Extract salt and encrypted data
            salt = encrypted_data[:32]
            encrypted_model_data = encrypted_data[32:]
            
            # Derive key and decrypt
            key = self._derive_key(salt)
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_model_data)
            
            # Load model state
            state_dict = torch.load(decrypted_data, map_location='cpu')
            model.load_state_dict(state_dict)
            
            return model
            
        except Exception as e:
            logger.error(f"Model decryption failed: {e}")
            raise SecurityException(f"Model decryption failed: {e}")


class WeightIntegrityChecker:
    """Check model weight integrity to detect tampering."""
    
    def __init__(self):
        self.weight_hashes = {}
        
    def compute_model_hash(self, model: nn.Module) -> str:
        """Compute cryptographic hash of model weights."""
        hash_obj = hashlib.sha256()
        
        # Sort parameters by name for consistent hashing
        for name, param in sorted(model.named_parameters()):
            if param.requires_grad:
                # Convert to bytes and hash
                param_bytes = param.data.detach().cpu().numpy().tobytes()
                hash_obj.update(f"{name}:".encode())
                hash_obj.update(param_bytes)
        
        return hash_obj.hexdigest()
    
    def register_model(self, model: nn.Module, model_id: str):
        """Register model weights for integrity checking."""
        model_hash = self.compute_model_hash(model)
        self.weight_hashes[model_id] = {
            'hash': model_hash,
            'timestamp': time.time(),
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        logger.info(f"Model {model_id} registered with hash: {model_hash[:16]}...")
    
    def verify_integrity(self, model: nn.Module, model_id: str) -> bool:
        """Verify model integrity against registered hash."""
        if model_id not in self.weight_hashes:
            logger.warning(f"Model {model_id} not registered for integrity checking")
            return False
        
        current_hash = self.compute_model_hash(model)
        expected_hash = self.weight_hashes[model_id]['hash']
        
        if current_hash == expected_hash:
            logger.debug(f"Model {model_id} integrity verified")
            return True
        else:
            logger.error(f"Model {model_id} integrity check failed!")
            logger.error(f"Expected: {expected_hash[:16]}...")
            logger.error(f"Got: {current_hash[:16]}...")
            return False


class DifferentialPrivacy:
    """Implement differential privacy for gradient-based training."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier for given privacy parameters."""
        # Simplified computation - in practice, use more sophisticated methods
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to bound sensitivity."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model: nn.Module, batch_size: int):
        """Add calibrated noise to gradients."""
        noise_scale = self.clip_norm * self.noise_multiplier / batch_size
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_scale, size=param.grad.shape, 
                                   device=param.grad.device, dtype=param.grad.dtype)
                param.grad.data.add_(noise)
    
    def get_privacy_spent(self, steps: int, batch_size: int, dataset_size: int) -> float:
        """Compute privacy budget spent."""
        # Simplified computation
        sampling_rate = batch_size / dataset_size
        return steps * sampling_rate * self.epsilon


class InputValidator:
    """Validate and sanitize inputs to prevent attacks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validation_rules = {
            'max_size': config.max_input_size,
            'allowed_dtypes': config.allowed_input_dtypes,
            'shape_constraints': [(1, 1024, 1024, 16)],  # (min_batch, max_h, max_w, max_channels)
        }
        
    def validate_tensor_input(self, tensor: torch.Tensor, input_name: str = "input") -> bool:
        """Validate tensor input for security."""
        try:
            # Check data type
            if str(tensor.dtype).replace('torch.', '') not in self.config.allowed_input_dtypes:
                raise SecurityException(f"Invalid dtype for {input_name}: {tensor.dtype}")
            
            # Check size
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size > self.config.max_input_size:
                raise SecurityException(f"Input {input_name} too large: {tensor_size} bytes")
            
            # Check for NaN/Inf
            if torch.isnan(tensor).any():
                raise SecurityException(f"NaN values detected in {input_name}")
            
            if torch.isinf(tensor).any():
                raise SecurityException(f"Infinite values detected in {input_name}")
            
            # Check value ranges (basic sanity check)
            if tensor.dtype in [torch.float32, torch.float64]:
                if tensor.abs().max() > 1e6:
                    logger.warning(f"Large values detected in {input_name}: {tensor.abs().max().item()}")
            
            # Check shape constraints
            if len(tensor.shape) > 4:  # Typically batch, height, width, channels
                raise SecurityException(f"Too many dimensions in {input_name}: {len(tensor.shape)}")
            
            return True
            
        except SecurityException:
            raise
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            raise SecurityException(f"Input validation failed for {input_name}")
    
    def sanitize_string_input(self, text: str) -> str:
        """Sanitize string inputs to prevent injection attacks."""
        # Remove potentially dangerous characters
        dangerous_patterns = [
            r'[<>"\']',  # HTML/XML tags and quotes
            r'\\x[0-9a-fA-F]{2}',  # Hex escape sequences
            r'\\[0-7]{1,3}',  # Octal escape sequences
            r'[\x00-\x1f\x7f-\x9f]',  # Control characters
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized)
        
        # Limit length
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized


class AccessController:
    """Role-based access control system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.users = {}
        self.roles = {
            'admin': {'permissions': {'all'}},
            'researcher': {'permissions': {'train', 'infer', 'view_metrics'}},
            'viewer': {'permissions': {'view_metrics'}},
        }
        self.sessions = {}
        self.failed_attempts = {}
        
    def create_user(self, username: str, role: str, api_key: Optional[str] = None) -> str:
        """Create new user with specified role."""
        if role not in self.roles:
            raise SecurityException(f"Invalid role: {role}")
        
        if api_key is None:
            api_key = secrets.token_urlsafe(32)
        
        # Hash the API key for storage
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.users[username] = {
            'role': role,
            'api_key_hash': api_key_hash,
            'created_at': time.time(),
            'last_login': None
        }
        
        logger.info(f"User {username} created with role {role}")
        return api_key
    
    def authenticate_user(self, username: str, api_key: str) -> bool:
        """Authenticate user with API key."""
        if username not in self.users:
            self._record_failed_attempt(username)
            return False
        
        # Check if account is locked due to failed attempts
        if self._is_account_locked(username):
            raise SecurityException(f"Account {username} is temporarily locked")
        
        # Verify API key
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if api_key_hash == self.users[username]['api_key_hash']:
            self.users[username]['last_login'] = time.time()
            self._reset_failed_attempts(username)
            return True
        else:
            self._record_failed_attempt(username)
            return False
    
    def create_session(self, username: str) -> str:
        """Create authenticated session."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'username': username,
            'created_at': time.time(),
            'expires_at': time.time() + self.config.session_timeout
        }
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username if valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if time.time() > session['expires_at']:
            del self.sessions[session_id]
            return None
        
        return session['username']
    
    def check_permission(self, username: str, operation: str) -> bool:
        """Check if user has permission for operation."""
        if username not in self.users:
            return False
        
        user_role = self.users[username]['role']
        role_permissions = self.roles[user_role]['permissions']
        
        return 'all' in role_permissions or operation in role_permissions
    
    def _record_failed_attempt(self, username: str):
        """Record failed authentication attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(time.time())
        
        # Keep only recent attempts
        cutoff_time = time.time() - 3600  # 1 hour
        self.failed_attempts[username] = [
            t for t in self.failed_attempts[username] if t > cutoff_time
        ]
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = len(self.failed_attempts[username])
        return recent_attempts >= self.config.max_failed_attempts
    
    def _reset_failed_attempts(self, username: str):
        """Reset failed attempts counter after successful login."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup separate audit logger
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_operation(self, username: str, operation: str, resource: str, 
                     success: bool, details: Optional[Dict[str, Any]] = None):
        """Log security-relevant operation."""
        log_entry = {
            'timestamp': time.time(),
            'username': username,
            'operation': operation,
            'resource': resource,
            'success': success,
            'ip_address': 'localhost',  # Would be actual IP in real deployment
            'details': details or {}
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, details: Optional[Dict[str, Any]] = None):
        """Log security event."""
        log_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'details': details or {}
        }
        
        self.audit_logger.warning(json.dumps(log_entry))


class SecurityException(Exception):
    """Custom exception for security-related errors."""
    pass


class SecureNeuralOperator:
    """Secure wrapper for neural operator models."""
    
    def __init__(self, model: nn.Module, config: SecurityConfig):
        self.model = model
        self.config = config
        
        # Initialize security components
        self.encryptor = ModelEncryption() if config.enable_model_encryption else None
        self.integrity_checker = WeightIntegrityChecker() if config.enable_weight_integrity_checks else None
        self.dp_engine = DifferentialPrivacy(config.differential_privacy_epsilon) if config.enable_gradient_privacy else None
        self.input_validator = InputValidator(config) if config.enable_input_validation else None
        self.access_controller = AccessController(config) if config.enable_rbac else None
        self.audit_logger = AuditLogger() if config.enable_audit_logging else None
        
        # Register model for integrity checking
        if self.integrity_checker:
            self.integrity_checker.register_model(model, "main_model")
    
    def secure_forward(self, x: torch.Tensor, username: str, session_id: str) -> torch.Tensor:
        """Secure forward pass with all security checks."""
        try:
            # Validate session
            if self.access_controller:
                validated_user = self.access_controller.validate_session(session_id)
                if validated_user != username:
                    raise SecurityException("Invalid session")
                
                if not self.access_controller.check_permission(username, 'infer'):
                    raise SecurityException("Insufficient permissions for inference")
            
            # Validate input
            if self.input_validator:
                self.input_validator.validate_tensor_input(x, "model_input")
            
            # Check model integrity
            if self.integrity_checker:
                if not self.integrity_checker.verify_integrity(self.model, "main_model"):
                    raise SecurityException("Model integrity check failed")
            
            # Perform inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
            
            # Log operation
            if self.audit_logger:
                self.audit_logger.log_operation(
                    username, "inference", "main_model", True,
                    {"input_shape": list(x.shape), "output_shape": list(output.shape)}
                )
            
            return output
            
        except SecurityException as e:
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "access_violation", "high", str(e),
                    {"username": username, "operation": "inference"}
                )
            raise
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_operation(
                    username, "inference", "main_model", False,
                    {"error": str(e)}
                )
            raise
    
    def secure_train_step(self, batch: Any, loss_fn: Callable, optimizer: torch.optim.Optimizer,
                         username: str, session_id: str) -> Dict[str, float]:
        """Secure training step with privacy protection."""
        try:
            # Security checks
            if self.access_controller:
                validated_user = self.access_controller.validate_session(session_id)
                if validated_user != username:
                    raise SecurityException("Invalid session")
                
                if not self.access_controller.check_permission(username, 'train'):
                    raise SecurityException("Insufficient permissions for training")
            
            # Validate inputs
            if self.input_validator:
                self.input_validator.validate_tensor_input(batch[0], "training_input")
                self.input_validator.validate_tensor_input(batch[1], "training_target")
            
            # Training step
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(batch[0])
            loss = loss_fn(outputs, batch[1])
            loss.backward()
            
            # Apply differential privacy
            if self.dp_engine:
                gradient_norm = self.dp_engine.clip_gradients(self.model)
                self.dp_engine.add_noise_to_gradients(self.model, batch[0].size(0))
            
            optimizer.step()
            
            # Log training step
            if self.audit_logger:
                self.audit_logger.log_operation(
                    username, "training", "main_model", True,
                    {"loss": loss.item(), "batch_size": batch[0].size(0)}
                )
            
            return {"loss": loss.item()}
            
        except SecurityException as e:
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "access_violation", "high", str(e),
                    {"username": username, "operation": "training"}
                )
            raise
    
    def save_secure_model(self, filepath: str, username: str, session_id: str):
        """Save model with encryption."""
        try:
            # Security checks
            if self.access_controller:
                validated_user = self.access_controller.validate_session(session_id)
                if validated_user != username:
                    raise SecurityException("Invalid session")
                
                if not self.access_controller.check_permission(username, 'save_model'):
                    raise SecurityException("Insufficient permissions to save model")
            
            if self.encryptor:
                encrypted_data = self.encryptor.encrypt_model(self.model)
                with open(filepath, 'wb') as f:
                    f.write(encrypted_data)
            else:
                torch.save(self.model.state_dict(), filepath)
            
            # Log save operation
            if self.audit_logger:
                self.audit_logger.log_operation(
                    username, "save_model", filepath, True
                )
            
        except SecurityException as e:
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "access_violation", "high", str(e),
                    {"username": username, "operation": "save_model"}
                )
            raise


def create_secure_neural_operator(
    model: nn.Module,
    enable_encryption: bool = True,
    enable_integrity_checks: bool = True,
    enable_differential_privacy: bool = True,
    epsilon: float = 1.0,
    enable_rbac: bool = True
) -> SecureNeuralOperator:
    """Create a secure neural operator with specified security features."""
    config = SecurityConfig(
        enable_model_encryption=enable_encryption,
        enable_weight_integrity_checks=enable_integrity_checks,
        enable_gradient_privacy=enable_differential_privacy,
        differential_privacy_epsilon=epsilon,
        enable_rbac=enable_rbac
    )
    
    return SecureNeuralOperator(model, config)