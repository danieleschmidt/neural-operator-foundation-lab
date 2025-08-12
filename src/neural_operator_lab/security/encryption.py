"""Encryption utilities for neural operator models and data."""

import os
import hashlib
import secrets
from typing import Optional, Dict, Any, Union
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption/decryption operations."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize encryption manager.
        
        Args:
            key: Optional encryption key. If None, generates a new key.
        """
        if key is None:
            self._key = Fernet.generate_key()
        else:
            self._key = key
        
        self._fernet = Fernet(self._key)
        logger.info("Encryption manager initialized")
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'EncryptionManager':
        """Create encryption manager from password.
        
        Args:
            password: Password to derive key from
            salt: Optional salt. If None, generates random salt.
            
        Returns:
            EncryptionManager instance
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        instance = cls(key)
        instance._salt = salt
        return instance
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self._fernet.encrypt(data)
        logger.debug("Data encrypted successfully")
        return encrypted
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data.
        
        Args:
            encrypted_data: Encrypted data as bytes
            
        Returns:
            Decrypted data as bytes
        """
        decrypted = self._fernet.decrypt(encrypted_data)
        logger.debug("Data decrypted successfully")
        return decrypted
    
    def decrypt_to_string(self, encrypted_data: bytes) -> str:
        """Decrypt data and return as string.
        
        Args:
            encrypted_data: Encrypted data as bytes
            
        Returns:
            Decrypted data as string
        """
        return self.decrypt(encrypted_data).decode('utf-8')
    
    def get_key(self) -> bytes:
        """Get the encryption key.
        
        Returns:
            Encryption key as bytes
        """
        return self._key
    
    def compute_hash(self, data: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex string of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()


class ModelEncryption:
    """Specialized encryption for neural operator models."""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        """Initialize model encryption.
        
        Args:
            encryption_manager: Optional encryption manager. Creates new if None.
        """
        self.encryption_manager = encryption_manager or EncryptionManager()
        self._model_metadata: Dict[str, Any] = {}
        logger.info("Model encryption initialized")
    
    def encrypt_model_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt model state dictionary.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Dictionary with encrypted model data
        """
        try:
            import torch
            import pickle
            
            # Serialize state dict
            serialized = pickle.dumps(state_dict)
            
            # Encrypt serialized data
            encrypted_data = self.encryption_manager.encrypt(serialized)
            
            # Compute hash for integrity
            data_hash = self.encryption_manager.compute_hash(serialized)
            
            encrypted_package = {
                'encrypted_data': encrypted_data,
                'data_hash': data_hash,
                'encryption_method': 'fernet',
                'metadata': self._model_metadata.copy()
            }
            
            logger.info("Model state encrypted successfully")
            return encrypted_package
            
        except ImportError:
            logger.error("PyTorch not available for model encryption")
            raise
        except Exception as e:
            logger.error(f"Model encryption failed: {e}")
            raise
    
    def decrypt_model_state(self, encrypted_package: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt model state dictionary.
        
        Args:
            encrypted_package: Encrypted model package
            
        Returns:
            Decrypted model state dictionary
        """
        try:
            import pickle
            
            encrypted_data = encrypted_package['encrypted_data']
            expected_hash = encrypted_package['data_hash']
            
            # Decrypt data
            decrypted_data = self.encryption_manager.decrypt(encrypted_data)
            
            # Verify integrity
            actual_hash = self.encryption_manager.compute_hash(decrypted_data)
            if actual_hash != expected_hash:
                raise ValueError("Data integrity check failed - hash mismatch")
            
            # Deserialize state dict
            state_dict = pickle.loads(decrypted_data)
            
            # Store metadata
            self._model_metadata = encrypted_package.get('metadata', {})
            
            logger.info("Model state decrypted successfully")
            return state_dict
            
        except Exception as e:
            logger.error(f"Model decryption failed: {e}")
            raise
    
    def save_encrypted_model(self, model_state: Dict[str, Any], filepath: str) -> None:
        """Save encrypted model to file.
        
        Args:
            model_state: Model state dictionary
            filepath: File path to save to
        """
        try:
            import torch
            
            encrypted_package = self.encrypt_model_state(model_state)
            torch.save(encrypted_package, filepath)
            
            logger.info(f"Encrypted model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save encrypted model: {e}")
            raise
    
    def load_encrypted_model(self, filepath: str) -> Dict[str, Any]:
        """Load encrypted model from file.
        
        Args:
            filepath: File path to load from
            
        Returns:
            Decrypted model state dictionary
        """
        try:
            import torch
            
            encrypted_package = torch.load(filepath, map_location='cpu')
            state_dict = self.decrypt_model_state(encrypted_package)
            
            logger.info(f"Encrypted model loaded from {filepath}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to load encrypted model: {e}")
            raise
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set model metadata.
        
        Args:
            metadata: Model metadata dictionary
        """
        self._model_metadata.update(metadata)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Model metadata dictionary
        """
        return self._model_metadata.copy()


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        URL-safe base64 encoded token
    """
    token_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(token_bytes).decode('ascii')


def secure_compare(a: str, b: str) -> bool:
    """Perform timing-safe string comparison.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal, False otherwise
    """
    return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))