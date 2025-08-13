#!/usr/bin/env python3
"""Test robust implementation without external dependencies."""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_robust_framework():
    """Test the robust framework functionality."""
    print("🧪 Testing Robust Framework...")
    
    try:
        from neural_operator_lab.robust_framework import (
            RobustLogger, ErrorContext, robust_execution, 
            error_boundary, InputValidator, HealthChecker
        )
        print("✅ Robust framework imports successful")
        
        # Test error context
        context = ErrorContext("test_function", "test_module")
        assert context.function_name == "test_function"
        assert context.module_name == "test_module"
        assert context.error_id is not None
        print("✅ ErrorContext working")
        
        # Test robust logger
        logger = RobustLogger("test_logger")
        logger.log_info("Test info message")
        logger.log_warning("Test warning message")
        print("✅ RobustLogger working")
        
        # Test robust execution decorator
        @robust_execution(max_retries=2, fallback_value="fallback")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "fallback"
        print("✅ Robust execution with fallback working")
        
        # Test error boundary
        with error_boundary("test_operation", reraise=False, fallback_value="safe"):
            raise RuntimeError("Test error in boundary")
        print("✅ Error boundary working")
        
        # Test input validator
        class MockTensor:
            def __init__(self, shape):
                self.shape = shape
                self.dtype = "float32"
            
            def isnan(self):
                return MockTensor((False,))
            
            def any(self):
                return False
            
            def isinf(self):
                return MockTensor((False,))
            
            def min(self):
                return 0.0
            
            def max(self):
                return 1.0
        
        mock_tensor = MockTensor((10, 10))
        InputValidator.validate_tensor_input(
            mock_tensor, "test_tensor", expected_dims=2
        )
        print("✅ InputValidator working")
        
        # Test health checker
        checker = HealthChecker()
        
        def test_check():
            return True
        
        checker.register_check("test", test_check, interval=1.0)
        result = checker.run_check("test")
        assert result['status'] == 'ok'
        print("✅ HealthChecker working")
        
        print("✅ All robust framework tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Robust framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_framework():
    """Test the security framework functionality."""
    print("\n🔒 Testing Security Framework...")
    
    try:
        from neural_operator_lab.security.enhanced_security import (
            SecurityConfig, SecureModelLoader, InputSanitizer,
            OutputSanitizer, secure_execution, SecurityAuditor
        )
        print("✅ Security framework imports successful")
        
        # Test security config
        config = SecurityConfig()
        assert config.enable_input_validation == True
        assert config.max_model_size_mb == 1000
        print("✅ SecurityConfig working")
        
        # Test secure model loader
        loader = SecureModelLoader(config)
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp.write(b"test model data")
            tmp_path = tmp.name
        
        try:
            # Test file validation
            is_valid = loader.validate_model_file(tmp_path)
            assert is_valid == True
            print("✅ Model file validation working")
        finally:
            os.unlink(tmp_path)
        
        # Test input sanitizer
        sanitizer = InputSanitizer(config)
        
        class MockTensor:
            def __init__(self, data):
                self.data = data
                self.shape = (10, 10)
            
            def isnan(self):
                return MockTensor([False])
            
            def any(self):
                return False
            
            def isinf(self):
                return MockTensor([False])
            
            def abs(self):
                return MockTensor([1.0])
            
            def max(self):
                return MockTensor(1.0)
            
            def item(self):
                return 1.0
        
        mock_tensor = MockTensor([1, 2, 3])
        sanitized = sanitizer.sanitize_tensor_input(mock_tensor, "test")
        assert sanitized is not None
        print("✅ InputSanitizer working")
        
        # Test config sanitization
        test_config = {"learning_rate": 0.001, "batch_size": 32}
        sanitized_config = sanitizer.sanitize_config_input(test_config)
        assert "learning_rate" in sanitized_config
        print("✅ Config sanitization working")
        
        # Test security auditor
        auditor = SecurityAuditor(config)
        
        class MockModel:
            def __init__(self):
                pass
            
            def parameters(self):
                return [MockParam() for _ in range(10)]
        
        class MockParam:
            def numel(self):
                return 1000
        
        mock_model = MockModel()
        audit_result = auditor.audit_model_security(mock_model)
        assert audit_result['score'] >= 0
        print("✅ SecurityAuditor working")
        
        print("✅ All security framework tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Security framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_manager():
    """Test the configuration manager."""
    print("\n⚙️ Testing Configuration Manager...")
    
    try:
        from neural_operator_lab.config_manager import (
            ConfigManager, ModelConfig, TrainingConfig, 
            ExperimentConfig, create_config
        )
        print("✅ Config manager imports successful")
        
        # Test model config
        model_config = ModelConfig(
            model_type="transformer",
            input_dim=3,
            output_dim=3,
            hidden_dim=128
        )
        assert model_config.model_type == "transformer"
        assert model_config.hidden_dim == 128
        print("✅ ModelConfig working")
        
        # Test training config
        training_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=16,
            max_epochs=50
        )
        assert training_config.learning_rate == 1e-3
        print("✅ TrainingConfig working")
        
        # Test experiment config
        exp_config = ExperimentConfig(
            name="test_experiment",
            model=model_config,
            training=training_config
        )
        assert exp_config.name == "test_experiment"
        assert exp_config.model.hidden_dim == 128
        print("✅ ExperimentConfig working")
        
        # Test config manager
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(config_dir=Path(temp_dir))
            
            # Save and load config
            saved_path = manager.save_config(exp_config, "test_config.json")
            assert saved_path.exists()
            
            loaded_config = manager.load_config("test_config.json", "experiment")
            assert loaded_config.name == "test_experiment"
            print("✅ Config save/load working")
            
            # Test template creation
            manager.create_template("test_template", exp_config)
            template_config = manager.load_from_template("test_template", name="from_template")
            assert template_config.name == "from_template"
            print("✅ Config templates working")
        
        print("✅ All config manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Config manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_foundation_models():
    """Test foundation models (without torch dependencies)."""
    print("\n🧠 Testing Foundation Models (Structure Only)...")
    
    try:
        # Test that imports work and classes are defined
        from neural_operator_lab.models.foundation import (
            UniversalPhysicsConfig, PhysicsEmbedding
        )
        print("✅ Foundation model imports successful")
        
        # Test config
        config = UniversalPhysicsConfig(
            d_model=256,
            num_layers=4,
            num_heads=8
        )
        assert config.d_model == 256
        assert config.num_layers == 4
        print("✅ UniversalPhysicsConfig working")
        
        print("✅ Foundation models structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Foundation models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_module():
    """Test validation module (structure only)."""
    print("\n🔍 Testing Validation Module (Structure Only)...")
    
    try:
        from neural_operator_lab.validation import ValidationMetrics
        print("✅ Validation module imports successful")
        
        # Test validation metrics
        metrics = ValidationMetrics(
            mse=0.001,
            relative_l2=0.05,
            max_error=0.1,
            spectral_error=0.02
        )
        
        metrics_dict = metrics.to_dict()
        assert 'mse' in metrics_dict
        assert metrics_dict['mse'] == 0.001
        print("✅ ValidationMetrics working")
        
        print("✅ Validation module structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Robust Neural Operator Implementation")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Robust Framework", test_robust_framework()))
    test_results.append(("Security Framework", test_security_framework()))
    test_results.append(("Config Manager", test_config_manager()))
    test_results.append(("Foundation Models", test_foundation_models()))
    test_results.append(("Validation Module", test_validation_module()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Robust implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)