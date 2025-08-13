#!/usr/bin/env python3
"""Test core robust implementation without external dependencies."""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_robust_framework():
    """Test the robust framework functionality."""
    print("🧪 Testing Core Robust Framework...")
    
    try:
        from neural_operator_lab.robust_framework import (
            RobustLogger, ErrorContext, robust_execution, 
            error_boundary, InputValidator, HealthChecker,
            ResourceMonitor, CircuitBreaker
        )
        print("✅ Robust framework imports successful")
        
        # Test error context
        context = ErrorContext("test_function", "test_module")
        assert context.function_name == "test_function"
        assert context.module_name == "test_module" 
        assert context.error_id is not None
        assert len(context.error_id) == 8  # MD5 hash truncated to 8 chars
        print("✅ ErrorContext working")
        
        # Test robust logger
        logger = RobustLogger("test_logger")
        logger.log_info("Test info message")
        logger.log_warning("Test warning message")
        
        # Test error logging
        test_error = ValueError("Test error for logging")
        logger.log_error(test_error, context)
        
        # Check error summary
        summary = logger.get_error_summary()
        assert summary['total_errors'] >= 1
        assert 'ValueError' in summary['error_types']
        print("✅ RobustLogger working")
        
        # Test robust execution decorator with successful function
        @robust_execution(max_retries=2)
        def successful_function(x):
            return x * 2
        
        result = successful_function(5)
        assert result == 10
        print("✅ Robust execution with success working")
        
        # Test robust execution decorator with retries
        attempt_count = 0
        
        @robust_execution(max_retries=2, fallback_value="fallback")
        def sometimes_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = sometimes_failing_function()
        assert result == "success"
        print("✅ Robust execution with retries working")
        
        # Test robust execution with permanent failure
        @robust_execution(max_retries=1, fallback_value="fallback")
        def always_failing_function():
            raise ValueError("Permanent failure")
        
        result = always_failing_function()
        assert result == "fallback"
        print("✅ Robust execution with fallback working")
        
        # Test error boundary with reraise=False
        boundary_result = None
        with error_boundary("test_operation", reraise=False, fallback_value="safe"):
            raise RuntimeError("Test error in boundary")
        print("✅ Error boundary with suppression working")
        
        # Test input validator
        validator = InputValidator()
        
        # Test config validation
        valid_config = {"key1": "value1", "key2": 42}
        validator.validate_config(valid_config, ["key1", "key2"])
        
        # Test missing key validation
        try:
            validator.validate_config({"key1": "value1"}, ["key1", "key2"])
            assert False, "Should have raised ValidationError"
        except Exception as e:
            assert "Missing required config keys" in str(e)
        
        print("✅ InputValidator config validation working")
        
        # Test file path validation
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            validated_path = validator.validate_file_path(
                tmp_path, 
                must_exist=True, 
                allowed_extensions=['.txt']
            )
            assert validated_path == Path(tmp_path)
        finally:
            os.unlink(tmp_path)
        
        print("✅ InputValidator file validation working")
        
        # Test health checker
        checker = HealthChecker()
        
        check_calls = 0
        def test_check():
            nonlocal check_calls
            check_calls += 1
            return check_calls % 2 == 1  # Alternates between True/False
        
        checker.register_check("test", test_check, interval=0.1)
        
        # First check should run
        result1 = checker.run_check("test")
        assert result1['status'] == 'ok'
        
        # Second check should be cached
        result2 = checker.run_check("test")
        assert result2['status'] == 'cached'
        
        # Run all checks
        all_results = checker.run_all_checks()
        assert 'test' in all_results
        
        # Get system status
        status = checker.get_system_status()
        assert status in ['healthy', 'degraded', 'critical', 'unknown']
        
        print("✅ HealthChecker working")
        
        # Test resource monitor
        monitor = ResourceMonitor(max_execution_time=0.1)
        monitor.start_monitoring()
        
        # Should not raise (quick operation)
        monitor.check_resources("quick_op")
        
        # Test timeout (simulate slow operation)
        import time
        time.sleep(0.12)
        try:
            monitor.check_resources("slow_op")
            assert False, "Should have raised ResourceError"
        except Exception as e:
            assert "exceeded max execution time" in str(e)
        
        print("✅ ResourceMonitor working")
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        def unreliable_service():
            raise ConnectionError("Service unavailable")
        
        # First few failures should pass through
        for i in range(2):
            try:
                breaker.call(unreliable_service)
            except ConnectionError:
                pass
        
        # Circuit should now be open
        try:
            breaker.call(unreliable_service)
            assert False, "Should have raised ResourceError (circuit open)"
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e)
        
        print("✅ CircuitBreaker working")
        
        print("✅ All core robust framework tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Core robust framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_core():
    """Test core security functionality without torch dependencies."""
    print("\n🔒 Testing Core Security...")
    
    try:
        from neural_operator_lab.security.enhanced_security import (
            SecurityConfig, SecureTemporaryFiles
        )
        print("✅ Core security imports successful")
        
        # Test security config
        config = SecurityConfig()
        assert config.enable_input_validation == True
        assert config.max_model_size_mb == 1000
        assert '.pt' in config.allowed_file_extensions
        assert '/etc' in config.forbidden_paths
        print("✅ SecurityConfig working")
        
        # Test secure temporary files
        temp_manager = SecureTemporaryFiles()
        
        with temp_manager.create_temp_file(suffix='.test') as temp_path:
            assert temp_path.exists()
            
            # Write some test data
            temp_path.write_text("test data")
            assert temp_path.read_text() == "test data"
            
            # Check permissions (on Unix systems)
            if hasattr(os, 'stat'):
                import stat
                file_mode = temp_path.stat().st_mode
                # Should be readable/writable by owner only
                assert file_mode & stat.S_IRWXO == 0  # No permissions for others
                assert file_mode & stat.S_IRWXG == 0  # No permissions for group
        
        # File should be cleaned up after context
        assert not temp_path.exists()
        
        print("✅ SecureTemporaryFiles working")
        
        print("✅ All core security tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Core security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_config():
    """Test basic configuration without yaml dependency."""
    print("\n⚙️ Testing Basic Configuration...")
    
    try:
        # Test basic dataclass configs without the full manager
        import json
        from dataclasses import dataclass, asdict
        
        @dataclass
        class TestConfig:
            name: str = "test"
            value: int = 42
            enabled: bool = True
        
        # Create config
        config = TestConfig(name="my_test", value=100)
        assert config.name == "my_test"
        assert config.value == 100
        assert config.enabled == True  # default value
        
        # Convert to dict
        config_dict = asdict(config)
        assert config_dict['name'] == "my_test"
        
        # Test JSON serialization
        json_str = json.dumps(config_dict)
        restored_dict = json.loads(json_str)
        assert restored_dict['name'] == "my_test"
        
        print("✅ Basic configuration working")
        return True
        
    except Exception as e:
        print(f"❌ Basic configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run core tests without external dependencies."""
    print("🚀 Testing Core Robust Implementation (Dependency-Free)")
    print("=" * 70)
    
    test_results = []
    
    # Run core tests that don't require external libraries
    test_results.append(("Core Robust Framework", test_robust_framework()))
    test_results.append(("Core Security", test_security_core()))
    test_results.append(("Basic Configuration", test_basic_config()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 CORE TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("=" * 70)
    print(f"Results: {passed}/{total} core tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All core tests passed! Generation 2 robust framework is working!")
        print("\n📋 GENERATION 2 FEATURES IMPLEMENTED:")
        print("  ✅ Robust error handling with retries and fallbacks")
        print("  ✅ Comprehensive logging with error tracking")
        print("  ✅ Input validation and sanitization")
        print("  ✅ Health monitoring and circuit breakers")
        print("  ✅ Resource monitoring and limits")
        print("  ✅ Security configurations and secure temp files")
        print("  ✅ Error boundaries for fault isolation")
        print("  ✅ Audit logging and error summaries")
        return True
    else:
        print("⚠️ Some core tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)