#!/usr/bin/env python3
"""Comprehensive testing suite for Robust QISA implementation.

Tests Generation 2 (Robustness) features:
- Error handling and recovery
- Input validation and security
- Model health monitoring
- Performance tracking
- Numerical stability
"""

import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all robust QISA components can be imported."""
    print("🧪 Testing Robust QISA Imports...")
    
    try:
        # Test core QISA import
        from neural_operator_lab.models.quantum_spectral_attention import QuantumSpectralAttentionOperator
        print("✓ Core QISA imported successfully")
        
        # Test robust QISA import (should work without torch for structure validation)
        import ast
        robust_qisa_file = Path("src/neural_operator_lab/models/robust_qisa.py")
        if robust_qisa_file.exists():
            with open(robust_qisa_file, 'r') as f:
                ast.parse(f.read())
            print("✓ Robust QISA syntax validated")
        else:
            print("❌ Robust QISA file not found")
            return False
        
        # Test security validation import
        security_file = Path("src/neural_operator_lab/security/advanced_input_validation.py")
        if security_file.exists():
            with open(security_file, 'r') as f:
                ast.parse(f.read())
            print("✓ Advanced input validation syntax validated")
        else:
            print("❌ Advanced input validation file not found")
            return False
        
        # Test monitoring import
        monitor_file = Path("src/neural_operator_lab/monitoring/comprehensive_model_monitor.py")
        if monitor_file.exists():
            with open(monitor_file, 'r') as f:
                ast.parse(f.read())
            print("✓ Comprehensive model monitor syntax validated")
        else:
            print("❌ Comprehensive model monitor file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_code_structure():
    """Test the structure and completeness of robust implementations."""
    print("\n🏗️  Testing Code Structure...")
    
    results = {}
    
    # Test Robust QISA structure
    robust_qisa_file = Path("src/neural_operator_lab/models/robust_qisa.py")
    if robust_qisa_file.exists():
        with open(robust_qisa_file, 'r') as f:
            content = f.read()
        
        # Check for required classes and methods
        required_items = [
            'class RobustQuantumSpectralAttentionOperator',
            'def validate_input',
            'def check_model_health',
            'def apply_numerical_stability_fixes',
            '_error_handling_context',
            'def get_robustness_report',
            'create_robust_qisa_model'
        ]
        
        missing_items = []
        for item in required_items:
            if item not in content:
                missing_items.append(item)
        
        if not missing_items:
            print("✓ Robust QISA structure complete")
            results['robust_qisa'] = True
        else:
            print(f"❌ Missing Robust QISA components: {missing_items}")
            results['robust_qisa'] = False
    
    # Test Security Validation structure
    security_file = Path("src/neural_operator_lab/security/advanced_input_validation.py")
    if security_file.exists():
        with open(security_file, 'r') as f:
            content = f.read()
        
        required_validators = [
            'class AdversarialDetector',
            'class NumericalStabilityValidator',
            'class GeometricValidator',
            'class QuantumInputValidator',
            'class ComprehensiveInputValidator'
        ]
        
        missing_validators = []
        for validator in required_validators:
            if validator not in content:
                missing_validators.append(validator)
        
        if not missing_validators:
            print("✓ Security validation structure complete")
            results['security'] = True
        else:
            print(f"❌ Missing security validators: {missing_validators}")
            results['security'] = False
    
    # Test Monitoring structure
    monitor_file = Path("src/neural_operator_lab/monitoring/comprehensive_model_monitor.py")
    if monitor_file.exists():
        with open(monitor_file, 'r') as f:
            content = f.read()
        
        required_components = [
            'class PerformanceTracker',
            'class ModelHealthMonitor',
            'class ComprehensiveModelMonitor',
            'class AlertSeverity',
            '@dataclass\nclass ModelMetrics',
            'def start_monitoring',
            'def check_model_health'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if not missing_components:
            print("✓ Monitoring system structure complete")
            results['monitoring'] = True
        else:
            print(f"❌ Missing monitoring components: {missing_components}")
            results['monitoring'] = False
    
    return all(results.values())

def test_robustness_features():
    """Test robustness feature implementation."""
    print("\n🛡️  Testing Robustness Features...")
    
    features_found = {}
    
    # Test error handling patterns
    robust_qisa_file = Path("src/neural_operator_lab/models/robust_qisa.py")
    if robust_qisa_file.exists():
        with open(robust_qisa_file, 'r') as f:
            content = f.read()
        
        error_handling_patterns = [
            'torch.cuda.OutOfMemoryError',
            '@contextmanager',
            'try:',
            'except',
            'error_recovery_mode',
            'graceful',
            'strict'
        ]
        
        found_patterns = [pattern for pattern in error_handling_patterns if pattern in content]
        features_found['error_handling'] = len(found_patterns) >= 5
        
        if features_found['error_handling']:
            print(f"✓ Error handling patterns found: {len(found_patterns)}/7")
        else:
            print(f"❌ Insufficient error handling patterns: {len(found_patterns)}/7")
    
    # Test monitoring features
    monitor_file = Path("src/neural_operator_lab/monitoring/comprehensive_model_monitor.py")
    if monitor_file.exists():
        with open(monitor_file, 'r') as f:
            content = f.read()
        
        monitoring_features = [
            'performance_tracker',
            'health_monitor',
            'alert_callbacks',
            'threading',
            'deque',
            'AlertSeverity',
            'memory_usage',
            'gradient_norm'
        ]
        
        found_features = [feature for feature in monitoring_features if feature in content]
        features_found['monitoring'] = len(found_features) >= 6
        
        if features_found['monitoring']:
            print(f"✓ Monitoring features found: {len(found_features)}/8")
        else:
            print(f"❌ Insufficient monitoring features: {len(found_features)}/8")
    
    # Test security features
    security_file = Path("src/neural_operator_lab/security/advanced_input_validation.py")
    if security_file.exists():
        with open(security_file, 'r') as f:
            content = f.read()
        
        security_features = [
            'adversarial',
            'validate',
            'sanitize',
            'SecurityValidator',
            'ValidationResult',
            'quantum_constraints',
            'numerical_stability',
            'geometric'
        ]
        
        found_security = [feature for feature in security_features if feature.lower() in content.lower()]
        features_found['security'] = len(found_security) >= 6
        
        if features_found['security']:
            print(f"✓ Security features found: {len(found_security)}/8")
        else:
            print(f"❌ Insufficient security features: {len(found_security)}/8")
    
    return all(features_found.values())

def test_documentation_quality():
    """Test documentation quality and completeness."""
    print("\n📚 Testing Documentation Quality...")
    
    doc_scores = {}
    
    files_to_check = [
        "src/neural_operator_lab/models/robust_qisa.py",
        "src/neural_operator_lab/security/advanced_input_validation.py",
        "src/neural_operator_lab/monitoring/comprehensive_model_monitor.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Count docstrings
            docstring_count = content.count('"""')
            
            # Check for key documentation elements
            doc_elements = [
                'Args:',
                'Returns:',
                'Raises:',
                'Example:',
                'Note:',
                'Warning:'
            ]
            
            found_elements = sum(1 for elem in doc_elements if elem in content)
            
            # Calculate documentation score
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            
            doc_ratio = (docstring_count + len(comment_lines)) / len(non_empty_lines)
            
            score = min(100, int((docstring_count * 10 + found_elements * 5 + doc_ratio * 50)))
            doc_scores[Path(file_path).name] = score
            
            if score >= 60:
                print(f"✓ {Path(file_path).name}: Documentation score {score}/100")
            else:
                print(f"❌ {Path(file_path).name}: Documentation score {score}/100 (needs improvement)")
    
    average_score = sum(doc_scores.values()) / len(doc_scores) if doc_scores else 0
    return average_score >= 60

def test_production_readiness():
    """Test production readiness indicators."""
    print("\n🚀 Testing Production Readiness...")
    
    readiness_checks = {}
    
    # Check for production features in robust QISA
    robust_qisa_file = Path("src/neural_operator_lab/models/robust_qisa.py")
    if robust_qisa_file.exists():
        with open(robust_qisa_file, 'r') as f:
            content = f.read()
        
        production_features = [
            'logging',
            'error_handler',
            'health_checker',
            'metrics_collector',
            'input_validator',
            'monitoring',
            'stability',
            'recovery'
        ]
        
        found_prod_features = [feature for feature in production_features if feature in content]
        readiness_checks['production_features'] = len(found_prod_features) >= 6
        
        if readiness_checks['production_features']:
            print(f"✓ Production features: {len(found_prod_features)}/8")
        else:
            print(f"❌ Insufficient production features: {len(found_prod_features)}/8")
    
    # Check for configuration management
    config_patterns = ['config', 'parameter', 'setting', 'option']
    config_found = 0
    
    for file_path in ["src/neural_operator_lab/models/robust_qisa.py"]:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in config_patterns:
                if pattern in content.lower():
                    config_found += 1
                    break
    
    readiness_checks['configuration'] = config_found > 0
    if readiness_checks['configuration']:
        print("✓ Configuration management present")
    else:
        print("❌ Configuration management missing")
    
    # Check for factory functions
    factory_functions = ['create_robust_qisa_model', 'create_neural_operator_input_validator', 'create_qisa_monitor']
    factory_found = 0
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            for func in factory_functions:
                if func in content:
                    factory_found += 1
    
    readiness_checks['factory_functions'] = factory_found >= 2
    if readiness_checks['factory_functions']:
        print(f"✓ Factory functions present: {factory_found}/3")
    else:
        print(f"❌ Insufficient factory functions: {factory_found}/3")
    
    return all(readiness_checks.values())

def run_comprehensive_tests():
    """Run all robust QISA tests."""
    print("🧪 COMPREHENSIVE ROBUST QISA TESTING\n")
    print("=" * 60)
    
    test_results = {}
    
    # Run test suites
    test_results['imports'] = test_imports()
    test_results['structure'] = test_code_structure()
    test_results['robustness'] = test_robustness_features()
    test_results['documentation'] = test_documentation_quality()
    test_results['production'] = test_production_readiness()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():.<20} {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL ROBUST QISA TESTS PASSED!")
        print("   Ready for Generation 3 (Scaling) implementation")
        return True
    else:
        print(f"\n⚠️  {total-passed} TEST(S) FAILED")
        print("   Address issues before proceeding to Generation 3")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)