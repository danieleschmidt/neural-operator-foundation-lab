#!/usr/bin/env python3
"""Validation test for project structure and basic functionality without PyTorch dependencies."""

import os
import sys
import importlib
import ast
import json
from pathlib import Path


def test_project_structure():
    """Test that all required directories and files exist."""
    print("🏗️  Testing project structure...")
    
    required_dirs = [
        'src/neural_operator_lab',
        'src/neural_operator_lab/models',
        'src/neural_operator_lab/training',
        'src/neural_operator_lab/data',
        'tests',
        'examples',
        'docs'
    ]
    
    required_files = [
        'README.md',
        'pyproject.toml',
        'requirements.txt',
        'setup.py',
        'src/neural_operator_lab/__init__.py',
        'src/neural_operator_lab/base.py',
        'src/neural_operator_lab/utils.py',
        'src/neural_operator_lab/models/__init__.py',
        'src/neural_operator_lab/models/fourier.py',
        'src/neural_operator_lab/models/transformer.py',
        'src/neural_operator_lab/models/probabilistic.py',
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is complete")
    return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("🐍 Testing Python syntax...")
    
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Add examples and tests
    for root, dirs, files in os.walk('examples'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Add root level Python files
    for file in ['setup.py', 'test_generation1.py', 'test_structure_validation.py']:
        if os.path.exists(file):
            python_files.append(file)
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse to check syntax
            ast.parse(content, filename=file_path)
            
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
        except Exception as e:
            syntax_errors.append((file_path, f"Read error: {str(e)}"))
    
    if syntax_errors:
        print("❌ Syntax errors found:")
        for file_path, error in syntax_errors:
            print(f"   {file_path}: {error}")
        return False
    
    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True


def test_imports_structure():
    """Test that imports are structured correctly (without actually importing torch)."""
    print("📦 Testing import structure...")
    
    # Test main __init__.py
    init_file = 'src/neural_operator_lab/__init__.py'
    
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Check for key exports
        required_exports = ['FourierNeuralOperator', 'TransformerNeuralOperator', 'ProbabilisticNeuralOperator']
        
        for export in required_exports:
            if export not in content:
                print(f"❌ Missing export {export} in {init_file}")
                return False
        
        # Check for proper conditional imports
        if 'try:' not in content or 'ImportError' not in content:
            print(f"❌ {init_file} should have defensive imports")
            return False
        
        print("✅ Import structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False


def test_configuration_files():
    """Test that configuration files are valid."""
    print("⚙️  Testing configuration files...")
    
    # Test pyproject.toml exists and has required sections
    if not os.path.exists('pyproject.toml'):
        print("❌ pyproject.toml not found")
        return False
    
    # Test requirements.txt
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_deps = ['torch', 'numpy', 'scipy']
        for dep in required_deps:
            if dep not in requirements.lower():
                print(f"❌ Missing dependency {dep} in requirements.txt")
                return False
    
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False
    
    print("✅ Configuration files are valid")
    return True


def test_model_implementations():
    """Test that model files contain required classes."""
    print("🧠 Testing model implementations...")
    
    model_tests = [
        ('src/neural_operator_lab/models/fourier.py', ['FourierNeuralOperator', 'FNOBlock']),
        ('src/neural_operator_lab/models/transformer.py', ['TransformerNeuralOperator', 'TNOBlock']),
        ('src/neural_operator_lab/models/probabilistic.py', ['ProbabilisticNeuralOperator', 'BayesianLinear']),
    ]
    
    for file_path, required_classes in model_tests:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for class_name in required_classes:
                if f"class {class_name}" not in content:
                    print(f"❌ Missing class {class_name} in {file_path}")
                    return False
        
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False
    
    print("✅ All model implementations contain required classes")
    return True


def test_examples():
    """Test that example files exist and have proper structure."""
    print("📚 Testing example files...")
    
    example_files = [
        'examples/basic_fno_example.py',
        'examples/transformer_neural_operator_demo.py'
    ]
    
    for example_file in example_files:
        if not os.path.exists(example_file):
            print(f"❌ Missing example file: {example_file}")
            return False
        
        try:
            with open(example_file, 'r') as f:
                content = f.read()
            
            # Check for main execution
            if 'if __name__ == "__main__":' not in content:
                print(f"❌ Example {example_file} missing main execution block")
                return False
        
        except Exception as e:
            print(f"❌ Error reading {example_file}: {e}")
            return False
    
    print("✅ Example files are properly structured")
    return True


def test_utilities():
    """Test utility functions exist."""
    print("🔧 Testing utility functions...")
    
    util_file = 'src/neural_operator_lab/utils.py'
    
    try:
        with open(util_file, 'r') as f:
            content = f.read()
        
        required_classes = ['SpectralConv1d', 'SpectralConv2d', 'SpectralConv3d', 'FourierFeatures']
        
        for class_name in required_classes:
            if f"class {class_name}" not in content:
                print(f"❌ Missing utility class {class_name}")
                return False
        
        print("✅ All utility functions are implemented")
        return True
        
    except Exception as e:
        print(f"❌ Error reading {util_file}: {e}")
        return False


def run_all_structure_tests():
    """Run all structure validation tests."""
    print("🚀 Neural Operator Foundation Lab - Structure Validation")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Structure", test_imports_structure),
        ("Configuration Files", test_configuration_files),
        ("Model Implementations", test_model_implementations),
        ("Example Files", test_examples),
        ("Utility Functions", test_utilities),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            if results[test_name]:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Structure Validation Summary:")
    print(f"✅ Passed: {passed}/{len(tests)} tests")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    if passed == len(tests):
        print("\n🎉 All structure tests passed! Generation 1 implementation is complete.")
        print("📦 Ready for dependency installation and functional testing.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please fix structure issues.")
        return False


if __name__ == "__main__":
    success = run_all_structure_tests()
    sys.exit(0 if success else 1)