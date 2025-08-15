"""Syntax validation script for new neural operator implementations."""

import ast
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(source_code)
        
        # Count classes and functions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        return True, {
            'classes': classes,
            'functions': functions,
            'lines': len(source_code.splitlines())
        }
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Run syntax validation on new neural operator files."""
    print("🔍 Syntax Validation for New Neural Operators")
    print("=" * 60)
    
    files_to_check = [
        'src/neural_operator_lab/models/geometric.py',
        'src/neural_operator_lab/models/physics_informed.py',
        'src/neural_operator_lab/models/adaptive.py',
        'tests/unit/test_geometric_operators.py',
        'tests/unit/test_physics_informed_operators.py',
        'tests/unit/test_adaptive_operators.py'
    ]
    
    total_classes = 0
    total_functions = 0
    total_lines = 0
    valid_files = 0
    
    for file_path in files_to_check:
        full_path = Path(file_path)
        
        if not full_path.exists():
            print(f"❌ {file_path} - File not found")
            continue
        
        is_valid, result = validate_python_syntax(full_path)
        
        if is_valid:
            print(f"✅ {file_path}")
            print(f"   📋 Classes: {len(result['classes'])}")
            print(f"   🔧 Functions: {len(result['functions'])}")
            print(f"   📄 Lines: {result['lines']}")
            
            if result['classes']:
                print(f"   🏗️  Key Classes: {', '.join(result['classes'][:5])}")
            
            total_classes += len(result['classes'])
            total_functions += len(result['functions'])
            total_lines += result['lines']
            valid_files += 1
        else:
            print(f"❌ {file_path} - {result}")
        
        print()
    
    print("📊 Summary Statistics")
    print("-" * 30)
    print(f"📁 Valid files: {valid_files}/{len(files_to_check)}")
    print(f"🏗️  Total classes: {total_classes}")
    print(f"🔧 Total functions: {total_functions}")
    print(f"📄 Total lines: {total_lines:,}")
    
    # Check for key neural operator classes
    print("\n🧬 Key Neural Operator Classes Check")
    print("-" * 40)
    
    key_files = [
        ('src/neural_operator_lab/models/geometric.py', 'GeometricNeuralOperator'),
        ('src/neural_operator_lab/models/physics_informed.py', 'PhysicsInformedUniversalNeuralOperator'),
        ('src/neural_operator_lab/models/adaptive.py', 'MultiResolutionAdaptiveNeuralOperator')
    ]
    
    implemented_operators = 0
    
    for file_path, expected_class in key_files:
        full_path = Path(file_path)
        if full_path.exists():
            is_valid, result = validate_python_syntax(full_path)
            if is_valid and expected_class in result['classes']:
                print(f"✅ {expected_class} - Implemented")
                implemented_operators += 1
            else:
                print(f"❌ {expected_class} - Missing or invalid")
        else:
            print(f"❌ {expected_class} - File not found")
    
    print(f"\n🎯 Implementation Coverage: {implemented_operators}/{len(key_files)} ({implemented_operators/len(key_files)*100:.1f}%)")
    
    if valid_files == len(files_to_check) and implemented_operators == len(key_files):
        print("\n🎉 SYNTAX VALIDATION: ✅ EXCELLENT")
        print("   • All files have valid Python syntax")
        print("   • All key neural operator classes implemented")
        print("   • Ready for runtime testing when dependencies available")
        return True
    else:
        print("\n⚠️  SYNTAX VALIDATION: ⚠️  NEEDS ATTENTION")
        print("   • Some files have syntax errors or missing implementations")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)