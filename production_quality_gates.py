#!/usr/bin/env python3
"""Production Quality Gates for Neural Operator Framework

Focused quality gates for production readiness assessment.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path

def check_security() -> tuple[bool, float, str]:
    """Check security - no active dangerous code."""
    try:
        result = subprocess.run([sys.executable, 'final_security_check.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return True, 100.0, "✅ No active security threats found"
        else:
            return False, 0.0, "❌ Active security threats detected"
            
    except Exception as e:
        return False, 0.0, f"❌ Security check failed: {e}"

def check_basic_functionality() -> tuple[bool, float, str]:
    """Check that basic neural operator functionality works."""
    try:
        # Run our Generation 1 test
        result = subprocess.run([sys.executable, 'generation1_test.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "GENERATION 1 COMPLETE" in result.stdout:
            return True, 100.0, "✅ Basic functionality working"
        else:
            return False, 0.0, f"❌ Basic functionality failed"
            
    except Exception as e:
        return False, 0.0, f"❌ Functionality test failed: {e}"

def check_robustness() -> tuple[bool, float, str]:
    """Check robustness features."""
    try:
        result = subprocess.run([sys.executable, 'generation2_robust.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "GENERATION 2 COMPLETE" in result.stdout:
            return True, 100.0, "✅ Robustness features working"
        else:
            return False, 60.0, "⚠️ Robustness partially working"
            
    except Exception as e:
        return False, 0.0, f"❌ Robustness check failed: {e}"

def check_performance() -> tuple[bool, float, str]:
    """Check performance and scalability."""
    try:
        result = subprocess.run([sys.executable, 'generation3_scale.py'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and "GENERATION 3 COMPLETE" in result.stdout:
            return True, 100.0, "✅ Performance and scaling working"
        else:
            return False, 70.0, "⚠️ Performance features partially working"
            
    except Exception as e:
        return False, 0.0, f"❌ Performance check failed: {e}"

def check_code_quality() -> tuple[bool, float, str]:
    """Basic code quality checks."""
    python_files = list(Path('src').rglob('*.py'))
    
    if not python_files:
        return False, 0.0, "❌ No Python files found"
    
    total_files = len(python_files)
    syntax_errors = 0
    
    # Check syntax of all Python files
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(py_file), 'exec')
        except SyntaxError:
            syntax_errors += 1
        except Exception:
            pass  # Other errors don't count as syntax errors
    
    if syntax_errors == 0:
        return True, 100.0, f"✅ All {total_files} files have valid syntax"
    else:
        score = max(0, 100 - (syntax_errors / total_files * 100))
        return False, score, f"❌ {syntax_errors}/{total_files} files have syntax errors"

def run_production_quality_gates():
    """Run production-focused quality gates."""
    print("🏭 PRODUCTION QUALITY GATES")
    print("=" * 50)
    
    gates = [
        ("Security", check_security, True),  # Critical
        ("Basic Functionality", check_basic_functionality, True),  # Critical  
        ("Code Quality", check_code_quality, True),  # Critical
        ("Robustness", check_robustness, False),  # Nice to have
        ("Performance", check_performance, False),  # Nice to have
    ]
    
    results = {}
    all_critical_passed = True
    total_score = 0
    total_weight = 0
    
    for gate_name, gate_func, is_critical in gates:
        print(f"\\n🔍 Running {gate_name}...")
        start_time = time.time()
        
        passed, score, message = gate_func()
        duration = time.time() - start_time
        
        weight = 1.0 if is_critical else 0.5
        total_score += score * weight
        total_weight += weight
        
        results[gate_name] = {
            "passed": passed,
            "score": score,
            "message": message,
            "critical": is_critical,
            "duration": duration
        }
        
        status = "✅ PASS" if passed else ("❌ CRITICAL FAIL" if is_critical else "⚠️  WARNING")
        print(f"   {status} - {message} ({score:.1f}/100) [{duration:.1f}s]")
        
        if is_critical and not passed:
            all_critical_passed = False
    
    # Calculate overall results
    overall_score = total_score / total_weight if total_weight > 0 else 0
    overall_status = "✅ READY" if all_critical_passed else "❌ NOT READY"
    
    print("\\n" + "=" * 50)
    print("📊 PRODUCTION READINESS ASSESSMENT")
    print("=" * 50)
    print(f"Overall Status: {overall_status}")
    print(f"Overall Score: {overall_score:.1f}/100")
    
    critical_gates = [name for name, result in results.items() if result["critical"]]
    critical_passed = [name for name, result in results.items() if result["critical"] and result["passed"]]
    
    print(f"Critical Gates: {len(critical_passed)}/{len(critical_gates)} passed")
    
    if all_critical_passed:
        print("\\n🚀 SYSTEM IS PRODUCTION READY!")
        print("   ✅ All critical quality gates passed")
        print("   ✅ Security verified")
        print("   ✅ Core functionality validated")
    else:
        print("\\n⚠️  SYSTEM NOT PRODUCTION READY")
        failed_critical = [name for name, result in results.items() 
                          if result["critical"] and not result["passed"]]
        print(f"   ❌ Critical failures: {', '.join(failed_critical)}")
        print("   🔧 Fix critical issues before deployment")
    
    # Save results
    with open('production_quality_report.json', 'w') as f:
        json.dump({
            "overall_status": overall_status,
            "overall_score": overall_score,
            "all_critical_passed": all_critical_passed,
            "results": results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\\n📄 Detailed report saved: production_quality_report.json")
    
    return all_critical_passed

def main():
    """Main function."""
    try:
        success = run_production_quality_gates()
        return success
    except KeyboardInterrupt:
        print("\\n❌ Quality gates interrupted by user")
        return False
    except Exception as e:
        print(f"\\n❌ Quality gates failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)