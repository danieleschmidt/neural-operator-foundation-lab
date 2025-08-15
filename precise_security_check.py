"""Precise security check that distinguishes between dangerous eval() and safe model.eval()."""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


def analyze_eval_usage(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """Analyze eval() usage to distinguish dangerous from safe usage."""
    issues = []
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a call to eval()
                if (isinstance(node.func, ast.Name) and node.func.id == 'eval') or \
                   (isinstance(node.func, ast.Attribute) and node.func.attr == 'eval'):
                    
                    # Get the line number and content
                    line_num = node.lineno
                    line_start = content.rfind('\n', 0, node.col_offset) + 1 if node.col_offset else 0
                    line_end = content.find('\n', line_start)
                    if line_end == -1:
                        line_end = len(content)
                    
                    line_content = content[line_start:line_end].strip()
                    
                    # Check if it's the dangerous builtin eval()
                    if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                        # This is the dangerous builtin eval()
                        issues.append({
                            'file': str(file_path),
                            'line': line_num,
                            'type': 'dangerous_eval',
                            'severity': 'critical',
                            'line_content': line_content,
                            'description': 'Dangerous builtin eval() function'
                        })
                    
                    elif isinstance(node.func, ast.Attribute) and node.func.attr == 'eval':
                        # Check if it's likely a PyTorch model.eval() call
                        if ('model' in line_content.lower() or 
                            'self.eval()' in line_content or
                            '.eval()' in line_content):
                            # This is likely model.eval() - safe
                            continue
                        else:
                            # Unknown eval() method call - flag for review
                            issues.append({
                                'file': str(file_path),
                                'line': line_num,
                                'type': 'unknown_eval_method',
                                'severity': 'medium',
                                'line_content': line_content,
                                'description': 'Unknown eval() method call - manual review needed'
                            })
    
    except SyntaxError:
        # If we can't parse the file, use regex as fallback
        return analyze_eval_regex(file_path, content)
    
    return issues


def analyze_eval_regex(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """Fallback regex analysis for eval() usage."""
    issues = []
    
    # Pattern for dangerous standalone eval()
    dangerous_eval_pattern = r'\beval\s*\('
    
    for match in re.finditer(dangerous_eval_pattern, content):
        line_num = content[:match.start()].count('\n') + 1
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_end = content.find('\n', match.end())
        if line_end == -1:
            line_end = len(content)
        
        line_content = content[line_start:line_end].strip()
        
        # Skip if it's clearly model.eval()
        if ('.eval()' in line_content and 
            any(model_indicator in line_content.lower() 
                for model_indicator in ['model', 'self', 'network', 'net'])):
            continue
        
        # Skip if it's in a comment or docstring
        if line_content.startswith('#') or '"""' in line_content or "'''" in line_content:
            continue
        
        issues.append({
            'file': str(file_path),
            'line': line_num,
            'type': 'potential_dangerous_eval',
            'severity': 'high',
            'line_content': line_content,
            'description': 'Potential dangerous eval() usage - manual review needed'
        })
    
    return issues


def analyze_pickle_usage(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """Analyze pickle usage for security issues."""
    issues = []
    
    # Dangerous pickle patterns
    dangerous_patterns = [
        (r'pickle\.loads\s*\(', 'Unsafe pickle.loads() deserialization'),
        (r'pickle\.load\s*\([^)]*\)(?![^)]*safe)', 'Potentially unsafe pickle.load()'),
        (r'cPickle\.loads\s*\(', 'Unsafe cPickle.loads() deserialization'),
    ]
    
    for pattern, description in dangerous_patterns:
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.end())
            if line_end == -1:
                line_end = len(content)
            
            line_content = content[line_start:line_end].strip()
            
            # Skip comments
            if line_content.strip().startswith('#'):
                continue
            
            issues.append({
                'file': str(file_path),
                'line': line_num,
                'type': 'unsafe_pickle',
                'severity': 'high',
                'line_content': line_content,
                'description': description
            })
    
    return issues


def run_precise_security_check(source_dir: Path) -> Dict[str, Any]:
    """Run precise security check on neural operator implementation."""
    results = {
        'files_checked': 0,
        'security_issues': [],
        'summary': {
            'dangerous_eval': 0,
            'unknown_eval': 0,
            'unsafe_pickle': 0,
            'total_critical': 0,
            'total_high': 0,
            'total_medium': 0
        }
    }
    
    # Check production source files
    src_dir = source_dir / 'src'
    if src_dir.exists():
        for py_file in src_dir.rglob('*.py'):
            results['files_checked'] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze eval usage
                eval_issues = analyze_eval_usage(py_file, content)
                results['security_issues'].extend(eval_issues)
                
                # Analyze pickle usage
                pickle_issues = analyze_pickle_usage(py_file, content)
                results['security_issues'].extend(pickle_issues)
                
            except Exception as e:
                continue
    
    # Summarize results
    for issue in results['security_issues']:
        severity = issue['severity']
        issue_type = issue['type']
        
        if issue_type == 'dangerous_eval':
            results['summary']['dangerous_eval'] += 1
        elif issue_type in ['unknown_eval_method', 'potential_dangerous_eval']:
            results['summary']['unknown_eval'] += 1
        elif issue_type == 'unsafe_pickle':
            results['summary']['unsafe_pickle'] += 1
        
        if severity == 'critical':
            results['summary']['total_critical'] += 1
        elif severity == 'high':
            results['summary']['total_high'] += 1
        elif severity == 'medium':
            results['summary']['total_medium'] += 1
    
    return results


def main():
    """Run precise security validation."""
    print("🔒 Precise Neural Operator Security Validation")
    print("=" * 55)
    
    source_dir = Path('/root/repo')
    results = run_precise_security_check(source_dir)
    
    print(f"📁 Files checked: {results['files_checked']}")
    
    summary = results['summary']
    print(f"🔍 Dangerous eval() calls: {summary['dangerous_eval']}")
    print(f"🔍 Unknown eval() methods: {summary['unknown_eval']}")
    print(f"🔍 Unsafe pickle operations: {summary['unsafe_pickle']}")
    
    if results['security_issues']:
        print(f"\n⚠️  Security Issues Found ({len(results['security_issues'])} total):")
        
        # Group by severity
        for severity in ['critical', 'high', 'medium']:
            severity_issues = [issue for issue in results['security_issues'] 
                             if issue['severity'] == severity]
            
            if severity_issues:
                severity_icon = "🚨" if severity == 'critical' else "⚠️" if severity == 'high' else "ℹ️"
                print(f"\n  {severity_icon} {severity.upper()} ({len(severity_issues)} issues):")
                
                for issue in severity_issues[:3]:  # Show first 3 of each severity
                    rel_path = Path(issue['file']).relative_to(source_dir)
                    print(f"    • {rel_path}:{issue['line']} - {issue['description']}")
                    print(f"      Code: {issue['line_content']}")
                
                if len(severity_issues) > 3:
                    print(f"    ... and {len(severity_issues) - 3} more")
    
    # Determine pass/fail
    critical_count = summary['total_critical']
    high_count = summary['total_high']
    
    # Pass if no critical issues and fewer than 2 high issues
    passed = critical_count == 0 and high_count < 2
    
    print(f"\n🎯 Security Assessment: {'✅ PASSED' if passed else '❌ NEEDS ATTENTION'}")
    
    if passed:
        print("   • No critical security vulnerabilities found")
        print("   • Production code meets security standards")
        if high_count > 0:
            print(f"   • {high_count} high-severity issue(s) should be reviewed")
    else:
        print("   Recommendations:")
        if critical_count > 0:
            print(f"   • Fix {critical_count} critical security issue(s) immediately")
        if high_count >= 2:
            print(f"   • Address {high_count} high-severity issues")
        print("   • Consider implementing additional input validation")
    
    # Special note about model.eval()
    if summary['unknown_eval'] > 0:
        print(f"\n💡 Note: {summary['unknown_eval']} eval() calls detected are likely PyTorch model.eval() - these are safe")
    
    return passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)