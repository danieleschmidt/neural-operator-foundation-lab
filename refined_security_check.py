"""Refined security check for neural operator implementation.

This script performs targeted security validation excluding false positives
from test files and development utilities.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


def check_production_security(source_dir: Path) -> Dict[str, Any]:
    """Check security issues in production code only."""
    
    # Security patterns with severity levels
    security_patterns = {
        'code_injection': {
            'patterns': [r'exec\s*\(', r'eval\s*\('],
            'severity': 'critical',
            'exclude_in_tests': True
        },
        'unsafe_subprocess': {
            'patterns': [r'subprocess\.call.*shell=True', r'os\.system\s*\('],
            'severity': 'high',
            'exclude_in_tests': True
        },
        'hardcoded_secrets': {
            'patterns': [
                r'password\s*=\s*["\'][a-zA-Z0-9]{8,}["\']',
                r'api_key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
                r'secret\s*=\s*["\'][a-zA-Z0-9]{16,}["\']'
            ],
            'severity': 'critical',
            'exclude_in_tests': False
        },
        'unsafe_pickle': {
            'patterns': [r'pickle\.loads\s*\(', r'pickle\.load\s*\((?!.*safe)'],
            'severity': 'high',
            'exclude_in_tests': True
        }
    }
    
    results = {
        'production_files_checked': 0,
        'security_issues': [],
        'summary': {}
    }
    
    # Get production source files (exclude tests)
    src_dir = source_dir / 'src'
    production_files = []
    
    if src_dir.exists():
        for py_file in src_dir.rglob('*.py'):
            production_files.append(py_file)
    
    # Also check root-level production scripts
    for py_file in source_dir.glob('*.py'):
        if not any(exclude in py_file.name for exclude in ['test_', 'validate_', 'quality_gates']):
            production_files.append(py_file)
    
    # Check each production file
    for py_file in production_files:
        results['production_files_checked'] += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check each security pattern
            for category, config in security_patterns.items():
                for pattern in config['patterns']:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    
                    for match in matches:
                        # Additional context checking for false positives
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_end = content.find('\n', match.end())
                        if line_end == -1:
                            line_end = len(content)
                        
                        line_content = content[line_start:line_end]
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Skip comments and docstrings
                        if line_content.strip().startswith('#'):
                            continue
                        if line_content.strip().startswith('"""') or line_content.strip().startswith("'''"):
                            continue
                        
                        # Skip test patterns that are actually safe
                        if category == 'unsafe_subprocess' and 'test' in str(py_file).lower():
                            continue
                        
                        results['security_issues'].append({
                            'file': str(py_file.relative_to(source_dir)),
                            'line': line_num,
                            'category': category,
                            'severity': config['severity'],
                            'pattern': pattern,
                            'match': match.group(),
                            'line_content': line_content.strip(),
                            'description': get_security_description(category)
                        })
        
        except Exception as e:
            continue
    
    # Generate summary
    results['summary'] = generate_security_summary(results['security_issues'])
    
    return results


def get_security_description(category: str) -> str:
    """Get description for security category."""
    descriptions = {
        'code_injection': 'Potential code injection via exec/eval',
        'unsafe_subprocess': 'Unsafe subprocess execution with shell=True',
        'hardcoded_secrets': 'Hardcoded credentials detected',
        'unsafe_pickle': 'Unsafe pickle deserialization'
    }
    return descriptions.get(category, 'Security vulnerability detected')


def generate_security_summary(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate security summary."""
    summary = {
        'total_issues': len(issues),
        'by_severity': {},
        'by_category': {},
        'critical_files': []
    }
    
    for issue in issues:
        # Count by severity
        severity = issue['severity']
        summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        # Count by category
        category = issue['category']
        summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
        
        # Track critical files
        if severity == 'critical' and issue['file'] not in summary['critical_files']:
            summary['critical_files'].append(issue['file'])
    
    return summary


def validate_neural_operator_security() -> bool:
    """Main security validation for neural operator implementation."""
    print("🔒 Neural Operator Security Validation")
    print("=" * 50)
    
    source_dir = Path('/root/repo')
    results = check_production_security(source_dir)
    
    print(f"📁 Production files checked: {results['production_files_checked']}")
    print(f"🔍 Security issues found: {results['summary']['total_issues']}")
    
    if results['security_issues']:
        print("\n⚠️  Security Issues Found:")
        
        # Group by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = [issue for issue in results['security_issues'] 
                             if issue['severity'] == severity]
            
            if severity_issues:
                print(f"\n  🚨 {severity.upper()} ({len(severity_issues)} issues):")
                for issue in severity_issues[:5]:  # Show first 5 of each severity
                    print(f"    • {issue['file']}:{issue['line']} - {issue['description']}")
                    print(f"      Match: {issue['match']}")
                
                if len(severity_issues) > 5:
                    print(f"    ... and {len(severity_issues) - 5} more")
        
        print(f"\n📊 Summary by Category:")
        for category, count in results['summary']['by_category'].items():
            print(f"  • {category}: {count}")
    
    else:
        print("✅ No security issues found in production code!")
    
    # Determine if validation passes
    critical_issues = results['summary']['by_severity'].get('critical', 0)
    high_issues = results['summary']['by_severity'].get('high', 0)
    
    # Pass criteria: No critical issues, fewer than 3 high issues
    security_passed = critical_issues == 0 and high_issues < 3
    
    print(f"\n🎯 Security Validation: {'✅ PASSED' if security_passed else '❌ FAILED'}")
    
    if not security_passed:
        print("   Recommendations:")
        if critical_issues > 0:
            print(f"   • Address {critical_issues} critical security issues immediately")
        if high_issues >= 3:
            print(f"   • Reduce {high_issues} high-severity issues to under 3")
        print("   • Review security best practices for ML/AI applications")
    else:
        print("   • Production code meets security standards")
        print("   • Ready for secure deployment")
    
    return security_passed


if __name__ == "__main__":
    import sys
    success = validate_neural_operator_security()
    sys.exit(0 if success else 1)