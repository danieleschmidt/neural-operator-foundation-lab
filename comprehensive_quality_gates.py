"""Comprehensive Quality Gates for Neural Operator Foundation Lab.

This script runs all quality gates including syntax validation, security checks,
performance analysis, and production readiness validation.
"""

import ast
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import re
import hashlib


class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, score: float, details: Dict[str, Any]):
        self.name = name
        self.passed = passed
        self.score = score  # 0.0 to 1.0
        self.details = details
        
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} (Score: {self.score:.2f})"


class SecurityGate:
    """Security validation gate."""
    
    def __init__(self):
        self.security_patterns = {
            'sql_injection': [
                r'exec\s*\(',
                r'eval\s*\(',
                r'subprocess\.call',
                r'os\.system',
                r'shell=True'
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'unsafe_deserialization': [
                r'pickle\.loads',
                r'pickle\.load',
                r'yaml\.load\(',
                r'marshal\.loads'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\\\',
                r'open\s*\(\s*["\'][^"\']*\.\.[^"\']*["\']'
            ]
        }
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run security validation."""
        results = {
            'files_checked': 0,
            'vulnerabilities': [],
            'security_score': 1.0
        }
        
        # Check all Python files
        for py_file in source_dir.rglob('*.py'):
            results['files_checked'] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_vulnerabilities = self._check_file_security(py_file, content)
                results['vulnerabilities'].extend(file_vulnerabilities)
                
            except Exception as e:
                results['vulnerabilities'].append({
                    'file': str(py_file),
                    'type': 'read_error',
                    'description': f"Could not read file: {e}"
                })
        
        # Calculate security score
        if results['vulnerabilities']:
            # Deduct points based on vulnerability severity
            severity_weights = {'high': 0.3, 'medium': 0.1, 'low': 0.05}
            total_deduction = sum(
                severity_weights.get(vuln.get('severity', 'medium'), 0.1)
                for vuln in results['vulnerabilities']
            )
            results['security_score'] = max(0.0, 1.0 - total_deduction)
        
        passed = results['security_score'] >= 0.8 and len(results['vulnerabilities']) == 0
        
        return QualityGateResult(
            name="Security Validation",
            passed=passed,
            score=results['security_score'],
            details=results
        )
    
    def _check_file_security(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Check security issues in a single file."""
        vulnerabilities = []
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    vulnerabilities.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': category,
                        'pattern': pattern,
                        'match': match.group(),
                        'severity': self._get_severity(category),
                        'description': self._get_description(category)
                    })
        
        return vulnerabilities
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category."""
        severity_map = {
            'sql_injection': 'high',
            'hardcoded_secrets': 'high',
            'unsafe_deserialization': 'high',
            'path_traversal': 'medium'
        }
        return severity_map.get(category, 'medium')
    
    def _get_description(self, category: str) -> str:
        """Get description for vulnerability category."""
        descriptions = {
            'sql_injection': 'Potential code injection vulnerability',
            'hardcoded_secrets': 'Hardcoded credentials detected',
            'unsafe_deserialization': 'Unsafe deserialization method',
            'path_traversal': 'Potential path traversal vulnerability'
        }
        return descriptions.get(category, 'Security issue detected')


class CodeQualityGate:
    """Code quality validation gate."""
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run code quality validation."""
        results = {
            'files_analyzed': 0,
            'total_lines': 0,
            'total_classes': 0,
            'total_functions': 0,
            'complexity_scores': [],
            'documentation_coverage': 0.0,
            'code_quality_score': 0.0
        }
        
        # Analyze all Python files
        for py_file in source_dir.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            file_analysis = self._analyze_file(py_file)
            if file_analysis:
                results['files_analyzed'] += 1
                results['total_lines'] += file_analysis['lines']
                results['total_classes'] += len(file_analysis['classes'])
                results['total_functions'] += len(file_analysis['functions'])
                results['complexity_scores'].extend(file_analysis['complexity_scores'])
                
                # Documentation coverage
                documented_items = (file_analysis['documented_classes'] + 
                                  file_analysis['documented_functions'])
                total_items = len(file_analysis['classes']) + len(file_analysis['functions'])
                
                if total_items > 0:
                    doc_coverage = documented_items / total_items
                    results['documentation_coverage'] += doc_coverage
        
        # Calculate averages
        if results['files_analyzed'] > 0:
            results['documentation_coverage'] /= results['files_analyzed']
            avg_complexity = sum(results['complexity_scores']) / len(results['complexity_scores']) if results['complexity_scores'] else 0
        else:
            avg_complexity = 0
        
        # Calculate overall code quality score
        doc_score = results['documentation_coverage']
        complexity_score = max(0, 1.0 - (avg_complexity - 5) / 20)  # Normalize complexity
        structure_score = min(1.0, (results['total_classes'] + results['total_functions']) / 100)
        
        results['code_quality_score'] = (doc_score * 0.4 + complexity_score * 0.4 + structure_score * 0.2)
        
        passed = (results['code_quality_score'] >= 0.7 and 
                 results['documentation_coverage'] >= 0.6 and
                 avg_complexity <= 15)
        
        return QualityGateResult(
            name="Code Quality",
            passed=passed,
            score=results['code_quality_score'],
            details=results
        )
    
    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'lines': len(content.splitlines()),
                'classes': [],
                'functions': [],
                'documented_classes': 0,
                'documented_functions': 0,
                'complexity_scores': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                    if ast.get_docstring(node):
                        analysis['documented_classes'] += 1
                    analysis['complexity_scores'].append(self._calculate_complexity(node))
                
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                    if ast.get_docstring(node):
                        analysis['documented_functions'] += 1
                    analysis['complexity_scores'].append(self._calculate_complexity(node))
            
            return analysis
            
        except Exception as e:
            return None
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity


class PerformanceGate:
    """Performance validation gate."""
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run performance validation."""
        results = {
            'performance_patterns': [],
            'optimization_opportunities': [],
            'performance_score': 0.8  # Default score
        }
        
        # Performance anti-patterns to check
        anti_patterns = {
            'nested_loops': r'for\s+\w+.*:\s*\n\s*for\s+\w+.*:',
            'global_variables': r'global\s+\w+',
            'inefficient_string_concat': r'\w+\s*\+=\s*["\']',
            'recursive_without_memoization': r'def\s+\w+.*:\s*.*\1\(',
            'large_data_structures': r'range\(\s*\d{6,}\s*\)',
        }
        
        pattern_count = 0
        total_files = 0
        
        for py_file in source_dir.rglob('*.py'):
            if 'test' in str(py_file):
                continue
                
            total_files += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern_name, pattern in anti_patterns.items():
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        pattern_count += len(matches)
                        results['performance_patterns'].append({
                            'file': str(py_file),
                            'pattern': pattern_name,
                            'count': len(matches)
                        })
                
                # Check for optimization opportunities
                optimization_checks = self._check_optimizations(content, py_file)
                results['optimization_opportunities'].extend(optimization_checks)
                
            except Exception as e:
                continue
        
        # Calculate performance score
        if total_files > 0:
            pattern_ratio = pattern_count / total_files
            results['performance_score'] = max(0.0, 1.0 - pattern_ratio * 0.1)
        
        passed = results['performance_score'] >= 0.7 and pattern_count < total_files * 2
        
        return QualityGateResult(
            name="Performance Analysis",
            passed=passed,
            score=results['performance_score'],
            details=results
        )
    
    def _check_optimizations(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for optimization opportunities."""
        opportunities = []
        
        # Check for missing torch.no_grad()
        if 'torch' in content and 'forward' in content:
            if 'torch.no_grad()' not in content and 'autocast' not in content:
                opportunities.append({
                    'file': str(file_path),
                    'type': 'missing_no_grad',
                    'description': 'Consider using torch.no_grad() for inference'
                })
        
        # Check for inefficient tensor operations
        if 'for' in content and ('tensor' in content or 'torch' in content):
            opportunities.append({
                'file': str(file_path),
                'type': 'potential_vectorization',
                'description': 'Check if loops can be vectorized with tensor operations'
            })
        
        return opportunities


class TestCoverageGate:
    """Test coverage validation gate."""
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run test coverage analysis."""
        results = {
            'test_files': 0,
            'source_files': 0,
            'test_functions': 0,
            'coverage_estimate': 0.0
        }
        
        # Count test files and functions
        test_dir = source_dir / 'tests'
        if test_dir.exists():
            for test_file in test_dir.rglob('test_*.py'):
                results['test_files'] += 1
                
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                            results['test_functions'] += 1
                
                except Exception:
                    continue
        
        # Count source files
        src_dir = source_dir / 'src'
        if src_dir.exists():
            for src_file in src_dir.rglob('*.py'):
                if '__init__.py' not in src_file.name:
                    results['source_files'] += 1
        
        # Estimate coverage
        if results['source_files'] > 0:
            results['coverage_estimate'] = min(1.0, results['test_functions'] / (results['source_files'] * 3))
        
        passed = (results['test_files'] >= 3 and 
                 results['test_functions'] >= 30 and
                 results['coverage_estimate'] >= 0.5)
        
        return QualityGateResult(
            name="Test Coverage",
            passed=passed,
            score=results['coverage_estimate'],
            details=results
        )


class DocumentationGate:
    """Documentation validation gate."""
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run documentation validation."""
        results = {
            'readme_exists': False,
            'api_docs_exist': False,
            'docstrings_coverage': 0.0,
            'documentation_score': 0.0
        }
        
        # Check for README
        readme_files = list(source_dir.glob('README.md')) + list(source_dir.glob('README.rst'))
        results['readme_exists'] = len(readme_files) > 0
        
        # Check for API documentation
        docs_dir = source_dir / 'docs'
        if docs_dir.exists():
            api_docs = list(docs_dir.rglob('*.md')) + list(docs_dir.rglob('*.rst'))
            results['api_docs_exist'] = len(api_docs) >= 3
        
        # Check docstring coverage
        total_functions = 0
        documented_functions = 0
        
        src_dir = source_dir / 'src'
        if src_dir.exists():
            for py_file in src_dir.rglob('*.py'):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                
                except Exception:
                    continue
        
        if total_functions > 0:
            results['docstrings_coverage'] = documented_functions / total_functions
        
        # Calculate documentation score
        readme_score = 0.3 if results['readme_exists'] else 0.0
        api_score = 0.3 if results['api_docs_exist'] else 0.0
        docstring_score = results['docstrings_coverage'] * 0.4
        
        results['documentation_score'] = readme_score + api_score + docstring_score
        
        passed = (results['readme_exists'] and
                 results['docstrings_coverage'] >= 0.6 and
                 results['documentation_score'] >= 0.7)
        
        return QualityGateResult(
            name="Documentation",
            passed=passed,
            score=results['documentation_score'],
            details=results
        )


class ProductionReadinessGate:
    """Production readiness validation gate."""
    
    def run(self, source_dir: Path) -> QualityGateResult:
        """Run production readiness validation."""
        results = {
            'dockerfile_exists': False,
            'requirements_exist': False,
            'config_management': False,
            'logging_implemented': False,
            'error_handling': False,
            'monitoring_ready': False,
            'production_score': 0.0
        }
        
        # Check for Docker support
        results['dockerfile_exists'] = (source_dir / 'Dockerfile').exists()
        
        # Check for requirements
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'poetry.lock']
        results['requirements_exist'] = any((source_dir / req).exists() for req in req_files)
        
        # Check for configuration management
        config_indicators = ['config.py', 'settings.py', 'config.yaml', 'config.json']
        for indicator in config_indicators:
            if list(source_dir.rglob(indicator)):
                results['config_management'] = True
                break
        
        # Check for logging
        logging_patterns = ['logging', 'logger', 'log_level']
        for py_file in source_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if any(pattern in content for pattern in logging_patterns):
                    results['logging_implemented'] = True
                    break
            except:
                continue
        
        # Check for error handling
        error_patterns = ['try:', 'except:', 'raise', 'Exception']
        error_count = 0
        total_files = 0
        
        for py_file in source_dir.rglob('*.py'):
            if 'test' in str(py_file):
                continue
                
            total_files += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if any(pattern in content for pattern in error_patterns):
                    error_count += 1
            except:
                continue
        
        if total_files > 0:
            results['error_handling'] = error_count / total_files >= 0.5
        
        # Check for monitoring
        monitoring_indicators = ['metrics', 'prometheus', 'health', 'monitoring']
        for py_file in source_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if any(indicator in content.lower() for indicator in monitoring_indicators):
                    results['monitoring_ready'] = True
                    break
            except:
                continue
        
        # Calculate production score
        criteria = [
            results['dockerfile_exists'],
            results['requirements_exist'],
            results['config_management'],
            results['logging_implemented'],
            results['error_handling'],
            results['monitoring_ready']
        ]
        
        results['production_score'] = sum(criteria) / len(criteria)
        
        passed = results['production_score'] >= 0.8 and sum(criteria) >= 5
        
        return QualityGateResult(
            name="Production Readiness",
            passed=passed,
            score=results['production_score'],
            details=results
        )


class ComprehensiveQualityGates:
    """Comprehensive quality gates runner."""
    
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.gates = [
            SecurityGate(),
            CodeQualityGate(),
            PerformanceGate(),
            TestCoverageGate(),
            DocumentationGate(),
            ProductionReadinessGate()
        ]
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 Running Comprehensive Quality Gates")
        print("=" * 60)
        
        start_time = time.time()
        results = []
        
        for gate in self.gates:
            print(f"\n🔍 Running {gate.__class__.__name__}...")
            result = gate.run(self.source_dir)
            results.append(result)
            print(f"   {result}")
            
            if result.details:
                self._print_gate_details(result)
        
        # Calculate overall results
        overall_results = self._calculate_overall_results(results)
        overall_results['execution_time'] = time.time() - start_time
        
        self._print_summary(overall_results, results)
        
        return overall_results
    
    def _print_gate_details(self, result: QualityGateResult):
        """Print detailed results for a gate."""
        details = result.details
        
        if result.name == "Security Validation":
            if details['vulnerabilities']:
                print(f"   ⚠️  Found {len(details['vulnerabilities'])} security issues")
                for vuln in details['vulnerabilities'][:3]:  # Show first 3
                    print(f"      • {vuln['type']} in {Path(vuln['file']).name}")
        
        elif result.name == "Code Quality":
            print(f"   📊 Documentation Coverage: {details['documentation_coverage']:.1%}")
            print(f"   📊 Total Classes: {details['total_classes']}")
            print(f"   📊 Total Functions: {details['total_functions']}")
        
        elif result.name == "Performance Analysis":
            if details['performance_patterns']:
                print(f"   ⚠️  Found {len(details['performance_patterns'])} performance patterns")
        
        elif result.name == "Test Coverage":
            print(f"   🧪 Test Files: {details['test_files']}")
            print(f"   🧪 Test Functions: {details['test_functions']}")
            print(f"   📊 Coverage Estimate: {details['coverage_estimate']:.1%}")
        
        elif result.name == "Production Readiness":
            ready_items = sum([
                details['dockerfile_exists'],
                details['requirements_exist'],
                details['config_management'],
                details['logging_implemented'],
                details['error_handling'],
                details['monitoring_ready']
            ])
            print(f"   ✅ Production Criteria Met: {ready_items}/6")
    
    def _calculate_overall_results(self, results: List[QualityGateResult]) -> Dict[str, Any]:
        """Calculate overall quality gate results."""
        total_gates = len(results)
        passed_gates = sum(1 for result in results if result.passed)
        average_score = sum(result.score for result in results) / total_gates
        
        # Weighted overall score (some gates are more critical)
        weights = {
            "Security Validation": 0.25,
            "Code Quality": 0.20,
            "Performance Analysis": 0.15,
            "Test Coverage": 0.15,
            "Documentation": 0.10,
            "Production Readiness": 0.15
        }
        
        weighted_score = sum(
            result.score * weights.get(result.name, 1.0/total_gates)
            for result in results
        )
        
        # Overall pass/fail
        critical_gates = ["Security Validation", "Production Readiness"]
        critical_passed = all(
            result.passed for result in results 
            if result.name in critical_gates
        )
        
        overall_passed = critical_passed and passed_gates >= total_gates * 0.8
        
        return {
            'overall_passed': overall_passed,
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'average_score': average_score,
            'weighted_score': weighted_score,
            'gate_results': {result.name: result for result in results}
        }
    
    def _print_summary(self, overall_results: Dict[str, Any], results: List[QualityGateResult]):
        """Print comprehensive summary."""
        print("\n" + "=" * 60)
        print("🏆 COMPREHENSIVE QUALITY GATES SUMMARY")
        print("=" * 60)
        
        # Overall status
        status = "✅ PASSED" if overall_results['overall_passed'] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Gates Passed: {overall_results['passed_gates']}/{overall_results['total_gates']}")
        print(f"Average Score: {overall_results['average_score']:.2f}")
        print(f"Weighted Score: {overall_results['weighted_score']:.2f}")
        print(f"Execution Time: {overall_results['execution_time']:.2f}s")
        
        print("\n📋 Gate Results:")
        for result in results:
            status_icon = "✅" if result.passed else "❌"
            print(f"  {status_icon} {result.name}: {result.score:.2f}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if not overall_results['overall_passed']:
            failed_gates = [r for r in results if not r.passed]
            for gate in failed_gates:
                print(f"  • Improve {gate.name}: Score {gate.score:.2f}")
        
        # Quality assessment
        if overall_results['weighted_score'] >= 0.9:
            quality_level = "EXCELLENT 🌟"
        elif overall_results['weighted_score'] >= 0.8:
            quality_level = "GOOD ✅"
        elif overall_results['weighted_score'] >= 0.7:
            quality_level = "ACCEPTABLE ⚠️"
        else:
            quality_level = "NEEDS IMPROVEMENT ❌"
        
        print(f"\n🎯 Quality Level: {quality_level}")
        
        print("\n🚀 Production Readiness Assessment:")
        prod_gate = next((r for r in results if r.name == "Production Readiness"), None)
        if prod_gate and prod_gate.passed:
            print("  ✅ Ready for production deployment")
        else:
            print("  ⚠️  Requires additional work before production deployment")


def main():
    """Run comprehensive quality gates."""
    source_dir = Path('/root/repo')
    
    # Validate source directory
    if not source_dir.exists():
        print(f"❌ Source directory {source_dir} does not exist")
        return False
    
    # Run quality gates
    runner = ComprehensiveQualityGates(source_dir)
    results = runner.run_all_gates()
    
    # Return success/failure
    return results['overall_passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)