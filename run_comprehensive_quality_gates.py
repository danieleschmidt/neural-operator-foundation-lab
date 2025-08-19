#!/usr/bin/env python3
"""Comprehensive Quality Gates for Neural Operator Foundation Lab."""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import importlib
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool = False, 
                 score: float = 0.0, details: Dict[str, Any] = None,
                 errors: List[str] = None, warnings: List[str] = None):
        self.name = name
        self.passed = passed
        self.score = score
        self.details = details or {}
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp
        }


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0, required: bool = True):
        self.name = name
        self.weight = weight
        self.required = required
    
    def run(self) -> QualityGateResult:
        """Run the quality gate check."""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Get human-readable description of the gate."""
        return f"Quality gate: {self.name}"


class SecurityGate(QualityGate):
    """Security validation gate."""
    
    def __init__(self):
        super().__init__("Security Validation", weight=2.0, required=True)
    
    def run(self) -> QualityGateResult:
        """Run security checks."""
        logger.info("🔒 Running Security Quality Gate...")
        
        errors = []
        warnings = []
        security_score = 0.0
        details = {}
        
        try:
            # Check for common security issues in Python files
            python_files = []
            for root, dirs, files in os.walk('src'):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            # Scan for potential security issues
            security_issues = []
            dangerous_patterns = [
                ('eval(', 'Use of eval() function'),
                ('exec(', 'Use of exec() function'),
                ('__import__(', 'Dynamic import'),
                ('subprocess.call', 'Subprocess execution'),
                ('os.system(', 'System command execution'),
                ('pickle.loads(', 'Unsafe pickle loading'),
            ]
            
            safe_files = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_issues = []
                    for pattern, description in dangerous_patterns:
                        if pattern in content:
                            file_issues.append(f"{description} found in {file_path}")
                    
                    if not file_issues:
                        safe_files += 1
                    else:
                        security_issues.extend(file_issues)
                
                except Exception as e:
                    warnings.append(f"Could not scan {file_path}: {e}")
            
            # Check for hardcoded secrets (basic patterns)
            secret_patterns = [
                (r'password\s*=\s*[\'"][^\'\"]+[\'"]', 'Hardcoded password'),
                (r'api_key\s*=\s*[\'"][^\'\"]+[\'"]', 'Hardcoded API key'),
                (r'secret\s*=\s*[\'"][^\'\"]+[\'"]', 'Hardcoded secret'),
            ]
            
            import re
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, description in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            warnings.append(f"{description} possibly found in {file_path}")
                
                except Exception as e:
                    continue
            
            # Calculate security score
            total_files = len(python_files)
            if total_files > 0:
                security_score = (safe_files / total_files) * 100.0
            else:
                security_score = 100.0
            
            # Determine pass/fail
            passed = len(security_issues) == 0 and security_score >= 80.0
            
            details = {
                'total_files_scanned': total_files,
                'safe_files': safe_files,
                'security_issues': security_issues,
                'security_score': security_score
            }
            
            if security_issues:
                errors.extend(security_issues)
            
        except Exception as e:
            errors.append(f"Security gate failed: {e}")
            passed = False
        
        return QualityGateResult(
            self.name, passed, security_score, details, errors, warnings
        )


class PerformanceGate(QualityGate):
    """Performance benchmarking gate."""
    
    def __init__(self):
        super().__init__("Performance Benchmarks", weight=1.5, required=False)
    
    def run(self) -> QualityGateResult:
        """Run performance benchmarks."""
        logger.info("⚡ Running Performance Quality Gate...")
        
        errors = []
        warnings = []
        performance_score = 0.0
        details = {}
        
        try:
            # Try to import torch for benchmarking
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False
                warnings.append("PyTorch not available for performance testing")
            
            if torch_available:
                # Simple performance test
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Create test tensors
                test_sizes = [32, 64, 128]
                benchmark_results = []
                
                for size in test_sizes:
                    x = torch.randn(4, size, size, 1).to(device)
                    
                    # Time tensor operations
                    start_time = time.time()
                    
                    # Simple operations
                    y = torch.fft.fft2(x.squeeze(-1))
                    y = torch.fft.ifft2(y).real.unsqueeze(-1)
                    result = torch.mean(y)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    elapsed = time.time() - start_time
                    
                    benchmark_results.append({
                        'size': size,
                        'time_seconds': elapsed,
                        'throughput': (4 * size * size) / elapsed  # elements per second
                    })
                
                # Calculate performance score based on throughput
                avg_throughput = sum(r['throughput'] for r in benchmark_results) / len(benchmark_results)
                performance_score = min(100.0, avg_throughput / 1e6 * 100)  # Scale to 0-100
                
                details = {
                    'device': str(device),
                    'benchmark_results': benchmark_results,
                    'average_throughput': avg_throughput,
                    'torch_version': torch.__version__ if torch_available else None
                }
                
                passed = performance_score >= 10.0  # Minimum threshold
                
            else:
                # Fallback CPU benchmark
                start_time = time.time()
                
                # Simple numerical computation
                result = sum(i**2 for i in range(10000))
                
                elapsed = time.time() - start_time
                performance_score = max(0, 100 - elapsed * 1000)  # Penalize slow computation
                
                details = {
                    'fallback_benchmark': True,
                    'computation_time': elapsed,
                    'result': result
                }
                
                passed = elapsed < 0.1  # Should complete in < 100ms
        
        except Exception as e:
            errors.append(f"Performance gate failed: {e}")
            passed = False
        
        return QualityGateResult(
            self.name, passed, performance_score, details, errors, warnings
        )


class CodeQualityGate(QualityGate):
    """Code quality and style gate."""
    
    def __init__(self):
        super().__init__("Code Quality", weight=1.0, required=False)
    
    def run(self) -> QualityGateResult:
        """Run code quality checks."""
        logger.info("📝 Running Code Quality Gate...")
        
        errors = []
        warnings = []
        quality_score = 0.0
        details = {}
        
        try:
            # Count lines of code
            python_files = []
            for root, dirs, files in os.walk('src'):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            total_lines = 0
            comment_lines = 0
            docstring_lines = 0
            blank_lines = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    in_docstring = False
                    for line in lines:
                        stripped = line.strip()
                        total_lines += 1
                        
                        if not stripped:
                            blank_lines += 1
                        elif stripped.startswith('#'):
                            comment_lines += 1
                        elif '"""' in stripped or "'''" in stripped:
                            docstring_lines += 1
                            in_docstring = not in_docstring
                        elif in_docstring:
                            docstring_lines += 1
                
                except Exception as e:
                    warnings.append(f"Could not analyze {file_path}: {e}")
            
            # Calculate quality metrics
            documentation_ratio = (comment_lines + docstring_lines) / max(total_lines, 1) * 100
            
            # Check for consistent naming conventions
            naming_issues = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic checks for naming conventions
                    lines = content.split('\\n')
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        # Check for camelCase functions (should be snake_case)
                        import re
                        if re.match(r'^def [a-z]+([A-Z][a-z]*)+', stripped):
                            naming_issues += 1
                
                except Exception:
                    continue
            
            # Calculate overall quality score
            quality_score = max(0, 100 - naming_issues * 5)
            quality_score = min(quality_score, documentation_ratio * 2)  # Boost for good documentation
            
            details = {
                'total_files': len(python_files),
                'total_lines': total_lines,
                'comment_lines': comment_lines,
                'docstring_lines': docstring_lines,
                'blank_lines': blank_lines,
                'documentation_ratio': documentation_ratio,
                'naming_issues': naming_issues,
                'code_to_comment_ratio': total_lines / max(comment_lines + docstring_lines, 1)
            }
            
            passed = quality_score >= 50.0 and documentation_ratio >= 10.0
            
        except Exception as e:
            errors.append(f"Code quality gate failed: {e}")
            passed = False
        
        return QualityGateResult(
            self.name, passed, quality_score, details, errors, warnings
        )


class TestCoverageGate(QualityGate):
    """Test coverage gate."""
    
    def __init__(self):
        super().__init__("Test Coverage", weight=1.5, required=False)
    
    def run(self) -> QualityGateResult:
        """Run test coverage analysis."""
        logger.info("🧪 Running Test Coverage Gate...")
        
        errors = []
        warnings = []
        coverage_score = 0.0
        details = {}
        
        try:
            # Count test files
            test_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
            
            # Count source files
            src_files = []
            for root, dirs, files in os.walk('src'):
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        src_files.append(os.path.join(root, file))
            
            # Simple coverage estimation based on test-to-source ratio
            if src_files:
                test_ratio = len(test_files) / len(src_files)
                coverage_score = min(100.0, test_ratio * 100)
            else:
                coverage_score = 0.0
            
            # Try to run existing tests
            test_results = []
            for test_file in test_files:
                try:
                    # This is a basic check - in practice, you'd use a proper test runner
                    result = subprocess.run(
                        [sys.executable, '-m', 'py_compile', test_file],
                        capture_output=True,
                        timeout=30
                    )
                    
                    test_results.append({
                        'file': test_file,
                        'compilable': result.returncode == 0,
                        'errors': result.stderr.decode() if result.stderr else None
                    })
                    
                except Exception as e:
                    test_results.append({
                        'file': test_file,
                        'compilable': False,
                        'errors': str(e)
                    })
            
            compilable_tests = sum(1 for r in test_results if r['compilable'])
            
            details = {
                'test_files_count': len(test_files),
                'source_files_count': len(src_files),
                'test_to_source_ratio': test_ratio if src_files else 0,
                'compilable_tests': compilable_tests,
                'test_results': test_results
            }
            
            passed = coverage_score >= 20.0  # At least 20% test coverage by file count
            
            if len(test_files) == 0:
                warnings.append("No test files found")
            
        except Exception as e:
            errors.append(f"Test coverage gate failed: {e}")
            passed = False
        
        return QualityGateResult(
            self.name, passed, coverage_score, details, errors, warnings
        )


class DocumentationGate(QualityGate):
    """Documentation completeness gate."""
    
    def __init__(self):
        super().__init__("Documentation", weight=1.0, required=False)
    
    def run(self) -> QualityGateResult:
        """Check documentation completeness."""
        logger.info("📚 Running Documentation Gate...")
        
        errors = []
        warnings = []
        doc_score = 0.0
        details = {}
        
        try:
            # Check for key documentation files
            required_docs = {
                'README.md': os.path.exists('README.md'),
                'LICENSE': os.path.exists('LICENSE'),
                'CONTRIBUTING.md': os.path.exists('CONTRIBUTING.md'),
                'requirements.txt': os.path.exists('requirements.txt'),
                'pyproject.toml': os.path.exists('pyproject.toml'),
                'setup.py': os.path.exists('setup.py')
            }
            
            # Check examples directory
            has_examples = os.path.exists('examples') and any(
                f.endswith('.py') for f in os.listdir('examples') if os.path.isfile(os.path.join('examples', f))
            )
            required_docs['examples/'] = has_examples
            
            # Check docs directory
            has_docs = os.path.exists('docs')
            required_docs['docs/'] = has_docs
            
            # Calculate documentation score
            present_docs = sum(1 for present in required_docs.values() if present)
            doc_score = (present_docs / len(required_docs)) * 100
            
            # Check README content quality
            readme_quality = 0
            if os.path.exists('README.md'):
                try:
                    with open('README.md', 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    
                    # Check for key sections
                    key_sections = [
                        'installation', 'usage', 'example', 'api', 
                        'contributing', 'license', 'getting started'
                    ]
                    
                    content_lower = readme_content.lower()
                    present_sections = sum(1 for section in key_sections if section in content_lower)
                    readme_quality = (present_sections / len(key_sections)) * 100
                    
                except Exception as e:
                    warnings.append(f"Could not analyze README.md: {e}")
            
            # Adjust score based on README quality
            doc_score = (doc_score + readme_quality) / 2
            
            details = {
                'required_docs_present': required_docs,
                'documentation_score': doc_score,
                'readme_quality': readme_quality,
                'missing_docs': [doc for doc, present in required_docs.items() if not present]
            }
            
            # Pass if most documentation is present
            passed = doc_score >= 70.0
            
            if not required_docs['README.md']:
                errors.append("README.md is missing")
            
            missing_count = sum(1 for present in required_docs.values() if not present)
            if missing_count > 2:
                warnings.append(f"{missing_count} important documentation files are missing")
        
        except Exception as e:
            errors.append(f"Documentation gate failed: {e}")
            passed = False
        
        return QualityGateResult(
            self.name, passed, doc_score, details, errors, warnings
        )


class ComprehensiveQualityRunner:
    """Run all quality gates and generate report."""
    
    def __init__(self):
        self.gates = [
            SecurityGate(),
            PerformanceGate(),
            CodeQualityGate(),
            TestCoverageGate(),
            DocumentationGate()
        ]
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        logger.info("🚀 Starting Comprehensive Quality Gates")
        
        results = []
        total_score = 0.0
        total_weight = 0.0
        required_failures = []
        
        for gate in self.gates:
            logger.info(f"Running {gate.name}...")
            
            try:
                result = gate.run()
                results.append(result)
                
                # Weight the scores
                weighted_score = result.score * gate.weight
                total_score += weighted_score
                total_weight += gate.weight
                
                # Track required gate failures
                if gate.required and not result.passed:
                    required_failures.append(gate.name)
                
                # Log result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{gate.name}: {status} (Score: {result.score:.1f})")
                
                if result.errors:
                    for error in result.errors:
                        logger.error(f"  ERROR: {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"  WARNING: {warning}")
            
            except Exception as e:
                error_result = QualityGateResult(
                    gate.name, 
                    passed=False, 
                    score=0.0, 
                    errors=[f"Gate execution failed: {e}"]
                )
                results.append(error_result)
                logger.error(f"{gate.name}: ❌ FAILED - {e}")
        
        # Calculate overall score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        overall_passed = len(required_failures) == 0 and overall_score >= 60.0
        
        # Generate summary
        summary = {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'required_failures': required_failures,
            'total_gates': len(self.gates),
            'passed_gates': sum(1 for r in results if r.passed),
            'failed_gates': sum(1 for r in results if not r.passed),
            'timestamp': time.time()
        }
        
        # Complete report
        report = {
            'summary': summary,
            'gate_results': [r.to_dict() for r in results],
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        for result in results:
            if not result.passed:
                if result.name == "Security Validation":
                    recommendations.append("🔒 Review and fix security issues before production deployment")
                elif result.name == "Performance Benchmarks":
                    recommendations.append("⚡ Consider performance optimizations or hardware upgrades")
                elif result.name == "Code Quality":
                    recommendations.append("📝 Improve code style, naming conventions, and documentation")
                elif result.name == "Test Coverage":
                    recommendations.append("🧪 Add more comprehensive tests to improve reliability")
                elif result.name == "Documentation":
                    recommendations.append("📚 Complete missing documentation for better usability")
            
            if result.warnings:
                recommendations.extend(f"⚠️ {warning}" for warning in result.warnings[:3])
        
        if not recommendations:
            recommendations.append("🎉 All quality gates passed! Ready for production.")
        
        return recommendations


def main():
    """Main entry point."""
    print("🌟 Neural Operator Foundation Lab - Comprehensive Quality Gates")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        runner = ComprehensiveQualityRunner()
        report = runner.run_all_gates()
        
        # Print summary
        print("\\n" + "=" * 70)
        print("📋 QUALITY GATES SUMMARY")
        print("=" * 70)
        
        summary = report['summary']
        status = "✅ PASSED" if summary['overall_passed'] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Overall Score: {summary['overall_score']:.1f}/100.0")
        print(f"Passed Gates: {summary['passed_gates']}/{summary['total_gates']}")
        
        if summary['required_failures']:
            print(f"Required Gate Failures: {', '.join(summary['required_failures'])}")
        
        # Print recommendations
        if report['recommendations']:
            print("\\n🔧 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        # Save detailed report
        report_path = Path("quality_gates_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n📄 Detailed report saved: {report_path}")
        
        # Print timing
        elapsed_time = time.time() - start_time
        print(f"⏱️ Total execution time: {elapsed_time:.1f} seconds")
        
        # Exit with appropriate code
        exit_code = 0 if summary['overall_passed'] else 1
        print(f"\\n🚪 Exiting with code: {exit_code}")
        
        return exit_code
    
    except Exception as e:
        logger.error(f"Quality gates runner failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)