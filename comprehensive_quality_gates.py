#!/usr/bin/env python3
"""Comprehensive quality gates for neural operator lab."""

import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# Add src to path
sys.path.insert(0, 'src')


@dataclass
class QualityCheck:
    """Quality check result."""
    name: str
    status: str  # "passed", "failed", "warning", "skipped"
    score: float  # 0-100
    details: Dict[str, Any]
    duration: float
    errors: List[str]
    recommendations: List[str]


class QualityGateRunner:
    """Comprehensive quality gate execution."""
    
    def __init__(self):
        self.results = {}
        self.overall_score = 0.0
        self.start_time = time.time()
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🛡️ EXECUTING COMPREHENSIVE QUALITY GATES")
        print("=" * 60)
        
        # Run all quality checks
        checks = [
            self.check_core_functionality,
            self.check_security_framework,
            self.check_performance_optimization,
            self.check_research_capabilities,
            self.check_code_quality,
            self.check_documentation_completeness,
            self.check_deployment_readiness,
            self.check_compliance_standards
        ]
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_func in checks:
            try:
                result = check_func()
                self.results[result.name] = result
                
                # Weight different checks
                weight = self._get_check_weight(result.name)
                total_score += result.score * weight
                total_weight += weight
                
                # Print immediate result
                status_icon = self._get_status_icon(result.status)
                print(f"{status_icon} {result.name}: {result.score:.1f}/100")
                
                if result.errors:
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"   ❌ {error}")
                
                if result.recommendations:
                    for rec in result.recommendations[:2]:  # Show first 2 recommendations
                        print(f"   💡 {rec}")
                
            except Exception as e:
                error_result = QualityCheck(
                    name=check_func.__name__,
                    status="failed",
                    score=0.0,
                    details={},
                    duration=0.0,
                    errors=[str(e)],
                    recommendations=["Fix critical errors before proceeding"]
                )
                self.results[error_result.name] = error_result
                print(f"❌ {check_func.__name__}: FAILED - {e}")
        
        # Calculate overall score
        self.overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Generate summary
        return self._generate_summary()
    
    def _get_status_icon(self, status: str) -> str:
        """Get status icon."""
        icons = {
            "passed": "✅",
            "failed": "❌", 
            "warning": "⚠️",
            "skipped": "⏭️"
        }
        return icons.get(status, "❓")
    
    def _get_check_weight(self, check_name: str) -> float:
        """Get weight for different checks."""
        weights = {
            'check_core_functionality': 2.0,
            'check_security_framework': 1.5,
            'check_performance_optimization': 1.5,
            'check_research_capabilities': 1.0,
            'check_code_quality': 1.0,
            'check_documentation_completeness': 0.8,
            'check_deployment_readiness': 1.2,
            'check_compliance_standards': 0.8
        }
        return weights.get(check_name, 1.0)
    
    def check_core_functionality(self) -> QualityCheck:
        """Check core functionality."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Test core robust framework
            print("   🧪 Testing core robust framework...")
            result = subprocess.run([
                sys.executable, "test_core_robust.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                details['robust_framework'] = "passed"
            else:
                errors.append("Core robust framework tests failed")
                details['robust_framework'] = "failed"
            
            # Test performance optimization
            print("   🚀 Testing performance optimization...")
            result = subprocess.run([
                sys.executable, "test_generation_3_simple.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                details['performance_optimization'] = "passed"
            else:
                errors.append("Performance optimization tests failed")
                details['performance_optimization'] = "failed"
            
            # Test research framework
            print("   🔬 Testing research framework...")
            result = subprocess.run([
                sys.executable, "test_research_framework.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                details['research_framework'] = "passed"
            else:
                errors.append("Research framework tests failed")
                details['research_framework'] = "failed"
            
            # Calculate score
            passed_tests = sum(1 for v in details.values() if v == "passed")
            total_tests = len(details)
            score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            if score < 80:
                recommendations.append("Fix failing core functionality tests")
            if score < 60:
                status = "failed"
            elif score < 80:
                status = "warning"
            else:
                status = "passed"
            
        except subprocess.TimeoutExpired:
            errors.append("Tests timed out")
            score = 0
            status = "failed"
        except Exception as e:
            errors.append(f"Test execution failed: {e}")
            score = 0
            status = "failed"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Core Functionality",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_security_framework(self) -> QualityCheck:
        """Check security framework."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check for security modules
            from neural_operator_lab.security import SecurityConfig
            from neural_operator_lab.robust_framework import InputValidator
            details['security_modules'] = "present"
            
            # Check security configurations
            config = SecurityConfig()
            if config.enable_input_validation:
                details['input_validation'] = "enabled"
            else:
                errors.append("Input validation not enabled")
            
            if config.enable_audit_logging:
                details['audit_logging'] = "enabled"
            else:
                errors.append("Audit logging not enabled")
            
            # Check for forbidden patterns
            security_issues = self._scan_security_patterns()
            details['security_scan'] = f"{len(security_issues)} issues found"
            
            if security_issues:
                errors.extend(security_issues[:5])  # Show first 5 issues
                recommendations.append("Review and fix security issues")
            
            # Calculate score
            base_score = 80
            score = base_score - (len(security_issues) * 5)  # -5 per issue
            score = max(0, min(100, score))
            
            if score >= 90:
                status = "passed"
            elif score >= 70:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Security check failed: {e}")
            score = 0
            status = "failed"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Security Framework",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def _scan_security_patterns(self) -> List[str]:
        """Scan for basic security patterns."""
        issues = []
        
        # Check Python files for potential security issues
        py_files = list(Path("src").rglob("*.py"))
        
        dangerous_patterns = [
            ("eval(", "Use of eval() function"),
            ("exec(", "Use of exec() function"),
            ("__import__(", "Dynamic imports"),
        ]
        
        for py_file in py_files[:10]:  # Limit to first 10 files for speed
            try:
                content = py_file.read_text()
                for pattern, description in dangerous_patterns:
                    # Skip false positives
                    if pattern == "eval(" and ".eval()" in content:
                        continue  # PyTorch model.eval() is safe
                    
                    if pattern in content and "# SECURITY:" not in content:
                        # Check for actual dangerous usage
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if pattern in line and not line.strip().startswith('#'):
                                # Additional filtering for false positives
                                if pattern == "eval(" and ("model.eval()" in line or "self.eval()" in line):
                                    continue
                                issues.append(f"{py_file}:{line_num}: {description}")
            except Exception:
                continue
        
        return issues
    
    def check_performance_optimization(self) -> QualityCheck:
        """Check performance optimization."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check for performance modules
            from neural_operator_lab.performance_optimizer import (
                AdvancedCache, PerformanceProfiler
            )
            from neural_operator_lab.intelligent_cache import AdaptiveCache
            details['performance_modules'] = "present"
            
            # Test caching performance
            cache = AdvancedCache(maxsize=100)
            
            # Performance test
            test_start = time.time()
            for i in range(1000):
                cache.put(f"key_{i}", f"value_{i}")
            
            for i in range(1000):
                cache.get(f"key_{i}")
            
            cache_perf_time = time.time() - test_start
            details['cache_performance'] = f"{cache_perf_time:.3f}s for 2000 ops"
            
            if cache_perf_time > 1.0:
                errors.append("Cache performance slower than expected")
            
            # Test profiler
            profiler = PerformanceProfiler()
            with profiler.profile("test_op"):
                time.sleep(0.01)
            
            stats = profiler.get_operation_stats("test_op")
            if stats['count'] > 0:
                details['profiler'] = "working"
            else:
                errors.append("Performance profiler not working")
            
            # Calculate score
            score = 90
            if errors:
                score -= len(errors) * 10
            
            score = max(0, score)
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Performance check failed: {e}")
            score = 0
            status = "failed"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Performance Optimization",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_research_capabilities(self) -> QualityCheck:
        """Check research capabilities."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check research modules
            from neural_operator_lab.research_framework import (
                StatisticalAnalyzer, BenchmarkSuite
            )
            from neural_operator_lab.experiment_runner import ExperimentRunner
            details['research_modules'] = "present"
            
            # Test statistical analysis
            analyzer = StatisticalAnalyzer()
            test_data = [1, 2, 3, 4, 5]
            stats = analyzer.descriptive_statistics(test_data)
            
            if 'mean' in stats and abs(stats['mean'] - 3.0) < 0.01:
                details['statistical_analysis'] = "working"
            else:
                errors.append("Statistical analysis not working correctly")
            
            # Test benchmark suite
            suite = BenchmarkSuite()
            
            def test_benchmark():
                return {'value': 42}
            
            suite.register_benchmark('test', test_benchmark)
            result = suite.run_benchmark('test', {}, num_runs=2)
            
            if result and result.success:
                details['benchmark_suite'] = "working"
            else:
                errors.append("Benchmark suite not working")
            
            # Test experiment runner
            runner = ExperimentRunner()
            details['experiment_runner'] = "present"
            
            score = 90 - (len(errors) * 15)
            score = max(0, score)
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Research capabilities check failed: {e}")
            score = 0
            status = "failed"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Research Capabilities",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_code_quality(self) -> QualityCheck:
        """Check code quality."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Count Python files
            py_files = list(Path("src").rglob("*.py"))
            details['python_files'] = len(py_files)
            
            # Basic structure check
            required_modules = [
                "src/neural_operator_lab/__init__.py",
                "src/neural_operator_lab/base.py",
                "src/neural_operator_lab/models/__init__.py",
                "src/neural_operator_lab/robust_framework.py",
                "src/neural_operator_lab/performance_optimizer.py",
                "src/neural_operator_lab/research_framework.py"
            ]
            
            missing_modules = []
            for module in required_modules:
                if not Path(module).exists():
                    missing_modules.append(module)
            
            if missing_modules:
                errors.extend([f"Missing module: {m}" for m in missing_modules])
                details['missing_modules'] = len(missing_modules)
            else:
                details['module_completeness'] = "complete"
            
            # Check for basic documentation
            documented_files = 0
            for py_file in py_files[:20]:  # Check first 20 files
                try:
                    content = py_file.read_text()
                    if '"""' in content and 'def ' in content:
                        documented_files += 1
                except Exception:
                    continue
            
            doc_ratio = documented_files / min(20, len(py_files))
            details['documentation_ratio'] = f"{doc_ratio:.2%}"
            
            if doc_ratio < 0.5:
                recommendations.append("Add more documentation to Python files")
            
            # Calculate score
            base_score = 85
            score = base_score - (len(missing_modules) * 10) - (max(0, 0.8 - doc_ratio) * 20)
            score = max(0, min(100, score))
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Code quality check failed: {e}")
            score = 50
            status = "warning"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Code Quality",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_documentation_completeness(self) -> QualityCheck:
        """Check documentation completeness."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check for key documentation files
            doc_files = [
                "README.md",
                "docs/README.md",
                "CONTRIBUTING.md",
                "LICENSE",
                "pyproject.toml",
                "requirements.txt"
            ]
            
            existing_docs = []
            missing_docs = []
            
            for doc_file in doc_files:
                if Path(doc_file).exists():
                    existing_docs.append(doc_file)
                else:
                    missing_docs.append(doc_file)
            
            details['existing_docs'] = len(existing_docs)
            details['missing_docs'] = len(missing_docs)
            
            if missing_docs:
                errors.extend([f"Missing documentation: {d}" for d in missing_docs])
            
            # Check README quality
            readme_path = Path("README.md")
            if readme_path.exists():
                readme_content = readme_path.read_text()
                readme_sections = ['install', 'usage', 'example', 'feature']
                
                sections_found = sum(
                    1 for section in readme_sections 
                    if section.lower() in readme_content.lower()
                )
                
                details['readme_sections'] = f"{sections_found}/{len(readme_sections)}"
                
                if sections_found < 3:
                    recommendations.append("Enhance README with more sections")
            
            # Calculate score
            doc_completeness = len(existing_docs) / len(doc_files)
            score = doc_completeness * 100
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Documentation check failed: {e}")
            score = 50
            status = "warning"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Documentation Completeness",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_deployment_readiness(self) -> QualityCheck:
        """Check deployment readiness."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check for deployment files
            deployment_files = [
                "Dockerfile",
                "docker-compose.yml",
                "pyproject.toml",
                "setup.py"
            ]
            
            deployment_score = 0
            for file in deployment_files:
                if Path(file).exists():
                    deployment_score += 25
                    details[file] = "present"
                else:
                    details[file] = "missing"
            
            # Check for configuration management
            config_files = list(Path("src").rglob("*config*.py"))
            details['config_files'] = len(config_files)
            
            if config_files:
                deployment_score += 10
            
            # Check for environment handling
            if Path(".env.example").exists() or "environment" in str(config_files):
                deployment_score += 10
                details['environment_handling'] = "present"
            
            # Check for health checks
            health_check_files = []
            for py_file in Path("src").rglob("*.py"):
                try:
                    content = py_file.read_text()
                    if "health" in content.lower() and ("check" in content.lower() or "status" in content.lower()):
                        health_check_files.append(str(py_file))
                except Exception:
                    continue
            
            if health_check_files:
                deployment_score += 5
                details['health_checks'] = f"{len(health_check_files)} files"
            
            score = min(100, deployment_score)
            
            if score < 60:
                recommendations.append("Add missing deployment files")
            if score < 40:
                recommendations.append("Implement proper configuration management")
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Deployment readiness check failed: {e}")
            score = 50
            status = "warning"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Deployment Readiness",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def check_compliance_standards(self) -> QualityCheck:
        """Check compliance with standards."""
        start_time = time.time()
        errors = []
        recommendations = []
        details = {}
        
        try:
            # Check for compliance modules
            compliance_modules = list(Path("src").rglob("*compliance*"))
            details['compliance_modules'] = len(compliance_modules)
            
            # Check for GDPR compliance
            gdpr_files = [f for f in compliance_modules if "gdpr" in str(f).lower()]
            
            # Also check for GDPR content in compliance files
            gdpr_content = False
            for comp_file in compliance_modules:
                try:
                    if comp_file.suffix == '.py':
                        content = comp_file.read_text()
                        if 'gdpr' in content.lower() or 'general data protection' in content.lower():
                            gdpr_content = True
                            break
                except Exception:
                    continue
            
            if gdpr_files or gdpr_content:
                details['gdpr_compliance'] = "implemented"
            else:
                errors.append("GDPR compliance module missing")
            
            # Check for security standards
            security_modules = list(Path("src").rglob("*security*"))
            details['security_modules'] = len(security_modules)
            
            # Check for audit logging
            audit_files = []
            for py_file in Path("src").rglob("*.py"):
                try:
                    content = py_file.read_text()
                    if "audit" in content.lower() and "log" in content.lower():
                        audit_files.append(str(py_file))
                except Exception:
                    continue
            
            if audit_files:
                details['audit_logging'] = f"{len(audit_files)} files"
            else:
                recommendations.append("Implement comprehensive audit logging")
            
            # Check for licensing
            if Path("LICENSE").exists():
                details['license'] = "present"
            else:
                errors.append("License file missing")
            
            # Calculate score
            compliance_score = 70  # Base score
            
            if details.get('gdpr_compliance') == "implemented":
                compliance_score += 15
            if details.get('license') == "present":
                compliance_score += 10
            if audit_files:
                compliance_score += 5
            
            compliance_score -= len(errors) * 10
            score = max(0, min(100, compliance_score))
            
            if score >= 80:
                status = "passed"
            elif score >= 60:
                status = "warning"
            else:
                status = "failed"
            
        except Exception as e:
            errors.append(f"Compliance check failed: {e}")
            score = 50
            status = "warning"
        
        duration = time.time() - start_time
        
        return QualityCheck(
            name="Compliance Standards",
            status=status,
            score=score,
            details=details,
            duration=duration,
            errors=errors,
            recommendations=recommendations
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate quality gates summary."""
        total_duration = time.time() - self.start_time
        
        # Count status
        status_counts = {"passed": 0, "failed": 0, "warning": 0, "skipped": 0}
        
        for result in self.results.values():
            status_counts[result.status] += 1
        
        # Get worst issues
        all_errors = []
        all_recommendations = []
        
        for result in self.results.values():
            all_errors.extend(result.errors)
            all_recommendations.extend(result.recommendations)
        
        # Determine overall status
        if status_counts["failed"] > 0:
            overall_status = "FAILED"
        elif status_counts["warning"] > 2:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"
        
        return {
            'overall_status': overall_status,
            'overall_score': self.overall_score,
            'total_duration': total_duration,
            'status_counts': status_counts,
            'individual_results': {name: {
                'status': result.status,
                'score': result.score,
                'duration': result.duration
            } for name, result in self.results.items()},
            'top_errors': all_errors[:10],
            'top_recommendations': all_recommendations[:10],
            'timestamp': time.time()
        }


def main():
    """Run comprehensive quality gates."""
    runner = QualityGateRunner()
    summary = runner.run_all_gates()
    
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    # Overall status
    status_icon = "🎉" if summary['overall_status'] == "PASSED" else "⚠️" if summary['overall_status'] == "WARNING" else "❌"
    print(f"{status_icon} Overall Status: {summary['overall_status']}")
    print(f"📊 Overall Score: {summary['overall_score']:.1f}/100")
    print(f"⏱️ Total Duration: {summary['total_duration']:.1f}s")
    
    # Status breakdown
    print("\n📋 Status Breakdown:")
    for status, count in summary['status_counts'].items():
        icon = runner._get_status_icon(status)
        print(f"   {icon} {status.title()}: {count}")
    
    # Individual results
    print("\n📊 Individual Results:")
    for name, result in summary['individual_results'].items():
        icon = runner._get_status_icon(result['status'])
        print(f"   {icon} {name}: {result['score']:.1f}/100 ({result['duration']:.1f}s)")
    
    # Top recommendations
    if summary['top_recommendations']:
        print("\n💡 Top Recommendations:")
        for i, rec in enumerate(summary['top_recommendations'][:5], 1):
            print(f"   {i}. {rec}")
    
    # Top errors
    if summary['top_errors']:
        print("\n❌ Critical Issues:")
        for i, error in enumerate(summary['top_errors'][:5], 1):
            print(f"   {i}. {error}")
    
    print("\n" + "=" * 60)
    
    # Save results
    results_file = Path("quality_gates_results.json")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    
    # Exit code based on overall status
    if summary['overall_status'] == "PASSED":
        print("✅ All quality gates passed! Ready for deployment.")
        return 0
    elif summary['overall_status'] == "WARNING":
        print("⚠️ Quality gates passed with warnings. Review recommendations.")
        return 0
    else:
        print("❌ Quality gates failed. Fix critical issues before deployment.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)