#!/usr/bin/env python3
"""
Terragon Metrics Collection System
Collects comprehensive value delivery and SDLC metrics for continuous improvement.
"""

import json
import os
import subprocess
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

class MetricsCollector:
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.metrics_dir = Path(".terragon/metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive repository health metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "repository": self._get_repository_info(),
            "code_quality": self._collect_code_quality_metrics(),
            "security": self._collect_security_metrics(),
            "performance": self._collect_performance_metrics(),
            "technical_debt": self._collect_technical_debt_metrics(),
            "testing": self._collect_testing_metrics(),
            "dependencies": self._collect_dependency_metrics(),
            "git_activity": self._collect_git_activity_metrics()
        }
        
        return metrics
    
    def _get_repository_info(self) -> Dict[str, Any]:
        """Get basic repository information"""
        try:
            # Get repository stats
            result = subprocess.run([
                "find", ".", "-name", "*.py", "-type", "f"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            python_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Count lines of code
            total_loc = 0
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        total_loc += len([line for line in f if line.strip() and not line.strip().startswith('#')])
                except (OSError, UnicodeDecodeError):
                    continue
            
            return {
                "python_files": len(python_files),
                "lines_of_code": total_loc,
                "maturity_level": self.config.get("repository", {}).get("maturity", "unknown"),
                "primary_language": self.config.get("repository", {}).get("primary_language", "python")
            }
            
        except subprocess.CalledProcessError:
            return {"error": "Failed to collect repository info"}
    
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics using static analysis tools"""
        metrics = {}
        
        # Flake8 linting metrics
        try:
            result = subprocess.run([
                "flake8", "--statistics", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                # Parse flake8 statistics
                lint_issues = len(result.stdout.strip().split('\n'))
                metrics["lint_issues"] = lint_issues
            else:
                metrics["lint_issues"] = 0
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            metrics["lint_issues"] = "tool_not_available"
        
        # MyPy type checking
        try:
            result = subprocess.run([
                "mypy", "--show-error-codes", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            type_errors = len([line for line in result.stdout.split('\n') if 'error:' in line])
            metrics["type_errors"] = type_errors
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            metrics["type_errors"] = "tool_not_available"
        
        # Code complexity (if radon available)
        try:
            result = subprocess.run([
                "radon", "cc", "--json", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                metrics["average_cyclomatic_complexity"] = round(avg_complexity, 2)
                metrics["total_functions"] = function_count
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metrics["average_cyclomatic_complexity"] = "tool_not_available"
        
        return metrics
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics"""
        metrics = {}
        
        # Bandit security scan
        try:
            result = subprocess.run([
                "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                metrics["security_issues"] = len(bandit_data.get("results", []))
                
                # Categorize by severity
                severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "LOW")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                metrics["security_issues_by_severity"] = severity_counts
            else:
                metrics["security_issues"] = 0
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metrics["security_issues"] = "tool_not_available"
        
        # Safety dependency scan
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                metrics["vulnerable_dependencies"] = len(safety_data)
            else:
                metrics["vulnerable_dependencies"] = 0
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metrics["vulnerable_dependencies"] = "tool_not_available"
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics"""
        metrics = {}
        
        # Check for performance tests
        try:
            result = subprocess.run([
                "find", "tests/", "-name", "*performance*", "-o", "-name", "*benchmark*"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            perf_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            metrics["performance_test_files"] = len([f for f in perf_files if f])
            
        except subprocess.CalledProcessError:
            metrics["performance_test_files"] = 0
        
        # Check for profiling or performance markers in code
        try:
            result = subprocess.run([
                "grep", "-r", "-c", "-i", "profile\\|benchmark\\|timer", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            performance_markers = len([line for line in result.stdout.split('\n') if line and ':' in line and int(line.split(':')[1]) > 0])
            metrics["performance_monitoring_usage"] = performance_markers
            
        except (subprocess.CalledProcessError, ValueError):
            metrics["performance_monitoring_usage"] = 0
        
        return metrics
    
    def _collect_technical_debt_metrics(self) -> Dict[str, Any]:
        """Collect technical debt indicators"""
        metrics = {}
        
        # Count TODO/FIXME comments
        try:
            result = subprocess.run([
                "grep", "-r", "-c", "-i", "TODO\\|FIXME\\|HACK\\|XXX", "--include=*.py", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            debt_markers = {}
            total_debt_comments = 0
            
            for line in result.stdout.split('\n'):
                if ':' in line:
                    file_path, count = line.split(':', 1)
                    count = int(count)
                    if count > 0:
                        debt_markers[file_path] = count
                        total_debt_comments += count
            
            metrics["total_debt_comments"] = total_debt_comments
            metrics["files_with_debt"] = len(debt_markers)
            
            if debt_markers:
                metrics["debt_hotspots"] = sorted(debt_markers.items(), key=lambda x: x[1], reverse=True)[:5]
            
        except (subprocess.CalledProcessError, ValueError):
            metrics["total_debt_comments"] = 0
        
        # Analyze deprecated usage
        try:
            result = subprocess.run([
                "grep", "-r", "-c", "-i", "deprecated", "--include=*.py", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            deprecated_usage = sum(int(line.split(':')[1]) for line in result.stdout.split('\n') 
                                 if ':' in line and line.split(':')[1].isdigit())
            metrics["deprecated_usage_count"] = deprecated_usage
            
        except (subprocess.CalledProcessError, ValueError):
            metrics["deprecated_usage_count"] = 0
        
        return metrics
    
    def _collect_testing_metrics(self) -> Dict[str, Any]:
        """Collect testing and coverage metrics"""
        metrics = {}
        
        # Test coverage
        try:
            result = subprocess.run([
                "pytest", "--cov=src/", "--cov-report=json", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            # Try to read coverage report
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    
                metrics["test_coverage_percentage"] = round(coverage_data["totals"]["percent_covered"], 2)
                metrics["lines_covered"] = coverage_data["totals"]["covered_lines"]
                metrics["lines_missing"] = coverage_data["totals"]["missing_lines"]
                
                # Clean up coverage file
                coverage_file.unlink()
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metrics["test_coverage_percentage"] = "tool_not_available"
        
        # Count test files and functions
        try:
            result = subprocess.run([
                "find", "tests/", "-name", "test_*.py", "-o", "-name", "*_test.py"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            test_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            metrics["test_files"] = len([f for f in test_files if f])
            
            # Count test functions
            test_function_count = 0
            for test_file in test_files:
                if test_file:
                    try:
                        with open(test_file, 'r') as f:
                            content = f.read()
                            test_function_count += len(re.findall(r'def test_\w+', content))
                    except (OSError, UnicodeDecodeError):
                        continue
            
            metrics["test_functions"] = test_function_count
            
        except subprocess.CalledProcessError:
            metrics["test_files"] = 0
            metrics["test_functions"] = 0
        
        return metrics
    
    def _collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics"""
        metrics = {}
        
        # Count dependencies
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        dependency_count = 0
        
        for req_file in req_files:
            if Path(req_file).exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                        if req_file.endswith('.toml'):
                            # Count dependencies in pyproject.toml
                            dependency_count += len(re.findall(r'"[^"]+"\s*=', content))
                        else:
                            # Count dependencies in requirements files
                            dependency_count += len([line for line in content.split('\n') 
                                                   if line.strip() and not line.startswith('#')])
                except OSError:
                    continue
        
        metrics["total_dependencies"] = dependency_count
        
        # Check for dependency updates available
        try:
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                outdated_deps = json.loads(result.stdout)
                metrics["outdated_dependencies"] = len(outdated_deps)
            else:
                metrics["outdated_dependencies"] = 0
                
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            metrics["outdated_dependencies"] = "tool_not_available"
        
        return metrics
    
    def _collect_git_activity_metrics(self) -> Dict[str, Any]:
        """Collect git activity and contribution metrics"""
        metrics = {}
        
        try:
            # Commit activity in last 30 days
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run([
                "git", "log", "--since", thirty_days_ago, "--oneline"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["commits_last_30_days"] = recent_commits
            
            # Get branch count
            result = subprocess.run([
                "git", "branch", "-r"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            branch_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["remote_branches"] = branch_count
            
            # Contributors
            result = subprocess.run([
                "git", "log", "--format=%an", "--since", thirty_days_ago
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            contributors = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
            metrics["active_contributors_last_30_days"] = len(contributors)
            
        except subprocess.CalledProcessError:
            metrics["commits_last_30_days"] = "git_not_available"
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to timestamped file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also save as latest
        latest_file = self.metrics_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics saved to {metrics_file}")
        return metrics_file
    
    def calculate_maturity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate repository maturity score based on collected metrics"""
        
        score = 0.0
        max_score = 100.0
        
        # Code quality (25 points)
        if metrics["code_quality"]["lint_issues"] == 0:
            score += 10
        elif isinstance(metrics["code_quality"]["lint_issues"], int):
            score += max(0, 10 - metrics["code_quality"]["lint_issues"])
        
        if metrics["code_quality"]["type_errors"] == 0:
            score += 10
        elif isinstance(metrics["code_quality"]["type_errors"], int):
            score += max(0, 10 - metrics["code_quality"]["type_errors"])
        
        if isinstance(metrics["code_quality"].get("average_cyclomatic_complexity"), (int, float)):
            if metrics["code_quality"]["average_cyclomatic_complexity"] <= 5:
                score += 5
            elif metrics["code_quality"]["average_cyclomatic_complexity"] <= 10:
                score += 3
        
        # Security (20 points)
        if metrics["security"]["security_issues"] == 0:
            score += 10
        elif isinstance(metrics["security"]["security_issues"], int):
            score += max(0, 10 - metrics["security"]["security_issues"])
        
        if metrics["security"]["vulnerable_dependencies"] == 0:
            score += 10
        elif isinstance(metrics["security"]["vulnerable_dependencies"], int):
            score += max(0, 10 - metrics["security"]["vulnerable_dependencies"])
        
        # Testing (25 points)
        if isinstance(metrics["testing"].get("test_coverage_percentage"), (int, float)):
            coverage = metrics["testing"]["test_coverage_percentage"]
            if coverage >= 90:
                score += 15
            elif coverage >= 80:
                score += 12
            elif coverage >= 70:
                score += 8
            elif coverage >= 50:
                score += 5
        
        if metrics["testing"]["test_files"] > 0:
            score += 5
        
        if metrics["testing"]["test_functions"] > 10:
            score += 5
        
        # Technical debt (15 points)
        if metrics["technical_debt"]["total_debt_comments"] == 0:
            score += 10
        elif metrics["technical_debt"]["total_debt_comments"] < 10:
            score += 7
        elif metrics["technical_debt"]["total_debt_comments"] < 25:
            score += 4
        
        if metrics["technical_debt"]["deprecated_usage_count"] == 0:
            score += 5
        
        # Dependencies (15 points)
        if isinstance(metrics["dependencies"].get("outdated_dependencies"), int):
            outdated = metrics["dependencies"]["outdated_dependencies"]
            if outdated == 0:
                score += 10
            elif outdated < 5:
                score += 7
            elif outdated < 10:
                score += 4
        
        if metrics["dependencies"]["total_dependencies"] > 0:
            score += 5
        
        return min(score, max_score)
    
    def run_collection(self) -> str:
        """Main metrics collection execution"""
        print("📊 Starting comprehensive metrics collection...")
        
        metrics = self.collect_repository_metrics()
        maturity_score = self.calculate_maturity_score(metrics)
        
        metrics["maturity_score"] = round(maturity_score, 1)
        
        print(f"   Repository maturity score: {maturity_score:.1f}/100")
        print(f"   Code quality: {len([k for k, v in metrics['code_quality'].items() if isinstance(v, int) and v == 0])} metrics optimal")
        print(f"   Security posture: {metrics['security']['security_issues']} issues found")
        print(f"   Test coverage: {metrics['testing'].get('test_coverage_percentage', 'N/A')}%")
        print(f"   Technical debt: {metrics['technical_debt']['total_debt_comments']} debt comments")
        
        metrics_file = self.save_metrics(metrics)
        return str(metrics_file)

if __name__ == "__main__":
    collector = MetricsCollector()
    metrics_file = collector.run_collection()
    print(f"\n✅ Metrics collection complete: {metrics_file}")