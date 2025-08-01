#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes the highest-value work items.
"""

import json
import os
import subprocess
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort_hours: float
    
    # WSJF Components
    user_business_value: float
    time_criticality: float
    risk_reduction: float
    opportunity_enablement: float
    
    # ICE Components  
    impact: float
    confidence: float
    ease: float
    
    # Technical Debt
    debt_impact: float
    debt_interest: float
    hotspot_multiplier: float
    
    # Composite scores
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
            
        # Calculate composite scores
        cost_of_delay = (self.user_business_value + self.time_criticality + 
                        self.risk_reduction + self.opportunity_enablement)
        self.wsjf_score = cost_of_delay / max(self.estimated_effort_hours, 0.5)
        
        self.ice_score = self.impact * self.confidence * self.ease
        
        self.technical_debt_score = ((self.debt_impact + self.debt_interest) * 
                                   self.hotspot_multiplier)

class ValueDiscoveryEngine:
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Fallback configuration for repositories without .terragon/config.yaml"""
        return {
            "scoring": {
                "weights": {"wsjf": 0.5, "ice": 0.2, "technicalDebt": 0.2, "security": 0.1},
                "thresholds": {"minScore": 10, "maxRisk": 0.8}
            },
            "discovery": {"sources": ["gitHistory", "staticAnalysis"]}
        }
    
    def discover_work_items(self) -> List[WorkItem]:
        """Comprehensive work item discovery from multiple sources"""
        items = []
        
        # Discover from each configured source
        for source in self.config.get("discovery", {}).get("sources", []):
            if source == "gitHistory":
                items.extend(self._discover_from_git_history())
            elif source == "staticAnalysis":
                items.extend(self._discover_from_static_analysis())
            elif source == "vulnerabilityDatabases":
                items.extend(self._discover_security_vulnerabilities())
            elif source == "performanceMonitoring":
                items.extend(self._discover_performance_issues())
            elif source == "codeMetrics":
                items.extend(self._discover_code_quality_issues())
                
        return items
    
    def _discover_from_git_history(self) -> List[WorkItem]:
        """Extract TODOs, FIXMEs, and debt markers from git history and current code"""
        items = []
        
        # Search for TODO/FIXME comments in codebase
        try:
            result = subprocess.run([
                "grep", "-r", "-n", "-i", 
                "-E", r"(TODO|FIXME|HACK|XXX|DEPRECATED).*",
                "--include=*.py", "--include=*.js", "--include=*.ts",
                "."
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    file_path, line_num, content = line.split(':', 2)
                    
                    # Extract the comment content
                    comment_match = re.search(r'(TODO|FIXME|HACK|XXX|DEPRECATED)[:\s]*(.*)', content, re.IGNORECASE)
                    if comment_match:
                        marker = comment_match.group(1).upper()
                        description = comment_match.group(2).strip()
                        
                        # Score based on marker type
                        urgency_scores = {
                            "FIXME": (8, 9, 6),  # high impact, confidence, medium ease
                            "HACK": (7, 8, 4),   # high impact, medium ease
                            "TODO": (6, 7, 8),   # medium impact, high ease
                            "XXX": (9, 6, 3),    # very high impact, low confidence/ease
                            "DEPRECATED": (5, 9, 7)  # medium impact, high confidence
                        }
                        
                        impact, confidence, ease = urgency_scores.get(marker, (5, 5, 5))
                        
                        item = WorkItem(
                            id=f"tech-debt-{hash(line) % 10000}",
                            title=f"{marker}: {description[:50]}...",
                            description=f"Address {marker.lower()} comment: {description}",
                            category="technical_debt",
                            source="git_history",
                            files_affected=[file_path],
                            estimated_effort_hours=2.0 if marker in ["TODO", "DEPRECATED"] else 4.0,
                            user_business_value=3 if marker == "DEPRECATED" else 2,
                            time_criticality=8 if marker == "FIXME" else 4,
                            risk_reduction=7 if marker in ["FIXME", "HACK"] else 3,
                            opportunity_enablement=2,
                            impact=impact,
                            confidence=confidence,
                            ease=ease,
                            debt_impact=impact * 2,
                            debt_interest=1.5,
                            hotspot_multiplier=self._get_hotspot_multiplier(file_path)
                        )
                        items.append(item)
                        
        except subprocess.CalledProcessError:
            pass  # grep not found or no matches
            
        return items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Extract issues from static analysis tools"""
        items = []
        
        # Run flake8 if available
        try:
            result = subprocess.run([
                "flake8", "--format=json", "src/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and result.stdout:
                flake8_issues = json.loads(result.stdout)
                for issue in flake8_issues:
                    item = WorkItem(
                        id=f"lint-{hash(str(issue)) % 10000}",
                        title=f"Fix linting issue: {issue['code']}",
                        description=f"Address {issue['code']}: {issue['text']}",
                        category="code_quality",
                        source="static_analysis",
                        files_affected=[issue['filename']],
                        estimated_effort_hours=0.5,
                        user_business_value=2,
                        time_criticality=3,
                        risk_reduction=4,
                        opportunity_enablement=2,
                        impact=4,
                        confidence=9,
                        ease=8,
                        debt_impact=3,
                        debt_interest=1.2,
                        hotspot_multiplier=self._get_hotspot_multiplier(issue['filename'])
                    )
                    items.append(item)
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
            
        return items
    
    def _discover_security_vulnerabilities(self) -> List[WorkItem]:
        """Discover security vulnerabilities using safety and bandit"""
        items = []
        
        # Run safety check for dependency vulnerabilities
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                safety_report = json.loads(result.stdout)
                for vuln in safety_report:
                    item = WorkItem(
                        id=f"security-{hash(str(vuln)) % 10000}",
                        title=f"Security: Update {vuln['package_name']}",
                        description=f"Vulnerability in {vuln['package_name']}: {vuln['advisory']}",
                        category="security",
                        source="vulnerability_database",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        estimated_effort_hours=1.0,
                        user_business_value=9,
                        time_criticality=8,
                        risk_reduction=9,
                        opportunity_enablement=3,
                        impact=9,
                        confidence=8,
                        ease=7,
                        debt_impact=8,
                        debt_interest=2.0,
                        hotspot_multiplier=1.5
                    )
                    items.append(item)
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
            
        return items
    
    def _discover_performance_issues(self) -> List[WorkItem]:
        """Discover performance issues from benchmarks and profiling"""
        items = []
        
        # Check for slow tests that could indicate performance issues
        try:
            result = subprocess.run([
                "pytest", "--collect-only", "-q", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                # Look for tests marked as slow
                slow_test_pattern = r"test.*slow|slow.*test"
                for line in result.stdout.split('\n'):
                    if re.search(slow_test_pattern, line, re.IGNORECASE):
                        item = WorkItem(
                            id=f"perf-{hash(line) % 10000}",
                            title="Optimize slow test performance",
                            description=f"Performance optimization needed: {line}",
                            category="performance",
                            source="performance_monitoring",
                            files_affected=["tests/"],
                            estimated_effort_hours=3.0,
                            user_business_value=6,
                            time_criticality=4,
                            risk_reduction=3,
                            opportunity_enablement=7,
                            impact=6,
                            confidence=6,
                            ease=5,
                            debt_impact=4,
                            debt_interest=1.3,
                            hotspot_multiplier=1.2
                        )
                        items.append(item)
                        
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return items
    
    def _discover_code_quality_issues(self) -> List[WorkItem]:
        """Discover code quality issues using complexity analysis"""
        items = []
        
        # Basic complexity heuristics
        python_files = list(Path("src/").rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Count lines of code
                loc = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
                
                # If file is very large, suggest refactoring
                if loc > 500:
                    item = WorkItem(
                        id=f"refactor-{hash(str(py_file)) % 10000}",
                        title=f"Refactor large file: {py_file.name}",
                        description=f"File has {loc} lines of code, consider breaking into smaller modules",
                        category="code_quality",
                        source="code_metrics",
                        files_affected=[str(py_file)],
                        estimated_effort_hours=6.0,
                        user_business_value=4,
                        time_criticality=2,
                        risk_reduction=6,
                        opportunity_enablement=8,
                        impact=5,
                        confidence=7,
                        ease=4,
                        debt_impact=6,
                        debt_interest=1.4,
                        hotspot_multiplier=self._get_hotspot_multiplier(str(py_file))
                    )
                    items.append(item)
                    
            except (OSError, UnicodeDecodeError):
                continue
                
        return items
    
    def _get_hotspot_multiplier(self, file_path: str) -> float:
        """Calculate hotspot multiplier based on file churn and complexity"""
        try:
            # Get git log for file churn
            result = subprocess.run([
                "git", "log", "--oneline", "--", file_path
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            commit_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Files with more commits are hotter spots
            if commit_count > 50:
                return 2.5
            elif commit_count > 20:
                return 2.0
            elif commit_count > 10:
                return 1.5
            else:
                return 1.0
                
        except subprocess.CalledProcessError:
            return 1.0
    
    def calculate_composite_scores(self, items: List[WorkItem]) -> List[WorkItem]:
        """Calculate final composite scores for all work items"""
        weights = self.config.get("scoring", {}).get("weights", {
            "wsjf": 0.6, "ice": 0.1, "technicalDebt": 0.2, "security": 0.1
        })
        
        for item in items:
            # Normalize scores to 0-100 scale
            normalized_wsjf = min(item.wsjf_score * 10, 100)
            normalized_ice = min(item.ice_score / 10, 100) 
            normalized_debt = min(item.technical_debt_score * 5, 100)
            
            # Calculate composite score
            item.composite_score = (
                weights["wsjf"] * normalized_wsjf +
                weights["ice"] * normalized_ice +
                weights["technicalDebt"] * normalized_debt
            )
            
            # Apply security boost
            if item.category == "security":
                security_boost = self.config.get("scoring", {}).get("thresholds", {}).get("securityBoost", 2.0)
                item.composite_score *= security_boost
        
        return sorted(items, key=lambda x: x.composite_score, reverse=True)
    
    def save_backlog(self, items: List[WorkItem], path: str = ".terragon/backlog.json"):
        """Save discovered work items to backlog file"""
        backlog_data = {
            "last_updated": datetime.now().isoformat(),
            "total_items": len(items),
            "items": [asdict(item) for item in items]
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(backlog_data, f, indent=2)
    
    def run_discovery(self) -> List[WorkItem]:
        """Main discovery execution"""
        print("🔍 Starting autonomous value discovery...")
        
        # Discover work items
        items = self.discover_work_items()
        print(f"   Discovered {len(items)} potential work items")
        
        # Calculate composite scores
        items = self.calculate_composite_scores(items)
        print(f"   Scored and prioritized {len(items)} items")
        
        # Filter by minimum score threshold
        min_score = self.config.get("scoring", {}).get("thresholds", {}).get("minScore", 10)
        qualified_items = [item for item in items if item.composite_score >= min_score]
        print(f"   {len(qualified_items)} items meet minimum score threshold ({min_score})")
        
        # Save backlog
        self.save_backlog(qualified_items)
        print(f"   Saved backlog to .terragon/backlog.json")
        
        return qualified_items

if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    items = engine.run_discovery()
    
    if items:
        print(f"\n🎯 Next best value item:")
        best = items[0]
        print(f"   [{best.id}] {best.title}")
        print(f"   Score: {best.composite_score:.1f} | Category: {best.category}")
        print(f"   Estimated effort: {best.estimated_effort_hours}h")
    else:
        print("\n✅ No high-value work items discovered. Repository is in good shape!")