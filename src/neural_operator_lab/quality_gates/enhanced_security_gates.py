"""Enhanced Security Validation Gates.

Implements comprehensive security validation with neural operator specific checks,
AI-powered threat detection, and advanced vulnerability assessment.
"""

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import tempfile
import shutil

from .progressive_gates import QualityGateResult, QualityGateGeneration
from .autonomous_validation import SelfImprovingGate

logger = logging.getLogger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityCategory(Enum):
    """Security vulnerability categories."""
    CODE_INJECTION = "code_injection"
    HARDCODED_SECRETS = "hardcoded_secrets"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    PATH_TRAVERSAL = "path_traversal"
    NEURAL_OPERATOR_SPECIFIC = "neural_operator_specific"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    CONFIGURATION_WEAKNESS = "configuration_weakness"
    DATA_PRIVACY = "data_privacy"
    MODEL_SECURITY = "model_security"
    SUPPLY_CHAIN = "supply_chain"


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    id: str
    category: SecurityCategory
    threat_level: SecurityThreatLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    remediation: Optional[str] = None
    false_positive_probability: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return {
            'id': self.id,
            'category': self.category.value,
            'threat_level': self.threat_level.value,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'cwe_id': self.cwe_id,
            'remediation': self.remediation,
            'false_positive_probability': self.false_positive_probability,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class SecurityScanResult:
    """Results of a security scan."""
    scan_type: str
    scan_duration: float
    files_scanned: int
    vulnerabilities: List[SecurityVulnerability]
    security_score: float
    confidence: float
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_count(self) -> int:
        """Count of critical vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.threat_level == SecurityThreatLevel.CRITICAL)
    
    @property
    def high_count(self) -> int:
        """Count of high severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.threat_level == SecurityThreatLevel.HIGH)
    
    @property
    def total_count(self) -> int:
        """Total vulnerability count."""
        return len(self.vulnerabilities)


class SecurityPatternDatabase:
    """Database of security patterns for detection."""
    
    def __init__(self):
        self.patterns = self._load_security_patterns()
        self.neural_operator_patterns = self._load_neural_operator_patterns()
        self.dependency_patterns = self._load_dependency_patterns()
    
    def _load_security_patterns(self) -> Dict[SecurityCategory, List[Dict[str, Any]]]:
        """Load standard security patterns."""
        return {
            SecurityCategory.CODE_INJECTION: [
                {
                    'pattern': r'exec\s*\(',
                    'description': 'Use of exec() function',
                    'threat_level': SecurityThreatLevel.CRITICAL,
                    'cwe_id': 'CWE-94',
                    'remediation': 'Avoid dynamic code execution. Use safer alternatives.'
                },
                {
                    'pattern': r'eval\s*\(',
                    'description': 'Use of eval() function',
                    'threat_level': SecurityThreatLevel.CRITICAL,
                    'cwe_id': 'CWE-94',
                    'remediation': 'Replace eval() with safer parsing methods like ast.literal_eval()'
                },
                {
                    'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                    'description': 'Shell injection via subprocess',
                    'threat_level': SecurityThreatLevel.HIGH,
                    'cwe_id': 'CWE-78',
                    'remediation': 'Use shell=False and pass arguments as a list'
                },
                {
                    'pattern': r'os\.system\s*\(',
                    'description': 'Use of os.system()',
                    'threat_level': SecurityThreatLevel.HIGH,
                    'cwe_id': 'CWE-78',
                    'remediation': 'Use subprocess with proper argument handling'
                }
            ],
            SecurityCategory.HARDCODED_SECRETS: [
                {
                    'pattern': r'(?i)(password|pwd)\s*=\s*["\'][^"\']{8,}["\']',
                    'description': 'Hardcoded password',
                    'threat_level': SecurityThreatLevel.CRITICAL,
                    'cwe_id': 'CWE-798',
                    'remediation': 'Use environment variables or secure credential stores'
                },
                {
                    'pattern': r'(?i)(api_key|apikey)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
                    'description': 'Hardcoded API key',
                    'threat_level': SecurityThreatLevel.CRITICAL,
                    'cwe_id': 'CWE-798',
                    'remediation': 'Store API keys in environment variables or secure vaults'
                },
                {
                    'pattern': r'(?i)(secret|token)\s*=\s*["\'][a-zA-Z0-9]{15,}["\']',
                    'description': 'Hardcoded secret or token',
                    'threat_level': SecurityThreatLevel.HIGH,
                    'cwe_id': 'CWE-798',
                    'remediation': 'Use secure configuration management'
                },
                {
                    'pattern': r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
                    'description': 'Embedded private key',
                    'threat_level': SecurityThreatLevel.CRITICAL,
                    'cwe_id': 'CWE-798',
                    'remediation': 'Store private keys securely outside of code'
                }
            ],
            SecurityCategory.UNSAFE_DESERIALIZATION: [
                {
                    'pattern': r'pickle\.loads?\s*\(',
                    'description': 'Unsafe pickle deserialization',
                    'threat_level': SecurityThreatLevel.HIGH,
                    'cwe_id': 'CWE-502',
                    'remediation': 'Use safer serialization formats like JSON or implement custom deserialization with validation'
                },
                {
                    'pattern': r'yaml\.load\s*\(',
                    'description': 'Unsafe YAML loading',
                    'threat_level': SecurityThreatLevel.HIGH,
                    'cwe_id': 'CWE-502',
                    'remediation': 'Use yaml.safe_load() instead of yaml.load()'
                },
                {
                    'pattern': r'marshal\.loads?\s*\(',
                    'description': 'Unsafe marshal deserialization',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'cwe_id': 'CWE-502',
                    'remediation': 'Validate input before unmarshaling'
                }
            ],
            SecurityCategory.PATH_TRAVERSAL: [
                {
                    'pattern': r'\.\./',
                    'description': 'Potential path traversal',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'cwe_id': 'CWE-22',
                    'remediation': 'Validate and sanitize file paths'
                },
                {
                    'pattern': r'open\s*\(\s*["\'][^"\']*\.\.[^"\']*["\']',
                    'description': 'File access with path traversal',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'cwe_id': 'CWE-22',
                    'remediation': 'Use os.path.abspath() and validate paths'
                }
            ]
        }
    
    def _load_neural_operator_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load neural operator specific security patterns."""
        return {
            'unsafe_model_loading': {
                'pattern': r'torch\.load\s*\([^)]*map_location\s*=\s*None',
                'description': 'Unsafe PyTorch model loading without device specification',
                'threat_level': SecurityThreatLevel.MEDIUM,
                'cwe_id': 'CWE-502',
                'remediation': 'Always specify map_location when loading models'
            },
            'pickle_model_loading': {
                'pattern': r'torch\.load\s*\([^)]*\).*\.pkl',
                'description': 'Loading pickle model files',
                'threat_level': SecurityThreatLevel.HIGH,
                'cwe_id': 'CWE-502',
                'remediation': 'Use state_dict instead of pickle for model serialization'
            },
            'unsafe_model_execution': {
                'pattern': r'eval\s*\(\s*model_code',
                'description': 'Dynamic model code execution',
                'threat_level': SecurityThreatLevel.CRITICAL,
                'cwe_id': 'CWE-94',
                'remediation': 'Define model architecture statically'
            },
            'data_leakage': {
                'pattern': r'print\s*\(.*(?:password|token|key)',
                'description': 'Potential sensitive data logging',
                'threat_level': SecurityThreatLevel.MEDIUM,
                'cwe_id': 'CWE-532',
                'remediation': 'Avoid logging sensitive information'
            },
            'model_poisoning_risk': {
                'pattern': r'(?:urllib|requests)\..*download.*\.(?:pt|pth|pkl)',
                'description': 'Downloading model files from external sources',
                'threat_level': SecurityThreatLevel.HIGH,
                'cwe_id': 'CWE-829',
                'remediation': 'Verify model integrity and use trusted sources'
            }
        }
    
    def _load_dependency_patterns(self) -> List[Dict[str, Any]]:
        """Load dependency security patterns."""
        return [
            {
                'name': 'torch',
                'vulnerable_versions': ['<1.12.0'],
                'description': 'PyTorch versions with known vulnerabilities',
                'threat_level': SecurityThreatLevel.HIGH
            },
            {
                'name': 'pillow',
                'vulnerable_versions': ['<8.3.2'],
                'description': 'Pillow versions with image processing vulnerabilities',
                'threat_level': SecurityThreatLevel.MEDIUM
            },
            {
                'name': 'numpy',
                'vulnerable_versions': ['<1.21.0'],
                'description': 'NumPy versions with potential security issues',
                'threat_level': SecurityThreatLevel.LOW
            }
        ]


class StaticCodeAnalyzer:
    """Static code analysis for security vulnerabilities."""
    
    def __init__(self):
        self.pattern_db = SecurityPatternDatabase()
        self.scan_cache: Dict[str, SecurityScanResult] = {}
    
    async def analyze_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Analyze a single file for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Cache key for this file
            file_hash = hashlib.md5(content.encode()).hexdigest()
            cache_key = f"{file_path}:{file_hash}"
            
            if cache_key in self.scan_cache:
                return self.scan_cache[cache_key].vulnerabilities
            
            # Analyze with standard patterns
            vulnerabilities.extend(await self._analyze_standard_patterns(file_path, content))
            
            # Analyze with neural operator patterns
            vulnerabilities.extend(await self._analyze_neural_operator_patterns(file_path, content))
            
            # AST-based analysis
            vulnerabilities.extend(await self._analyze_ast_patterns(file_path, content))
            
            # Cache results
            scan_result = SecurityScanResult(
                scan_type="static_analysis",
                scan_duration=0.0,
                files_scanned=1,
                vulnerabilities=vulnerabilities,
                security_score=self._calculate_file_security_score(vulnerabilities),
                confidence=0.85
            )
            self.scan_cache[cache_key] = scan_result
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return vulnerabilities
    
    async def _analyze_standard_patterns(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Analyze file content against standard security patterns."""
        vulnerabilities = []
        
        for category, patterns in self.pattern_db.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    # Extract code snippet
                    lines = content.split('\n')
                    start_line = max(0, line_number - 2)
                    end_line = min(len(lines), line_number + 1)
                    code_snippet = '\n'.join(lines[start_line:end_line])
                    
                    # Generate vulnerability ID
                    vuln_id = hashlib.sha256(f"{file_path}:{line_number}:{pattern}".encode()).hexdigest()[:12]
                    
                    vulnerability = SecurityVulnerability(
                        id=vuln_id,
                        category=category,
                        threat_level=pattern_info['threat_level'],
                        title=pattern_info['description'],
                        description=f"{pattern_info['description']} detected in {file_path.name}",
                        file_path=str(file_path),
                        line_number=line_number,
                        code_snippet=code_snippet,
                        cwe_id=pattern_info.get('cwe_id'),
                        remediation=pattern_info.get('remediation'),
                        confidence=0.8 - self._calculate_false_positive_probability(match.group(), pattern)
                    )
                    
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    async def _analyze_neural_operator_patterns(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Analyze neural operator specific patterns."""
        vulnerabilities = []
        
        for pattern_name, pattern_info in self.pattern_db.neural_operator_patterns.items():
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                vuln_id = hashlib.sha256(f"{file_path}:{line_number}:{pattern_name}".encode()).hexdigest()[:12]
                
                vulnerability = SecurityVulnerability(
                    id=vuln_id,
                    category=SecurityCategory.NEURAL_OPERATOR_SPECIFIC,
                    threat_level=pattern_info['threat_level'],
                    title=pattern_info['description'],
                    description=f"Neural operator security issue: {pattern_info['description']}",
                    file_path=str(file_path),
                    line_number=line_number,
                    cwe_id=pattern_info.get('cwe_id'),
                    remediation=pattern_info.get('remediation'),
                    confidence=0.9,
                    metadata={'pattern_name': pattern_name}
                )
                
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    async def _analyze_ast_patterns(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Analyze using Abstract Syntax Tree for more accurate detection."""
        vulnerabilities = []
        
        try:
            tree = ast.parse(content)
            
            # Check for dangerous function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    vuln = await self._check_dangerous_call(node, file_path, content)
                    if vuln:
                        vulnerabilities.append(vuln)
                
                # Check for hardcoded strings that might be secrets
                elif isinstance(node, ast.Str):
                    vuln = await self._check_hardcoded_secret(node, file_path, content)
                    if vuln:
                        vulnerabilities.append(vuln)
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            logger.warning(f"AST analysis failed for {file_path}: {e}")
        
        return vulnerabilities
    
    async def _check_dangerous_call(self, node: ast.Call, file_path: Path, content: str) -> Optional[SecurityVulnerability]:
        """Check for dangerous function calls using AST."""
        dangerous_functions = {
            'exec': SecurityThreatLevel.CRITICAL,
            'eval': SecurityThreatLevel.CRITICAL,
            'compile': SecurityThreatLevel.HIGH,
            '__import__': SecurityThreatLevel.MEDIUM
        }
        
        if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
            line_number = node.lineno
            vuln_id = hashlib.sha256(f"{file_path}:{line_number}:ast_call".encode()).hexdigest()[:12]
            
            return SecurityVulnerability(
                id=vuln_id,
                category=SecurityCategory.CODE_INJECTION,
                threat_level=dangerous_functions[node.func.id],
                title=f"Dangerous function call: {node.func.id}",
                description=f"Use of {node.func.id}() function detected",
                file_path=str(file_path),
                line_number=line_number,
                cwe_id='CWE-94',
                remediation=f"Avoid using {node.func.id}() or implement proper input validation",
                confidence=0.95
            )
        
        return None
    
    async def _check_hardcoded_secret(self, node: ast.Str, file_path: Path, content: str) -> Optional[SecurityVulnerability]:
        """Check for hardcoded secrets using AST."""
        if len(node.s) < 10:  # Skip short strings
            return None
        
        # Simple entropy check for potential secrets
        entropy = self._calculate_entropy(node.s)
        
        if entropy > 4.0:  # High entropy suggests possible secret
            line_number = node.lineno
            vuln_id = hashlib.sha256(f"{file_path}:{line_number}:high_entropy".encode()).hexdigest()[:12]
            
            return SecurityVulnerability(
                id=vuln_id,
                category=SecurityCategory.HARDCODED_SECRETS,
                threat_level=SecurityThreatLevel.MEDIUM,
                title="High entropy string (possible secret)",
                description="String with high entropy detected - may be a hardcoded secret",
                file_path=str(file_path),
                line_number=line_number,
                cwe_id='CWE-798',
                remediation="Review string content and move secrets to secure configuration",
                confidence=0.6,
                false_positive_probability=0.4
            )
        
        return None
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not string:
            return 0
        
        # Count character frequencies
        freq = {}
        for char in string:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        for count in freq.values():
            p = count / len(string)
            if p > 0:
                entropy -= p * (p.bit_length() - 1)
        
        return entropy
    
    def _calculate_false_positive_probability(self, match_text: str, pattern: str) -> float:
        """Calculate probability that a match is a false positive."""
        # Simple heuristics for false positive detection
        fp_probability = 0.0
        
        # Check for common false positive patterns
        if 'test' in match_text.lower() or 'example' in match_text.lower():
            fp_probability += 0.3
        
        if 'demo' in match_text.lower() or 'sample' in match_text.lower():
            fp_probability += 0.2
        
        if len(match_text) < 10:
            fp_probability += 0.1
        
        return min(fp_probability, 0.8)
    
    def _calculate_file_security_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate security score for a file."""
        if not vulnerabilities:
            return 1.0
        
        # Weight vulnerabilities by threat level
        weights = {
            SecurityThreatLevel.CRITICAL: 1.0,
            SecurityThreatLevel.HIGH: 0.7,
            SecurityThreatLevel.MEDIUM: 0.4,
            SecurityThreatLevel.LOW: 0.2,
            SecurityThreatLevel.INFO: 0.1
        }
        
        total_impact = sum(weights.get(v.threat_level, 0.5) * v.confidence for v in vulnerabilities)
        
        # Normalize to 0-1 scale
        max_impact = len(vulnerabilities) * 1.0  # If all were critical with 100% confidence
        
        if max_impact == 0:
            return 1.0
        
        normalized_impact = total_impact / max_impact
        return max(0.0, 1.0 - normalized_impact)


class DependencyScanner:
    """Scanner for dependency vulnerabilities."""
    
    def __init__(self):
        self.vulnerability_db = self._load_vulnerability_database()
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database for dependencies."""
        return {
            'torch': {
                'vulnerable_versions': ['<1.12.0', '1.13.0'],
                'vulnerabilities': [
                    {
                        'id': 'CVE-2022-45907',
                        'description': 'PyTorch vulnerability in torch.jit',
                        'severity': 'HIGH',
                        'fixed_version': '1.13.1'
                    }
                ]
            },
            'pillow': {
                'vulnerable_versions': ['<8.3.2'],
                'vulnerabilities': [
                    {
                        'id': 'CVE-2021-34552',
                        'description': 'Buffer overflow in Pillow',
                        'severity': 'HIGH',
                        'fixed_version': '8.3.2'
                    }
                ]
            }
        }
    
    async def scan_dependencies(self, source_dir: Path) -> SecurityScanResult:
        """Scan project dependencies for vulnerabilities."""
        vulnerabilities = []
        scan_start = time.time()
        
        # Find dependency files
        dependency_files = [
            source_dir / 'requirements.txt',
            source_dir / 'pyproject.toml',
            source_dir / 'setup.py',
            source_dir / 'Pipfile'
        ]
        
        for dep_file in dependency_files:
            if dep_file.exists():
                file_vulns = await self._scan_dependency_file(dep_file)
                vulnerabilities.extend(file_vulns)
        
        scan_duration = time.time() - scan_start
        
        return SecurityScanResult(
            scan_type="dependency_scan",
            scan_duration=scan_duration,
            files_scanned=len([f for f in dependency_files if f.exists()]),
            vulnerabilities=vulnerabilities,
            security_score=self._calculate_dependency_security_score(vulnerabilities),
            confidence=0.9
        )
    
    async def _scan_dependency_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a specific dependency file."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if file_path.name == 'requirements.txt':
                vulnerabilities.extend(await self._scan_requirements_txt(file_path, content))
            elif file_path.name == 'pyproject.toml':
                vulnerabilities.extend(await self._scan_pyproject_toml(file_path, content))
            
        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")
        
        return vulnerabilities
    
    async def _scan_requirements_txt(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan requirements.txt for vulnerable dependencies."""
        vulnerabilities = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse dependency
            if '==' in line:
                package, version = line.split('==', 1)
                package = package.strip()
                version = version.strip()
                
                # Check against vulnerability database
                if package in self.vulnerability_db:
                    vuln_info = self.vulnerability_db[package]
                    
                    for vuln in vuln_info.get('vulnerabilities', []):
                        if self._is_version_vulnerable(version, vuln_info['vulnerable_versions']):
                            vuln_id = hashlib.sha256(f"{file_path}:{package}:{vuln['id']}".encode()).hexdigest()[:12]
                            
                            vulnerability = SecurityVulnerability(
                                id=vuln_id,
                                category=SecurityCategory.DEPENDENCY_VULNERABILITY,
                                threat_level=self._severity_to_threat_level(vuln['severity']),
                                title=f"Vulnerable dependency: {package}",
                                description=f"{package} {version} - {vuln['description']}",
                                file_path=str(file_path),
                                line_number=line_num,
                                remediation=f"Update {package} to version {vuln['fixed_version']} or later",
                                confidence=0.95,
                                metadata={
                                    'package': package,
                                    'version': version,
                                    'cve_id': vuln['id'],
                                    'fixed_version': vuln['fixed_version']
                                }
                            )
                            
                            vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    async def _scan_pyproject_toml(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan pyproject.toml for vulnerable dependencies."""
        # Simplified TOML parsing for dependencies
        vulnerabilities = []
        
        # Look for dependency patterns in TOML
        dependency_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        matches = re.finditer(dependency_pattern, content)
        
        for match in matches:
            package = match.group(1)
            version_spec = match.group(2)
            
            # Extract version if it's a simple version specification
            if '>=' in version_spec:
                version = version_spec.split('>=')[1].strip()
            elif '==' in version_spec:
                version = version_spec.split('==')[1].strip()
            else:
                continue
            
            # Check vulnerability database
            if package in self.vulnerability_db:
                vuln_info = self.vulnerability_db[package]
                
                for vuln in vuln_info.get('vulnerabilities', []):
                    if self._is_version_vulnerable(version, vuln_info['vulnerable_versions']):
                        line_num = content[:match.start()].count('\n') + 1
                        vuln_id = hashlib.sha256(f"{file_path}:{package}:{vuln['id']}".encode()).hexdigest()[:12]
                        
                        vulnerability = SecurityVulnerability(
                            id=vuln_id,
                            category=SecurityCategory.DEPENDENCY_VULNERABILITY,
                            threat_level=self._severity_to_threat_level(vuln['severity']),
                            title=f"Vulnerable dependency: {package}",
                            description=f"{package} {version} - {vuln['description']}",
                            file_path=str(file_path),
                            line_number=line_num,
                            remediation=f"Update {package} to version {vuln['fixed_version']} or later",
                            confidence=0.95
                        )
                        
                        vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _is_version_vulnerable(self, version: str, vulnerable_versions: List[str]) -> bool:
        """Check if a version is vulnerable based on version specs."""
        # Simplified version comparison
        for vuln_spec in vulnerable_versions:
            if vuln_spec.startswith('<'):
                # Extract version from spec like "<1.12.0"
                threshold = vuln_spec[1:]
                if self._compare_versions(version, threshold) < 0:
                    return True
            elif vuln_spec == version:
                return True
        
        return False
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
        # Simple version comparison
        def normalize(v):
            return [int(x) for x in v.split('.')]
        
        try:
            v1_parts = normalize(v1)
            v2_parts = normalize(v2)
            
            # Pad with zeros to same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for a, b in zip(v1_parts, v2_parts):
                if a < b:
                    return -1
                elif a > b:
                    return 1
            
            return 0
        except ValueError:
            # Fallback to string comparison
            return -1 if v1 < v2 else (0 if v1 == v2 else 1)
    
    def _severity_to_threat_level(self, severity: str) -> SecurityThreatLevel:
        """Convert severity string to threat level."""
        severity_map = {
            'CRITICAL': SecurityThreatLevel.CRITICAL,
            'HIGH': SecurityThreatLevel.HIGH,
            'MEDIUM': SecurityThreatLevel.MEDIUM,
            'LOW': SecurityThreatLevel.LOW
        }
        return severity_map.get(severity.upper(), SecurityThreatLevel.MEDIUM)
    
    def _calculate_dependency_security_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate security score for dependencies."""
        if not vulnerabilities:
            return 1.0
        
        # Weight by threat level
        threat_weights = {
            SecurityThreatLevel.CRITICAL: 0.4,
            SecurityThreatLevel.HIGH: 0.3,
            SecurityThreatLevel.MEDIUM: 0.2,
            SecurityThreatLevel.LOW: 0.1
        }
        
        total_impact = sum(threat_weights.get(v.threat_level, 0.2) for v in vulnerabilities)
        
        # Normalize based on number of vulnerabilities
        max_impact = len(vulnerabilities) * 0.4  # If all were critical
        
        if max_impact == 0:
            return 1.0
        
        return max(0.0, 1.0 - (total_impact / max_impact))


class ComprehensiveSecurityGate(SelfImprovingGate):
    """Comprehensive security gate with multi-layer validation."""
    
    def __init__(self):
        super().__init__("Comprehensive Security Validation", QualityGateGeneration.GENERATION_1)
        self.static_analyzer = StaticCodeAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.scan_history: List[Dict[str, Any]] = []
    
    async def _execute_generation_specific(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute comprehensive security validation."""
        source_dir = Path(context.get('source_dir', '/root/repo'))
        scan_start = time.time()
        
        # Multi-layer security scanning
        scan_results = await self._perform_comprehensive_scan(source_dir)
        
        # Aggregate results
        aggregated_results = await self._aggregate_scan_results(scan_results)
        
        # AI-powered threat assessment
        threat_assessment = await self._perform_threat_assessment(aggregated_results)
        
        # Calculate overall security posture
        security_posture = await self._calculate_security_posture(
            aggregated_results, threat_assessment
        )
        
        scan_duration = time.time() - scan_start
        
        # Store scan history
        scan_record = {
            'timestamp': time.time(),
            'scan_duration': scan_duration,
            'results': aggregated_results,
            'threat_assessment': threat_assessment,
            'security_posture': security_posture
        }
        self.scan_history.append(scan_record)
        
        # Generate result
        return QualityGateResult(
            name=self.name,
            generation=self.generation,
            passed=security_posture['overall_passed'],
            score=security_posture['security_score'],
            confidence=security_posture['confidence'],
            execution_time=scan_duration,
            details={
                'scan_results': aggregated_results,
                'threat_assessment': threat_assessment,
                'security_posture': security_posture,
                'scan_duration': scan_duration,
                'scan_layers': ['static_analysis', 'dependency_scan', 'configuration_scan']
            },
            recommendations=security_posture.get('recommendations', []),
            next_actions=security_posture.get('next_actions', [])
        )
    
    async def _perform_comprehensive_scan(self, source_dir: Path) -> Dict[str, SecurityScanResult]:
        """Perform comprehensive multi-layer security scan."""
        scan_results = {}
        
        # Static code analysis
        logger.info("🔍 Running static code analysis")
        static_result = await self._run_static_analysis(source_dir)
        scan_results['static_analysis'] = static_result
        
        # Dependency scanning
        logger.info("📦 Scanning dependencies")
        dependency_result = await self.dependency_scanner.scan_dependencies(source_dir)
        scan_results['dependency_scan'] = dependency_result
        
        # Configuration scanning
        logger.info("⚙️  Scanning configuration files")
        config_result = await self._scan_configuration_files(source_dir)
        scan_results['configuration_scan'] = config_result
        
        return scan_results
    
    async def _run_static_analysis(self, source_dir: Path) -> SecurityScanResult:
        """Run static code analysis on all Python files."""
        all_vulnerabilities = []
        files_scanned = 0
        scan_start = time.time()
        
        # Find all Python files
        python_files = list(source_dir.rglob('*.py'))
        
        # Filter out files that should be skipped
        python_files = [f for f in python_files if not self._should_skip_file(f)]
        
        # Analyze files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(asyncio.run, self.static_analyzer.analyze_file(py_file)): py_file
                for py_file in python_files
            }
            
            for future in as_completed(futures):
                try:
                    file_vulnerabilities = future.result()
                    all_vulnerabilities.extend(file_vulnerabilities)
                    files_scanned += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze file: {e}")
        
        scan_duration = time.time() - scan_start
        security_score = self._calculate_overall_security_score(all_vulnerabilities, files_scanned)
        
        return SecurityScanResult(
            scan_type="static_analysis",
            scan_duration=scan_duration,
            files_scanned=files_scanned,
            vulnerabilities=all_vulnerabilities,
            security_score=security_score,
            confidence=0.85,
            scan_metadata={
                'analyzer_version': '1.0',
                'patterns_checked': len(self.static_analyzer.pattern_db.patterns)
            }
        )
    
    async def _scan_configuration_files(self, source_dir: Path) -> SecurityScanResult:
        """Scan configuration files for security issues."""
        vulnerabilities = []
        files_scanned = 0
        scan_start = time.time()
        
        # Configuration file patterns
        config_patterns = [
            '*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.cfg',
            '.env*', 'Dockerfile*', 'docker-compose*'
        ]
        
        config_files = []
        for pattern in config_patterns:
            config_files.extend(source_dir.rglob(pattern))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_vulns = await self._analyze_configuration_file(config_file, content)
                vulnerabilities.extend(file_vulns)
                files_scanned += 1
                
            except Exception as e:
                logger.warning(f"Failed to scan config file {config_file}: {e}")
        
        scan_duration = time.time() - scan_start
        security_score = self._calculate_config_security_score(vulnerabilities, files_scanned)
        
        return SecurityScanResult(
            scan_type="configuration_scan",
            scan_duration=scan_duration,
            files_scanned=files_scanned,
            vulnerabilities=vulnerabilities,
            security_score=security_score,
            confidence=0.8
        )
    
    async def _analyze_configuration_file(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Analyze a configuration file for security issues."""
        vulnerabilities = []
        
        # Check for hardcoded secrets in config files
        secret_patterns = [
            r'(?i)(password|pwd|secret|token|key)\s*[:=]\s*["\'][^"\']{8,}["\']',
            r'(?i)(api_key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'
        ]
        
        for pattern in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                vuln_id = hashlib.sha256(f"{file_path}:{line_number}:config_secret".encode()).hexdigest()[:12]
                
                vulnerability = SecurityVulnerability(
                    id=vuln_id,
                    category=SecurityCategory.HARDCODED_SECRETS,
                    threat_level=SecurityThreatLevel.HIGH,
                    title="Hardcoded secret in configuration",
                    description=f"Potential hardcoded secret detected in {file_path.name}",
                    file_path=str(file_path),
                    line_number=line_number,
                    cwe_id='CWE-798',
                    remediation="Move secrets to environment variables or secure vault",
                    confidence=0.8
                )
                
                vulnerabilities.append(vulnerability)
        
        # Check for insecure Dockerfile configurations
        if file_path.name.startswith('Dockerfile'):
            dockerfile_vulns = await self._check_dockerfile_security(file_path, content)
            vulnerabilities.extend(dockerfile_vulns)
        
        return vulnerabilities
    
    async def _check_dockerfile_security(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Check Dockerfile for security issues."""
        vulnerabilities = []
        
        # Check for running as root
        if 'USER root' in content or not re.search(r'USER\s+\w+', content):
            vuln_id = hashlib.sha256(f"{file_path}:root_user".encode()).hexdigest()[:12]
            
            vulnerability = SecurityVulnerability(
                id=vuln_id,
                category=SecurityCategory.CONFIGURATION_WEAKNESS,
                threat_level=SecurityThreatLevel.MEDIUM,
                title="Container running as root",
                description="Dockerfile does not specify non-root user",
                file_path=str(file_path),
                cwe_id='CWE-250',
                remediation="Add USER directive to run container as non-root user",
                confidence=0.9
            )
            
            vulnerabilities.append(vulnerability)
        
        # Check for secrets in environment variables
        env_pattern = r'ENV\s+\w*(?:PASSWORD|SECRET|KEY|TOKEN)\w*\s+\S+'
        matches = re.finditer(env_pattern, content, re.IGNORECASE)
        
        for match in matches:
            line_number = content[:match.start()].count('\n') + 1
            vuln_id = hashlib.sha256(f"{file_path}:{line_number}:env_secret".encode()).hexdigest()[:12]
            
            vulnerability = SecurityVulnerability(
                id=vuln_id,
                category=SecurityCategory.HARDCODED_SECRETS,
                threat_level=SecurityThreatLevel.HIGH,
                title="Secret in Dockerfile ENV",
                description="Potential secret exposed in Dockerfile ENV directive",
                file_path=str(file_path),
                line_number=line_number,
                cwe_id='CWE-798',
                remediation="Use build args or runtime environment variables for secrets",
                confidence=0.85
            )
            
            vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning."""
        skip_patterns = [
            '__pycache__', '.git', 'node_modules', 'venv', '.venv',
            'test_', '_test.py', 'tests/', '.pytest_cache',
            'build/', 'dist/', '.egg-info'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _calculate_overall_security_score(self, vulnerabilities: List[SecurityVulnerability], files_scanned: int) -> float:
        """Calculate overall security score for static analysis."""
        if not vulnerabilities:
            return 1.0
        
        # Weight vulnerabilities by threat level and confidence
        threat_weights = {
            SecurityThreatLevel.CRITICAL: 1.0,
            SecurityThreatLevel.HIGH: 0.7,
            SecurityThreatLevel.MEDIUM: 0.4,
            SecurityThreatLevel.LOW: 0.2,
            SecurityThreatLevel.INFO: 0.1
        }
        
        weighted_impact = sum(
            threat_weights.get(v.threat_level, 0.5) * v.confidence * (1 - v.false_positive_probability)
            for v in vulnerabilities
        )
        
        # Normalize by files scanned
        if files_scanned == 0:
            return 1.0
        
        impact_per_file = weighted_impact / files_scanned
        
        # Convert to 0-1 scale where 0 is worst, 1 is best
        return max(0.0, 1.0 - min(1.0, impact_per_file))
    
    def _calculate_config_security_score(self, vulnerabilities: List[SecurityVulnerability], files_scanned: int) -> float:
        """Calculate security score for configuration files."""
        if not vulnerabilities:
            return 1.0
        
        # Configuration vulnerabilities are generally more serious
        impact = sum(0.8 if v.threat_level in [SecurityThreatLevel.CRITICAL, SecurityThreatLevel.HIGH] else 0.4 
                    for v in vulnerabilities)
        
        if files_scanned == 0:
            return 1.0
        
        normalized_impact = impact / max(files_scanned, 1)
        return max(0.0, 1.0 - min(1.0, normalized_impact * 0.5))
    
    async def _aggregate_scan_results(self, scan_results: Dict[str, SecurityScanResult]) -> Dict[str, Any]:
        """Aggregate results from all security scans."""
        all_vulnerabilities = []
        total_files_scanned = 0
        scan_summaries = {}
        
        for scan_type, result in scan_results.items():
            all_vulnerabilities.extend(result.vulnerabilities)
            total_files_scanned += result.files_scanned
            
            scan_summaries[scan_type] = {
                'vulnerabilities_found': len(result.vulnerabilities),
                'files_scanned': result.files_scanned,
                'security_score': result.security_score,
                'scan_duration': result.scan_duration,
                'critical_count': result.critical_count,
                'high_count': result.high_count
            }
        
        # Categorize vulnerabilities
        vulnerability_categories = {}
        for vuln in all_vulnerabilities:
            category = vuln.category.value
            if category not in vulnerability_categories:
                vulnerability_categories[category] = []
            vulnerability_categories[category].append(vuln.to_dict())
        
        # Calculate overall metrics
        critical_vulns = [v for v in all_vulnerabilities if v.threat_level == SecurityThreatLevel.CRITICAL]
        high_vulns = [v for v in all_vulnerabilities if v.threat_level == SecurityThreatLevel.HIGH]
        
        return {
            'total_vulnerabilities': len(all_vulnerabilities),
            'critical_vulnerabilities': len(critical_vulns),
            'high_vulnerabilities': len(high_vulns),
            'total_files_scanned': total_files_scanned,
            'vulnerability_categories': vulnerability_categories,
            'scan_summaries': scan_summaries,
            'vulnerabilities': [v.to_dict() for v in all_vulnerabilities]
        }
    
    async def _perform_threat_assessment(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered threat assessment."""
        threat_assessment = {
            'overall_risk_level': 'LOW',
            'threat_vectors': [],
            'attack_surface_analysis': {},
            'risk_factors': [],
            'mitigation_priorities': []
        }
        
        total_vulns = aggregated_results['total_vulnerabilities']
        critical_vulns = aggregated_results['critical_vulnerabilities']
        high_vulns = aggregated_results['high_vulnerabilities']
        
        # Determine overall risk level
        if critical_vulns > 0:
            threat_assessment['overall_risk_level'] = 'CRITICAL'
        elif high_vulns > 3:
            threat_assessment['overall_risk_level'] = 'HIGH'
        elif high_vulns > 0 or total_vulns > 10:
            threat_assessment['overall_risk_level'] = 'MEDIUM'
        else:
            threat_assessment['overall_risk_level'] = 'LOW'
        
        # Identify threat vectors
        vuln_categories = aggregated_results['vulnerability_categories']
        
        if 'code_injection' in vuln_categories:
            threat_assessment['threat_vectors'].append({
                'vector': 'Code Injection',
                'severity': 'CRITICAL',
                'description': 'Remote code execution vulnerabilities detected'
            })
        
        if 'hardcoded_secrets' in vuln_categories:
            threat_assessment['threat_vectors'].append({
                'vector': 'Credential Exposure',
                'severity': 'HIGH',
                'description': 'Hardcoded credentials may be exposed'
            })
        
        if 'dependency_vulnerability' in vuln_categories:
            threat_assessment['threat_vectors'].append({
                'vector': 'Supply Chain',
                'severity': 'MEDIUM',
                'description': 'Vulnerable dependencies detected'
            })
        
        # Risk factors
        if critical_vulns > 0:
            threat_assessment['risk_factors'].append("Critical vulnerabilities present")
        
        if len(vuln_categories) > 3:
            threat_assessment['risk_factors'].append("Multiple vulnerability categories detected")
        
        # Mitigation priorities
        if critical_vulns > 0:
            threat_assessment['mitigation_priorities'].append({
                'priority': 1,
                'action': 'Fix critical vulnerabilities immediately',
                'timeline': 'Within 24 hours'
            })
        
        if high_vulns > 0:
            threat_assessment['mitigation_priorities'].append({
                'priority': 2,
                'action': 'Address high-severity vulnerabilities',
                'timeline': 'Within 1 week'
            })
        
        return threat_assessment
    
    async def _calculate_security_posture(self, aggregated_results: Dict[str, Any], 
                                        threat_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall security posture."""
        # Base security score from scan results
        scan_scores = [
            scan['security_score'] 
            for scan in aggregated_results['scan_summaries'].values()
        ]
        
        base_score = sum(scan_scores) / len(scan_scores) if scan_scores else 0.0
        
        # Adjust based on threat assessment
        risk_level = threat_assessment['overall_risk_level']
        risk_adjustments = {
            'CRITICAL': -0.5,
            'HIGH': -0.3,
            'MEDIUM': -0.1,
            'LOW': 0.0
        }
        
        adjusted_score = max(0.0, base_score + risk_adjustments.get(risk_level, 0.0))
        
        # Determine if security posture passes
        overall_passed = (
            adjusted_score >= 0.8 and
            aggregated_results['critical_vulnerabilities'] == 0 and
            aggregated_results['high_vulnerabilities'] <= 2
        )
        
        # Generate recommendations
        recommendations = []
        if aggregated_results['critical_vulnerabilities'] > 0:
            recommendations.append("Fix critical vulnerabilities immediately")
        
        if aggregated_results['high_vulnerabilities'] > 0:
            recommendations.append("Address high-severity vulnerabilities")
        
        if 'dependency_vulnerability' in aggregated_results['vulnerability_categories']:
            recommendations.append("Update vulnerable dependencies")
        
        if 'hardcoded_secrets' in aggregated_results['vulnerability_categories']:
            recommendations.append("Remove hardcoded secrets and use secure configuration")
        
        # Next actions
        next_actions = []
        if not overall_passed:
            next_actions.append("Implement security fixes before deployment")
            next_actions.append("Conduct security code review")
        
        next_actions.extend([
            "Set up automated security scanning in CI/CD",
            "Implement security monitoring and alerting"
        ])
        
        return {
            'security_score': adjusted_score,
            'overall_passed': overall_passed,
            'confidence': 0.9,
            'base_score': base_score,
            'risk_adjustment': risk_adjustments.get(risk_level, 0.0),
            'recommendations': recommendations,
            'next_actions': next_actions,
            'security_metrics': {
                'total_vulnerabilities': aggregated_results['total_vulnerabilities'],
                'critical_vulnerabilities': aggregated_results['critical_vulnerabilities'],
                'high_vulnerabilities': aggregated_results['high_vulnerabilities'],
                'files_scanned': aggregated_results['total_files_scanned'],
                'risk_level': risk_level
            }
        }