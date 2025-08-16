"""Tests for progressive quality gates system."""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from neural_operator_lab.quality_gates.progressive_gates import (
    QualityGateResult,
    QualityGateGeneration,
    AdaptiveQualityGate,
    IntelligentSecurityGate,
    ProgressiveQualityGateSystem,
    IntelligentQualityOrchestrator
)


class TestQualityGateResult:
    """Test quality gate result functionality."""
    
    def test_quality_gate_result_creation(self):
        """Test creating quality gate results."""
        result = QualityGateResult(
            name="Test Gate",
            generation=QualityGateGeneration.GENERATION_1,
            passed=True,
            score=0.85,
            confidence=0.9,
            execution_time=1.5,
            details={"test": "data"}
        )
        
        assert result.name == "Test Gate"
        assert result.generation == QualityGateGeneration.GENERATION_1
        assert result.passed is True
        assert result.score == 0.85
        assert result.confidence == 0.9
        assert result.execution_time == 1.5
        assert result.details == {"test": "data"}
    
    def test_quality_level_assessment(self):
        """Test quality level assessment."""
        test_cases = [
            (0.96, "EXCEPTIONAL"),
            (0.92, "EXCELLENT"),
            (0.85, "GOOD"),
            (0.75, "ACCEPTABLE"),
            (0.65, "NEEDS_IMPROVEMENT"),
            (0.45, "CRITICAL")
        ]
        
        for score, expected_level in test_cases:
            result = QualityGateResult(
                name="Test",
                generation=QualityGateGeneration.GENERATION_1,
                passed=True,
                score=score,
                confidence=0.8,
                execution_time=1.0,
                details={}
            )
            assert result.quality_level == expected_level
    
    def test_to_dict_conversion(self):
        """Test converting result to dictionary."""
        result = QualityGateResult(
            name="Test Gate",
            generation=QualityGateGeneration.GENERATION_2,
            passed=False,
            score=0.6,
            confidence=0.7,
            execution_time=2.0,
            details={"error": "test error"},
            recommendations=["Fix error"],
            next_actions=["Review code"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "Test Gate"
        assert result_dict["generation"] == "robust_monitoring"
        assert result_dict["passed"] is False
        assert result_dict["score"] == 0.6
        assert result_dict["confidence"] == 0.7
        assert result_dict["quality_level"] == "NEEDS_IMPROVEMENT"
        assert result_dict["recommendations"] == ["Fix error"]
        assert result_dict["next_actions"] == ["Review code"]
    
    def test_score_validation(self):
        """Test score validation."""
        with pytest.raises(AssertionError):
            QualityGateResult(
                name="Test",
                generation=QualityGateGeneration.GENERATION_1,
                passed=True,
                score=1.5,  # Invalid score > 1.0
                confidence=0.8,
                execution_time=1.0,
                details={}
            )
        
        with pytest.raises(AssertionError):
            QualityGateResult(
                name="Test",
                generation=QualityGateGeneration.GENERATION_1,
                passed=True,
                score=0.8,
                confidence=1.5,  # Invalid confidence > 1.0
                execution_time=1.0,
                details={}
            )


class MockAdaptiveGate(AdaptiveQualityGate):
    """Mock adaptive gate for testing."""
    
    def __init__(self, name: str, generation: QualityGateGeneration, should_pass: bool = True):
        super().__init__(name, generation)
        self.should_pass = should_pass
        self.execution_count = 0
    
    async def execute_generation_1(self, context):
        self.execution_count += 1
        return QualityGateResult(
            name=self.name,
            generation=self.generation,
            passed=self.should_pass,
            score=0.8 if self.should_pass else 0.4,
            confidence=0.9,
            execution_time=0.1,
            details={"generation": "1", "context": context}
        )
    
    async def execute_generation_2(self, context):
        self.execution_count += 1
        return QualityGateResult(
            name=self.name,
            generation=self.generation,
            passed=self.should_pass,
            score=0.85 if self.should_pass else 0.3,
            confidence=0.95,
            execution_time=0.15,
            details={"generation": "2", "context": context}
        )
    
    async def execute_generation_3(self, context):
        self.execution_count += 1
        return QualityGateResult(
            name=self.name,
            generation=self.generation,
            passed=self.should_pass,
            score=0.9 if self.should_pass else 0.2,
            confidence=0.98,
            execution_time=0.2,
            details={"generation": "3", "context": context}
        )


class TestAdaptiveQualityGate:
    """Test adaptive quality gate functionality."""
    
    @pytest.mark.asyncio
    async def test_gate_execution(self):
        """Test gate execution across generations."""
        gate = MockAdaptiveGate("Test Gate", QualityGateGeneration.GENERATION_1, should_pass=True)
        context = {"test": "data"}
        
        result = await gate.execute(context)
        
        assert result.name == "Test Gate"
        assert result.passed is True
        assert result.score == 0.8
        assert result.execution_time > 0
        assert gate.execution_count == 1
        assert len(gate.execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_generation_evolution(self):
        """Test execution across different generations."""
        gate = MockAdaptiveGate("Test Gate", QualityGateGeneration.GENERATION_1)
        context = {"test": "data"}
        
        # Generation 1
        result1 = await gate.execute(context)
        assert result1.score == 0.8
        
        # Evolution to Generation 2
        gate.generation = QualityGateGeneration.GENERATION_2
        result2 = await gate.execute(context)
        assert result2.score == 0.85
        
        # Evolution to Generation 3
        gate.generation = QualityGateGeneration.GENERATION_3
        result3 = await gate.execute(context)
        assert result3.score == 0.9
        
        assert gate.execution_count == 3
        assert len(gate.execution_history) == 3
    
    @pytest.mark.asyncio
    async def test_learning_from_execution(self):
        """Test learning from execution history."""
        gate = MockAdaptiveGate("Test Gate", QualityGateGeneration.GENERATION_1)
        context = {"test": "data", "pattern": "test_pattern"}
        
        # Execute multiple times
        for _ in range(5):
            await gate.execute(context)
        
        # Check learning data
        learning_insights = gate.get_learning_insights()
        assert learning_insights["total_executions"] == 5
        assert learning_insights["average_score"] == 0.8
        assert learning_insights["success_rate"] == 1.0
        assert len(gate.learning_data) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in gate execution."""
        gate = MockAdaptiveGate("Test Gate", QualityGateGeneration.GENERATION_1)
        
        # Mock a method to raise an exception
        async def failing_execute(context):
            raise ValueError("Test error")
        
        gate.execute_generation_1 = failing_execute
        
        result = await gate.execute({"test": "data"})
        
        assert result.passed is False
        assert result.score == 0.0
        assert "Test error" in result.details["error"]
        assert "Fix error: Test error" in result.recommendations


class TestIntelligentSecurityGate:
    """Test intelligent security gate functionality."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            
            # Create test Python files
            (source_dir / "test_file.py").write_text("""
import subprocess
import os

def test_function():
    password = "secret123"
    subprocess.call("ls", shell=True)
    os.system("echo test")
    exec("print('test')")
""")
            
            (source_dir / "safe_file.py").write_text("""
import numpy as np

def safe_function():
    return np.array([1, 2, 3])
""")
            
            yield source_dir
    
    @pytest.mark.asyncio
    async def test_generation_1_execution(self, temp_source_dir):
        """Test Generation 1 security validation."""
        gate = IntelligentSecurityGate()
        context = {"source_dir": temp_source_dir}
        
        result = await gate.execute_generation_1(context)
        
        assert result.name == "Intelligent Security Validation"
        assert result.generation == QualityGateGeneration.GENERATION_1
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert result.details["files_scanned"] >= 2
        
        # Should detect vulnerabilities in test_file.py
        vulnerabilities = result.details["vulnerabilities"]
        assert len(vulnerabilities) > 0
        
        # Check for specific vulnerability types
        vuln_types = [v["type"] for v in vulnerabilities]
        assert "code_injection" in vuln_types
        assert "hardcoded_secrets" in vuln_types
    
    @pytest.mark.asyncio
    async def test_generation_2_execution(self, temp_source_dir):
        """Test Generation 2 security validation."""
        gate = IntelligentSecurityGate()
        gate.generation = QualityGateGeneration.GENERATION_2
        context = {"source_dir": temp_source_dir}
        
        result = await gate.execute_generation_2(context)
        
        assert result.generation == QualityGateGeneration.GENERATION_2
        assert result.confidence >= 0.9  # Generation 2 has higher confidence
        assert "security_analysis" in result.details
        assert "monitoring_status" in result.details
        assert "threat_validation" in result.details
        assert result.details["real_time_monitoring"] is True
    
    @pytest.mark.asyncio
    async def test_generation_3_execution(self, temp_source_dir):
        """Test Generation 3 security validation."""
        gate = IntelligentSecurityGate()
        gate.generation = QualityGateGeneration.GENERATION_3
        context = {"source_dir": temp_source_dir}
        
        result = await gate.execute_generation_3(context)
        
        assert result.generation == QualityGateGeneration.GENERATION_3
        assert result.confidence >= 0.95  # Generation 3 has highest confidence
        assert "scan_results" in result.details
        assert "risk_assessment" in result.details
        assert "adaptive_posture" in result.details
        assert result.details["ai_powered"] is True
        assert result.details["scaling_enabled"] is True
    
    def test_should_skip_file(self):
        """Test file skipping logic."""
        gate = IntelligentSecurityGate()
        
        # Files that should be skipped
        skip_files = [
            Path("__pycache__/test.py"),
            Path(".git/config"),
            Path("node_modules/package.json"),
            Path("venv/lib/python3.9/site-packages/test.py"),
            Path("test_file.py")
        ]
        
        for file_path in skip_files:
            assert gate._should_skip_file(file_path)
        
        # Files that should not be skipped
        keep_files = [
            Path("src/main.py"),
            Path("neural_operator_lab/models.py"),
            Path("scripts/train.py")
        ]
        
        for file_path in keep_files:
            assert not gate._should_skip_file(file_path)
    
    @pytest.mark.asyncio
    async def test_file_security_analysis(self, temp_source_dir):
        """Test individual file security analysis."""
        gate = IntelligentSecurityGate()
        test_file = temp_source_dir / "test_file.py"
        
        analysis = await gate._scan_file_basic(test_file)
        
        assert len(analysis) > 0
        
        # Check for detected vulnerabilities
        vuln_types = [v["type"] for v in analysis]
        assert "code_injection" in vuln_types
        assert "hardcoded_secrets" in vuln_types
        
        # Check vulnerability details
        for vuln in analysis:
            assert "file" in vuln
            assert "line" in vuln
            assert "type" in vuln
            assert "severity" in vuln
    
    def test_security_score_calculation(self):
        """Test security score calculation."""
        gate = IntelligentSecurityGate()
        
        # No vulnerabilities should give perfect score
        score = gate._calculate_security_score([], 10)
        assert score == 1.0
        
        # Some vulnerabilities should reduce score
        vulnerabilities = [
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
            {"severity": "LOW"}
        ]
        score = gate._calculate_security_score(vulnerabilities, 10)
        assert 0.0 <= score < 1.0
        
        # More vulnerabilities should reduce score further
        many_vulnerabilities = vulnerabilities * 5
        score_many = gate._calculate_security_score(many_vulnerabilities, 10)
        assert score_many < score


class TestProgressiveQualityGateSystem:
    """Test progressive quality gate system."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_system_initialization(self, temp_source_dir):
        """Test system initialization."""
        system = ProgressiveQualityGateSystem(temp_source_dir)
        
        assert system.source_dir == temp_source_dir
        assert system.current_generation == QualityGateGeneration.GENERATION_1
        assert len(system.gates) > 0
        assert len(system.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_progressive_validation(self, temp_source_dir):
        """Test progressive validation execution."""
        system = ProgressiveQualityGateSystem(temp_source_dir)
        
        # Add mock gate for testing
        mock_gate = MockAdaptiveGate("Mock Gate", QualityGateGeneration.GENERATION_1)
        system.gates = [mock_gate]
        
        results = await system.execute_progressive_validation()
        
        assert "generations_completed" in results
        assert "total_execution_time" in results
        assert "overall_passed" in results
        assert "generation_results" in results
        
        # Check that all generations were executed
        assert len(results["generations_completed"]) == 3
        assert "basic_functionality" in results["generations_completed"]
        assert "robust_monitoring" in results["generations_completed"]
        assert "optimized_scaling" in results["generations_completed"]
        
        # Check gate was executed for each generation
        assert mock_gate.execution_count == 3
    
    @pytest.mark.asyncio
    async def test_generation_execution(self, temp_source_dir):
        """Test individual generation execution."""
        system = ProgressiveQualityGateSystem(temp_source_dir)
        
        # Add mock gate
        mock_gate = MockAdaptiveGate("Mock Gate", QualityGateGeneration.GENERATION_1, should_pass=True)
        system.gates = [mock_gate]
        
        # Test Generation 1
        gen1_results = await system._execute_generation(QualityGateGeneration.GENERATION_1)
        
        assert gen1_results["generation"] == "basic_functionality"
        assert gen1_results["gates_executed"] == 1
        assert gen1_results["gates_passed"] == 1
        assert gen1_results["passed"] is True
        assert len(gen1_results["gate_results"]) == 1
        assert gen1_results["average_score"] > 0
    
    @pytest.mark.asyncio
    async def test_gate_evolution(self, temp_source_dir):
        """Test gate evolution through generations."""
        system = ProgressiveQualityGateSystem(temp_source_dir)
        
        # Add mock gate
        mock_gate = MockAdaptiveGate("Mock Gate", QualityGateGeneration.GENERATION_1)
        system.gates = [mock_gate]
        
        # Initially Generation 1
        assert mock_gate.generation == QualityGateGeneration.GENERATION_1
        
        # Evolve to Generation 2
        await system._evolve_gates_to_generation(QualityGateGeneration.GENERATION_2)
        assert mock_gate.generation == QualityGateGeneration.GENERATION_2
        
        # Evolve to Generation 3
        await system._evolve_gates_to_generation(QualityGateGeneration.GENERATION_3)
        assert mock_gate.generation == QualityGateGeneration.GENERATION_3
    
    def test_system_insights(self, temp_source_dir):
        """Test system insights generation."""
        system = ProgressiveQualityGateSystem(temp_source_dir)
        
        insights = system.get_system_insights()
        
        assert "current_generation" in insights
        assert "total_gates" in insights
        assert "execution_history_count" in insights
        assert "gate_insights" in insights
        
        assert insights["current_generation"] == "basic_functionality"
        assert insights["total_gates"] == len(system.gates)
        assert insights["execution_history_count"] == 0
        assert len(insights["gate_insights"]) == len(system.gates)


class TestIntelligentQualityOrchestrator:
    """Test intelligent quality orchestrator."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            
            # Create basic project structure
            (source_dir / "README.md").write_text("# Test Project")
            (source_dir / "src").mkdir()
            (source_dir / "src" / "main.py").write_text("def main(): pass")
            
            yield source_dir
    
    def test_orchestrator_initialization(self, temp_source_dir):
        """Test orchestrator initialization."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        assert orchestrator.source_dir == temp_source_dir
        assert orchestrator.quality_system is not None
        assert len(orchestrator.orchestration_history) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_validation(self, temp_source_dir):
        """Test autonomous validation orchestration."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        # Mock the quality system to avoid long execution times
        mock_validation_results = {
            "generations_completed": ["basic_functionality", "robust_monitoring", "optimized_scaling"],
            "total_execution_time": 1.0,
            "overall_passed": True,
            "generation_results": {
                "basic_functionality": {
                    "gates_executed": 1,
                    "gates_passed": 1,
                    "average_score": 0.85,
                    "passed": True,
                    "gate_results": [
                        {
                            "name": "Mock Gate",
                            "passed": True,
                            "score": 0.85,
                            "recommendations": []
                        }
                    ]
                }
            }
        }
        
        with patch.object(orchestrator.quality_system, 'execute_progressive_validation', 
                         new_callable=AsyncMock, return_value=mock_validation_results):
            
            results = await orchestrator.orchestrate_autonomous_validation()
            
            assert "orchestration_time" in results
            assert "validation_results" in results
            assert "analysis" in results
            assert "improvement_plan" in results
            assert "auto_improvements" in results
            assert "overall_quality_score" in results
            assert "production_ready" in results
            
            # Check orchestration history
            assert len(orchestrator.orchestration_history) == 1
    
    @pytest.mark.asyncio
    async def test_analysis_generation(self, temp_source_dir):
        """Test validation results analysis."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        # Mock validation results
        validation_results = {
            "overall_passed": True,
            "generation_results": {
                "basic_functionality": {
                    "average_score": 0.9,
                    "gate_results": [
                        {"name": "Gate1", "passed": True, "score": 0.9, "recommendations": []},
                        {"name": "Gate2", "passed": True, "score": 0.85, "recommendations": []}
                    ]
                },
                "robust_monitoring": {
                    "average_score": 0.8,
                    "gate_results": [
                        {"name": "Gate1", "passed": False, "score": 0.6, "recommendations": ["Fix issue"]}
                    ]
                }
            }
        }
        
        analysis = await orchestrator._analyze_validation_results(validation_results)
        
        assert "overall_quality_score" in analysis
        assert "production_ready" in analysis
        assert "critical_issues" in analysis
        assert "recommendations" in analysis
        assert "strengths" in analysis
        assert "weaknesses" in analysis
        
        # Should identify critical issues from failed gates
        assert len(analysis["critical_issues"]) > 0
        assert analysis["critical_issues"][0]["gate"] == "Gate1"
    
    @pytest.mark.asyncio
    async def test_improvement_plan_generation(self, temp_source_dir):
        """Test improvement plan generation."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        # Mock analysis with issues
        analysis = {
            "overall_quality_score": 0.6,
            "production_ready": False,
            "critical_issues": [
                {"gate": "Security Gate", "score": 0.4, "recommendations": ["Fix vulnerabilities"]}
            ],
            "recommendations": ["Improve code quality"]
        }
        
        improvement_plan = await orchestrator._generate_improvement_plan(analysis)
        
        assert "priority_actions" in improvement_plan
        assert "automated_fixes" in improvement_plan
        assert "manual_reviews" in improvement_plan
        assert "long_term_goals" in improvement_plan
        
        # Should have priority actions for critical issues
        assert len(improvement_plan["priority_actions"]) > 0
        assert improvement_plan["priority_actions"][0]["priority"] == "HIGH"
        
        # Should have automated fixes for low quality score
        assert len(improvement_plan["automated_fixes"]) > 0
    
    @pytest.mark.asyncio
    async def test_autonomous_improvements(self, temp_source_dir):
        """Test autonomous improvements execution."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        improvement_plan = {
            "automated_fixes": [
                "Run code formatting",
                "Fix import ordering",
                "Remove unused imports",
                "Dangerous operation"  # Should be skipped
            ]
        }
        
        auto_improvements = await orchestrator._execute_autonomous_improvements(improvement_plan)
        
        assert "executed" in auto_improvements
        assert "skipped" in auto_improvements
        assert "errors" in auto_improvements
        
        # Should execute safe fixes
        assert "Run code formatting" in auto_improvements["executed"]
        assert "Fix import ordering" in auto_improvements["executed"]
        assert "Remove unused imports" in auto_improvements["executed"]
        
        # Should skip unsafe fixes
        skipped_fixes = [item["fix"] for item in auto_improvements["skipped"]]
        assert "Dangerous operation" in skipped_fixes
    
    @pytest.mark.asyncio
    async def test_report_generation(self, temp_source_dir):
        """Test orchestration report generation."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        # Mock orchestration results
        orchestration_results = {
            "orchestration_time": 10.5,
            "overall_quality_score": 0.85,
            "production_ready": True,
            "validation_results": {
                "generation_results": {
                    "basic_functionality": {
                        "gates_executed": 2,
                        "gates_passed": 2,
                        "average_score": 0.85,
                        "passed": True
                    }
                }
            },
            "analysis": {
                "critical_issues": [],
                "recommendations": ["Continue monitoring"]
            },
            "improvement_plan": {
                "priority_actions": [],
                "automated_fixes": ["Code formatting"]
            },
            "auto_improvements": {
                "executed": ["Code formatting"],
                "skipped": [],
                "errors": []
            }
        }
        
        await orchestrator._generate_orchestration_report(orchestration_results)
        
        # Check that report file was created
        report_path = temp_source_dir / "quality_orchestration_report.md"
        assert report_path.exists()
        
        # Check report content
        report_content = report_path.read_text()
        assert "Quality Orchestration Report" in report_content
        assert "Executive Summary" in report_content
        assert "Overall Quality Score" in report_content
        assert "0.85" in report_content
        assert "✅ YES" in report_content  # Production ready
    
    def test_report_formatting(self, temp_source_dir):
        """Test report formatting functionality."""
        orchestrator = IntelligentQualityOrchestrator(temp_source_dir)
        
        results = {
            "overall_quality_score": 0.92,
            "production_ready": True,
            "orchestration_time": 15.3,
            "validation_results": {
                "generation_results": {
                    "basic_functionality": {
                        "gates_executed": 3,
                        "gates_passed": 3,
                        "average_score": 0.92,
                        "passed": True
                    },
                    "robust_monitoring": {
                        "gates_executed": 2,
                        "gates_passed": 1,
                        "average_score": 0.75,
                        "passed": False
                    }
                }
            },
            "analysis": {
                "critical_issues": [{"gate": "Security", "score": 0.6}],
                "recommendations": ["Improve security", "Add monitoring"]
            },
            "improvement_plan": {
                "priority_actions": [
                    {"action": "Fix security", "priority": "HIGH"},
                    {"action": "Add tests", "priority": "MEDIUM"}
                ]
            },
            "auto_improvements": {
                "executed": ["Format code", "Fix imports"],
                "skipped": [],
                "errors": []
            }
        }
        
        report_content = orchestrator._format_orchestration_report(results)
        
        # Check report structure
        assert "# Quality Orchestration Report" in report_content
        assert "## Executive Summary" in report_content
        assert "## Generation Results" in report_content
        assert "## Analysis" in report_content
        assert "## Improvement Plan" in report_content
        
        # Check specific content
        assert "0.92/1.00" in report_content
        assert "✅ YES" in report_content
        assert "15.3s" in report_content
        assert "Basic Functionality" in report_content
        assert "Robust Monitoring" in report_content
        assert "✅ PASSED" in report_content
        assert "❌ FAILED" in report_content
        assert "**HIGH**: Fix security" in report_content
        assert "✅ Format code" in report_content


if __name__ == "__main__":
    pytest.main([__file__])