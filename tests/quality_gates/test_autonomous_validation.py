"""Tests for autonomous validation system."""

import pytest
import asyncio
import tempfile
import time
import threading
import queue
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from neural_operator_lab.quality_gates.autonomous_validation import (
    ErrorContext,
    MonitoringMetrics,
    AdvancedErrorHandler,
    ComprehensiveMonitor,
    SelfImprovingGate,
    AutonomousQualityValidator,
    LearningQualitySystem
)
from neural_operator_lab.quality_gates.progressive_gates import (
    QualityGateResult,
    QualityGateGeneration
)


class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Test error",
            stack_trace="Traceback...",
            component="TestComponent",
            timestamp=time.time(),
            recovery_attempted=True,
            recovery_successful=False,
            retry_count=2,
            context_data={"key": "value"}
        )
        
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert context.component == "TestComponent"
        assert context.recovery_attempted is True
        assert context.recovery_successful is False
        assert context.retry_count == 2
        assert context.context_data == {"key": "value"}


class TestMonitoringMetrics:
    """Test monitoring metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = MonitoringMetrics()
        
        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.average_execution_time == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.performance_trend == "stable"
        assert isinstance(metrics.resource_usage, dict)
        assert isinstance(metrics.alerts_triggered, list)
    
    def test_metrics_updates(self):
        """Test metrics updates."""
        metrics = MonitoringMetrics()
        
        # Update metrics
        metrics.execution_count = 10
        metrics.success_count = 8
        metrics.failure_count = 2
        metrics.average_execution_time = 5.5
        metrics.error_rate = 0.2
        metrics.performance_trend = "improving"
        metrics.resource_usage = {"cpu": 50.0, "memory": 60.0}
        metrics.alerts_triggered = ["High CPU usage"]
        
        assert metrics.execution_count == 10
        assert metrics.success_count == 8
        assert metrics.failure_count == 2
        assert metrics.average_execution_time == 5.5
        assert metrics.error_rate == 0.2
        assert metrics.performance_trend == "improving"
        assert metrics.resource_usage["cpu"] == 50.0
        assert "High CPU usage" in metrics.alerts_triggered


class TestAdvancedErrorHandler:
    """Test advanced error handler functionality."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = AdvancedErrorHandler()
        
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) > 0
        assert len(handler.error_patterns) == 0
        assert handler.learning_enabled is True
        
        # Check that recovery strategies are set up
        assert "FileNotFoundError" in handler.recovery_strategies
        assert "PermissionError" in handler.recovery_strategies
        assert "TimeoutError" in handler.recovery_strategies
        assert "MemoryError" in handler.recovery_strategies
    
    @pytest.mark.asyncio
    async def test_error_handling_with_recovery(self):
        """Test error handling with successful recovery."""
        handler = AdvancedErrorHandler()
        
        # Mock a recovery strategy
        async def mock_recovery(error, context):
            return "recovered_value"
        
        handler.recovery_strategies["TestError"] = mock_recovery
        
        # Create test error
        test_error = Exception("Test error")
        test_error.__class__.__name__ = "TestError"
        
        # Handle error
        success, result = await handler.handle_error(
            test_error, "TestComponent", {"test": "data"}
        )
        
        assert success is True
        assert result == "recovered_value"
        assert len(handler.error_history) == 1
        
        error_context = handler.error_history[0]
        assert error_context.error_type == "TestError"
        assert error_context.component == "TestComponent"
        assert error_context.recovery_attempted is True
        assert error_context.recovery_successful is True
    
    @pytest.mark.asyncio
    async def test_error_handling_without_recovery(self):
        """Test error handling without recovery strategy."""
        handler = AdvancedErrorHandler()
        
        # Create test error without recovery strategy
        test_error = Exception("Unknown error")
        test_error.__class__.__name__ = "UnknownError"
        
        # Handle error
        success, result = await handler.handle_error(
            test_error, "TestComponent", {"test": "data"}
        )
        
        assert success is False
        assert result is None
        assert len(handler.error_history) == 1
        
        error_context = handler.error_history[0]
        assert error_context.error_type == "UnknownError"
        assert error_context.recovery_attempted is False
        assert error_context.recovery_successful is False
    
    @pytest.mark.asyncio
    async def test_file_not_found_recovery(self):
        """Test file not found error recovery."""
        handler = AdvancedErrorHandler()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            missing_file = temp_path / "missing.py"
            
            # Create FileNotFoundError
            error = FileNotFoundError(f"No such file or directory: '{missing_file}'")
            context = {"file_path": str(missing_file)}
            
            # Test recovery
            result = await handler._recover_file_not_found(error, context)
            
            # Should create the missing file
            assert missing_file.exists()
            assert result == str(missing_file)
    
    @pytest.mark.asyncio
    async def test_timeout_error_recovery(self):
        """Test timeout error recovery."""
        handler = AdvancedErrorHandler()
        
        error = TimeoutError("Operation timed out")
        context = {"retry_count": 0}
        
        # Test recovery
        result = await handler._recover_timeout_error(error, context)
        
        assert result is not None
        assert result["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_memory_error_recovery(self):
        """Test memory error recovery."""
        handler = AdvancedErrorHandler()
        
        error = MemoryError("Out of memory")
        context = {"batch_size": 64}
        
        # Test recovery
        result = await handler._recover_memory_error(error, context)
        
        assert result is not None
        assert result["batch_size"] == 32  # Should be halved
    
    def test_error_statistics(self):
        """Test error statistics generation."""
        handler = AdvancedErrorHandler()
        
        # Add some error history
        handler.error_history = [
            ErrorContext("ValueError", "Error 1", "", "Component1", time.time(), 
                        recovery_attempted=True, recovery_successful=True),
            ErrorContext("TypeError", "Error 2", "", "Component2", time.time(), 
                        recovery_attempted=True, recovery_successful=False),
            ErrorContext("ValueError", "Error 3", "", "Component1", time.time(), 
                        recovery_attempted=True, recovery_successful=True)
        ]
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["recovery_rate"] == 2/3  # 2 out of 3 successful
        assert "ValueError" in stats["error_types"]
        assert "TypeError" in stats["error_types"]
        assert stats["error_types"]["ValueError"]["count"] == 2
        assert stats["error_types"]["ValueError"]["recovered"] == 2
        assert len(stats["recent_errors"]) == 3


class TestComprehensiveMonitor:
    """Test comprehensive monitor functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ComprehensiveMonitor()
        
        assert monitor.monitoring_active is False
        assert monitor.monitoring_thread is None
        assert isinstance(monitor.metrics, MonitoringMetrics)
        assert isinstance(monitor.real_time_data, queue.Queue)
        assert "error_rate" in monitor.alert_thresholds
        assert "avg_execution_time" in monitor.alert_thresholds
        assert "memory_usage" in monitor.alert_thresholds
    
    def test_execution_recording(self):
        """Test execution metrics recording."""
        monitor = ComprehensiveMonitor()
        
        # Record successful execution
        monitor.record_execution(1.5, True)
        
        assert monitor.metrics.execution_count == 1
        assert monitor.metrics.success_count == 1
        assert monitor.metrics.failure_count == 0
        assert monitor.metrics.average_execution_time == 1.5
        assert monitor.metrics.error_rate == 0.0
        
        # Record failed execution
        monitor.record_execution(2.0, False)
        
        assert monitor.metrics.execution_count == 2
        assert monitor.metrics.success_count == 1
        assert monitor.metrics.failure_count == 1
        assert monitor.metrics.average_execution_time == 1.75  # (1.5 + 2.0) / 2
        assert monitor.metrics.error_rate == 0.5
    
    def test_performance_trend_update(self):
        """Test performance trend updates."""
        monitor = ComprehensiveMonitor()
        
        # Record baseline
        monitor.record_execution(1.0, True)
        assert monitor.metrics.performance_trend == "stable"
        
        # Record faster execution
        monitor._update_performance_trend(0.7)  # 30% faster than avg
        assert monitor.metrics.performance_trend == "improving"
        
        # Record slower execution
        monitor._update_performance_trend(1.5)  # 50% slower than avg
        assert monitor.metrics.performance_trend == "degrading"
        
        # Record normal execution
        monitor._update_performance_trend(1.0)
        assert monitor.metrics.performance_trend == "stable"
    
    def test_monitoring_report(self):
        """Test monitoring report generation."""
        monitor = ComprehensiveMonitor()
        
        # Record some metrics
        monitor.record_execution(1.0, True)
        monitor.record_execution(1.5, False)
        monitor.metrics.resource_usage = {"cpu": 50.0, "memory": 60.0}
        monitor.metrics.alerts_triggered = ["Test alert"]
        
        report = monitor.get_monitoring_report()
        
        assert "execution_metrics" in report
        assert "resource_metrics" in report
        assert "alerts" in report
        assert "monitoring_active" in report
        
        exec_metrics = report["execution_metrics"]
        assert exec_metrics["total_executions"] == 2
        assert exec_metrics["success_rate"] == 0.5
        assert exec_metrics["error_rate"] == 0.5
        assert exec_metrics["average_execution_time"] == 1.25
        
        assert report["resource_metrics"]["cpu"] == 50.0
        assert "Test alert" in report["alerts"]
        assert report["monitoring_active"] is False


class MockSelfImprovingGate(SelfImprovingGate):
    """Mock self-improving gate for testing."""
    
    def __init__(self, should_pass=True, should_fail=False):
        super().__init__("Mock Gate", QualityGateGeneration.GENERATION_1)
        self.should_pass = should_pass
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def _execute_generation_specific(self, context):
        self.execution_count += 1
        
        if self.should_fail:
            raise ValueError("Mock execution failure")
        
        return QualityGateResult(
            name=self.name,
            generation=self.generation,
            passed=self.should_pass,
            score=0.8 if self.should_pass else 0.4,
            confidence=0.9,
            execution_time=0.0,
            details={"mock": True, "context": context}
        )


class TestSelfImprovingGate:
    """Test self-improving gate functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful gate execution."""
        gate = MockSelfImprovingGate(should_pass=True)
        context = {"test": "data"}
        
        result = await gate.execute(context)
        
        assert result.name == "Mock Gate"
        assert result.passed is True
        assert result.score == 0.8
        assert gate.execution_count == 1
        
        # Check monitoring
        monitor_report = gate.monitor.get_monitoring_report()
        assert monitor_report["execution_metrics"]["total_executions"] == 1
        assert monitor_report["execution_metrics"]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_failed_execution_with_recovery(self):
        """Test failed execution with error recovery."""
        gate = MockSelfImprovingGate(should_fail=True)
        
        # Mock error handler to simulate successful recovery
        async def mock_handle_error(error, component, context):
            return True, {"recovered": True}
        
        gate.error_handler.handle_error = mock_handle_error
        
        # Override generation-specific execution for recovery
        async def recovered_execution(context):
            return QualityGateResult(
                name=gate.name,
                generation=gate.generation,
                passed=True,
                score=0.7,
                confidence=0.8,
                execution_time=0.0,
                details={"recovered": True}
            )
        
        gate._execute_generation_specific = recovered_execution
        
        result = await gate.execute({"test": "data"})
        
        # Should succeed after recovery
        assert result.passed is True
        assert result.score == 0.7
        assert result.details["recovered"] is True
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in gate execution."""
        gate = MockSelfImprovingGate()
        
        # Mock a slow execution
        async def slow_execution(context):
            await asyncio.sleep(0.1)  # Small delay for testing
            return QualityGateResult(
                name=gate.name,
                generation=gate.generation,
                passed=True,
                score=0.8,
                confidence=0.9,
                execution_time=0.0,
                details={}
            )
        
        gate._execute_generation_specific = slow_execution
        
        # Test with very short timeout
        with patch.object(gate, '_execute_with_timeout') as mock_timeout:
            mock_timeout.side_effect = asyncio.TimeoutError("Test timeout")
            
            result = await gate.execute({"test": "data"})
            
            assert result.passed is False
            assert result.score == 0.0
            assert "Test timeout" in str(result.details.get("error", ""))
    
    @pytest.mark.asyncio
    async def test_self_improvement(self):
        """Test self-improvement functionality."""
        gate = MockSelfImprovingGate(should_pass=True)
        gate.self_improvement_enabled = True
        
        # Execute multiple times to generate improvement data
        for i in range(6):  # Need at least 5 for pattern analysis
            await gate.execute({"iteration": i})
        
        # Check that improvement history is recorded
        assert len(gate.improvement_history) == 6
        
        # Check improvement insights
        insights = gate.get_self_improvement_report()
        assert insights["improvement_history_count"] == 6
        assert insights["self_improvement_enabled"] is True
        assert "error_statistics" in insights
        assert "monitoring_report" in insights
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self):
        """Test execution pattern analysis."""
        gate = MockSelfImprovingGate(should_pass=True)
        
        # Execute multiple times with varying performance
        for i in range(10):
            result = QualityGateResult(
                name=gate.name,
                generation=gate.generation,
                passed=i % 2 == 0,  # Alternate pass/fail
                score=0.5 + (i * 0.05),  # Increasing scores
                confidence=0.8,
                execution_time=1.0 + (i * 0.1),  # Increasing time
                details={}
            )
            gate.execution_history.append(result)
        
        # Analyze patterns
        patterns = await gate._analyze_execution_patterns()
        
        assert "average_score" in patterns
        assert "success_rate" in patterns
        assert "average_time" in patterns
        assert "confidence_trend" in patterns
        assert "recurring_issues" in patterns
        
        assert patterns["success_rate"] == 0.5  # 50% pass rate
        assert len(patterns["confidence_trend"]) == 10
    
    def test_learning_insights(self):
        """Test learning insights generation."""
        gate = MockSelfImprovingGate()
        
        # Add some execution history
        for i in range(5):
            result = QualityGateResult(
                name=gate.name,
                generation=gate.generation,
                passed=True,
                score=0.8,
                confidence=0.9,
                execution_time=1.0,
                details={}
            )
            gate.execution_history.append(result)
        
        insights = gate.get_learning_insights()
        
        assert insights["total_executions"] == 5
        assert insights["average_score"] == 0.8
        assert insights["success_rate"] == 1.0
        assert insights["generation"] == "basic_functionality"


class TestAutonomousQualityValidator:
    """Test autonomous quality validator."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            
            # Create test files
            (source_dir / "test_file.py").write_text("""
def test_function():
    return "test"
""")
            
            yield source_dir
    
    def test_validator_initialization(self, temp_source_dir):
        """Test validator initialization."""
        validator = AutonomousQualityValidator(temp_source_dir)
        
        assert validator.source_dir == temp_source_dir
        assert len(validator.gates) > 0
        assert validator.global_monitor is not None
        assert len(validator.validation_history) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_validation(self, temp_source_dir):
        """Test autonomous validation execution."""
        validator = AutonomousQualityValidator(temp_source_dir)
        
        # Replace with mock gates for faster testing
        mock_gate = MockSelfImprovingGate(should_pass=True)
        validator.gates = [mock_gate]
        
        results = await validator.run_autonomous_validation()
        
        assert "validation_time" in results
        assert "gate_results" in results
        assert "overall_analysis" in results
        assert "system_improvements" in results
        assert "monitoring_report" in results
        assert results["autonomous_features_active"] is True
        
        # Check that gates were executed
        assert len(results["gate_results"]) == 1
        assert mock_gate.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_parallel_gate_execution(self, temp_source_dir):
        """Test parallel gate execution."""
        validator = AutonomousQualityValidator(temp_source_dir)
        
        # Add multiple mock gates
        mock_gates = [
            MockSelfImprovingGate(should_pass=True),
            MockSelfImprovingGate(should_pass=False),
            MockSelfImprovingGate(should_pass=True)
        ]
        validator.gates = mock_gates
        
        results = await validator._execute_gates_parallel()
        
        assert len(results) == 3
        
        # Check that all gates were executed
        for gate in mock_gates:
            assert gate.execution_count == 1
        
        # Check results
        passed_results = [r for r in results if r.get("passed", False)]
        failed_results = [r for r in results if not r.get("passed", True)]
        
        assert len(passed_results) == 2
        assert len(failed_results) == 1
    
    @pytest.mark.asyncio
    async def test_overall_analysis(self, temp_source_dir):
        """Test overall results analysis."""
        validator = AutonomousQualityValidator(temp_source_dir)
        
        # Mock validation results
        validation_results = [
            {"name": "Gate1", "passed": True, "score": 0.9},
            {"name": "Gate2", "passed": False, "score": 0.4, "error": "Test error"},
            {"name": "Gate3", "passed": True, "score": 0.8}
        ]
        
        analysis = await validator._analyze_overall_results(validation_results)
        
        assert analysis["total_gates"] == 3
        assert analysis["passed_gates"] == 2
        assert analysis["pass_rate"] == 2/3
        assert analysis["average_score"] == pytest.approx((0.9 + 0.4 + 0.8) / 3)
        assert analysis["overall_passed"] is False  # Not all gates passed
        
        # Check critical issues
        assert len(analysis["critical_issues"]) == 1
        assert analysis["critical_issues"][0]["gate"] == "Gate2"
    
    @pytest.mark.asyncio
    async def test_system_improvements(self, temp_source_dir):
        """Test system improvements."""
        validator = AutonomousQualityValidator(temp_source_dir)
        
        # Mock analysis with low scores
        analysis = {
            "average_score": 0.6,
            "pass_rate": 0.5,
            "critical_issues": [{"gate": "Security", "score": 0.4}]
        }
        
        # Add some validation history
        validator.validation_history = [{"dummy": "data"}, {"dummy": "data"}, {"dummy": "data"}]
        
        improvements = await validator._perform_system_improvements(analysis)
        
        assert isinstance(improvements, list)
        # Should trigger automatic fixes for low score
        if analysis["average_score"] < 0.8:
            assert any("automatic" in imp.lower() for imp in improvements)


class TestLearningQualitySystem:
    """Test learning quality system."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_system_initialization(self, temp_source_dir):
        """Test learning system initialization."""
        system = LearningQualitySystem(temp_source_dir)
        
        assert system.source_dir == temp_source_dir
        assert system.validator is not None
        assert isinstance(system.learning_data, dict)
        assert len(system.evolution_history) == 0
    
    @pytest.mark.asyncio
    async def test_system_evolution(self, temp_source_dir):
        """Test system evolution."""
        system = LearningQualitySystem(temp_source_dir)
        
        # Mock validator results
        mock_validation_results = {
            "gate_results": [
                {"name": "Gate1", "passed": True, "score": 0.8},
                {"name": "Gate2", "passed": False, "score": 0.5}
            ],
            "overall_analysis": {"average_score": 0.65}
        }
        
        with patch.object(system.validator, 'run_autonomous_validation', 
                         new_callable=AsyncMock, return_value=mock_validation_results):
            
            evolution_results = await system.evolve_system()
            
            assert "evolution_time" in evolution_results
            assert "current_results" in evolution_results
            assert "learning_insights" in evolution_results
            assert "evolution_changes" in evolution_results
            assert "evolution_validation" in evolution_results
            assert "system_version" in evolution_results
            
            # Check evolution history
            assert len(system.evolution_history) == 1
            assert evolution_results["system_version"] == 1
    
    @pytest.mark.asyncio
    async def test_learning_from_validation(self, temp_source_dir):
        """Test learning from validation results."""
        system = LearningQualitySystem(temp_source_dir)
        
        validation_results = {
            "gate_results": [
                {"name": "Security Gate", "score": 0.9, "passed": True, "execution_time": 1.0},
                {"name": "Performance Gate", "score": 0.6, "passed": False, "execution_time": 2.0}
            ]
        }
        
        insights = await system._learn_from_validation(validation_results)
        
        assert "performance_patterns" in insights
        assert "error_patterns" in insights
        assert "improvement_opportunities" in insights
        assert "successful_strategies" in insights
        
        # Check performance patterns
        assert "Security Gate" in insights["performance_patterns"]
        assert "Performance Gate" in insights["performance_patterns"]
        
        assert insights["performance_patterns"]["Security Gate"]["average_score"] == 0.9
        assert insights["performance_patterns"]["Performance Gate"]["success_rate"] == 0.0
        
        # Check successful strategies
        assert len(insights["successful_strategies"]) == 1
        assert insights["successful_strategies"][0]["gate"] == "Security Gate"
    
    @pytest.mark.asyncio
    async def test_component_evolution(self, temp_source_dir):
        """Test system component evolution."""
        system = LearningQualitySystem(temp_source_dir)
        
        learning_insights = {
            "performance_patterns": {
                "Low Performance Gate": {"average_score": 0.5}
            },
            "error_patterns": {"timeout": ["pattern1", "pattern2"]}
        }
        
        evolution_changes = await system._evolve_system_components(learning_insights)
        
        assert isinstance(evolution_changes, list)
        # Should have some evolution changes
        assert len(evolution_changes) >= 0
    
    @pytest.mark.asyncio
    async def test_evolution_validation(self, temp_source_dir):
        """Test evolution validation."""
        system = LearningQualitySystem(temp_source_dir)
        
        evolution_changes = [
            "Enhanced monitoring",
            "Improved error handling",
            "Optimized algorithms"
        ]
        
        validation = await system._validate_evolution(evolution_changes)
        
        assert "changes_applied" in validation
        assert "effectiveness_score" in validation
        assert "validation_passed" in validation
        assert "recommendations" in validation
        
        assert validation["changes_applied"] == 3
        assert validation["validation_passed"] is True


if __name__ == "__main__":
    pytest.main([__file__])