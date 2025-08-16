"""Tests for research quality gates."""

import pytest
import asyncio
import tempfile
import json
import statistics
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from neural_operator_lab.quality_gates.research_quality_gates import (
    ResearchQualityMetrics,
    ExperimentalResult,
    PublicationReadinessAssessment,
    AIAnalysisEngine,
    DistributedQualityProcessor,
    ResearchQualityValidator,
    ExperimentalValidation,
    PublicationReadinessGate
)
from neural_operator_lab.quality_gates.progressive_gates import QualityGateGeneration


class TestExperimentalResult:
    """Test experimental result functionality."""
    
    def test_experimental_result_creation(self):
        """Test creating experimental results."""
        result = ExperimentalResult(
            experiment_name="test_experiment",
            baseline_performance=0.7,
            novel_performance=0.85,
            improvement_percentage=21.43,
            p_value=0.01,
            confidence_interval=(15.0, 28.0),
            sample_size=100,
            effect_size=0.8,
            statistical_power=0.9,
            reproducibility_score=0.95
        )
        
        assert result.experiment_name == "test_experiment"
        assert result.baseline_performance == 0.7
        assert result.novel_performance == 0.85
        assert result.improvement_percentage == 21.43
        assert result.p_value == 0.01
        assert result.sample_size == 100
        assert result.effect_size == 0.8
        assert result.statistical_power == 0.9
        assert result.reproducibility_score == 0.95
    
    def test_statistical_significance(self):
        """Test statistical significance assessment."""
        # Statistically significant result
        significant_result = ExperimentalResult(
            experiment_name="significant",
            baseline_performance=0.7,
            novel_performance=0.85,
            improvement_percentage=21.43,
            p_value=0.01,  # < 0.05
            confidence_interval=(15.0, 28.0),
            sample_size=100,
            effect_size=0.8,
            statistical_power=0.9,
            reproducibility_score=0.95
        )
        
        assert significant_result.is_statistically_significant is True
        
        # Not statistically significant result
        non_significant_result = ExperimentalResult(
            experiment_name="non_significant",
            baseline_performance=0.7,
            novel_performance=0.72,
            improvement_percentage=2.86,
            p_value=0.15,  # > 0.05
            confidence_interval=(-1.0, 6.7),
            sample_size=30,
            effect_size=0.1,
            statistical_power=0.3,
            reproducibility_score=0.8
        )
        
        assert non_significant_result.is_statistically_significant is False
    
    def test_practical_significance(self):
        """Test practical significance assessment."""
        # Practically significant result
        practical_result = ExperimentalResult(
            experiment_name="practical",
            baseline_performance=0.7,
            novel_performance=0.85,
            improvement_percentage=21.43,  # > 5%
            p_value=0.01,
            confidence_interval=(15.0, 28.0),
            sample_size=100,
            effect_size=0.8,  # > 0.3
            statistical_power=0.9,
            reproducibility_score=0.95
        )
        
        assert practical_result.has_practical_significance is True
        
        # Not practically significant result
        non_practical_result = ExperimentalResult(
            experiment_name="non_practical",
            baseline_performance=0.7,
            novel_performance=0.72,
            improvement_percentage=2.86,  # < 5%
            p_value=0.01,
            confidence_interval=(1.0, 4.7),
            sample_size=100,
            effect_size=0.1,  # < 0.3
            statistical_power=0.9,
            reproducibility_score=0.95
        )
        
        assert non_practical_result.has_practical_significance is False


class TestPublicationReadinessAssessment:
    """Test publication readiness assessment."""
    
    def test_assessment_creation(self):
        """Test creating publication readiness assessment."""
        assessment = PublicationReadinessAssessment(
            code_quality_score=0.9,
            documentation_completeness=0.85,
            experimental_rigor=0.88,
            reproducibility_score=0.92,
            novelty_assessment=0.8,
            benchmark_coverage=0.75,
            statistical_validity=0.9,
            overall_readiness=0.87,
            missing_components=["Extended documentation"],
            recommendations=["Improve benchmark coverage"]
        )
        
        assert assessment.code_quality_score == 0.9
        assert assessment.documentation_completeness == 0.85
        assert assessment.experimental_rigor == 0.88
        assert assessment.reproducibility_score == 0.92
        assert assessment.novelty_assessment == 0.8
        assert assessment.benchmark_coverage == 0.75
        assert assessment.statistical_validity == 0.9
        assert assessment.overall_readiness == 0.87
        assert "Extended documentation" in assessment.missing_components
        assert "Improve benchmark coverage" in assessment.recommendations
    
    def test_publication_readiness(self):
        """Test publication readiness determination."""
        # Ready for publication
        ready_assessment = PublicationReadinessAssessment(
            code_quality_score=0.9,
            documentation_completeness=0.85,
            experimental_rigor=0.88,
            reproducibility_score=0.92,
            novelty_assessment=0.8,
            benchmark_coverage=0.75,
            statistical_validity=0.95,
            overall_readiness=0.87
        )
        
        assert ready_assessment.is_publication_ready is True
        
        # Not ready for publication (low overall readiness)
        not_ready_assessment = PublicationReadinessAssessment(
            code_quality_score=0.7,
            documentation_completeness=0.6,
            experimental_rigor=0.65,
            reproducibility_score=0.8,
            novelty_assessment=0.5,
            benchmark_coverage=0.4,
            statistical_validity=0.7,
            overall_readiness=0.7  # < 0.85
        )
        
        assert not_ready_assessment.is_publication_ready is False
        
        # Not ready for publication (low reproducibility)
        low_repro_assessment = PublicationReadinessAssessment(
            code_quality_score=0.9,
            documentation_completeness=0.85,
            experimental_rigor=0.88,
            reproducibility_score=0.8,  # < 0.9
            novelty_assessment=0.8,
            benchmark_coverage=0.75,
            statistical_validity=0.95,
            overall_readiness=0.87
        )
        
        assert low_repro_assessment.is_publication_ready is False


class TestAIAnalysisEngine:
    """Test AI analysis engine functionality."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            
            # Create test files with neural operator patterns
            (source_dir / "models" / "transformer_operator.py").mkdir(parents=True, exist_ok=True)
            (source_dir / "models" / "transformer_operator.py").write_text("""
class TransformerNeuralOperator:
    def __init__(self):
        self.transformer_neural_operator = True
        
    def forward(self, x):
        return x
""")
            
            (source_dir / "models" / "probabilistic_operator.py").write_text("""
class ProbabilisticNeuralOperator:
    def __init__(self):
        self.probabilistic_neural_operator = True
        
    def forward_with_uncertainty(self, x):
        return x, uncertainty
""")
            
            yield source_dir
    
    def test_ai_engine_initialization(self):
        """Test AI engine initialization."""
        engine = AIAnalysisEngine()
        
        assert len(engine.analysis_cache) == 0
        assert engine.model_confidence_threshold == 0.8
        assert len(engine.analysis_history) == 0
    
    @pytest.mark.asyncio
    async def test_algorithmic_novelty_analysis(self, temp_source_dir):
        """Test algorithmic novelty analysis."""
        engine = AIAnalysisEngine()
        
        analysis = await engine.analyze_algorithmic_novelty(temp_source_dir)
        
        assert "novel_algorithms_detected" in analysis
        assert "innovation_score" in analysis
        assert "similarity_to_existing" in analysis
        assert "potential_contributions" in analysis
        assert "confidence" in analysis
        
        # Should detect novel patterns in test files
        novel_patterns = analysis["novel_algorithms_detected"]
        assert len(novel_patterns) > 0
        
        pattern_names = [p["pattern"] for p in novel_patterns]
        assert "transformer_neural_operator" in pattern_names
        assert "probabilistic_neural_operator" in pattern_names
        
        # Should have reasonable innovation score
        assert 0.0 <= analysis["innovation_score"] <= 1.0
        assert analysis["confidence"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_experimental_design_analysis(self):
        """Test experimental design analysis."""
        engine = AIAnalysisEngine()
        
        experiment_configs = [
            {
                "name": "neural_operator_benchmark",
                "baselines": ["fourier_neural_operator", "deep_operator_network"],
                "datasets": ["navier_stokes", "darcy_flow", "wave_equation"],
                "sample_size": 100,
                "statistical_validation": True,
                "reproducibility_measures": True
            },
            {
                "name": "ablation_study",
                "baselines": ["baseline"],
                "datasets": ["single_dataset"],
                "sample_size": 30,
                "statistical_validation": False,
                "reproducibility_measures": False
            }
        ]
        
        analysis = await engine.analyze_experimental_design(experiment_configs)
        
        assert "design_quality_score" in analysis
        assert "statistical_power_assessment" in analysis
        assert "bias_detection" in analysis
        assert "methodology_recommendations" in analysis
        assert "confidence" in analysis
        
        # Should have reasonable design quality
        assert 0.0 <= analysis["design_quality_score"] <= 1.0
        
        # Should assess power for both experiments
        power_assessment = analysis["statistical_power_assessment"]
        assert "neural_operator_benchmark" in power_assessment
        assert "ablation_study" in power_assessment
        
        # Should detect biases in second experiment
        biases = analysis["bias_detection"]
        assert len(biases) > 0
        
        # Should provide recommendations
        recommendations = analysis["methodology_recommendations"]
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_novel_pattern_detection(self, temp_source_dir):
        """Test novel pattern detection."""
        engine = AIAnalysisEngine()
        
        patterns = await engine._detect_novel_patterns(temp_source_dir)
        
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert "pattern" in pattern
            assert "file" in pattern
            assert "novelty_score" in pattern
            assert "description" in pattern
            assert 0.0 <= pattern["novelty_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_innovation_score_calculation(self):
        """Test innovation score calculation."""
        engine = AIAnalysisEngine()
        
        # Test with no patterns
        score_empty = await engine._calculate_innovation_score([])
        assert score_empty == 0.0
        
        # Test with patterns
        patterns = [
            {"novelty_score": 0.8, "pattern": "transformer_neural_operator"},
            {"novelty_score": 0.9, "pattern": "probabilistic_neural_operator"},
            {"novelty_score": 0.7, "pattern": "foundation_neural_operator"}
        ]
        
        score = await engine._calculate_innovation_score(patterns)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high with good patterns
    
    @pytest.mark.asyncio
    async def test_potential_contributions_identification(self):
        """Test potential contributions identification."""
        engine = AIAnalysisEngine()
        
        novel_patterns = [
            {"pattern": "probabilistic_neural_operator"},
            {"pattern": "transformer_neural_operator"},
            {"pattern": "foundation_neural_operator"}
        ]
        
        contributions = await engine._identify_potential_contributions(novel_patterns, 0.8)
        
        assert len(contributions) > 0
        assert "Uncertainty quantification in neural operators" in contributions
        assert "Attention mechanisms for PDE solving" in contributions
        assert "Foundation model approach for PDEs" in contributions


class TestDistributedQualityProcessor:
    """Test distributed quality processor."""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization."""
        async with DistributedQualityProcessor(max_workers=4) as processor:
            assert processor.max_workers == 4
            assert processor.process_pool is not None
            assert processor.thread_pool is not None
    
    def test_gate_partitioning(self):
        """Test gate partitioning for distributed processing."""
        processor = DistributedQualityProcessor(max_workers=4)
        
        # Create mock gates
        gates = [Mock(name=f"Gate{i}") for i in range(10)]
        
        partitions = processor._partition_gates(gates)
        
        assert len(partitions) <= 4  # Should not exceed max_workers
        assert sum(len(partition) for partition in partitions) == 10  # All gates included
    
    def test_small_gate_partitioning(self):
        """Test partitioning with fewer gates than workers."""
        processor = DistributedQualityProcessor(max_workers=8)
        
        gates = [Mock(name=f"Gate{i}") for i in range(3)]
        
        partitions = processor._partition_gates(gates)
        
        assert len(partitions) == 3  # One gate per partition
        assert all(len(partition) == 1 for partition in partitions)


class TestResearchQualityValidator:
    """Test research quality validator."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            
            # Create project structure
            (source_dir / "README.md").write_text("# Neural Operator Research")
            (source_dir / "src").mkdir()
            (source_dir / "tests").mkdir()
            (source_dir / "docs").mkdir()
            (source_dir / "examples").mkdir()
            
            # Create research metadata
            research_config = {
                "experiments": [
                    {
                        "name": "neural_operator_benchmark",
                        "baselines": ["fno", "deeponet"],
                        "datasets": ["navier_stokes", "darcy"],
                        "sample_size": 100
                    }
                ]
            }
            
            with open(source_dir / "research_config.json", "w") as f:
                json.dump(research_config, f)
            
            # Create test files with statistical patterns
            (source_dir / "src" / "analysis.py").write_text("""
import scipy.stats
from scipy import stats

def statistical_test():
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = cohen_d(group1, group2)
    return p_value, effect_size

def cohen_d(x, y):
    return (mean(x) - mean(y)) / pooled_std(x, y)
""")
            
            (source_dir / "src" / "config.py").write_text("""
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
""")
            
            # Create Docker support
            (source_dir / "Dockerfile").write_text("FROM python:3.9")
            (source_dir / "requirements.txt").write_text("torch>=1.0.0\nnumpy>=1.20.0")
            
            yield source_dir
    
    def test_validator_initialization(self, temp_source_dir):
        """Test validator initialization."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        assert validator.source_dir == temp_source_dir
        assert validator.ai_engine is not None
        assert len(validator.experimental_results) == 0
        assert len(validator.research_metadata) == 0
        assert len(validator.validation_cache) == 0
    
    @pytest.mark.asyncio
    async def test_research_metadata_loading(self, temp_source_dir):
        """Test research metadata loading."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        await validator._load_research_metadata()
        
        assert len(validator.research_metadata) > 0
        assert "experiments" in validator.research_metadata
        assert len(validator.research_metadata["experiments"]) == 1
    
    @pytest.mark.asyncio
    async def test_experimental_design_validation(self, temp_source_dir):
        """Test experimental design validation."""
        validator = ResearchQualityValidator(temp_source_dir)
        await validator._load_research_metadata()
        
        validation = await validator._validate_experimental_design()
        
        assert "design_quality" in validation
        assert "experiments_found" in validation
        assert "ai_analysis" in validation
        assert "baseline_validation" in validation
        assert "power_analysis" in validation
        assert "overall_score" in validation
        
        assert validation["experiments_found"] == 1
        assert 0.0 <= validation["overall_score"] <= 1.0
    
    def test_experiment_config_extraction(self, temp_source_dir):
        """Test experiment configuration extraction."""
        validator = ResearchQualityValidator(temp_source_dir)
        validator.research_metadata = {
            "experiments": [
                {
                    "name": "test_experiment",
                    "baselines": ["baseline1", "baseline2"],
                    "datasets": ["dataset1", "dataset2", "dataset3"]
                }
            ]
        }
        
        configs = validator._extract_experiment_configs()
        
        assert len(configs) == 1
        assert configs[0]["name"] == "test_experiment"
        assert len(configs[0]["baselines"]) == 2
        assert len(configs[0]["datasets"]) == 3
    
    def test_baseline_validation(self, temp_source_dir):
        """Test baseline validation."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        experiment_configs = [
            {
                "name": "good_experiment",
                "baselines": ["baseline1", "baseline2", "baseline3"]
            },
            {
                "name": "poor_experiment",
                "baselines": ["baseline1"]
            },
            {
                "name": "no_baselines",
                "baselines": []
            }
        ]
        
        validation = validator._validate_baselines(experiment_configs)
        
        assert "score" in validation
        assert "issues" in validation
        assert "recommendations" in validation
        
        # Should have decent score (good experiment gets 1.0, poor gets 0.4, none gets 0.0)
        expected_score = (1.0 + 0.4 + 0.0) / 3
        assert validation["score"] == pytest.approx(expected_score, rel=0.1)
    
    def test_statistical_power_analysis(self, temp_source_dir):
        """Test statistical power analysis."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        experiment_configs = [
            {
                "name": "high_power",
                "sample_size": 150,
                "expected_effect_size": 0.5
            },
            {
                "name": "medium_power",
                "sample_size": 60,
                "expected_effect_size": 0.3
            },
            {
                "name": "low_power",
                "sample_size": 20,
                "expected_effect_size": 0.1
            }
        ]
        
        analysis = validator._analyze_statistical_power(experiment_configs)
        
        assert "individual_powers" in analysis
        assert "average_power" in analysis
        assert "adequate_power_count" in analysis
        
        assert len(analysis["individual_powers"]) == 3
        
        # High power experiment should have power >= 0.8
        assert analysis["individual_powers"][0] >= 0.8
        
        # Low power experiment should have lower power
        assert analysis["individual_powers"][2] < analysis["individual_powers"][0]
    
    @pytest.mark.asyncio
    async def test_statistical_methodology_validation(self, temp_source_dir):
        """Test statistical methodology validation."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        validation = await validator._validate_statistical_methodology()
        
        assert "hypothesis_testing" in validation
        assert "multiple_comparisons" in validation
        assert "effect_size_reporting" in validation
        assert "confidence_intervals" in validation
        assert "statistical_assumptions" in validation
        assert "overall_score" in validation
        
        # Should find statistical tests in test files
        hypothesis_testing = validation["hypothesis_testing"]
        assert hypothesis_testing["score"] > 0
        assert "ttest" in hypothesis_testing["tests_found"]
        
        # Should find effect size calculations
        effect_size = validation["effect_size_reporting"]
        assert effect_size["score"] > 0
        assert "cohen_d" in effect_size["effects_found"]
    
    def test_hypothesis_testing_check(self, temp_source_dir):
        """Test hypothesis testing check."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        result = validator._check_hypothesis_testing()
        
        assert "score" in result
        assert "tests_found" in result
        assert "recommendations" in result
        
        # Should find statistical tests
        assert result["score"] > 0
        assert len(result["tests_found"]) > 0
        assert "ttest" in result["tests_found"] or "scipy.stats" in result["tests_found"]
    
    @pytest.mark.asyncio
    async def test_reproducibility_check(self, temp_source_dir):
        """Test reproducibility check."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        check = await validator._check_reproducibility()
        
        assert "random_seed_control" in check
        assert "environment_specification" in check
        assert "data_versioning" in check
        assert "code_determinism" in check
        assert "docker_support" in check
        assert "overall_score" in check
        
        # Should find seed control in config files
        seed_control = check["random_seed_control"]
        assert seed_control["score"] > 0
        
        # Should find environment specification
        env_spec = check["environment_specification"]
        assert env_spec["score"] > 0
        assert "requirements.txt" in env_spec["env_files_found"]
        
        # Should find Docker support
        docker_support = check["docker_support"]
        assert docker_support["score"] > 0
        assert "Dockerfile" in docker_support["docker_files_found"]
    
    def test_documentation_assessment(self, temp_source_dir):
        """Test documentation completeness assessment."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        score = validator._assess_documentation_completeness()
        
        # Should have README
        assert score >= 0.2  # At least 1/5 for README
        
        # Should be less than perfect since not all docs exist
        assert score < 1.0
    
    def test_benchmark_coverage_assessment(self, temp_source_dir):
        """Test benchmark coverage assessment."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        score = validator._assess_benchmark_coverage()
        
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_publication_readiness_assessment(self, temp_source_dir):
        """Test publication readiness assessment."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        # Mock component results
        experimental_validation = {"overall_score": 0.8}
        novelty_assessment = {"innovation_score": 0.7}
        statistical_validation = {"overall_score": 0.85}
        reproducibility_check = {"overall_score": 0.9}
        
        assessment = await validator._assess_publication_readiness(
            experimental_validation,
            novelty_assessment,
            statistical_validation,
            reproducibility_check
        )
        
        assert isinstance(assessment, PublicationReadinessAssessment)
        assert 0.0 <= assessment.overall_readiness <= 1.0
        assert 0.0 <= assessment.code_quality_score <= 1.0
        assert 0.0 <= assessment.experimental_rigor <= 1.0
        assert 0.0 <= assessment.reproducibility_score <= 1.0
        assert 0.0 <= assessment.novelty_assessment <= 1.0
        
        # Should have recommendations if not publication ready
        if not assessment.is_publication_ready:
            assert len(assessment.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_complete_research_validation(self, temp_source_dir):
        """Test complete research quality validation."""
        validator = ResearchQualityValidator(temp_source_dir)
        
        results = await validator.validate_research_quality()
        
        assert "validation_time" in results
        assert "experimental_validation" in results
        assert "novelty_assessment" in results
        assert "statistical_validation" in results
        assert "reproducibility_check" in results
        assert "publication_assessment" in results
        assert "overall_research_quality" in results
        
        assert results["validation_time"] > 0
        assert 0.0 <= results["overall_research_quality"] <= 1.0


class TestExperimentalValidation:
    """Test experimental validation functionality."""
    
    def test_experimental_validation_initialization(self):
        """Test experimental validation initialization."""
        validation = ExperimentalValidation()
        
        assert len(validation.experiments) == 0
        assert len(validation.validation_results) == 0
    
    @pytest.mark.asyncio
    async def test_single_experiment_execution(self):
        """Test single experiment execution."""
        validation = ExperimentalValidation()
        
        config = {
            "name": "test_experiment",
            "baseline_performance": 0.7,
            "novel_performance": 0.85,
            "sample_size": 100
        }
        
        result = await validation._run_single_experiment(config)
        
        assert isinstance(result, ExperimentalResult)
        assert result.experiment_name == "test_experiment"
        assert result.baseline_performance == 0.7
        assert result.novel_performance == 0.85
        assert result.sample_size == 100
        assert result.improvement_percentage > 0
        assert 0.0 <= result.p_value <= 1.0
        assert len(result.confidence_interval) == 2
        assert result.effect_size > 0
        assert 0.0 <= result.statistical_power <= 1.0
        assert 0.0 <= result.reproducibility_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_experimental_validation_execution(self):
        """Test experimental validation execution."""
        validation = ExperimentalValidation()
        
        experiment_configs = [
            {
                "name": "experiment_1",
                "baseline_performance": 0.7,
                "novel_performance": 0.85,
                "sample_size": 100
            },
            {
                "name": "experiment_2",
                "baseline_performance": 0.6,
                "novel_performance": 0.75,
                "sample_size": 50
            }
        ]
        
        results = await validation.run_experimental_validation(experiment_configs)
        
        assert "experiments_run" in results
        assert "statistically_significant" in results
        assert "practically_significant" in results
        assert "overall_validity" in results
        assert "individual_results" in results
        
        assert results["experiments_run"] == 2
        assert len(results["individual_results"]) == 2
        assert 0.0 <= results["overall_validity"] <= 1.0
        
        # Should have created experimental result objects
        assert len(validation.experiments) == 2


class TestPublicationReadinessGate:
    """Test publication readiness gate."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_gate_initialization(self):
        """Test gate initialization."""
        gate = PublicationReadinessGate()
        
        assert gate.name == "Publication Readiness"
        assert gate.generation == QualityGateGeneration.GENERATION_3
        assert gate.research_validator is None
    
    @pytest.mark.asyncio
    async def test_gate_execution(self, temp_source_dir):
        """Test gate execution."""
        gate = PublicationReadinessGate()
        context = {"source_dir": temp_source_dir}
        
        # Mock the research validator to avoid long execution
        mock_research_results = {
            "publication_assessment": PublicationReadinessAssessment(
                code_quality_score=0.9,
                documentation_completeness=0.85,
                experimental_rigor=0.88,
                reproducibility_score=0.92,
                novelty_assessment=0.8,
                benchmark_coverage=0.75,
                statistical_validity=0.9,
                overall_readiness=0.87,
                recommendations=["Minor improvements needed"]
            ),
            "overall_research_quality": 0.87,
            "experimental_validation": {"overall_score": 0.8},
            "novelty_assessment": {"innovation_score": 0.8},
            "statistical_validation": {"overall_score": 0.9},
            "reproducibility_check": {"overall_score": 0.92}
        }
        
        with patch.object(gate, 'research_validator') as mock_validator:
            mock_validator.validate_research_quality = AsyncMock(return_value=mock_research_results)
            
            result = await gate._execute_generation_specific(context)
            
            assert result.name == "Publication Readiness"
            assert result.generation == QualityGateGeneration.GENERATION_3
            assert result.score == 0.87
            assert result.confidence == 0.95
            assert isinstance(result.passed, bool)
            
            # Should have detailed results
            assert "research_results" in result.details
            assert "publication_assessment" in result.details
            assert "research_quality_components" in result.details
            
            # Should have recommendations
            assert len(result.recommendations) > 0
            assert len(result.next_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__])