"""Research framework for neural operators with comparative studies and benchmarking."""

from .benchmark_suite import ResearchBenchmarkSuite, ComparisonStudy
from .statistical_analysis import StatisticalAnalyzer, SignificanceTests
from .experiment_framework import ExperimentManager, HypothesisTester
from .reproducibility import ReproducibilityManager, ResultsValidator
from .publication_tools import PublicationPreparer, FigureGenerator

__all__ = [
    "ResearchBenchmarkSuite",
    "ComparisonStudy", 
    "StatisticalAnalyzer",
    "SignificanceTests",
    "ExperimentManager",
    "HypothesisTester",
    "ReproducibilityManager",
    "ResultsValidator",
    "PublicationPreparer", 
    "FigureGenerator"
]