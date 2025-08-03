"""Neural Operator Foundation Lab

A comprehensive framework for training and benchmarking neural operators
on high-dimensional PDEs (turbulence, electromagnetics, elasticity).
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core imports for easy access
from neural_operator_lab.config import configure_security
from neural_operator_lab.models import (
    FourierNeuralOperator,
    ProbabilisticNeuralOperator, 
    TransformerNeuralOperator
)
from neural_operator_lab.data import PDEDataset, create_pde_dataset
from neural_operator_lab.training import Trainer, TrainerConfig

# Advanced features
from neural_operator_lab.analytics import MetricsTracker, PerformanceAnalyzer
from neural_operator_lab.security import InputValidator, SecureModelLoader
from neural_operator_lab.optimization import MemoryOptimizer

__all__ = [
    "__version__",
    "configure_security",
    "FourierNeuralOperator",
    "ProbabilisticNeuralOperator",
    "TransformerNeuralOperator",
    "PDEDataset",
    "create_pde_dataset", 
    "Trainer",
    "TrainerConfig",
    "MetricsTracker",
    "PerformanceAnalyzer",
    "InputValidator",
    "SecureModelLoader",
    "MemoryOptimizer",
]