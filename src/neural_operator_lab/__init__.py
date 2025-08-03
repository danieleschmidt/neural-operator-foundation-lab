"""Neural Operator Foundation Lab

A comprehensive framework for training and benchmarking neural operators
on high-dimensional PDEs (turbulence, electromagnetics, elasticity).
"""

__version__ = "0.1.0"
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
]