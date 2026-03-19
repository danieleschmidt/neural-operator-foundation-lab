"""Neural Operator Foundation Lab — tools for learning mappings between function spaces."""

from .fno import FourierNeuralOperator1d, FourierNeuralOperator2d, SpectralConv1d, SpectralConv2d
from .deeponet import DeepONet
from .data import PDEDataGenerator
from .trainer import OperatorTrainer

__version__ = "0.2.0"
__all__ = [
    "FourierNeuralOperator1d",
    "FourierNeuralOperator2d",
    "SpectralConv1d",
    "SpectralConv2d",
    "DeepONet",
    "PDEDataGenerator",
    "OperatorTrainer",
]
