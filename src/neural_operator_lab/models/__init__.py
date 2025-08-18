"""Neural operator model implementations."""

from .fourier import FourierNeuralOperator
from .probabilistic import ProbabilisticNeuralOperator
from .transformer import TransformerNeuralOperator
from .geometric import GeometricNeuralOperator
from .physics_informed import PhysicsInformedUniversalNeuralOperator
from .adaptive import MultiResolutionAdaptiveNeuralOperator

# Import foundation models if available
try:
    from .foundation import UniversalPhysicsTransformer
    _HAS_FOUNDATION = True
except ImportError:
    _HAS_FOUNDATION = False
    UniversalPhysicsTransformer = None

# Import error handling utilities
try:
    from .error_handling import (
        RobustNeuralOperator, 
        MemoryAwareForward,
        GradientHealthChecker,
        NumericalStabilizer,
        make_model_robust
    )
    _HAS_ERROR_HANDLING = True
except ImportError:
    _HAS_ERROR_HANDLING = False

__all__ = [
    'FourierNeuralOperator',
    'ProbabilisticNeuralOperator', 
    'TransformerNeuralOperator',
    'GeometricNeuralOperator',
    'PhysicsInformedUniversalNeuralOperator',
    'MultiResolutionAdaptiveNeuralOperator'
]

if _HAS_FOUNDATION:
    __all__.append('UniversalPhysicsTransformer')

if _HAS_ERROR_HANDLING:
    __all__.extend([
        'RobustNeuralOperator',
        'MemoryAwareForward', 
        'GradientHealthChecker',
        'NumericalStabilizer',
        'make_model_robust'
    ])