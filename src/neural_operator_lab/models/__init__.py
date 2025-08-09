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