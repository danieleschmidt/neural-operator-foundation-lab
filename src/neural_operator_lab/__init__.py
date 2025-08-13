"""Neural Operator Foundation Lab

A comprehensive framework for training and benchmarking neural operators
on high-dimensional PDEs (turbulence, electromagnetics, elasticity).
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt" 
__email__ = "daniel@terragon.ai"

# Always available core imports
from neural_operator_lab.config import configure_security

# Conditional imports that depend on external libraries
__all__ = ["__version__", "configure_security"]

# Try to import torch-dependent modules
try:
    import torch
    _HAS_TORCH = True
    
    from neural_operator_lab.models import (
        FourierNeuralOperator,
        ProbabilisticNeuralOperator, 
        TransformerNeuralOperator
    )
    from neural_operator_lab.training import Trainer, TrainerConfig
    
    __all__.extend([
        "FourierNeuralOperator",
        "ProbabilisticNeuralOperator", 
        "TransformerNeuralOperator",
        "Trainer",
        "TrainerConfig"
    ])
    
except ImportError:
    _HAS_TORCH = False

# Try to import optional data modules
try:
    from neural_operator_lab.data import PDEDataset, create_pde_dataset
    __all__.extend(["PDEDataset", "create_pde_dataset"])
except ImportError:
    pass

# Try to import analytics modules  
try:
    from neural_operator_lab.analytics import MetricsTracker, PerformanceAnalyzer
    __all__.extend(["MetricsTracker", "PerformanceAnalyzer"])
except ImportError:
    pass

# Try to import security modules
try:
    from neural_operator_lab.security import InputValidator, SecureModelLoader
    __all__.extend(["InputValidator", "SecureModelLoader"])
except ImportError:
    pass

# Try to import optimization modules
try:
    from neural_operator_lab.optimization import MemoryOptimizer
    __all__.extend(["MemoryOptimizer"])
except ImportError:
    pass

# Try to import validation modules
try:
    from neural_operator_lab.validation import ModelValidator, DataValidator, ValidationMetrics
    __all__.extend(["ModelValidator", "DataValidator", "ValidationMetrics"])
except ImportError:
    pass

def get_available_modules():
    """Get list of available modules based on installed dependencies."""
    available = ["config"]
    
    if _HAS_TORCH:
        available.extend(["models", "training", "base", "utils"])
    
    try:
        import h5py
        available.append("data")
    except ImportError:
        pass
        
    try:
        import numpy
        available.append("analytics")
    except ImportError:
        pass
    
    return available

def check_dependencies():
    """Check which optional dependencies are installed."""
    deps = {}
    
    # Core dependencies
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = None
        
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
        
    try:
        import h5py
        deps['h5py'] = h5py.__version__
    except ImportError:
        deps['h5py'] = None
        
    try:
        import scipy
        deps['scipy'] = scipy.__version__
    except ImportError:
        deps['scipy'] = None
        
    try:
        import matplotlib
        deps['matplotlib'] = matplotlib.__version__
    except ImportError:
        deps['matplotlib'] = None
    
    return deps