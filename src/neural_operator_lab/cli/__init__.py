"""Command line interface for neural operator lab."""

from .main import main, create_parser
from .commands import TrainCommand, EvaluateCommand, BenchmarkCommand

__all__ = [
    'main',
    'create_parser', 
    'TrainCommand',
    'EvaluateCommand',
    'BenchmarkCommand'
]