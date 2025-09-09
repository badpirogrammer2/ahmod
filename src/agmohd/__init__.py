"""
AGMOHD: Adaptive Gradient Momentum with Hindrance Detection

A revolutionary PyTorch optimizer that adapts to training dynamics through
intelligent hindrance detection and adaptive momentum control.
"""

__version__ = "1.0.0"
__author__ = "AGMOHD Team"
__license__ = "MIT"

from .agmohd import AGMOHD
from .hindrance_detector import HindranceDetector
from .momentum_controller import MomentumController
from .lr_scheduler import CyclicalLRScheduler
from .gradient_processor import GradientProcessor
from .performance_monitor import PerformanceMonitor

__all__ = [
    'AGMOHD',
    'HindranceDetector',
    'MomentumController',
    'CyclicalLRScheduler',
    'GradientProcessor',
    'PerformanceMonitor'
]
