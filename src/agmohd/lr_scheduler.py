"""
Learning Rate Scheduler for AGMOHD Optimizer

Advanced learning rate scheduling with cyclical patterns and
adaptive adjustments based on training dynamics.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import math


class CyclicalLRScheduler:
    """
    Cyclical Learning Rate Scheduler for AGMOHD optimizer.

    Supports multiple scheduling modes:
    1. Cyclical: Triangular cyclical learning rates
    2. Cosine: Cosine annealing schedule
    3. Step: Step decay schedule
    4. None: Constant learning rate

    Args:
        base_lr (float): Base learning rate
        mode (str): Scheduling mode
        cycle_length (int): Length of cycle for cyclical mode
        lr_min (float): Minimum learning rate
        lr_max (float): Maximum learning rate
    """

    def __init__(
        self,
        base_lr: float = 0.01,
        mode: str = 'cyclical',
        cycle_length: int = 100,
        lr_min: float = 1e-6,
        lr_max: float = 0.1
    ):
        self.base_lr = base_lr
        self.mode = mode
        self.cycle_length = cycle_length
        self.lr_min = lr_min
        self.lr_max = lr_max

        # State tracking
        self.step_count = 0
        self.lr_history = [base_lr]

        # Cyclical parameters
        self.cycle_step = 0

        # Logging
        self.logger = logging.getLogger(__name__)

    def get_lr(self, step: int) -> float:
        """
        Get learning rate for current step.

        Args:
            step: Current training step

        Returns:
            learning_rate: Learning rate for this step
        """

        self.step_count = step

        if self.mode == 'cyclical':
            lr = self._cyclical_lr(step)
        elif self.mode == 'cosine':
            lr = self._cosine_lr(step)
        elif self.mode == 'step':
            lr = self._step_lr(step)
        elif self.mode == 'none':
            lr = self.base_lr
        else:
            raise ValueError(f"Unknown LR scheduler mode: {self.mode}")

        # Store in history
        self.lr_history.append(lr)
        if len(self.lr_history) > 1000:  # Keep last 1000 values
            self.lr_history = self.lr_history[-500:]

        return lr

    def _cyclical_lr(self, step: int) -> float:
        """Triangular cyclical learning rate schedule."""

        cycle_position = step % self.cycle_length
        cycle_progress = cycle_position / self.cycle_length

        if cycle_progress <= 0.5:
            # Increasing phase
            lr_range = self.lr_max - self.base_lr
            lr = self.base_lr + (lr_range * cycle_progress * 2)
        else:
            # Decreasing phase
            lr_range = self.lr_max - self.lr_min
            lr = self.lr_max - (lr_range * (cycle_progress - 0.5) * 2)

        return max(self.lr_min, min(self.lr_max, lr))

    def _cosine_lr(self, step: int) -> float:
        """Cosine annealing learning rate schedule."""

        # Cosine annealing with warm restarts
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.cycle_length))
        lr = self.lr_min + (self.base_lr - self.lr_min) * cosine_decay

        return lr

    def _step_lr(self, step: int) -> float:
        """Step decay learning rate schedule."""

        # Decay every cycle_length steps
        decay_factor = 0.1 ** (step // self.cycle_length)
        lr = self.base_lr * decay_factor

        return max(self.lr_min, lr)

    def get_lr_stats(self) -> Dict[str, Any]:
        """Get learning rate statistics."""
        return {
            'current_lr': self.lr_history[-1] if self.lr_history else self.base_lr,
            'base_lr': self.base_lr,
            'min_lr': self.lr_min,
            'max_lr': self.lr_max,
            'mode': self.mode,
            'cycle_length': self.cycle_length,
            'step_count': self.step_count,
            'average_lr': np.mean(self.lr_history) if self.lr_history else 0.0,
            'lr_variance': np.var(self.lr_history) if len(self.lr_history) > 1 else 0.0
        }

    def reset(self):
        """Reset scheduler state."""
        self.step_count = 0
        self.lr_history = [self.base_lr]
        self.cycle_step = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            'step_count': self.step_count,
            'lr_history': self.lr_history,
            'base_lr': self.base_lr,
            'mode': self.mode,
            'cycle_length': self.cycle_length,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.step_count = state.get('step_count', 0)
        self.lr_history = state.get('lr_history', [self.base_lr])
        self.base_lr = state.get('base_lr', self.base_lr)
        self.mode = state.get('mode', self.mode)
        self.cycle_length = state.get('cycle_length', self.cycle_length)
        self.lr_min = state.get('lr_min', self.lr_min)
        self.lr_max = state.get('lr_max', self.lr_max)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'base_lr={self.base_lr}, '
                f'mode={self.mode}, '
                f'current_lr={self.lr_history[-1] if self.lr_history else self.base_lr:.6f})')
