"""
Momentum Controller for AGMOHD Optimizer

Adaptive momentum control system that adjusts momentum based on
training hindrance levels and optimization requirements.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import logging


class MomentumController:
    """
    Adaptive momentum controller for AGMOHD optimizer.

    Dynamically adjusts momentum based on:
    1. Current hindrance levels
    2. Training stability
    3. Optimization requirements

    Args:
        initial_momentum (float): Initial momentum value
        schedule (str): Momentum scheduling strategy
        min_momentum (float): Minimum momentum value
        max_momentum (float): Maximum momentum value
        adaptation_rate (float): Rate of momentum adaptation
    """

    def __init__(
        self,
        initial_momentum: float = 0.9,
        schedule: str = 'adaptive',
        min_momentum: float = 0.5,
        max_momentum: float = 0.95,
        adaptation_rate: float = 0.1
    ):
        self.initial_momentum = initial_momentum
        self.schedule = schedule
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.adaptation_rate = adaptation_rate

        # Current state
        self.current_momentum = initial_momentum
        self.momentum_history = [initial_momentum]

        # Adaptation parameters
        self.momentum_decay = 0.95   # Decay factor for high hindrance
        self.momentum_growth = 1.05  # Growth factor for low hindrance

        # Thresholds
        self.high_hindrance_threshold = 0.7
        self.low_hindrance_threshold = 0.2

        # Logging
        self.logger = logging.getLogger(__name__)

    def adjust_momentum(self, hindrance_level: float) -> float:
        """
        Adjust momentum based on current hindrance level.

        Args:
            hindrance_level: Current training hindrance level (0.0 to 1.0)

        Returns:
            adjusted_momentum: New momentum value
        """

        if self.schedule == 'fixed':
            return self.current_momentum
        elif self.schedule == 'adaptive':
            return self._adaptive_adjustment(hindrance_level)
        elif self.schedule == 'nesterov':
            return self._nesterov_adjustment(hindrance_level)
        else:
            raise ValueError(f"Unknown momentum schedule: {self.schedule}")

    def _adaptive_adjustment(self, hindrance_level: float) -> float:
        """Adaptive momentum adjustment based on hindrance level."""

        if hindrance_level > self.high_hindrance_threshold:
            # High hindrance: reduce momentum for stability
            new_momentum = max(
                self.min_momentum,
                self.current_momentum * self.momentum_decay
            )
            self.logger.debug(".4f")
        elif hindrance_level < self.low_hindrance_threshold:
            # Low hindrance: increase momentum for speed
            new_momentum = min(
                self.max_momentum,
                self.current_momentum * self.momentum_growth
            )
            self.logger.debug(".4f")
        else:
            # Moderate hindrance: gradual adjustment
            target_momentum = self.initial_momentum
            momentum_diff = target_momentum - self.current_momentum
            adjustment = self.adaptation_rate * momentum_diff
            new_momentum = self.current_momentum + adjustment

            # Clamp to bounds
            new_momentum = max(self.min_momentum, min(self.max_momentum, new_momentum))

        # Smooth momentum changes
        new_momentum = self._smooth_momentum_change(new_momentum)

        # Update state
        self.current_momentum = new_momentum
        self.momentum_history.append(new_momentum)

        # Keep history manageable
        if len(self.momentum_history) > 100:
            self.momentum_history.pop(0)

        return new_momentum

    def _nesterov_adjustment(self, hindrance_level: float) -> float:
        """Nesterov-style momentum adjustment."""

        # Base momentum adjustment
        base_momentum = self._adaptive_adjustment(hindrance_level)

        # Nesterov modification: slightly higher momentum for better lookahead
        nesterov_boost = 0.02
        nesterov_momentum = min(self.max_momentum, base_momentum + nesterov_boost)

        return nesterov_momentum

    def _smooth_momentum_change(self, new_momentum: float) -> float:
        """Smooth momentum changes to prevent oscillations."""

        if len(self.momentum_history) < 3:
            return new_momentum

        # Calculate recent momentum trend
        recent_momenta = self.momentum_history[-3:]
        trend = np.polyfit(range(len(recent_momenta)), recent_momenta, 1)[0]

        # If trend is too steep, dampen the change
        if abs(trend) > 0.01:  # Significant momentum change trend
            damping_factor = 0.7
            momentum_diff = new_momentum - self.current_momentum
            damped_diff = momentum_diff * damping_factor
            smoothed_momentum = self.current_momentum + damped_diff

            return smoothed_momentum

        return new_momentum

    def get_current_momentum(self) -> float:
        """Get current momentum value."""
        return self.current_momentum

    def get_momentum_stats(self) -> Dict[str, Any]:
        """Get momentum statistics."""
        return {
            'current_momentum': self.current_momentum,
            'initial_momentum': self.initial_momentum,
            'min_momentum': self.min_momentum,
            'max_momentum': self.max_momentum,
            'schedule': self.schedule,
            'adaptation_rate': self.adaptation_rate,
            'history_length': len(self.momentum_history),
            'average_momentum': np.mean(self.momentum_history) if self.momentum_history else 0.0,
            'momentum_variance': np.var(self.momentum_history) if len(self.momentum_history) > 1 else 0.0
        }

    def reset(self):
        """Reset momentum controller to initial state."""
        self.current_momentum = self.initial_momentum
        self.momentum_history = [self.initial_momentum]

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            'current_momentum': self.current_momentum,
            'momentum_history': self.momentum_history,
            'initial_momentum': self.initial_momentum,
            'schedule': self.schedule,
            'min_momentum': self.min_momentum,
            'max_momentum': self.max_momentum,
            'adaptation_rate': self.adaptation_rate
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.current_momentum = state.get('current_momentum', self.initial_momentum)
        self.momentum_history = state.get('momentum_history', [self.initial_momentum])
        self.initial_momentum = state.get('initial_momentum', self.initial_momentum)
        self.schedule = state.get('schedule', self.schedule)
        self.min_momentum = state.get('min_momentum', self.min_momentum)
        self.max_momentum = state.get('max_momentum', self.max_momentum)
        self.adaptation_rate = state.get('adaptation_rate', self.adaptation_rate)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'initial_momentum={self.initial_momentum}, '
                f'schedule={self.schedule}, '
                f'current_momentum={self.current_momentum:.3f})')
