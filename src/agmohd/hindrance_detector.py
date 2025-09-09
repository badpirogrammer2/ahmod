"""
Hindrance Detector for AGMOHD Optimizer

Advanced hindrance detection engine that analyzes training dynamics
to identify instabilities and optimization challenges.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging


class HindranceDetector:
    """
    Advanced hindrance detection engine for AGMOHD optimizer.

    Detects various types of training instabilities:
    1. Gradient explosions and vanishing
    2. Loss stability issues
    3. Training plateau regions
    4. Oscillatory behavior

    Args:
        threshold (float): Base threshold for hindrance detection
        adaptive_sensitivity (bool): Enable adaptive sensitivity adjustment
        gradient_window (int): Window size for gradient analysis
        loss_window (int): Window size for loss stability analysis
    """

    def __init__(
        self,
        threshold: float = 0.1,
        adaptive_sensitivity: bool = True,
        gradient_window: int = 10,
        loss_window: int = 20
    ):
        self.threshold = threshold
        self.adaptive_sensitivity = adaptive_sensitivity
        self.gradient_window = gradient_window
        self.loss_window = loss_window

        # State tracking
        self.gradient_history = []
        self.loss_history = []
        self.hindrance_history = []

        # Adaptive parameters
        self.adaptive_threshold = threshold
        self.sensitivity_factor = 1.0

        # Detection thresholds
        self.explosion_threshold = 10.0  # Gradient explosion multiplier
        self.vanishing_threshold = 1e-6   # Gradient vanishing threshold
        self.plateau_threshold = 0.01     # Loss plateau variance threshold
        self.oscillation_threshold = 0.05 # Oscillation detection threshold

        # Logging
        self.logger = logging.getLogger(__name__)

    def detect_hindrance(
        self,
        gradients: List[torch.Tensor],
        loss_history: Optional[List[float]] = None
    ) -> float:
        """
        Detect training hindrance from gradients and loss history.

        Args:
            gradients: List of gradient tensors
            loss_history: Recent loss values (optional)

        Returns:
            hindrance_level: Float between 0.0 (no hindrance) and 1.0 (severe hindrance)
        """

        # Analyze gradient magnitudes
        gradient_hindrance = self._analyze_gradient_magnitudes(gradients)

        # Analyze loss stability if available
        loss_hindrance = 0.0
        if loss_history and len(loss_history) >= 3:
            loss_hindrance = self._analyze_loss_stability(loss_history)

        # Analyze gradient stability
        stability_hindrance = self._analyze_gradient_stability(gradients)

        # Combine hindrance factors
        combined_hindrance = self._combine_hindrance_factors(
            gradient_hindrance, loss_hindrance, stability_hindrance
        )

        # Update adaptive threshold
        if self.adaptive_sensitivity:
            self._update_adaptive_threshold(combined_hindrance)

        # Store hindrance history
        self.hindrance_history.append(combined_hindrance)
        if len(self.hindrance_history) > 50:
            self.hindrance_history.pop(0)

        return combined_hindrance

    def _analyze_gradient_magnitudes(self, gradients: List[torch.Tensor]) -> float:
        """Analyze gradient magnitudes for explosion/vanishing detection."""

        if not gradients:
            return 0.0

        # Calculate gradient norms
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                norm = torch.norm(grad).item()
                if not np.isfinite(norm):
                    return 1.0  # Severe hindrance if gradients are NaN/Inf
                grad_norms.append(norm)

        if not grad_norms:
            return 0.0

        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)

        # Store gradient history
        self.gradient_history.append(mean_norm)
        if len(self.gradient_history) > self.gradient_window:
            self.gradient_history.pop(0)

        # Detect gradient explosion
        if mean_norm > self.explosion_threshold:
            return min(1.0, mean_norm / (self.explosion_threshold * 2))

        # Detect gradient vanishing
        if mean_norm < self.vanishing_threshold:
            return min(1.0, self.vanishing_threshold / mean_norm)

        # Detect high variance (unstable gradients)
        if len(self.gradient_history) >= 3:
            recent_norms = self.gradient_history[-3:]
            variance = np.var(recent_norms)
            if variance > mean_norm * 0.5:  # High variance relative to mean
                return min(1.0, variance / (mean_norm * 0.5))

        return 0.0

    def _analyze_loss_stability(self, loss_history: List[float]) -> float:
        """Analyze loss stability for plateau and oscillation detection."""

        if len(loss_history) < self.loss_window:
            return 0.0

        recent_losses = loss_history[-self.loss_window:]

        # Calculate loss variance (plateau detection)
        loss_variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)

        # Plateau detection
        if loss_variance < self.plateau_threshold:
            plateau_factor = min(1.0, self.plateau_threshold / (loss_variance + 1e-8))
            return plateau_factor * 0.7  # Plateau hindrance weight

        # Oscillation detection
        if len(recent_losses) >= 5:
            # Check for alternating up/down pattern
            oscillation_score = self._detect_oscillation(recent_losses)
            if oscillation_score > self.oscillation_threshold:
                return min(1.0, oscillation_score * 2.0)

        # Divergence detection (rapid loss increase)
        if len(loss_history) >= 10:
            recent_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            if recent_trend > mean_loss * 0.1:  # Rapid increase
                return min(1.0, recent_trend / (mean_loss * 0.1))

        return 0.0

    def _analyze_gradient_stability(self, gradients: List[torch.Tensor]) -> float:
        """Analyze gradient stability over time."""

        if len(self.gradient_history) < 3:
            return 0.0

        # Calculate gradient trend
        recent_grads = self.gradient_history[-5:]
        if len(recent_grads) >= 3:
            # Check for erratic gradient behavior
            diffs = np.diff(recent_grads)
            erratic_score = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-8)

            if erratic_score > 2.0:  # Highly erratic
                return min(1.0, erratic_score / 4.0)

        return 0.0

    def _detect_oscillation(self, losses: List[float]) -> float:
        """Detect oscillatory behavior in loss values."""

        if len(losses) < 5:
            return 0.0

        # Calculate second differences to detect oscillation
        first_diff = np.diff(losses)
        second_diff = np.diff(first_diff)

        # Count sign changes in second differences
        sign_changes = np.sum(np.diff(np.sign(second_diff)) != 0)

        # Normalize by sequence length
        oscillation_score = sign_changes / len(second_diff)

        return oscillation_score

    def _combine_hindrance_factors(
        self,
        gradient_hindrance: float,
        loss_hindrance: float,
        stability_hindrance: float
    ) -> float:
        """Combine different hindrance factors into final hindrance level."""

        # Weighted combination
        weights = {
            'gradient': 0.4,    # Gradient issues are most critical
            'loss': 0.35,       # Loss stability is important
            'stability': 0.25   # Gradient stability matters
        }

        combined = (
            weights['gradient'] * gradient_hindrance +
            weights['loss'] * loss_hindrance +
            weights['stability'] * stability_hindrance
        )

        # Apply adaptive threshold
        if combined > self.adaptive_threshold:
            # Scale hindrance level based on adaptive threshold
            scaled_hindrance = (combined - self.adaptive_threshold) / (1.0 - self.adaptive_threshold)
            return min(1.0, max(0.0, scaled_hindrance))

        return 0.0

    def _update_adaptive_threshold(self, current_hindrance: float):
        """Update adaptive threshold based on training history."""

        if len(self.hindrance_history) < 10:
            return

        # Calculate recent average hindrance
        recent_avg = np.mean(self.hindrance_history[-10:])

        # Adjust threshold based on training stability
        if recent_avg < 0.1:  # Very stable training
            self.adaptive_threshold = max(0.05, self.adaptive_threshold * 0.95)
        elif recent_avg > 0.3:  # Unstable training
            self.adaptive_threshold = min(0.3, self.adaptive_threshold * 1.05)

        # Adjust sensitivity factor
        if current_hindrance > 0.5:
            self.sensitivity_factor = min(2.0, self.sensitivity_factor * 1.1)
        elif current_hindrance < 0.1:
            self.sensitivity_factor = max(0.5, self.sensitivity_factor * 0.95)

    def get_hindrance_level(self) -> float:
        """Get current hindrance level."""
        return self.hindrance_history[-1] if self.hindrance_history else 0.0

    def get_hindrance_stats(self) -> Dict[str, Any]:
        """Get detailed hindrance statistics."""
        return {
            'current_level': self.get_hindrance_level(),
            'average_level': np.mean(self.hindrance_history) if self.hindrance_history else 0.0,
            'max_level': max(self.hindrance_history) if self.hindrance_history else 0.0,
            'adaptive_threshold': self.adaptive_threshold,
            'sensitivity_factor': self.sensitivity_factor,
            'gradient_history_length': len(self.gradient_history),
            'loss_history_length': len(self.loss_history)
        }

    def reset(self):
        """Reset hindrance detector state."""
        self.gradient_history.clear()
        self.loss_history.clear()
        self.hindrance_history.clear()
        self.adaptive_threshold = self.threshold
        self.sensitivity_factor = 1.0

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            'gradient_history': self.gradient_history,
            'loss_history': self.loss_history,
            'hindrance_history': self.hindrance_history,
            'adaptive_threshold': self.adaptive_threshold,
            'sensitivity_factor': self.sensitivity_factor,
            'threshold': self.threshold
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.gradient_history = state.get('gradient_history', [])
        self.loss_history = state.get('loss_history', [])
        self.hindrance_history = state.get('hindrance_history', [])
        self.adaptive_threshold = state.get('adaptive_threshold', self.threshold)
        self.sensitivity_factor = state.get('sensitivity_factor', 1.0)
        self.threshold = state.get('threshold', self.threshold)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'threshold={self.threshold}, '
                f'adaptive_sensitivity={self.adaptive_sensitivity}, '
                f'gradient_window={self.gradient_window}, '
                f'loss_window={self.loss_window}, '
                f'current_hindrance={self.get_hindrance_level():.3f})')
