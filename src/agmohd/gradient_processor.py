"""
Gradient Processor for AGMOHD Optimizer

Advanced gradient processing with adaptive clipping, normalization,
and noise filtering capabilities.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging


class GradientProcessor:
    """
    Advanced gradient processing for AGMOHD optimizer.

    Features:
    1. Adaptive gradient clipping
    2. Gradient normalization
    3. Noise filtering
    4. RTX optimizations

    Args:
        clipping_method (str): Gradient clipping method
        use_rtx_optimizations (bool): Enable RTX optimizations
        clip_value (float): Clipping threshold
        noise_filter_strength (float): Noise filtering strength
    """

    def __init__(
        self,
        clipping_method: str = 'adaptive',
        use_rtx_optimizations: bool = True,
        clip_value: float = 1.0,
        noise_filter_strength: float = 0.1
    ):
        self.clipping_method = clipping_method
        self.use_rtx_optimizations = use_rtx_optimizations
        self.clip_value = clip_value
        self.noise_filter_strength = noise_filter_strength

        # RTX optimizations
        if use_rtx_optimizations and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Logging
        self.logger = logging.getLogger(__name__)

    def process_gradients(
        self,
        gradients: List[torch.Tensor],
        hindrance_level: float
    ) -> List[torch.Tensor]:
        """
        Process gradients with adaptive techniques.

        Args:
            gradients: List of gradient tensors
            hindrance_level: Current training hindrance level

        Returns:
            processed_gradients: Processed gradient tensors
        """

        if not gradients:
            return gradients

        # Apply gradient clipping
        if self.clipping_method != 'none':
            gradients = self._clip_gradients(gradients, hindrance_level)

        # Apply noise filtering
        if self.noise_filter_strength > 0:
            gradients = self._filter_noise(gradients)

        # Normalize gradients if needed
        gradients = self._normalize_gradients(gradients)

        return gradients

    def _clip_gradients(
        self,
        gradients: List[torch.Tensor],
        hindrance_level: float
    ) -> List[torch.Tensor]:
        """Apply adaptive gradient clipping."""

        if self.clipping_method == 'global_norm':
            return self._global_norm_clipping(gradients)
        elif self.clipping_method == 'adaptive':
            return self._adaptive_clipping(gradients, hindrance_level)
        else:
            return gradients

    def _global_norm_clipping(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply global norm gradient clipping."""

        # Calculate global norm
        global_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients if g is not None))

        # Clip if above threshold
        if global_norm > self.clip_value:
            clip_coef = self.clip_value / (global_norm + 1e-6)
            gradients = [g * clip_coef if g is not None else g for g in gradients]

        return gradients

    def _adaptive_clipping(
        self,
        gradients: List[torch.Tensor],
        hindrance_level: float
    ) -> List[torch.Tensor]:
        """Apply adaptive gradient clipping based on hindrance level."""

        # Adjust clipping threshold based on hindrance
        adaptive_clip_value = self.clip_value

        if hindrance_level > 0.5:  # High hindrance
            adaptive_clip_value *= 0.5  # More aggressive clipping
        elif hindrance_level < 0.2:  # Low hindrance
            adaptive_clip_value *= 2.0  # Less aggressive clipping

        # Apply global norm clipping with adaptive threshold
        global_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients if g is not None))

        if global_norm > adaptive_clip_value:
            clip_coef = adaptive_clip_value / (global_norm + 1e-6)
            gradients = [g * clip_coef if g is not None else g for g in gradients]

        return gradients

    def _filter_noise(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply noise filtering to gradients."""

        filtered_gradients = []

        for grad in gradients:
            if grad is None:
                filtered_gradients.append(grad)
                continue

            # Simple noise filtering using median filtering
            if grad.numel() > 10:  # Only filter if tensor is large enough
                # Calculate median absolute deviation
                median = torch.median(torch.abs(grad))
                mad = torch.median(torch.abs(grad - median))

                # Filter out extreme values
                if mad > 0:
                    modified_z_score = 0.6745 * (grad - median) / mad
                    # Clip extreme values
                    grad = torch.where(
                        torch.abs(modified_z_score) > 3.5,
                        median,  # Replace with median
                        grad     # Keep original
                    )

            filtered_gradients.append(grad)

        return filtered_gradients

    def _normalize_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize gradients to prevent scale issues."""

        # Calculate gradient norms
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                norm = torch.norm(grad)
                grad_norms.append(norm.item() if norm > 0 else 1e-8)

        if not grad_norms:
            return gradients

        # Normalize by median norm to be robust to outliers
        median_norm = np.median(grad_norms)

        if median_norm > 0:
            normalized_gradients = []
            for grad in gradients:
                if grad is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > 0:
                        # Normalize to median scale
                        scale_factor = median_norm / grad_norm
                        grad = grad * scale_factor
                normalized_gradients.append(grad)
            return normalized_gradients

        return gradients

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get gradient processing statistics."""
        return {
            'clipping_method': self.clipping_method,
            'clip_value': self.clip_value,
            'noise_filter_strength': self.noise_filter_strength,
            'use_rtx_optimizations': self.use_rtx_optimizations
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'clipping_method={self.clipping_method}, '
                f'clip_value={self.clip_value}, '
                f'noise_filter_strength={self.noise_filter_strength})')
