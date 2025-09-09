"""
Performance Monitor for AGMOHD Optimizer

Comprehensive training performance monitoring and analytics system.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time
from collections import deque


class PerformanceMonitor:
    """
    Performance monitoring system for AGMOHD optimizer.

    Tracks and analyzes:
    1. Training metrics (loss, accuracy, etc.)
    2. Optimization parameters (LR, momentum, etc.)
    3. System resources (GPU memory, utilization)
    4. Convergence patterns and trends

    Args:
        max_history (int): Maximum history length to keep
        enable_gpu_monitoring (bool): Enable GPU monitoring
        log_interval (int): Logging interval in steps
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_gpu_monitoring: bool = True,
        log_interval: int = 100
    ):
        self.max_history = max_history
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.log_interval = log_interval

        # Data storage
        self.metrics_history = deque(maxlen=max_history)
        self.system_stats = deque(maxlen=max_history)

        # Current state
        self.start_time = time.time()
        self.step_count = 0

        # Performance tracking
        self.best_loss = float('inf')
        self.best_step = 0
        self.convergence_threshold = 1e-4

        # Logging
        self.logger = logging.getLogger(__name__)

    def update(
        self,
        step: int,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        hindrance: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ):
        """
        Update performance metrics.

        Args:
            step: Current training step
            loss: Current loss value
            lr: Current learning rate
            momentum: Current momentum value
            hindrance: Current hindrance level
            gradient_norm: Current gradient norm
            **kwargs: Additional metrics
        """

        self.step_count = step

        # Collect metrics
        metrics = {
            'step': step,
            'timestamp': time.time() - self.start_time,
            'loss': loss,
            'learning_rate': lr,
            'momentum': momentum,
            'hindrance_level': hindrance,
            'gradient_norm': gradient_norm,
            **kwargs
        }

        # Add system stats if enabled
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            system_stats = self._get_system_stats()
            metrics.update(system_stats)

        # Store in history
        self.metrics_history.append(metrics)

        # Update best metrics
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            self.best_step = step

    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""

        stats = {}

        if torch.cuda.is_available():
            # GPU memory stats
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024   # MB
            stats['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0

            # GPU temperature (if available)
            try:
                stats['gpu_temperature'] = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else None
            except:
                stats['gpu_temperature'] = None

        # CPU stats (simplified)
        import psutil
        stats['cpu_percent'] = psutil.cpu_percent()
        stats['memory_percent'] = psutil.virtual_memory().percent

        return stats

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics_history:
            return {}

        return self.metrics_history[-1]

    def get_metrics_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        if last_n is None:
            return list(self.metrics_history)
        else:
            return list(self.metrics_history)[-last_n:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""

        if not self.metrics_history:
            return {}

        # Extract metrics
        losses = [m['loss'] for m in self.metrics_history if m['loss'] is not None]
        lrs = [m['learning_rate'] for m in self.metrics_history if m['learning_rate'] is not None]
        hindrances = [m['hindrance_level'] for m in self.metrics_history if m['hindrance_level'] is not None]

        summary = {
            'total_steps': self.step_count,
            'training_time': time.time() - self.start_time,
            'best_loss': self.best_loss,
            'best_step': self.best_step,
            'convergence_achieved': self._check_convergence(),
        }

        # Statistical summaries
        if losses:
            summary.update({
                'final_loss': losses[-1],
                'average_loss': np.mean(losses),
                'loss_std': np.std(losses),
                'loss_trend': self._calculate_trend(losses),
                'loss_improvement': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 else 0
            })

        if lrs:
            summary.update({
                'final_lr': lrs[-1],
                'average_lr': np.mean(lrs),
                'lr_range': max(lrs) - min(lrs)
            })

        if hindrances:
            summary.update({
                'average_hindrance': np.mean(hindrances),
                'max_hindrance': max(hindrances),
                'hindrance_stability': np.std(hindrances)
            })

        return summary

    def _calculate_trend(self, values: List[float], window: int = 10) -> float:
        """Calculate trend in values over recent window."""

        if len(values) < window:
            return 0.0

        recent = values[-window:]
        x = np.arange(len(recent))

        # Linear regression slope
        slope = np.polyfit(x, recent, 1)[0]

        return slope

    def _check_convergence(self, threshold: float = 1e-4, window: int = 20) -> bool:
        """Check if training has converged."""

        if len(self.metrics_history) < window:
            return False

        recent_losses = [m['loss'] for m in list(self.metrics_history)[-window:]
                        if m['loss'] is not None]

        if len(recent_losses) < window // 2:
            return False

        # Check if loss variation is below threshold
        loss_std = np.std(recent_losses)
        avg_loss = np.mean(recent_losses)

        return loss_std / (avg_loss + 1e-8) < threshold

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Get detailed convergence analysis."""

        analysis = {
            'is_converged': self._check_convergence(),
            'convergence_step': None,
            'convergence_time': None,
            'stability_score': 0.0,
            'oscillation_detected': False
        }

        if not self.metrics_history:
            return analysis

        # Find convergence point
        window_size = 20
        for i in range(window_size, len(self.metrics_history)):
            recent_metrics = list(self.metrics_history)[i-window_size:i]
            recent_losses = [m['loss'] for m in recent_metrics if m['loss'] is not None]

            if len(recent_losses) >= window_size // 2:
                loss_std = np.std(recent_losses)
                avg_loss = np.mean(recent_losses)

                if loss_std / (avg_loss + 1e-8) < self.convergence_threshold:
                    analysis['is_converged'] = True
                    analysis['convergence_step'] = recent_metrics[-1]['step']
                    analysis['convergence_time'] = recent_metrics[-1]['timestamp']
                    break

        # Calculate stability score
        if len(self.metrics_history) > 10:
            recent_losses = [m['loss'] for m in list(self.metrics_history)[-10:]
                           if m['loss'] is not None]
            if recent_losses:
                analysis['stability_score'] = 1.0 / (1.0 + np.std(recent_losses))

        # Detect oscillations
        if len(self.metrics_history) > 20:
            losses = [m['loss'] for m in self.metrics_history if m['loss'] is not None]
            if len(losses) > 10:
                # Simple oscillation detection
                diffs = np.diff(losses)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                oscillation_ratio = sign_changes / len(diffs)
                analysis['oscillation_detected'] = oscillation_ratio > 0.3

        return analysis

    def reset(self):
        """Reset performance monitor."""
        self.metrics_history.clear()
        self.system_stats.clear()
        self.start_time = time.time()
        self.step_count = 0
        self.best_loss = float('inf')
        self.best_step = 0

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        data = {
            'summary': self.get_performance_summary(),
            'convergence_analysis': self.get_convergence_analysis(),
            'metrics_history': list(self.metrics_history),
            'export_time': time.time()
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def __repr__(self):
        summary = self.get_performance_summary()
        return (f'{self.__class__.__name__}('
                f'steps={summary.get("total_steps", 0)}, '
                f'best_loss={summary.get("best_loss", "N/A"):.4f}, '
                f'training_time={summary.get("training_time", 0):.1f}s)')
