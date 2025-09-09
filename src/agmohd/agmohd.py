"""
AGMOHD: Adaptive Gradient Momentum with Hindrance Detection

Core optimizer implementation with advanced hindrance detection,
adaptive momentum control, and cyclical learning rates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import os
import json
from datetime import datetime

from .hindrance_detector import HindranceDetector
from .momentum_controller import MomentumController
from .lr_scheduler import CyclicalLRScheduler
from .gradient_processor import GradientProcessor
from .performance_monitor import PerformanceMonitor


class AGMOHD(torch.optim.Optimizer):
    """
    Adaptive Gradient Momentum with Hindrance Detection Optimizer

    A revolutionary optimizer that adapts to training dynamics through:
    1. Intelligent hindrance detection
    2. Adaptive momentum control
    3. Cyclical learning rate scheduling
    4. Advanced gradient processing

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Learning rate (default: 0.01)
        beta (float): Momentum factor (default: 0.9)
        weight_decay (float): Weight decay (L2 penalty) (default: 1e-4)
        eps (float): Term added for numerical stability (default: 1e-8)
        hindrance_threshold (float): Threshold for hindrance detection (default: 0.1)
        lr_scheduler (str): Learning rate scheduler ('cyclical', 'cosine', 'step', 'none') (default: 'cyclical')
        momentum_schedule (str): Momentum scheduling ('adaptive', 'fixed', 'nesterov') (default: 'adaptive')
        gradient_clipping (str): Gradient clipping method ('adaptive', 'global_norm', 'none') (default: 'adaptive')
        device (str): Target device ('auto', 'cuda', 'cpu') (default: 'auto')
        parallel_mode (str): Parallel processing mode ('thread', 'process', 'none') (default: 'thread')
        use_rtx_optimizations (bool): Enable RTX-specific optimizations (default: True)
        hindrance_detector (HindranceDetector): Custom hindrance detector (optional)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.9,
        weight_decay: float = 1e-4,
        eps: float = 1e-8,
        hindrance_threshold: float = 0.1,
        lr_scheduler: str = 'cyclical',
        momentum_schedule: str = 'adaptive',
        gradient_clipping: str = 'adaptive',
        device: str = 'auto',
        parallel_mode: str = 'thread',
        use_rtx_optimizations: bool = True,
        hindrance_detector: Optional[HindranceDetector] = None,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            eps=eps,
            hindrance_threshold=hindrance_threshold,
            lr_scheduler=lr_scheduler,
            momentum_schedule=momentum_schedule,
            gradient_clipping=gradient_clipping,
            device=device,
            parallel_mode=parallel_mode,
            use_rtx_optimizations=use_rtx_optimizations,
            **kwargs
        )

        super(AGMOHD, self).__init__(params, defaults)

        # Initialize device
        self.device = self._setup_device(device)

        # Initialize core components
        self.hindrance_detector = hindrance_detector or HindranceDetector(
            threshold=hindrance_threshold,
            adaptive_sensitivity=True
        )

        self.momentum_controller = MomentumController(
            initial_momentum=beta,
            schedule=momentum_schedule
        )

        self.lr_scheduler = CyclicalLRScheduler(
            base_lr=lr,
            mode=lr_scheduler
        )

        self.gradient_processor = GradientProcessor(
            clipping_method=gradient_clipping,
            use_rtx_optimizations=use_rtx_optimizations
        )

        self.performance_monitor = PerformanceMonitor()

        # Training state
        self.state = {}
        self.step_count = 0
        self.loss_history = []
        self.gradient_history = []

        # Configuration
        self.enable_monitoring_flag = False
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoints'

        # RTX optimizations
        self.use_rtx_optimizations = use_rtx_optimizations
        self._setup_rtx_optimizations()

        # Parallel processing
        self.parallel_mode = parallel_mode
        self._setup_parallel_processing()

        # Logging
        self.logger = logging.getLogger(__name__)

    def _setup_device(self, device: str) -> torch.device:
        """Setup the target device for computations."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device('cuda')
        elif device == 'cpu':
            return torch.device('cpu')
        else:
            raise ValueError(f"Invalid device: {device}")

    def _setup_rtx_optimizations(self):
        """Setup RTX-specific optimizations if available."""
        if self.use_rtx_optimizations and torch.cuda.is_available():
            # Enable TF32 for RTX 40-series and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Set optimal cuDNN settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def _setup_parallel_processing(self):
        """Setup parallel processing capabilities."""
        if self.parallel_mode == 'thread':
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        elif self.parallel_mode == 'process':
            import concurrent.futures
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        else:
            self.executor = None

    def _get_param_groups_device(self) -> torch.device:
        """Get the device of the first parameter group."""
        for group in self.param_groups:
            for p in group['params']:
                if p.device.type != 'meta':
                    return p.device
        return self.device

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float, optional): The loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update step count
        self.step_count += 1

        # Collect gradients and parameters
        gradients = []
        params_data = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Move to device if necessary
                if p.device != self.device:
                    p.data = p.data.to(self.device)
                    p.grad.data = p.grad.data.to(self.device)

                gradients.append(p.grad.data.clone())
                params_data.append(p.data.clone())

        if not gradients:
            return loss

        # Detect hindrance
        hindrance_level = self.hindrance_detector.detect_hindrance(
            gradients, self.loss_history
        )

        # Update momentum based on hindrance
        current_momentum = self.momentum_controller.adjust_momentum(hindrance_level)

        # Update learning rate
        current_lr = self.lr_scheduler.get_lr(self.step_count)

        # Process gradients
        processed_gradients = self.gradient_processor.process_gradients(
            gradients, hindrance_level
        )

        # Apply optimization step
        self._apply_step(processed_gradients, params_data, current_lr, current_momentum)

        # Update training statistics
        if loss is not None:
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 100:  # Keep last 100 losses
                self.loss_history.pop(0)

        self.gradient_history.append(torch.stack([g.norm() for g in gradients]).mean().item())
        if len(self.gradient_history) > 50:  # Keep last 50 gradient norms
            self.gradient_history.pop(0)

        # Update performance monitor
        self.performance_monitor.update(
            step=self.step_count,
            loss=loss.item() if loss is not None else None,
            lr=current_lr,
            momentum=current_momentum,
            hindrance=hindrance_level,
            gradient_norm=torch.stack([g.norm() for g in gradients]).mean().item()
        )

        # Log if monitoring is enabled
        if self.enable_monitoring_flag:
            self._log_training_stats()

        return loss

    def _apply_step(self, gradients: List[torch.Tensor],
                   params_data: List[torch.Tensor],
                   lr: float, momentum: float):
        """Apply the optimization step to parameters."""

        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = gradients[param_idx]

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                # Update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Apply update
                p.data.add_(momentum_buffer, alpha=-lr)

                # Update state
                state['step'] += 1
                param_idx += 1

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step': self.step_count,
            'current_loss': self.loss_history[-1] if self.loss_history else None,
            'current_lr': self.lr_scheduler.get_lr(self.step_count),
            'current_momentum': self.momentum_controller.get_current_momentum(),
            'hindrance_level': self.hindrance_detector.get_hindrance_level(),
            'gradient_norm': self.gradient_history[-1] if self.gradient_history else None,
            'memory_usage': self._get_memory_usage(),
            'gpu_utilization': self._get_gpu_utilization() if torch.cuda.is_available() else None
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if torch.cuda.is_available():
            return torch.cuda.utilization()
        return 0.0

    def enable_monitoring(self, log_dir: str = './logs'):
        """Enable training monitoring and logging."""
        self.enable_monitoring_flag = True
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def set_checkpoint_dir(self, checkpoint_dir: str = './checkpoints'):
        """Set checkpoint directory for automatic saving."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _log_training_stats(self):
        """Log current training statistics."""
        stats = self.get_training_stats()
        self.logger.info(f"Step {stats['step']}: Loss={stats['current_loss']:.4f}, "
                        f"LR={stats['current_lr']:.6f}, Momentum={stats['current_momentum']:.3f}, "
                        f"Hindrance={stats['hindrance_level']:.3f}")

    def save_checkpoint(self, filepath: str):
        """Save optimizer state to checkpoint."""
        checkpoint = {
            'step_count': self.step_count,
            'state': self.state,
            'param_groups': self.param_groups,
            'loss_history': self.loss_history,
            'gradient_history': self.gradient_history,
            'hindrance_detector_state': self.hindrance_detector.get_state(),
            'momentum_controller_state': self.momentum_controller.get_state(),
            'lr_scheduler_state': self.lr_scheduler.get_state(),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def load_checkpoint(self, filepath: str):
        """Load optimizer state from checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        self.step_count = checkpoint['step_count']
        self.state = checkpoint['state']
        self.param_groups = checkpoint['param_groups']
        self.loss_history = checkpoint['loss_history']
        self.gradient_history = checkpoint['gradient_history']

        # Restore component states
        if 'hindrance_detector_state' in checkpoint:
            self.hindrance_detector.set_state(checkpoint['hindrance_detector_state'])
        if 'momentum_controller_state' in checkpoint:
            self.momentum_controller.set_state(checkpoint['momentum_controller_state'])
        if 'lr_scheduler_state' in checkpoint:
            self.lr_scheduler.set_state(checkpoint['lr_scheduler_state'])

    def train_auto(self, train_loader, val_loader=None, num_epochs=10,
                  early_stopping=True, save_best=True, **kwargs):
        """
        Automated training with monitoring and optimization.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs
            early_stopping: Enable early stopping
            save_best: Save best model checkpoint
            **kwargs: Additional training arguments
        """
        best_val_loss = float('inf')
        patience = kwargs.get('patience', 5)
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)

                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0

                        if save_best:
                            checkpoint_path = os.path.join(
                                self.checkpoint_dir,
                                f'best_model_epoch_{epoch+1}.pt'
                            )
                            self.save_checkpoint(checkpoint_path)
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}")

    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.zero_grad()

            # Forward pass (assuming batch contains input and target)
            outputs = self.model(batch['input'])
            loss = self.criterion(outputs, batch['target'])

            # Backward pass
            loss.backward()

            # Optimization step
            self.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch['input'])
                loss = self.criterion(outputs, batch['target'])

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'lr={self.defaults["lr"]}, '
                f'beta={self.defaults["beta"]}, '
                f'weight_decay={self.defaults["weight_decay"]}, '
                f'hindrance_threshold={self.defaults["hindrance_threshold"]}, '
                f'lr_scheduler={self.defaults["lr_scheduler"]}, '
                f'momentum_schedule={self.defaults["momentum_schedule"]}, '
                f'gradient_clipping={self.defaults["gradient_clipping"]}, '
                f'device={self.device}, '
                f'parallel_mode={self.parallel_mode}, '
                f'use_rtx_optimizations={self.use_rtx_optimizations})')
