# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AGMOHD Optimizer for Transformers."""

import math
import warnings
from typing import Optional, Union

import torch
from torch.optim import Optimizer

from .hindrance_detector import HindranceDetector
from .momentum_controller import MomentumController
from .lr_scheduler import CyclicalLRScheduler
from .gradient_processor import GradientProcessor


class AGMOHD(Optimizer):
    """
    Adaptive Gradient Momentum with Hindrance Detection Optimizer.

    A revolutionary optimizer that adapts to training dynamics through
    intelligent hindrance detection and adaptive momentum control.

    This optimizer is designed to be compatible with Hugging Face Transformers
    and can be used with the Trainer class.

    Args:
        params (`iterable`):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        beta (`float`, *optional*, defaults to 0.9):
            Momentum factor.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty).
        eps (`float`, *optional*, defaults to 1e-8):
            Term added for numerical stability.
        hindrance_threshold (`float`, *optional*, defaults to 0.1):
            Threshold for hindrance detection.
        lr_scheduler (`str`, *optional*, defaults to "cyclical"):
            Learning rate scheduler mode ("cyclical", "cosine", "step", "none").
        momentum_schedule (`str`, *optional*, defaults to "adaptive"):
            Momentum scheduling ("adaptive", "fixed", "nesterov").
        gradient_clipping (`str`, *optional*, defaults to "adaptive"):
            Gradient clipping method ("adaptive", "global_norm", "none").
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        hindrance_threshold: float = 0.1,
        lr_scheduler: str = "cyclical",
        momentum_schedule: str = "adaptive",
        gradient_clipping: str = "adaptive",
        **kwargs
    ):
        if lr < 0.0:
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
            **kwargs
        )

        super().__init__(params, defaults)

        # Initialize core components
        self.hindrance_detector = HindranceDetector(
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
            use_rtx_optimizations=False  # Disable RTX for compatibility
        )

        # Training state
        self.step_count = 0
        self.loss_history = []

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update step count
        self.step_count += 1

        # Collect gradients
        gradients = []
        params_with_grad = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)
                    params_with_grad.append(p)

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
        self._apply_step(processed_gradients, params_with_grad, current_lr, current_momentum)

        # Update loss history
        if loss is not None:
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)

        return loss

    def _apply_step(self, gradients, params, lr, momentum):
        """Apply the optimization step to parameters."""
        for grad, p in zip(gradients, params):
            if grad is None:
                continue

            state = self.state[p]

            # Initialize state
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p)

            # Apply weight decay
            if p.grad is not None and self.defaults["weight_decay"] != 0:
                grad = grad.add(p, alpha=self.defaults["weight_decay"])

            # Update momentum buffer
            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

            # Apply update
            p.add_(momentum_buffer, alpha=-lr)

            state["step"] += 1

    def get_lr(self):
        """Get current learning rate."""
        return self.lr_scheduler.get_lr(self.step_count)

    def get_momentum(self):
        """Get current momentum."""
        return self.momentum_controller.get_current_momentum()

    def get_hindrance_level(self):
        """Get current hindrance level."""
        return self.hindrance_detector.get_hindrance_level()


# AGMOHD Schedule (similar to AdafactorSchedule)
class AGMOHDSchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    A schedule for AGMOHD that returns the current learning rate.
    """

    def __init__(self, optimizer, initial_lr=0.0):
        def lr_lambda(_):
            return initial_lr

        super().__init__(optimizer, lr_lambda)


def get_agmohd_schedule(optimizer, initial_lr=0.0):
    """
    Get a proxy schedule for AGMOHD.

    Args:
        optimizer: The AGMOHD optimizer.
        initial_lr: Initial learning rate.

    Returns:
        AGMOHDSchedule instance.
    """
    return AGMOHDSchedule(optimizer, initial_lr)
