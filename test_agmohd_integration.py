#!/usr/bin/env python3
"""
Test script for AGMOHD optimizer integration with Transformers-style usage.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the src directory to path to import AGMOHD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agmohd.agmohd_transformers import AGMOHD, get_agmohd_schedule


def test_basic_functionality():
    """Test basic AGMOHD optimizer functionality."""
    print("Testing basic AGMOHD functionality...")

    # Create a simple model
    model = nn.Linear(10, 1)

    # Create AGMOHD optimizer
    optimizer = AGMOHD(model.parameters(), lr=1e-3, beta=0.9)

    # Create some dummy data
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Training loop
    for step in range(5):
        optimizer.zero_grad()

        output = model(input_data)
        loss = nn.functional.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        print(".4f")

    print("✓ Basic functionality test passed!")


def test_with_transformers_style():
    """Test AGMOHD with Transformers-style parameter groups."""
    print("\nTesting Transformers-style parameter groups...")

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # Different learning rates for different layers (Transformers style)
    optimizer = AGMOHD([
        {'params': model[0].parameters(), 'lr': 1e-3},
        {'params': model[2].parameters(), 'lr': 2e-3}
    ], lr=1e-3, beta=0.9)

    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    for step in range(3):
        optimizer.zero_grad()

        output = model(input_data)
        loss = nn.functional.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        print(".4f")

    print("✓ Transformers-style parameter groups test passed!")


def test_scheduler_integration():
    """Test AGMOHD scheduler integration."""
    print("\nTesting AGMOHD scheduler integration...")

    model = nn.Linear(10, 1)
    optimizer = AGMOHD(model.parameters(), lr=1e-3)

    # Create scheduler
    scheduler = get_agmohd_schedule(optimizer, initial_lr=1e-3)

    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    for step in range(5):
        optimizer.zero_grad()

        output = model(input_data)
        loss = nn.functional.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        current_lr = optimizer.get_lr()
        current_momentum = optimizer.get_momentum()
        hindrance_level = optimizer.get_hindrance_level()

        print(".6f")

    print("✓ Scheduler integration test passed!")


def test_adaptive_features():
    """Test AGMOHD adaptive features."""
    print("\nTesting AGMOHD adaptive features...")

    model = nn.Linear(10, 1)
    optimizer = AGMOHD(
        model.parameters(),
        lr=1e-3,
        hindrance_threshold=0.1,
        momentum_schedule='adaptive',
        gradient_clipping='adaptive'
    )

    # Simulate training with varying loss
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    for step in range(10):
        optimizer.zero_grad()

        output = model(input_data)
        # Add some noise to create varying loss
        noise = torch.randn_like(target) * 0.1 * (step % 3)
        loss = nn.functional.mse_loss(output, target + noise)

        loss.backward()
        optimizer.step()

        print(".4f")

    print("✓ Adaptive features test passed!")


def main():
    """Run all tests."""
    print("AGMOHD Integration Test Suite")
    print("=" * 40)

    try:
        test_basic_functionality()
        test_with_transformers_style()
        test_scheduler_integration()
        test_adaptive_features()

        print("\n" + "=" * 40)
        print("🎉 All tests passed! AGMOHD is ready for Transformers integration.")
        print("\nTo integrate into Hugging Face Transformers:")
        print("1. Follow the steps in integration_guide.md")
        print("2. Add the AGMOHD code to src/transformers/optimization.py")
        print("3. Update src/transformers/__init__.py")
        print("4. Add tests to tests/optimization/")
        print("5. Submit a pull request")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
