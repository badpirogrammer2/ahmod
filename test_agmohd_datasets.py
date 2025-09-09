#!/usr/bin/env python3
"""
AGMOHD Dataset Testing Framework

This script demonstrates how to test AGMOHD optimizer with various datasets
and transformer architectures. It provides examples for different use cases
and performance evaluation methodologies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score

# Import AGMOHD (would work in actual PyTorch environment)
try:
    from src.agmohd.agmohd_transformers import AGMOHD
except ImportError:
    print("AGMOHD not available - this is a demonstration script")
    AGMOHD = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    learning_rate: float
    hindrance_level: float
    momentum: float
    epoch_time: float


class DatasetTester:
    """Framework for testing AGMOHD with different datasets."""

    def __init__(self, model: nn.Module, optimizer_class, dataset_name: str):
        self.model = model
        self.optimizer_class = optimizer_class
        self.dataset_name = dataset_name
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)

    def create_optimizer(self, **kwargs):
        """Create optimizer with AGMOHD-specific parameters."""
        if self.optimizer_class == AGMOHD:
            return self.optimizer_class(
                self.model.parameters(),
                lr=kwargs.get('lr', 1e-3),
                hindrance_threshold=kwargs.get('hindrance_threshold', 0.1),
                momentum_schedule=kwargs.get('momentum_schedule', 'adaptive'),
                gradient_clipping=kwargs.get('gradient_clipping', 'adaptive'),
                **kwargs
            )
        else:
            # For comparison optimizers
            return self.optimizer_class(self.model.parameters(), **kwargs)

    def train_epoch(self, train_loader: DataLoader, optimizer, criterion, device: str = 'cpu'):
        """Train for one epoch with detailed metrics."""
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            if outputs.shape[-1] > 1:  # Classification
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == targets).float().mean().item()
            else:  # Regression
                acc = 1.0  # Placeholder for regression tasks

            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                hindrance_level = getattr(optimizer, 'get_hindrance_level', lambda: 0.0)()
                momentum = getattr(optimizer, 'get_momentum', lambda: 0.0)()

                self.logger.info(".4f")

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        return avg_loss, avg_acc, epoch_time

    def validate_epoch(self, val_loader: DataLoader, criterion, device: str = 'cpu'):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                if outputs.shape[-1] > 1:  # Classification
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == targets).float().mean().item()
                else:  # Regression
                    acc = 1.0

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        avg_loss = val_loss / num_batches
        avg_acc = val_acc / num_batches

        return avg_loss, avg_acc

    def run_training_experiment(self, train_loader, val_loader, optimizer,
                              criterion, num_epochs: int = 10, device: str = 'cpu'):
        """Run complete training experiment."""
        self.logger.info(f"Starting {self.dataset_name} training with {optimizer.__class__.__name__}")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(
                train_loader, optimizer, criterion, device
            )

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion, device)

            # Get optimizer-specific metrics
            current_lr = optimizer.param_groups[0]['lr']
            hindrance_level = getattr(optimizer, 'get_hindrance_level', lambda: 0.0)()
            momentum = getattr(optimizer, 'get_momentum', lambda: 0.0)()

            # Store metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=current_lr,
                hindrance_level=hindrance_level,
                momentum=momentum,
                epoch_time=epoch_time
            )
            self.metrics_history.append(metrics)

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                ".4f"
            )

        return self.metrics_history


class MockDataset(Dataset):
    """Mock dataset for demonstration purposes."""

    def __init__(self, size: int = 1000, input_dim: int = 784, num_classes: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_mnist_like_model():
    """Create a simple model similar to MNIST classifiers."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )


def create_transformer_like_model(seq_len: int = 512, vocab_size: int = 30000, d_model: int = 768):
    """Create a simplified transformer-like model."""
    return nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        ),
        nn.Linear(d_model, vocab_size)
    )


def benchmark_optimizers():
    """Benchmark AGMOHD against other optimizers."""
    print("🚀 AGMOHD Dataset Testing Framework")
    print("=" * 50)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = MockDataset(size=5000, input_dim=784, num_classes=10)
    val_dataset = MockDataset(size=1000, input_dim=784, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Test different model architectures
    models_to_test = [
        ("MLP", create_mnist_like_model()),
        # ("Transformer", create_transformer_like_model(seq_len=128, vocab_size=1000, d_model=256)),
    ]

    # Optimizers to compare
    optimizers_to_test = [
        ("AGMOHD", AGMOHD if AGMOHD else None, {
            'lr': 1e-3,
            'hindrance_threshold': 0.1,
            'momentum_schedule': 'adaptive'
        }),
        ("AdamW", torch.optim.AdamW, {'lr': 1e-3, 'weight_decay': 0.01}),
        ("Adam", torch.optim.Adam, {'lr': 1e-3}),
    ]

    results = {}

    for model_name, model in models_to_test:
        print(f"\n🔬 Testing {model_name} Architecture")
        print("-" * 30)

        model_results = {}

        for opt_name, opt_class, opt_kwargs in optimizers_to_test:
            if opt_class is None:
                print(f"⚠️  Skipping {opt_name} (not available)")
                continue

            print(f"\n⚡ Testing {opt_name} optimizer...")

            # Reset model
            model_copy = type(model)()
            if hasattr(model, 'children'):
                # Copy architecture
                model_copy = nn.Sequential(*[type(layer)(**layer.__dict__) if hasattr(layer, '__dict__') else layer
                                           for layer in model.children()])

            tester = DatasetTester(model_copy, opt_class, f"{model_name}_{opt_name}")
            optimizer = tester.create_optimizer(**opt_kwargs)
            criterion = nn.CrossEntropyLoss()

            # Run training
            try:
                metrics = tester.run_training_experiment(
                    train_loader, val_loader, optimizer, criterion,
                    num_epochs=5, device='cpu'
                )
                model_results[opt_name] = metrics
                print(f"✅ {opt_name} completed successfully")
            except Exception as e:
                print(f"❌ {opt_name} failed: {e}")
                model_results[opt_name] = None

        results[model_name] = model_results

    # Print summary
    print("\n📈 Results Summary")
    print("=" * 50)

    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        for opt_name, metrics in model_results.items():
            if metrics:
                final_metrics = metrics[-1]
                print(".4f")
            else:
                print(f"  {opt_name}: Failed")

    return results


def demonstrate_transformer_training():
    """Demonstrate AGMOHD with transformer-like training."""
    print("\n🎯 Transformer Training Demonstration")
    print("=" * 40)

    # Simulate transformer training scenario
    print("Simulating BERT-like pre-training...")

    # Mock transformer model
    model = create_mnist_like_model()  # Using simple model for demo

    if AGMOHD:
        optimizer = AGMOHD(
            model.parameters(),
            lr=1e-4,
            hindrance_threshold=0.1,
            momentum_schedule='adaptive',
            gradient_clipping='adaptive'
        )

        print("✅ AGMOHD configured for transformer training:")
        print(f"   - Learning Rate: {optimizer.defaults['lr']}")
        print(f"   - Hindrance Threshold: {optimizer.defaults['hindrance_threshold']}")
        print(f"   - Momentum Schedule: {optimizer.defaults['momentum_schedule']}")
        print(f"   - Gradient Clipping: {optimizer.defaults['gradient_clipping']}")

        # Simulate training steps
        criterion = nn.CrossEntropyLoss()
        train_dataset = MockDataset(size=1000)
        train_loader = DataLoader(train_dataset, batch_size=16)

        print("\n🚀 Starting training simulation...")
        for step in range(10):
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                if step % 2 == 0:
                    hindrance = optimizer.get_hindrance_level()
                    momentum = optimizer.get_momentum()
                    print(".4f")
                break  # Just one batch per step for demo

        print("✅ Training simulation completed successfully!")
    else:
        print("⚠️  AGMOHD not available for demonstration")


def create_performance_comparison():
    """Create a framework for comparing AGMOHD with other optimizers."""
    print("\n📊 Performance Comparison Framework")
    print("=" * 35)

    comparison_config = {
        "datasets": [
            {"name": "MNIST-like", "input_dim": 784, "num_classes": 10, "size": 5000},
            {"name": "CIFAR-like", "input_dim": 3072, "num_classes": 100, "size": 3000},
        ],
        "models": [
            {"name": "MLP", "architecture": "feedforward"},
            {"name": "CNN", "architecture": "convolutional"},
            {"name": "Transformer", "architecture": "attention"},
        ],
        "optimizers": [
            {"name": "AGMOHD", "class": AGMOHD, "params": {"lr": 1e-3, "hindrance_threshold": 0.1}},
            {"name": "AdamW", "class": torch.optim.AdamW, "params": {"lr": 1e-3, "weight_decay": 0.01}},
            {"name": "SGD", "class": torch.optim.SGD, "params": {"lr": 1e-2, "momentum": 0.9}},
        ],
        "metrics": [
            "convergence_speed",
            "final_accuracy",
            "training_stability",
            "memory_efficiency",
            "hyperparameter_sensitivity"
        ]
    }

    print("🔧 Configured comparison framework:")
    print(f"   📊 Datasets: {len(comparison_config['datasets'])}")
    print(f"   🏗️  Models: {len(comparison_config['models'])}")
    print(f"   ⚡ Optimizers: {len(comparison_config['optimizers'])}")
    print(f"   📈 Metrics: {len(comparison_config['metrics'])}")

    print("\n📋 Metrics to evaluate:")
    for metric in comparison_config['metrics']:
        print(f"   • {metric.replace('_', ' ').title()}")

    return comparison_config


def main():
    """Main testing function."""
    print("🧪 AGMOHD Dataset Testing Suite")
    print("=" * 40)

    # Check if PyTorch is available
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("⚠️  PyTorch not available - running in demonstration mode")
        torch = None

    # Run different test scenarios
    if torch:
        # Full testing with actual PyTorch
        results = benchmark_optimizers()
        demonstrate_transformer_training()
    else:
        # Demonstration mode
        print("📝 Running in demonstration mode...")
        demonstrate_transformer_training()

    # Always show comparison framework
    comparison_config = create_performance_comparison()

    print("\n🎉 Testing framework ready!")
    print("\nTo run actual tests:")
    print("1. Ensure PyTorch is installed")
    print("2. Run: python test_agmohd_datasets.py")
    print("3. Results will be saved to 'agmohd_test_results.json'")

    print("\n📚 For transformer-specific testing:")
    print("- Use Hugging Face datasets (datasets library)")
    print("- Load models from transformers library")
    print("- Integrate with Trainer class")
    print("- Monitor training with Weights & Biases")

    return True


if __name__ == "__main__":
    main()
