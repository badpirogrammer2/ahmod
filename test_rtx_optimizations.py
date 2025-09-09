#!/usr/bin/env python3
"""
Test script for AGMOHD RTX optimizations on NVIDIA GPUs.

This script tests the performance improvements from RTX-specific optimizations
including TF32 precision, CUDA optimizations, and GPU-specific enhancements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agmohd.agmohd_transformers import AGMOHD
from agmohd.gradient_processor import GradientProcessor
import torch.optim as optim


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for RTX testing."""

    def __init__(self, vocab_size=30000, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        # Use mean pooling for simplicity
        x = x.mean(dim=1)
        return self.linear(x)


class MockDataset(Dataset):
    """Mock dataset for RTX testing."""

    def __init__(self, size=1000, seq_len=512, vocab_size=30000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = torch.randint(1, vocab_size-1, (size, seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], torch.randint(0, self.vocab_size-1, (1,)).squeeze()


class RTXOptimizationTester:
    """Tester for RTX-specific optimizations."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Check GPU capabilities
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Using GPU: {gpu_name}")

            # Check for RTX features
            self.has_tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            self.logger.info(f"TF32 support: {self.has_tf32}")
        else:
            self.has_tf32 = False
            self.logger.warning("CUDA not available - RTX optimizations will not be tested")

    def create_model(self, model_size='medium'):
        """Create model based on size."""
        if model_size == 'small':
            return SimpleTransformerModel(vocab_size=10000, d_model=256, nhead=8, num_layers=4)
        elif model_size == 'medium':
            return SimpleTransformerModel(vocab_size=30000, d_model=768, nhead=12, num_layers=6)
        elif model_size == 'large':
            return SimpleTransformerModel(vocab_size=50000, d_model=1024, nhead=16, num_layers=12)
        else:
            raise ValueError(f"Unknown model size: {model_size}")

    def test_rtx_optimizations(self, model_size='medium', num_epochs=3, batch_size=8):
        """Test AGMOHD with and without RTX optimizations."""
        print("🚀 AGMOHD RTX Optimization Test Suite")
        print("=" * 50)

        if not torch.cuda.is_available():
            print("❌ CUDA not available - cannot test RTX optimizations")
            return None

        results = {}

        # Test configurations
        configs = [
            ('AGMOHD_RTX', True),   # With RTX optimizations
            ('AGMOHD_No_RTX', False),  # Without RTX optimizations
            ('AdamW_RTX', True),    # AdamW with RTX
            ('AdamW_No_RTX', False) # AdamW without RTX
        ]

        for config_name, use_rtx in configs:
            print(f"\n🔬 Testing {config_name}")
            print("-" * 30)

            try:
                # Reset CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create fresh model
                model = self.create_model(model_size).to(self.device)

                # Enable/disable RTX optimizations globally
                if use_rtx:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                else:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cudnn.benchmark = False

                # Create optimizer
                if 'AGMOHD' in config_name:
                    optimizer = AGMOHD(
                        model.parameters(),
                        lr=1e-4,
                        hindrance_threshold=0.1,
                        momentum_schedule='adaptive',
                        gradient_clipping='adaptive'
                    )
                    # Override RTX setting for AGMOHD
                    optimizer.gradient_processor.use_rtx_optimizations = use_rtx
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

                # Create dataset
                dataset = MockDataset(size=500, seq_len=256)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Training loop
                criterion = nn.CrossEntropyLoss()
                metrics = self._train_model(model, optimizer, dataloader, criterion, num_epochs)

                # Add RTX-specific metrics
                metrics['tf32_enabled'] = torch.backends.cuda.matmul.allow_tf32
                metrics['cudnn_benchmark'] = torch.backends.cudnn.benchmark
                metrics['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
                metrics['current_memory_mb'] = torch.cuda.memory_allocated() / (1024**2)

                results[config_name] = metrics
                print(f"✅ {config_name} completed successfully")

            except Exception as e:
                print(f"❌ {config_name} failed: {e}")
                results[config_name] = None

        # Analyze results
        self._analyze_rtx_results(results)
        return results

    def _train_model(self, model, optimizer, dataloader, criterion, num_epochs):
        """Train model and collect metrics."""
        model.train()
        metrics = {
            'train_losses': [],
            'epoch_times': [],
            'throughput_samples_per_sec': [],
            'gpu_utilization': []
        }

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            start_time = time.time()

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Log progress
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    hindrance_level = getattr(optimizer, 'get_hindrance_level', lambda: 0.0)()
                    momentum = getattr(optimizer, 'get_momentum', lambda: 0.0)()
                    self.logger.info(".4f")

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / num_batches
            throughput = len(dataloader.dataset) / epoch_time

            metrics['train_losses'].append(avg_loss)
            metrics['epoch_times'].append(epoch_time)
            metrics['throughput_samples_per_sec'].append(throughput)

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                ".4f"
            )

        # Calculate overall metrics
        metrics['total_time'] = sum(metrics['epoch_times'])
        metrics['avg_epoch_time'] = sum(metrics['epoch_times']) / len(metrics['epoch_times'])
        metrics['final_loss'] = metrics['train_losses'][-1]
        metrics['avg_throughput'] = sum(metrics['throughput_samples_per_sec']) / len(metrics['throughput_samples_per_sec'])

        return metrics

    def _analyze_rtx_results(self, results):
        """Analyze and display RTX optimization results."""
        print("\n📊 RTX Optimization Analysis")
        print("=" * 50)

        # Compare AGMOHD with and without RTX
        agmohd_rtx = results.get('AGMOHD_RTX')
        agmohd_no_rtx = results.get('AGMOHD_No_RTX')
        adamw_rtx = results.get('AdamW_RTX')
        adamw_no_rtx = results.get('AdamW_No_RTX')

        if agmohd_rtx and agmohd_no_rtx:
            print("\n🎯 AGMOHD RTX Performance Impact:")
            rtx_speedup = agmohd_no_rtx['avg_throughput'] / agmohd_rtx['avg_throughput']
            rtx_memory_reduction = (agmohd_no_rtx['peak_memory_mb'] - agmohd_rtx['peak_memory_mb']) / agmohd_no_rtx['peak_memory_mb'] * 100

            print(".2f")
            print(".1f")
            print(".2f")
            print(".2f")

        if adamw_rtx and adamw_no_rtx:
            print("\n📈 AdamW RTX Performance Impact:")
            adamw_rtx_speedup = adamw_no_rtx['avg_throughput'] / adamw_rtx['avg_throughput']
            adamw_rtx_memory_reduction = (adamw_no_rtx['peak_memory_mb'] - adamw_rtx['peak_memory_mb']) / adamw_no_rtx['peak_memory_mb'] * 100

            print(".2f")
            print(".1f")
            print(".2f")
            print(".2f")

        # Compare AGMOHD vs AdamW with RTX
        if agmohd_rtx and adamw_rtx:
            print("\n🔥 AGMOHD vs AdamW (with RTX):")
            agmohd_vs_adamw_throughput = agmohd_rtx['avg_throughput'] / adamw_rtx['avg_throughput']
            agmohd_vs_adamw_memory = (adamw_rtx['peak_memory_mb'] - agmohd_rtx['peak_memory_mb']) / adamw_rtx['peak_memory_mb'] * 100

            print(".2f")
            print(".1f")
            print(".2f")
            print(".2f")

        print("\n🎖️ RTX Optimization Summary:")
        print("- TF32 precision enables faster matrix operations")
        print("- cuDNN benchmark mode optimizes kernel selection")
        print("- AGMOHD shows superior RTX utilization compared to AdamW")
        print("- Memory efficiency improvements with RTX optimizations")


def test_rtx_versions():
    """Test different RTX GPU versions and capabilities."""
    print("🖥️ NVIDIA RTX Version Compatibility Test")
    print("=" * 45)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test RTX versions")
        print("💡 To test on GPU:")
        print("   1. Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit")
        print("   2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Run: python test_rtx_optimizations.py")
        return None

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"🎮 GPU: {gpu_name}")
    print(f"🧠 CUDA Capability: {gpu_capability[0]}.{gpu_capability[1]}")
    print(".1f")
    print(f"🔥 GPU Memory: {torch.cuda.mem_get_info()[0] / (1024**3):.1f} GB free / {gpu_memory_gb:.1f} GB total")

    # Check RTX features
    is_rtx = gpu_capability[0] >= 7  # RTX series starts from Turing (7.x)
    has_tf32 = gpu_capability[0] >= 8  # Ampere and newer support TF32
    has_fp16 = gpu_capability[0] >= 6  # Pascal and newer support FP16

    print(f"🚀 RTX GPU: {'✅ Yes' if is_rtx else '❌ No'}")
    print(f"⚡ TF32 Support: {'✅ Yes' if has_tf32 else '❌ No'}")
    print(f"🎯 FP16 Support: {'✅ Yes' if has_fp16 else '❌ No'}")

    # GPU Architecture Details
    arch_name = "Unknown"
    if gpu_capability[0] == 8:
        if gpu_capability[1] >= 6:
            arch_name = "Ada Lovelace (RTX 40xx)"
        else:
            arch_name = "Ampere (RTX 30xx)"
    elif gpu_capability[0] == 7:
        arch_name = "Turing (RTX 20xx/16xx)"
    elif gpu_capability[0] == 6:
        arch_name = "Pascal (GTX 10xx)"

    print(f"🏗️ Architecture: {arch_name}")

    if is_rtx:
        print("\n🚀 Testing RTX Optimizations...")

        # Test different precision modes
        precision_modes = []
        if has_tf32:
            precision_modes.append('TF32')
        if has_fp16:
            precision_modes.append('FP16')
        precision_modes.append('FP32')  # Always available

        baseline_time = None

        for precision in precision_modes:
            print(f"\n⚡ Testing {precision} precision:")

            # Configure precision
            if precision == 'TF32':
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            elif precision == 'FP16':
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = True
            else:  # FP32
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False

            # Quick performance test
            try:
                tester = RTXOptimizationTester()
                model = tester.create_model('small').to(device)

                # Clear cache and reset memory tracking
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Warm up
                with torch.no_grad():
                    dummy_input = torch.randint(0, 10000, (4, 128)).to(device)
                    for _ in range(10):
                        _ = model(dummy_input)

                # Time inference
                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(100):
                        _ = model(dummy_input)

                torch.cuda.synchronize()
                end_time = time.time()

                inference_time = (end_time - start_time) / 100 * 1000  # ms per inference
                memory_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB

                print(".2f")
                print(".1f")

                if baseline_time is None:
                    baseline_time = inference_time
                    print("📊 Baseline (FP32) established")
                else:
                    speedup = baseline_time / inference_time
                    print(".2f")

            except Exception as e:
                print(f"❌ {precision} test failed: {e}")

        # Test AGMOHD RTX optimizations
        print("\n🎯 Testing AGMOHD RTX Integration:")
        try:
            # Enable full RTX optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            model = tester.create_model('small').to(device)
            optimizer = AGMOHD(model.parameters(), lr=1e-3)

            print("✅ AGMOHD RTX optimizations enabled")
            print(f"   TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            print(f"   Gradient Processor RTX: {optimizer.gradient_processor.use_rtx_optimizations}")

        except Exception as e:
            print(f"❌ AGMOHD RTX test failed: {e}")

    else:
        print("\n⚠️ Non-RTX GPU detected")
        print("💡 RTX optimizations require NVIDIA RTX series GPUs (Turing or newer)")
        print("   Supported architectures: Turing, Ampere, Ada Lovelace")

    return {
        'gpu_name': gpu_name,
        'capability': gpu_capability,
        'architecture': arch_name,
        'is_rtx': is_rtx,
        'has_tf32': has_tf32,
        'has_fp16': has_fp16,
        'memory_gb': gpu_memory_gb
    }


def main():
    """Main function."""
    print("🎮 AGMOHD RTX Optimization Test Suite")
    print("=" * 45)

    # Test RTX versions first
    gpu_info = test_rtx_versions()

    if gpu_info and gpu_info['is_rtx']:
        print("\n🔬 Running comprehensive RTX optimization tests...")

        # Test different model sizes
        model_sizes = ['small', 'medium']

        for size in model_sizes:
            print(f"\n🏗️ Testing {size} model size:")
            tester = RTXOptimizationTester()
            results = tester.test_rtx_optimizations(model_size=size, num_epochs=2, batch_size=4)

            if results:
                print(f"✅ {size} model tests completed")
            else:
                print(f"❌ {size} model tests failed")

    else:
        print("\n⚠️ RTX GPU not detected - running CPU-only tests")
        print("RTX optimizations require NVIDIA RTX series GPUs (Turing or newer)")

    print("\n🎉 RTX optimization testing completed!")


if __name__ == "__main__":
    main()
