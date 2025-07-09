#!/usr/bin/env python3
"""
PoC Training Validation for MotionAGFormer + MambaGCN
Validates that the integrated model can be trained successfully with:
- Small-scale training (batch_size=8, 1-2 epochs)
- Loss monitoring and gradient checking
- Baseline comparison
"""

from model.MotionAGFormer import MotionAGFormer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PoC_TrainingLogger:
    """Simple training logger for PoC validation"""

    def __init__(self, log_dir="poc_training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"poc_training_{timestamp}.log")
        self.metrics_file = os.path.join(
            log_dir, f"poc_metrics_{timestamp}.json")

        self.metrics = {
            "baseline": {},
            "mamba_gcn": {},
            "training_time": {},
            "memory_usage": {}
        }

        self.log("🚀 PoC Training Validation Started")
        self.log(f"PyTorch version: {torch.__version__}")
        self.log(f"CUDA available: {torch.cuda.is_available()}")

    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


def create_synthetic_dataset(num_samples=200, n_frames=27, n_joints=17):
    """Create synthetic 2D->3D pose data for training validation"""

    # Generate synthetic 2D poses (input)
    input_2d = torch.randn(num_samples, n_frames, n_joints, 2)

    # Generate synthetic 3D poses (target)
    # Add some correlation with 2D to make learning possible
    target_3d = torch.randn(num_samples, n_frames, n_joints, 3)
    target_3d[:, :, :, :2] = input_2d + \
        torch.randn_like(input_2d) * 0.1  # Add noise

    return input_2d, target_3d


def check_gradients(model, logger):
    """Check gradient norms and detect potential issues"""
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    large_grad_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

            if param_norm.item() < 1e-8:
                zero_grad_count += 1
            elif param_norm.item() > 10.0:
                large_grad_count += 1
                logger.log(
                    f"⚠️  Large gradient in {name}: {param_norm.item():.6f}")
        else:
            logger.log(f"⚠️  No gradient for parameter: {name}")

    total_norm = total_norm ** 0.5

    gradient_health = {
        "total_norm": total_norm,
        "param_count": param_count,
        "zero_grad_count": zero_grad_count,
        "large_grad_count": large_grad_count,
        "avg_grad_norm": total_norm / max(param_count, 1)
    }

    return gradient_health


def check_memory_usage():
    """Check GPU memory usage if available"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return {
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved
        }
    return {"allocated_gb": 0, "reserved_gb": 0}


def train_model(model, train_loader, model_name, logger, epochs=2):
    """Train model for specified epochs and monitor metrics"""

    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logger.log(f"\n{'='*60}")
    logger.log(f"🏋️  Training {model_name}")
    logger.log(f"   Device: {device}")
    logger.log(
        f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"   Epochs: {epochs}")
    logger.log(f"   Batches per epoch: {len(train_loader)}")

    model.train()

    training_metrics = {
        "losses": [],
        "gradient_norms": [],
        "memory_usage": [],
        "epoch_times": []
    }

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []

        logger.log(f"\n📊 Epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Check gradients
            grad_health = check_gradients(model, logger)

            # Check for gradient explosion
            if grad_health["total_norm"] > 100.0:
                logger.log(
                    f"⚠️  Potential gradient explosion! Norm: {grad_health['total_norm']:.2f}")

            # Clip gradients if needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())
            training_metrics["losses"].append(loss.item())
            training_metrics["gradient_norms"].append(
                grad_health["total_norm"])

            # Memory check
            memory_info = check_memory_usage()
            training_metrics["memory_usage"].append(
                memory_info["allocated_gb"])

            if batch_idx == 0 or (batch_idx + 1) % 5 == 0:
                logger.log(f"   Batch {batch_idx + 1:2d}: Loss={loss.item():.6f}, "
                           f"Grad_norm={grad_health['total_norm']:.4f}, "
                           f"Mem={memory_info['allocated_gb']:.2f}GB")

        epoch_time = time.time() - epoch_start_time
        training_metrics["epoch_times"].append(epoch_time)

        avg_loss = np.mean(epoch_losses)
        logger.log(f"   📈 Epoch {epoch + 1} Summary:")
        logger.log(f"      Average Loss: {avg_loss:.6f}")
        logger.log(f"      Epoch Time: {epoch_time:.2f}s")
        logger.log(
            f"      Loss Trend: {'📉 Decreasing' if len(training_metrics['losses']) > 10 and training_metrics['losses'][-1] < training_metrics['losses'][-10] else '📊 Monitoring'}")

    total_training_time = time.time() - total_start_time
    logger.log(
        f"\n✅ Training {model_name} completed in {total_training_time:.2f}s")

    # Training health assessment
    final_loss = training_metrics["losses"][-1]
    initial_loss = training_metrics["losses"][0]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    logger.log(f"📊 Training Assessment:")
    logger.log(f"   Initial Loss: {initial_loss:.6f}")
    logger.log(f"   Final Loss: {final_loss:.6f}")
    logger.log(f"   Loss Reduction: {loss_reduction:.2f}%")
    logger.log(f"   Max Memory: {max(training_metrics['memory_usage']):.2f}GB")
    logger.log(
        f"   Avg Gradient Norm: {np.mean(training_metrics['gradient_norms']):.4f}")

    training_success = (
        loss_reduction > 0 and  # Loss should decrease
        final_loss < 100.0 and  # Loss should be reasonable
        max(training_metrics['gradient_norms']
            ) < 1000.0  # No gradient explosion
    )

    logger.log(
        f"   Status: {'✅ SUCCESS' if training_success else '❌ ISSUES DETECTED'}")

    return training_metrics, training_success


def run_poc_validation():
    """Run complete PoC training validation"""

    logger = PoC_TrainingLogger()

    # Configuration
    batch_size = 8
    epochs = 2
    n_frames = 27  # Smaller for faster training
    n_joints = 17
    dim_feat = 64  # Smaller for faster training
    num_samples = 80  # Small dataset for PoC

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"🔧 PoC Configuration:")
    logger.log(f"   Batch size: {batch_size}")
    logger.log(f"   Epochs: {epochs}")
    logger.log(f"   Sequence length: {n_frames}")
    logger.log(f"   Dataset size: {num_samples}")
    logger.log(f"   Feature dimension: {dim_feat}")
    logger.log(f"   Device: {device}")

    # Create synthetic dataset
    logger.log(f"\n📊 Creating synthetic dataset...")
    input_data, target_data = create_synthetic_dataset(
        num_samples, n_frames, n_joints)

    # Create data loader
    dataset = TensorDataset(input_data, target_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.log(f"   Input shape: {input_data.shape}")
    logger.log(f"   Target shape: {target_data.shape}")
    logger.log(f"   Batches: {len(train_loader)}")

    # Model configurations to test
    model_configs = [
        {
            "name": "Baseline MotionAGFormer",
            "config": {
                "use_mamba_gcn": False,
                "hierarchical": False
            }
        },
        {
            "name": "MotionAGFormer + MambaGCN",
            "config": {
                "use_mamba_gcn": True,
                "mamba_gcn_use_mamba": True,
                "mamba_gcn_use_attention": False,
                "hierarchical": False
            }
        },
        {
            "name": "MotionAGFormer + MambaGCN (Full)",
            "config": {
                "use_mamba_gcn": True,
                "mamba_gcn_use_mamba": True,
                "mamba_gcn_use_attention": True,
                "hierarchical": False
            }
        }
    ]

    results = {}

    # Test each configuration
    for model_info in model_configs:
        model_name = model_info["name"]
        config = model_info["config"]

        try:
            # Create model
            model = MotionAGFormer(
                n_layers=4,  # Smaller for faster training
                dim_in=2,
                dim_feat=dim_feat,
                dim_rep=128,
                dim_out=3,
                n_frames=n_frames,
                **config
            ).to(device)

            # Train model
            training_metrics, success = train_model(
                model, train_loader, model_name, logger, epochs
            )

            results[model_name] = {
                "success": success,
                "metrics": training_metrics,
                "config": config
            }

            # Store metrics for comparison
            if "Baseline" in model_name:
                logger.metrics["baseline"] = training_metrics
            else:
                logger.metrics["mamba_gcn"] = training_metrics

        except Exception as e:
            logger.log(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {
                "success": False,
                "error": str(e),
                "config": config
            }

    # Performance comparison
    logger.log(f"\n{'='*60}")
    logger.log(f"📊 PoC TRAINING VALIDATION SUMMARY")
    logger.log(f"{'='*60}")

    successful_models = 0
    total_models = len(model_configs)

    for model_name, result in results.items():
        if result["success"]:
            successful_models += 1
            final_loss = result["metrics"]["losses"][-1]
            training_time = sum(result["metrics"]["epoch_times"])
            max_memory = max(result["metrics"]["memory_usage"])

            logger.log(f"✅ {model_name}")
            logger.log(f"   Final Loss: {final_loss:.6f}")
            logger.log(f"   Training Time: {training_time:.2f}s")
            logger.log(f"   Max Memory: {max_memory:.2f}GB")
        else:
            logger.log(f"❌ {model_name}")
            if "error" in result:
                logger.log(f"   Error: {result['error']}")

    # Overall assessment
    logger.log(f"\n🎯 OVERALL RESULTS:")
    logger.log(
        f"   Success Rate: {successful_models}/{total_models} ({successful_models/total_models*100:.1f}%)")

    if successful_models >= 2:  # At least baseline + one MambaGCN variant
        logger.log(f"✅ PoC TRAINING VALIDATION PASSED!")
        logger.log(f"   - Models can be trained successfully")
        logger.log(f"   - Losses decrease as expected")
        logger.log(f"   - No memory overflow or gradient explosion")
        logger.log(f"   - Integration is stable")
        validation_passed = True
    else:
        logger.log(f"❌ PoC TRAINING VALIDATION FAILED!")
        logger.log(f"   - Check training logs for issues")
        validation_passed = False

    # Save all metrics
    logger.metrics["validation_passed"] = validation_passed
    logger.metrics["success_rate"] = successful_models / total_models
    logger.save_metrics()

    logger.log(f"\n📁 Logs saved to: {logger.log_file}")
    logger.log(f"📁 Metrics saved to: {logger.metrics_file}")

    return validation_passed, results


if __name__ == "__main__":
    print("🧪 Starting PoC Training Validation for MotionAGFormer + MambaGCN")
    print("=" * 70)

    success, results = run_poc_validation()

    if success:
        print("\n🎉 PoC Training Validation COMPLETED SUCCESSFULLY!")
        print("✅ Task 2.3: PoC 训练验证 - PASSED")
    else:
        print("\n⚠️  PoC Training Validation encountered issues.")
        print("❌ Task 2.3: PoC 训练验证 - NEEDS ATTENTION")

    exit(0 if success else 1)
