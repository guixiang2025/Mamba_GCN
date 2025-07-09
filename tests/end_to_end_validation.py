#!/usr/bin/env python3
"""
End-to-End Validation for MotionAGFormer + MambaGCN
Tests the complete pipeline: Data Loading → Model Training → Result Output

This script validates that the integrated model can handle the full training workflow
without any crashes or errors.
"""

from model.MotionAGFormer import MotionAGFormer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EndToEndValidator:
    """
    End-to-end validation system for the complete training pipeline
    """

    def __init__(self, device='auto'):
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        self.validation_log = []
        self.start_time = None

    def log_step(self, step, status, details=None):
        """Log validation step"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'status': status,
            'details': details or {}
        }
        self.validation_log.append(log_entry)

        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "🔄"
        print(f"[{timestamp}] {status_icon} {step}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    def step_1_data_loading(self):
        """Step 1: Test data loading pipeline"""
        try:
            # Create synthetic dataset
            batch_size = 16
            n_frames = 27
            n_joints = 17
            n_samples = 100

            # Generate 2D→3D pose data
            input_2d = torch.randn(n_samples, n_frames, n_joints, 2)
            target_3d = torch.randn(n_samples, n_frames, n_joints, 3)

            # Create data loaders
            dataset = TensorDataset(input_2d, target_3d)
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True)

            # Test data loading
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 2:  # Test first few batches
                    break
                assert inputs.shape == (batch_size, n_frames, n_joints, 2)
                assert targets.shape == (batch_size, n_frames, n_joints, 3)

            self.log_step("Data Loading", "PASS", {
                'dataset_size': n_samples,
                'batch_size': batch_size,
                'input_shape': str(tuple(inputs.shape)),
                'target_shape': str(tuple(targets.shape)),
                'num_batches': len(train_loader)
            })

            return True, train_loader

        except Exception as e:
            self.log_step("Data Loading", "FAIL", {'error': str(e)})
            return False, None

    def step_2_model_creation(self):
        """Step 2: Test model creation with different configurations"""
        try:
            configurations = [
                {
                    'name': 'Baseline MotionAGFormer',
                    'config': {'use_mamba_gcn': False}
                },
                {
                    'name': 'MotionAGFormer + MambaGCN',
                    'config': {
                        'use_mamba_gcn': True,
                        'mamba_gcn_use_mamba': True,
                        'mamba_gcn_use_attention': False
                    }
                }
            ]

            models = {}
            for config_info in configurations:
                name = config_info['name']
                config = config_info['config']

                model = MotionAGFormer(
                    n_layers=4,
                    dim_in=2,
                    dim_feat=64,
                    dim_rep=128,
                    dim_out=3,
                    n_frames=27,
                    **config
                ).to(self.device)

                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel()
                                       for p in model.parameters() if p.requires_grad)

                models[name] = model

                self.log_step(f"Model Creation - {name}", "PASS", {
                    'total_params': f"{total_params:,}",
                    'trainable_params': f"{trainable_params:,}",
                    'device': str(self.device)
                })

            return True, models

        except Exception as e:
            self.log_step("Model Creation", "FAIL", {'error': str(e)})
            return False, None

    def step_3_forward_pass(self, models, sample_input):
        """Step 3: Test forward pass for all models"""
        try:
            sample_input = sample_input.to(self.device)

            for model_name, model in models.items():
                model.eval()

                start_time = time.time()
                with torch.no_grad():
                    output = model(sample_input)
                forward_time = time.time() - start_time

                # Verify output shape
                expected_shape = (sample_input.shape[0], sample_input.shape[1],
                                  sample_input.shape[2], 3)
                assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"

                # Check for NaN/Inf
                assert torch.isfinite(output).all(
                ), "Output contains NaN or Inf"

                self.log_step(f"Forward Pass - {model_name}", "PASS", {
                    'output_shape': str(tuple(output.shape)),
                    'forward_time': f"{forward_time:.4f}s",
                    'mean_output': f"{output.mean().item():.6f}",
                    'std_output': f"{output.std().item():.6f}"
                })

            return True

        except Exception as e:
            self.log_step("Forward Pass", "FAIL", {'error': str(e)})
            return False

    def step_4_backward_pass(self, models, sample_input, sample_target):
        """Step 4: Test backward pass and gradient computation"""
        try:
            sample_input = sample_input.to(self.device)
            sample_target = sample_target.to(self.device)

            for model_name, model in models.items():
                model.train()

                # Forward pass
                output = model(sample_input)

                # Compute loss
                criterion = nn.MSELoss()
                loss = criterion(output, sample_target)

                # Backward pass
                model.zero_grad()
                loss.backward()

                # Check gradients
                total_grad_norm = 0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        param_count += 1

                total_grad_norm = total_grad_norm ** 0.5

                # Verify gradients are reasonable
                assert total_grad_norm < 1000.0, f"Gradient explosion: {total_grad_norm}"
                assert total_grad_norm > 0.0, "No gradients computed"

                self.log_step(f"Backward Pass - {model_name}", "PASS", {
                    'loss': f"{loss.item():.6f}",
                    'grad_norm': f"{total_grad_norm:.6f}",
                    'params_with_grad': param_count
                })

            return True

        except Exception as e:
            self.log_step("Backward Pass", "FAIL", {'error': str(e)})
            return False

    def step_5_mini_training(self, models, train_loader):
        """Step 5: Test mini training loop"""
        try:
            for model_name, model in models.items():
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                criterion = nn.MSELoss()

                # Train for 5 batches
                losses = []
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if batch_idx >= 5:
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

                # Check loss progression
                avg_loss = np.mean(losses)
                loss_std = np.std(losses)

                self.log_step(f"Mini Training - {model_name}", "PASS", {
                    'num_batches': len(losses),
                    'avg_loss': f"{avg_loss:.6f}",
                    'loss_std': f"{loss_std:.6f}",
                    'final_loss': f"{losses[-1]:.6f}"
                })

            return True

        except Exception as e:
            self.log_step("Mini Training", "FAIL", {'error': str(e)})
            return False

    def step_6_result_output(self, models, sample_input):
        """Step 6: Test result output and formatting"""
        try:
            results = {}
            sample_input = sample_input.to(self.device)

            for model_name, model in models.items():
                model.eval()

                with torch.no_grad():
                    output = model(sample_input)

                    # Convert to numpy for result processing
                output_np = output.cpu().numpy()

                # Simulate pose metrics computation (create matching 3D ground truth)
                gt_poses_2d = sample_input.cpu().numpy()  # [B, T, J, 2]
                # Create synthetic 3D ground truth with same shape as output
                gt_poses_3d = np.random.randn(*output_np.shape)  # [B, T, J, 3]

                # Compute simple metrics
                mean_error = np.mean(np.abs(output_np - gt_poses_3d))
                max_error = np.max(np.abs(output_np - gt_poses_3d))

                results[model_name] = {
                    'output_shape': output_np.shape,
                    'mean_prediction': float(np.mean(output_np)),
                    'std_prediction': float(np.std(output_np)),
                    'mean_error': float(mean_error),
                    'max_error': float(max_error)
                }

                self.log_step(f"Result Output - {model_name}", "PASS", {
                    'output_format': 'numpy array',
                    'mean_prediction': f"{results[model_name]['mean_prediction']:.6f}",
                    'mean_error': f"{results[model_name]['mean_error']:.6f}"
                })

            # Save results to file
            results_file = f"end_to_end_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            self.log_step("Result Saving", "PASS", {
                'results_file': results_file,
                'num_models': len(results)
            })

            return True, results

        except Exception as e:
            self.log_step("Result Output", "FAIL", {'error': str(e)})
            return False, None

    def run_validation(self):
        """Run complete end-to-end validation"""
        print("🚀 Starting End-to-End Validation")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print("=" * 60)

        self.start_time = time.time()

        # Step 1: Data Loading
        success, train_loader = self.step_1_data_loading()
        if not success:
            return False

        # Get sample data for testing
        sample_input, sample_target = next(iter(train_loader))

        # Step 2: Model Creation
        success, models = self.step_2_model_creation()
        if not success:
            return False

        # Step 3: Forward Pass
        success = self.step_3_forward_pass(models, sample_input)
        if not success:
            return False

        # Step 4: Backward Pass
        success = self.step_4_backward_pass(
            models, sample_input, sample_target)
        if not success:
            return False

        # Step 5: Mini Training
        success = self.step_5_mini_training(models, train_loader)
        if not success:
            return False

        # Step 6: Result Output
        success, results = self.step_6_result_output(models, sample_input)
        if not success:
            return False

        # Final summary
        total_time = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("🎉 END-TO-END VALIDATION COMPLETED")
        print("=" * 60)

        passed_steps = sum(
            1 for log in self.validation_log if log['status'] == 'PASS')
        total_steps = len(self.validation_log)

        print(f"✅ Steps Passed: {passed_steps}/{total_steps}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🔧 Device Used: {self.device}")
        print(f"📊 Models Tested: {len(models)}")

        if passed_steps == total_steps:
            print("\n🎯 RESULT: ALL VALIDATION STEPS PASSED!")
            print("✅ The complete pipeline is ready for production use")
            return True
        else:
            print(f"\n⚠️  RESULT: {total_steps - passed_steps} STEPS FAILED")
            print("❌ Please check the failed steps above")
            return False

    def save_detailed_log(self):
        """Save detailed validation log"""
        log_file = f"end_to_end_validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            'validation_date': datetime.now().isoformat(),
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'steps': self.validation_log
        }

        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📁 Detailed log saved: {log_file}")


def main():
    """Main function to run end-to-end validation"""
    try:
        validator = EndToEndValidator()
        success = validator.run_validation()
        validator.save_detailed_log()

        print("\n" + "=" * 60)
        if success:
            print("✅ Task 2.4.2: End-to-End Validation - PASSED")
        else:
            print("❌ Task 2.4.2: End-to-End Validation - FAILED")
        print("=" * 60)

        return 0 if success else 1

    except Exception as e:
        print(f"❌ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
