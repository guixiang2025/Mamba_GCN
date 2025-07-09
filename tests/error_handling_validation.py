#!/usr/bin/env python3
"""
Error Handling Validation for MotionAGFormer + MambaGCN
Tests error resilience and graceful failure handling

This script validates that the system handles various error conditions gracefully
without crashing or producing undefined behavior.
"""

from model.MotionAGFormer import MotionAGFormer
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import warnings
from typing import Tuple, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ErrorHandlingValidator:
    """Validates error handling and edge case resilience"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_results = []

    def log_test(self, test_name: str, expected_error: str, actual_result: str, passed: bool):
        """Log test result"""
        self.test_results.append({
            'test': test_name,
            'expected': expected_error,
            'actual': actual_result,
            'passed': passed
        })

        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
        if not passed:
            print(f"    Expected: {expected_error}")
            print(f"    Actual: {actual_result}")

    def test_invalid_input_shapes(self):
        """Test handling of invalid input tensor shapes"""
        print("\n🔍 Testing Invalid Input Shapes...")

        model = MotionAGFormer(
            n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
        ).to(self.device)

        invalid_inputs = [
            # Wrong number of dimensions
            (torch.randn(16, 27, 17), "3D tensor instead of 4D"),
            (torch.randn(16, 27, 17, 2, 5), "5D tensor instead of 4D"),

            # Wrong feature dimension
            (torch.randn(16, 27, 17, 5), "Wrong feature dimension (5 instead of 2)"),

            # Wrong joint count
            (torch.randn(16, 27, 20, 2), "Wrong joint count (20 instead of 17)"),

            # Zero dimensions
            (torch.randn(0, 27, 17, 2), "Batch size 0"),
            (torch.randn(16, 0, 17, 2), "Time dimension 0"),
        ]

        for invalid_input, description in invalid_inputs:
            try:
                invalid_input = invalid_input.to(self.device)
                output = model(invalid_input)
                self.log_test(
                    f"Invalid shape: {description}", "RuntimeError", "No error raised", False)
            except (RuntimeError, ValueError, IndexError) as e:
                self.log_test(
                    f"Invalid shape: {description}", "Appropriate error", f"Caught: {type(e).__name__}", True)
            except Exception as e:
                self.log_test(
                    f"Invalid shape: {description}", "Appropriate error", f"Unexpected: {type(e).__name__}", False)

    def test_memory_edge_cases(self):
        """Test handling of extreme memory conditions"""
        print("\n🔍 Testing Memory Edge Cases...")

        # Test with very large batch size (should handle gracefully or fail cleanly)
        try:
            model = MotionAGFormer(
                n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
            ).to(self.device)

            large_input = torch.randn(1000, 27, 17, 2).to(self.device)
            output = model(large_input)
            self.log_test("Large batch size (1000)",
                          "Success or clean OOM", "Success", True)

        except torch.cuda.OutOfMemoryError:
            self.log_test("Large batch size (1000)",
                          "Success or clean OOM", "Clean CUDA OOM", True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log_test("Large batch size (1000)",
                              "Success or clean OOM", "Clean OOM", True)
            else:
                self.log_test("Large batch size (1000)",
                              "Success or clean OOM", f"Runtime error: {e}", False)
        except Exception as e:
            self.log_test("Large batch size (1000)", "Success or clean OOM",
                          f"Unexpected: {type(e).__name__}", False)

    def test_model_configuration_errors(self):
        """Test handling of invalid model configurations"""
        print("\n🔍 Testing Model Configuration Errors...")

        invalid_configs = [
            # Invalid layer count
            {"n_layers": 0, "description": "Zero layers"},
            {"n_layers": -1, "description": "Negative layers"},

            # Invalid dimensions
            {"dim_feat": 0, "description": "Zero feature dimension"},
            {"dim_feat": -10, "description": "Negative feature dimension"},

            # Invalid frame count
            {"n_frames": 0, "description": "Zero frames"},
            {"n_frames": -5, "description": "Negative frames"},
        ]

        for config in invalid_configs:
            desc = config.pop("description")
            try:
                base_config = {
                    "n_layers": 2, "dim_in": 2, "dim_feat": 32,
                    "dim_out": 3, "n_frames": 27
                }
                base_config.update(config)

                model = MotionAGFormer(**base_config)
                self.log_test(
                    f"Invalid config: {desc}", "Configuration error", "No error raised", False)

            except (ValueError, RuntimeError, TypeError) as e:
                self.log_test(
                    f"Invalid config: {desc}", "Configuration error", f"Caught: {type(e).__name__}", True)
            except Exception as e:
                self.log_test(
                    f"Invalid config: {desc}", "Configuration error", f"Unexpected: {type(e).__name__}", False)

    def test_gradient_edge_cases(self):
        """Test handling of gradient computation edge cases"""
        print("\n🔍 Testing Gradient Edge Cases...")

        model = MotionAGFormer(
            n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
        ).to(self.device)

        # Test with very small loss (potential underflow)
        try:
            x = torch.randn(4, 27, 17, 2).to(self.device)
            target = torch.randn(4, 27, 17, 3).to(self.device)

            output = model(x)
            loss = nn.MSELoss()(output, target) * 1e-10  # Very small loss

            model.zero_grad()
            loss.backward()

            # Check if gradients exist and are finite
            grad_exists = any(p.grad is not None for p in model.parameters())
            grad_finite = all(torch.isfinite(p.grad).all()
                              for p in model.parameters() if p.grad is not None)

            if grad_exists and grad_finite:
                self.log_test("Very small loss gradients",
                              "Finite gradients or zero", "Success", True)
            else:
                self.log_test("Very small loss gradients",
                              "Finite gradients or zero", "Some gradients invalid", False)

        except Exception as e:
            self.log_test("Very small loss gradients",
                          "Finite gradients or zero", f"Error: {type(e).__name__}", False)

        # Test gradient clipping resilience
        try:
            x = torch.randn(4, 27, 17, 2).to(self.device)
            target = torch.randn(4, 27, 17, 3).to(
                self.device) * 1000  # Large target

            output = model(x)
            loss = nn.MSELoss()(output, target)  # Potentially large loss

            model.zero_grad()
            loss.backward()

            # Apply gradient clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)

            # Check if clipping worked
            total_norm_after = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after ** 0.5

            if total_norm_after <= 1.1:  # Allow small numerical errors
                self.log_test("Gradient clipping", "Norm <= 1.0",
                              f"Norm: {total_norm_after:.4f}", True)
            else:
                self.log_test("Gradient clipping", "Norm <= 1.0",
                              f"Norm: {total_norm_after:.4f}", False)

        except Exception as e:
            self.log_test("Gradient clipping", "Norm <= 1.0",
                          f"Error: {type(e).__name__}", False)

    def test_device_handling(self):
        """Test handling of device mismatches and transfers"""
        print("\n🔍 Testing Device Handling...")

        if torch.cuda.is_available():
            # Test device mismatch
            try:
                model_cpu = MotionAGFormer(
                    n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
                )  # On CPU

                input_cuda = torch.randn(4, 27, 17, 2).cuda()  # On CUDA

                output = model_cpu(input_cuda)
                self.log_test("Device mismatch (CPU model, CUDA input)",
                              "RuntimeError", "No error raised", False)

            except RuntimeError as e:
                if "device" in str(e).lower() or "cuda" in str(e).lower():
                    self.log_test("Device mismatch (CPU model, CUDA input)",
                                  "Device error", "Appropriate device error", True)
                else:
                    self.log_test("Device mismatch (CPU model, CUDA input)",
                                  "Device error", f"Other RuntimeError: {e}", False)
            except Exception as e:
                self.log_test("Device mismatch (CPU model, CUDA input)",
                              "Device error", f"Unexpected: {type(e).__name__}", False)
        else:
            self.log_test("Device mismatch test",
                          "CUDA not available", "Skipped (no CUDA)", True)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        print("\n🔍 Testing Numerical Stability...")

        model = MotionAGFormer(
            n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
        ).to(self.device)

        extreme_inputs = [
            (torch.ones(4, 27, 17, 2) * 1000, "Very large values (1000)"),
            (torch.ones(4, 27, 17, 2) * 1e-6, "Very small values (1e-6)"),
            (torch.randn(4, 27, 17, 2) * 0, "All zeros"),
        ]

        for extreme_input, description in extreme_inputs:
            try:
                extreme_input = extreme_input.to(self.device)
                output = model(extreme_input)

                # Check if output is finite
                if torch.isfinite(output).all():
                    self.log_test(
                        f"Numerical stability: {description}", "Finite output", "All finite", True)
                else:
                    self.log_test(
                        f"Numerical stability: {description}", "Finite output", "Contains NaN/Inf", False)

            except Exception as e:
                self.log_test(
                    f"Numerical stability: {description}", "Finite output", f"Error: {type(e).__name__}", False)

    def test_concurrent_usage(self):
        """Test handling of concurrent model usage"""
        print("\n🔍 Testing Concurrent Usage...")

        try:
            model = MotionAGFormer(
                n_layers=2, dim_in=2, dim_feat=32, dim_out=3, n_frames=27
            ).to(self.device)

            # Simulate concurrent forward passes
            inputs = [torch.randn(2, 27, 17, 2).to(self.device)
                      for _ in range(3)]

            model.eval()
            outputs = []
            for inp in inputs:
                with torch.no_grad():
                    out = model(inp)
                    outputs.append(out)

            # Check all outputs are valid
            all_valid = all(torch.isfinite(out).all() for out in outputs)

            if all_valid:
                self.log_test("Concurrent forward passes",
                              "All valid outputs", "Success", True)
            else:
                self.log_test("Concurrent forward passes",
                              "All valid outputs", "Some invalid outputs", False)

        except Exception as e:
            self.log_test("Concurrent forward passes",
                          "All valid outputs", f"Error: {type(e).__name__}", False)

    def run_all_tests(self):
        """Run all error handling tests"""
        print("🛡️ Starting Error Handling Validation")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print("=" * 60)

        # Suppress warnings during testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.test_invalid_input_shapes()
            self.test_memory_edge_cases()
            self.test_model_configuration_errors()
            self.test_gradient_edge_cases()
            self.test_device_handling()
            self.test_numerical_stability()
            self.test_concurrent_usage()

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results if result['passed'])

        print("\n" + "=" * 60)
        print("🛡️ ERROR HANDLING VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"📊 Success Rate: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("\n🎯 RESULT: ALL ERROR HANDLING TESTS PASSED!")
            print("✅ System demonstrates robust error handling")
            return True
        else:
            print(f"\n⚠️  RESULT: {total_tests - passed_tests} TESTS FAILED")
            print("❌ Some error handling needs improvement")

            # Show failed tests
            print("\n📋 Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  ❌ {result['test']}: {result['actual']}")

            return False


def main():
    """Main function to run error handling validation"""
    try:
        validator = ErrorHandlingValidator()
        success = validator.run_all_tests()

        print("\n" + "=" * 60)
        if success:
            print("✅ Task 2.4.3: Error Handling Validation - PASSED")
        else:
            print("❌ Task 2.4.3: Error Handling Validation - NEEDS IMPROVEMENT")
        print("=" * 60)

        return 0 if success else 1

    except Exception as e:
        print(f"❌ Error handling validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
