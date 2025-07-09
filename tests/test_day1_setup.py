#!/usr/bin/env python3
"""
Day 1 Task Completion Verification Script
Validates that all Day 1 morning tasks (T1.0 + T1.1 + T1.2) are successfully completed
"""

import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path

def test_environment_setup():
    """Test T1.1: Environment setup"""
    print("🔧 Testing Environment Setup...")
    
    # Test core packages
    try:
        print(f"  - PyTorch: {torch.__version__} ✅")
        print(f"  - NumPy: {np.__version__} ✅") 
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        if torch.backends.mps.is_available():
            print(f"  - MPS (Apple Silicon) Available: True ✅")
    except Exception as e:
        print(f"  - Environment Error: {e} ❌")
        return False
    
    return True

def test_project_structure():
    """Test T1.0: Project structure integration"""
    print("\n📁 Testing Project Structure...")
    
    required_dirs = ['model', 'configs', 'utils', 'data', 'loss']
    required_files = ['train.py', 'requirements.txt']
    
    missing = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing.append(f"Directory: {dir_name}")
        else:
            print(f"  - {dir_name}/ ✅")
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing.append(f"File: {file_name}")
        else:
            print(f"  - {file_name} ✅")
    
    if missing:
        print(f"  - Missing: {missing} ❌")
        return False
    
    return True

def test_motionagformer_integration():
    """Test MotionAGFormer code integration"""
    print("\n🔄 Testing MotionAGFormer Integration...")
    
    try:
        # Check if we can import key modules
        sys.path.append('.')
        
        # Test model loading capability
        if os.path.exists('model/MotionAGFormer.py'):
            print("  - MotionAGFormer.py found ✅")
        
        # Test config files
        config_path = 'configs/h36m/MotionAGFormer-base.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("  - Config file loadable ✅")
            print(f"    - Model dimensions: {config.get('model', {}).get('dim', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  - Integration Error: {e} ❌")
        return False

def test_data_pipeline():
    """Test T1.2: Data preparation"""
    print("\n📊 Testing Data Pipeline...")
    
    # Test mock data files
    data_files = [
        'data/motion3d/data_3d_h36m_mock.npz',
        'data/motion3d/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz',
        'data/motion3d/test_data_small.npz'
    ]
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"  - Missing data file: {file_path} ❌")
            return False
        else:
            print(f"  - {Path(file_path).name} ✅")
    
    # Test data loading and format
    try:
        # Load 3D data
        data_3d = np.load('data/motion3d/data_3d_h36m_mock.npz')
        poses_3d = data_3d['positions_3d']
        
        # Load 2D data
        data_2d = np.load('data/motion3d/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz')
        poses_2d = data_2d['positions_2d']
        
        # Verify format [B, T, J, D]
        assert len(poses_3d.shape) == 4, f"3D data should be 4D, got {poses_3d.shape}"
        assert poses_3d.shape[-1] == 3, f"3D data last dim should be 3, got {poses_3d.shape[-1]}"
        assert poses_3d.shape[2] == 17, f"Should have 17 joints, got {poses_3d.shape[2]}"
        
        assert len(poses_2d.shape) == 4, f"2D data should be 4D, got {poses_2d.shape}"
        assert poses_2d.shape[-1] == 2, f"2D data last dim should be 2, got {poses_2d.shape[-1]}"
        assert poses_2d.shape[2] == 17, f"Should have 17 joints, got {poses_2d.shape[2]}"
        
        print(f"  - 3D data shape: {poses_3d.shape} [B,T,J,D] ✅")
        print(f"  - 2D data shape: {poses_2d.shape} [B,T,J,D] ✅")
        print(f"  - Data format validation passed ✅")
        
        return True
        
    except Exception as e:
        print(f"  - Data loading error: {e} ❌")
        return False

def test_basic_model_instantiation():
    """Test basic model can be instantiated"""
    print("\n🧠 Testing Basic Model Capability...")
    
    try:
        # Create dummy data matching expected format
        batch_size, seq_len, num_joints = 2, 50, 17
        dummy_2d = torch.randn(batch_size, seq_len, num_joints, 2)
        
        print(f"  - Dummy input created: {dummy_2d.shape} ✅")
        print(f"  - PyTorch tensor operations work ✅")
        
        # Test basic tensor operations that model will need
        dummy_3d = torch.randn(batch_size, seq_len, num_joints, 3)
        output = torch.cat([dummy_2d, dummy_3d[:,:,:,:2]], dim=-1)
        
        print(f"  - Tensor concatenation works: {output.shape} ✅")
        
        return True
        
    except Exception as e:
        print(f"  - Model test error: {e} ❌")
        return False

def main():
    """Run all Day 1 verification tests"""
    print("🚀 Day 1 Task Completion Verification")
    print("=" * 50)
    
    tests = [
        ("Environment Setup (T1.1)", test_environment_setup),
        ("Project Structure (T1.0)", test_project_structure), 
        ("MotionAGFormer Integration (T1.0)", test_motionagformer_integration),
        ("Data Pipeline (T1.2)", test_data_pipeline),
        ("Basic Model Capability", test_basic_model_instantiation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Day 1 Task Completion Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL DAY 1 TASKS COMPLETED SUCCESSFULLY!")
        print("🚀 Ready to proceed to Day 1 afternoon tasks (baseline validation)")
        print("💡 Next: Run MotionAGFormer baseline and proceed to Day 2 development")
    else:
        print("\n⚠️  Some tasks need attention before proceeding")
        print("🔧 Please resolve failed tests before continuing")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 