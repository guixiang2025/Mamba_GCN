#!/usr/bin/env python3
"""
Mock data generator for Mamba-GCN development
Creates synthetic Human3.6M-like data for testing and development
"""

import numpy as np
import os

def create_mock_h36m_data():
    """Create mock Human3.6M data for development"""
    print("🔧 Creating mock Human3.6M data for development...")
    
    # Parameters matching Human3.6M dataset
    n_sequences = 100  # Small dataset for quick testing
    seq_length = 250   # Sequence length
    n_joints = 17      # Human3.6M joint count
    
    # Create synthetic 3D poses (in mm, Human3.6M scale)
    poses_3d = np.random.randn(n_sequences, seq_length, n_joints, 3) * 100
    
    # Create synthetic 2D poses (in pixels, typical image scale)
    poses_2d = np.random.randn(n_sequences, seq_length, n_joints, 2) * 300 + 500
    
    # Add some realistic structure to the data
    # Make sure poses are somewhat human-like (basic skeletal constraints)
    for i in range(n_sequences):
        # Add some temporal smoothness
        for j in range(n_joints):
            for k in range(3):  # x, y, z
                poses_3d[i, :, j, k] = np.cumsum(np.random.randn(seq_length) * 5)
            for k in range(2):  # x, y
                poses_2d[i, :, j, k] = np.cumsum(np.random.randn(seq_length) * 3) + 500
    
    # Create metadata
    metadata = {
        'n_sequences': n_sequences,
        'seq_length': seq_length,
        'n_joints': n_joints,
        'joint_names': [
            'Hip', 'RHip', 'RKnee', 'RFoot',
            'LHip', 'LKnee', 'LFoot',
            'Spine', 'Thorax', 'Neck', 'Head',
            'LShoulder', 'LElbow', 'LWrist',
            'RShoulder', 'RElbow', 'RWrist'
        ],
        'data_type': 'mock_synthetic',
        'scale': 'mm_for_3d_pixels_for_2d'
    }
    
    # Save the mock data
    os.makedirs('data/motion3d', exist_ok=True)
    
    # Save in format similar to MotionBERT preprocessing
    np.savez_compressed('data/motion3d/data_3d_h36m_mock.npz',
                       positions_3d=poses_3d,
                       metadata=metadata)
    
    np.savez_compressed('data/motion3d/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz',
                       positions_2d=poses_2d,
                       metadata=metadata)
    
    # Create a small test dataset (10 samples)
    test_indices = np.random.choice(n_sequences, 10, replace=False)
    
    np.savez_compressed('data/motion3d/test_data_small.npz',
                       positions_3d=poses_3d[test_indices],
                       positions_2d=poses_2d[test_indices],
                       metadata=dict(metadata, n_sequences=10))
    
    print(f"✅ Mock data created successfully!")
    print(f"   - 3D poses: {poses_3d.shape} (sequences, frames, joints, xyz)")
    print(f"   - 2D poses: {poses_2d.shape} (sequences, frames, joints, xy)")
    print(f"   - Test data: 10 sequences for quick validation")
    print(f"   - Files saved in: data/motion3d/")
    
    return poses_3d, poses_2d, metadata

def verify_data_format():
    """Verify that mock data matches expected format [B,T,J,D]"""
    print("\n🔍 Verifying data format...")
    
    # Load and check 3D data
    data_3d = np.load('data/motion3d/data_3d_h36m_mock.npz')
    poses_3d = data_3d['positions_3d']
    
    print(f"   3D data shape: {poses_3d.shape}")
    print(f"   Expected format: [Batch, Time, Joints, Dims] = [B, T, J, 3]")
    
    # Load and check 2D data  
    data_2d = np.load('data/motion3d/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz')
    poses_2d = data_2d['positions_2d']
    
    print(f"   2D data shape: {poses_2d.shape}")
    print(f"   Expected format: [Batch, Time, Joints, Dims] = [B, T, J, 2]")
    
    # Verify dimensions match expected [B,T,J,D] format
    assert len(poses_3d.shape) == 4, f"3D data should be 4D, got {poses_3d.shape}"
    assert poses_3d.shape[-1] == 3, f"3D data last dim should be 3, got {poses_3d.shape[-1]}"
    assert poses_3d.shape[2] == 17, f"Should have 17 joints, got {poses_3d.shape[2]}"
    
    assert len(poses_2d.shape) == 4, f"2D data should be 4D, got {poses_2d.shape}"
    assert poses_2d.shape[-1] == 2, f"2D data last dim should be 2, got {poses_2d.shape[-1]}"
    assert poses_2d.shape[2] == 17, f"Should have 17 joints, got {poses_2d.shape[2]}"
    
    print("✅ Data format verification passed!")
    return True

if __name__ == "__main__":
    print("🚀 Mock Data Generator for Mamba-GCN")
    print("=" * 50)
    
    # Create mock data
    poses_3d, poses_2d, metadata = create_mock_h36m_data()
    
    # Verify format
    verify_data_format()
    
    print("\n🎉 Mock data setup complete! Ready for development.")
    print("💡 Note: Replace with real Human3.6M data for actual training.") 