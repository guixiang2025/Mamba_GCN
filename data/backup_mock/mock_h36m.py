#!/usr/bin/env python3
"""
Mock H36M Data Reader for Mamba-GCN Development
Adapts synthetic mock data to MotionAGFormer's expected format
"""

import numpy as np
import random
from utils.data import split_clips

random.seed(0)


class DataReaderMockH36M(object):
    """Mock data reader that mimics DataReaderH36M but uses synthetic data"""
    
    def __init__(self, n_frames, sample_stride=1, data_stride_train=81, data_stride_test=243, 
                 read_confidence=True, dt_root='data/motion3d', dt_file='mock'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
        # Load mock data
        self._load_mock_data(dt_root)
        
    def _load_mock_data(self, dt_root):
        """Load mock data and create dataset structure"""
        print("🔄 Loading mock data for development...")
        
        # Load 3D and 2D mock data
        data_3d = np.load(f'{dt_root}/data_3d_h36m_mock.npz')
        data_2d = np.load(f'{dt_root}/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz')
        
        poses_3d = data_3d['positions_3d']  # [100, 250, 17, 3]
        poses_2d = data_2d['positions_2d']  # [100, 250, 17, 2]
        
        n_sequences, seq_len, n_joints, _ = poses_3d.shape
        
        # Flatten to frame-level data (similar to H36M format)
        # [100, 250, 17, 3] -> [25000, 17, 3]
        poses_3d_flat = poses_3d.reshape(-1, n_joints, 3)
        poses_2d_flat = poses_2d.reshape(-1, n_joints, 2)
        
        # Add confidence scores (mock as 1.0)
        confidence = np.ones((poses_2d_flat.shape[0], n_joints, 1))
        poses_2d_with_conf = np.concatenate([poses_2d_flat, confidence], axis=-1)
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * poses_3d_flat.shape[0])
        
        # Create mock dataset structure matching H36M
        self.dt_dataset = {
            'train': {
                'joint_2d': poses_2d_with_conf[:split_idx],  # [20000, 17, 3]
                'joint3d_image': poses_3d_flat[:split_idx],  # [20000, 17, 3]
                'confidence': confidence[:split_idx, :, 0],   # [20000, 17]
                'camera_name': ['54138969'] * split_idx,      # Mock camera
                'source': [f'mock_seq_{i//seq_len:03d}' for i in range(split_idx)],
                'action': ['Walking'] * split_idx,            # Mock action
                '2.5d_factor': np.ones((split_idx, 1)),      # Mock factor
                'joints_2.5d_image': poses_3d_flat[:split_idx]  # Same as 3d for simplicity
            },
            'test': {
                'joint_2d': poses_2d_with_conf[split_idx:],   # [5000, 17, 3]
                'joint3d_image': poses_3d_flat[split_idx:],   # [5000, 17, 3]
                'confidence': confidence[split_idx:, :, 0],   # [5000, 17]
                'camera_name': ['54138969'] * (poses_3d_flat.shape[0] - split_idx),
                'source': [f'mock_seq_{i//seq_len:03d}' for i in range(split_idx, poses_3d_flat.shape[0])],
                'action': ['Walking'] * (poses_3d_flat.shape[0] - split_idx),
                '2.5d_factor': np.ones((poses_3d_flat.shape[0] - split_idx, 1)),
                'joints_2.5d_image': poses_3d_flat[split_idx:]
            }
        }
        
        print(f"✅ Mock dataset loaded:")
        print(f"   - Train: {len(self.dt_dataset['train']['joint_2d'])} frames")
        print(f"   - Test: {len(self.dt_dataset['test']['joint_2d'])} frames")

    def read_2d(self):
        """Read 2D pose data with confidence"""
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride].astype(np.float32)
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride].astype(np.float32)
        
        # Normalize to [-1, 1] range (mock normalization)
        # For simplicity, we assume data is already in reasonable range
        trainset[:, :, :2] = trainset[:, :, :2] / 1000.0  # Scale down from pixel range
        testset[:, :, :2] = testset[:, :, :2] / 1000.0
        
        return trainset, testset

    def read_3d(self):
        """Read 3D pose data"""
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride].astype(np.float32)
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride].astype(np.float32)
        
        # Normalize 3D coordinates (mock normalization)
        train_labels = train_labels / 1000.0  # Scale from mm to meters-like range
        test_labels = test_labels / 1000.0
        
        return train_labels, test_labels

    def read_hw(self):
        """Mock camera resolution"""
        if self.test_hw is not None:
            return self.test_hw
        
        n_test_frames = len(self.dt_dataset['test']['camera_name'])
        self.test_hw = np.ones((n_test_frames, 2)) * 1000  # Mock 1000x1000 resolution
        return self.test_hw

    def get_split_id(self):
        """Generate frame splits for sequence windows"""
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
            
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]
        
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        
        return self.split_id_train, self.split_id_test

    def turn_into_test_clips(self, data):
        """Convert frame-level data to clips"""
        split_id_train, split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw(self):
        """Get camera resolution for test set"""
        test_hw = self.read_hw()
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]
        return test_hw

    def get_sliced_data(self):
        """Get windowed training and test data"""
        train_data, test_data = self.read_2d()
        train_labels, test_labels = self.read_3d()
        split_id_train, split_id_test = self.get_split_id()
        
        train_data = train_data[split_id_train]
        test_data = test_data[split_id_test] 
        train_labels = train_labels[split_id_train]
        test_labels = test_labels[split_id_test]
        
        return train_data, test_data, train_labels, test_labels

    def denormalize(self, test_data, all_sequence=False):
        """Denormalize test data for evaluation"""
        if all_sequence:
            test_data = self.turn_into_test_clips(test_data)

        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        
        # Denormalize back to original scale
        data = data * 1000.0  # Scale back to mm
        
        return data


def create_mock_datareader(args):
    """Factory function to create mock data reader"""
    return DataReaderMockH36M(
        n_frames=args.n_frames,
        sample_stride=1,
        data_stride_train=81,
        data_stride_test=243,
        read_confidence=True,
        dt_root=args.data_root
    ) 