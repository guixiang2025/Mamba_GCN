#!/usr/bin/env python3
"""
Real H36M Data Reader for Mamba-GCN 
Uses actual Human3.6M dataset from /raw directory
"""

import numpy as np
import random
import os
import pickle
from utils.data import read_pkl, split_clips

random.seed(0)


class DataReaderRealH36M(object):
    """Real H36M data reader using actual Human3.6M dataset"""

    def __init__(self, n_frames, sample_stride=1, data_stride_train=81, data_stride_test=243,
                 read_confidence=True, dt_root='data/motion3d/human36m/raw/motion3d',
                 dt_file='h36m_sh_conf_cam_source_final.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None

        # Load real H36M data
        self.dt_dataset = read_pkl(f'{dt_root}/{dt_file}')
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

        print(f"🔄 Loading real Human3.6M data from {dt_root}/{dt_file}")
        print(f"   Train frames: {len(self.dt_dataset['train']['joint_2d'])}")
        print(f"   Test frames: {len(self.dt_dataset['test']['joint_2d'])}")

    def read_2d(self):
        """Read 2D pose data with confidence"""
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(
            np.float32)
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(
            np.float32)

        # Normalize to [-1, 1] range based on camera resolution
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            res_w: float = 1000.0
            res_h: float = 1000.0
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000.0, 1002.0
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000.0, 1000.0
            else:
                assert 0, f'{idx} data item has an invalid camera name'
            trainset[idx, :, :] = trainset[idx, :, :] / \
                res_w * 2 - [1, res_h / res_w]

        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000.0, 1002.0
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000.0, 1000.0
            else:
                assert 0, f'{idx} data item has an invalid camera name'
            testset[idx, :, :] = testset[idx, :, :] / \
                res_w * 2 - [1, res_h / res_w]

        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(
                    np.float32)
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(
                    np.float32)
                if len(train_confidence.shape) == 2:
                    train_confidence = train_confidence[:, :, None]
                    test_confidence = test_confidence[:, :, None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:, :, 0:1]
                test_confidence = np.ones(testset.shape)[:, :, 0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)
            testset = np.concatenate((testset, test_confidence), axis=2)

        return trainset, testset

    def read_3d(self):
        """Read 3D pose data"""
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)

        # Normalize based on camera resolution
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, f'{idx} data item has an invalid camera name'
            train_labels[idx, :, :2] = train_labels[idx,
                                                    :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2

        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, f'{idx} data item has an invalid camera name'
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / \
                res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2

        return train_labels, test_labels

    def read_hw(self):
        """Read hardware/camera info"""
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, f'{idx} data item has an invalid camera name'
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def get_split_id(self):
        """Get video split IDs for temporal clips"""
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]
        self.split_id_train = split_clips(
            vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(
            vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test

    def turn_into_test_clips(self, data):
        """Converts (total_frames, ...) tensor to (n_clips, n_frames, ...) based on split_id_test"""
        split_id_train, split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw(self):
        """Get hardware info for test set"""
        test_hw = self.read_hw()
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]
        return test_hw

    def get_sliced_data(self):
        """Get sliced data for training and testing"""
        train_data, test_data = self.read_2d()
        train_labels, test_labels = self.read_3d()
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]
        return train_data, test_data, train_labels, test_labels

    def denormalize(self, test_data, all_sequence=False):
        """Denormalize predicted data"""
        if all_sequence:
            test_data = self.turn_into_test_clips(test_data)

        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(
            test_hw), f"Data n_clips is {len(data)} while test_hw size is {len(test_hw)}"

        # Denormalize (x,y,z) coordinates for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (
                data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data


def create_real_datareader(args):
    """Factory function to create real H36M data reader"""
    return DataReaderRealH36M(
        n_frames=args.n_frames,
        sample_stride=1,
        data_stride_train=81,
        data_stride_test=243,
        read_confidence=True,
        dt_root='data/motion3d/human36m/raw/motion3d'
    )
