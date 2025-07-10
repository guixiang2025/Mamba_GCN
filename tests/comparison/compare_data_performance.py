#!/usr/bin/env python3
"""
Mock vs Real Data Performance Comparison
比较模拟数据和真实Human3.6M数据的模型性能
"""

import torch
import numpy as np
from datetime import datetime
import os

from data.reader.mock_h36m import DataReaderMockH36M
from data.reader.real_h36m import DataReaderRealH36M
from model.MotionAGFormer import MotionAGFormer
from loss.pose3d import loss_mpjpe
from utils.tools import set_random_seed


def quick_test(datareader, data_type, model_type='baseline'):
    """Quick performance test"""
    print(f"\n🧪 测试 {data_type} 数据 ({model_type})...")
    
    # Get small sample
    train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
    
    # Take small sample for quick test
    sample_size = min(100, len(test_data))
    test_data_sample = test_data[:sample_size]
    test_labels_sample = test_labels[:sample_size]
    
    print(f"   测试样本: {test_data_sample.shape} -> {test_labels_sample.shape}")
    
    # Create model
    if model_type == 'baseline':
        model = MotionAGFormer(
            n_layers=4, dim_in=2, dim_feat=64, dim_out=3, n_frames=243,
            use_mamba_gcn=False
        )
    else:  # mamba_gcn
        model = MotionAGFormer(
            n_layers=4, dim_in=2, dim_feat=64, dim_out=3, n_frames=243,
            use_mamba_gcn=True, mamba_gcn_use_mamba=True, mamba_gcn_use_attention=False
        )
    
    model.eval()
    
    # Convert to tensors
    input_tensor = torch.FloatTensor(test_data_sample)
    target_tensor = torch.FloatTensor(test_labels_sample)
    
    # Forward pass
    with torch.no_grad():
        pred_tensor = model(input_tensor)
    
    # Calculate loss
    loss_value = loss_mpjpe(pred_tensor, target_tensor).item()
    
    # Calculate MPJPE in original scale (approximate)
    pred_np = pred_tensor.numpy()
    target_np = target_tensor.numpy()
    mpjpe = np.mean(np.sqrt(np.sum((pred_np - target_np) ** 2, axis=-1))) * 1000  # Convert to mm
    
    print(f"   损失: {loss_value:.4f}")
    print(f"   MPJPE: {mpjpe:.2f}mm")
    
    return {
        'data_type': data_type,
        'model_type': model_type,
        'loss': loss_value,
        'mpjpe': mpjpe,
        'sample_size': sample_size
    }


def main():
    print("📋 Mock vs Real Data Performance Comparison")
    print("=" * 60)
    
    set_random_seed(42)
    results = []
    
    # Test with Mock Data
    try:
        mock_reader = DataReaderMockH36M(
            n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
            read_confidence=True, dt_root='data/motion3d'
        )
        
        # Test baseline and mamba_gcn with mock data
        results.append(quick_test(mock_reader, 'Mock', 'baseline'))
        results.append(quick_test(mock_reader, 'Mock', 'mamba_gcn'))
        
    except Exception as e:
        print(f"❌ Mock数据测试失败: {e}")
    
    # Test with Real Data
    try:
        real_reader = DataReaderRealH36M(
            n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
            read_confidence=True, dt_root='data/motion3d/human36m/raw/motion3d'
        )
        
        # Test baseline and mamba_gcn with real data
        results.append(quick_test(real_reader, 'Real', 'baseline'))
        results.append(quick_test(real_reader, 'Real', 'mamba_gcn'))
        
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 性能比较总结")
    print("=" * 60)
    
    for result in results:
        print(f"{result['data_type']} + {result['model_type']:<10}: "
              f"MPJPE = {result['mpjpe']:.2f}mm, Loss = {result['loss']:.4f}")
    
    # Find best configuration
    if results:
        best_result = min(results, key=lambda x: x['mpjpe'])
        print(f"\n🏆 最佳配置: {best_result['data_type']} + {best_result['model_type']} "
              f"(MPJPE: {best_result['mpjpe']:.2f}mm)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data_comparison_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 详细结果保存到: {results_file}")


if __name__ == '__main__':
    main()
