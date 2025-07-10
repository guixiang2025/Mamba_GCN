#!/usr/bin/env python3
"""
Data Migration Script: Mock → Real Human3.6M Data
处理Task 2.5中识别的缺口 - 从模拟数据转换为真实Human3.6M数据
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path


def check_real_data_availability():
    """检查真实Human3.6M数据的可用性"""
    print("🔍 检查真实Human3.6M数据...")

    data_files = [
        'data/motion3d/human36m/raw/motion3d/h36m_sh_conf_cam_source_final.pkl',
        'data/motion3d/human36m/raw/motion3d/data_train_3dhp.npz',
        'data/motion3d/human36m/raw/motion3d/data_test_3dhp.npz',
        'data/motion3d/human36m/raw/motion3d/H36M-243/train',
        'data/motion3d/human36m/raw/motion3d/H36M-243/test'
    ]

    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"  ❌ 缺失: {file_path}")
        else:
            print(f"  ✅ 存在: {file_path}")

    if missing_files:
        print(f"\n⚠️  发现 {len(missing_files)} 个缺失文件")
        return False
    else:
        print("\n✅ 所有真实数据文件都可用")
        return True


def backup_mock_files():
    """备份当前的mock数据和相关文件"""
    print("\n💾 备份当前mock数据文件...")

    backup_dir = "data/backup_mock"
    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = [
        'data/reader/mock_h36m.py',
        'scripts/train_mock.py',
        'data/motion3d/data_3d_h36m_mock.npz',
        'data/motion3d/data_2d_h36m_cpn_ft_h36m_dbb_mock.npz',
        'data/motion3d/test_data_small.npz'
    ]

    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"  ✅ 备份: {file_path} → {backup_path}")

    print(f"📦 Mock数据备份完成: {backup_dir}")


def create_real_training_script():
    """创建使用真实数据的训练脚本"""
    print("\n🚂 创建真实数据训练脚本...")

    train_real_content = '''#!/usr/bin/env python3
"""
Real H36M Training Script for MotionAGFormer + MambaGCN
Uses actual Human3.6M dataset for authentic training and evaluation
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import from project modules
from data.reader.real_h36m import DataReaderRealH36M
from model.MotionAGFormer import MotionAGFormer
from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity
from utils.tools import set_random_seed, get_config
from utils.learning import AverageMeter


class RealH36MDataset(Dataset):
    """Dataset wrapper for real H36M data"""
    def __init__(self, data_2d, data_3d):
        self.data_2d = torch.FloatTensor(data_2d)
        self.data_3d = torch.FloatTensor(data_3d)
        
    def __len__(self):
        return len(self.data_2d)
    
    def __getitem__(self, idx):
        return self.data_2d[idx], self.data_3d[idx]


def parse_args():
    parser = argparse.ArgumentParser(description='MotionAGFormer + MambaGCN Real H36M Training')
    parser.add_argument('--config', type=str, default='configs/h36m/MotionAGFormer-base.yaml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--model_type', type=str, default='mamba_gcn', 
                       choices=['baseline', 'mamba_gcn', 'full'],
                       help='Model configuration type')
    parser.add_argument('--save_dir', type=str, default='checkpoints/real_h36m',
                       help='Directory to save models')
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def create_model(args, model_type='baseline'):
    """Create model based on type"""
    
    if model_type == 'baseline':
        # Original MotionAGFormer
        model = MotionAGFormer(
            n_layers=args.n_layers,
            dim_in=args.dim_in,
            dim_feat=args.dim_feat,
            dim_out=args.dim_out,
            n_frames=args.n_frames,
            use_mamba_gcn=False
        )
    elif model_type == 'mamba_gcn':
        # MambaGCN enhanced
        model = MotionAGFormer(
            n_layers=args.n_layers,
            dim_in=args.dim_in,
            dim_feat=args.dim_feat,
            dim_out=args.dim_out,
            n_frames=args.n_frames,
            use_mamba_gcn=True,
            mamba_gcn_use_mamba=True,
            mamba_gcn_use_attention=False
        )
    elif model_type == 'full':
        # Full architecture with all components
        model = MotionAGFormer(
            n_layers=args.n_layers,
            dim_in=args.dim_in,
            dim_feat=args.dim_feat,
            dim_out=args.dim_out,
            n_frames=args.n_frames,
            use_mamba_gcn=True,
            mamba_gcn_use_mamba=True,
            mamba_gcn_use_attention=True
        )
    
    return model


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data_2d, data_3d) in enumerate(pbar):
        data_2d, data_3d = data_2d.to(device), data_3d.to(device)
        
        optimizer.zero_grad()
        pred_3d = model(data_2d)
        
        # Compute loss
        loss = criterion(pred_3d, data_3d)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), data_2d.size(0))
        pbar.set_postfix({'Loss': losses.avg})
        
    return losses.avg


def evaluate(model, test_loader, device, datareader):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data_2d, data_3d in tqdm(test_loader, desc='Evaluating'):
            data_2d, data_3d = data_2d.to(device), data_3d.to(device)
            
            pred_3d = model(data_2d)
            
            # Accumulate for MPJPE calculation
            predictions.append(pred_3d.cpu().numpy())
            targets.append(data_3d.cpu().numpy())
            
            total_samples += data_2d.size(0)
    
    # Calculate MPJPE
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Denormalize predictions for proper MPJPE calculation
    predictions_denorm = datareader.denormalize(predictions, all_sequence=True)
    targets_denorm = datareader.denormalize(targets, all_sequence=True)
    
    # Calculate MPJPE in mm
    mpjpe = np.mean(np.sqrt(np.sum((predictions_denorm - targets_denorm) ** 2, axis=-1)))
    
    return mpjpe


def main():
    opts = parse_args()
    
    # Load config
    args = get_config(opts.config)
    
    # Override settings
    args.epochs = opts.epochs
    args.batch_size = opts.batch_size
    args.n_frames = 243
    
    # Set device
    device = get_device(opts.device)
    print(f"🔧 使用设备: {device}")
    
    # Set random seed
    set_random_seed(42)
    
    # Create real data reader
    print("📊 加载真实Human3.6M数据...")
    datareader = DataReaderRealH36M(
        n_frames=args.n_frames,
        sample_stride=1,
        data_stride_train=81,
        data_stride_test=243,
        read_confidence=True,
        dt_root='data/motion3d/human36m/raw/motion3d'
    )
    
    # Get data
    train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
    print(f"✅ 真实数据加载完成:")
    print(f"   - 训练: {train_data.shape} -> {train_labels.shape}")
    print(f"   - 测试: {test_data.shape} -> {test_labels.shape}")
    
    # Create datasets and loaders
    train_dataset = RealH36MDataset(train_data, train_labels)
    test_dataset = RealH36MDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Create model
    print(f"🧠 创建 {opts.model_type} 模型...")
    model = create_model(args, opts.model_type)
    model = model.to(device)
    
    # Setup training
    criterion = loss_mpjpe
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # Create save directory
    os.makedirs(opts.save_dir, exist_ok=True)
    
    best_mpjpe = float('inf')
    
    print(f"🚂 开始训练 ({opts.epochs} epochs)...")
    for epoch in range(opts.epochs):
        print(f"\\nEpoch {epoch+1}/{opts.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        mpjpe = evaluate(model, test_loader, device, datareader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, MPJPE: {mpjpe:.2f}mm")
        
        # Save best model
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mpjpe': mpjpe,
                'model_type': opts.model_type
            }, os.path.join(opts.save_dir, f'best_{opts.model_type}.pth'))
            print(f"💾 保存最佳模型 (MPJPE: {mpjpe:.2f}mm)")
    
    print(f"\\n🎉 训练完成！最佳MPJPE: {best_mpjpe:.2f}mm")
    print(f"📁 模型保存在: {opts.save_dir}")


if __name__ == '__main__':
    main()
'''

    with open('scripts/train_real.py', 'w', encoding='utf-8') as f:
        f.write(train_real_content)

    print("✅ 真实数据训练脚本已创建: scripts/train_real.py")


def update_baseline_validation():
    """更新baseline_validation.py以使用真实数据"""
    print("\n🔧 更新baseline_validation.py...")

    # Read existing file
    with open('baseline_validation.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace mock data reader import with real data reader
    content = content.replace(
        'from data.reader.mock_h36m import DataReaderMockH36M',
        'from data.reader.real_h36m import DataReaderRealH36M'
    )

    # Replace mock data reader instantiation
    content = content.replace(
        'DataReaderMockH36M(',
        'DataReaderRealH36M('
    )

    # Update data root path
    content = content.replace(
        "dt_root=args.data_root",
        "dt_root='data/motion3d/human36m/raw/motion3d'"
    )

    # Update comments
    content = content.replace(
        '# 创建 mock data reader',
        '# 创建真实Human3.6M data reader'
    )
    content = content.replace(
        'print("📊 加载mock数据...")',
        'print("📊 加载真实Human3.6M数据...")'
    )

    # Write updated file
    with open('baseline_validation_real.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 更新完成: baseline_validation_real.py")


def create_comparison_script():
    """创建mock vs real数据性能比较脚本"""
    print("\n📊 创建性能比较脚本...")

    comparison_content = '''#!/usr/bin/env python3
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
    print(f"\\n🧪 测试 {data_type} 数据 ({model_type})...")
    
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
    print("\\n" + "=" * 60)
    print("📊 性能比较总结")
    print("=" * 60)
    
    for result in results:
        print(f"{result['data_type']} + {result['model_type']:<10}: "
              f"MPJPE = {result['mpjpe']:.2f}mm, Loss = {result['loss']:.4f}")
    
    # Find best configuration
    if results:
        best_result = min(results, key=lambda x: x['mpjpe'])
        print(f"\\n🏆 最佳配置: {best_result['data_type']} + {best_result['model_type']} "
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
'''

    with open('compare_data_performance.py', 'w', encoding='utf-8') as f:
        f.write(comparison_content)

    print("✅ 性能比较脚本已创建: compare_data_performance.py")


def update_demo_script():
    """更新demo.py以支持真实数据"""
    print("\n🎭 更新demo.py...")

    # Read existing demo.py
    with open('demo.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Add real data import
    content = content.replace(
        'from data.reader.mock_h36m import DataReaderMockH36M',
        '''from data.reader.mock_h36m import DataReaderMockH36M
from data.reader.real_h36m import DataReaderRealH36M'''
    )

    # Add real data option to demo
    real_data_method = '''
    def load_real_data(self, subset='test', max_samples=1000):
        """加载真实Human3.6M数据"""
        print(f"\\n📊 加载真实Human3.6M数据 ({subset})...")
        
        try:
            datareader = DataReaderRealH36M(
                n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
                read_confidence=True, dt_root='data/motion3d/human36m/raw/motion3d'
            )
            
            train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
            
            if subset == 'train':
                data_2d, data_3d = train_data, train_labels
            else:
                data_2d, data_3d = test_data, test_labels
            
            # Limit sample size for demo
            if len(data_2d) > max_samples:
                indices = np.random.choice(len(data_2d), max_samples, replace=False)
                data_2d = data_2d[indices]
                data_3d = data_3d[indices]
            
            data_2d = torch.FloatTensor(data_2d).to(self.device)
            data_3d = torch.FloatTensor(data_3d).to(self.device)
            
            print(f"   ✅ 真实数据加载完成: {data_2d.shape} -> {data_3d.shape}")
            return data_2d, data_3d, datareader
            
        except Exception as e:
            print(f"   ❌ 真实数据加载失败: {e}")
            print("   🔄 fallback到mock数据...")
            return None, None, None'''

    # Insert the method after create_demo_data method
    insertion_point = content.find(
        'def compare_models(self, data_2d, data_3d):')
    if insertion_point != -1:
        content = content[:insertion_point] + real_data_method + \
            '\n\n    ' + content[insertion_point:]

    # Update demo execution modes
    content = content.replace(
        "modes = ['quick', 'full', 'config']",
        "modes = ['quick', 'full', 'config', 'real']"
    )

    # Add real data execution mode
    real_mode_code = '''
        elif mode == 'real':
            print("\\n🎯 真实数据演示模式")
            # Try to load real data
            real_2d, real_3d, real_reader = self.load_real_data(subset='test', max_samples=500)
            
            if real_2d is not None:
                print("\\n📊 使用真实Human3.6M数据进行演示...")
                data_2d, data_3d = real_2d, real_3d
                
                # Model comparison with real data
                results = self.compare_models(data_2d, data_3d)
                self.results[f'real_data'] = results
                
                # Generate report
                self.generate_report()
                
                print("\\n🎉 真实数据演示完成！")
            else:
                print("\\n⚠️  真实数据不可用，使用mock数据继续演示")
                mode = 'quick'  # Fallback to quick mode'''

    # Insert real mode handling
    config_mode_end = content.find('print("\\n🎉 配置演示完成！")')
    if config_mode_end != -1:
        next_elif = content.find('else:', config_mode_end)
        if next_elif != -1:
            content = content[:next_elif] + real_mode_code + \
                '\n        ' + content[next_elif:]

    # Write updated demo
    with open('demo_real.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 更新完成: demo_real.py (支持真实数据)")


def main():
    parser = argparse.ArgumentParser(description='数据迁移: Mock → Real Human3.6M')
    parser.add_argument('--check-only', action='store_true', help='仅检查数据可用性')
    parser.add_argument('--backup', action='store_true', help='备份现有mock文件')
    parser.add_argument('--force', action='store_true', help='强制执行迁移')
    args = parser.parse_args()

    print("🔄 Human3.6M数据迁移工具")
    print("=" * 50)
    print("目标: 将项目从模拟数据转换为真实Human3.6M数据")
    print("=" * 50)

    # Step 1: Check data availability
    data_available = check_real_data_availability()

    if not data_available:
        print("\\n❌ 真实数据不完整，无法进行迁移")
        print("💡 请确保以下数据文件存在:")
        print("  - data/motion3d/human36m/raw/motion3d/h36m_sh_conf_cam_source_final.pkl")
        print("  - data/motion3d/human36m/raw/motion3d/H36M-243/train/")
        print("  - data/motion3d/human36m/raw/motion3d/H36M-243/test/")
        return False

    if args.check_only:
        print("\\n✅ 数据检查完成，所有真实数据都可用")
        return True

    # Step 2: Backup mock files
    if args.backup:
        backup_mock_files()

    # Step 3: Create new files for real data
    create_real_training_script()
    update_baseline_validation()
    create_comparison_script()
    update_demo_script()

    # Step 4: Test real data loading
    print("\\n🧪 测试真实数据加载...")
    try:
        from data.reader.real_h36m import DataReaderRealH36M

        reader = DataReaderRealH36M(
            n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
            read_confidence=True, dt_root='data/motion3d/human36m/raw/motion3d'
        )

        train_data, test_data, train_labels, test_labels = reader.get_sliced_data()
        print(f"  ✅ 训练数据: {train_data.shape} -> {train_labels.shape}")
        print(f"  ✅ 测试数据: {test_data.shape} -> {test_labels.shape}")

    except Exception as e:
        print(f"  ❌ 数据加载测试失败: {e}")
        return False

    # Step 5: Summary
    print("\\n" + "=" * 50)
    print("✅ 数据迁移完成总结")
    print("=" * 50)
    print("🆕 新创建的文件:")
    print("  - data/reader/real_h36m.py          # 真实数据读取器")
    print("  - scripts/train_real.py             # 真实数据训练脚本")
    print("  - baseline_validation_real.py       # 真实数据基线验证")
    print("  - compare_data_performance.py       # Mock vs Real 性能比较")
    print("  - demo_real.py                      # 支持真实数据的演示")

    print("\\n🔧 使用方法:")
    print("  # 性能比较")
    print("  python compare_data_performance.py")
    print()
    print("  # 真实数据训练")
    print("  python scripts/train_real.py --model_type mamba_gcn --epochs 5")
    print()
    print("  # 真实数据基线验证")
    print("  python baseline_validation_real.py")
    print()
    print("  # 真实数据演示")
    print("  python demo_real.py real")

    print("\\n🎯 下一步建议:")
    print("  1. 运行性能比较了解真实 vs 模拟数据的差异")
    print("  2. 使用真实数据重新训练和验证模型")
    print("  3. 更新最终交付文档中的性能指标")

    print("\\n🎉 迁移完成！项目现在可以使用真实Human3.6M数据了")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
