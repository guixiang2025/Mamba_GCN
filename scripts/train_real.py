#!/usr/bin/env python3
"""
Real H36M Training Script for MotionAGFormer + MambaGCN
Uses actual Human3.6M dataset for authentic training and evaluation
Enhanced with comprehensive logging and results recording
"""

from utils.learning import AverageMeter
from utils.tools import set_random_seed, get_config
from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity
from model.MotionAGFormer import MotionAGFormer
from data.reader.real_h36m import DataReaderRealH36M
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import sys
import time
import json
from datetime import datetime
sys.path.append('.')
sys.path.append('/home/hpe/Mamba_GCN')


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.log_file = os.path.join(save_dir, 'training.log')
        self.metrics_file = os.path.join(save_dir, 'metrics.json')

        self.metrics = {
            'train_losses': [],
            'train_mpjpe': [],
            'test_mpjpe': [],
            'epoch_times': [],
            'learning_rates': [],
            'improvements': []
        }

        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write(
                f"🚀 Real H36M 训练日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")

    def log(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        print(log_message)

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def save_metrics(self):
        """保存训练指标"""
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                metrics_json[key] = [float(v) if hasattr(
                    v, 'item') else v for v in value]
            else:
                metrics_json[key] = float(value) if hasattr(
                    value, 'item') else value

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)


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
    parser = argparse.ArgumentParser(
        description='MotionAGFormer + MambaGCN Real H36M Training')
    parser.add_argument('--config', type=str, default='configs/h36m/MotionAGFormer-base.yaml',
                        help='Config file path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cpu, cuda, cuda:0, cuda:1 (overrides config)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['baseline', 'mamba_gcn', 'full'],
                        help='Model configuration type (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save models (overrides config)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames in sequence (overrides config)')
    return parser.parse_args()


def merge_config(config, opts):
    """合并配置：命令行 > 配置文件 > 默认值"""

    # 设置默认值（如果配置文件中没有）
    if not hasattr(config, 'epochs'):
        config.epochs = 5
    if not hasattr(config, 'batch_size'):
        config.batch_size = 16
    if not hasattr(config, 'learning_rate'):
        config.learning_rate = 1e-4
    if not hasattr(config, 'device'):
        config.device = 'cuda:1'
    if not hasattr(config, 'model_type'):
        config.model_type = 'baseline'
    if not hasattr(config, 'save_dir'):
        config.save_dir = 'checkpoints/real_h36m_enhanced'
    if not hasattr(config, 'n_frames'):
        config.n_frames = 243

    # 命令行参数覆盖配置文件（如果提供）
    if opts.epochs is not None:
        config.epochs = opts.epochs
    if opts.batch_size is not None:
        config.batch_size = opts.batch_size
    if opts.learning_rate is not None:
        config.learning_rate = opts.learning_rate
    if opts.device is not None:
        config.device = opts.device
    if opts.model_type is not None:
        config.model_type = opts.model_type
    if opts.save_dir is not None:
        config.save_dir = opts.save_dir
    if opts.n_frames is not None:
        config.n_frames = opts.n_frames

    return config


def get_device(device_str):
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def create_model(config, model_type=None):
    """Create model based on type"""

    if model_type is None:
        model_type = config.model_type

    if model_type == 'baseline':
        # Original MotionAGFormer
        model = MotionAGFormer(
            n_layers=config.n_layers,
            dim_in=2,  # 2D input
            dim_feat=config.dim_feat,
            dim_out=config.dim_out,
            n_frames=config.n_frames,
            use_mamba_gcn=False
        )
    elif model_type == 'mamba_gcn':
        # MambaGCN enhanced
        model = MotionAGFormer(
            n_layers=config.n_layers,
            dim_in=2,  # 2D input
            dim_feat=config.dim_feat,
            dim_out=config.dim_out,
            n_frames=config.n_frames,
            use_mamba_gcn=True,
            mamba_gcn_use_mamba=True,
            mamba_gcn_use_attention=False
        )
    elif model_type == 'full':
        # Full architecture with all components
        model = MotionAGFormer(
            n_layers=config.n_layers,
            dim_in=2,  # 2D input
            dim_feat=config.dim_feat,
            dim_out=config.dim_out,
            n_frames=config.n_frames,
            use_mamba_gcn=True,
            mamba_gcn_use_mamba=True,
            mamba_gcn_use_attention=True
        )

    return model


def train_epoch(model, train_loader, optimizer, criterion, device, logger):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    train_losses = []

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (data_2d, data_3d) in enumerate(pbar):
        data_2d, data_3d = data_2d.to(device), data_3d.to(device)

        optimizer.zero_grad()
        pred_3d = model(data_2d)

        # Compute loss
        loss = criterion(pred_3d, data_3d)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data_2d.size(0))
        train_losses.append(loss.item())
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 定期清理内存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    return losses.avg, train_losses


def evaluate(model, test_loader, device, logger):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for data_2d, data_3d in tqdm(test_loader, desc='Evaluating', leave=False):
            data_2d, data_3d = data_2d.to(device), data_3d.to(device)

            pred_3d = model(data_2d)

            # Accumulate for MPJPE calculation
            predictions.append(pred_3d.cpu().numpy())
            targets.append(data_3d.cpu().numpy())

            total_samples += data_2d.size(0)

    # Calculate MPJPE
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # 正确的MPJPE计算：将标准化坐标转换回毫米
    # Human3.6M的标准化是：coord = coord / 1000 * 2, 所以反向转换是：coord = coord * 1000 / 2
    predictions_mm = predictions * 500  # 500 = 1000/2，转换回毫米
    targets_mm = targets * 500

    # 计算每个关节的欧几里德距离，然后平均 (mm)
    predictions_flat = predictions_mm.reshape(-1, 17, 3)
    targets_flat = targets_mm.reshape(-1, 17, 3)

    # MPJPE: 计算每个关节的L2距离，然后对所有关节和所有样本求平均
    joint_distances = np.sqrt(
        np.sum((predictions_flat - targets_flat) ** 2, axis=-1))  # [N*T, 17]
    mpjpe = np.mean(joint_distances)

    return mpjpe


def generate_training_report(save_dir, metrics, initial_mpjpe, best_mpjpe, total_time, model_type):
    """生成训练报告"""
    report_file = os.path.join(save_dir, 'training_report.md')

    with open(report_file, 'w') as f:
        f.write(f"# Real H36M {model_type.upper()} 训练报告\n\n")
        f.write(
            f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 🎯 训练配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|----|\n")
        f.write(f"| 架构 | MotionAGFormer + {model_type.upper()} |\n")
        f.write(f"| Epoch数量 | {len(metrics['test_mpjpe'])} |\n")
        f.write("| 数据集 | Real Human3.6M |\n")
        f.write("| 设备 | CUDA:1 |\n")
        f.write("| 序列长度 | 243 frames |\n\n")

        f.write("## 📊 训练结果\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        f.write(f"| 初始MPJPE | {initial_mpjpe:.2f}mm |\n")
        f.write(f"| 最终MPJPE | {metrics['test_mpjpe'][-1]:.2f}mm |\n")
        f.write(f"| 最佳MPJPE | {best_mpjpe:.2f}mm |\n")
        f.write(
            f"| 总体改善 | {(initial_mpjpe - best_mpjpe) / initial_mpjpe * 100:.1f}% |\n")
        f.write(f"| 训练时间 | {total_time:.1f}秒 |\n")
        f.write(
            f"| 平均每epoch | {total_time / len(metrics['test_mpjpe']):.1f}秒 |\n\n")

        f.write("## 📈 逐Epoch结果\n\n")
        f.write("| Epoch | 训练损失 | 训练MPJPE | 测试MPJPE | 改善% | 用时(s) |\n")
        f.write("|-------|----------|-----------|-----------|--------|--------|\n")

        for i in range(len(metrics['test_mpjpe'])):
            improvement = (initial_mpjpe -
                           metrics['test_mpjpe'][i]) / initial_mpjpe * 100
            f.write(
                f"| {i+1} | {metrics['train_losses'][i]:.4f} | {metrics['train_mpjpe'][i]:.2f}mm | {metrics['test_mpjpe'][i]:.2f}mm | {improvement:+.1f}% | {metrics['epoch_times'][i]:.1f}s |\n")

        f.write("\n## 🏆 结论\n\n")
        if best_mpjpe < initial_mpjpe * 0.8:
            f.write("✅ **训练成功**: 模型性能显著改善\n")
        else:
            f.write("⚠️  **需要更多训练**: 建议增加训练epoch数以获得更好性能\n")

        if best_mpjpe < 40:
            f.write("🎯 **已达到目标**: MPJPE < 40mm 的性能目标已实现\n")
        else:
            f.write(f"🎯 **接近目标**: 当前最佳MPJPE {best_mpjpe:.2f}mm，距离40mm目标还需改善\n")


def main():
    opts = parse_args()

    # Load config and merge with command line arguments
    config = get_config(opts.config)
    config = merge_config(config, opts)

    # Set device
    device = get_device(config.device)
    print(f"🔧 使用设备: {device}")

    # Check GPU memory
    if 'cuda' in device:
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        print(f"📊 GPU {gpu_id} 内存状态:")
        print(
            f"   - 总内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f}GB")
        print(
            f"   - 已分配: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.1f}GB")
        print(f"   - 缓存: {torch.cuda.memory_reserved(gpu_id) / 1024**3:.1f}GB")

    # Set random seed
    set_random_seed(42)

    # Initialize logger
    logger = TrainingLogger(config.save_dir)
    logger.log("🎯 Real H36M 训练开始")
    logger.log(f"📊 训练配置:")
    logger.log(f"   - 架构: MotionAGFormer + {config.model_type.upper()}")
    logger.log(f"   - Epoch数量: {config.epochs}")
    logger.log(f"   - 批次大小: {config.batch_size}")
    logger.log(f"   - 学习率: {config.learning_rate}")
    logger.log(f"   - 设备: {device}")
    logger.log(f"   - 序列长度: {config.n_frames}")
    logger.log(f"   - 保存目录: {config.save_dir}")

    # Create real data reader
    logger.log("\n📊 加载真实Human3.6M数据...")
    try:
        datareader = DataReaderRealH36M(
            n_frames=config.n_frames,
            sample_stride=1,
            data_stride_train=81,
            data_stride_test=243,
            read_confidence=True,
            dt_root='data/motion3d/human36m/raw/motion3d'
        )

        # Get data
        train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
        logger.log(f"✅ 真实数据加载完成:")
        logger.log(f"   - 训练集: {train_data.shape} -> {train_labels.shape}")
        logger.log(f"   - 测试集: {test_data.shape} -> {test_labels.shape}")

    except Exception as e:
        logger.log(f"❌ 数据加载失败: {e}")
        return False

    # Create datasets and loaders
    train_dataset = RealH36MDataset(
        train_data[:, :, :, :2], train_labels)  # 只使用2D坐标
    test_dataset = RealH36MDataset(
        test_data[:, :, :, :2], test_labels)  # 只使用2D坐标

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    logger.log(f"   - 训练批次: {len(train_loader)}")
    logger.log(f"   - 测试批次: {len(test_loader)}")

    # Create model
    logger.log(f"\n🧠 创建 {config.model_type.upper()} 模型...")
    model = create_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.log(f"✅ 模型创建成功:")
    logger.log(f"   - 总参数: {total_params:,}")
    logger.log(f"   - 可训练参数: {trainable_params:,}")

    # Setup training
    criterion = loss_mpjpe
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.9)

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # 训练前评估
    logger.log("\n📊 训练前基线评估...")
    initial_mpjpe = evaluate(model, test_loader, device, logger)
    logger.log(f"   初始MPJPE: {initial_mpjpe:.2f}mm")

    best_mpjpe = float('inf')
    training_start_time = time.time()

    logger.log(f"\n🚂 开始训练 ({config.epochs} epochs)...")

    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        logger.log(f"\n📅 Epoch {epoch+1}/{config.epochs}")

        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, logger)

        # Evaluate
        test_mpjpe = evaluate(model, test_loader, device, logger)

        # Convert training loss to mm scale
        train_mpjpe = train_loss * 1000

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Calculate improvement
        improvement = (initial_mpjpe - test_mpjpe) / initial_mpjpe * 100

        # Record metrics
        logger.metrics['train_losses'].append(train_loss)
        logger.metrics['train_mpjpe'].append(train_mpjpe)
        logger.metrics['test_mpjpe'].append(test_mpjpe)
        logger.metrics['epoch_times'].append(epoch_time)
        logger.metrics['learning_rates'].append(current_lr)
        logger.metrics['improvements'].append(improvement)

        # Log epoch results
        logger.log(f"   训练损失: {train_loss:.6f}")
        logger.log(f"   训练MPJPE: {train_mpjpe:.2f}mm")
        logger.log(f"   测试MPJPE: {test_mpjpe:.2f}mm")
        logger.log(f"   性能改善: {improvement:+.1f}%")
        logger.log(f"   学习率: {current_lr:.2e}")
        logger.log(f"   用时: {epoch_time:.1f}秒")

        # Save best model
        if test_mpjpe < best_mpjpe:
            best_mpjpe = test_mpjpe
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mpjpe': test_mpjpe,
                'model_type': config.model_type,
                'config': config
            }, os.path.join(config.save_dir, f'best_{config.model_type}.pth'))
            logger.log(f"   🏆 新的最佳MPJPE: {best_mpjpe:.2f}mm (已保存)")

        # Save current model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mpjpe': test_mpjpe,
            'model_type': config.model_type,
            'config': config
        }, os.path.join(config.save_dir, f'epoch_{epoch+1}_{config.model_type}.pth'))

    # Training complete
    total_training_time = time.time() - training_start_time

    logger.log(f"\n🎉 训练完成总结:")
    logger.log(f"   总训练时间: {total_training_time:.1f}秒")
    logger.log(f"   平均每epoch: {total_training_time/config.epochs:.1f}秒")
    logger.log(f"   初始MPJPE: {initial_mpjpe:.2f}mm")
    logger.log(f"   最终MPJPE: {test_mpjpe:.2f}mm")
    logger.log(f"   最佳MPJPE: {best_mpjpe:.2f}mm")
    logger.log(
        f"   总体改善: {(initial_mpjpe - best_mpjpe) / initial_mpjpe * 100:.1f}%")

    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mpjpe': test_mpjpe,
        'model_type': config.model_type,
        'config': config,
        'final': True
    }, os.path.join(config.save_dir, f'final_{config.model_type}.pth'))

    # Save training metrics
    logger.save_metrics()

    # Generate training report
    generate_training_report(config.save_dir, logger.metrics,
                             initial_mpjpe, best_mpjpe, total_training_time, config.model_type)

    logger.log(f"\n📁 所有文件已保存到: {config.save_dir}")
    logger.log(f"   - 训练日志: training.log")
    logger.log(f"   - 训练指标: metrics.json")
    logger.log(f"   - 最佳模型: best_{config.model_type}.pth")
    logger.log(f"   - 最终模型: final_{config.model_type}.pth")
    logger.log(f"   - 训练报告: training_report.md")

    print(f"\n🎉 训练成功完成！最佳MPJPE: {best_mpjpe:.2f}mm")
    print(f"📁 模型保存在: {config.save_dir}")
    print(f"🎯 配置总结:")
    print(f"   - 架构: {config.model_type}")
    print(f"   - Epochs: {config.epochs}")
    print(f"   - 批次大小: {config.batch_size}")
    print(f"   - 学习率: {config.learning_rate}")
    print(f"   - 设备: {device}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
