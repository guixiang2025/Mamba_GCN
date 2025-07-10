#!/usr/bin/env python3
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
        print(f"\nEpoch {epoch+1}/{opts.epochs}")
        
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
    
    print(f"\n🎉 训练完成！最佳MPJPE: {best_mpjpe:.2f}mm")
    print(f"📁 模型保存在: {opts.save_dir}")


if __name__ == '__main__':
    main()
