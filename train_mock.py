#!/usr/bin/env python3
"""
Simplified training script for MotionAGFormer with mock data
Used for baseline verification and development
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
from data.reader.mock_h36m import DataReaderMockH36M
from model.MotionAGFormer import MotionAGFormer
from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity
from utils.tools import set_random_seed, get_config
from utils.learning import AverageMeter


class MockDataset(Dataset):
    """Simple dataset wrapper for mock data"""
    def __init__(self, data_2d, data_3d):
        self.data_2d = torch.FloatTensor(data_2d)
        self.data_3d = torch.FloatTensor(data_3d)
        
    def __len__(self):
        return len(self.data_2d)
    
    def __getitem__(self, idx):
        return self.data_2d[idx], self.data_3d[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", 
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for mock training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for mock training")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, mps, or cpu")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    opts = parser.parse_args()
    return opts


def get_device(device_arg):
    """Auto-detect best available device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def create_model(args):
    """Create MotionAGFormer model"""
    model = MotionAGFormer(
        n_layers=args.n_layers,
        dim_in=args.dim_in,
        dim_feat=args.dim_feat,
        dim_rep=args.dim_rep,
        dim_out=args.dim_out,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        n_frames=args.n_frames,
        num_joints=args.num_joints,
        use_adaptive_fusion=args.use_adaptive_fusion,
        use_temporal_similarity=args.use_temporal_similarity,
        neighbour_num=args.neighbour_num,
        use_tcn=args.use_tcn,
        graph_only=args.graph_only
    )
    return model


def train_one_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    
    losses = {
        '3d_pose': AverageMeter(),
        '3d_scale': AverageMeter(), 
        '3d_velocity': AverageMeter(),
        'total': AverageMeter()
    }
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        
        # Normalize target (root relative)
        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
        
        # Forward pass
        pred = model(x)
        
        # Calculate losses
        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_vel = loss_velocity(pred, y)
        
        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_vel
        
        # Update losses
        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_vel.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss_total.item():.4f}")
    
    return losses


def evaluate(model, test_loader, device, args):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_mpjpe = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            pred = model(x)
            
            # Normalize (root relative)
            if args.root_rel:
                pred = pred - pred[..., 0:1, :]
                y = y - y[..., 0:1, :]
            
            # Calculate metrics
            mpjpe = loss_mpjpe(pred, y)
            total_mpjpe += mpjpe.item()
            num_batches += 1
    
    avg_mpjpe = total_mpjpe / num_batches
    return avg_mpjpe


def main():
    opts = parse_args()
    
    # Load config
    args = get_config(opts.config)
    
    # Override some settings for mock training
    args.epochs = opts.epochs
    args.batch_size = opts.batch_size
    args.n_frames = 243  # Use full sequence length
    
    # Set device
    device = get_device(opts.device)
    print(f"🔧 Using device: {device}")
    
    # Set random seed
    set_random_seed(42)
    
    # Create mock data reader
    print("📊 Loading mock data...")
    datareader = DataReaderMockH36M(
        n_frames=args.n_frames,
        sample_stride=1,
        data_stride_train=81,
        data_stride_test=243,
        read_confidence=True,
        dt_root=args.data_root
    )
    
    # Get data
    train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
    print(f"✅ Data loaded:")
    print(f"   - Train: {train_data.shape} -> {train_labels.shape}")
    print(f"   - Test: {test_data.shape} -> {test_labels.shape}")
    
    # Create datasets and loaders
    train_dataset = MockDataset(train_data, train_labels)
    test_dataset = MockDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Create model
    print("🧠 Creating MotionAGFormer model...")
    model = create_model(args)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total params: {total_params:,}")
    print(f"   - Trainable params: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                           weight_decay=args.weight_decay)
    
    # Test forward pass
    print("🔍 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        x_sample, y_sample = sample_batch[0][:1].to(device), sample_batch[1][:1].to(device)
        pred_sample = model(x_sample)
        print(f"   - Input shape: {x_sample.shape}")
        print(f"   - Output shape: {pred_sample.shape}")
        print(f"   - Target shape: {y_sample.shape}")
        print("✅ Forward pass successful!")
    
    if opts.eval_only:
        print("📊 Running evaluation only...")
        avg_mpjpe = evaluate(model, test_loader, device, args)
        print(f"   - Average MPJPE: {avg_mpjpe:.2f}mm")
        return
    
    # Training loop
    print(f"🚀 Starting training for {args.epochs} epochs...")
    best_mpjpe = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, device, args)
        
        # Evaluate
        avg_mpjpe = evaluate(model, test_loader, device, args)
        
        # Print epoch results
        print(f"📈 Epoch {epoch + 1} Results:")
        print(f"   - Train Loss: {train_losses['total'].avg:.4f}")
        print(f"   - Train MPJPE: {train_losses['3d_pose'].avg:.2f}mm")
        print(f"   - Test MPJPE: {avg_mpjpe:.2f}mm")
        
        # Save best model
        if avg_mpjpe < best_mpjpe:
            best_mpjpe = avg_mpjpe
            print(f"   ⭐ New best MPJPE: {best_mpjpe:.2f}mm")
    
    print(f"\n🎉 Training completed!")
    print(f"   - Best MPJPE: {best_mpjpe:.2f}mm")
    print(f"   - Final train loss: {train_losses['total'].avg:.4f}")


if __name__ == "__main__":
    main() 