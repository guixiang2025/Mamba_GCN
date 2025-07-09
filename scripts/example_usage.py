#!/usr/bin/env python3
"""
Usage example for MotionAGFormer with MambaGCN integration.
Shows how to use the new three-branch architecture.
"""

from model.MotionAGFormer import MotionAGFormer
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("🚀 MotionAGFormer + MambaGCN Usage Example")
    print("=" * 50)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Example configuration
    batch_size = 4
    n_frames = 243  # Typical sequence length
    n_joints = 17   # Human3.6M skeleton
    input_dim = 2   # 2D pose coordinates

    # Create sample input data [B, T, J, C]
    input_data = torch.randn(batch_size, n_frames,
                             n_joints, input_dim).to(device)
    print(f"Input shape: {input_data.shape}")

    # Configuration 1: Original MotionAGFormer (baseline)
    print(f"\n🔹 Configuration 1: Original MotionAGFormer")
    model_original = MotionAGFormer(
        n_layers=6,
        dim_in=input_dim,
        dim_feat=128,
        dim_rep=256,
        dim_out=3,
        n_frames=n_frames,
        use_mamba_gcn=False  # Disable MambaGCN
    ).to(device)

    with torch.no_grad():
        output_original = model_original(input_data)

    print(
        f"   Parameters: {sum(p.numel() for p in model_original.parameters()):,}")
    print(f"   Output shape: {output_original.shape}")

    # Configuration 2: MotionAGFormer + MambaGCN (Mamba + GCN)
    print(f"\n🔹 Configuration 2: MotionAGFormer + MambaGCN (Mamba+GCN)")
    model_mamba_gcn = MotionAGFormer(
        n_layers=6,
        dim_in=input_dim,
        dim_feat=128,
        dim_rep=256,
        dim_out=3,
        n_frames=n_frames,
        use_mamba_gcn=True,           # Enable MambaGCN
        mamba_gcn_use_mamba=True,     # Use Mamba branch
        mamba_gcn_use_attention=False  # Disable attention branch
    ).to(device)

    with torch.no_grad():
        output_mamba_gcn = model_mamba_gcn(input_data)

    print(
        f"   Parameters: {sum(p.numel() for p in model_mamba_gcn.parameters()):,}")
    print(f"   Output shape: {output_mamba_gcn.shape}")

    # Configuration 3: Full three-branch (Mamba + GCN + Attention)
    print(f"\n🔹 Configuration 3: Full Three-Branch (Mamba+GCN+Attention)")
    model_full = MotionAGFormer(
        n_layers=6,
        dim_in=input_dim,
        dim_feat=128,
        dim_rep=256,
        dim_out=3,
        n_frames=n_frames,
        use_mamba_gcn=True,          # Enable MambaGCN
        mamba_gcn_use_mamba=True,    # Use Mamba branch
        mamba_gcn_use_attention=True  # Enable attention branch
    ).to(device)

    with torch.no_grad():
        output_full = model_full(input_data)

    print(
        f"   Parameters: {sum(p.numel() for p in model_full.parameters()):,}")
    print(f"   Output shape: {output_full.shape}")

    # Demonstrate training setup
    print(f"\n🔧 Training Setup Example")
    model_full.train()

    # Dummy loss computation
    target = torch.randn(batch_size, n_frames, n_joints, 3).to(device)
    criterion = torch.nn.MSELoss()

    # Forward pass
    pred = model_full(input_data)
    loss = criterion(pred, target)

    # Backward pass
    loss.backward()

    print(f"   Loss: {loss.item():.6f}")
    print(f"   Gradients computed successfully!")

    # Model configuration summary
    print(f"\n📋 Model Configuration Summary")
    print(f"   Original MotionAGFormer: 2 branches (ST-Attention + ST-Graph)")
    print(f"   + MambaGCN: 3 branches (ST-Attention + ST-Graph + MambaGCN)")
    print(f"   + Linear complexity temporal modeling via Mamba")
    print(f"   + Human skeleton structure awareness via GCN")
    print(f"   + Adaptive fusion with learnable weights")

    print(f"\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
