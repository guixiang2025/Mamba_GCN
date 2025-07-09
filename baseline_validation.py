#!/usr/bin/env python3
"""
Task 1.4: MotionAGFormer 基线验证脚本
=====================================
完成基线模型的验证，确保环境和代码的正确性
"""

import os
import sys
import time
import argparse
from pathlib import Path

def check_dependencies():
    """T1.4.1: 检查运行环境和依赖"""
    print("🔍 检查运行环境...")
    
    required_packages = [
        'torch', 'numpy', 'scipy', 'matplotlib', 
        'tqdm', 'yaml', 'timm', 'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: 缺失")
    
    if missing_packages:
        print(f"\n⚠️  缺失依赖: {missing_packages}")
        print("请先安装依赖: pip install torch numpy scipy matplotlib tqdm pyyaml timm pillow")
        return False
    
    print("✅ 所有核心依赖已安装")
    return True

def load_config():
    """加载基线配置"""
    config_path = "configs/h36m/MotionAGFormer-base.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def create_mock_model(config):
    """T1.4.2: 创建基线模型实例"""
    print("\n🏗️  创建基线模型...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model.MotionAGFormer import MotionAGFormer
        
        model = MotionAGFormer(
            n_layers=config.get('n_layers', 16),
            dim_in=config.get('dim_in', 3),
            dim_feat=config.get('dim_feat', 128),
            dim_rep=config.get('dim_rep', 512),
            dim_out=config.get('dim_out', 3),
            num_joints=config.get('num_joints', 17),
            n_frames=config.get('n_frames', 243)
        )
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建成功")
        print(f"   📊 总参数: {total_params:,}")
        print(f"   🎯 可训练参数: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_forward_pass(model, config):
    """T1.4.3: 测试前向传播"""
    print("\n🚀 测试前向传播...")
    
    try:
        import torch
        
        # 创建测试输入
        batch_size = 2
        n_frames = config.get('n_frames', 243) 
        num_joints = config.get('num_joints', 17)
        dim_in = config.get('dim_in', 3)
        
        x = torch.randn(batch_size, n_frames, num_joints, dim_in)
        print(f"   📥 输入形状: {x.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(x)
            end_time = time.time()
        
        print(f"   📤 输出形状: {output.shape}")
        print(f"   ⏱️  推理时间: {end_time - start_time:.4f}s")
        
        # 验证输出维度
        expected_shape = (batch_size, n_frames, num_joints, config.get('dim_out', 3))
        if output.shape == expected_shape:
            print(f"✅ 输出维度正确: {output.shape}")
            return True
        else:
            print(f"❌ 输出维度错误: 期望 {expected_shape}, 实际 {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return False

def simulate_training_step(model, config):
    """T1.4.4: 模拟训练步骤"""
    print("\n🏋️‍♂️ 模拟训练步骤...")
    
    try:
        import torch
        import torch.optim as optim
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.0005))
        
        # 创建训练数据
        batch_size = config.get('batch_size', 16)
        n_frames = config.get('n_frames', 243)
        num_joints = config.get('num_joints', 17)
        
        x = torch.randn(batch_size, n_frames, num_joints, config.get('dim_in', 3))
        y = torch.randn(batch_size, n_frames, num_joints, config.get('dim_out', 3))
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        loss.backward()
        optimizer.step()
        
        print(f"✅ 训练步骤完成")
        print(f"   📊 Loss: {loss.item():.6f}")
        print(f"   🎯 可训练性: 正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return False

def check_data_pipeline():
    """检查数据加载管道"""
    print("\n📊 检查数据管道...")
    
    # 检查数据目录
    data_dirs = ["data/motion3d", "data/preprocess", "data/reader"]
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"✅ {data_dir}: 存在")
        else:
            print(f"⚠️  {data_dir}: 不存在")
    
    # 检查 mock 数据
    mock_data_script = "data/create_mock_data.py"
    if os.path.exists(mock_data_script):
        print(f"✅ Mock数据生成器: 可用")
        return True
    else:
        print(f"❌ Mock数据生成器: 缺失")
        return False

def main():
    """主函数：执行基线验证"""
    parser = argparse.ArgumentParser(description='MotionAGFormer 基线验证')
    parser.add_argument('--skip-deps', action='store_true', help='跳过依赖检查')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 Task 1.4: MotionAGFormer 基线验证")
    print("=" * 60)
    
    # T1.4.1: 检查依赖
    if not args.skip_deps:
        if not check_dependencies():
            print("\n❌ 环境依赖检查失败，请先安装所需依赖")
            return False
    
    # 加载配置
    config = load_config()
    if config is None:
        return False
    
    # T1.4.2: 创建模型
    model = create_mock_model(config)
    if model is None:
        return False
    
    # T1.4.3: 测试前向传播
    if not test_forward_pass(model, config):
        return False
    
    # T1.4.4: 模拟训练
    if not simulate_training_step(model, config):
        return False
    
    # 检查数据管道
    data_ok = check_data_pipeline()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 基线验证总结")
    print("=" * 60)
    print("✅ 模型架构: 正常")
    print("✅ 前向传播: 正常") 
    print("✅ 训练能力: 正常")
    print(f"{'✅' if data_ok else '⚠️'} 数据管道: {'正常' if data_ok else '需要配置'}")
    
    print("\n🎉 基线验证完成！可以进入 Mamba-GCN 开发阶段")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 