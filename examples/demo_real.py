#!/usr/bin/env python3
"""
🎯 MotionAGFormer + MambaGCN 一键演示脚本

这个脚本展示了完整的 Mamba-GCN 模型使用流程：
1. 模型创建和配置
2. 数据加载和预处理
3. 快速训练验证
4. 推理和结果可视化
5. 性能对比分析

使用方法:
    python demo.py                     # 默认演示
    python demo.py --quick             # 快速演示 (1 epoch)
    python demo.py --full              # 完整演示 (包含所有配置)
    python demo.py --config baseline   # 指定模型配置
"""

from model.MotionAGFormer import MotionAGFormer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MambaGCNDemo:
    """
    Mamba-GCN 演示类
    """

    def __init__(self, device='auto'):
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.demo_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"🚀 Mamba-GCN Demo Initialize")
        print(f"   Device: {self.device}")
        print(f"   Time: {self.demo_time}")

    def create_demo_data(self, batch_size=8, n_frames=27, n_joints=17):
        """创建演示数据"""

        print(f"\n📊 Creating demo data...")
        print(f"   Shape: [{batch_size}, {n_frames}, {n_joints}, 2→3]")

        # 创建模拟的 2D 姿态序列
        input_2d = torch.randn(batch_size, n_frames, n_joints, 2)

        # 创建对应的 3D 真值 (用于计算损失)
        target_3d = torch.randn(batch_size, n_frames, n_joints, 3)

        # 移动到设备
        input_2d = input_2d.to(self.device)
        target_3d = target_3d.to(self.device)

        # 创建数据加载器
        dataset = TensorDataset(input_2d, target_3d)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"   ✅ Demo data created")
        return dataloader

    def create_models(self, n_frames=27):
        """创建不同配置的模型"""

        print(f"\n🧠 Creating models...")

        model_configs = {
            'baseline': {
                'name': 'Baseline MotionAGFormer',
                'config': {
                    'n_layers': 4, 'dim_in': 2, 'dim_feat': 64, 'dim_out': 3,
                    'n_frames': n_frames, 'use_mamba_gcn': False
                }
            },
            'mamba_gcn': {
                'name': 'MotionAGFormer + MambaGCN',
                'config': {
                    'n_layers': 4, 'dim_in': 2, 'dim_feat': 64, 'dim_out': 3,
                    'n_frames': n_frames, 'use_mamba_gcn': True,
                    'mamba_gcn_use_mamba': True, 'mamba_gcn_use_attention': False
                }
            },
            'mamba_gcn_full': {
                'name': 'MotionAGFormer + MambaGCN (Full)',
                'config': {
                    'n_layers': 4, 'dim_in': 2, 'dim_feat': 64, 'dim_out': 3,
                    'n_frames': n_frames, 'use_mamba_gcn': True,
                    'mamba_gcn_use_mamba': True, 'mamba_gcn_use_attention': True
                }
            }
        }

        models = {}
        for key, config in model_configs.items():
            print(f"   🔧 {config['name']}...")

            try:
                model = MotionAGFormer(**config['config'])
                model = model.to(self.device)
                models[key] = {
                    'model': model,
                    'name': config['name'],
                    'params': sum(p.numel() for p in model.parameters()),
                    'config': config['config']
                }
                print(f"      ✅ Parameters: {models[key]['params']:,}")

            except Exception as e:
                print(f"      ❌ Failed: {e}")
                models[key] = None

        return models

    def quick_training(self, models, dataloader, epochs=2):
        """快速训练验证"""

        print(f"\n🚂 Quick training validation ({epochs} epochs)...")

        criterion = nn.MSELoss()
        training_results = {}

        for model_key, model_info in models.items():
            if model_info is None:
                continue

            print(f"\n   🎯 Training {model_info['name']}...")

            model = model_info['model']
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            model.train()
            start_time = time.time()
            epoch_losses = []

            try:
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch_idx, (input_2d, target_3d) in enumerate(dataloader):
                        # Forward pass
                        optimizer.zero_grad()
                        output_3d = model(input_2d)
                        loss = criterion(output_3d, target_3d)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(dataloader)
                    epoch_losses.append(avg_loss)
                    print(
                        f"      Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

                training_time = time.time() - start_time

                # 计算改进率
                loss_improvement = (
                    (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0]) * 100 if len(epoch_losses) > 1 else 0

                training_results[model_key] = {
                    'name': model_info['name'],
                    'params': model_info['params'],
                    'initial_loss': epoch_losses[0],
                    'final_loss': epoch_losses[-1],
                    'loss_improvement': loss_improvement,
                    'training_time': training_time,
                    'losses': epoch_losses,
                    'status': 'success'
                }

                print(f"      ✅ Training time: {training_time:.2f}s")
                print(f"      📈 Loss improvement: {loss_improvement:.2f}%")

            except Exception as e:
                print(f"      ❌ Training failed: {e}")
                training_results[model_key] = {
                    'name': model_info['name'],
                    'status': 'failed',
                    'error': str(e)
                }

        self.results['training'] = training_results
        return training_results

    def inference_benchmark(self, models, batch_size=8, n_frames=27, n_joints=17, num_runs=10):
        """推理性能基准测试"""

        print(f"\n⚡ Inference benchmark ({num_runs} runs)...")

        # 创建测试数据
        test_input = torch.randn(batch_size, n_frames,
                                 n_joints, 2).to(self.device)

        inference_results = {}

        for model_key, model_info in models.items():
            if model_info is None:
                continue

            print(f"   🔍 {model_info['name']}...")

            model = model_info['model']
            model.eval()

            try:
                # Warmup
                with torch.no_grad():
                    _ = model(test_input)

                # Benchmark
                inference_times = []
                with torch.no_grad():
                    for run in range(num_runs):
                        start_time = time.time()
                        output = model(test_input)
                        end_time = time.time()
                        inference_times.append(
                            (end_time - start_time) * 1000)  # ms

                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                fps = 1000 / avg_time * batch_size

                inference_results[model_key] = {
                    'name': model_info['name'],
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'fps': fps,
                    'output_shape': list(output.shape),
                    'status': 'success'
                }

                print(f"      ⏱️  Avg time: {avg_time:.2f}±{std_time:.2f}ms")
                print(f"      🎥 FPS: {fps:.1f}")

            except Exception as e:
                print(f"      ❌ Inference failed: {e}")
                inference_results[model_key] = {
                    'name': model_info['name'],
                    'status': 'failed',
                    'error': str(e)
                }

        self.results['inference'] = inference_results
        return inference_results

    def create_comparison_chart(self):
        """创建性能对比图表"""

        print(f"\n📊 Creating comparison charts...")

        if 'training' not in self.results or 'inference' not in self.results:
            print("   ⚠️ No results to plot")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            'MotionAGFormer + MambaGCN Performance Comparison', fontsize=16)

        # 1. 训练损失对比
        ax1 = axes[0, 0]
        successful_models = {k: v for k, v in self.results['training'].items(
        ) if v.get('status') == 'success'}

        if successful_models:
            model_names = [v['name'] for v in successful_models.values()]
            final_losses = [v['final_loss']
                            for v in successful_models.values()]

            bars1 = ax1.bar(range(len(model_names)), final_losses, color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
            ax1.set_title('Final Training Loss')
            ax1.set_ylabel('MSE Loss')
            ax1.set_xticks(range(len(model_names)))
            ax1.set_xticklabels([name.split(
                '+')[0].strip() if '+' in name else name for name in model_names], rotation=45)

            # 添加数值标签
            for bar, loss in zip(bars1, final_losses):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

        # 2. 参数量对比
        ax2 = axes[0, 1]
        if successful_models:
            params = [v['params'] /
                      1000 for v in successful_models.values()]  # in K
            bars2 = ax2.bar(range(len(model_names)), params, color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
            ax2.set_title('Model Parameters')
            ax2.set_ylabel('Parameters (K)')
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels([name.split(
                '+')[0].strip() if '+' in name else name for name in model_names], rotation=45)

            for bar, param in zip(bars2, params):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                         f'{param:.0f}K', ha='center', va='bottom', fontsize=9)

        # 3. 推理速度对比
        ax3 = axes[1, 0]
        successful_inference = {
            k: v for k, v in self.results['inference'].items() if v.get('status') == 'success'}

        if successful_inference:
            inf_names = [v['name'] for v in successful_inference.values()]
            inf_times = [v['avg_time_ms']
                         for v in successful_inference.values()]
            bars3 = ax3.bar(range(len(inf_names)), inf_times, color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'][:len(inf_names)])
            ax3.set_title('Inference Time')
            ax3.set_ylabel('Time (ms)')
            ax3.set_xticks(range(len(inf_names)))
            ax3.set_xticklabels([name.split(
                '+')[0].strip() if '+' in name else name for name in inf_names], rotation=45)

            for bar, time_ms in zip(bars3, inf_times):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{time_ms:.1f}ms', ha='center', va='bottom', fontsize=9)

        # 4. 性能改进对比
        ax4 = axes[1, 1]
        if successful_models:
            improvements = [v['loss_improvement']
                            for v in successful_models.values()]
            bars4 = ax4.bar(range(len(model_names)), improvements, color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
            ax4.set_title('Loss Improvement')
            ax4.set_ylabel('Improvement (%)')
            ax4.set_xticks(range(len(model_names)))
            ax4.set_xticklabels([name.split(
                '+')[0].strip() if '+' in name else name for name in model_names], rotation=45)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            for bar, imp in zip(bars4, improvements):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{imp:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存图表
        chart_path = f"demo_comparison_{self.demo_time}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"   ✅ Chart saved: {chart_path}")
        return chart_path

    def print_summary(self):
        """打印演示总结"""

        print(f"\n" + "="*60)
        print(f"🎯 Mamba-GCN Demo Summary")
        print(f"="*60)

        if 'training' in self.results:
            print(f"\n📊 Training Results:")
            for model_key, result in self.results['training'].items():
                if result.get('status') == 'success':
                    print(f"   ✅ {result['name']}")
                    print(f"      Parameters: {result['params']:,}")
                    print(f"      Final Loss: {result['final_loss']:.6f}")
                    print(
                        f"      Improvement: {result['loss_improvement']:.2f}%")
                    print(
                        f"      Training Time: {result['training_time']:.2f}s")
                else:
                    print(
                        f"   ❌ {result['name']}: {result.get('error', 'Failed')}")

        if 'inference' in self.results:
            print(f"\n⚡ Inference Results:")
            for model_key, result in self.results['inference'].items():
                if result.get('status') == 'success':
                    print(f"   ✅ {result['name']}")
                    print(f"      Avg Time: {result['avg_time_ms']:.2f}ms")
                    print(f"      FPS: {result['fps']:.1f}")
                    print(f"      Output: {result['output_shape']}")
                else:
                    print(
                        f"   ❌ {result['name']}: {result.get('error', 'Failed')}")

        # 最佳性能
        if 'training' in self.results and 'inference' in self.results:
            successful_training = {
                k: v for k, v in self.results['training'].items() if v.get('status') == 'success'}
            successful_inference = {
                k: v for k, v in self.results['inference'].items() if v.get('status') == 'success'}

            if successful_training and successful_inference:
                # 最低损失
                best_loss_model = min(
                    successful_training.values(), key=lambda x: x['final_loss'])
                print(f"\n🏆 Best Performance:")
                print(
                    f"   🎯 Lowest Loss: {best_loss_model['name']} ({best_loss_model['final_loss']:.6f})")

                # 最快推理
                fastest_model = min(
                    successful_inference.values(), key=lambda x: x['avg_time_ms'])
                print(
                    f"   ⚡ Fastest Inference: {fastest_model['name']} ({fastest_model['avg_time_ms']:.1f}ms)")

                # 最佳改进
                best_improvement = max(
                    successful_training.values(), key=lambda x: x['loss_improvement'])
                print(
                    f"   📈 Best Improvement: {best_improvement['name']} ({best_improvement['loss_improvement']:.2f}%)")

        print(f"\n✨ Demo completed successfully!")
        print(f"   Time: {self.demo_time}")
        print(f"   Device: {self.device}")


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description='MotionAGFormer + MambaGCN Demo')
    parser.add_argument('--config', type=str, default='all', choices=['baseline', 'mamba_gcn', 'mamba_gcn_full', 'all'],
                        help='Model configuration to demo')
    parser.add_argument('--quick', action='store_true',
                        help='Quick demo (1 epoch)')
    parser.add_argument('--full', action='store_true',
                        help='Full demo (more epochs)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--frames', type=int, default=27,
                        help='Number of frames')

    args = parser.parse_args()

    # 确定训练轮数
    if args.quick:
        epochs = 1
    elif args.full:
        epochs = 5
    else:
        epochs = 2

    print(f"🚀 Starting Mamba-GCN Demo")
    print(f"   Config: {args.config}")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {args.device}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Frames: {args.frames}")

    # 创建演示实例
    demo = MambaGCNDemo(device=args.device)

    # 创建演示数据
    dataloader = demo.create_demo_data(
        batch_size=args.batch_size,
        n_frames=args.frames,
        n_joints=17
    )

    # 创建模型
    models = demo.create_models(n_frames=args.frames)

    # 过滤模型（如果指定了特定配置）
    if args.config != 'all':
        models = {args.config: models.get(args.config)}

    # 验证是否有可用模型
    available_models = {k: v for k, v in models.items() if v is not None}
    if not available_models:
        print("❌ No models available for demo!")
        return

    # 快速训练
    training_results = demo.quick_training(
        available_models, dataloader, epochs=epochs)

    # 推理基准测试
    inference_results = demo.inference_benchmark(available_models,
                                                 batch_size=args.batch_size,
                                                 n_frames=args.frames)

    # 创建对比图表
    try:
        demo.create_comparison_chart()
    except Exception as e:
        print(f"   ⚠️ Chart creation failed: {e}")

    # 打印总结
    demo.print_summary()


if __name__ == "__main__":
    main()
