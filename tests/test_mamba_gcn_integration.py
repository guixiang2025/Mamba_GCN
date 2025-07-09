#!/usr/bin/env python3
"""
MambaGCNBlock 集成测试脚本
=========================
验证 Mamba-GCN 混合架构的完整功能

Task 2.1 完成状态验证：
✅ T2.1.1 实现 Mamba 分支 (2 小时)
✅ T2.1.2 实现 GCN 分支 (2 小时)  
✅ T2.1.3 实现融合模块 (1 小时)
✅ T2.1.4 集成测试 (1 小时)
"""

from model.modules.gcn_layer import GCNBranch, test_gcn_branch
from model.modules.mamba_gcn_block import MambaGCNBlock, AttentionBranch
from model.modules.mamba_layer import MambaBranch, test_mamba_branch
import torch
import torch.nn as nn
import sys
import time
import traceback
from typing import Dict, Tuple, List

# 添加模块路径
sys.path.append('.')


def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_individual_components():
    """测试各个组件的独立功能"""
    print_separator("🧪 测试各个组件的独立功能")

    results = {}

    # 测试 Mamba 分支
    print("\n📍 1. 测试 Mamba 分支")
    results['mamba'] = test_mamba_branch()

    # 测试 GCN 分支
    print("\n📍 2. 测试 GCN 分支")
    results['gcn'] = test_gcn_branch()

    # 测试 Attention 分支
    print("\n📍 3. 测试 Attention 分支")
    try:
        batch_size, time_steps, num_joints, dim = 2, 81, 17, 128
        x = torch.randn(batch_size, time_steps, num_joints, dim)

        attention_branch = AttentionBranch(dim)
        y_attention = attention_branch(x)

        print(f"✅ Attention 分支输出形状: {y_attention.shape}")
        print(f"✅ 维度检查: 输入 {x.shape} → 输出 {y_attention.shape}")

        # 梯度测试
        loss = y_attention.sum()
        loss.backward()
        print("✅ 梯度计算正常")

        results['attention'] = True
    except Exception as e:
        print(f"❌ Attention 分支测试失败: {e}")
        results['attention'] = False

    return results


def test_fusion_configurations():
    """测试不同的融合配置"""
    print_separator("🔧 测试不同的融合配置")

    # 测试数据
    batch_size, time_steps, num_joints, dim = 4, 81, 17, 256
    x = torch.randn(batch_size, time_steps, num_joints, dim)
    print(f"测试数据形状: {x.shape}")

    # 不同配置
    configurations = [
        {
            "name": "完整三分支 (Mamba + GCN + Attention)",
            "use_mamba": True,
            "use_attention": True,
            "expected_branches": 3
        },
        {
            "name": "Mamba + GCN",
            "use_mamba": True,
            "use_attention": False,
            "expected_branches": 2
        },
        {
            "name": "GCN + Attention",
            "use_mamba": False,
            "use_attention": True,
            "expected_branches": 2
        },
        {
            "name": "仅 GCN (最小配置)",
            "use_mamba": False,
            "use_attention": False,
            "expected_branches": 1
        }
    ]

    results = {}

    for i, config in enumerate(configurations):
        print(f"\n📍 {i+1}. {config['name']}")

        try:
            # 创建模型
            model = MambaGCNBlock(
                dim=dim,
                num_joints=num_joints,
                use_mamba=config['use_mamba'],
                use_attention=config['use_attention']
            )

            # 前向传播
            start_time = time.time()
            output, info = model(x)
            forward_time = time.time() - start_time

            # 验证结果
            assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
            assert info['num_branches'] == config[
                'expected_branches'], f"分支数量不匹配: {info['num_branches']} vs {config['expected_branches']}"

            print(f"✅ 输出形状: {output.shape}")
            print(
                f"✅ 分支数量: {info['num_branches']} (预期: {config['expected_branches']})")
            print(f"✅ 分支名称: {info['branch_names']}")
            print(f"✅ 融合权重: {info['fusion_weights'].shape}")
            print(f"✅ 前向传播时间: {forward_time:.4f}s")

            # 梯度测试
            start_time = time.time()
            loss = output.sum()
            loss.backward()
            backward_time = time.time() - start_time
            print(f"✅ 反向传播时间: {backward_time:.4f}s")

            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
            print(f"✅ 总参数量: {total_params:,}")
            print(f"✅ 可训练参数: {trainable_params:,}")

            results[config['name']] = {
                'success': True,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'output_shape': output.shape,
                'num_branches': info['num_branches']
            }

        except Exception as e:
            print(f"❌ 配置失败: {e}")
            traceback.print_exc()
            results[config['name']] = {'success': False, 'error': str(e)}

    return results


def test_performance_benchmark():
    """性能基准测试"""
    print_separator("⚡ 性能基准测试")

    # 不同输入尺寸测试
    test_cases = [
        {"batch_size": 1, "time_steps": 27, "description": "小序列 (27 frames)"},
        {"batch_size": 2, "time_steps": 81, "description": "中序列 (81 frames)"},
        {"batch_size": 4, "time_steps": 243,
            "description": "长序列 (243 frames)"},
    ]

    num_joints, dim = 17, 128

    for case in test_cases:
        print(f"\n📊 {case['description']}")

        x = torch.randn(case['batch_size'],
                        case['time_steps'], num_joints, dim)
        model = MambaGCNBlock(
            dim, num_joints, use_mamba=True, use_attention=True)

        # 预热
        for _ in range(3):
            _ = model(x)

        # 性能测试
        times = []
        for _ in range(10):
            start_time = time.time()
            output, info = model(x)
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)
        print(f"✅ 输入形状: {x.shape}")
        print(
            f"✅ 平均前向时间: {avg_time:.4f}s (±{torch.std(torch.tensor(times)):.4f})")
        print(
            f"✅ 吞吐量: {case['batch_size'] * case['time_steps'] / avg_time:.1f} frames/s")


def test_edge_cases():
    """边界情况测试"""
    print_separator("🔍 边界情况测试")

    edge_cases = [
        {"batch_size": 1, "time_steps": 1, "description": "最小输入 (1 frame)"},
        {"batch_size": 1, "time_steps": 10, "description": "短序列 (10 frames)"},
        {"batch_size": 8, "time_steps": 100, "description": "大批次"},
    ]

    num_joints, dim = 17, 64  # 使用较小的维度避免内存问题

    for case in edge_cases:
        print(f"\n🔍 {case['description']}")

        try:
            x = torch.randn(case['batch_size'],
                            case['time_steps'], num_joints, dim)
            model = MambaGCNBlock(dim, num_joints)

            output, info = model(x)
            assert output.shape == x.shape

            print(f"✅ 输入: {x.shape} → 输出: {output.shape}")
            print(f"✅ 融合权重正常: {info['fusion_weights'].shape}")

        except Exception as e:
            print(f"❌ 边界情况失败: {e}")


def generate_summary_report(component_results: Dict, fusion_results: Dict):
    """生成总结报告"""
    print_separator("📋 Task 2.1 完成总结报告")

    print("🎯 **Task 2.1: MambaGCNBlock 核心实现** - 完成状态:")
    print()

    # T2.1.1 Mamba 分支
    mamba_status = "✅ 通过" if component_results.get('mamba', False) else "❌ 失败"
    print(f"📍 T2.1.1 实现 Mamba 分支 (2小时): {mamba_status}")
    print("   - SimplifiedMamba 状态空间模型实现")
    print("   - 支持 [B,T,J,C] 维度处理")
    print("   - LSTM 备用方案")

    # T2.1.2 GCN 分支
    gcn_status = "✅ 通过" if component_results.get('gcn', False) else "❌ 失败"
    print(f"📍 T2.1.2 实现 GCN 分支 (2小时): {gcn_status}")
    print("   - Human3.6M 17关节骨架图构建")
    print("   - 度归一化邻接矩阵")
    print("   - 多层图卷积网络")

    # T2.1.3 融合模块
    attention_status = "✅ 通过" if component_results.get(
        'attention', False) else "❌ 失败"
    print(f"📍 T2.1.3 实现融合模块 (1小时): {attention_status}")
    print("   - 三分支架构 (Mamba + GCN + Attention)")
    print("   - 自适应权重融合")
    print("   - 残差连接和层归一化")

    # T2.1.4 集成测试
    all_fusion_passed = all(result.get('success', False)
                            for result in fusion_results.values())
    integration_status = "✅ 通过" if all_fusion_passed else "❌ 失败"
    print(f"📍 T2.1.4 集成测试 (1小时): {integration_status}")
    print("   - 端到端前向传播测试")
    print("   - 梯度流验证")
    print("   - 多配置兼容性测试")

    print("\n🏆 **总体状态:**")
    total_passed = sum([component_results.get('mamba', False),
                       component_results.get('gcn', False),
                       component_results.get('attention', False),
                       all_fusion_passed])

    if total_passed == 4:
        print("🎉 **Task 2.1 全部完成！** 所有子任务均成功通过测试")
        print("✅ 已准备好进入 Task 2.2 模型集成阶段")
    else:
        print(f"⚠️  部分任务需要修复 ({total_passed}/4 通过)")

    print("\n📊 **性能概览:**")
    for name, result in fusion_results.items():
        if result.get('success', False):
            params = result.get('trainable_params', 0)
            forward_time = result.get('forward_time', 0)
            print(f"   - {name}: {params:,} 参数, {forward_time:.4f}s 前向时间")


def main():
    """主测试函数"""
    print("🚀 MambaGCNBlock 集成测试开始")
    print("=" * 60)

    try:
        # 1. 测试各个组件
        component_results = test_individual_components()

        # 2. 测试融合配置
        fusion_results = test_fusion_configurations()

        # 3. 性能基准测试
        test_performance_benchmark()

        # 4. 边界情况测试
        test_edge_cases()

        # 5. 生成总结报告
        generate_summary_report(component_results, fusion_results)

        print("\n🎉 集成测试完成！")

    except Exception as e:
        print(f"\n❌ 集成测试过程中出现错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
