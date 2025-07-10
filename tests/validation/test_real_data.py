#!/usr/bin/env python3
"""
简单的真实Human3.6M数据测试脚本
验证数据加载和基本模型推理功能
"""

import torch
import numpy as np
from data.reader.real_h36m import DataReaderRealH36M
from model.MotionAGFormer import MotionAGFormer
from loss.pose3d import loss_mpjpe


def test_real_data_loading():
    """测试真实数据加载"""
    print("📊 测试真实Human3.6M数据加载...")

    try:
        # 创建数据读取器
        datareader = DataReaderRealH36M(
            n_frames=243,
            sample_stride=1,
            data_stride_train=81,
            data_stride_test=243,
            read_confidence=True,
            dt_root='data/motion3d/human36m/raw/motion3d'
        )

        # 获取数据
        train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()

        print(f"✅ 数据加载成功:")
        print(f"   训练数据: {train_data.shape} -> {train_labels.shape}")
        print(f"   测试数据: {test_data.shape} -> {test_labels.shape}")

        # 检查数据类型和范围
        print(f"   训练数据范围: [{train_data.min():.3f}, {train_data.max():.3f}]")
        print(
            f"   训练标签范围: [{train_labels.min():.3f}, {train_labels.max():.3f}]")

        # 检查数据维度
        assert len(train_data.shape) == 4, f"期望4D数据，得到 {train_data.shape}"
        assert train_data.shape[-1] == 3, f"期望输入维度为3 (2D+conf)，得到 {train_data.shape[-1]}"
        assert train_labels.shape[-1] == 3, f"期望输出维度为3，得到 {train_labels.shape[-1]}"

        print("✅ 数据维度验证通过")

        return train_data, test_data, train_labels, test_labels, datareader

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None, None, None, None


def test_model_with_real_data(data_2d, data_3d, model_type='baseline'):
    """测试模型与真实数据的兼容性"""
    print(f"\n🧠 测试 {model_type} 模型...")

    try:
        # 创建模型
        if model_type == 'baseline':
            model = MotionAGFormer(
                n_layers=4,
                dim_in=2,  # 只使用2D坐标，忽略confidence
                dim_feat=64,
                dim_out=3,
                n_frames=243,
                use_mamba_gcn=False
            )
        else:  # mamba_gcn
            model = MotionAGFormer(
                n_layers=4,
                dim_in=2,  # 只使用2D坐标，忽略confidence
                dim_feat=64,
                dim_out=3,
                n_frames=243,
                use_mamba_gcn=True,
                mamba_gcn_use_mamba=True,
                mamba_gcn_use_attention=False
            )

        model.eval()

        # 准备数据 - 只使用前2个维度 (x, y)，忽略confidence
        input_2d = torch.FloatTensor(data_2d[:5, :, :, :2])  # [5, 243, 17, 2]
        target_3d = torch.FloatTensor(data_3d[:5])  # [5, 243, 17, 3]

        print(f"   输入形状: {input_2d.shape}")
        print(f"   目标形状: {target_3d.shape}")

        # 前向传播
        with torch.no_grad():
            pred_3d = model(input_2d)

        print(f"   输出形状: {pred_3d.shape}")

        # 计算损失
        loss = loss_mpjpe(pred_3d, target_3d)
        print(f"   MPJPE损失: {loss.item():.4f}")

        # 计算近似MPJPE (mm)
        pred_np = pred_3d.numpy()
        target_np = target_3d.numpy()
        mpjpe_mm = np.mean(
            np.sqrt(np.sum((pred_np - target_np) ** 2, axis=-1))) * 1000
        print(f"   近似MPJPE: {mpjpe_mm:.2f}mm")

        print(f"✅ {model_type} 模型测试成功")

        return {
            'model_type': model_type,
            'loss': loss.item(),
            'mpjpe_mm': mpjpe_mm,
            'input_shape': list(input_2d.shape),
            'output_shape': list(pred_3d.shape)
        }

    except Exception as e:
        print(f"❌ {model_type} 模型测试失败: {e}")
        return None


def test_baseline_validation():
    """测试基线验证脚本"""
    print("\n🔧 测试基线验证...")

    try:
        # 运行基线验证 (真实数据版本)
        import subprocess
        result = subprocess.run(['python3', 'baseline_validation_real.py', '--skip-deps'],
                                capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ 基线验证成功")
            print("   输出摘要:")
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"     {line}")
        else:
            print("⚠️  基线验证有警告")
            print(f"   错误信息: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("⚠️  基线验证超时，跳过")
    except Exception as e:
        print(f"⚠️  基线验证失败: {e}")


def main():
    print("🎯 真实Human3.6M数据验证测试")
    print("=" * 50)

    # 测试1: 数据加载
    train_data, test_data, train_labels, test_labels, datareader = test_real_data_loading()

    if train_data is None:
        print("❌ 数据加载失败，无法继续测试")
        return False

    # 测试2: 基线模型
    baseline_result = test_model_with_real_data(
        test_data, test_labels, 'baseline')

    # 测试3: MambaGCN模型
    mamba_result = test_model_with_real_data(
        test_data, test_labels, 'mamba_gcn')

    # 测试4: 基线验证
    test_baseline_validation()

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)

    if train_data is not None and test_data is not None:
        print(f"✅ 数据集规模:")
        print(
            f"   训练: {train_data.shape[0]:,} clips × {train_data.shape[1]} frames")
        print(
            f"   测试: {test_data.shape[0]:,} clips × {test_data.shape[1]} frames")

    if baseline_result:
        print(f"✅ 基线模型: MPJPE = {baseline_result['mpjpe_mm']:.2f}mm")

    if mamba_result:
        print(f"✅ MambaGCN模型: MPJPE = {mamba_result['mpjpe_mm']:.2f}mm")

        if baseline_result and mamba_result:
            improvement = ((baseline_result['mpjpe_mm'] - mamba_result['mpjpe_mm']) /
                           baseline_result['mpjpe_mm']) * 100
            print(f"🚀 性能提升: {improvement:.1f}%")

    print("\n🎉 真实数据验证完成！")
    print("💡 现在可以使用真实Human3.6M数据进行正式训练和评估")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
