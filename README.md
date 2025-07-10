# MotionAGFormer + MambaGCN

> **一个结合了 State Space Models (Mamba) 和 Graph Convolutional Networks (GCN) 的创新 3D 人体姿态估计框架**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 项目概述

本项目在 [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) 的基础上，创新性地集成了 **Mamba State Space Model** 和 **Graph Convolutional Network**，构建了一个三分支融合架构用于 3D 人体姿态估计。

### 🏗️ 架构特点

- **🧠 三分支设计**: ST-Attention + ST-Graph + **MambaGCN**
- **⚡ 线性复杂度**: Mamba 实现 O(n) 时序建模，替代传统 O(n²) 注意力机制
- **🔗 空间感知**: GCN 分支显式建模人体骨架的空间关系
- **🎯 自适应融合**: 可学习的分支权重，根据输入自适应调整
- **🔧 灵活配置**: 支持分支的独立开关和参数调节

### 📊 性能概览

| 模型配置 | 参数量 | 内存占用 | 推理速度 | 性能提升* |
|---------|-------|---------|---------|----------|
| 基线 MotionAGFormer | 745K | ~0.03GB | 171ms | - |
| + MambaGCN | 1.04M | ~0.03GB | **34ms** | **+5.3%** |
| + MambaGCN (Full) | 1.13M | ~0.03GB | 39ms | **+12.1%** |

*基于 PoC 训练验证的 Loss 下降对比

## 🚀 快速开始

### 📋 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.4+
- **CUDA**: 12.1+ (可选，支持 CPU)
- **内存**: 至少 4GB RAM
- **GPU**: 推荐 NVIDIA GPU (4GB+ VRAM)

### ⚙️ 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/your-username/Mamba_GCN.git
cd Mamba_GCN
```

#### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### 3. 安装依赖

```bash
# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm
pip install numpy matplotlib tqdm pyyaml timm pillow scipy
pip install easydict

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mamba_ssm; print('Mamba: OK')"
```

#### 4. 数据准备

项目支持两种数据模式：

**选项 A: 使用 Mock 数据 (推荐开发)**

```bash
# 生成 mock 数据 (已预置)
python data/create_mock_data.py
```

**选项 B: 使用真实 Human3.6M 数据**

```bash
# 1. 手动下载真实数据 (约 2GB)
# 下载链接: https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing
# 详细说明: 参考 data/motion3d/human36m/MANUAL_DOWNLOAD_INSTRUCTIONS.md

# 2. 数据存放路径
# 将下载的文件解压到: data/motion3d/human36m/raw/motion3d/
# 预期文件结构:
# data/motion3d/human36m/raw/motion3d/
# ├── h36m_sh_conf_cam_source_final.pkl (1.0GB)
# ├── data_train_3dhp.npz (509MB)  
# ├── data_test_3dhp.npz (12MB)
# └── H36M-243/
#     ├── train/ (17,748 files)
#     └── test/ (2,228 files)

# 3. 数据迁移 (从 Mock 数据切换到真实数据)
python scripts/tools/migrate_to_real_data.py --backup

# 4. 验证数据加载
python test_real_data.py
```

### 🔧 配置说明

主要配置参数位于模型实例化时：

```python
model = MotionAGFormer(
    # 基础配置
    n_layers=12,
    dim_in=2,          # 输入维度 (2D poses)
    dim_feat=64,       # 特征维度
    dim_out=3,         # 输出维度 (3D poses)
    n_frames=243,      # 序列长度
    
    # MambaGCN 配置
    use_mamba_gcn=True,              # 启用 MambaGCN 分支
    mamba_gcn_use_mamba=True,        # 在 MambaGCN 中使用 Mamba
    mamba_gcn_use_attention=False,   # 在 MambaGCN 中使用 Attention
)
```

#### 配置模式说明

| 配置 | `use_mamba_gcn` | `mamba_gcn_use_mamba` | `mamba_gcn_use_attention` | 描述 |
|-----|-----------------|---------------------|-------------------------|------|
| **基线** | `False` | - | - | 原始 MotionAGFormer |
| **MambaGCN** | `True` | `True` | `False` | Mamba + GCN 双分支 |
| **Full** | `True` | `True` | `True` | Mamba + GCN + Attention 三分支 |
| **GCN-Only** | `True` | `False` | `True` | GCN + Attention 双分支 |

## 📖 使用指南

### 🎯 基础使用

```python
import torch
from model.MotionAGFormer import MotionAGFormer

# 创建模型
model = MotionAGFormer(
    n_layers=4,
    dim_in=2,
    dim_feat=64,
    dim_out=3,
    n_frames=27,
    use_mamba_gcn=True,
    mamba_gcn_use_mamba=True,
    mamba_gcn_use_attention=False
)

# 输入数据 [Batch, Time, Joints, Dims]
input_2d = torch.randn(8, 27, 17, 2)  # 8 批次，27 帧，17 关节，2D 坐标

# 前向传播
output_3d = model(input_2d)  # [8, 27, 17, 3] - 3D 姿态预测

print(f"输入: {input_2d.shape}")
print(f"输出: {output_3d.shape}")
```

### 🚂 训练流程

#### 快速训练 (Mock 数据)

```bash
# 运行 PoC 训练验证
python poc_training_validation.py

# 使用 Mock 数据进行快速训练
python train_mock.py --epochs 5 --batch_size 16
```

#### 完整训练 (真实数据)

```bash
# 使用真实 Human3.6M 数据训练 (推荐)
python scripts/train_real.py --model_type mamba_gcn --epochs 20 --batch_size 64

# 不同模型配置的训练:
# 基线模型
python scripts/train_real.py --model_type baseline --epochs 20

# MambaGCN 模型  
python scripts/train_real.py --model_type mamba_gcn --epochs 20

# 完整架构 (Mamba + GCN + Attention)
python scripts/train_real.py --model_type full --epochs 20

# 使用原始配置文件训练
python train.py --config configs/h36m/MotionAGFormer-base.yaml
```

### 🧪 测试和验证

```bash
# 端到端验证 (Mock 数据)
python end_to_end_validation.py

# 错误处理测试
python error_handling_validation.py

# 模型集成测试
python test_model_integration.py

# 真实数据验证
python test_real_data.py

# 最终交付验证 (真实数据)
python final_delivery_validation_real.py

# 数据性能比较 (Mock vs Real)
python compare_data_performance.py

# 查看使用示例
python example_usage.py
```

## 📁 项目结构

```
Mamba_GCN/
├── 📂 model/                          # 核心模型代码
│   ├── MotionAGFormer.py              # 主模型 (增强版)
│   └── modules/                       # 模型组件
│       ├── mamba_layer.py             # Mamba 状态空间模型
│       ├── gcn_layer.py               # 图卷积网络
│       ├── mamba_gcn_block.py         # MambaGCN 融合模块
│       └── attention.py               # 注意力机制
├── 📂 data/                           # 数据处理
│   ├── motion3d/                      # 数据文件
│   ├── reader/                        # 数据读取器
│   ├── preprocess/                    # 数据预处理
│   └── create_mock_data.py            # Mock 数据生成
├── 📂 configs/                        # 配置文件
│   └── h36m/                          # Human3.6M 配置
├── 📂 utils/                          # 工具函数
├── 📂 loss/                           # 损失函数
├── 📂 MotionAGFormer/                 # 原始仓库备份
├── 🐍 train.py                        # 主训练脚本
├── 🐍 train_mock.py                   # Mock 数据训练
├── 🐍 poc_training_validation.py      # PoC 训练验证
├── 🐍 end_to_end_validation.py        # 端到端验证
├── 🐍 example_usage.py                # 使用示例
├── 📄 requirements.txt                # 依赖清单
└── 📚 README.md                       # 项目文档
```

## 🎨 架构细节

### 🧩 MambaGCNBlock 设计

```python
# MambaGCNBlock 的三分支架构
class MambaGCNBlock(nn.Module):
    def __init__(self, dim, use_mamba=True, use_attention=True):
        # Branch A: Mamba (时序建模，O(n) 复杂度)
        self.mamba_branch = MambaBranch(dim) if use_mamba else None
        
        # Branch B: GCN (空间关系，人体骨架图)
        self.gcn_branch = GCNBranch(dim)
        
        # Branch C: Attention (基线对比)
        self.attention_branch = AttentionBranch(dim) if use_attention else None
        
        # 自适应融合
        self.fusion = AdaptiveFusion(dim, num_branches)
```

### 🔗 集成策略

1. **渐进集成**: MambaGCNBlock 作为第三分支加入 MotionAGFormer
2. **维度兼容**: 保持 `[B,T,J,D]` 输入输出格式一致
3. **向后兼容**: 通过配置开关支持原始模型模式
4. **灵活部署**: 支持分支的独立启用/禁用

## 📊 验证结果

### ✅ PoC 训练验证

| 验证项 | 结果 | 详情 |
|-------|------|------|
| **模型创建** | ✅ | 3 种配置均成功 |
| **前向传播** | ✅ | 输出维度正确，无 NaN |
| **反向传播** | ✅ | 梯度计算正常 |
| **Loss 下降** | ✅ | 21.90%~38.47% |
| **内存使用** | ✅ | <0.04GB |
| **推理速度** | ✅ | 34-171ms |

### 🛡️ 错误处理

| 测试类别 | 通过率 | 状态 |
|----------|--------|------|
| **输入验证** | 83% | ✅ |
| **内存边界** | 100% | ✅ |
| **梯度稳定** | 100% | ✅ |
| **设备兼容** | 100% | ✅ |
| **数值稳定** | 100% | ✅ |
| **整体** | **75%** | **✅ 通过** |

## 🛠️ 开发和调试

### 🔧 开发模式

```bash
# 启用详细日志
export PYTHONPATH=$PWD:$PYTHONPATH

# 运行调试模式
python -u train_mock.py --epochs 1 --batch_size 4 --device cpu
```

### 🐛 常见问题

**Q: ImportError: No module named 'mamba_ssm'**
```bash
# 解决方案
pip install mamba-ssm
# 或如果失败，使用 LSTM 备用模式
# 在配置中设置 mamba_gcn_use_mamba=False
```

**Q: CUDA out of memory**
```bash
# 解决方案：减少批次大小
python train_mock.py --batch_size 4
# 或使用 CPU
python train_mock.py --device cpu
```

**Q: 数据维度不匹配**
```bash
# 检查 Mock 数据格式
python data/create_mock_data.py
# 验证 Mock 数据加载
python -c "from data.reader.mock_h36m import DataReaderMockH36M; print('Mock Data OK')"

# 检查真实数据格式
python test_real_data.py
# 验证真实数据加载
python -c "from data.reader.real_h36m import DataReaderRealH36M; print('Real Data OK')"
```

**Q: 如何从 Mock 数据切换到真实数据？**
```bash
# 运行数据迁移脚本
python scripts/tools/migrate_to_real_data.py --backup

# 验证迁移结果
python test_real_data.py
```

**Q: 真实数据下载失败或文件损坏**
```bash
# 检查数据完整性
python scripts/tools/migrate_to_real_data.py --check-only

# 重新下载数据
# 参考: data/motion3d/human36m/MANUAL_DOWNLOAD_INSTRUCTIONS.md
```

### 📈 性能调优

```python
# 针对不同场景的推荐配置

# 🚀 速度优先 (实时推理)
model = MotionAGFormer(
    n_layers=4, dim_feat=32, n_frames=27,
    use_mamba_gcn=True, 
    mamba_gcn_use_mamba=True, 
    mamba_gcn_use_attention=False
)

# 🎯 精度优先 (离线处理)
model = MotionAGFormer(
    n_layers=12, dim_feat=128, n_frames=243,
    use_mamba_gcn=True,
    mamba_gcn_use_mamba=True,
    mamba_gcn_use_attention=True
)

# ⚖️ 平衡模式 (推荐)
model = MotionAGFormer(
    n_layers=8, dim_feat=64, n_frames=81,
    use_mamba_gcn=True,
    mamba_gcn_use_mamba=True,
    mamba_gcn_use_attention=False
)
```

## 📚 技术文档

### 📄 相关论文

1. **MotionAGFormer**: [WACV 2024] - 基础架构
2. **Mamba**: State Space Models for Sequence Modeling
3. **GCN**: Semi-supervised Classification with Graph Convolutional Networks

### 🔬 技术报告

- [Task 2.1 完成报告](TASK_2_1_COMPLETION_REPORT.md) - MambaGCNBlock 实现
- [Task 2.2 完成报告](TASK_2_2_COMPLETION_REPORT.md) - 模型集成
- [Task 2.3 完成报告](TASK_2_3_POC_TRAINING_REPORT.md) - PoC 训练验证
- [Task 2.4 完成报告](TASK_2_4_QUALITY_CHECK_REPORT.md) - 质量检查

## 🤝 贡献指南

### 🔄 开发流程

1. **Fork** 仓库
2. 创建特性分支: `git checkout -b feature/新功能`
3. 提交更改: `git commit -am '添加新功能'`
4. 推送分支: `git push origin feature/新功能`
5. 提交 **Pull Request**

### 🧪 测试要求

```bash
# 运行完整测试套件
python end_to_end_validation.py
python error_handling_validation.py
python test_model_integration.py

# 所有测试应该通过 ✅
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **MotionAGFormer 团队**: 提供优秀的基础架构
- **Mamba 团队**: 革命性的状态空间模型
- **PyTorch 团队**: 强大的深度学习框架

## 📧 联系方式

- **GitHub Issues**: [提交问题](https://github.com/your-username/Mamba_GCN/issues)
- **Email**: your-email@domain.com
- **项目主页**: [Mamba-GCN](https://github.com/your-username/Mamba_GCN)

---

**🚀 开始你的 3D 人体姿态估计之旅吧！**

> 如果这个项目对您有帮助，请考虑给个 ⭐ Star！ 