"""
MambaGCNBlock Implementation for 3D Pose Estimation
==================================================
Mamba-GCN 混合架构的核心模块
三分支融合：Mamba (时序) + GCN (空间) + Attention (基线)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_layer import MambaBranch
from .gcn_layer import GCNBranch


class AttentionBranch(nn.Module):
    """
    Attention 分支：作为基线对比
    使用简化的自注意力机制
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Multi-head attention layers
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - batch, time, joints, channels
        Returns:
            y: [B, T, J, C] - attention enhanced features
        """
        B, T, J, C = x.shape
        identity = x

        # Reshape for attention: [B, T, J, C] → [B*J, T, C]
        x_reshaped = x.view(B * J, T, C)

        # Multi-head attention
        q = self.q_proj(x_reshaped).view(
            B * J, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_reshaped).view(
            B * J, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_reshaped).view(
            B * J, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [B*J, num_heads, T, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(B * J, T, C)

        # Output projection
        output = self.out_proj(attn_output)

        # Reshape back: [B*J, T, C] → [B, T, J, C]
        output = output.view(B, T, J, C)

        # Residual connection + normalization
        output = self.norm(output + identity)

        return output


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块：动态融合三个分支的输出
    使用可学习的权重进行加权融合
    """

    def __init__(self, dim, num_branches=3):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches

        # 分支权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_branches),
            nn.Softmax(dim=-1)
        )

        # 特征融合层
        self.fusion_proj = nn.Linear(dim * num_branches, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, branch_outputs, input_features):
        """
        Args:
            branch_outputs: List of [B, T, J, C] from each branch
            input_features: [B, T, J, C] original input for weight generation
        Returns:
            fused_output: [B, T, J, C] fused features
        """
        B, T, J, C = input_features.shape

        # 生成自适应权重：基于输入特征的全局平均
        global_feat = input_features.mean(dim=(1, 2))  # [B, C]
        branch_weights = self.weight_generator(
            global_feat)  # [B, num_branches]

        # 堆叠分支输出：[B, T, J, C, num_branches]
        stacked_outputs = torch.stack(branch_outputs, dim=-1)

        # 扩展权重维度以匹配 stacked_outputs
        # [B, num_branches] → [B, 1, 1, 1, num_branches]
        weights = branch_weights.view(B, 1, 1, 1, self.num_branches)

        # 加权融合
        weighted_outputs = stacked_outputs * \
            weights  # [B, T, J, C, num_branches]
        weighted_sum = weighted_outputs.sum(dim=-1)   # [B, T, J, C]

        # 残差连接 + 层归一化
        output = self.norm(weighted_sum + input_features)

        return output, branch_weights


class MambaGCNBlock(nn.Module):
    """
    MambaGCN 核心模块：三分支融合架构

    Architecture:
    Input [B,T,J,C] 
    ├── Branch A: Mamba (时序建模)
    ├── Branch B: GCN (空间关系) 
    ├── Branch C: Attention (基线对比)
    └── Adaptive Fusion → Output [B,T,J,C]
    """

    def __init__(self, dim, num_joints=17, use_mamba=True, use_attention=True):
        """
        Args:
            dim: 特征维度
            num_joints: 关节数量 (默认17)
            use_mamba: 是否使用 Mamba 分支
            use_attention: 是否使用 Attention 分支
        """
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.use_mamba = use_mamba
        self.use_attention = use_attention

        # 分支模块
        self.branches = nn.ModuleDict()
        self.branch_names = []

        # Branch A: Mamba 时序分支
        if use_mamba:
            self.branches['mamba'] = MambaBranch(
                dim, num_joints, use_mamba=True)
            self.branch_names.append('mamba')

        # Branch B: GCN 空间分支 (必须包含)
        self.branches['gcn'] = GCNBranch(dim, num_joints)
        self.branch_names.append('gcn')

        # Branch C: Attention 基线分支
        if use_attention:
            self.branches['attention'] = AttentionBranch(dim)
            self.branch_names.append('attention')

        # 自适应融合模块
        num_branches = len(self.branch_names)
        self.fusion = AdaptiveFusion(dim, num_branches)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - input pose features
        Returns:
            output: [B, T, J, C] - enhanced pose features
            info: dict with branch outputs and fusion weights
        """
        identity = x

        # 通过各分支处理
        branch_outputs = []
        branch_info = {}

        for branch_name in self.branch_names:
            branch_output = self.branches[branch_name](x)
            branch_outputs.append(branch_output)
            branch_info[f'{branch_name}_output'] = branch_output

        # 自适应融合
        fused_output, fusion_weights = self.fusion(branch_outputs, x)

        # 输出投影
        projected_output = self.output_proj(fused_output)

        # 最终残差连接 + 层归一化
        final_output = self.final_norm(projected_output + identity)

        # 返回详细信息用于分析
        info = {
            'branch_info': branch_info,
            'fusion_weights': fusion_weights,
            'fused_output': fused_output,
            'num_branches': len(self.branch_names),
            'branch_names': self.branch_names
        }

        return final_output, info

    def get_model_info(self):
        """获取模型配置信息"""
        return {
            'dim': self.dim,
            'num_joints': self.num_joints,
            'use_mamba': self.use_mamba,
            'use_attention': self.use_attention,
            'branch_names': self.branch_names,
            'num_branches': len(self.branch_names)
        }


def test_mamba_gcn_block():
    """测试完整的 MambaGCNBlock"""
    print("🧪 测试 MambaGCNBlock...")

    # 创建测试数据
    batch_size, time_steps, num_joints, dim = 2, 81, 17, 128
    x = torch.randn(batch_size, time_steps, num_joints, dim)
    print(f"输入数据形状: {x.shape}")

    # 测试不同配置
    configs = [
        {"use_mamba": True, "use_attention": True, "name": "完整三分支"},
        {"use_mamba": True, "use_attention": False, "name": "Mamba+GCN"},
        {"use_mamba": False, "use_attention": True, "name": "GCN+Attention"},
    ]

    for config in configs:
        print(f"\n🔧 测试配置: {config['name']}")

        try:
            # 创建模型
            model = MambaGCNBlock(
                dim=dim,
                num_joints=num_joints,
                use_mamba=config['use_mamba'],
                use_attention=config['use_attention']
            )

            # 前向传播
            output, info = model(x)

            print(f"✅ 输出形状: {output.shape}")
            print(f"✅ 分支数量: {info['num_branches']}")
            print(f"✅ 分支名称: {info['branch_names']}")
            print(f"✅ 融合权重形状: {info['fusion_weights'].shape}")

            # 梯度测试
            loss = output.sum()
            loss.backward()
            print("✅ 梯度计算正常")

            # 模型信息
            model_info = model.get_model_info()
            print(f"✅ 模型配置: {model_info}")

        except Exception as e:
            print(f"❌ 配置 {config['name']} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 MambaGCNBlock 测试完成!")


if __name__ == "__main__":
    test_mamba_gcn_block()
