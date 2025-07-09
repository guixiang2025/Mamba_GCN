"""
GCN Layer Implementation for 3D Pose Estimation  
===============================================
基于图卷积网络的空间关节依赖建模模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Human36MGraph:
    """
    Human3.6M 数据集的 17 关节骨架图结构
    根据人体骨架的自然连接关系构建邻接矩阵
    """

    def __init__(self):
        # Human3.6M 17 关节定义
        self.joints = [
            'Hip',        # 0
            'RHip',       # 1
            'RKnee',      # 2
            'RFoot',      # 3
            'LHip',       # 4
            'LKnee',      # 5
            'LFoot',      # 6
            'Spine',      # 7
            'Thorax',     # 8
            'Nose',       # 9
            'Head',       # 10
            'LShoulder',  # 11
            'LElbow',     # 12
            'LWrist',     # 13
            'RShoulder',  # 14
            'RElbow',     # 15
            'RWrist'      # 16
        ]

        # 定义骨架连接关系 (parent-child pairs)
        self.skeleton_edges = [
            (0, 1),   # Hip -> RHip
            (1, 2),   # RHip -> RKnee
            (2, 3),   # RKnee -> RFoot
            (0, 4),   # Hip -> LHip
            (4, 5),   # LHip -> LKnee
            (5, 6),   # LKnee -> LFoot
            (0, 7),   # Hip -> Spine
            (7, 8),   # Spine -> Thorax
            (8, 9),   # Thorax -> Nose
            (9, 10),  # Nose -> Head
            (8, 11),  # Thorax -> LShoulder
            (11, 12),  # LShoulder -> LElbow
            (12, 13),  # LElbow -> LWrist
            (8, 14),  # Thorax -> RShoulder
            (14, 15),  # RShoulder -> RElbow
            (15, 16)  # RElbow -> RWrist
        ]

        self.num_joints = len(self.joints)

    def get_adjacency_matrix(self, add_self_loops=True):
        """
        构建邻接矩阵
        Args:
            add_self_loops: 是否添加自环
        Returns:
            adj_matrix: [J, J] 邻接矩阵
        """
        adj = np.zeros((self.num_joints, self.num_joints))

        # 添加边连接 (双向)
        for i, j in self.skeleton_edges:
            adj[i, j] = 1
            adj[j, i] = 1

        # 添加自环
        if add_self_loops:
            adj += np.eye(self.num_joints)

        return adj

    def get_normalized_adjacency(self):
        """
        获取度归一化的邻接矩阵 D^(-1/2) * A * D^(-1/2)
        """
        adj = self.get_adjacency_matrix()

        # 计算度矩阵
        degree = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        # 归一化邻接矩阵
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        return adj_normalized


class GraphConvLayer(nn.Module):
    """
    图卷积层: X' = D^(-1/2) * A * D^(-1/2) * X * W
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重矩阵
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        """
        Args:
            x: [B, T, J, C] 输入特征
            adj: [J, J] 邻接矩阵
        Returns:
            output: [B, T, J, C'] 输出特征
        """
        B, T, J, C = x.shape

        # Reshape for matrix operations
        x_reshaped = x.view(B * T, J, C)  # [B*T, J, C]

        # Graph convolution: adj @ x @ weight
        support = torch.matmul(x_reshaped, self.weight)  # [B*T, J, C']
        output = torch.matmul(adj, support)  # [B*T, J, C']

        if self.bias is not None:
            output += self.bias

        # Reshape back
        output = output.view(B, T, J, -1)  # [B, T, J, C']

        return output


class GCNBranch(nn.Module):
    """
    GCN 分支：专门处理空间关节依赖
    输入: [B, T, J, C]
    输出: [B, T, J, C]
    """

    def __init__(self, dim, num_joints=17, hidden_dim=None, num_layers=2):
        """
        Args:
            dim: 特征维度
            num_joints: 关节数量 (默认17)
            hidden_dim: 隐藏层维度 (默认与dim相同)
            num_layers: GCN层数
        """
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim or dim
        self.num_layers = num_layers

        # 构建 Human3.6M 图结构
        self.graph = Human36MGraph()
        adj_matrix = self.graph.get_normalized_adjacency()
        self.register_buffer('adj_matrix', torch.FloatTensor(adj_matrix))

        # 构建 GCN 层
        self.gcn_layers = nn.ModuleList()

        # 第一层
        self.gcn_layers.append(GraphConvLayer(dim, self.hidden_dim))

        # 中间层
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GraphConvLayer(
                self.hidden_dim, self.hidden_dim))

        # 最后一层
        if num_layers > 1:
            self.gcn_layers.append(GraphConvLayer(self.hidden_dim, dim))

        # 激活函数和归一化
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - batch, time, joints, channels
        Returns:
            y: [B, T, J, C] - 空间增强的特征
        """
        identity = x  # 残差连接

        # 通过 GCN 层
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, self.adj_matrix)

            # 除了最后一层，都加激活函数和dropout
            if i < len(self.gcn_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)

        # 残差连接 + 层归一化
        output = self.norm(x + identity)

        return output

    def get_graph_info(self):
        """返回图结构信息"""
        return {
            'num_joints': self.num_joints,
            'edges': self.graph.skeleton_edges,
            'adj_matrix_shape': self.adj_matrix.shape,
            'joints': self.graph.joints
        }


def test_gcn_branch():
    """测试 GCN 分支"""
    print("🧪 测试 GCN 分支...")

    # 创建测试数据
    batch_size, time_steps, num_joints, dim = 2, 81, 17, 128
    x = torch.randn(batch_size, time_steps, num_joints, dim)

    # 创建 GCN 分支
    gcn_branch = GCNBranch(dim, num_joints)

    try:
        y_gcn = gcn_branch(x)
        print(f"✅ GCN 分支输出形状: {y_gcn.shape}")
        print(f"✅ 维度检查: 输入 {x.shape} → 输出 {y_gcn.shape}")

        # 梯度测试
        loss = y_gcn.sum()
        loss.backward()
        print("✅ 梯度计算正常")

        # 图结构信息
        graph_info = gcn_branch.get_graph_info()
        print(
            f"✅ 图结构: {graph_info['num_joints']} 关节, {len(graph_info['edges'])} 条边")

        return True

    except Exception as e:
        print(f"❌ GCN 分支测试失败: {e}")
        return False


if __name__ == "__main__":
    test_gcn_branch()
