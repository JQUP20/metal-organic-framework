"""
Graph Neural Network Models for MOF Property Prediction

实现CGCNN和简化的MEGNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.nn import GATConv, GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")


class CGCNNConv(MessagePassing):
    """
    CGCNN卷积层

    基于论文: Crystal Graph Convolutional Neural Networks (Xie & Grossman, 2018)
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr='add')
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # 边门控网络
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.Sigmoid()
        )

        # 边特征网络
        self.edge_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.Softplus()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播

        参数:
            x: (N, node_dim) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, edge_dim) 边特征

        返回:
            out: (N, node_dim) 更新后的节点特征
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        消息函数

        参数:
            x_i: (E, node_dim) 目标节点特征
            x_j: (E, node_dim) 源节点特征
            edge_attr: (E, edge_dim) 边特征
        """
        # 拼接节点和边特征
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # 计算门控和消息
        gate = self.edge_gate(z)
        message = self.edge_update(z)

        return gate * message

    def update(self, aggr_out, x):
        """
        更新函数（残差连接）
        """
        return x + aggr_out


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network

    用于预测晶体材料性质
    """

    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 24,
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        num_fc_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()

        # 节点嵌入
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)

        # 边嵌入
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)

        # CGCNN卷积层
        self.conv_layers = nn.ModuleList([
            CGCNNConv(hidden_dim, hidden_dim)
            for _ in range(num_conv_layers)
        ])

        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_conv_layers)
        ])

        # 全连接层
        fc_layers = []
        for i in range(num_fc_layers):
            if i == 0:
                fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fc_layers.append(nn.Softplus())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))

        self.fc_layers = nn.Sequential(*fc_layers)

        # 输出层
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG Data对象
                - x: (N, node_input_dim) 节点特征
                - edge_index: (2, E) 边索引
                - edge_attr: (E, edge_input_dim) 边特征
                - batch: (N,) 批次索引

        返回:
            out: (batch_size,) 预测值
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None

        # 嵌入
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # CGCNN卷积
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.softplus(x)

        # 池化（图级）
        if batch is None:
            # 单个图
            x_pool = x.mean(dim=0, keepdim=True)
        else:
            # 批量图
            x_pool = global_mean_pool(x, batch)

        # 全连接层
        x_pool = self.fc_layers(x_pool)

        # 输出
        out = self.output(x_pool).squeeze(-1)

        return out


class SimpleMEGNetBlock(nn.Module):
    """
    简化的MEGNet块

    包含边更新、节点更新、全局更新
    """

    def __init__(self, dim: int):
        super().__init__()

        # 边更新
        self.edge_update = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Softplus(),
            nn.Linear(dim, dim)
        )

        # 节点更新
        self.node_update = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Softplus(),
            nn.Linear(dim, dim)
        )

        # 全局更新
        self.global_update = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Softplus(),
            nn.Linear(dim, dim)
        )

    def forward(self, v, e, u, edge_index, batch):
        """
        前向传播

        参数:
            v: (N, dim) 节点特征
            e: (E, dim) 边特征
            u: (B, dim) 全局特征
            edge_index: (2, E) 边索引
            batch: (N,) 批次索引
        """
        # 1. 边更新
        row, col = edge_index
        v_i, v_j = v[row], v[col]
        u_expanded_e = u[batch[row]]  # 将全局特征扩展到边

        e_input = torch.cat([e, v_i, u_expanded_e], dim=-1)
        e_new = e + self.edge_update(e_input)

        # 2. 节点更新（聚合边信息）
        # 简化：使用scatter_mean聚合边到节点
        from torch_scatter import scatter_mean
        e_agg = scatter_mean(e_new, row, dim=0, dim_size=v.size(0))
        u_expanded_v = u[batch]

        v_input = torch.cat([v, e_agg, u_expanded_v], dim=-1)
        v_new = v + self.node_update(v_input)

        # 3. 全局更新
        v_total = global_mean_pool(v_new, batch)
        e_total = global_mean_pool(e_new, batch[row])

        u_input = torch.cat([u, v_total, e_total], dim=-1)
        u_new = u + self.global_update(u_input)

        return v_new, e_new, u_new


class SimpleMEGNet(nn.Module):
    """
    简化的MEGNet模型

    基于论文: MatErials Graph Network (Chen et al., 2019)
    """

    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 24,
        global_input_dim: int = 1,  # 可以添加温度、压力等全局特征
        hidden_dim: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()

        # 嵌入层
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_input_dim, hidden_dim)

        # MEGNet块
        self.blocks = nn.ModuleList([
            SimpleMEGNetBlock(hidden_dim)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG Data对象
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # 初始化全局特征（简化：使用零向量）
        num_graphs = batch.max().item() + 1
        u = torch.zeros(num_graphs, 1, device=x.device)

        # 嵌入
        v = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)
        u = self.global_embedding(u)

        # MEGNet块
        for block in self.blocks:
            v, e, u = block(v, e, u, edge_index, batch)

        # 输出（使用全局特征）
        out = self.output(u).squeeze(-1)

        return out


def create_sample_data():
    """创建样本数据用于测试"""
    try:
        from torch_geometric.data import Data, Batch
    except ImportError:
        print("PyTorch Geometric not installed")
        return None

    # 创建两个小图
    data1 = Data(
        x=torch.randn(5, 4),  # 5个节点，4维特征
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3],
                                 [1, 2, 3, 4, 0, 4, 0, 1, 2]], dtype=torch.long),
        edge_attr=torch.randn(9, 24),  # 9条边，24维特征
        y=torch.tensor([3.5])
    )

    data2 = Data(
        x=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 0],
                                 [1, 2, 3, 0, 3]], dtype=torch.long),
        edge_attr=torch.randn(5, 24),
        y=torch.tensor([2.8])
    )

    # 创建批次
    batch = Batch.from_data_list([data1, data2])

    return batch


if __name__ == '__main__':
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Testing CGCNN and SimpleMEGNet models...")

        # 创建样本数据
        batch_data = create_sample_data()

        if batch_data is not None:
            # 测试CGCNN
            print("\n[CGCNN Test]")
            model_cgcnn = CGCNN(
                node_input_dim=4,
                edge_input_dim=24,
                hidden_dim=64,
                num_conv_layers=3
            )
            out_cgcnn = model_cgcnn(batch_data)
            print(f"Input: {batch_data}")
            print(f"Output shape: {out_cgcnn.shape}")
            print(f"Output: {out_cgcnn}")

            # 测试SimpleMEGNet
            print("\n[SimpleMEGNet Test]")
            model_megnet = SimpleMEGNet(
                node_input_dim=4,
                edge_input_dim=24,
                hidden_dim=64,
                num_blocks=2
            )
            out_megnet = model_megnet(batch_data)
            print(f"Output shape: {out_megnet.shape}")
            print(f"Output: {out_megnet}")

            print("\n✓ Models working correctly!")
    else:
        print("PyTorch Geometric not installed. Cannot run tests.")
