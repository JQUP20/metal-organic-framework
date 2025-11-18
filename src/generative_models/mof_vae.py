"""
MOF Variational Autoencoder (MOF-VAE)

基于图神经网络的MOF生成模型，用于：
- 学习MOF结构的低维潜在表示
- 从潜在空间生成新的MOF结构
- 基于性质的逆向设计

参考：
- "Inverse design of porous materials using artificial neural networks" (Yao et al., 2018)
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")


class CGCNNConvEncoder(MessagePassing):
    """
    CGCNN卷积层（用于编码器）
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
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = self.edge_gate(z)
        message = self.edge_update(z)
        return gate * message

    def update(self, aggr_out, x):
        return x + aggr_out


class MOFEncoder(nn.Module):
    """
    MOF编码器：从图结构编码到潜在空间

    输入: 图 G = (V, E)
    输出: μ, log_σ² (潜在分布参数)
    """

    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 24,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_conv_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # 节点和边嵌入
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)

        # CGCNN卷积层
        self.conv_layers = nn.ModuleList([
            CGCNNConvEncoder(hidden_dim, hidden_dim)
            for _ in range(num_conv_layers)
        ])

        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_conv_layers)
        ])

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(dropout)
        )

        # 潜在分布参数
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG Data对象

        返回:
            mu: (batch_size, latent_dim) 均值
            logvar: (batch_size, latent_dim) log方差
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 嵌入
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # CGCNN卷积
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.softplus(x)

        # 图级池化
        x_pool = global_mean_pool(x, batch)

        # 全连接层
        h = self.fc(x_pool)

        # 潜在分布参数
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class MOFDecoder(nn.Module):
    """
    MOF解码器：从潜在向量重构图结构

    简化版本：
    - 预测节点特征
    - 预测边特征
    - 预测邻接矩阵（边是否存在）

    注：完整的图生成需要更复杂的架构（如GraphRNN, GraphVAE）
    这里提供简化版本用于教学
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        max_num_nodes: int = 100,
        node_output_dim: int = 4,
        edge_output_dim: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_num_nodes = max_num_nodes
        self.node_output_dim = node_output_dim
        self.edge_output_dim = edge_output_dim

        # 潜在向量到隐藏层
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )

        # 节点特征预测
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, max_num_nodes * node_output_dim)
        )

        # 邻接矩阵预测（边是否存在）
        self.adj_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, max_num_nodes * max_num_nodes)
        )

        # 边特征预测
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 2 * node_output_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, edge_output_dim)
        )

    def forward(self, z):
        """
        前向传播

        参数:
            z: (batch_size, latent_dim) 潜在向量

        返回:
            node_features: (batch_size, max_num_nodes, node_output_dim)
            adj_matrix: (batch_size, max_num_nodes, max_num_nodes)
            edge_features: (batch_size, max_num_nodes, max_num_nodes, edge_output_dim)
        """
        batch_size = z.size(0)

        # 潜在向量到隐藏层
        h = self.fc_latent(z)

        # 节点特征
        node_logits = self.node_decoder(h)
        node_features = node_logits.view(batch_size, self.max_num_nodes, self.node_output_dim)

        # 邻接矩阵（对称化）
        adj_logits = self.adj_decoder(h)
        adj_matrix = adj_logits.view(batch_size, self.max_num_nodes, self.max_num_nodes)
        adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2  # 对称化
        adj_prob = torch.sigmoid(adj_matrix)

        # 边特征（基于节点特征对）
        edge_features = self._compute_edge_features(h, node_features)

        return node_features, adj_prob, edge_features

    def _compute_edge_features(self, h, node_features):
        """
        计算边特征（基于节点特征对）

        参数:
            h: (batch_size, hidden_dim) 隐藏状态
            node_features: (batch_size, max_num_nodes, node_output_dim)

        返回:
            edge_features: (batch_size, max_num_nodes, max_num_nodes, edge_output_dim)
        """
        batch_size = h.size(0)

        # 扩展节点特征对
        node_i = node_features.unsqueeze(2).repeat(1, 1, self.max_num_nodes, 1)
        node_j = node_features.unsqueeze(1).repeat(1, self.max_num_nodes, 1, 1)

        # 拼接节点特征和全局特征
        h_expanded = h.unsqueeze(1).unsqueeze(1).repeat(1, self.max_num_nodes, self.max_num_nodes, 1)
        edge_input = torch.cat([h_expanded, node_i, node_j], dim=-1)

        # 预测边特征
        edge_features = self.edge_decoder(edge_input)

        return edge_features


class MOFVAE(nn.Module):
    """
    MOF Variational Autoencoder

    完整的VAE模型，包括：
    - 编码器：图 → 潜在分布
    - 重参数化技巧
    - 解码器：潜在向量 → 图
    - ELBO损失函数
    """

    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 24,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_num_nodes: int = 100,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        beta: float = 1.0  # KL散度权重（β-VAE）
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # 编码器
        self.encoder = MOFEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout
        )

        # 解码器
        self.decoder = MOFDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_num_nodes=max_num_nodes,
            node_output_dim=node_input_dim,
            edge_output_dim=edge_input_dim,
            dropout=dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧

        z ~ N(μ, σ²) 等价于 z = μ + σ * ε, 其中 ε ~ N(0, 1)

        参数:
            mu: (batch_size, latent_dim) 均值
            logvar: (batch_size, latent_dim) log方差

        返回:
            z: (batch_size, latent_dim) 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, data):
        """编码：图 → 潜在分布参数"""
        return self.encoder(data)

    def decode(self, z):
        """解码：潜在向量 → 图"""
        return self.decoder(z)

    def forward(self, data):
        """
        前向传播（完整的编码-解码过程）

        参数:
            data: PyG Data对象

        返回:
            node_recon: 重构的节点特征
            adj_recon: 重构的邻接矩阵
            edge_recon: 重构的边特征
            mu: 潜在分布均值
            logvar: 潜在分布log方差
        """
        # 编码
        mu, logvar = self.encode(data)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        node_recon, adj_recon, edge_recon = self.decode(z)

        return node_recon, adj_recon, edge_recon, mu, logvar

    def loss_function(
        self,
        data,
        node_recon,
        adj_recon,
        edge_recon,
        mu,
        logvar
    ) -> Dict[str, torch.Tensor]:
        """
        ELBO损失函数

        Loss = Reconstruction Loss + β * KL Divergence

        参数:
            data: 原始图数据
            node_recon: 重构的节点特征
            adj_recon: 重构的邻接矩阵概率
            edge_recon: 重构的边特征
            mu: 潜在分布均值
            logvar: 潜在分布log方差

        返回:
            loss_dict: 包含各部分损失的字典
        """
        batch_size = mu.size(0)

        # 1. 重构损失（节点特征）
        # 简化版本：直接使用MSE
        # 实际应该根据数据类型选择（连续→MSE，离散→CE）
        node_loss = F.mse_loss(
            node_recon.view(batch_size, -1),
            data.x.view(batch_size, -1),
            reduction='sum'
        ) / batch_size

        # 2. 邻接矩阵重构损失
        # 从edge_index重构邻接矩阵
        adj_true = self._edge_index_to_adj_matrix(
            data.edge_index,
            data.batch if hasattr(data, 'batch') else None,
            batch_size,
            self.decoder.max_num_nodes
        ).to(adj_recon.device)

        adj_loss = F.binary_cross_entropy(
            adj_recon.view(batch_size, -1),
            adj_true.view(batch_size, -1),
            reduction='sum'
        ) / batch_size

        # 3. KL散度损失
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # 总损失
        recon_loss = node_loss + adj_loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'node_loss': node_loss,
            'adj_loss': adj_loss,
            'kl_loss': kl_loss
        }

    def _edge_index_to_adj_matrix(
        self,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        batch_size: int,
        max_num_nodes: int
    ) -> torch.Tensor:
        """
        将edge_index转换为邻接矩阵

        参数:
            edge_index: (2, E) 边索引
            batch: (N,) 批次索引
            batch_size: 批次大小
            max_num_nodes: 最大节点数

        返回:
            adj_matrix: (batch_size, max_num_nodes, max_num_nodes)
        """
        if batch is None:
            batch = torch.zeros(edge_index.max() + 1, dtype=torch.long, device=edge_index.device)

        adj_matrix = torch.zeros(
            batch_size, max_num_nodes, max_num_nodes,
            dtype=torch.float,
            device=edge_index.device
        )

        # 填充邻接矩阵
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[:, i]
            b = batch[src].item()

            # 计算相对节点索引（在各自图中的索引）
            # 简化：直接使用全局索引（可能超出max_num_nodes）
            if src < max_num_nodes and tgt < max_num_nodes:
                adj_matrix[b, src, tgt] = 1.0

        return adj_matrix

    def sample(self, num_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        从先验分布p(z) = N(0, I)中采样生成新MOF

        参数:
            num_samples: 采样数量
            device: 设备

        返回:
            z: (num_samples, latent_dim) 采样的潜在向量
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            node_recon, adj_recon, edge_recon = self.decode(z)
        return node_recon, adj_recon, edge_recon

    def interpolate(
        self,
        data1,
        data2,
        num_steps: int = 10
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        在两个MOF之间的潜在空间插值

        参数:
            data1, data2: 两个MOF图数据
            num_steps: 插值步数

        返回:
            interpolated_structures: 插值生成的结构列表
        """
        self.eval()
        with torch.no_grad():
            # 编码
            mu1, _ = self.encode(data1)
            mu2, _ = self.encode(data2)

            # 线性插值
            alphas = torch.linspace(0, 1, num_steps, device=mu1.device)

            interpolated_structures = []
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                node_recon, adj_recon, edge_recon = self.decode(z)
                interpolated_structures.append((node_recon, adj_recon, edge_recon))

        return interpolated_structures


class MOFVAETrainer:
    """
    MOF-VAE训练器

    功能：
    - 训练循环
    - 验证
    - 检查点保存
    - 损失记录
    """

    def __init__(
        self,
        model: MOFVAE,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_losses = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # 前向传播
            node_recon, adj_recon, edge_recon, mu, logvar = self.model(batch)

            # 计算损失
            losses = self.model.loss_function(
                batch, node_recon, adj_recon, edge_recon, mu, logvar
            )

            # 反向传播
            self.optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 记录损失
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.train_losses.append(avg_losses)

        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_losses = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            # 前向传播
            node_recon, adj_recon, edge_recon, mu, logvar = self.model(batch)

            # 计算损失
            losses = self.model.loss_function(
                batch, node_recon, adj_recon, edge_recon, mu, logvar
            )

            # 记录损失
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.val_losses.append(avg_losses)

        return avg_losses

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint loaded from {filepath}")


def create_sample_vae_data():
    """创建示例VAE数据用于测试"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None

    # 创建小图
    data_list = []
    for _ in range(4):
        num_nodes = np.random.randint(5, 10)
        data = Data(
            x=torch.randn(num_nodes, 4),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
            edge_attr=torch.randn(num_nodes * 2, 24)
        )
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == '__main__':
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Testing MOF-VAE model...")

        # 创建模型
        model = MOFVAE(
            node_input_dim=4,
            edge_input_dim=24,
            hidden_dim=64,
            latent_dim=32,
            max_num_nodes=20,
            num_conv_layers=2
        )

        print(f"\nModel architecture:")
        print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters())} parameters")
        print(f"  Decoder: {sum(p.numel() for p in model.decoder.parameters())} parameters")
        print(f"  Total: {sum(p.numel() for p in model.parameters())} parameters")

        # 测试前向传播
        batch_data = create_sample_vae_data()
        if batch_data is not None:
            print(f"\n[Forward Pass Test]")
            print(f"Input batch: {batch_data}")

            node_recon, adj_recon, edge_recon, mu, logvar = model(batch_data)
            print(f"Node reconstruction shape: {node_recon.shape}")
            print(f"Adjacency reconstruction shape: {adj_recon.shape}")
            print(f"Latent mu shape: {mu.shape}")
            print(f"Latent logvar shape: {logvar.shape}")

            # 测试损失函数
            losses = model.loss_function(
                batch_data, node_recon, adj_recon, edge_recon, mu, logvar
            )
            print(f"\n[Loss Test]")
            for key, value in losses.items():
                print(f"  {key}: {value.item():.4f}")

            # 测试采样
            print(f"\n[Sampling Test]")
            node_gen, adj_gen, edge_gen = model.sample(num_samples=2)
            print(f"Generated nodes shape: {node_gen.shape}")
            print(f"Generated adjacency shape: {adj_gen.shape}")

            print("\n✓ MOF-VAE model working correctly!")
    else:
        print("PyTorch Geometric not installed. Cannot run tests.")
