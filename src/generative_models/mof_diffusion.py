"""
MOF Diffusion Model (MOF-Diffusion)

基于扩散模型的MOF生成，使用DDPM (Denoising Diffusion Probabilistic Models)框架

参考:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Crystal Diffusion Variational Autoencoder" (Xie et al., 2022)
- "DiffCSP: Diffusion for Crystal Structure Prediction" (Jiao et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
import math

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")


class SinusoidalPositionEmbeddings(nn.Module):
    """
    时间步的正弦位置编码

    将时间步t编码为向量，用于条件生成
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        参数:
            time: (batch_size,) 时间步

        返回:
            embeddings: (batch_size, dim) 时间嵌入
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeConditionalLayer(nn.Module):
    """
    时间条件层

    将时间嵌入融入到特征中
    """

    def __init__(self, feature_dim: int, time_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(time_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (N, feature_dim) 特征
            t_emb: (batch_size, time_dim) 时间嵌入

        返回:
            x_cond: (N, feature_dim) 时间条件特征
        """
        # 扩展时间嵌入到所有节点
        return x + self.layer(t_emb)


class NoisePredictor(nn.Module):
    """
    噪声预测网络

    预测在时间步t添加到节点特征的噪声
    使用图神经网络架构
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 24,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        self.node_dim = node_dim
        self.time_dim = time_dim

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 输入嵌入
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)

        # GNN层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 时间条件层
        self.time_cond_layers = nn.ModuleList([
            TimeConditionalLayer(hidden_dim, time_dim)
            for _ in range(num_layers)
        ])

        # 输出层（预测噪声）
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            x: (N, node_dim) 含噪声的节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, edge_dim) 边特征
            t: (batch_size,) 时间步
            batch: (N,) 批次索引

        返回:
            noise_pred: (N, node_dim) 预测的噪声
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 节点和边嵌入
        h = self.node_embedding(x)
        edge_feat = self.edge_embedding(edge_attr)

        # GNN传播
        for gnn_layer, time_cond_layer in zip(self.gnn_layers, self.time_cond_layers):
            # 图卷积
            h = gnn_layer(h, edge_index, edge_feat)

            # 时间条件
            if batch is not None:
                # 扩展时间嵌入到每个节点
                t_emb_expanded = t_emb[batch]
            else:
                t_emb_expanded = t_emb.expand(h.size(0), -1)

            h = time_cond_layer(h, t_emb_expanded)

        # 预测噪声
        noise_pred = self.output(h)

        return noise_pred


class GNNLayer(MessagePassing):
    """
    简单的GNN层（用于噪声预测网络）
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr='add')

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class GaussianDiffusion:
    """
    高斯扩散过程

    实现DDPM的前向和反向过程
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'linear'
    ):
        """
        参数:
            num_timesteps: 扩散步数T
            beta_start: 初始β值
            beta_end: 最终β值
            schedule: 噪声调度方案 ('linear', 'cosine')
        """
        self.num_timesteps = num_timesteps

        # 噪声调度
        if schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = torch.from_numpy(betas).float()

        # 计算DDPM所需的系数
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # 前向过程系数
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 反向过程系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """
        余弦噪声调度（更平滑）

        参考: Improved DDPM (Nichol & Dhariwal, 2021)
        """
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散过程 q(x_t | x_0)

        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        参数:
            x_start: (N, D) 原始数据
            t: (batch_size,) 时间步
            noise: (N, D) 噪声（可选）

        返回:
            x_t: (N, D) 含噪声的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 获取系数
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # 添加噪声
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        反向去噪过程 p(x_{t-1} | x_t)

        x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

        参数:
            model: 噪声预测模型
            x_t: (N, D) 当前含噪声数据
            t: (batch_size,) 时间步
            edge_index: (2, E) 边索引
            edge_attr: (E, edge_dim) 边特征
            batch: (N,) 批次索引

        返回:
            x_t_prev: (N, D) 去噪后的数据
        """
        # 预测噪声
        noise_pred = model(x_t, edge_index, edge_attr, t, batch)

        # 计算x_{t-1}的均值
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        # 添加噪声（t > 0时）
        if t[0] > 0:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        从a中提取索引t的值，并reshape为与x_shape兼容的形状

        参数:
            a: (T,) 系数数组
            t: (batch_size,) 时间步
            x_shape: 目标形状

        返回:
            out: (batch_size, 1, ..., 1) 扩展后的系数
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class MOFDiffusion(nn.Module):
    """
    MOF扩散模型

    完整的扩散模型，包括：
    - 噪声预测网络
    - 前向扩散过程
    - 反向去噪过程
    - 训练和采样方法
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 24,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear'
    ):
        super().__init__()

        # 噪声预测网络
        self.noise_predictor = NoisePredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers
        )

        # 扩散过程
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            schedule=beta_schedule
        )

        self.node_dim = node_dim
        self.num_timesteps = num_timesteps

    def forward(
        self,
        x_start: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播

        参数:
            x_start: (N, node_dim) 原始节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, edge_dim) 边特征
            batch: (N,) 批次索引

        返回:
            noise_pred: 预测的噪声
            noise: 真实噪声
            t: 采样的时间步
        """
        batch_size = batch.max().item() + 1 if batch is not None else 1
        device = x_start.device

        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # 生成噪声
        noise = torch.randn_like(x_start)

        # 前向扩散（添加噪声）
        if batch is not None:
            t_expanded = t[batch]
        else:
            t_expanded = t.expand(x_start.size(0))

        x_t = self.diffusion.q_sample(x_start, t_expanded, noise)

        # 预测噪声
        noise_pred = self.noise_predictor(x_t, edge_index, edge_attr, t, batch)

        return noise_pred, noise, t

    @torch.no_grad()
    def sample(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        从噪声生成MOF结构（反向去噪）

        参数:
            num_nodes: 节点数量
            edge_index: (2, E) 边索引（图结构固定）
            edge_attr: (E, edge_dim) 边特征
            batch: (N,) 批次索引
            return_trajectory: 是否返回完整的去噪轨迹

        返回:
            x_0: (N, node_dim) 生成的节点特征
        """
        device = edge_index.device
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # 从纯噪声开始
        x = torch.randn(num_nodes, self.node_dim, device=device)

        trajectory = [x] if return_trajectory else None

        # 反向去噪过程
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.diffusion.p_sample(
                self.noise_predictor, x, t, edge_index, edge_attr, batch
            )

            if return_trajectory and (i % 100 == 0 or i == 0):
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    def compute_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扩散模型损失（简化的L2损失）

        Loss = ||ε - ε_θ(x_t, t)||²

        参数:
            noise_pred: 预测的噪声
            noise: 真实噪声

        返回:
            loss: 标量损失
        """
        return F.mse_loss(noise_pred, noise)


class MOFDiffusionTrainer:
    """
    MOF扩散模型训练器
    """

    def __init__(
        self,
        model: MOFDiffusion,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []

    def train_step(self, batch) -> float:
        """训练一步"""
        self.model.train()
        batch = batch.to(self.device)

        # 前向传播
        noise_pred, noise, t = self.model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch if hasattr(batch, 'batch') else None
        )

        # 计算损失
        loss = self.model.compute_loss(noise_pred, noise)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        print(f"Checkpoint loaded from {filepath}")


if __name__ == '__main__':
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Testing MOF Diffusion model...")

        # 创建模型
        model = MOFDiffusion(
            node_dim=4,
            edge_dim=24,
            hidden_dim=64,
            time_dim=32,
            num_layers=2,
            num_timesteps=100  # 减少用于测试
        )

        print(f"\nModel architecture:")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

        # 创建测试数据
        num_nodes = 10
        num_edges = 20

        x = torch.randn(num_nodes, 4)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 24)

        # 测试前向传播（训练）
        print(f"\n[Training Forward Pass]")
        noise_pred, noise, t = model(x, edge_index, edge_attr)
        print(f"Noise prediction shape: {noise_pred.shape}")
        print(f"Sampled timestep: {t.item()}")

        # 计算损失
        loss = model.compute_loss(noise_pred, noise)
        print(f"Loss: {loss.item():.4f}")

        # 测试采样
        print(f"\n[Sampling Test]")
        x_gen = model.sample(num_nodes, edge_index, edge_attr)
        print(f"Generated features shape: {x_gen.shape}")

        # 测试轨迹采样
        print(f"\n[Trajectory Sampling Test]")
        x_final, trajectory = model.sample(
            num_nodes, edge_index, edge_attr, return_trajectory=True
        )
        print(f"Trajectory length: {len(trajectory)}")
        print(f"Final generated features shape: {x_final.shape}")

        print("\n✓ MOF Diffusion model working correctly!")
    else:
        print("PyTorch Geometric not installed. Cannot run tests.")
