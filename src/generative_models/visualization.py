"""
生成模型可视化工具

用于可视化VAE潜在空间、扩散过程、优化历史等。

作者: AI & MOF Course
日期: 2024-11
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Tuple, Union
import torch


class LatentSpaceVisualizer:
    """
    潜在空间可视化器

    用于可视化VAE学习的潜在空间结构
    """

    def __init__(self, figsize=(10, 8)):
        """
        参数:
            figsize: 图形大小
        """
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_latent_space_2d(
        self,
        latent_vectors: np.ndarray,
        properties: Optional[np.ndarray] = None,
        method: str = 'pca',
        title: str = 'Latent Space Visualization',
        property_name: str = 'Property',
        highlight_points: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制2D潜在空间

        参数:
            latent_vectors: 潜在向量 (N, latent_dim)
            properties: 性质值 (N,)，用于着色
            method: 降维方法 ('pca' 或 'tsne')
            title: 图形标题
            property_name: 性质名称
            highlight_points: 要高亮的点的索引
            save_path: 保存路径
        """
        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            latent_2d = reducer.fit_transform(latent_vectors)
            explained_var = reducer.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%} var.)'
            ylabel = f'PC2 ({explained_var[1]:.1%} var.)'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = reducer.fit_transform(latent_vectors)
            xlabel = 't-SNE Dimension 1'
            ylabel = 't-SNE Dimension 2'
        else:
            raise ValueError(f"Unknown method: {method}")

        # 绘图
        plt.figure(figsize=self.figsize)

        if properties is not None:
            # 按性质着色
            scatter = plt.scatter(
                latent_2d[:, 0],
                latent_2d[:, 1],
                c=properties,
                cmap='viridis',
                s=60,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5
            )
            plt.colorbar(scatter, label=property_name)
        else:
            # 统一颜色
            plt.scatter(
                latent_2d[:, 0],
                latent_2d[:, 1],
                s=60,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5
            )

        # 高亮特定点
        if highlight_points is not None:
            plt.scatter(
                latent_2d[highlight_points, 0],
                latent_2d[highlight_points, 1],
                s=200,
                c='red',
                marker='*',
                edgecolors='black',
                linewidth=2,
                label='Highlighted',
                zorder=5
            )
            plt.legend()

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return latent_2d

    def plot_latent_distribution(
        self,
        latent_vectors: np.ndarray,
        dims_to_plot: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制潜在空间各维度的分布

        参数:
            latent_vectors: 潜在向量 (N, latent_dim)
            dims_to_plot: 要绘制的维度（默认全部）
            save_path: 保存路径
        """
        latent_dim = latent_vectors.shape[1]

        if dims_to_plot is None:
            dims_to_plot = list(range(min(latent_dim, 16)))  # 最多绘制16维

        n_dims = len(dims_to_plot)
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, dim in enumerate(dims_to_plot):
            ax = axes[idx]

            # 绘制直方图
            ax.hist(latent_vectors[:, dim], bins=30, alpha=0.7, color='steelblue', edgecolor='black')

            # 叠加正态分布（理论分布）
            mu, sigma = latent_vectors[:, dim].mean(), latent_vectors[:, dim].std()
            x = np.linspace(latent_vectors[:, dim].min(), latent_vectors[:, dim].max(), 100)
            ax.plot(x, len(latent_vectors) * sigma * np.sqrt(2 * np.pi) *
                   (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)),
                   'r--', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}²)')

            ax.set_xlabel(f'Dimension {dim}', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Latent Dim {dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_dims, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_interpolation_path(
        self,
        latent_vectors: np.ndarray,
        start_idx: int,
        end_idx: int,
        interpolation_points: np.ndarray,
        properties: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        在潜在空间中绘制插值路径

        参数:
            latent_vectors: 所有潜在向量
            start_idx: 起始点索引
            end_idx: 终止点索引
            interpolation_points: 插值点
            properties: 性质值（用于着色）
            save_path: 保存路径
        """
        # 使用PCA降维到2D
        pca = PCA(n_components=2, random_state=42)
        latent_2d = pca.fit_transform(latent_vectors)
        interp_2d = pca.transform(interpolation_points)

        plt.figure(figsize=self.figsize)

        # 绘制背景点
        if properties is not None:
            scatter = plt.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                c=properties, cmap='viridis',
                s=40, alpha=0.3, edgecolors='none'
            )
            plt.colorbar(scatter, label='Property')
        else:
            plt.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                s=40, alpha=0.3, color='gray', edgecolors='none'
            )

        # 绘制插值路径
        plt.plot(interp_2d[:, 0], interp_2d[:, 1],
                'r-', linewidth=2, alpha=0.7, label='Interpolation Path')
        plt.scatter(interp_2d[:, 0], interp_2d[:, 1],
                   s=80, c='red', alpha=0.7, edgecolors='darkred', linewidth=1)

        # 标记起点和终点
        plt.scatter(latent_2d[start_idx, 0], latent_2d[start_idx, 1],
                   s=300, c='green', marker='*', edgecolors='darkgreen',
                   linewidth=2, label='Start', zorder=5)
        plt.scatter(latent_2d[end_idx, 0], latent_2d[end_idx, 1],
                   s=300, c='blue', marker='*', edgecolors='darkblue',
                   linewidth=2, label='End', zorder=5)

        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title('Interpolation in Latent Space', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class DiffusionVisualizer:
    """
    扩散模型可视化器

    用于可视化扩散过程的中间步骤
    """

    def __init__(self, figsize=(14, 10)):
        """
        参数:
            figsize: 图形大小
        """
        self.figsize = figsize

    def plot_diffusion_process(
        self,
        trajectory: List[torch.Tensor],
        timesteps_to_show: Optional[List[int]] = None,
        feature_dim: int = 0,
        save_path: Optional[str] = None
    ):
        """
        绘制扩散过程

        参数:
            trajectory: 扩散轨迹 [(T, N, D)]
            timesteps_to_show: 要显示的时间步
            feature_dim: 要可视化的特征维度
            save_path: 保存路径
        """
        num_timesteps = len(trajectory)

        if timesteps_to_show is None:
            # 均匀选择8个时间步
            timesteps_to_show = np.linspace(0, num_timesteps-1, 8, dtype=int).tolist()

        n_steps = len(timesteps_to_show)
        n_cols = 4
        n_rows = (n_steps + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, t in enumerate(timesteps_to_show):
            ax = axes[idx]

            # 获取该时间步的特征
            x_t = trajectory[t].cpu().numpy()
            feature_values = x_t[:, feature_dim] if x_t.ndim > 1 else x_t

            # 绘制直方图
            ax.hist(feature_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel(f'Feature {feature_dim} Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'Timestep t={t}/{num_timesteps-1}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_steps, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Diffusion Process (Feature {feature_dim})',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_noise_schedule(
        self,
        betas: np.ndarray,
        alphas: np.ndarray,
        alphas_cumprod: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        绘制噪声调度

        参数:
            betas: β值
            alphas: α值 (1 - β)
            alphas_cumprod: 累积α值
            save_path: 保存路径
        """
        timesteps = np.arange(len(betas))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # β调度
        axes[0].plot(timesteps, betas, linewidth=2, color='steelblue')
        axes[0].set_xlabel('Timestep', fontsize=12)
        axes[0].set_ylabel('β_t', fontsize=12)
        axes[0].set_title('Beta Schedule', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # α调度
        axes[1].plot(timesteps, alphas, linewidth=2, color='green')
        axes[1].set_xlabel('Timestep', fontsize=12)
        axes[1].set_ylabel('α_t = 1 - β_t', fontsize=12)
        axes[1].set_title('Alpha Schedule', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 累积α
        axes[2].plot(timesteps, alphas_cumprod, linewidth=2, color='darkred')
        axes[2].set_xlabel('Timestep', fontsize=12)
        axes[2].set_ylabel('ᾱ_t = ∏α_i', fontsize=12)
        axes[2].set_title('Cumulative Alpha', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class OptimizationVisualizer:
    """
    优化过程可视化器

    用于可视化贝叶斯优化和逆向设计过程
    """

    def __init__(self, figsize=(12, 5)):
        """
        参数:
            figsize: 图形大小
        """
        self.figsize = figsize

    def plot_optimization_history(
        self,
        property_history: List[float],
        title: str = 'Optimization History',
        property_name: str = 'Property Value',
        target_value: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制优化历史

        参数:
            property_history: 性质值历史
            title: 标题
            property_name: 性质名称
            target_value: 目标值（可选）
            save_path: 保存路径
        """
        iterations = np.arange(len(property_history))
        best_so_far = np.maximum.accumulate(property_history)

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # 左图：优化历史
        axes[0].plot(iterations, property_history, 'o-', alpha=0.6,
                    label='Evaluated', markersize=5)
        axes[0].plot(iterations, best_so_far, 'r-', linewidth=2.5,
                    label='Best so far')

        if target_value is not None:
            axes[0].axhline(y=target_value, color='g', linestyle='--',
                          linewidth=2, label=f'Target: {target_value:.2f}')

        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel(property_name, fontsize=12)
        axes[0].set_title(title, fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 右图：每次迭代的改进
        improvements = np.diff(best_so_far)
        axes[1].bar(range(len(improvements)), improvements, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Improvement', fontsize=12)
        axes[1].set_title('Improvement per Iteration', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_acquisition_function(
        self,
        X_sample: np.ndarray,
        y_sample: np.ndarray,
        X_plot: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        acquisition: np.ndarray,
        next_point: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制采集函数（仅1D情况）

        参数:
            X_sample: 已采样的X
            y_sample: 已采样的y
            X_plot: 绘图用的X
            mean: GP预测均值
            std: GP预测标准差
            acquisition: 采集函数值
            next_point: 下一个采样点
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 上图：GP预测
        axes[0].plot(X_plot, mean, 'b-', linewidth=2, label='GP Mean')
        axes[0].fill_between(
            X_plot.flatten(),
            (mean - 1.96 * std).flatten(),
            (mean + 1.96 * std).flatten(),
            alpha=0.3,
            label='95% CI'
        )
        axes[0].plot(X_sample, y_sample, 'ro', markersize=8,
                    label='Observations', zorder=5)

        if next_point is not None:
            axes[0].axvline(x=next_point, color='green', linestyle='--',
                          linewidth=2, label='Next Sample', zorder=4)

        axes[0].set_ylabel('Property Value', fontsize=12)
        axes[0].set_title('Gaussian Process Prediction', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 下图：采集函数
        axes[1].plot(X_plot, acquisition, 'g-', linewidth=2)
        axes[1].fill_between(X_plot.flatten(), 0, acquisition.flatten(),
                            alpha=0.3, color='green')

        if next_point is not None:
            max_acq = acquisition[np.abs(X_plot - next_point).argmin()]
            axes[1].plot(next_point, max_acq, 'r*', markersize=20,
                        label='Max Acquisition', zorder=5)

        axes[1].set_xlabel('Latent Space', fontsize=12)
        axes[1].set_ylabel('Acquisition Value', fontsize=12)
        axes[1].set_title('Acquisition Function', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_pareto_front(
        self,
        objectives: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str] = ['Objective 1', 'Objective 2'],
        save_path: Optional[str] = None
    ):
        """
        绘制Pareto前沿（仅2D情况）

        参数:
            objectives: 目标值 (N, 2)
            pareto_mask: Pareto最优解的mask
            objective_names: 目标名称
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)

        # 非Pareto点
        plt.scatter(
            objectives[~pareto_mask, 0],
            objectives[~pareto_mask, 1],
            s=60,
            alpha=0.5,
            color='gray',
            label='Non-Pareto'
        )

        # Pareto前沿
        pareto_points = objectives[pareto_mask]
        sorted_idx = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_idx]

        plt.scatter(
            pareto_sorted[:, 0],
            pareto_sorted[:, 1],
            s=100,
            c='red',
            marker='*',
            edgecolors='darkred',
            linewidth=1.5,
            label='Pareto Front',
            zorder=5
        )

        plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'r--', linewidth=2, alpha=0.7, zorder=4)

        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
        plt.title('Pareto Front', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class TrainingVisualizer:
    """
    训练过程可视化器

    用于可视化模型训练过程
    """

    def __init__(self, figsize=(14, 5)):
        """
        参数:
            figsize: 图形大小
        """
        self.figsize = figsize

    def plot_vae_training(
        self,
        train_losses: List[Dict[str, float]],
        val_losses: List[Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        绘制VAE训练曲线

        参数:
            train_losses: 训练损失历史
            val_losses: 验证损失历史
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        epochs = range(1, len(train_losses) + 1)

        # 总损失
        axes[0].plot(epochs, [d['loss'] for d in train_losses],
                    label='Train', linewidth=2, alpha=0.8)
        axes[0].plot(epochs, [d['loss'] for d in val_losses],
                    label='Val', linewidth=2, alpha=0.8)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Total Loss (ELBO)', fontsize=11)
        axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 重构损失
        axes[1].plot(epochs, [d['recon_loss'] for d in train_losses],
                    linewidth=2, alpha=0.8, label='Train')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Reconstruction Loss', fontsize=11)
        axes[1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # KL散度
        axes[2].plot(epochs, [d['kl_loss'] for d in train_losses],
                    color='orange', linewidth=2, alpha=0.8, label='Train')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('KL Divergence', fontsize=11)
        axes[2].set_title('KL Divergence', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# 示例使用
if __name__ == "__main__":
    print("生成模型可视化工具")
    print("=" * 60)

    # 测试潜在空间可视化
    print("\n1. 测试潜在空间可视化...")
    latent_viz = LatentSpaceVisualizer()

    # 生成模拟数据
    np.random.seed(42)
    latent_vectors = np.random.randn(200, 16)
    properties = 5.0 + 2.0 * latent_vectors[:, 0] + np.random.randn(200) * 0.5

    latent_viz.plot_latent_space_2d(
        latent_vectors,
        properties,
        method='pca',
        title='Test Latent Space',
        property_name='CO2 Uptake (mmol/g)'
    )

    print("✓ 潜在空间可视化测试完成")

    # 测试优化可视化
    print("\n2. 测试优化历史可视化...")
    opt_viz = OptimizationVisualizer()

    property_history = [4.5, 5.2, 5.1, 6.0, 6.3, 6.8, 6.9, 7.2, 7.4, 7.5]
    opt_viz.plot_optimization_history(
        property_history,
        property_name='CO2 Uptake (mmol/g)',
        target_value=7.0
    )

    print("✓ 优化历史可视化测试完成")

    print("\n所有测试完成！")
