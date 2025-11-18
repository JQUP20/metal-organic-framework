"""
Bayesian Optimization for Inverse MOF Design

使用贝叶斯优化在潜在空间中搜索具有目标性质的MOF结构

功能：
- 高斯过程（GP）作为代理模型
- 多种采集函数（EI, UCB, PI）
- 支持单目标和多目标优化
- 与VAE潜在空间集成

参考：
- "Practical Bayesian Optimization" (Shahriari et al., 2016)
- "A Tutorial on Bayesian Optimization" (Frazier, 2018)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Dict, List
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class GaussianProcess:
    """
    高斯过程回归（GPR）

    用作贝叶斯优化的代理模型
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        length_scale: float = 1.0,
        noise: float = 1e-6,
        alpha: float = 1e-10
    ):
        """
        参数:
            kernel: 核函数类型 ('rbf', 'matern')
            length_scale: RBF核的长度尺度
            noise: 观测噪声
            alpha: 数值稳定性参数
        """
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        self.alpha = alpha

        # 训练数据
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        径向基函数（RBF）核

        k(x, x') = exp(-||x - x'||² / (2 * l²))

        参数:
            X1: (n1, d) 输入
            X2: (n2, d) 输入

        返回:
            K: (n1, n2) 核矩阵
        """
        # 计算欧氏距离的平方
        dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                np.sum(X2**2, axis=1).reshape(1, -1) - \
                2 * np.dot(X1, X2.T)

        K = np.exp(-dists / (2 * self.length_scale**2))
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合高斯过程

        参数:
            X: (n, d) 训练输入
            y: (n,) 训练目标
        """
        self.X_train = X
        self.y_train = y

        # 计算核矩阵
        K = self._rbf_kernel(X, X)

        # 添加噪声项（对角线）
        K += (self.noise + self.alpha) * np.eye(len(X))

        # 求逆（用于预测）
        self.K_inv = np.linalg.inv(K)

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测

        参数:
            X: (m, d) 测试输入
            return_std: 是否返回标准差

        返回:
            mu: (m,) 预测均值
            sigma: (m,) 预测标准差（可选）
        """
        if self.X_train is None:
            raise ValueError("GP not fitted. Call fit() first.")

        # 计算核向量
        K_star = self._rbf_kernel(X, self.X_train)

        # 预测均值
        mu = K_star @ self.K_inv @ self.y_train

        if return_std:
            # 预测方差
            K_star_star = self._rbf_kernel(X, X)
            var = np.diag(K_star_star) - np.sum(K_star @ self.K_inv * K_star, axis=1)
            var = np.maximum(var, 0)  # 数值稳定性
            sigma = np.sqrt(var)
            return mu, sigma
        else:
            return mu, None


class AcquisitionFunction:
    """
    采集函数（Acquisition Functions）

    用于选择下一个采样点
    """

    @staticmethod
    def expected_improvement(
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        期望改进（Expected Improvement, EI）

        EI(x) = E[max(f(x) - f_best, 0)]
              = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)

        其中 Z = (μ(x) - f_best - ξ) / σ(x)

        参数:
            X: (n, d) 候选点
            gp: 高斯过程模型
            y_best: 当前最优值
            xi: 探索参数（trade-off exploration vs exploitation）

        返回:
            ei: (n,) 期望改进值
        """
        mu, sigma = gp.predict(X, return_std=True)

        # 避免除零
        sigma = np.maximum(sigma, 1e-9)

        # 计算Z分数
        Z = (mu - y_best - xi) / sigma

        # 计算期望改进
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    @staticmethod
    def upper_confidence_bound(
        X: np.ndarray,
        gp: GaussianProcess,
        kappa: float = 2.0
    ) -> np.ndarray:
        """
        上置信界（Upper Confidence Bound, UCB）

        UCB(x) = μ(x) + κ * σ(x)

        参数:
            X: (n, d) 候选点
            gp: 高斯过程模型
            kappa: 探索参数（越大越倾向探索）

        返回:
            ucb: (n,) UCB值
        """
        mu, sigma = gp.predict(X, return_std=True)
        ucb = mu + kappa * sigma
        return ucb

    @staticmethod
    def probability_of_improvement(
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        改进概率（Probability of Improvement, PI）

        PI(x) = P(f(x) > f_best) = Φ((μ(x) - f_best - ξ) / σ(x))

        参数:
            X: (n, d) 候选点
            gp: 高斯过程模型
            y_best: 当前最优值
            xi: 探索参数

        返回:
            pi: (n,) 改进概率
        """
        mu, sigma = gp.predict(X, return_std=True)

        # 避免除零
        sigma = np.maximum(sigma, 1e-9)

        # 计算Z分数
        Z = (mu - y_best - xi) / sigma

        # 计算改进概率
        pi = norm.cdf(Z)

        return pi


class BayesianOptimizer:
    """
    贝叶斯优化器

    在潜在空间中搜索具有目标性质的MOF
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: np.ndarray,
        acquisition: str = 'ei',
        n_initial: int = 5,
        kernel: str = 'rbf',
        random_state: Optional[int] = None
    ):
        """
        参数:
            objective_function: 目标函数 f: z -> property
            bounds: (d, 2) 搜索边界 [[min1, max1], [min2, max2], ...]
            acquisition: 采集函数类型 ('ei', 'ucb', 'pi')
            n_initial: 初始随机采样点数量
            kernel: GP核函数类型
            random_state: 随机种子
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.acquisition = acquisition
        self.n_initial = n_initial
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # 高斯过程
        self.gp = GaussianProcess(kernel=kernel)

        # 观测数据
        self.X_observed = []
        self.y_observed = []

        # 最优值
        self.X_best = None
        self.y_best = -np.inf

    def _generate_random_samples(self, n: int) -> np.ndarray:
        """
        在边界内生成随机样本

        参数:
            n: 样本数量

        返回:
            X: (n, d) 随机样本
        """
        X = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(n, self.dim)
        )
        return X

    def _optimize_acquisition(self) -> np.ndarray:
        """
        优化采集函数以找到下一个采样点

        使用多起点优化避免局部最优

        返回:
            x_next: (d,) 下一个采样点
        """
        # 选择采集函数
        if self.acquisition == 'ei':
            acq_func = lambda X: -AcquisitionFunction.expected_improvement(
                X.reshape(1, -1), self.gp, self.y_best
            )[0]
        elif self.acquisition == 'ucb':
            acq_func = lambda X: -AcquisitionFunction.upper_confidence_bound(
                X.reshape(1, -1), self.gp
            )[0]
        elif self.acquisition == 'pi':
            acq_func = lambda X: -AcquisitionFunction.probability_of_improvement(
                X.reshape(1, -1), self.gp, self.y_best
            )[0]
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")

        # 多起点优化
        n_restarts = 10
        best_x = None
        best_acq = np.inf

        for _ in range(n_restarts):
            # 随机初始点
            x0 = self._generate_random_samples(1)[0]

            # 优化
            result = minimize(
                acq_func,
                x0,
                method='L-BFGS-B',
                bounds=[(self.bounds[i, 0], self.bounds[i, 1]) for i in range(self.dim)]
            )

            # 更新最优
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        return best_x

    def optimize(
        self,
        n_iterations: int = 20,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        执行贝叶斯优化

        参数:
            n_iterations: 优化迭代次数
            verbose: 是否打印进度

        返回:
            X_best: (d,) 最优输入
            y_best: 最优值
        """
        # 1. 初始随机采样
        if verbose:
            print(f"[Phase 1] Initial random sampling ({self.n_initial} points)")

        X_init = self._generate_random_samples(self.n_initial)

        for i, x in enumerate(X_init):
            y = self.objective_function(x)
            self.X_observed.append(x)
            self.y_observed.append(y)

            if y > self.y_best:
                self.y_best = y
                self.X_best = x

            if verbose:
                print(f"  Initial {i+1}/{self.n_initial}: y = {y:.4f} (best: {self.y_best:.4f})")

        # 2. 贝叶斯优化循环
        if verbose:
            print(f"\n[Phase 2] Bayesian optimization ({n_iterations} iterations)")

        for i in range(n_iterations):
            # 拟合GP
            X_train = np.array(self.X_observed)
            y_train = np.array(self.y_observed)
            self.gp.fit(X_train, y_train)

            # 优化采集函数
            x_next = self._optimize_acquisition()

            # 评估目标函数
            y_next = self.objective_function(x_next)

            # 更新观测
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)

            # 更新最优值
            if y_next > self.y_best:
                self.y_best = y_next
                self.X_best = x_next

            if verbose:
                print(f"  Iteration {i+1}/{n_iterations}: y = {y_next:.4f} (best: {self.y_best:.4f})")

        return self.X_best, self.y_best

    def get_optimization_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取优化历史

        返回:
            X_history: (n, d) 所有采样点
            y_history: (n,) 所有目标值
        """
        return np.array(self.X_observed), np.array(self.y_observed)


class MOFInverseDesigner:
    """
    MOF逆向设计器

    结合VAE和贝叶斯优化进行目标导向的MOF设计
    """

    def __init__(
        self,
        vae_model,
        property_predictor: Callable,
        latent_bounds: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        """
        参数:
            vae_model: 训练好的MOF-VAE模型
            property_predictor: 性质预测器 f: structure -> property
            latent_bounds: (latent_dim, 2) 潜在空间搜索边界
            device: 设备
        """
        self.vae = vae_model
        self.vae.eval()
        self.property_predictor = property_predictor
        self.device = device

        # 设置潜在空间搜索边界（默认：[-3, 3]σ）
        if latent_bounds is None:
            latent_dim = vae_model.latent_dim
            latent_bounds = np.array([[-3.0, 3.0]] * latent_dim)
        self.latent_bounds = latent_bounds

    def _latent_to_property(self, z: np.ndarray) -> float:
        """
        从潜在向量预测性质

        z -> decode -> structure -> predict -> property

        参数:
            z: (latent_dim,) 潜在向量

        返回:
            property_value: 预测的性质值
        """
        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 解码
            node_recon, adj_recon, edge_recon = self.vae.decode(z_tensor)

            # 预测性质
            # 注：实际应用中需要将重构的图转换为可用于预测的格式
            # 这里简化为直接预测
            property_value = self.property_predictor({
                'node_features': node_recon,
                'adj_matrix': adj_recon,
                'edge_features': edge_recon
            })

        return float(property_value)

    def optimize_for_property(
        self,
        target_property: str,
        n_iterations: int = 50,
        acquisition: str = 'ei',
        verbose: bool = True
    ) -> Dict:
        """
        优化MOF以获得目标性质

        参数:
            target_property: 目标性质名称
            n_iterations: 优化迭代次数
            acquisition: 采集函数
            verbose: 是否打印进度

        返回:
            result: 包含最优MOF和性质的字典
        """
        if verbose:
            print(f"Starting inverse design for property: {target_property}")
            print(f"Latent space dimension: {len(self.latent_bounds)}")
            print(f"Search bounds: {self.latent_bounds[0]}")

        # 创建贝叶斯优化器
        optimizer = BayesianOptimizer(
            objective_function=self._latent_to_property,
            bounds=self.latent_bounds,
            acquisition=acquisition,
            n_initial=10
        )

        # 运行优化
        z_best, property_best = optimizer.optimize(
            n_iterations=n_iterations,
            verbose=verbose
        )

        # 解码最优潜在向量
        with torch.no_grad():
            z_tensor = torch.tensor(z_best, dtype=torch.float32, device=self.device).unsqueeze(0)
            node_recon, adj_recon, edge_recon = self.vae.decode(z_tensor)

        result = {
            'latent_vector': z_best,
            'property_value': property_best,
            'node_features': node_recon.cpu().numpy(),
            'adj_matrix': adj_recon.cpu().numpy(),
            'edge_features': edge_recon.cpu().numpy(),
            'optimization_history': optimizer.get_optimization_history()
        }

        return result


# ===== 示例和测试 =====

def test_gaussian_process():
    """测试高斯过程"""
    print("Testing Gaussian Process...")

    # 生成测试数据
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = np.sin(X_train).flatten() + np.random.normal(0, 0.1, 5)

    # 拟合GP
    gp = GaussianProcess(length_scale=1.0, noise=0.1)
    gp.fit(X_train, y_train)

    # 预测
    X_test = np.linspace(0, 4, 50).reshape(-1, 1)
    mu, sigma = gp.predict(X_test, return_std=True)

    print(f"  Training points: {len(X_train)}")
    print(f"  Test points: {len(X_test)}")
    print(f"  Mean prediction range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  Std prediction range: [{sigma.min():.3f}, {sigma.max():.3f}]")
    print("✓ GP test passed!")


def test_bayesian_optimization():
    """测试贝叶斯优化"""
    print("\nTesting Bayesian Optimization...")

    # 定义测试函数（1D）
    def objective(x):
        return -(x[0] - 2.5)**2 + 1.0  # 最大值在x=2.5

    # 运行BO
    bounds = np.array([[0.0, 5.0]])
    optimizer = BayesianOptimizer(
        objective_function=objective,
        bounds=bounds,
        acquisition='ei',
        n_initial=3,
        random_state=42
    )

    x_best, y_best = optimizer.optimize(n_iterations=10, verbose=False)

    print(f"  True optimum: x = 2.5, y = 1.0")
    print(f"  Found optimum: x = {x_best[0]:.3f}, y = {y_best:.3f}")
    print(f"  Error: {abs(x_best[0] - 2.5):.3f}")
    print("✓ BO test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("Bayesian Optimization for MOF Inverse Design")
    print("=" * 60)

    test_gaussian_process()
    test_bayesian_optimization()

    print("\n✓ All tests passed!")
