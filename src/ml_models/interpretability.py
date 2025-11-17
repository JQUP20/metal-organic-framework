"""
SHAP-based Model Interpretability Analysis

提供基于 SHAP 的模型可解释性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Use: pip install shap")
    print("Model interpretability features will be limited.")


class SHAPAnalyzer:
    """
    SHAP 可解释性分析器

    功能：
    - 计算 SHAP 值
    - 绘制多种 SHAP 可视化图表
    - 特征贡献分析
    - 特征交互分析
    """

    def __init__(self, model, X: Union[np.ndarray, pd.DataFrame],
                 feature_names: Optional[List[str]] = None):
        """
        初始化 SHAP 分析器

        参数:
            model: 训练好的模型
            X: 特征矩阵（用于计算 SHAP 值）
            feature_names: 特征名称列表
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required. Install with: pip install shap")

        self.model = model
        self.X = X
        self.feature_names = feature_names

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        self.explainer = None
        self.shap_values = None

    def compute_shap_values(
        self,
        X_explain: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        max_samples: int = 1000,
        algorithm: str = 'auto'
    ) -> np.ndarray:
        """
        计算 SHAP 值

        参数:
            X_explain: 要解释的样本（None则使用初始化时的X）
            max_samples: 最大样本数（大数据集时采样）
            algorithm: 算法类型 ('auto', 'tree', 'linear', 'kernel')

        返回:
            SHAP 值矩阵
        """
        if X_explain is None:
            X_explain = self.X

        # 采样（如果样本太多）
        if len(X_explain) > max_samples:
            print(f"Sampling {max_samples} from {len(X_explain)} samples for SHAP computation...")
            indices = np.random.choice(len(X_explain), max_samples, replace=False)
            X_explain = X_explain[indices] if isinstance(X_explain, np.ndarray) else X_explain.iloc[indices]

        # 选择合适的 explainer
        if algorithm == 'auto':
            # 自动选择
            if hasattr(self.model, 'tree_'):
                algorithm = 'tree'
            elif hasattr(self.model, 'feature_importances_'):
                algorithm = 'tree'
            elif hasattr(self.model, 'coef_'):
                algorithm = 'linear'
            else:
                algorithm = 'kernel'

        print(f"Using {algorithm} explainer...")

        if algorithm == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif algorithm == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_explain)
        elif algorithm == 'kernel':
            # Kernel SHAP 较慢，使用背景样本
            background = shap.sample(X_explain, min(100, len(X_explain)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # 计算 SHAP 值
        print("Computing SHAP values...")
        if algorithm == 'kernel':
            self.shap_values = self.explainer.shap_values(X_explain, silent=True)
        else:
            self.shap_values = self.explainer.shap_values(X_explain)

        print(f"SHAP values computed for {len(X_explain)} samples.")
        return self.shap_values

    def summary_plot(
        self,
        plot_type: str = 'dot',
        max_display: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制 SHAP 摘要图（蜂群图或条形图）

        参数:
            plot_type: 图表类型 ('dot', 'bar', 'violin')
            max_display: 显示的最大特征数
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        plt.figure(figsize=figsize)

        shap.summary_plot(
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def waterfall_plot(
        self,
        sample_idx: int = 0,
        max_display: int = 20,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制单个样本的瀑布图

        参数:
            sample_idx: 样本索引
            max_display: 显示的最大特征数
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        plt.figure(figsize=figsize)

        # 创建 Explanation 对象
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X[sample_idx] if isinstance(self.X, np.ndarray) else self.X.iloc[sample_idx].values,
                feature_names=self.feature_names
            )
            shap.waterfall_plot(explanation, max_display=max_display, show=False)
        else:
            # 旧版本 SHAP
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                self.X[sample_idx] if isinstance(self.X, np.ndarray) else self.X.iloc[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def dependence_plot(
        self,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = 'auto',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制 SHAP 依赖图

        参数:
            feature: 特征名称或索引
            interaction_feature: 交互特征（'auto' 自动选择）
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        plt.figure(figsize=figsize)

        shap.dependence_plot(
            feature,
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def force_plot(
        self,
        sample_idx: int = 0,
        matplotlib: bool = True,
        save_path: Optional[str] = None
    ):
        """
        绘制力图（单个样本）

        参数:
            sample_idx: 样本索引
            matplotlib: 使用 matplotlib（否则返回 HTML）
            save_path: 保存路径（可选，仅matplotlib=True时有效）
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        X_sample = self.X[sample_idx] if isinstance(self.X, np.ndarray) else self.X.iloc[sample_idx]

        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            X_sample,
            feature_names=self.feature_names,
            matplotlib=matplotlib,
            show=not matplotlib
        )

        if matplotlib:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            plt.show()
        else:
            return force_plot

    def feature_importance(
        self,
        method: str = 'mean_abs',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        基于 SHAP 值计算特征重要性

        参数:
            method: 计算方法
                   - 'mean_abs': 平均绝对 SHAP 值
                   - 'mean': 平均 SHAP 值（考虑方向）
                   - 'std': SHAP 值标准差
            top_n: 返回前N个特征（None则全部）

        返回:
            特征重要性 DataFrame
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        if method == 'mean_abs':
            importance = np.abs(self.shap_values).mean(axis=0)
        elif method == 'mean':
            importance = self.shap_values.mean(axis=0)
        elif method == 'std':
            importance = self.shap_values.std(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 创建 DataFrame
        if self.feature_names is not None:
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })

        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n)

        return df

    def interaction_values(
        self,
        X_explain: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        max_samples: int = 500
    ) -> np.ndarray:
        """
        计算 SHAP 交互值

        参数:
            X_explain: 要解释的样本
            max_samples: 最大样本数

        返回:
            交互值矩阵 (n_samples, n_features, n_features)
        """
        if X_explain is None:
            X_explain = self.X

        # 采样
        if len(X_explain) > max_samples:
            print(f"Sampling {max_samples} from {len(X_explain)} samples...")
            indices = np.random.choice(len(X_explain), max_samples, replace=False)
            X_explain = X_explain[indices] if isinstance(X_explain, np.ndarray) else X_explain.iloc[indices]

        # 只有 TreeExplainer 支持交互值
        if not isinstance(self.explainer, shap.TreeExplainer):
            print("Computing tree explainer for interaction values...")
            self.explainer = shap.TreeExplainer(self.model)

        print("Computing SHAP interaction values (this may take a while)...")
        interaction_values = self.explainer.shap_interaction_values(X_explain)

        return interaction_values

    def plot_interaction_heatmap(
        self,
        interaction_values: np.ndarray,
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        绘制特征交互热力图

        参数:
            interaction_values: SHAP 交互值
            top_n: 显示前N个特征
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        # 计算平均交互强度
        interaction_matrix = np.abs(interaction_values).mean(axis=0)

        # 选择最重要的特征
        main_effects = np.abs(np.diagonal(interaction_matrix))
        top_indices = np.argsort(main_effects)[-top_n:][::-1]

        # 子集
        interaction_subset = interaction_matrix[top_indices][:, top_indices]

        # 特征名称
        if self.feature_names is not None:
            labels = [self.feature_names[i] for i in top_indices]
        else:
            labels = [f'F{i}' for i in top_indices]

        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(
            interaction_subset,
            xticklabels=labels,
            yticklabels=labels,
            cmap='RdYlBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Mean |SHAP interaction|'}
        )
        plt.title(f'Top {top_n} Feature Interactions')
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def decision_plot(
        self,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 100,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制决策图

        参数:
            sample_indices: 样本索引列表（None则随机选择）
            max_samples: 最大样本数
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        if sample_indices is None:
            n_samples = min(max_samples, len(self.shap_values))
            sample_indices = np.random.choice(len(self.shap_values), n_samples, replace=False)

        plt.figure(figsize=figsize)

        shap.decision_plot(
            self.explainer.expected_value,
            self.shap_values[sample_indices],
            self.X[sample_indices] if isinstance(self.X, np.ndarray) else self.X.iloc[sample_indices],
            feature_names=self.feature_names,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


class PermutationImportance:
    """
    排列重要性（模型无关方法）
    """

    def __init__(self, model, X: np.ndarray, y: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 metric: str = 'rmse'):
        """
        初始化

        参数:
            model: 训练好的模型
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
            metric: 评估指标 ('rmse', 'mae', 'r2')
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.metric = metric

    def compute_importance(self, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
        """
        计算排列重要性

        参数:
            n_repeats: 排列重复次数
            random_state: 随机种子

        返回:
            重要性 DataFrame
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # 选择指标函数
        if self.metric == 'rmse':
            score_func = lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mae':
            score_func = lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)
        elif self.metric == 'r2':
            score_func = r2_score
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # 基线得分
        y_pred = self.model.predict(self.X)
        baseline_score = score_func(self.y, y_pred)

        # 计算每个特征的重要性
        importances = []
        np.random.seed(random_state)

        for i, feature in enumerate(self.feature_names):
            scores = []

            for _ in range(n_repeats):
                # 复制数据
                X_permuted = self.X.copy()

                # 随机打乱特征 i
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

                # 计算得分
                y_pred_permuted = self.model.predict(X_permuted)
                score = score_func(self.y, y_pred_permuted)
                scores.append(baseline_score - score)

            importances.append({
                'feature': feature,
                'importance_mean': np.mean(scores),
                'importance_std': np.std(scores)
            })

        df = pd.DataFrame(importances)
        df = df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

        return df


if __name__ == '__main__':
    print("SHAP Interpretability Analyzer")
    if SHAP_AVAILABLE:
        print("SHAP library is available.")
    else:
        print("SHAP library is NOT available. Install with: pip install shap")
