"""
Model Evaluator for MOF Property Prediction

提供模型评估和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    模型评估器

    功能：
    - 计算评估指标
    - 绘制性能图表
    - 学习曲线分析
    - 预测误差分析
    """

    def __init__(self):
        """初始化评估器"""
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归评估指标

        参数:
            y_true: 真实值
            y_pred: 预测值

        返回:
            指标字典
        """
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }

        # MAPE (避免除零)
        try:
            if np.all(y_true != 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            else:
                metrics['mape'] = np.nan
        except:
            metrics['mape'] = np.nan

        return metrics

    def print_metrics(self, metrics: Dict[str, float], title: str = "Model Performance"):
        """
        打印评估指标

        参数:
            metrics: 指标字典
            title: 标题
        """
        print(f"\n{title}")
        print("="*50)
        print(f"R² Score:  {metrics['r2']:.4f}")
        print(f"RMSE:      {metrics['rmse']:.4f}")
        print(f"MAE:       {metrics['mae']:.4f}")
        print(f"MSE:       {metrics['mse']:.4f}")
        if not np.isnan(metrics.get('mape', np.nan)):
            print(f"MAPE:      {metrics['mape']:.2f}%")
        print("="*50)

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs True Values",
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ):
        """
        绘制预测值 vs 真实值图

        参数:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 计算指标
        metrics = self.calculate_metrics(y_true, y_pred)

        # 1. 散点图
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=30)

        # 添加理想线（y=x）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Predictions vs True\nR²={metrics["r2"]:.4f}, RMSE={metrics["rmse"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 残差图
        ax2 = axes[1]
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ):
        """
        绘制误差分布图

        参数:
            y_true: 真实值
            y_pred: 预测值
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        residuals = y_pred - y_true
        abs_errors = np.abs(residuals)
        rel_errors = abs_errors / (np.abs(y_true) + 1e-10) * 100  # 避免除零

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. 残差直方图
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Residuals Distribution\nMean={residuals.mean():.4f}')

        # 2. 绝对误差直方图
        ax2 = axes[1]
        ax2.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Absolute Error Distribution\nMAE={abs_errors.mean():.4f}')

        # 3. 相对误差直方图
        ax3 = axes[2]
        # 限制相对误差范围以便可视化
        rel_errors_clipped = np.clip(rel_errors, 0, np.percentile(rel_errors, 95))
        ax3.hist(rel_errors_clipped, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Relative Error Distribution\nMedian={np.median(rel_errors):.2f}%')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_learning_curve(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        n_jobs: int = -1,
        train_sizes: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制学习曲线

        参数:
            estimator: 模型
            X: 特征矩阵
            y: 目标变量
            cv: 交叉验证折数
            n_jobs: 并行任务数
            train_sizes: 训练集大小（比例）
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        print("Computing learning curve...")
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=n_jobs,
            verbose=0
        )

        # 转换为正值（RMSE）
        train_scores = -train_scores
        val_scores = -val_scores

        # 计算均值和标准差
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # 绘图
        plt.figure(figsize=figsize)

        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue',
                label='Training score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2, color='blue')

        plt.plot(train_sizes_abs, val_mean, 'o-', color='red',
                label='Cross-validation score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.2, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # 添加诊断信息
        final_gap = val_mean[-1] - train_mean[-1]
        if final_gap > train_mean[-1] * 0.3:
            diagnosis = "High Variance (Overfitting)"
        elif train_mean[-1] > val_mean[0] * 0.5:
            diagnosis = "High Bias (Underfitting)"
        else:
            diagnosis = "Good Fit"

        plt.text(0.02, 0.98, f'Diagnosis: {diagnosis}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_validation_curve(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: np.ndarray,
        cv: int = 5,
        n_jobs: int = -1,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制验证曲线（参数影响）

        参数:
            estimator: 模型
            X: 特征矩阵
            y: 目标变量
            param_name: 参数名称
            param_range: 参数范围
            cv: 交叉验证折数
            n_jobs: 并行任务数
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        print(f"Computing validation curve for {param_name}...")
        train_scores, val_scores = validation_curve(
            estimator, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=n_jobs
        )

        # 转换为正值
        train_scores = -train_scores
        val_scores = -val_scores

        # 计算均值和标准差
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # 绘图
        plt.figure(figsize=figsize)

        plt.plot(param_range, train_mean, 'o-', color='blue',
                label='Training score', linewidth=2, markersize=8)
        plt.fill_between(param_range,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2, color='blue')

        plt.plot(param_range, val_mean, 'o-', color='red',
                label='Cross-validation score', linewidth=2, markersize=8)
        plt.fill_between(param_range,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.2, color='red')

        plt.xlabel(param_name)
        plt.ylabel('RMSE')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # 标记最佳参数
        best_idx = np.argmin(val_mean)
        best_param = param_range[best_idx]
        plt.axvline(x=best_param, color='green', linestyle='--',
                   label=f'Best: {best_param}')
        plt.legend(loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        metric: str = 'rmse',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        比较多个模型的性能

        参数:
            results_dict: 结果字典 {model_name: {'y_true': ..., 'y_pred': ...}}
            metric: 比较指标 ('r2', 'rmse', 'mae')
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        model_names = []
        metric_values = []

        for model_name, data in results_dict.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            metrics = self.calculate_metrics(y_true, y_pred)
            model_names.append(model_name)
            metric_values.append(metrics[metric])

        # 创建条形图
        plt.figure(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(model_names, metric_values, color=colors, alpha=0.8, edgecolor='black')

        # 在条形上添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 标记最佳模型
        if metric == 'r2':
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)

        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制特征重要性图

        参数:
            importance_df: 特征重要性 DataFrame (需包含 'feature' 和 'importance' 列)
            top_n: 显示前N个特征
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        # 选择前N个
        df_plot = importance_df.head(top_n).copy()

        plt.figure(figsize=figsize)

        # 创建水平条形图
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_plot)))
        bars = plt.barh(df_plot['feature'], df_plot['importance'],
                       color=colors, edgecolor='black', alpha=0.8)

        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()  # 最重要的在顶部
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


if __name__ == '__main__':
    # 示例用法
    evaluator = ModelEvaluator()

    # 生成示例数据
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 50
    y_pred = y_true + np.random.randn(100) * 2

    # 计算指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics)

    # 绘图
    evaluator.plot_predictions(y_true, y_pred)
    evaluator.plot_error_distribution(y_true, y_pred)
