"""
MOF Property Prediction Model Trainer

支持多种传统机器学习算法：
- Linear Regression (Lasso, Ridge, ElasticNet)
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple, Optional, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Use: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Use: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Use: pip install catboost")


class MOFModelTrainer:
    """
    MOF性质预测模型训练器

    功能：
    - 支持多种算法
    - 自动特征缩放
    - 超参数优化
    - 交叉验证
    - 模型保存/加载
    """

    def __init__(self, random_state: int = 42):
        """
        初始化训练器

        参数:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.best_params = None

        # 可用模型列表
        self.available_models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
            'svr': SVR,
            'rf': RandomForestRegressor,
        }

        if XGBOOST_AVAILABLE:
            self.available_models['xgboost'] = xgb.XGBRegressor
        if LIGHTGBM_AVAILABLE:
            self.available_models['lightgbm'] = lgb.LGBMRegressor
        if CATBOOST_AVAILABLE:
            self.available_models['catboost'] = CatBoostRegressor

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """
        获取默认超参数网格

        参数:
            model_type: 模型类型

        返回:
            参数网格字典
        """
        params = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'elasticnet': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf'],
                'gamma': ['scale', 'auto']
            },
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 5, 10]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 1, 5]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            }
        }

        return params.get(model_type, {})

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        scaler_type: str = 'standard'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备数据：划分训练/测试集并缩放

        参数:
            X: 特征矩阵
            y: 目标变量
            test_size: 测试集比例
            scaler_type: 缩放器类型 ('standard' 或 'robust')

        返回:
            X_train, X_test, y_train, y_test
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # 特征缩放
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'xgboost',
        params: Optional[Dict[str, Any]] = None,
        optimize: bool = True,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        训练模型

        参数:
            X_train: 训练特征
            y_train: 训练目标
            model_type: 模型类型
            params: 自定义参数（None则使用默认参数）
            optimize: 是否进行超参数优化
            cv: 交叉验证折数
            n_jobs: 并行任务数
            verbose: 详细程度

        返回:
            训练结果字典
        """
        if model_type not in self.available_models:
            raise ValueError(f"Model type '{model_type}' not available. "
                           f"Choose from: {list(self.available_models.keys())}")

        # 获取模型类
        model_class = self.available_models[model_type]

        # 设置基本参数
        base_params = {'random_state': self.random_state} if 'random_state' in model_class().get_params() else {}

        # 特殊处理某些模型
        if model_type == 'catboost':
            base_params['verbose'] = 0
        elif model_type in ['xgboost', 'lightgbm']:
            base_params['n_jobs'] = n_jobs
        elif model_type == 'rf':
            base_params['n_jobs'] = n_jobs

        if optimize and params is None:
            # 超参数优化
            param_grid = self.get_default_params(model_type)

            if not param_grid:
                print(f"No default parameter grid for {model_type}. Using default settings.")
                self.model = model_class(**base_params)
                self.model.fit(X_train, y_train)
            else:
                print(f"Optimizing hyperparameters for {model_type}...")
                grid_search = GridSearchCV(
                    estimator=model_class(**base_params),
                    param_grid=param_grid,
                    cv=cv,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                grid_search.fit(X_train, y_train)

                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_

                print(f"Best parameters: {self.best_params}")
                print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

        else:
            # 使用指定参数或默认参数
            if params:
                model_params = {**base_params, **params}
            else:
                model_params = base_params

            self.model = model_class(**model_params)

            if verbose:
                print(f"Training {model_type} with parameters: {model_params}")

            self.model.fit(X_train, y_train)

        # 交叉验证评估
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=n_jobs
        )

        results = {
            'model_type': model_type,
            'best_params': self.best_params,
            'cv_rmse_mean': -cv_scores.mean(),
            'cv_rmse_std': cv_scores.std(),
            'cv_scores': -cv_scores
        }

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        参数:
            X: 特征矩阵

        返回:
            预测值
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取特征重要性

        参数:
            top_n: 返回前N个重要特征（None则返回全部）

        返回:
            特征重要性 DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        # 尝试获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not support feature importance.")

        # 创建 DataFrame
        if self.feature_names is not None:
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
        else:
            df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            })

        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n)

        return df

    def save_model(self, filepath: str):
        """
        保存模型

        参数:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("No model to save.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        加载模型

        参数:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params')

        print(f"Model loaded from {filepath}")


def train_multiple_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_types: list = ['linear', 'ridge', 'rf', 'xgboost'],
    test_size: float = 0.2,
    optimize: bool = True,
    cv: int = 5
) -> Dict[str, MOFModelTrainer]:
    """
    训练多个模型并比较

    参数:
        X: 特征矩阵
        y: 目标变量
        model_types: 要训练的模型类型列表
        test_size: 测试集比例
        optimize: 是否优化超参数
        cv: 交叉验证折数

    返回:
        模型字典 {model_type: trainer}
    """
    from .model_evaluator import ModelEvaluator

    # 准备数据
    trainer = MOFModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=test_size)

    results = {}
    evaluator = ModelEvaluator()

    print(f"\nTraining {len(model_types)} models...\n")
    print("="*80)

    for model_type in model_types:
        print(f"\n[{model_type.upper()}]")
        print("-"*80)

        # 创建训练器
        trainer = MOFModelTrainer()
        trainer.scaler = evaluator.scaler  # 共享scaler
        trainer.feature_names = evaluator.feature_names if hasattr(evaluator, 'feature_names') else None

        # 训练
        train_results = trainer.train(
            X_train, y_train,
            model_type=model_type,
            optimize=optimize,
            cv=cv,
            verbose=0
        )

        # 评估
        y_pred = trainer.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred)

        print(f"CV RMSE: {train_results['cv_rmse_mean']:.4f} ± {train_results['cv_rmse_std']:.4f}")
        print(f"Test R²: {metrics['r2']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test MAE: {metrics['mae']:.4f}")

        results[model_type] = {
            'trainer': trainer,
            'train_results': train_results,
            'test_metrics': metrics,
            'predictions': y_pred
        }

    print("\n" + "="*80)
    print("\nModel Comparison:")
    print("-"*80)

    comparison_df = pd.DataFrame({
        model: {
            'CV_RMSE': res['train_results']['cv_rmse_mean'],
            'Test_R2': res['test_metrics']['r2'],
            'Test_RMSE': res['test_metrics']['rmse'],
            'Test_MAE': res['test_metrics']['mae']
        }
        for model, res in results.items()
    }).T

    comparison_df = comparison_df.sort_values('Test_RMSE')
    print(comparison_df)

    return results


if __name__ == '__main__':
    # 示例用法
    print("MOF Model Trainer")
    print(f"Available models: {list(MOFModelTrainer().available_models.keys())}")
