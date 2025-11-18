"""
MOF Intelligent Agent System

集成文献挖掘、AutoML和自学习智能体的完整系统

包含：
1. LiteratureMiner - 文献挖掘与知识抽取
2. AutoMLOptimizer - 自适应模型优化
3. MOFAgent - 自学习型MOF发现智能体
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ========== 1. 文献挖掘工具 ==========

class LiteratureMiner:
    """
    文献挖掘与知识抽取

    功能：
    - 自动标注MOF文献
    - 提取结构信息
    - 提取合成条件
    - 提取性质数据
    - 生成知识图谱
    """

    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.knowledge_base = defaultdict(list)

    def mine_paper(self, paper_text: str) -> Dict:
        """从单篇论文提取MOF信息"""
        # 简化的提取逻辑（实际应使用LLM）
        extracted = {
            "mof_names": self._extract_mof_names(paper_text),
            "metals": self._extract_metals(paper_text),
            "linkers": self._extract_linkers(paper_text),
            "synthesis_conditions": self._extract_synthesis(paper_text),
            "properties": self._extract_properties(paper_text)
        }
        return extracted

    def _extract_mof_names(self, text: str) -> List[str]:
        """提取MOF名称"""
        # 简单的模式匹配
        import re
        patterns = [r'MOF-\d+', r'UiO-\d+', r'MIL-\d+', r'ZIF-\d+', r'HKUST-\d+']
        names = []
        for pattern in patterns:
            names.extend(re.findall(pattern, text))
        return list(set(names))

    def _extract_metals(self, text: str) -> List[str]:
        """提取金属中心"""
        metals = ['Zn', 'Cu', 'Zr', 'Fe', 'Co', 'Ni', 'Cr', 'Al', 'Mg']
        found = [m for m in metals if m in text or m.lower() in text.lower()]
        return found

    def _extract_linkers(self, text: str) -> List[str]:
        """提取有机配体"""
        linkers = ['BDC', 'BTC', 'DOBDC', 'NDC', 'BPDC']
        found = [l for l in linkers if l in text]
        return found

    def _extract_synthesis(self, text: str) -> Dict:
        """提取合成条件"""
        # 简化版：寻找温度和溶剂
        import re
        temp_match = re.search(r'(\d+)\s*[°]?C', text)
        temperature = temp_match.group(1) if temp_match else None

        solvents = ['DMF', 'DEF', 'methanol', 'ethanol', 'water']
        found_solvents = [s for s in solvents if s in text]

        return {
            "temperature": temperature,
            "solvents": found_solvents
        }

    def _extract_properties(self, text: str) -> Dict:
        """提取性质数据"""
        import re
        props = {}

        # 表面积
        sa_match = re.search(r'(\d+(?:\.\d+)?)\s*m[²2]/g', text)
        if sa_match:
            props['surface_area'] = float(sa_match.group(1))

        # CO2吸附
        co2_match = re.search(r'(\d+(?:\.\d+)?)\s*mmol/g.*CO[₂2]', text)
        if co2_match:
            props['co2_uptake'] = float(co2_match.group(1))

        return props

    def build_knowledge_graph(self, papers: List[str]) -> Dict:
        """从多篇论文构建知识图谱"""
        for paper in papers:
            info = self.mine_paper(paper)

            # 添加到知识库
            for mof_name in info['mof_names']:
                self.knowledge_base[mof_name].append({
                    'metals': info['metals'],
                    'linkers': info['linkers'],
                    'synthesis': info['synthesis_conditions'],
                    'properties': info['properties']
                })

        return dict(self.knowledge_base)

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """语义搜索相关MOF"""
        # 简化的搜索（实际应使用向量相似度）
        query_lower = query.lower()
        results = []

        for mof_name, entries in self.knowledge_base.items():
            score = 0
            for entry in entries:
                # 检查金属匹配
                if any(m.lower() in query_lower for m in entry['metals']):
                    score += 2
                # 检查配体匹配
                if any(l.lower() in query_lower for l in entry['linkers']):
                    score += 1

            if score > 0:
                results.append((mof_name, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in results[:top_k]]


# ========== 2. AutoML优化器 ==========

class AutoMLOptimizer:
    """
    AutoML自适应MOF筛选

    功能：
    - 自动特征工程
    - 自动模型选择
    - 超参数优化
    - 性能监控与调整
    """

    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.best_model = None
        self.best_score = -np.inf
        self.history = []

    def auto_optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_description: str,
        max_iterations: int = 10
    ) -> Dict:
        """自动优化MOF性质预测模型"""
        print(f"[AutoML] Starting optimization for: {task_description}")
        print(f"[AutoML] Data shape: X={X.shape}, y={y.shape}")

        # 1. 特征工程
        X_engineered = self.engineer_features(X, task_description)

        # 2. 模型搜索
        for i in range(max_iterations):
            # 选择模型和超参数
            model_config = self.suggest_model(i)

            # 训练和评估
            score = self.train_and_evaluate(X_engineered, y, model_config)

            # 记录
            self.history.append({
                'iteration': i,
                'model': model_config['name'],
                'score': score,
                'params': model_config['params']
            })

            # 更新最佳
            if score > self.best_score:
                self.best_score = score
                self.best_model = model_config

            print(f"  Iteration {i+1}/{max_iterations}: "
                  f"Model={model_config['name']}, Score={score:.4f}")

        print(f"\n[AutoML] Best score: {self.best_score:.4f}")
        print(f"[AutoML] Best model: {self.best_model['name']}")

        return {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'history': self.history
        }

    def engineer_features(self, X: np.ndarray, task_desc: str) -> np.ndarray:
        """自动特征工程"""
        # 简化：添加多项式特征
        X_eng = X.copy()

        # 平方特征
        X_squared = X ** 2
        X_eng = np.hstack([X_eng, X_squared])

        # 交互特征（前5维）
        if X.shape[1] >= 5:
            for i in range(min(5, X.shape[1])):
                for j in range(i+1, min(5, X.shape[1])):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    X_eng = np.hstack([X_eng, interaction])

        print(f"  [Feature Engineering] {X.shape[1]} → {X_eng.shape[1]} features")
        return X_eng

    def suggest_model(self, iteration: int) -> Dict:
        """建议模型配置"""
        models = [
            {'name': 'RandomForest', 'params': {'n_estimators': 100, 'max_depth': 10}},
            {'name': 'XGBoost', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
            {'name': 'Ridge', 'params': {'alpha': 1.0}},
            {'name': 'ElasticNet', 'params': {'alpha': 0.5, 'l1_ratio': 0.5}}
        ]
        return models[iteration % len(models)]

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, config: Dict) -> float:
        """训练并评估模型"""
        from sklearn.model_selection import cross_val_score

        # 创建模型
        if config['name'] == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**config['params'], random_state=42)
        elif config['name'] == 'XGBoost':
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(**config['params'], random_state=42)
            except:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
        elif config['name'] == 'Ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(**config['params'])
        else:  # ElasticNet
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(**config['params'])

        # 交叉验证
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return scores.mean()


# ========== 3. MOF智能体 ==========

class MOFAgent:
    """
    自学习型MOF发现智能体

    功能：
    - 自主文献调研
    - 设计新MOF
    - 预测性质
    - 学习和改进
    - 生成报告
    """

    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.literature_miner = LiteratureMiner(llm_model)
        self.automl = AutoMLOptimizer(llm_model)
        self.memory = []  # 经验记录
        self.discovered_mofs = []

    def discover_mof(
        self,
        goal: str,
        max_iterations: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        自主发现MOF

        Args:
            goal: 研究目标（如"Discover MOF with CO2 uptake > 5 mmol/g"）
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息

        Returns:
            discovery_report: 发现报告
        """
        if verbose:
            print("=" * 70)
            print("MOF Discovery Agent")
            print("=" * 70)
            print(f"\nGoal: {goal}\n")

        # 解析目标
        target_prop, target_value = self._parse_goal(goal)

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # 1. 文献调研
            relevant_mofs = self._literature_search(target_prop)
            if verbose:
                print(f"[Literature] Found {len(relevant_mofs)} relevant MOFs")

            # 2. 设计候选
            candidates = self._design_candidates(relevant_mofs, target_prop)
            if verbose:
                print(f"[Design] Generated {len(candidates)} candidates")

            # 3. 预测性质
            predictions = self._predict_properties(candidates, target_prop)

            # 4. 评估和选择
            best_candidate = max(predictions, key=lambda x: x['predicted_value'])

            if verbose:
                print(f"[Prediction] Best candidate: {best_candidate['name']}")
                print(f"             Predicted {target_prop}: {best_candidate['predicted_value']:.2f}")

            # 5. 记录发现
            self.discovered_mofs.append(best_candidate)
            self.memory.append({
                'iteration': iteration,
                'candidate': best_candidate,
                'achieved_goal': best_candidate['predicted_value'] >= target_value
            })

            # 检查是否达成目标
            if best_candidate['predicted_value'] >= target_value:
                if verbose:
                    print(f"\n✓ Goal achieved in iteration {iteration + 1}!")
                break

        # 生成报告
        report = self._generate_report(goal, target_value)

        return report

    def _parse_goal(self, goal: str) -> Tuple[str, float]:
        """解析目标"""
        import re
        # 简化的解析
        if "co2" in goal.lower():
            target_prop = "CO2_uptake"
            match = re.search(r'>\s*(\d+(?:\.\d+)?)', goal)
            target_value = float(match.group(1)) if match else 5.0
        else:
            target_prop = "surface_area"
            target_value = 3000.0

        return target_prop, target_value

    def _literature_search(self, target_prop: str) -> List[str]:
        """文献搜索"""
        # 模拟搜索结果
        if "co2" in target_prop.lower():
            return ["HKUST-1", "Mg-MOF-74", "UiO-66-NH2"]
        else:
            return ["MOF-5", "NU-1000", "MOF-177"]

    def _design_candidates(self, reference_mofs: List[str], target_prop: str) -> List[Dict]:
        """设计候选MOF"""
        candidates = []

        for i, ref_mof in enumerate(reference_mofs):
            # 基于参考MOF变异设计
            candidate = {
                'name': f"Design-{i+1}-variant",
                'base_mof': ref_mof,
                'modifications': ['functionalization', 'metal_substitution'][i % 2]
            }
            candidates.append(candidate)

        # 添加全新设计
        candidates.append({
            'name': "Novel-Design-1",
            'base_mof': None,
            'modifications': 'de_novo_design'
        })

        return candidates

    def _predict_properties(self, candidates: List[Dict], target_prop: str) -> List[Dict]:
        """预测候选MOF的性质"""
        predictions = []

        for cand in candidates:
            # 简化的预测（实际应使用ML模型）
            if "variant" in cand['name']:
                predicted_value = np.random.uniform(4.0, 6.5)
            else:
                predicted_value = np.random.uniform(3.5, 7.0)

            cand['predicted_value'] = predicted_value
            cand['property'] = target_prop
            predictions.append(cand)

        return predictions

    def _generate_report(self, goal: str, target_value: float) -> Dict:
        """生成发现报告"""
        success = any(m['achieved_goal'] for m in self.memory)

        report = {
            'goal': goal,
            'target_value': target_value,
            'success': success,
            'iterations': len(self.memory),
            'discovered_mofs': self.discovered_mofs,
            'best_mof': max(self.discovered_mofs, key=lambda x: x['predicted_value']),
            'learning_curve': [m['candidate']['predicted_value'] for m in self.memory]
        }

        return report

    def learn_from_experience(self):
        """从经验中学习"""
        # 分析成功和失败的设计
        successes = [m for m in self.memory if m['achieved_goal']]
        failures = [m for m in self.memory if not m['achieved_goal']]

        print(f"\n[Learning] Analyzed {len(self.memory)} iterations")
        print(f"  Successes: {len(successes)}")
        print(f"  Failures: {len(failures)}")

        # 提取设计规则（简化）
        if successes:
            print("\n[Insights] Successful strategies:")
            for s in successes[:3]:
                print(f"  - {s['candidate']['modifications']}")


# ========== 使用示例 ==========

def example_literature_mining():
    """示例：文献挖掘"""
    print("\n" + "="*70)
    print("Literature Mining Example")
    print("="*70)

    miner = LiteratureMiner()

    # 模拟论文文本
    papers = [
        "MOF-5 was synthesized using Zn metal and BDC linker in DMF at 120°C. "
        "The surface area was 3800 m²/g and CO₂ uptake was 4.5 mmol/g.",

        "UiO-66 contains Zr metal centers and BDC linkers. Synthesized in DMF at 120°C. "
        "Surface area: 1200 m²/g, CO₂ uptake: 3.0 mmol/g."
    ]

    # 构建知识图谱
    kg = miner.build_knowledge_graph(papers)

    print(f"\nExtracted knowledge from {len(papers)} papers:")
    for mof_name, info in kg.items():
        print(f"\n{mof_name}:")
        print(f"  Entries: {len(info)}")
        for entry in info[:1]:  # 显示第一个条目
            print(f"  Metals: {entry['metals']}")
            print(f"  Linkers: {entry['linkers']}")
            print(f"  Properties: {entry['properties']}")

    # 语义搜索
    query = "Zr-based MOF for CO2 capture"
    results = miner.semantic_search(query)
    print(f"\nSemantic search for '{query}':")
    print(f"  Results: {results}")


def example_automl():
    """示例：AutoML优化"""
    print("\n" + "="*70)
    print("AutoML Optimization Example")
    print("="*70)

    # 生成模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.5

    optimizer = AutoMLOptimizer()

    result = optimizer.auto_optimize(
        X, y,
        task_description="Predict CO2 uptake from MOF features",
        max_iterations=4
    )

    print(f"\nOptimization complete!")
    print(f"  Best model: {result['best_model']['name']}")
    print(f"  Best R² score: {result['best_score']:.4f}")


def example_mof_agent():
    """示例：MOF智能体"""
    print("\n" + "="*70)
    print("MOF Discovery Agent Example")
    print("="*70)

    agent = MOFAgent()

    # 运行发现任务
    report = agent.discover_mof(
        goal="Discover a MOF with CO2 uptake > 5.5 mmol/g",
        max_iterations=3,
        verbose=True
    )

    print("\n" + "="*70)
    print("Discovery Report")
    print("="*70)
    print(f"\nGoal: {report['goal']}")
    print(f"Success: {'✓ Yes' if report['success'] else '✗ No'}")
    print(f"Iterations: {report['iterations']}")
    print(f"\nBest MOF discovered:")
    print(f"  Name: {report['best_mof']['name']}")
    print(f"  Predicted value: {report['best_mof']['predicted_value']:.2f} mmol/g")
    print(f"  Learning curve: {[f'{v:.2f}' for v in report['learning_curve']]}")

    # 学习
    agent.learn_from_experience()


if __name__ == "__main__":
    # 运行所有示例
    example_literature_mining()
    example_automl()
    example_mof_agent()

    print("\n" + "="*70)
    print("✓ All LLM tools demonstrated successfully!")
    print("="*70)
