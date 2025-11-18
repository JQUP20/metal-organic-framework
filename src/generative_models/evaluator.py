"""
生成模型评估工具

评估生成MOF的质量、多样性、新颖性等指标。

作者: AI & MOF Course
日期: 2024-11
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from scipy.spatial.distance import pdist, cdist
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import pairwise_distances


class GeneratedMOFEvaluator:
    """
    生成MOF评估器

    评估生成MOF的各种指标
    """

    def __init__(self, verbose: bool = True):
        """
        参数:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose

    def evaluate_validity(
        self,
        generated_structures: List[Dict],
        validator: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        评估生成结构的有效性

        参数:
            generated_structures: 生成的结构列表
            validator: 验证函数（可选）

        返回:
            validity_metrics: 有效性指标字典
        """
        if validator is None:
            # 默认简单验证器
            validator = self._default_validator

        valid_count = 0
        invalid_reasons = []

        for struct in generated_structures:
            is_valid, reason = validator(struct)
            if is_valid:
                valid_count += 1
            else:
                invalid_reasons.append(reason)

        validity_rate = valid_count / len(generated_structures)

        if self.verbose:
            print(f"有效性评估:")
            print(f"  总样本数: {len(generated_structures)}")
            print(f"  有效样本数: {valid_count}")
            print(f"  有效率: {validity_rate:.2%}")

            if invalid_reasons:
                print(f"  无效原因统计:")
                unique_reasons, counts = np.unique(invalid_reasons, return_counts=True)
                for reason, count in zip(unique_reasons, counts):
                    print(f"    - {reason}: {count}")

        return {
            'validity_rate': validity_rate,
            'valid_count': valid_count,
            'total_count': len(generated_structures),
            'invalid_reasons': invalid_reasons
        }

    def _default_validator(self, structure: Dict) -> Tuple[bool, str]:
        """
        默认验证器

        检查基本的结构完整性
        """
        # 检查必要的键
        if 'node_features' not in structure:
            return False, "missing_node_features"

        if 'adj_matrix' not in structure:
            return False, "missing_adj_matrix"

        node_features = structure['node_features']
        adj_matrix = structure['adj_matrix']

        # 检查形状一致性
        if node_features.shape[0] != adj_matrix.shape[0]:
            return False, "shape_mismatch"

        # 检查是否有NaN或Inf
        if torch.isnan(node_features).any() or torch.isinf(node_features).any():
            return False, "invalid_node_values"

        # 检查邻接矩阵是否对称
        if not torch.allclose(adj_matrix, adj_matrix.T, atol=1e-5):
            return False, "non_symmetric_adj"

        # 检查是否有足够的边
        num_edges = (adj_matrix > 0.5).sum().item()
        num_nodes = node_features.shape[0]

        if num_edges < num_nodes - 1:  # 至少需要形成连通图
            return False, "too_few_edges"

        if num_edges > num_nodes * (num_nodes - 1):  # 不能超过完全图
            return False, "too_many_edges"

        return True, "valid"

    def evaluate_uniqueness(
        self,
        generated_structures: List[Dict],
        similarity_threshold: float = 0.95
    ) -> Dict[str, float]:
        """
        评估生成结构的唯一性

        参数:
            generated_structures: 生成的结构列表
            similarity_threshold: 相似度阈值（超过此值认为是重复）

        返回:
            uniqueness_metrics: 唯一性指标字典
        """
        if len(generated_structures) <= 1:
            return {'uniqueness_rate': 1.0, 'unique_count': len(generated_structures)}

        # 提取特征向量用于比较
        feature_vectors = []
        for struct in generated_structures:
            # 使用节点特征的统计量和邻接矩阵的统计量作为fingerprint
            node_feat = struct['node_features']
            adj_mat = struct['adj_matrix']

            fingerprint = torch.cat([
                node_feat.mean(dim=0),
                node_feat.std(dim=0),
                torch.tensor([adj_mat.sum().item(), adj_mat.max().item()])
            ])

            feature_vectors.append(fingerprint.cpu().numpy())

        feature_vectors = np.array(feature_vectors)

        # 计算成对相似度
        distances = pairwise_distances(feature_vectors, metric='euclidean')

        # 归一化到[0, 1]
        max_dist = distances.max()
        if max_dist > 0:
            similarities = 1 - (distances / max_dist)
        else:
            similarities = np.ones_like(distances)

        # 找出独特的结构
        unique_mask = np.ones(len(generated_structures), dtype=bool)

        for i in range(len(generated_structures)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(generated_structures)):
                if unique_mask[j] and similarities[i, j] > similarity_threshold:
                    unique_mask[j] = False

        unique_count = unique_mask.sum()
        uniqueness_rate = unique_count / len(generated_structures)

        if self.verbose:
            print(f"\n唯一性评估:")
            print(f"  总样本数: {len(generated_structures)}")
            print(f"  独特样本数: {unique_count}")
            print(f"  唯一率: {uniqueness_rate:.2%}")
            print(f"  相似度阈值: {similarity_threshold}")

        return {
            'uniqueness_rate': uniqueness_rate,
            'unique_count': unique_count,
            'total_count': len(generated_structures),
            'similarity_threshold': similarity_threshold,
            'similarity_matrix': similarities
        }

    def evaluate_diversity(
        self,
        generated_structures: List[Dict]
    ) -> Dict[str, float]:
        """
        评估生成结构的多样性

        参数:
            generated_structures: 生成的结构列表

        返回:
            diversity_metrics: 多样性指标字典
        """
        if len(generated_structures) <= 1:
            return {'diversity_score': 0.0}

        # 提取特征
        feature_vectors = []
        for struct in generated_structures:
            node_feat = struct['node_features']
            adj_mat = struct['adj_matrix']

            # 更丰富的特征
            fingerprint = torch.cat([
                node_feat.mean(dim=0),
                node_feat.std(dim=0),
                node_feat.min(dim=0)[0],
                node_feat.max(dim=0)[0],
                torch.tensor([
                    adj_mat.sum().item(),
                    (adj_mat > 0.5).float().sum().item(),  # 边数
                    adj_mat.sum(dim=0).max().item(),  # 最大度数
                    adj_mat.sum(dim=0).mean().item()  # 平均度数
                ])
            ])

            feature_vectors.append(fingerprint.cpu().numpy())

        feature_vectors = np.array(feature_vectors)

        # 计算多样性指标

        # 1. 平均成对距离（Diversity Score）
        pairwise_dists = pdist(feature_vectors, metric='euclidean')
        avg_distance = pairwise_dists.mean()

        # 2. 特征空间覆盖率（通过PCA解释的方差）
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(feature_vectors)
        explained_variance = pca.explained_variance_ratio_

        # 3. 熵（特征分布的均匀程度）
        # 对每个特征维度计算熵
        entropies = []
        for dim in range(feature_vectors.shape[1]):
            hist, _ = np.histogram(feature_vectors[:, dim], bins=20)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 避免log(0)
            entropy = -np.sum(hist * np.log(hist))
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)

        if self.verbose:
            print(f"\n多样性评估:")
            print(f"  平均成对距离: {avg_distance:.4f}")
            print(f"  PC1解释方差: {explained_variance[0]:.2%}")
            print(f"  特征空间覆盖: {1 - explained_variance[0]:.2%}")
            print(f"  平均熵: {mean_entropy:.4f}")

        return {
            'diversity_score': avg_distance,
            'pairwise_distances': pairwise_dists,
            'pca_variance_ratio': explained_variance,
            'mean_entropy': mean_entropy,
            'coverage': 1 - explained_variance[0]  # 覆盖率
        }

    def evaluate_novelty(
        self,
        generated_structures: List[Dict],
        reference_structures: List[Dict],
        k: int = 5
    ) -> Dict[str, float]:
        """
        评估生成结构的新颖性（与参考集的差异）

        参数:
            generated_structures: 生成的结构
            reference_structures: 参考结构（如训练集）
            k: k-最近邻

        返回:
            novelty_metrics: 新颖性指标字典
        """
        # 提取特征
        def extract_features(structures):
            features = []
            for struct in structures:
                node_feat = struct['node_features']
                adj_mat = struct['adj_matrix']

                fingerprint = torch.cat([
                    node_feat.mean(dim=0),
                    node_feat.std(dim=0),
                    torch.tensor([
                        adj_mat.sum().item(),
                        (adj_mat > 0.5).float().sum().item()
                    ])
                ])

                features.append(fingerprint.cpu().numpy())

            return np.array(features)

        gen_features = extract_features(generated_structures)
        ref_features = extract_features(reference_structures)

        # 计算每个生成样本到参考集的距离
        distances = cdist(gen_features, ref_features, metric='euclidean')

        # k-最近邻距离
        k = min(k, ref_features.shape[0])
        knn_distances = np.partition(distances, k-1, axis=1)[:, :k]
        avg_knn_distance = knn_distances.mean(axis=1)

        # 最近邻距离
        nearest_distances = distances.min(axis=1)

        # 新颖性得分（平均k-NN距离）
        novelty_score = avg_knn_distance.mean()

        if self.verbose:
            print(f"\n新颖性评估:")
            print(f"  生成样本数: {len(generated_structures)}")
            print(f"  参考样本数: {len(reference_structures)}")
            print(f"  k值: {k}")
            print(f"  平均{k}-NN距离: {novelty_score:.4f}")
            print(f"  平均最近邻距离: {nearest_distances.mean():.4f}")
            print(f"  最小最近邻距离: {nearest_distances.min():.4f}")

        return {
            'novelty_score': novelty_score,
            'knn_distances': avg_knn_distance,
            'nearest_distances': nearest_distances,
            'k': k
        }

    def evaluate_property_distribution(
        self,
        generated_properties: np.ndarray,
        reference_properties: np.ndarray,
        property_name: str = 'Property'
    ) -> Dict[str, float]:
        """
        评估生成MOF的性质分布与参考分布的差异

        参数:
            generated_properties: 生成MOF的性质值
            reference_properties: 参考MOF的性质值
            property_name: 性质名称

        返回:
            distribution_metrics: 分布指标字典
        """
        # Wasserstein距离（Earth Mover's Distance）
        wasserstein_dist = wasserstein_distance(generated_properties, reference_properties)

        # Kolmogorov-Smirnov检验
        ks_statistic, ks_pvalue = ks_2samp(generated_properties, reference_properties)

        # 均值和标准差差异
        mean_diff = abs(generated_properties.mean() - reference_properties.mean())
        std_diff = abs(generated_properties.std() - reference_properties.std())

        # 范围覆盖
        gen_range = (generated_properties.min(), generated_properties.max())
        ref_range = (reference_properties.min(), reference_properties.max())

        range_overlap = (
            max(gen_range[0], ref_range[0]),
            min(gen_range[1], ref_range[1])
        )

        if range_overlap[1] > range_overlap[0]:
            overlap_size = range_overlap[1] - range_overlap[0]
            ref_size = ref_range[1] - ref_range[0]
            coverage = overlap_size / ref_size if ref_size > 0 else 0
        else:
            coverage = 0

        if self.verbose:
            print(f"\n性质分布评估 ({property_name}):")
            print(f"  生成: μ={generated_properties.mean():.3f}, "
                  f"σ={generated_properties.std():.3f}, "
                  f"范围=[{gen_range[0]:.3f}, {gen_range[1]:.3f}]")
            print(f"  参考: μ={reference_properties.mean():.3f}, "
                  f"σ={reference_properties.std():.3f}, "
                  f"范围=[{ref_range[0]:.3f}, {ref_range[1]:.3f}]")
            print(f"  Wasserstein距离: {wasserstein_dist:.4f}")
            print(f"  KS统计量: {ks_statistic:.4f} (p={ks_pvalue:.4f})")
            print(f"  均值差异: {mean_diff:.4f}")
            print(f"  标准差差异: {std_diff:.4f}")
            print(f"  范围覆盖率: {coverage:.2%}")

        return {
            'wasserstein_distance': wasserstein_dist,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'range_coverage': coverage,
            'generated_stats': {
                'mean': generated_properties.mean(),
                'std': generated_properties.std(),
                'min': gen_range[0],
                'max': gen_range[1]
            },
            'reference_stats': {
                'mean': reference_properties.mean(),
                'std': reference_properties.std(),
                'min': ref_range[0],
                'max': ref_range[1]
            }
        }

    def comprehensive_evaluation(
        self,
        generated_structures: List[Dict],
        reference_structures: List[Dict],
        generated_properties: Optional[np.ndarray] = None,
        reference_properties: Optional[np.ndarray] = None,
        property_name: str = 'Property'
    ) -> Dict[str, Dict]:
        """
        综合评估

        参数:
            generated_structures: 生成的结构
            reference_structures: 参考结构
            generated_properties: 生成MOF的性质（可选）
            reference_properties: 参考MOF的性质（可选）
            property_name: 性质名称

        返回:
            comprehensive_metrics: 综合指标字典
        """
        print("=" * 60)
        print("综合评估报告")
        print("=" * 60)

        results = {}

        # 1. 有效性
        print("\n【1/5】有效性评估")
        results['validity'] = self.evaluate_validity(generated_structures)

        # 2. 唯一性
        print("\n【2/5】唯一性评估")
        results['uniqueness'] = self.evaluate_uniqueness(generated_structures)

        # 3. 多样性
        print("\n【3/5】多样性评估")
        results['diversity'] = self.evaluate_diversity(generated_structures)

        # 4. 新颖性
        print("\n【4/5】新颖性评估")
        results['novelty'] = self.evaluate_novelty(
            generated_structures,
            reference_structures
        )

        # 5. 性质分布（如果提供）
        if generated_properties is not None and reference_properties is not None:
            print("\n【5/5】性质分布评估")
            results['property_distribution'] = self.evaluate_property_distribution(
                generated_properties,
                reference_properties,
                property_name
            )

        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)

        # 打印总结
        print("\n📊 评估总结:")
        print(f"  ✓ 有效率: {results['validity']['validity_rate']:.1%}")
        print(f"  ✓ 唯一率: {results['uniqueness']['uniqueness_rate']:.1%}")
        print(f"  ✓ 多样性得分: {results['diversity']['diversity_score']:.3f}")
        print(f"  ✓ 新颖性得分: {results['novelty']['novelty_score']:.3f}")

        if 'property_distribution' in results:
            print(f"  ✓ Wasserstein距离: "
                  f"{results['property_distribution']['wasserstein_distance']:.3f}")

        return results


# 示例使用
if __name__ == "__main__":
    print("生成MOF评估工具")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成一些模拟的MOF结构
    def create_mock_structure():
        num_nodes = 30
        node_features = torch.randn(num_nodes, 4)
        adj_matrix = torch.rand(num_nodes, num_nodes)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
        adj_matrix = (adj_matrix > 0.7).float()  # 二值化

        return {
            'node_features': node_features,
            'adj_matrix': adj_matrix
        }

    # 生成结构
    generated = [create_mock_structure() for _ in range(50)]
    reference = [create_mock_structure() for _ in range(100)]

    # 生成性质
    gen_props = np.random.randn(50) * 2 + 5
    ref_props = np.random.randn(100) * 2 + 5

    # 评估
    evaluator = GeneratedMOFEvaluator(verbose=True)

    results = evaluator.comprehensive_evaluation(
        generated_structures=generated,
        reference_structures=reference,
        generated_properties=gen_props,
        reference_properties=ref_props,
        property_name='CO2 Uptake (mmol/g)'
    )

    print("\n✓ 评估完成！")
