"""
MOF Graph Builder

将MOF晶体结构转换为图表示，用于GNN模型训练
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    from ase.io import read as ase_read
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("ASE not available. Install with: pip install ase")

try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Pymatgen not available. Install with: pip install pymatgen")


class MOFGraphBuilder:
    """
    MOF结构到图的转换器

    功能：
    - 从CIF文件读取MOF结构
    - 构建原子图（节点=原子，边=键/邻近）
    - 提取节点和边特征
    - 支持周期性边界条件
    """

    def __init__(
        self,
        cutoff_radius: float = 8.0,
        max_neighbors: int = 12,
        use_voronoi: bool = False
    ):
        """
        初始化图构建器

        参数:
            cutoff_radius: 距离截断半径（Å）
            max_neighbors: 最大邻居数
            use_voronoi: 是否使用Voronoi多面体确定邻居
        """
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.use_voronoi = use_voronoi

        # 元素周期表信息（简化版）
        self.element_properties = self._init_element_properties()

    def _init_element_properties(self) -> Dict:
        """初始化元素属性字典"""
        # 简化的元素属性（实际应用中可使用mendeleev库）
        properties = {
            'H': {'atomic_number': 1, 'mass': 1.008, 'electronegativity': 2.20, 'covalent_radius': 0.31},
            'C': {'atomic_number': 6, 'mass': 12.01, 'electronegativity': 2.55, 'covalent_radius': 0.76},
            'N': {'atomic_number': 7, 'mass': 14.01, 'electronegativity': 3.04, 'covalent_radius': 0.71},
            'O': {'atomic_number': 8, 'mass': 16.00, 'electronegativity': 3.44, 'covalent_radius': 0.66},
            'F': {'atomic_number': 9, 'mass': 19.00, 'electronegativity': 3.98, 'covalent_radius': 0.57},
            'S': {'atomic_number': 16, 'mass': 32.07, 'electronegativity': 2.58, 'covalent_radius': 1.05},
            'Zn': {'atomic_number': 30, 'mass': 65.38, 'electronegativity': 1.65, 'covalent_radius': 1.22},
            'Cu': {'atomic_number': 29, 'mass': 63.55, 'electronegativity': 1.90, 'covalent_radius': 1.32},
            'Zr': {'atomic_number': 40, 'mass': 91.22, 'electronegativity': 1.33, 'covalent_radius': 1.75},
            'Fe': {'atomic_number': 26, 'mass': 55.85, 'electronegativity': 1.83, 'covalent_radius': 1.32},
        }
        return properties

    def build_graph_from_cif(
        self,
        cif_path: str
    ) -> Dict[str, np.ndarray]:
        """
        从CIF文件构建图

        参数:
            cif_path: CIF文件路径

        返回:
            graph_dict: 包含图数据的字典
                - node_features: (N, F) 节点特征矩阵
                - edge_index: (2, E) 边索引 [source, target]
                - edge_features: (E, D) 边特征矩阵
                - positions: (N, 3) 原子坐标
                - cell: (3, 3) 晶胞矩阵
        """
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required. Install with: pip install ase")

        # 读取结构
        atoms = ase_read(cif_path)

        # 提取信息
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        # 构建节点特征
        node_features = self._get_node_features(atoms)

        # 构建边
        edge_index, edge_features = self._get_edges(atoms)

        graph_dict = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'positions': positions,
            'cell': np.array(cell),
            'symbols': symbols,
            'n_atoms': n_atoms
        }

        return graph_dict

    def _get_node_features(self, atoms: 'Atoms') -> np.ndarray:
        """
        提取节点（原子）特征

        参数:
            atoms: ASE Atoms对象

        返回:
            node_features: (N, F) 特征矩阵
        """
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        # 特征列表
        features_list = []

        for symbol in symbols:
            # 获取元素属性
            if symbol in self.element_properties:
                props = self.element_properties[symbol]
                feature = [
                    props['atomic_number'],
                    props['mass'],
                    props['electronegativity'],
                    props['covalent_radius']
                ]
            else:
                # 未知元素，使用默认值
                feature = [50.0, 50.0, 2.0, 1.0]

            features_list.append(feature)

        node_features = np.array(features_list, dtype=np.float32)

        # 归一化
        node_features = self._normalize_features(node_features)

        return node_features

    def _get_edges(
        self,
        atoms: 'Atoms'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建边（基于距离截断）

        参数:
            atoms: ASE Atoms对象

        返回:
            edge_index: (2, E) 边索引
            edge_features: (E, D) 边特征
        """
        from ase.neighborlist import neighbor_list

        # 使用ASE的neighbor_list计算邻居
        # 'i': 源原子索引, 'j': 目标原子索引, 'd': 距离, 'D': 距离向量
        i, j, d, D = neighbor_list(
            'ijdD',
            atoms,
            self.cutoff_radius
        )

        # 构建边索引
        edge_index = np.stack([i, j], axis=0)

        # 构建边特征（使用距离和Gaussian扩展）
        edge_features = self._get_edge_features(d, D)

        return edge_index, edge_features

    def _get_edge_features(
        self,
        distances: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        计算边特征

        参数:
            distances: (E,) 距离数组
            vectors: (E, 3) 距离向量

        返回:
            edge_features: (E, D) 边特征矩阵
        """
        # Gaussian距离扩展（类似CGCNN）
        gaussian_features = self._gaussian_expansion(distances)

        # 距离本身
        dist_feature = distances.reshape(-1, 1)

        # 方向向量（归一化）
        direction = vectors / (distances.reshape(-1, 1) + 1e-10)

        # 合并特征
        edge_features = np.concatenate([
            dist_feature,
            gaussian_features,
            direction
        ], axis=1).astype(np.float32)

        return edge_features

    def _gaussian_expansion(
        self,
        distances: np.ndarray,
        n_gaussians: int = 20,
        dmin: float = 0.0,
        dmax: float = 8.0,
        width: float = 0.5
    ) -> np.ndarray:
        """
        Gaussian径向基函数扩展

        参数:
            distances: (E,) 距离
            n_gaussians: Gaussian函数数量
            dmin: 最小距离
            dmax: 最大距离
            width: Gaussian宽度

        返回:
            expanded: (E, n_gaussians)
        """
        # 创建Gaussian中心
        centers = np.linspace(dmin, dmax, n_gaussians)

        # 计算Gaussian值
        # expanded[i, j] = exp(-((dist[i] - centers[j])^2) / (2 * width^2))
        diff = distances.reshape(-1, 1) - centers.reshape(1, -1)
        expanded = np.exp(-(diff ** 2) / (2 * width ** 2))

        return expanded

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征

        参数:
            features: (N, F) 特征矩阵

        返回:
            normalized: (N, F) 归一化后的特征
        """
        # 标准化（零均值，单位方差）
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-10

        normalized = (features - mean) / std

        return normalized

    def to_pytorch_geometric(self, graph_dict: Dict) -> 'torch_geometric.data.Data':
        """
        转换为PyTorch Geometric Data对象

        参数:
            graph_dict: 图字典

        返回:
            data: PyG Data对象
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric not installed. "
                            "Install with: pip install torch-geometric")

        data = Data(
            x=torch.tensor(graph_dict['node_features'], dtype=torch.float),
            edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph_dict['edge_features'], dtype=torch.float),
            pos=torch.tensor(graph_dict['positions'], dtype=torch.float)
        )

        return data

    def batch_build_graphs(
        self,
        cif_paths: List[str],
        targets: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        批量构建图

        参数:
            cif_paths: CIF文件路径列表
            targets: 目标值数组（可选）
            verbose: 是否显示进度

        返回:
            graphs: 图字典列表
        """
        graphs = []

        for i, cif_path in enumerate(cif_paths):
            if verbose and i % 100 == 0:
                print(f"Processing {i}/{len(cif_paths)}")

            try:
                graph = self.build_graph_from_cif(cif_path)

                if targets is not None:
                    graph['target'] = targets[i]

                graphs.append(graph)

            except Exception as e:
                print(f"Error processing {cif_path}: {e}")
                continue

        return graphs


def create_simple_graph_example():
    """创建一个简单的图示例（用于测试）"""
    # 简单的MOF单元：Zn + 4 O
    graph_dict = {
        'node_features': np.array([
            [30, 65.38, 1.65, 1.22],  # Zn
            [8, 16.00, 3.44, 0.66],   # O1
            [8, 16.00, 3.44, 0.66],   # O2
            [8, 16.00, 3.44, 0.66],   # O3
            [8, 16.00, 3.44, 0.66],   # O4
        ], dtype=np.float32),
        'edge_index': np.array([
            [0, 0, 0, 0, 1, 2, 3, 4],  # source
            [1, 2, 3, 4, 0, 0, 0, 0],  # target
        ], dtype=np.int64),
        'edge_features': np.random.randn(8, 24).astype(np.float32),
        'positions': np.array([
            [0.0, 0.0, 0.0],    # Zn
            [1.5, 0.0, 0.0],    # O1
            [0.0, 1.5, 0.0],    # O2
            [-1.5, 0.0, 0.0],   # O3
            [0.0, -1.5, 0.0],   # O4
        ], dtype=np.float32),
        'cell': np.eye(3) * 10.0,
        'symbols': ['Zn', 'O', 'O', 'O', 'O'],
        'n_atoms': 5
    }

    return graph_dict


if __name__ == '__main__':
    print("MOF Graph Builder")

    # 测试
    builder = MOFGraphBuilder(cutoff_radius=8.0)

    # 创建示例图
    graph = create_simple_graph_example()
    print(f"\nExample graph:")
    print(f"  Nodes: {graph['n_atoms']}")
    print(f"  Edges: {graph['edge_index'].shape[1]}")
    print(f"  Node features shape: {graph['node_features'].shape}")
    print(f"  Edge features shape: {graph['edge_features'].shape}")
