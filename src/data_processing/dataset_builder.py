"""
MOF 数据集构建与清洗工具
从 CoRE-MOF、CSD、QMOF 数据库中筛选和清洗数据
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ase.io import read
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MOFDatasetBuilder:
    """MOF 数据集构建器"""

    def __init__(self, data_dir):
        """
        初始化数据集构建器

        Parameters:
        -----------
        data_dir : str
            数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.structures = []
        self.metadata = []

    def load_structures(self, file_pattern='*.cif', max_files=None):
        """
        加载晶体结构文件

        Parameters:
        -----------
        file_pattern : str
            文件匹配模式
        max_files : int
            最大加载文件数 (用于测试)

        Returns:
        --------
        n_loaded : int
            成功加载的文件数
        """
        cif_files = list(self.data_dir.glob(file_pattern))

        if max_files:
            cif_files = cif_files[:max_files]

        print(f"发现 {len(cif_files)} 个结构文件")

        for cif_file in tqdm(cif_files, desc="加载结构"):
            try:
                atoms = read(cif_file)
                self.structures.append({
                    'name': cif_file.stem,
                    'path': str(cif_file),
                    'atoms': atoms
                })
            except Exception as e:
                print(f"加载失败: {cif_file.name} - {e}")

        print(f"成功加载 {len(self.structures)} 个结构")
        return len(self.structures)

    def extract_basic_features(self):
        """
        提取基础结构特征

        Returns:
        --------
        df : pd.DataFrame
            特征数据框
        """
        features = []

        for struct in tqdm(self.structures, desc="提取特征"):
            atoms = struct['atoms']

            feature = {
                'name': struct['name'],
                'path': struct['path'],
                'n_atoms': len(atoms),
                'chemical_formula': atoms.get_chemical_formula(),
                'volume': atoms.get_volume(),
                'density': self._calculate_density(atoms),
                'cell_a': atoms.get_cell().lengths()[0],
                'cell_b': atoms.get_cell().lengths()[1],
                'cell_c': atoms.get_cell().lengths()[2],
                'cell_alpha': atoms.get_cell().angles()[0],
                'cell_beta': atoms.get_cell().angles()[1],
                'cell_gamma': atoms.get_cell().angles()[2],
            }

            # 元素组成统计
            composition = self._get_composition(atoms)
            feature.update(composition)

            features.append(feature)

        df = pd.DataFrame(features)
        return df

    def _calculate_density(self, atoms):
        """计算晶体密度 (g/cm³)"""
        from ase.data import atomic_masses

        mass = sum([atomic_masses[atom.number] for atom in atoms])
        volume_cm3 = atoms.get_volume() * 1e-24  # Å³ to cm³
        density = mass / volume_cm3 / 6.022e23  # g/cm³

        return density

    def _get_composition(self, atoms):
        """获取元素组成"""
        symbols = atoms.get_chemical_symbols()
        unique_elements = set(symbols)

        composition = {}
        for elem in unique_elements:
            composition[f'n_{elem}'] = symbols.count(elem)

        # 常见金属标记
        metals = {'Zn', 'Cu', 'Zr', 'Cr', 'Fe', 'Al', 'Co', 'Ni', 'Mg', 'Ca'}
        composition['has_metal'] = any(elem in metals for elem in unique_elements)

        # 金属种类
        metal_types = [elem for elem in unique_elements if elem in metals]
        composition['metal_types'] = ','.join(sorted(metal_types)) if metal_types else 'None'

        return composition

    def clean_dataset(self, df, filters=None):
        """
        清洗数据集

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
        filters : dict
            过滤条件，例如:
            {'n_atoms': (10, 1000), 'density': (0.1, 3.0)}

        Returns:
        --------
        df_clean : pd.DataFrame
            清洗后的数据
        """
        df_clean = df.copy()
        n_original = len(df_clean)

        print("开始数据清洗...")

        # 1. 移除缺失值
        df_clean = df_clean.dropna()
        print(f"移除缺失值: {n_original - len(df_clean)} 条")
        n_original = len(df_clean)

        # 2. 移除异常值
        if filters is None:
            filters = {
                'n_atoms': (10, 5000),
                'density': (0.1, 5.0),
                'volume': (100, 100000),
            }

        for col, (min_val, max_val) in filters.items():
            if col in df_clean.columns:
                mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
                df_clean = df_clean[mask]
                print(f"{col} 范围 [{min_val}, {max_val}]: 保留 {len(df_clean)} 条")

        # 3. 移除重复结构
        df_clean = df_clean.drop_duplicates(subset=['chemical_formula', 'volume'])
        print(f"移除重复结构: {n_original - len(df_clean)} 条")

        print(f"清洗完成: {len(df)} → {len(df_clean)} 条数据")

        return df_clean

    def split_dataset(self, df, train_ratio=0.8, val_ratio=0.1, random_state=42):
        """
        划分数据集

        Parameters:
        -----------
        df : pd.DataFrame
            数据集
        train_ratio : float
            训练集比例
        val_ratio : float
            验证集比例
        random_state : int
            随机种子

        Returns:
        --------
        splits : dict
            {'train': df_train, 'val': df_val, 'test': df_test}
        """
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        n = len(df_shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        df_train = df_shuffled[:n_train]
        df_val = df_shuffled[n_train:n_train + n_val]
        df_test = df_shuffled[n_train + n_val:]

        print(f"数据集划分:")
        print(f"  训练集: {len(df_train)} ({len(df_train) / n * 100:.1f}%)")
        print(f"  验证集: {len(df_val)} ({len(df_val) / n * 100:.1f}%)")
        print(f"  测试集: {len(df_test)} ({len(df_test) / n * 100:.1f}%)")

        return {
            'train': df_train,
            'val': df_val,
            'test': df_test
        }

    def save_dataset(self, df, output_path, format='csv'):
        """
        保存数据集

        Parameters:
        -----------
        df : pd.DataFrame
            数据集
        output_path : str
            输出路径
        format : str
            格式 ('csv', 'json', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

        print(f"数据集已保存: {output_path}")

    def generate_statistics(self, df):
        """
        生成数据集统计报告

        Parameters:
        -----------
        df : pd.DataFrame
            数据集

        Returns:
        --------
        stats : dict
            统计信息
        """
        stats = {
            'n_samples': len(df),
            'numeric_features': {},
            'categorical_features': {},
        }

        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['numeric_features'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
            }

        # 分类特征统计
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['name', 'path']:  # 跳过标识符
                stats['categorical_features'][col] = df[col].value_counts().to_dict()

        return stats

    def print_statistics(self, stats):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计报告".center(60))
        print("=" * 60)
        print(f"样本数: {stats['n_samples']}")

        print("\n数值特征:")
        for feat, values in stats['numeric_features'].items():
            print(f"  {feat}:")
            print(f"    平均值: {values['mean']:.2f}")
            print(f"    标准差: {values['std']:.2f}")
            print(f"    范围: [{values['min']:.2f}, {values['max']:.2f}]")

        if stats['categorical_features']:
            print("\n分类特征:")
            for feat, counts in stats['categorical_features'].items():
                print(f"  {feat}: {len(counts)} 个类别")

        print("=" * 60)


class CoREMOFDownloader:
    """CoRE MOF 数据库下载器"""

    @staticmethod
    def download_info():
        """打印下载信息"""
        info = """
        CoRE MOF 数据库下载指南:

        1. 访问官方网站:
           https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs

        2. 下载文件:
           - CoRE MOF 2019-ASR.tar.gz (所有结构)
           - 或访问 Materials Cloud: https://archive.materialscloud.org/

        3. 解压到数据目录:
           tar -xzf CoRE-MOF-2019-ASR.tar.gz -C data/datasets/

        4. 使用本工具加载:
           builder = MOFDatasetBuilder('data/datasets/CoRE-MOF-2019-ASR')
           builder.load_structures()
        """
        print(info)


class QMOFDownloader:
    """QMOF 数据库下载器"""

    @staticmethod
    def download_info():
        """打印下载信息"""
        info = """
        QMOF 数据库下载指南:

        1. 访问官方仓库:
           https://github.com/Andrew-S-Rosen/QMOF

        2. 下载预计算数据:
           git clone https://github.com/Andrew-S-Rosen/QMOF.git

        3. 数据包含:
           - CIF 结构文件
           - DFT 计算的电子性质
           - 能带结构、态密度

        4. 使用 pandas 加载 CSV:
           import pandas as pd
           qmof_data = pd.read_csv('QMOF/qmof.csv')
        """
        print(info)


if __name__ == "__main__":
    print("MOF 数据集构建工具模块已加载")
    print("\n使用示例:")
    print("  from dataset_builder import MOFDatasetBuilder")
    print("  builder = MOFDatasetBuilder('data/datasets/mof_structures')")
    print("  builder.load_structures()")
    print("  df = builder.extract_basic_features()")
    print("  df_clean = builder.clean_dataset(df)")
    print("  splits = builder.split_dataset(df_clean)")
