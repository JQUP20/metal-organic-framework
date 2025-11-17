"""
MOF 几何特征提取工具
集成 Zeo++、Poreblazer 等工具计算孔结构参数
"""

import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read, write


class ZeoppFeatureExtractor:
    """
    Zeo++ 特征提取器
    计算比表面积、孔径、孔体积等几何参数
    """

    def __init__(self, zeopp_path='network'):
        """
        初始化

        Parameters:
        -----------
        zeopp_path : str
            Zeo++ 可执行文件路径 (默认假设已添加到 PATH)
        """
        self.zeopp_path = zeopp_path
        self._check_installation()

    def _check_installation(self):
        """检查 Zeo++ 是否安装"""
        try:
            result = subprocess.run(
                [self.zeopp_path, '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                print("警告: Zeo++ 未正确安装，请参考安装指南")
        except FileNotFoundError:
            print("警告: 找不到 Zeo++，部分功能将不可用")
            print("安装指南: http://www.zeoplusplus.org/")

    def calculate_all_features(self, cif_file, probe_radius=1.86):
        """
        计算所有几何特征

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径
        probe_radius : float
            探针半径 (Å)，默认 1.86 Å (N₂ 分子)

        Returns:
        --------
        features : dict
            几何特征字典
        """
        features = {}

        try:
            # 计算比表面积
            sa_features = self.calculate_surface_area(cif_file, probe_radius)
            features.update(sa_features)

            # 计算孔径
            pore_features = self.calculate_pore_diameter(cif_file)
            features.update(pore_features)

            # 计算孔体积
            vol_features = self.calculate_pore_volume(cif_file, probe_radius)
            features.update(vol_features)

        except Exception as e:
            print(f"特征提取失败: {e}")
            features = self._default_features()

        return features

    def calculate_surface_area(self, cif_file, probe_radius=1.86, num_samples=20000):
        """
        计算比表面积

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径
        probe_radius : float
            探针半径 (Å)
        num_samples : int
            采样点数

        Returns:
        --------
        features : dict
            {'ASA_m2/g': 可及比表面积, 'ASA_m2/cm3': 体积比表面积, 'NASA_m2/g': 非可及比表面积}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / 'output'

            cmd = [
                self.zeopp_path,
                '-ha',  # 高精度
                '-sa', str(probe_radius), str(probe_radius), str(num_samples),
                cif_file,
                str(output_prefix)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)

                # 读取结果
                sa_file = f"{output_prefix}.sa"
                if os.path.exists(sa_file):
                    with open(sa_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            parts = lines[0].split()
                            return {
                                'ASA_m2/g': float(parts[1]),      # 可及比表面积
                                'ASA_m2/cm3': float(parts[2]),    # 体积比表面积
                                'NASA_m2/g': float(parts[3]),     # 非可及比表面积
                            }
            except Exception as e:
                print(f"Zeo++ 比表面积计算失败: {e}")

        return {'ASA_m2/g': np.nan, 'ASA_m2/cm3': np.nan, 'NASA_m2/g': np.nan}

    def calculate_pore_diameter(self, cif_file):
        """
        计算孔径参数

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径

        Returns:
        --------
        features : dict
            {'Di': 最大内嵌球直径, 'Df': 最大自由球直径, 'Dif': 最大内嵌球沿自由球路径直径}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / 'output'

            cmd = [
                self.zeopp_path,
                '-ha',
                '-res',  # 计算孔径
                cif_file,
                str(output_prefix)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)

                # 读取结果
                res_file = f"{output_prefix}.res"
                if os.path.exists(res_file):
                    with open(res_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            parts = lines[0].split()
                            return {
                                'LCD': float(parts[1]),  # Largest Cavity Diameter
                                'PLD': float(parts[2]),  # Pore Limiting Diameter
                                'LCPLD': float(parts[3]),  # Largest Cavity along Pore Limiting Diameter path
                            }
            except Exception as e:
                print(f"Zeo++ 孔径计算失败: {e}")

        return {'LCD': np.nan, 'PLD': np.nan, 'LCPLD': np.nan}

    def calculate_pore_volume(self, cif_file, probe_radius=1.86, num_samples=50000):
        """
        计算孔体积

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径
        probe_radius : float
            探针半径 (Å)
        num_samples : int
            采样点数

        Returns:
        --------
        features : dict
            {'AV_cm3/g': 可及孔体积, 'AV_cm3/cm3': 孔隙率, 'NAV_cm3/g': 非可及孔体积}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / 'output'

            cmd = [
                self.zeopp_path,
                '-ha',
                '-vol', str(probe_radius), str(probe_radius), str(num_samples),
                cif_file,
                str(output_prefix)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)

                # 读取结果
                vol_file = f"{output_prefix}.vol"
                if os.path.exists(vol_file):
                    with open(vol_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            parts = lines[0].split()
                            return {
                                'AV_cm3/g': float(parts[1]),     # 可及孔体积
                                'AV_cm3/cm3': float(parts[2]),   # 孔隙率
                                'NAV_cm3/g': float(parts[3]),    # 非可及孔体积
                            }
            except Exception as e:
                print(f"Zeo++ 孔体积计算失败: {e}")

        return {'AV_cm3/g': np.nan, 'AV_cm3/cm3': np.nan, 'NAV_cm3/g': np.nan}

    def _default_features(self):
        """返回默认特征 (NaN)"""
        return {
            'ASA_m2/g': np.nan,
            'ASA_m2/cm3': np.nan,
            'NASA_m2/g': np.nan,
            'LCD': np.nan,
            'PLD': np.nan,
            'LCPLD': np.nan,
            'AV_cm3/g': np.nan,
            'AV_cm3/cm3': np.nan,
            'NAV_cm3/g': np.nan,
        }


class SimplifiedGeometricExtractor:
    """
    简化几何特征提取器
    不依赖外部工具，仅使用 ASE 和简单几何计算
    """

    def __init__(self):
        pass

    def calculate_features(self, cif_file):
        """
        计算简化几何特征

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径

        Returns:
        --------
        features : dict
            几何特征
        """
        atoms = read(cif_file)

        features = {
            # 晶胞参数
            'cell_a': atoms.get_cell().lengths()[0],
            'cell_b': atoms.get_cell().lengths()[1],
            'cell_c': atoms.get_cell().lengths()[2],
            'cell_alpha': atoms.get_cell().angles()[0],
            'cell_beta': atoms.get_cell().angles()[1],
            'cell_gamma': atoms.get_cell().angles()[2],
            'volume': atoms.get_volume(),

            # 原子数
            'n_atoms': len(atoms),
            'density': self._calculate_density(atoms),

            # 估算孔隙率 (简化方法)
            'estimated_porosity': self._estimate_porosity(atoms),
        }

        return features

    def _calculate_density(self, atoms):
        """计算晶体密度 (g/cm³)"""
        from ase.data import atomic_masses

        mass = sum([atomic_masses[atom.number] for atom in atoms])
        volume_cm3 = atoms.get_volume() * 1e-24
        density = mass / volume_cm3 / 6.022e23

        return density

    def _estimate_porosity(self, atoms):
        """
        估算孔隙率 (简化方法)
        基于原子体积与晶胞体积的比值
        """
        from ase.data import covalent_radii

        # 原子总体积 (van der Waals 体积)
        atom_volume = 0
        for atom in atoms:
            radius = covalent_radii[atom.number] * 1.5  # 近似 vdW 半径
            atom_volume += (4/3) * np.pi * radius**3

        cell_volume = atoms.get_volume()
        porosity = max(0, 1 - atom_volume / cell_volume)

        return porosity


class MOFidExtractor:
    """
    MOFid 特征提取器
    识别 MOF 的拓扑结构和组成
    """

    def __init__(self):
        self._check_installation()

    def _check_installation(self):
        """检查 MOFid 是否安装"""
        try:
            import mofid
            self.mofid_available = True
        except ImportError:
            print("警告: MOFid 未安装，请运行 pip install mofid-cygwin")
            self.mofid_available = False

    def extract_features(self, cif_file):
        """
        提取 MOFid 特征

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径

        Returns:
        --------
        features : dict
            MOFid 识别的特征
        """
        if not self.mofid_available:
            return {'mofid': None, 'topology': None, 'metal': None, 'linker': None}

        try:
            import mofid

            # 运行 MOFid
            result = mofid.run_mofid(cif_file)

            features = {
                'mofid': result.get('mofid', None),
                'topology': result.get('topology', None),
                'metal': result.get('metal', None),
                'linker': result.get('linker', None),
            }

            return features

        except Exception as e:
            print(f"MOFid 提取失败: {e}")
            return {'mofid': None, 'topology': None, 'metal': None, 'linker': None}


def batch_extract_features(cif_files, use_zeopp=False):
    """
    批量提取特征

    Parameters:
    -----------
    cif_files : list
        CIF 文件路径列表
    use_zeopp : bool
        是否使用 Zeo++ (需要安装)

    Returns:
    --------
    df : pd.DataFrame
        特征数据框
    """
    from tqdm import tqdm

    if use_zeopp:
        extractor = ZeoppFeatureExtractor()
    else:
        extractor = SimplifiedGeometricExtractor()

    features_list = []

    for cif_file in tqdm(cif_files, desc="提取特征"):
        try:
            features = extractor.calculate_features(cif_file) if not use_zeopp else extractor.calculate_all_features(cif_file)
            features['name'] = Path(cif_file).stem
            features['path'] = cif_file
            features_list.append(features)
        except Exception as e:
            print(f"提取失败: {cif_file} - {e}")

    df = pd.DataFrame(features_list)
    return df


if __name__ == "__main__":
    print("MOF 特征提取工具模块已加载")
    print("\n使用示例:")
    print("  # 简化方法 (不需要 Zeo++)")
    print("  from geometric_features import SimplifiedGeometricExtractor")
    print("  extractor = SimplifiedGeometricExtractor()")
    print("  features = extractor.calculate_features('path/to/mof.cif')")
    print("\n  # 完整方法 (需要 Zeo++)")
    print("  from geometric_features import ZeoppFeatureExtractor")
    print("  extractor = ZeoppFeatureExtractor()")
    print("  features = extractor.calculate_all_features('path/to/mof.cif')")
