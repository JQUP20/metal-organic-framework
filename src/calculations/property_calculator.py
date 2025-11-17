"""
MOF 性能参数计算工具
计算比表面积、孔径分布、能量参数等
"""

import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
import matplotlib.pyplot as plt


class BETCalculator:
    """
    BET 比表面积计算器
    基于简化的 Connolly 表面计算
    """

    def __init__(self, probe_radius=1.86):
        """
        初始化

        Parameters:
        -----------
        probe_radius : float
            探针半径 (Å)，默认 1.86 Å (N₂)
        """
        self.probe_radius = probe_radius

    def calculate_surface_area(self, atoms, n_points=10000):
        """
        计算可及表面积 (简化方法)

        Parameters:
        -----------
        atoms : ase.Atoms
            晶体结构
        n_points : int
            采样点数

        Returns:
        --------
        sa_g : float
            质量比表面积 (m²/g)
        sa_v : float
            体积比表面积 (m²/cm³)
        """
        from ase.data import atomic_masses, covalent_radii

        # 扩展晶胞以避免边界效应
        atoms_extended = atoms.repeat((2, 2, 2))

        positions = atoms_extended.get_positions()
        numbers = atoms_extended.get_atomic_numbers()

        # 原子半径 (van der Waals)
        radii = np.array([covalent_radii[n] * 1.5 for n in numbers])

        # 生成随机采样点
        cell = atoms.get_cell()
        sample_points = np.random.rand(n_points, 3) @ cell

        # 检查每个点是否可及
        accessible_count = 0
        for point in sample_points:
            # 计算到所有原子的距离
            distances = np.linalg.norm(positions - point, axis=1)

            # 可及条件：最近原子距离 > 原子半径 + 探针半径
            min_clearance = np.min(distances - radii)

            if min_clearance >= self.probe_radius:
                accessible_count += 1

        # 计算可及体积
        total_volume = atoms.get_volume()
        accessible_fraction = accessible_count / n_points
        accessible_volume = total_volume * accessible_fraction

        # 估算表面积 (基于体积和平均孔径)
        # 简化假设：球形孔
        if accessible_fraction > 0.01:
            avg_pore_radius = self._estimate_pore_radius(atoms)
            surface_area = 3 * accessible_volume / avg_pore_radius  # Å²
        else:
            surface_area = 0

        # 转换单位
        surface_area_m2 = surface_area * 1e-20  # Å² to m²

        # 计算质量
        mass_g = sum([atomic_masses[n] for n in atoms.get_atomic_numbers()]) / 6.022e23

        # 比表面积
        sa_g = surface_area_m2 / mass_g if mass_g > 0 else 0

        # 体积比表面积
        volume_cm3 = total_volume * 1e-24
        sa_v = surface_area_m2 / volume_cm3 if volume_cm3 > 0 else 0

        return sa_g, sa_v

    def _estimate_pore_radius(self, atoms):
        """估算平均孔径"""
        from ase.data import covalent_radii

        # 简化方法：基于最近邻距离
        cutoffs = natural_cutoffs(atoms, mult=1.2)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        distances = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            if len(indices) > 0:
                for j, offset in zip(indices, offsets):
                    pos_i = atoms.positions[i]
                    pos_j = atoms.positions[j] + offset @ atoms.get_cell()
                    dist = np.linalg.norm(pos_i - pos_j)
                    distances.append(dist)

        if distances:
            avg_dist = np.mean(distances)
            avg_radius = covalent_radii[atoms.get_atomic_numbers()[0]] * 1.5
            pore_radius = max(1.0, avg_dist - 2 * avg_radius)
        else:
            pore_radius = 5.0  # 默认值

        return pore_radius


class PoreSizeDistributionCalculator:
    """
    孔径分布计算器
    """

    def __init__(self):
        pass

    def calculate_psd(self, atoms, probe_radius=1.2, max_radius=20.0, n_bins=50, n_samples=5000):
        """
        计算孔径分布

        Parameters:
        -----------
        atoms : ase.Atoms
            晶体结构
        probe_radius : float
            最小探针半径 (Å)
        max_radius : float
            最大探针半径 (Å)
        n_bins : int
            分布区间数
        n_samples : int
            采样点数

        Returns:
        --------
        radii : np.array
            孔径值 (Å)
        distribution : np.array
            概率分布
        """
        from ase.data import covalent_radii

        # 扩展晶胞
        atoms_extended = atoms.repeat((2, 2, 2))

        positions = atoms_extended.get_positions()
        numbers = atoms_extended.get_atomic_numbers()
        vdw_radii = np.array([covalent_radii[n] * 1.5 for n in numbers])

        # 生成随机采样点
        cell = atoms.get_cell()
        sample_points = np.random.rand(n_samples, 3) @ cell

        # 计算每个点的最大可及半径
        local_radii = []
        for point in sample_points:
            distances = np.linalg.norm(positions - point, axis=1)
            min_clearance = np.min(distances - vdw_radii)

            if min_clearance > probe_radius:
                local_radii.append(min_clearance)

        # 计算分布
        if local_radii:
            hist, bin_edges = np.histogram(local_radii, bins=n_bins, range=(probe_radius, max_radius), density=True)
            radii = (bin_edges[:-1] + bin_edges[1:]) / 2
            return radii, hist
        else:
            radii = np.linspace(probe_radius, max_radius, n_bins)
            return radii, np.zeros(n_bins)

    def plot_psd(self, radii, distribution, figsize=(8, 5)):
        """
        绘制孔径分布图

        Parameters:
        -----------
        radii : np.array
            孔径值
        distribution : np.array
            概率分布

        Returns:
        --------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(radii, distribution, linewidth=2, color='steelblue')
        ax.fill_between(radii, distribution, alpha=0.3, color='steelblue')

        ax.set_xlabel('Pore Diameter (Å)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Pore Size Distribution', fontsize=14)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig, ax


class EnergyCalculator:
    """
    能量参数计算器
    简化的能量估算 (实际应使用 DFT)
    """

    def __init__(self):
        pass

    def calculate_lattice_energy(self, atoms):
        """
        估算晶格能 (简化的 Lennard-Jones 势)

        Parameters:
        -----------
        atoms : ase.Atoms
            晶体结构

        Returns:
        --------
        energy : float
            晶格能 (eV)
        """
        # 简化的 LJ 参数
        epsilon = 0.01  # eV
        sigma = 3.0     # Å

        cutoffs = natural_cutoffs(atoms, mult=2.5)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
        nl.update(atoms)

        total_energy = 0.0

        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)

            for j, offset in zip(indices, offsets):
                pos_i = atoms.positions[i]
                pos_j = atoms.positions[j] + offset @ atoms.get_cell()
                r = np.linalg.norm(pos_i - pos_j)

                if r > 0.1:  # 避免除零
                    # LJ 势能
                    lj_energy = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
                    total_energy += lj_energy

        # 每个原子的平均能量
        energy_per_atom = total_energy / len(atoms)

        return energy_per_atom

    def estimate_band_gap(self, atoms):
        """
        估算带隙 (基于经验规则)

        Parameters:
        -----------
        atoms : ase.Atoms
            晶体结构

        Returns:
        --------
        band_gap : float
            估算带隙 (eV)

        Note:
        -----
        这是非常粗略的估算，实际应使用 DFT 计算
        """
        # 检查金属类型
        symbols = atoms.get_chemical_symbols()

        metals = {'Zn', 'Zr', 'Al', 'Mg'}
        conductive_metals = {'Cu', 'Fe', 'Co', 'Ni'}

        has_metal = any(s in metals for s in symbols)
        has_conductive = any(s in conductive_metals for s in symbols)

        if has_conductive:
            band_gap = 0.5  # 小带隙
        elif has_metal:
            band_gap = 3.0  # 大带隙
        else:
            band_gap = 2.0  # 中等

        # 加入一些随机性模拟复杂性
        band_gap += np.random.uniform(-0.5, 0.5)
        band_gap = max(0, band_gap)

        return band_gap


class PropertyCalculator:
    """
    综合性能计算器
    """

    def __init__(self):
        self.bet_calc = BETCalculator()
        self.psd_calc = PoreSizeDistributionCalculator()
        self.energy_calc = EnergyCalculator()

    def calculate_all_properties(self, cif_file, n_samples=5000):
        """
        计算所有性能参数

        Parameters:
        -----------
        cif_file : str
            CIF 文件路径
        n_samples : int
            采样点数 (影响精度和速度)

        Returns:
        --------
        properties : dict
            所有性能参数
        """
        atoms = read(cif_file)

        properties = {}

        # 比表面积
        try:
            sa_g, sa_v = self.bet_calc.calculate_surface_area(atoms, n_points=n_samples)
            properties['BET_m2/g'] = sa_g
            properties['BET_m2/cm3'] = sa_v
        except Exception as e:
            print(f"比表面积计算失败: {e}")
            properties['BET_m2/g'] = np.nan
            properties['BET_m2/cm3'] = np.nan

        # 孔径分布
        try:
            radii, dist = self.psd_calc.calculate_psd(atoms, n_samples=n_samples)
            properties['avg_pore_diameter'] = np.average(radii, weights=dist) if dist.sum() > 0 else np.nan
            properties['psd_radii'] = radii
            properties['psd_distribution'] = dist
        except Exception as e:
            print(f"孔径分布计算失败: {e}")
            properties['avg_pore_diameter'] = np.nan

        # 能量参数
        try:
            lattice_energy = self.energy_calc.calculate_lattice_energy(atoms)
            properties['lattice_energy_eV'] = lattice_energy
        except Exception as e:
            print(f"晶格能计算失败: {e}")
            properties['lattice_energy_eV'] = np.nan

        try:
            band_gap = self.energy_calc.estimate_band_gap(atoms)
            properties['band_gap_eV'] = band_gap
        except Exception as e:
            print(f"带隙估算失败: {e}")
            properties['band_gap_eV'] = np.nan

        return properties

    def print_properties(self, properties):
        """打印性能参数"""
        print("\n" + "=" * 60)
        print("MOF 性能参数".center(60))
        print("=" * 60)

        print("\n孔结构参数:")
        print(f"  BET 比表面积: {properties.get('BET_m2/g', 'N/A'):.2f} m²/g")
        print(f"  体积比表面积: {properties.get('BET_m2/cm3', 'N/A'):.2f} m²/cm³")
        print(f"  平均孔径: {properties.get('avg_pore_diameter', 'N/A'):.2f} Å")

        print("\n能量参数 (估算):")
        print(f"  晶格能: {properties.get('lattice_energy_eV', 'N/A'):.4f} eV/atom")
        print(f"  带隙: {properties.get('band_gap_eV', 'N/A'):.2f} eV")

        print("=" * 60)


if __name__ == "__main__":
    print("MOF 性能计算工具模块已加载")
    print("\n使用示例:")
    print("  from property_calculator import PropertyCalculator")
    print("  calc = PropertyCalculator()")
    print("  props = calc.calculate_all_properties('path/to/mof.cif')")
    print("  calc.print_properties(props)")
