"""
MOF 结构可视化工具
使用 ASE 和 py3Dmol 可视化 MOF 晶体结构
"""

import warnings
warnings.filterwarnings('ignore')

from ase.io import read, write
from ase import Atoms
import py3Dmol
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


class MOFVisualizer:
    """MOF 结构可视化类"""

    def __init__(self, structure_file):
        """
        初始化可视化器

        Parameters:
        -----------
        structure_file : str
            CIF 或其他晶体结构文件路径
        """
        self.structure = read(structure_file)
        self.atoms = self.structure

    def view_3d(self, style='sphere', size=(400, 400), show_cell=True):
        """
        3D 交互式可视化

        Parameters:
        -----------
        style : str
            显示风格: 'sphere', 'stick', 'line', 'cartoon'
        size : tuple
            窗口大小 (宽, 高)
        show_cell : bool
            是否显示晶胞边界

        Returns:
        --------
        view : py3Dmol.view
            3D 可视化对象
        """
        # 转换为 XYZ 格式字符串
        xyz_str = self._atoms_to_xyz_string()

        # 创建 py3Dmol 视图
        view = py3Dmol.view(width=size[0], height=size[1])
        view.addModel(xyz_str, 'xyz')

        # 设置显示风格
        if style == 'sphere':
            view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
        elif style == 'stick':
            view.setStyle({'stick': {'radius': 0.2}})
        elif style == 'line':
            view.setStyle({'line': {}})
        else:
            view.setStyle({style: {}})

        # 显示晶胞
        if show_cell:
            self._add_unit_cell(view)

        view.zoomTo()
        return view

    def _atoms_to_xyz_string(self):
        """将 ASE Atoms 对象转换为 XYZ 格式字符串"""
        positions = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()

        xyz_lines = [str(len(symbols)), ""]
        for symbol, pos in zip(symbols, positions):
            xyz_lines.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        return '\n'.join(xyz_lines)

    def _add_unit_cell(self, view):
        """添加晶胞边界框"""
        cell = self.atoms.get_cell()

        # 晶胞的 8 个顶点
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        # 转换到笛卡尔坐标
        cart_vertices = [np.dot(v, cell) for v in vertices]

        # 晶胞的 12 条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
        ]

        # 绘制边
        for edge in edges:
            start = cart_vertices[edge[0]]
            end = cart_vertices[edge[1]]
            view.addCylinder({
                'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                'radius': 0.05,
                'color': 'black',
                'opacity': 0.5
            })

    def plot_2d_projection(self, axis='z', repeat=(1, 1, 1), figsize=(8, 8)):
        """
        2D 投影图

        Parameters:
        -----------
        axis : str
            投影轴 ('x', 'y', 'z')
        repeat : tuple
            重复晶胞 (nx, ny, nz)
        figsize : tuple
            图像大小
        """
        atoms_repeated = self.atoms.repeat(repeat)
        positions = atoms_repeated.get_positions()
        symbols = atoms_repeated.get_chemical_symbols()

        # 选择投影轴
        axis_map = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
        idx = axis_map[axis]

        fig, ax = plt.subplots(figsize=figsize)

        # 原子颜色映射
        color_map = {
            'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue',
            'Zn': 'silver', 'Cu': 'orange', 'Zr': 'cyan',
            'Fe': 'brown', 'Al': 'pink', 'Cr': 'green'
        }

        # 绘制原子
        for symbol, pos in zip(symbols, positions):
            color = color_map.get(symbol, 'purple')
            ax.scatter(pos[idx[0]], pos[idx[1]],
                      c=color, s=100, edgecolors='black', linewidths=0.5,
                      label=symbol if symbol not in ax.get_legend_handles_labels()[1] else "")

        ax.set_xlabel(f'{["X", "Y", "Z"][idx[0]]} (Å)', fontsize=12)
        ax.set_ylabel(f'{["X", "Y", "Z"][idx[1]]} (Å)', fontsize=12)
        ax.set_title(f'MOF Structure - {axis.upper()} Projection', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        plt.tight_layout()

        return fig, ax

    def export_formats(self, output_prefix):
        """
        导出多种格式

        Parameters:
        -----------
        output_prefix : str
            输出文件前缀
        """
        formats = {
            'cif': 'CIF 格式',
            'xyz': 'XYZ 格式',
            'pdb': 'PDB 格式',
            'vasp': 'POSCAR 格式',
            'json': 'JSON 格式'
        }

        exported = {}
        for fmt, desc in formats.items():
            try:
                filename = f"{output_prefix}.{fmt}"
                write(filename, self.atoms, format=fmt)
                exported[fmt] = filename
                print(f"✓ 已导出 {desc}: {filename}")
            except Exception as e:
                print(f"✗ 导出 {desc} 失败: {e}")

        return exported

    def get_structure_info(self):
        """获取结构信息"""
        cell = self.atoms.get_cell()
        volume = self.atoms.get_volume()
        composition = self.atoms.get_chemical_formula()
        n_atoms = len(self.atoms)

        info = {
            '化学式': composition,
            '原子数': n_atoms,
            '晶胞参数 (Å)': {
                'a': cell.lengths()[0],
                'b': cell.lengths()[1],
                'c': cell.lengths()[2],
            },
            '晶胞角度 (°)': {
                'α': cell.angles()[0],
                'β': cell.angles()[1],
                'γ': cell.angles()[2],
            },
            '晶胞体积 (Å³)': volume,
        }

        return info

    def print_info(self):
        """打印结构信息"""
        info = self.get_structure_info()

        print("=" * 50)
        print("MOF 结构信息".center(50))
        print("=" * 50)
        print(f"化学式: {info['化学式']}")
        print(f"原子数: {info['原子数']}")
        print(f"\n晶胞参数:")
        print(f"  a = {info['晶胞参数 (Å)']['a']:.3f} Å")
        print(f"  b = {info['晶胞参数 (Å)']['b']:.3f} Å")
        print(f"  c = {info['晶胞参数 (Å)']['c']:.3f} Å")
        print(f"\n晶胞角度:")
        print(f"  α = {info['晶胞角度 (°)']['α']:.2f}°")
        print(f"  β = {info['晶胞角度 (°)']['β']:.2f}°")
        print(f"  γ = {info['晶胞角度 (°)']['γ']:.2f}°")
        print(f"\n晶胞体积: {info['晶胞体积 (Å³)']:.2f} Å³")
        print("=" * 50)


def compare_structures(file1, file2, style='sphere'):
    """
    并排比较两个 MOF 结构

    Parameters:
    -----------
    file1, file2 : str
        结构文件路径
    style : str
        显示风格

    Returns:
    --------
    view : py3Dmol.view
        并排显示的 3D 视图
    """
    atoms1 = read(file1)
    atoms2 = read(file2)

    view = py3Dmol.view(width=800, height=400, viewergrid=(1, 2))

    # 左侧结构
    xyz1 = atoms_to_xyz_string(atoms1)
    view.addModel(xyz1, 'xyz', viewer=(0, 0))
    view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}}, viewer=(0, 0))

    # 右侧结构
    xyz2 = atoms_to_xyz_string(atoms2)
    view.addModel(xyz2, 'xyz', viewer=(0, 1))
    view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}}, viewer=(0, 1))

    view.zoomTo()
    return view


def atoms_to_xyz_string(atoms):
    """辅助函数：Atoms 转 XYZ 字符串"""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    xyz_lines = [str(len(symbols)), ""]
    for symbol, pos in zip(symbols, positions):
        xyz_lines.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    return '\n'.join(xyz_lines)


if __name__ == "__main__":
    print("MOF 可视化工具模块已加载")
    print("使用示例:")
    print("  from mof_visualizer import MOFVisualizer")
    print("  viz = MOFVisualizer('path/to/mof.cif')")
    print("  viz.view_3d()  # Jupyter Notebook 中显示")
    print("  viz.plot_2d_projection()  # 2D 投影图")
