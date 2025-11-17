"""
基础测试
运行: pytest tests/test_basic.py
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """测试模块导入"""
    from visualization.mof_visualizer import MOFVisualizer
    from data_processing.dataset_builder import MOFDatasetBuilder
    from feature_extraction.geometric_features import SimplifiedGeometricExtractor
    from calculations.property_calculator import PropertyCalculator

    assert MOFVisualizer is not None
    assert MOFDatasetBuilder is not None
    assert SimplifiedGeometricExtractor is not None
    assert PropertyCalculator is not None


def test_example_file_exists():
    """测试示例文件是否存在"""
    example_cif = Path(__file__).parent.parent / 'data' / 'examples' / 'example_mof.cif'
    assert example_cif.exists(), f"示例文件不存在: {example_cif}"


if __name__ == '__main__':
    print("运行基础测试...")
    test_imports()
    print("✓ 模块导入测试通过")
    test_example_file_exists()
    print("✓ 示例文件测试通过")
    print("\n所有测试通过！")
