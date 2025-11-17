"""
Generate Sample MOF Adsorption Dataset

生成模拟的 MOF 吸附数据集用于教学演示
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def generate_mof_adsorption_data(
    n_samples: int = 1000,
    random_state: int = 42,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    生成模拟的 MOF 吸附数据集

    数据集包含：
    - 几何特征（比表面积、孔体积、孔径等）
    - 化学特征（密度、元素比例等）
    - 目标变量（CO₂ 吸附量、CH₄ 吸附量、选择性）

    参数:
        n_samples: 样本数量
        random_state: 随机种子
        save_path: 保存路径（可选）

    返回:
        DataFrame 包含所有特征和目标变量
    """
    np.random.seed(random_state)

    # ========== 基础几何特征 ==========

    # 比表面积 (m²/g): 500-5000
    surface_area = np.random.gamma(shape=2, scale=1000, size=n_samples)
    surface_area = np.clip(surface_area, 500, 5000)

    # 孔体积 (cm³/g): 0.2-2.0
    # 与比表面积正相关
    pore_volume = 0.0002 * surface_area + np.random.normal(0, 0.1, n_samples)
    pore_volume = np.clip(pore_volume, 0.2, 2.0)

    # 密度 (g/cm³): 0.3-2.0
    # 与孔体积负相关
    density = 2.5 - pore_volume + np.random.normal(0, 0.15, n_samples)
    density = np.clip(density, 0.3, 2.0)

    # 孔限制直径 PLD (Å): 3-20
    pld = np.random.gamma(shape=3, scale=3, size=n_samples)
    pld = np.clip(pld, 3, 20)

    # 最大腔体直径 LCD (Å): 通常大于PLD
    lcd = pld + np.random.gamma(shape=2, scale=5, size=n_samples)
    lcd = np.clip(lcd, pld + 1, 40)

    # 孔隙率 (void fraction): 0.3-0.9
    void_fraction = pore_volume * density
    void_fraction = np.clip(void_fraction, 0.3, 0.9)

    # ========== 化学特征 ==========

    # 金属中心类型（简化为数值）
    # 0: Zn, 1: Cu, 2: Zr, 3: Fe, 4: Al
    metal_type = np.random.randint(0, 5, n_samples)

    # 元素比例
    C_ratio = np.random.beta(5, 2, n_samples) * 0.5 + 0.3  # 0.3-0.8
    O_ratio = np.random.beta(3, 3, n_samples) * 0.3 + 0.1  # 0.1-0.4
    N_ratio = np.random.beta(2, 5, n_samples) * 0.2        # 0-0.2
    H_ratio = 1 - C_ratio - O_ratio - N_ratio - 0.05      # 剩余

    # 归一化
    total = C_ratio + O_ratio + N_ratio + H_ratio
    C_ratio /= total
    O_ratio /= total
    N_ratio /= total
    H_ratio /= total

    # 官能团数量（简化）
    n_functional_groups = np.random.poisson(lam=2, size=n_samples)

    # 电负性（加权平均）
    electronegativity = (
        C_ratio * 2.55 +
        O_ratio * 3.44 +
        N_ratio * 3.04 +
        H_ratio * 2.20
    ) + np.random.normal(0, 0.1, n_samples)

    # ========== 目标变量：CO₂ 吸附量 ==========

    # CO₂ 吸附量主要受以下因素影响：
    # 1. 比表面积（正相关）
    # 2. 孔体积（正相关）
    # 3. 孔径（有最佳值，约6-12 Å）
    # 4. 极性（N、O 含量，正相关）

    # 孔径效应（钟形曲线）
    pore_size_effect = np.exp(-((pld - 9)**2) / (2 * 4**2))

    # 极性效应
    polarity_effect = (O_ratio + N_ratio) * 5

    # CO₂ 吸附量 (mmol/g)
    co2_uptake = (
        0.002 * surface_area +           # 比表面积贡献
        1.5 * pore_volume +               # 孔体积贡献
        2.0 * pore_size_effect +          # 孔径效应
        polarity_effect +                 # 极性效应
        np.random.normal(0, 0.5, n_samples)  # 噪声
    )
    co2_uptake = np.clip(co2_uptake, 0.5, 15)

    # ========== 目标变量：CH₄ 吸附量 ==========

    # CH₄ 吸附量也受类似因素影响，但对孔径要求不同
    # CH₄ 分子更大（动力学直径 3.8 Å）

    # 孔径效应（最佳约 10-15 Å）
    pore_size_effect_ch4 = np.exp(-((pld - 12)**2) / (2 * 5**2))

    # CH₄ 对极性不太敏感
    ch4_uptake = (
        0.0015 * surface_area +
        1.2 * pore_volume +
        1.5 * pore_size_effect_ch4 +
        0.5 * polarity_effect +           # 较小的极性效应
        np.random.normal(0, 0.4, n_samples)
    )
    ch4_uptake = np.clip(ch4_uptake, 0.3, 10)

    # ========== 目标变量：CO₂/CH₄ 选择性 ==========

    # 选择性 = (x_CO2/y_CO2) / (x_CH4/y_CH4)
    # 简化为吸附量之比（理想情况）
    selectivity = co2_uptake / (ch4_uptake + 0.1)  # 避免除零
    selectivity = np.clip(selectivity, 1, 50)

    # ========== 创建 DataFrame ==========

    df = pd.DataFrame({
        # 标识符
        'mof_id': [f'MOF_{i:04d}' for i in range(n_samples)],

        # 几何特征
        'surface_area': surface_area,
        'pore_volume': pore_volume,
        'density': density,
        'pld': pld,
        'lcd': lcd,
        'void_fraction': void_fraction,

        # 衍生几何特征
        'surface_area_per_volume': surface_area / pore_volume,
        'pore_size_ratio': lcd / pld,

        # 化学特征
        'metal_type': metal_type,
        'C_ratio': C_ratio,
        'O_ratio': O_ratio,
        'N_ratio': N_ratio,
        'H_ratio': H_ratio,
        'n_functional_groups': n_functional_groups,
        'electronegativity': electronegativity,

        # 目标变量
        'co2_uptake': co2_uptake,
        'ch4_uptake': ch4_uptake,
        'selectivity': selectivity
    })

    # 保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"数据集已保存到: {save_path}")
        print(f"样本数: {n_samples}")
        print(f"特征数: {len(df.columns) - 4}")  # 减去 mof_id 和 3 个目标变量

    return df


def generate_mof_bandgap_data(
    n_samples: int = 500,
    random_state: int = 42,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    生成模拟的 MOF 能带结构数据集

    参数:
        n_samples: 样本数量
        random_state: 随机种子
        save_path: 保存路径（可选）

    返回:
        DataFrame 包含特征和带隙
    """
    np.random.seed(random_state)

    # 金属中心（影响带隙）
    # 0: Zn (半导体), 1: Cu (较窄带隙), 2: Zr (宽带隙), 3: Fe (窄带隙), 4: Ti (可调)
    metal_type = np.random.randint(0, 5, n_samples)

    # 配体共轭程度（0-10，越高越共轭）
    conjugation_degree = np.random.randint(0, 11, n_samples)

    # 金属-配体键长 (Å)
    bond_length = np.random.uniform(1.8, 2.5, n_samples)

    # 维度（1D, 2D, 3D）
    dimensionality = np.random.randint(1, 4, n_samples)

    # 电负性差异
    electronegativity_diff = np.random.uniform(0.5, 2.5, n_samples)

    # 带隙 (eV)
    # 影响因素：
    # - 金属类型
    # - 配体共轭程度（共轭越高，带隙越小）
    # - 维度（3D 通常带隙较小）

    metal_bandgap_base = {
        0: 3.5,  # Zn
        1: 2.0,  # Cu
        2: 4.5,  # Zr
        3: 1.5,  # Fe
        4: 3.0   # Ti
    }

    bandgap = np.array([metal_bandgap_base[m] for m in metal_type])
    bandgap -= conjugation_degree * 0.2  # 共轭降低带隙
    bandgap -= (dimensionality - 1) * 0.3  # 高维度降低带隙
    bandgap += electronegativity_diff * 0.5  # 电负性差异增加带隙
    bandgap += np.random.normal(0, 0.3, n_samples)  # 噪声

    bandgap = np.clip(bandgap, 0.5, 6.0)

    df = pd.DataFrame({
        'mof_id': [f'MOF_BG_{i:04d}' for i in range(n_samples)],
        'metal_type': metal_type,
        'conjugation_degree': conjugation_degree,
        'bond_length': bond_length,
        'dimensionality': dimensionality,
        'electronegativity_diff': electronegativity_diff,
        'bandgap': bandgap
    })

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"能带数据集已保存到: {save_path}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample MOF datasets')
    parser.add_argument('--type', choices=['adsorption', 'bandgap', 'both'],
                       default='adsorption', help='Dataset type')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples')
    parser.add_argument('--output_dir', default='../../data/examples',
                       help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ['adsorption', 'both']:
        print("Generating adsorption dataset...")
        df_ads = generate_mof_adsorption_data(
            n_samples=args.n_samples,
            save_path=output_dir / 'mof_adsorption_dataset.csv'
        )
        print(f"\nAdsorption dataset summary:")
        print(df_ads.describe())

    if args.type in ['bandgap', 'both']:
        print("\nGenerating bandgap dataset...")
        df_bg = generate_mof_bandgap_data(
            n_samples=args.n_samples // 2,
            save_path=output_dir / 'mof_bandgap_dataset.csv'
        )
        print(f"\nBandgap dataset summary:")
        print(df_bg.describe())

    print("\nDone!")
