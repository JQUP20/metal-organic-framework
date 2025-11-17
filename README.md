# AI 与 MOF 基础课程

<div align="center">

**AI 赋能的金属有机框架(MOF)材料设计与性能预测**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ASE](https://img.shields.io/badge/ASE-3.22+-orange.svg)](https://wiki.fysik.dtu.dk/ase/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-purple.svg)](https://shap.readthedocs.io/)

</div>

---

## 📚 课程概述

本仓库包含"AI 与 MOF 基础课程"的完整教学材料，涵盖理论讲解和实操练习。

### 已完成内容

#### ✅ 第一天：AI 与 MOF 的基础认知与科学范式
#### ✅ 第二天：传统机器学习方法在 MOF 性质预测中的应用

### 第一天：理论部分
- ✅ 人工智能的科学革命：从符号主义到深度学习的演进路径
- ✅ AI 基本理论框架：监督学习、无监督学习、强化学习、生成模型
- ✅ 机器学习典型流程：数据预处理 → 特征工程 → 模型训练 → 验证与解释
- ✅ 深度学习简介：神经网络结构、反向传播算法、过拟合与泛化能力
- ✅ MOF 材料基础知识：结构组成（有机配体-金属节点-拓扑网络）
- ✅ MOF 的发展历程与研究热点：从 MOF-5、ZIF-8 到多功能杂化框架
- ✅ MOF 在能源、环境与医药领域的应用
- ✅ AI 与 MOF 的融合趋势：从实验发现到智能预测与自主设计

### 第一天：实操部分
- ✅ Linux 与 Python 科学计算环境搭建 (Anaconda/Mamba)
- ✅ MOF 结构可视化与格式转换 (ASE, py3Dmol)
- ✅ 数据集构建：从 CoRE-MOF、CSD、QMOF 数据库筛选与清洗
- ✅ 特征提取：Zeo++、Poreblazer、MOFid 工具的使用
- ✅ Python 实现 MOF 比表面积、孔径分布、能量参数计算

### 第二天：理论部分
- ✅ 机器学习在材料科学中的应用模式与发展历程
- ✅ MOF 结构-性质关系的定量表征方法 (QSAR/QSPR)
- ✅ 特征工程：几何、化学、拓扑特征的提取与缩放
- ✅ 传统机器学习算法详解：
  - 线性回归 (Linear Regression, Ridge, Lasso, ElasticNet)
  - 支持向量机 (SVM/SVR)
  - 随机森林 (Random Forest)
  - 梯度提升 (XGBoost, LightGBM, CatBoost)
- ✅ 模型评估与验证：交叉验证、学习曲线、过拟合诊断
- ✅ 可解释性分析：SHAP、特征重要性、排列重要性

### 第二天：实操部分
- ✅ MOF 吸附数据集构建与探索性分析
- ✅ 多种机器学习模型训练与超参数优化 (GridSearchCV)
- ✅ 模型性能对比与可视化 (R², RMSE, MAE)
- ✅ SHAP 可解释性分析：
  - Summary Plot (蜂群图)
  - Dependence Plot (特征依赖图)
  - Waterfall Plot (单样本解释)
  - Feature Interaction (特征交互)
- ✅ 案例实战：CO₂/CH₄ 吸附预测与选择性分析
- ✅ 模型保存与部署

---

## 📁 项目结构

```
metal-organic-framework/
├── config/                      # 配置文件
│   └── setup_guide.md          # 环境搭建指南
├── data/                        # 数据目录
│   ├── datasets/               # 数据集 (CoRE-MOF, QMOF 等)
│   └── examples/               # 示例数据
│       ├── example_mof.cif                    # 示例 MOF 结构
│       ├── mof_adsorption_dataset.csv         # 吸附数据集 ✨
│       └── mof_bandgap_dataset.csv            # 能带数据集 ✨
├── docs/                        # 文档
│   ├── theory/                 # 理论文档
│   │   ├── 01_ai_fundamentals.md              # AI 基础理论
│   │   ├── 02_mof_fundamentals.md             # MOF 基础知识
│   │   ├── 03_ai_mof_integration.md           # AI 与 MOF 融合趋势
│   │   └── 04_traditional_ml_methods.md       # 传统机器学习方法 ✨
│   └── tutorials/              # 教程
├── notebooks/                   # Jupyter Notebooks
│   ├── day1_tutorial.ipynb     # 第一天实操教程
│   └── day2_tutorial.ipynb     # 第二天实操教程 ✨
├── src/                         # 源代码
│   ├── visualization/          # 可视化工具
│   │   └── mof_visualizer.py
│   ├── data_processing/        # 数据处理
│   │   ├── dataset_builder.py
│   │   └── generate_sample_data.py            # 数据生成工具 ✨
│   ├── feature_extraction/     # 特征提取
│   │   └── geometric_features.py
│   ├── calculations/           # 性能计算
│   │   └── property_calculator.py
│   └── ml_models/              # 机器学习模型 ✨
│       ├── model_trainer.py              # 模型训练器
│       ├── model_evaluator.py            # 模型评估器
│       └── interpretability.py           # 可解释性分析
├── tests/                       # 测试代码
├── requirements.txt             # Python 依赖 (pip)
├── environment.yml              # Conda 环境配置
└── README.md                    # 本文件

✨ = 第二天新增内容
```

---

## 🚀 快速开始

### 1. 环境搭建

#### 方法一：使用 Conda (推荐)

```bash
# 克隆仓库
git clone <repository-url>
cd metal-organic-framework

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate ai-mof-course
```

#### 方法二：使用 pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 详细安装指南

查看完整的环境搭建指南：[config/setup_guide.md](config/setup_guide.md)

---

### 2. 运行示例

#### Jupyter Notebook 教程

```bash
# 启动 Jupyter Lab
jupyter lab

# 打开 notebooks/day1_tutorial.ipynb
```

#### Python 脚本示例

```python
# MOF 结构可视化
from src.visualization.mof_visualizer import MOFVisualizer

viz = MOFVisualizer('data/examples/example_mof.cif')
viz.print_info()
viz.plot_2d_projection(axis='z')
```

```python
# 特征提取
from src.feature_extraction.geometric_features import SimplifiedGeometricExtractor

extractor = SimplifiedGeometricExtractor()
features = extractor.calculate_features('data/examples/example_mof.cif')
print(features)
```

```python
# 性能计算
from src.calculations.property_calculator import PropertyCalculator

calc = PropertyCalculator()
props = calc.calculate_all_properties('data/examples/example_mof.cif')
calc.print_properties(props)
```

**第二天新增：机器学习模型**

```python
# 训练 MOF 性质预测模型
from src.ml_models.model_trainer import MOFModelTrainer
import pandas as pd

# 加载数据
df = pd.read_csv('data/examples/mof_adsorption_dataset.csv')
X = df.drop(['mof_id', 'co2_uptake', 'ch4_uptake', 'selectivity'], axis=1)
y = df['co2_uptake']

# 训练模型
trainer = MOFModelTrainer(random_state=42)
X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)

results = trainer.train(
    X_train, y_train,
    model_type='xgboost',  # 或 'rf', 'lightgbm', 'svr'
    optimize=True,
    cv=5
)

# 评估
from src.ml_models.model_evaluator import ModelEvaluator
evaluator = ModelEvaluator()
y_pred = trainer.predict(X_test)
metrics = evaluator.calculate_metrics(y_test, y_pred)
evaluator.print_metrics(metrics)
evaluator.plot_predictions(y_test, y_pred)
```

```python
# SHAP 可解释性分析
from src.ml_models.interpretability import SHAPAnalyzer

shap_analyzer = SHAPAnalyzer(
    model=trainer.model,
    X=X_test,
    feature_names=X.columns.tolist()
)

shap_values = shap_analyzer.compute_shap_values(X_test)
shap_analyzer.summary_plot(plot_type='dot', max_display=20)  # 蜂群图
shap_analyzer.waterfall_plot(sample_idx=0)  # 单样本解释
```

---

## 📖 学习路径

### 第一天：AI 与 MOF 基础 (6-8 小时)

**理论学习 (2-3 小时)**
1. 阅读 [AI 基础理论](docs/theory/01_ai_fundamentals.md)
2. 阅读 [MOF 基础知识](docs/theory/02_mof_fundamentals.md)
3. 阅读 [AI 与 MOF 融合趋势](docs/theory/03_ai_mof_integration.md)

**环境搭建 (1-2 小时)**
1. 按照 [环境搭建指南](config/setup_guide.md) 安装依赖
2. 验证安装：运行测试脚本

**实操练习 (3-4 小时)**
1. 运行 [第一天实操教程](notebooks/day1_tutorial.ipynb)
2. 尝试可视化不同 MOF 结构
3. 提取几何特征并分析
4. 计算性能参数

### 第二天：传统机器学习方法 (6-8 小时)

**理论学习 (2-3 小时)**
1. 阅读 [传统机器学习方法](docs/theory/04_traditional_ml_methods.md)
2. 理解特征工程与模型选择
3. 学习 SHAP 可解释性分析

**实操练习 (4-5 小时)**
1. 运行 [第二天实操教程](notebooks/day2_tutorial.ipynb)
2. 训练多种机器学习模型（Linear, RF, XGBoost, LightGBM）
3. 超参数优化与模型比较
4. SHAP 可解释性分析
5. 完成练习题：
   - 预测 CH₄ 吸附量
   - 预测 CO₂/CH₄ 选择性
   - 特征工程实验

**进阶任务 (可选)**
1. 下载真实 QMOF 数据集
2. 复现文献中的机器学习模型
3. 探索特征交互效应

---

## 🔧 核心工具

### 第一天工具

**MOF 可视化**
- **mof_visualizer.py**: 3D/2D 可视化、格式转换
- 支持格式：CIF, XYZ, PDB, VASP, JSON

**数据处理**
- **dataset_builder.py**: 数据集构建、清洗、划分
- 支持 CoRE-MOF、QMOF、CSD 数据库

**特征提取**
- **geometric_features.py**: 几何特征提取
- 集成 Zeo++、MOFid（可选）
- 简化方法（无需外部工具）

**性能计算**
- **property_calculator.py**: 比表面积、孔径分布、能量参数
- BET 计算、PSD 分析

### 第二天工具 ✨

**机器学习模型**
- **model_trainer.py**: 多算法模型训练器
  - 支持算法：Linear, Ridge, Lasso, SVM, Random Forest, XGBoost, LightGBM, CatBoost
  - 超参数自动优化（GridSearchCV）
  - 交叉验证
  - 模型保存/加载

**模型评估**
- **model_evaluator.py**: 全面的模型评估工具
  - 评估指标：R², RMSE, MAE, MAPE
  - 可视化：预测散点图、残差图、学习曲线
  - 模型对比
  - 特征重要性可视化

**可解释性分析**
- **interpretability.py**: SHAP 可解释性分析
  - SHAP Summary Plot (蜂群图)
  - SHAP Dependence Plot (特征依赖图)
  - SHAP Waterfall Plot (单样本解释)
  - SHAP Force Plot (力图)
  - 特征交互分析
  - 排列重要性

**数据生成**
- **generate_sample_data.py**: 生成模拟 MOF 数据集
  - 吸附数据集（CO₂、CH₄）
  - 能带结构数据集

---

## 📊 数据资源

### 推荐数据库
1. **CoRE MOF Database** (14,000+ 结构)
   - 官网: [CoRE-MOF GitHub](https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs)

2. **QMOF Database** (20,000+ 结构 + DFT 数据)
   - 官网: [QMOF GitHub](https://github.com/Andrew-S-Rosen/QMOF)

3. **Cambridge Structural Database (CSD)**
   - 官网: [CSD](https://www.ccdc.cam.ac.uk/solutions/csd-core/components/csd/)

4. **hMOF Database** (130万+ 假设结构)
   - 用于高通量筛选

### 下载说明
详见 `src/data_processing/dataset_builder.py` 中的下载指南

---

## 🛠️ 外部工具安装 (可选)

### Zeo++ (孔结构分析)
```bash
wget http://www.zeoplusplus.org/Zeo++-0.3.tar.gz
tar -xzf Zeo++-0.3.tar.gz
cd Zeo++-0.3
make
export PATH=$PATH:$(pwd)
```

### Poreblazer (孔隙率计算)
```bash
git clone https://github.com/SarkisovTeam/Poreblazer.git
cd Poreblazer
make
export PATH=$PATH:$(pwd)
```

### Avogadro (可视化软件)
```bash
# Ubuntu/Debian
sudo apt-get install avogadro

# 或下载 AppImage
wget https://github.com/OpenChemistry/avogadrolibs/releases/download/1.95.1/Avogadro2-x86_64.AppImage
chmod +x Avogadro2-x86_64.AppImage
```

---

## 🧪 测试

```bash
# 运行测试 (如果已编写测试)
pytest tests/

# 或手动测试
python src/visualization/mof_visualizer.py
python src/feature_extraction/geometric_features.py
```

---

## 📝 常见问题

### Q: Zeo++ 安装失败？
**A**: Zeo++ 编译需要 C++ 编译器，确保已安装 `build-essential` (Ubuntu) 或 `Xcode Command Line Tools` (Mac)。也可以使用简化方法，不依赖 Zeo++。

### Q: 内存不足？
**A**: 减小采样点数 `n_samples` 参数，例如从 10000 降至 2000。

### Q: Jupyter Notebook 无法显示 3D 可视化？
**A**: 确保已安装 `py3Dmol` 和 `ipywidgets`，并重启 Jupyter。

### Q: 如何加速批量计算？
**A**: 使用多进程并行，参考 `joblib` 或 `multiprocessing` 库。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📜 许可证

MIT License

---

## 📧 联系方式

如有问题，请提交 Issue 或联系课程负责人。

---

## 🌟 致谢

- ASE (Atomic Simulation Environment)
- Zeo++ 开发团队
- CoRE-MOF、QMOF 数据库贡献者
- 所有开源社区贡献者

---

## 📅 课程安排

- **第一天**: AI 与 MOF 基础认知 ✅
  - 理论：AI基础、MOF基础、融合趋势
  - 实操：环境搭建、可视化、特征提取、性能计算

- **第二天**: 传统机器学习方法在 MOF 性质预测中的应用 ✅
  - 理论：机器学习算法、特征工程、模型评估、可解释性
  - 实操：多算法训练、超参数优化、SHAP分析、案例实战

- **第三天**: 深度学习与图神经网络 (规划中)
  - 理论：神经网络、图神经网络、注意力机制
  - 实操：PyTorch基础、GNN模型、MOF图表示

- **第四天**: 生成模型与 MOF 逆向设计 (规划中)
  - 理论：VAE、GAN、扩散模型
  - 实操：MOF生成、性能导向设计

- **第五天**: 高通量筛选与主动学习 (规划中)
  - 理论：主动学习策略、不确定性量化
  - 实操：高通量计算、候选MOF筛选

---

## 🎯 学习成果

完成本课程后，你将能够：

✅ 理解 AI 在材料科学中的应用原理
✅ 熟练使用 Python 处理和分析 MOF 结构
✅ 构建 MOF 性质预测的机器学习模型
✅ 使用 SHAP 等工具解释模型决策
✅ 掌握特征工程和超参数优化技巧
✅ 为深度学习和高级方法打下基础

---

**祝学习愉快！** 🎓

如果觉得有帮助，请给个 ⭐ Star！

---

## 🔄 更新日志

**v0.2.0** (2024-11)
- ✅ 新增第二天完整课程材料
- ✅ 新增机器学习模型训练器（8种算法）
- ✅ 新增SHAP可解释性分析工具
- ✅ 新增模拟数据集生成工具
- ✅ 新增第二天实操Jupyter Notebook
- ✅ 新增传统机器学习理论文档（60+页）

**v0.1.0** (2024-11)
- ✅ 第一天课程材料
- ✅ MOF可视化和特征提取工具
- ✅ 性能计算模块
- ✅ 基础教程和文档
