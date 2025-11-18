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
#### ✅ 第三天：图神经网络（GNN）与 MOF 结构-性能建模
#### ✅ 第四天：生成模型与逆向 MOF 设计
#### ✅ 第五天：大语言模型（LLM）在 MOF 智能设计中的应用

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

### 第三天：理论部分
- ✅ 图神经网络基础：从分子图到晶体图表示
- ✅ 图的数学定义：节点、边、邻接矩阵
- ✅ MOF的图表示方法：
  - 节点特征设计（原子属性）
  - 边特征设计（距离、键角、Gaussian扩展）
  - 周期性边界条件处理
- ✅ 消息传递神经网络 (MPNN) 框架
- ✅ 主流GNN模型详解：
  - CGCNN (Crystal Graph Convolutional Neural Network)
  - MEGNet (Materials Graph Network)
  - SchNet, ALIGNN, DimeNet
- ✅ GNN在MOF中的应用：性质预测、结构筛选
- ✅ 模型训练与优化策略
- ✅ 案例研究：QMOF带隙预测、CO₂吸附预测

### 第三天：实操部分
- ✅ MOF结构到图的转换工具 (graph_builder.py)
- ✅ CGCNN模型实现（门控消息传递）
- ✅ SimpleMEGNet模型实现（三级图）
- ✅ Gaussian距离扩展
- ✅ PyTorch Geometric集成
- ✅ 批量图构建
- ✅ 模型测试与验证

### 第四天：理论部分
- ✅ 生成模型 vs 预测模型：从"性质预测"到"结构生成"的范式转变
- ✅ 变分自编码器（VAE）：
  - ELBO损失函数与重参数化技巧
  - MOF-VAE架构设计
  - 潜在空间学习与插值
- ✅ 扩散模型（Diffusion Models）：
  - DDPM前向和反向过程
  - 噪声调度策略（linear, cosine）
  - 时间条件生成
  - E(3)等变扩散模型
- ✅ 逆向设计策略：
  - 潜在空间优化
  - 条件生成
  - 多目标优化
- ✅ 贝叶斯优化（Bayesian Optimization）：
  - 高斯过程代理模型
  - 采集函数（EI, UCB, PI）
  - 探索-利用平衡
- ✅ 可合成性评估：SA Score、稳定性预测
- ✅ 案例研究：CO₂捕获MOF的目标导向设计

### 第四天：实操部分
- ✅ MOF-VAE模型实现：
  - GNN编码器（图→潜在分布）
  - 图解码器（潜在向量→图）
  - ELBO损失与训练循环
  - 潜在空间采样与插值
- ✅ MOF扩散模型实现：
  - 正弦位置编码（时间嵌入）
  - 噪声预测网络
  - 前向扩散与反向去噪
  - DDPM采样算法
- ✅ 贝叶斯优化工具：
  - 高斯过程实现
  - 三种采集函数（EI/UCB/PI）
  - 多起点优化
  - MOF逆向设计器
- ✅ 端到端逆向设计流程：VAE + BO
- ✅ 性质导向的MOF生成与优化

### 第五天：理论部分
- ✅ 大语言模型的崛起与科学研究新范式
  - ChatGPT → MatGPT → ChemLLM 演进
  - 从通用LLM到材料专用LLM
  - LLM在材料科学中的科学范式转变
- ✅ LLM在材料科学中的认知与生成能力
  - Text-to-Structure（文本到结构）生成
  - Text-to-Experiment（文本到实验）指导
  - 三种实现途径：RAG、端到端、混合方法
- ✅ 材料知识图谱与LLM的融合
  - 知识图谱基础（MaterialsKG、MATTERverse）
  - 检索增强生成（RAG）系统
  - 向量数据库与语义检索
- ✅ LLM在MOF研究中的应用
  - 文献挖掘与知识抽取
  - 语义筛选与智能检索
  - AutoML自动化分析
  - 提示工程（Prompt Engineering）
- ✅ 多模态智能体（Multi-agent）助力自主材料发现
  - 智能体架构设计
  - 自主发现循环
  - 经验回放与持续学习

### 第五天：实操部分
- ✅ Text-to-Structure生成工具：
  - LLM参数提取（从自然语言描述）
  - 三种生成方法：检索、生成、混合
  - 结构验证与排序
- ✅ 文献语料的自动标注与知识抽取：
  - 基于正则表达式的信息提取
  - LLM增强的语义解析
  - 知识图谱构建
- ✅ RAG（检索增强生成）系统：
  - 文档分块与向量化
  - 向量数据库构建
  - 语义相似度检索
  - 上下文增强的答案生成
- ✅ 提示工程工具包：
  - 预定义提示模板（提取、设计、预测）
  - Few-shot学习管理器
  - Chain-of-Thought（思维链）提示
  - 自动提示优化
- ✅ AutoML与LLM集成：
  - 自动特征工程
  - 智能模型选择
  - 超参数优化
- ✅ 自学习型MOF智能体：
  - 文献搜索→设计→预测→评估循环
  - 经验记忆与知识积累
  - 自主材料发现
- ✅ 端到端设计流程：文本→结构→性质预测

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
│   │   ├── 04_traditional_ml_methods.md       # 传统机器学习方法
│   │   ├── 05_gnn_for_mof.md                  # 图神经网络理论 ⭐
│   │   ├── 06_generative_models_for_mof.md    # 生成模型理论 🎨
│   │   └── 07_llm_for_mof.md                  # LLM应用理论 🤖
│   ├── tutorials/              # 教程
│   ├── DAY3_SUMMARY.md         # 第三天课程总结 ⭐
│   ├── DAY4_SUMMARY.md         # 第四天课程总结 🎨
│   └── DAY5_SUMMARY.md         # 第五天课程总结 🤖
├── notebooks/                   # Jupyter Notebooks
│   ├── day1_tutorial.ipynb     # 第一天实操教程
│   ├── day2_tutorial.ipynb     # 第二天实操教程
│   ├── day4_tutorial.ipynb     # 第四天实操教程 🎨
│   └── day5_tutorial.ipynb     # 第五天实操教程 🤖
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
│   ├── ml_models/              # 机器学习模型 (Day 2)
│   │   ├── model_trainer.py              # 模型训练器
│   │   ├── model_evaluator.py            # 模型评估器
│   │   └── interpretability.py           # 可解释性分析
│   ├── graph_models/           # 图神经网络模型 (Day 3) ⭐
│   │   ├── graph_builder.py              # MOF到图转换
│   │   └── gnn_models.py                 # CGCNN & MEGNet实现
│   ├── generative_models/      # 生成模型 (Day 4) 🎨
│   │   ├── mof_vae.py                    # MOF-VAE模型
│   │   ├── mof_diffusion.py              # MOF扩散模型
│   │   ├── bayesian_optimizer.py         # 贝叶斯优化工具
│   │   ├── visualization.py              # 可视化工具
│   │   └── evaluator.py                  # 生成MOF评估工具
│   └── llm_tools/              # LLM工具 (Day 5) 🤖
│       ├── text_to_structure.py          # Text-to-Structure生成
│       ├── mof_agent_system.py           # MOF智能体系统
│       ├── prompt_engineering.py         # 提示工程工具
│       └── rag_system.py                 # RAG检索增强生成
├── tests/                       # 测试代码
├── requirements.txt             # Python 依赖 (pip)
├── environment.yml              # Conda 环境配置
└── README.md                    # 本文件

✨ = 第二天新增内容
⭐ = 第三天新增内容
🎨 = 第四天新增内容
🤖 = 第五天新增内容
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

**第三天新增：图神经网络**

```python
# 构建MOF图
from src.graph_models import MOFGraphBuilder

builder = MOFGraphBuilder(cutoff_radius=8.0, max_neighbors=12)
graph = builder.build_graph_from_cif('data/examples/example_mof.cif')

# 转换为PyTorch Geometric格式
data = builder.to_pytorch_geometric(graph)
data.y = torch.tensor([3.5])  # 目标值（如吸附量）

print(f"节点数: {data.x.shape[0]}")
print(f"边数: {data.edge_index.shape[1]}")
```

```python
# 训练CGCNN模型
import torch
from src.graph_models import CGCNN
from torch_geometric.loader import DataLoader

# 准备数据集
dataset = [...]  # PyG Data对象列表
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型
model = CGCNN(
    node_input_dim=4,
    edge_input_dim=24,
    hidden_dim=128,
    num_conv_layers=4,
    dropout=0.1
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

# 预测
model.eval()
with torch.no_grad():
    prediction = model(data)
    print(f"预测值: {prediction.item():.3f}")
```

**第四天新增：生成模型与逆向设计**

```python
# 训练MOF-VAE模型
from src.generative_models.mof_vae import MOFVAE, MOFVAETrainer
import torch
from torch_geometric.loader import DataLoader

# 加载数据集
train_dataset = [...]  # MOF图数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建VAE模型
vae_model = MOFVAE(
    node_input_dim=4,
    edge_input_dim=24,
    hidden_dim=128,
    latent_dim=64,  # 潜在空间维度
    max_num_nodes=100,
    beta=0.5  # KL散度权重
)

# 训练
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
trainer = MOFVAETrainer(vae_model, optimizer, device='cuda')

for epoch in range(200):
    train_loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: Loss = {train_loss['loss']:.4f}")

# 生成新MOF
node_gen, adj_gen, edge_gen = vae_model.sample(num_samples=10)
print(f"生成了 10 个新的 MOF 结构")
```

```python
# 贝叶斯优化逆向设计
from src.generative_models.bayesian_optimizer import MOFInverseDesigner

# 定义性质预测函数
def predict_co2_uptake(structure_dict):
    # 从生成的结构预测CO2吸附量
    node = structure_dict['node_features']
    adj = structure_dict['adj_matrix']
    # ... 特征提取和预测
    return predicted_uptake

# 创建逆向设计器
designer = MOFInverseDesigner(
    vae_model=vae_model,
    property_predictor=predict_co2_uptake,
    latent_bounds=[[-3.0, 3.0]] * 64  # 搜索范围
)

# 优化：寻找高CO2吸附量的MOF
result = designer.optimize_for_property(
    target_property='CO2_uptake',
    n_iterations=50,
    acquisition='ei',  # 期望改进
    verbose=True
)

print(f"最优CO2吸附量: {result['property_value']:.2f} mmol/g")
print(f"优化完成，共评估 {len(result['optimization_history'][0])} 个MOF")
```

```python
# 扩散模型生成MOF
from src.generative_models.mof_diffusion import MOFDiffusion

# 创建扩散模型
diffusion_model = MOFDiffusion(
    node_dim=4,
    edge_dim=24,
    hidden_dim=128,
    num_timesteps=1000
)

# 从噪声生成MOF（需要提供图拓扑）
template_edge_index = ...  # 图的边索引（拓扑）
template_edge_attr = ...   # 边特征

x_generated = diffusion_model.sample(
    num_nodes=50,
    edge_index=template_edge_index,
    edge_attr=template_edge_attr
)

print(f"生成的节点特征: {x_generated.shape}")
```

```python
# 可视化潜在空间
from src.generative_models.visualization import LatentSpaceVisualizer

# 创建可视化器
viz = LatentSpaceVisualizer()

# 编码训练数据到潜在空间
latent_vectors = []
properties = []
for batch in train_loader:
    with torch.no_grad():
        mu, logvar = vae_model.encode(batch)
        z = vae_model.reparameterize(mu, logvar)
        latent_vectors.append(z.cpu().numpy())
        properties.append(batch.y.cpu().numpy())

latent_vectors = np.concatenate(latent_vectors)
properties = np.concatenate(properties)

# 绘制2D潜在空间（按性质着色）
viz.plot_latent_space_2d(
    latent_vectors,
    properties,
    method='tsne',
    property_name='CO2 Uptake (mmol/g)'
)
```

```python
# 评估生成MOF的质量
from src.generative_models.evaluator import GeneratedMOFEvaluator

# 生成一批MOF
generated_mofs = []
with torch.no_grad():
    for _ in range(100):
        node, adj, edge = vae_model.sample(num_samples=1)
        generated_mofs.append({
            'node_features': node[0],
            'adj_matrix': adj[0],
            'edge_features': edge[0]
        })

# 创建评估器
evaluator = GeneratedMOFEvaluator(verbose=True)

# 综合评估
results = evaluator.comprehensive_evaluation(
    generated_structures=generated_mofs,
    reference_structures=train_dataset[:100],
    generated_properties=np.array([...]),  # 预测的性质
    reference_properties=np.array([...]),   # 参考性质
    property_name='CO2 Uptake'
)

print(f"有效率: {results['validity']['validity_rate']:.1%}")
print(f"唯一率: {results['uniqueness']['uniqueness_rate']:.1%}")
print(f"多样性得分: {results['diversity']['diversity_score']:.3f}")
```

**第五天新增：LLM智能设计工具**

```python
# Text-to-Structure: 从文本生成MOF
from src.llm_tools.text_to_structure import Text2StructureGenerator

# 初始化生成器（可接入真实LLM API）
generator = Text2StructureGenerator()

# 自然语言描述
query = "Design a copper-based MOF with paddle-wheel clusters for CO2 capture"

# 生成MOF候选
result = generator.generate_from_text(
    query,
    method="hybrid",  # retrieval, generation, 或 hybrid
    top_k=3,
    verbose=True
)

# 查看结果
for i, candidate in enumerate(result['candidates']):
    print(f"{i+1}. {candidate['name']}")
    print(f"   Metal: {candidate['metal']}, Linker: {candidate['linker']}")
    print(f"   Predicted CO2 uptake: {candidate['predicted_co2']:.2f} mmol/g")
```

```python
# RAG系统：检索增强生成
from src.llm_tools.rag_system import MOF_RAG_System, KnowledgeBase

# 构建知识库
kb = KnowledgeBase()
kb.add_paper(
    title="UiO-66: A Highly Stable Zr-MOF",
    authors="Cavka et al.",
    year=2008,
    abstract="UiO-66 is a zirconium-based MOF with exceptional stability..."
)

# 创建RAG系统
rag = MOF_RAG_System()
rag.add_documents(
    kb.get_all_documents(),
    kb.get_metadata(),
    chunk_size=300
)

# 查询知识库
result = rag.query(
    "What is the CO2 uptake of UiO-66?",
    top_k=3,
    return_sources=True
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents retrieved")
```

```python
# MOF智能体：自主发现系统
from src.llm_tools.mof_agent_system import MOFAgent

# 创建智能体
agent = MOFAgent()

# 定义目标
goal = {
    "property": "CO2_uptake",
    "target_value": 8.0,  # mmol/g
    "constraints": {
        "stability": "high",
        "cost": "low"
    }
}

# 启动自主发现
result = agent.discover_mof(goal, max_iterations=5)

print(f"Best MOF found: {result['best_mof']['name']}")
print(f"Predicted CO2 uptake: {result['best_mof']['predicted_value']:.2f} mmol/g")
print(f"Iterations completed: {result['iterations_completed']}")
```

```python
# 提示工程：优化LLM性能
from src.llm_tools.prompt_engineering import PromptTemplate, ChainOfThoughtPrompt

# 使用预定义模板
template = PromptTemplate.get_template('extraction')
prompt = template.format(
    text="UiO-66 is a Zr-MOF with BDC linker, showing CO2 uptake of 3.0 mmol/g."
)

# Chain-of-Thought推理
cot = ChainOfThoughtPrompt()
cot_prompt = cot.generate(
    task="predict MOF stability",
    context={"mof_name": "UiO-66", "metal": "Zr", "linker": "BDC"}
)

# Few-shot学习
from src.llm_tools.prompt_engineering import FewShotManager

few_shot = FewShotManager()
few_shot.add_example(
    input_text="Cu-BTC MOF",
    output_text="Metal: Cu, Linker: BTC, Topology: paddle-wheel"
)

few_shot_prompt = few_shot.generate_prompt(
    new_input="Zr-UiO-66",
    task="extract MOF components"
)
```

```python
# AutoML：自动机器学习
from src.llm_tools.mof_agent_system import AutoMLOptimizer
import pandas as pd

# 加载数据
df = pd.read_csv('data/examples/mof_adsorption_dataset.csv')
X = df.drop(['mof_id', 'co2_uptake'], axis=1)
y = df['co2_uptake']

# 创建AutoML优化器
automl = AutoMLOptimizer()

# 自动优化
result = automl.auto_optimize(
    X, y,
    task_description="Predict CO2 uptake for MOFs",
    max_iterations=10
)

print(f"Best model: {result['best_model']['type']}")
print(f"Best score: {result['best_score']:.4f}")
print(f"Optimized features: {result['engineered_features'][:5]}...")
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

### 第三天：图神经网络 (6-8 小时)

**理论学习 (2-3 小时)**
1. 阅读 [图神经网络理论](docs/theory/05_gnn_for_mof.md)
2. 理解消息传递机制
3. 学习主流GNN模型（CGCNN, MEGNet, ALIGNN等）

**实操练习 (4-5 小时)**
1. 阅读 [第三天课程总结](docs/DAY3_SUMMARY.md)
2. 使用 `graph_builder.py` 将MOF转换为图
3. 测试 CGCNN 和 SimpleMEGNet 模型
4. 理解Gaussian距离扩展
5. 对比GNN与传统ML的性能

**进阶任务 (可选)**
1. 实现注意力可视化
2. 在QMOF数据集上训练GNN
3. 尝试实现ALIGNN模型
4. 探索多任务学习

### 第四天：生成模型与逆向设计 (8-10 小时)

**理论学习 (3-4 小时)**
1. 阅读 [生成模型理论](docs/theory/06_generative_models_for_mof.md)
2. 理解VAE、扩散模型的原理
3. 学习贝叶斯优化与逆向设计策略
4. 了解可合成性评估方法

**实操练习 (5-6 小时)**
1. 阅读 [第四天课程总结](docs/DAY4_SUMMARY.md)
2. 运行 [第四天实操教程](notebooks/day4_tutorial.ipynb)
3. 训练 MOF-VAE 模型并理解潜在空间
4. 使用 VAE 生成新 MOF 结构
5. 实现潜在空间插值可视化
6. 运行贝叶斯优化进行逆向设计
7. 测试扩散模型生成过程
8. 使用可视化工具分析潜在空间
9. 使用评估工具评估生成MOF质量
10. 完成练习题：
   - 优化特定性质（如高CO₂吸附量）
   - 多目标优化实验
   - 评估生成MOF的合理性
   - 调整VAE超参数观察效果

**进阶任务 (可选)**
1. 实现条件VAE（cVAE）
2. 探索E(3)等变扩散模型
3. 实现多目标Pareto优化
4. 结合DFT计算验证生成的MOF
5. 尝试强化学习辅助生成

### 第五天：LLM智能设计 (8-10 小时)

**理论学习 (3-4 小时)**
1. 阅读 [LLM应用理论](docs/theory/07_llm_for_mof.md)
2. 理解Text-to-Structure生成原理
3. 学习RAG（检索增强生成）系统
4. 了解提示工程技术
5. 掌握智能体架构设计

**实操练习 (5-6 小时)**
1. 阅读 [第五天课程总结](docs/DAY5_SUMMARY.md)
2. 运行 [第五天实操教程](notebooks/day5_tutorial.ipynb)
3. 使用Text-to-Structure工具生成MOF
4. 构建RAG系统并进行文献问答
5. 测试提示工程模板和优化
6. 运行AutoML自动优化实验
7. 体验MOF智能体自主发现流程
8. 完成练习题：
   - 创建自定义提示模板
   - 构建领域知识库
   - 设计多轮对话智能体

**进阶任务 (可选)**
1. 接入真实LLM API（OpenAI、Anthropic、本地LLaMA）
2. 使用sentence-transformers进行真实嵌入
3. 集成FAISS或Chroma向量数据库
4. 实现多智能体协作系统
5. 构建完整的MOF发现平台
6. 评估幻觉和提升输出质量

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

### 第三天工具 ⭐

**图构建**
- **graph_builder.py**: MOF结构到图的转换
  - CIF文件读取（基于ASE）
  - 邻接矩阵构建（距离截断/KNN）
  - 节点特征提取（元素属性）
  - 边特征提取（Gaussian距离扩展）
  - 周期性边界条件处理
  - PyTorch Geometric格式转换

**图神经网络模型**
- **gnn_models.py**: GNN模型实现
  - **CGCNN**: Crystal Graph Convolutional Neural Network
    - 门控消息传递
    - 残差连接
    - 专为晶体材料设计
  - **SimpleMEGNet**: 简化的Materials Graph Network
    - 三级图（节点、边、全局）
    - 丰富的信息流
    - 更强的表达能力

### 第四天工具 🎨

**生成模型**
- **mof_vae.py**: MOF变分自编码器
  - **MOFEncoder**: GNN编码器（图→潜在分布）
  - **MOFDecoder**: 图解码器（潜在向量→图）
  - **MOFVAE**: 完整VAE模型
    - 重参数化技巧
    - ELBO损失函数
    - 潜在空间采样与插值
  - **MOFVAETrainer**: 训练器
    - 支持训练/验证循环
    - 检查点保存/加载
    - 损失记录

- **mof_diffusion.py**: MOF扩散模型
  - **SinusoidalPositionEmbeddings**: 时间步编码
  - **NoisePredictor**: 噪声预测网络（基于GNN）
  - **GaussianDiffusion**: 扩散过程管理
    - 前向扩散（加噪）
    - 反向去噪（采样）
    - 多种噪声调度（linear, cosine）
  - **MOFDiffusion**: 完整扩散模型
    - DDPM算法
    - 支持轨迹采样
  - **MOFDiffusionTrainer**: 训练器

**逆向设计工具**
- **bayesian_optimizer.py**: 贝叶斯优化工具
  - **GaussianProcess**: 高斯过程代理模型
    - RBF/Matérn核函数
    - 预测均值和方差
  - **AcquisitionFunction**: 采集函数
    - 期望改进（EI）
    - 上置信界（UCB）
    - 改进概率（PI）
  - **BayesianOptimizer**: 贝叶斯优化器
    - 自动初始采样
    - 多起点优化
    - 历史记录跟踪
  - **MOFInverseDesigner**: MOF逆向设计器
    - VAE + BO集成
    - 性质导向生成
    - 潜在空间优化

**可视化工具**
- **visualization.py**: 生成模型可视化
  - **LatentSpaceVisualizer**: 潜在空间可视化
    - 2D降维可视化（PCA/t-SNE）
    - 潜在分布直方图
    - 插值路径可视化
  - **DiffusionVisualizer**: 扩散过程可视化
    - 去噪轨迹可视化
    - 噪声调度曲线
  - **OptimizationVisualizer**: 优化过程可视化
    - 优化历史曲线
    - 采集函数可视化
    - Pareto前沿绘制
  - **TrainingVisualizer**: 训练曲线可视化
    - VAE损失（ELBO, 重构, KL）
    - 多指标对比

**评估工具**
- **evaluator.py**: 生成MOF质量评估
  - **GeneratedMOFEvaluator**: 综合评估器
    - 有效性评估（Validity）
      - 结构完整性检查
      - 物理合理性验证
    - 唯一性评估（Uniqueness）
      - 结构相似度计算
      - 去重统计
    - 多样性评估（Diversity）
      - 平均成对距离
      - 特征空间覆盖
      - 分布熵
    - 新颖性评估（Novelty）
      - k-最近邻距离
      - 与训练集的差异
    - 性质分布评估
      - Wasserstein距离
      - KS检验
      - 统计量对比

### 第五天工具 🤖

**Text-to-Structure生成**
- **text_to_structure.py**: 文本到MOF结构生成
  - **MockLLM**: 模拟LLM（演示用）
  - **Text2StructureGenerator**: 主生成器
    - LLM参数提取
    - 三种生成方法（retrieval/generation/hybrid）
    - 候选排序与验证
  - **StructureValidator**: 结构验证器
    - 必要字段检查
    - 性质合理性验证
  - 易于集成真实LLM API

**RAG检索增强生成**
- **rag_system.py**: 完整RAG系统
  - **SimpleVectorStore**: 向量数据库
    - 文档存储与索引
    - 余弦相似度检索
    - Top-K检索
  - **SimpleEmbedder**: 文本嵌入器
    - 关键词权重嵌入（简化版）
    - 可替换为sentence-transformers
  - **MOF_RAG_System**: RAG主系统
    - 文档分块与嵌入
    - 语义检索
    - 上下文增强生成
    - 来源追踪
  - **KnowledgeBase**: 知识库管理
    - 论文管理
    - MOF数据管理
    - 元数据跟踪

**智能体系统**
- **mof_agent_system.py**: MOF智能体与AutoML
  - **LiteratureMiner**: 文献挖掘器
    - MOF名称提取
    - 金属/配体识别
    - 合成条件提取
    - 性质数据提取
    - 知识图谱构建
  - **AutoMLOptimizer**: AutoML优化器
    - 自动特征工程
    - 模型搜索与选择
    - 超参数优化
    - 交叉验证
  - **MOFAgent**: 自主发现智能体
    - 文献搜索循环
    - 候选设计
    - 性质预测
    - 结果评估与学习
    - 经验记忆

**提示工程**
- **prompt_engineering.py**: 提示工程工具包
  - **PromptTemplate**: 预定义模板
    - 信息提取模板
    - MOF设计模板
    - 性质预测模板
    - 文献总结模板
  - **FewShotManager**: Few-shot学习
    - 示例管理
    - 自动提示生成
    - 示例选择策略
  - **ChainOfThoughtPrompt**: CoT提示
    - 逐步推理生成
    - 中间步骤展示
  - **PromptOptimizer**: 提示优化器
    - 自动变体生成
    - 性能评估
    - 最优提示选择

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

- **第三天**: 图神经网络（GNN）与 MOF 结构-性能建模 ✅
  - 理论：图表示、消息传递、主流GNN模型（CGCNN, MEGNet, ALIGNN等）
  - 实操：图构建、CGCNN/MEGNet实现、PyTorch Geometric

- **第四天**: 生成模型与 MOF 逆向设计 ✅
  - 理论：VAE、扩散模型、贝叶斯优化
  - 实操：MOF-VAE、扩散模型、逆向设计

- **第五天**: 大语言模型（LLM）在 MOF 智能设计中的应用 ✅
  - 理论：LLM演进、Text-to-Structure、RAG系统、智能体
  - 实操：文本生成MOF、文献挖掘、AutoML、自主发现

---

## 🎯 学习成果

完成本课程后，你将能够：

✅ 理解 AI 在材料科学中的应用原理
✅ 熟练使用 Python 处理和分析 MOF 结构
✅ 构建 MOF 性质预测的机器学习模型（传统ML + GNN）
✅ 使用 SHAP 等工具解释模型决策
✅ 掌握特征工程和超参数优化技巧
✅ 理解图神经网络的工作原理
✅ 实现并训练CGCNN和MEGNet模型
✅ 将MOF结构转换为图表示
✅ 掌握生成模型（VAE、Diffusion）原理与实现
✅ 使用贝叶斯优化进行逆向设计
✅ 理解大语言模型在材料科学中的应用
✅ 构建Text-to-Structure生成系统
✅ 实现RAG检索增强生成系统
✅ 掌握提示工程技术和最佳实践
✅ 开发自主发现的MOF智能体
✅ 集成AutoML进行自动化优化
✅ 为AI辅助材料发现打下坚实基础

---

**祝学习愉快！** 🎓

如果觉得有帮助，请给个 ⭐ Star！

---

## 🔄 更新日志

**v0.5.0** (2024-11)
- ✅ 新增第五天完整课程材料
- ✅ 新增LLM应用理论文档（100+页）
  - LLM演进（ChatGPT → MatGPT → ChemLLM）
  - Text-to-Structure生成方法
  - RAG检索增强生成系统
  - 提示工程技术
  - 智能体架构设计
- ✅ 新增Text-to-Structure生成工具 (text_to_structure.py)
  - 三种生成方法（retrieval/generation/hybrid）
  - 结构验证与排序
  - 易于集成真实LLM API
- ✅ 新增RAG系统实现 (rag_system.py)
  - 向量数据库与语义检索
  - 文档分块与嵌入
  - 上下文增强生成
  - 知识库管理
- ✅ 新增智能体系统 (mof_agent_system.py)
  - 文献挖掘与知识抽取
  - AutoML自动优化
  - 自主发现循环
- ✅ 新增提示工程工具包 (prompt_engineering.py)
  - 预定义模板库
  - Few-shot学习管理
  - Chain-of-Thought提示
  - 自动提示优化
- ✅ 新增第五天实操教程 (day5_tutorial.ipynb)
  - 7个完整部分
  - 可视化和练习
- ✅ 新增第五天课程总结 (DAY5_SUMMARY.md)

**v0.4.1** (2024-11)
- ✅ 增强第四天课程材料
- ✅ 新增第四天实操教程 (day4_tutorial.ipynb)
  - 7个完整部分的Jupyter教程
  - VAE训练与潜在空间探索
  - 扩散模型采样
  - 贝叶斯优化逆向设计
  - 端到端设计流程
- ✅ 新增生成模型可视化工具 (visualization.py)
  - 潜在空间可视化（PCA/t-SNE）
  - 扩散过程可视化
  - 优化历史可视化
  - 训练曲线可视化
- ✅ 新增生成MOF评估工具 (evaluator.py)
  - 有效性评估（结构完整性）
  - 唯一性评估（去重）
  - 多样性评估（覆盖范围）
  - 新颖性评估（与训练集差异）
  - 性质分布评估（统计检验）
- ✅ 更新README包含新工具的完整文档

**v0.4.0** (2024-11)
- ✅ 新增第四天完整课程材料
- ✅ 新增生成模型理论文档（80+页）
  - VAE理论与ELBO推导
  - 扩散模型（DDPM）详解
  - 贝叶斯优化与逆向设计
  - 可合成性评估
- ✅ 新增MOF-VAE模型实现 (mof_vae.py)
  - GNN编码器/解码器
  - 重参数化技巧
  - 潜在空间操作
- ✅ 新增MOF扩散模型 (mof_diffusion.py)
  - DDPM算法
  - 噪声调度策略
  - 采样生成
- ✅ 新增贝叶斯优化工具 (bayesian_optimizer.py)
  - 高斯过程
  - 采集函数（EI/UCB/PI）
  - MOF逆向设计器
- ✅ 新增第四天课程总结 (DAY4_SUMMARY.md)

**v0.3.0** (2024-11)
- ✅ 新增第三天完整课程材料
- ✅ 新增图神经网络理论文档（70+页）
  - GNN基础、消息传递机制
  - CGCNN, MEGNet, SchNet, ALIGNN, DimeNet详解
  - MOF图表示方法
  - 案例研究和性能对比
- ✅ 新增MOF图构建工具 (graph_builder.py)
  - 自动构建邻接矩阵
  - Gaussian距离扩展
  - 周期性边界条件处理
- ✅ 新增GNN模型实现
  - CGCNN（门控消息传递）
  - SimpleMEGNet（三级图）
- ✅ 新增第三天课程总结 (DAY3_SUMMARY.md)
- ✅ 更新依赖：PyTorch Geometric, DGL

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
