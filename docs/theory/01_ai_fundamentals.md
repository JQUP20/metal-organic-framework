# 人工智能基础理论

## 1. 人工智能的科学革命：从符号主义到深度学习的演进路径

### 1.1 符号主义时代 (1950s-1980s)
- **核心思想**：智能可以通过符号操作和逻辑推理实现
- **代表成果**：
  - 专家系统 (MYCIN, DENDRAL)
  - 知识图谱
  - 基于规则的推理系统
- **局限性**：
  - 知识获取瓶颈
  - 难以处理不确定性
  - 缺乏学习能力

### 1.2 连接主义时代 (1980s-2010s)
- **核心思想**：通过神经网络模拟大脑学习
- **里程碑**：
  - 1986: 反向传播算法
  - 1998: LeNet (卷积神经网络)
  - 2006: 深度学习概念提出 (Hinton)
- **突破**：
  - 端到端学习
  - 特征自动提取
  - 非线性建模能力

### 1.3 深度学习革命 (2012-至今)
- **ImageNet 时刻 (2012)**：
  - AlexNet 将图像识别错误率从 26% 降至 15%
  - GPU 加速训练成为可能
- **后续发展**：
  - 2014: GAN (生成对抗网络)
  - 2017: Transformer (注意力机制)
  - 2018: BERT, GPT (大语言模型)
  - 2020s: ChatGPT, Diffusion Models
- **影响**：
  - 计算机视觉接近人类水平
  - 自然语言处理质的飞跃
  - 科学研究范式变革 (AI4Science)

---

## 2. AI 基本理论框架

### 2.1 监督学习 (Supervised Learning)
**定义**：从标注数据中学习输入到输出的映射关系

**核心组件**：
- **训练集**：{(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- **损失函数**：L(ŷ, y) 衡量预测与真实值的差异
- **优化目标**：min θ Σ L(f(x; θ), y)

**典型任务**：
- 分类 (Classification)
  - 二分类：垃圾邮件检测
  - 多分类：图像识别 (猫/狗/鸟...)
  - 多标签：文本标注
- 回归 (Regression)
  - 房价预测
  - MOF 性能预测 (吸附量、带隙等)

**常用算法**：
```
线性模型：线性回归、逻辑回归
树模型：决策树、随机森林、XGBoost
神经网络：MLP, CNN, RNN, Transformer
```

**在 MOF 研究中的应用**：
- 预测 MOF 的 CO₂ 吸附容量
- 分类 MOF 的拓扑结构
- 预测合成成功率

---

### 2.2 无监督学习 (Unsupervised Learning)
**定义**：从无标注数据中发现隐藏的模式和结构

**主要任务**：
1. **聚类 (Clustering)**
   - K-means, DBSCAN, 层次聚类
   - 应用：MOF 结构家族自动分类

2. **降维 (Dimensionality Reduction)**
   - PCA (主成分分析)
   - t-SNE, UMAP (流形学习)
   - 应用：高维 MOF 特征可视化

3. **异常检测 (Anomaly Detection)**
   - 孤立森林、One-Class SVM
   - 应用：识别不稳定的 MOF 结构

**自编码器 (Autoencoder)**：
```
输入 → 编码器 → 潜在表示 → 解码器 → 重构输出
```
- 变分自编码器 (VAE)：用于 MOF 分子生成

---

### 2.3 强化学习 (Reinforcement Learning)
**定义**：智能体通过与环境交互，学习最优策略

**核心概念**：
- **状态 (State)**：环境的当前描述
- **动作 (Action)**：智能体的选择
- **奖励 (Reward)**：行为的即时反馈
- **策略 (Policy)**：π(a|s) 状态到动作的映射
- **价值函数 (Value Function)**：V(s) 未来累积奖励的期望

**经典算法**：
- Q-learning, DQN (深度 Q 网络)
- Policy Gradient, PPO (近端策略优化)
- AlphaGo, AlphaFold 背后的技术

**在 MOF 研究中的应用**：
- 自动化合成路线规划
- 配体-金属组合优化
- 反应条件智能调控

---

### 2.4 生成模型 (Generative Models)
**定义**：学习数据分布，生成新样本

**主要类型**：

1. **生成对抗网络 (GAN)**
   ```
   生成器 G：噪声 z → 假样本 x'
   判别器 D：区分真假样本
   目标：min_G max_D V(D, G)
   ```
   - 应用：MOF 结构生成、逆向设计

2. **变分自编码器 (VAE)**
   - 学习连续潜在空间
   - 应用：MOF 配体设计

3. **扩散模型 (Diffusion Models)**
   - 最新技术，生成质量最高
   - 应用：3D MOF 晶体结构生成

4. **大语言模型 (LLM)**
   - GPT, BERT 架构
   - 应用：化学文献挖掘、SMILES 生成

---

## 3. 机器学习典型流程

### 3.1 整体流程图
```
数据收集 → 数据预处理 → 特征工程 → 模型训练 → 验证评估 → 模型解释 → 部署应用
   ↑                                                                       ↓
   └─────────────────────────── 迭代优化 ───────────────────────────────┘
```

### 3.2 数据预处理
**缺失值处理**：
- 删除：样本或特征缺失率过高
- 填充：均值/中位数/众数
- 插值：KNN、回归预测

**异常值处理**：
- 检测：3σ 原则、IQR 方法、孤立森林
- 处理：删除、Winsorization、对数变换

**数据标准化**：
```python
# 标准化 (Z-score)
X_std = (X - μ) / σ

# 归一化 (Min-Max)
X_norm = (X - X_min) / (X_max - X_min)
```

**不平衡数据**：
- 过采样：SMOTE
- 欠采样：Tomek Links
- 加权：class_weight 参数

---

### 3.3 特征工程
**MOF 特征示例**：
```
几何特征：孔径、比表面积、孔隙率
化学特征：金属种类、配体官能团、电负性
拓扑特征：连接度、环数、Voronoi 分析
能量特征：结合能、HOMO-LUMO 能隙
```

**特征选择方法**：
1. **过滤法 (Filter)**：
   - 相关系数、卡方检验
   - 互信息

2. **包装法 (Wrapper)**：
   - 递归特征消除 (RFE)
   - 遗传算法

3. **嵌入法 (Embedded)**：
   - Lasso (L1 正则化)
   - 树模型的特征重要性

**特征变换**：
- 多项式特征
- 交互特征
- 领域知识引导的特征构造

---

### 3.4 模型训练
**数据集划分**：
```
训练集 (60-80%)：学习参数
验证集 (10-20%)：调整超参数
测试集 (10-20%)：最终评估
```

**交叉验证**：
```
K-Fold CV：将数据分为 K 份，轮流作为验证集
Stratified K-Fold：保持类别比例
Leave-One-Out：n 个样本 → n 次验证 (小数据集)
```

**超参数优化**：
- 网格搜索 (Grid Search)
- 随机搜索 (Random Search)
- 贝叶斯优化 (Bayesian Optimization)
- AutoML：Auto-sklearn, TPOT

---

### 3.5 验证与评估
**分类任务指标**：
```
准确率 (Accuracy) = (TP + TN) / (TP + TN + FP + FN)
精确率 (Precision) = TP / (TP + FP)
召回率 (Recall) = TP / (TP + FN)
F1-score = 2 × (Precision × Recall) / (Precision + Recall)
AUC-ROC：受试者工作特征曲线下面积
```

**回归任务指标**：
```
MAE (平均绝对误差) = Σ|yᵢ - ŷᵢ| / n
RMSE (均方根误差) = √(Σ(yᵢ - ŷᵢ)² / n)
R² (决定系数) = 1 - SS_res / SS_tot
```

**模型对比**：
- 基线模型：简单规则、随机猜测
- 多模型对比：选择最优架构
- 显著性检验：t-test, Wilcoxon

---

### 3.6 模型解释
**为什么需要解释性？**
- 科学发现：理解物理化学规律
- 建立信任：验证模型合理性
- 调试改进：发现问题所在

**解释方法**：
1. **全局解释**：
   - 特征重要性 (Feature Importance)
   - 部分依赖图 (Partial Dependence Plot)

2. **局部解释**：
   - LIME (局部可解释模型)
   - SHAP (Shapley Additive Explanations)

3. **注意力机制**：
   - Transformer 的 Attention Weights
   - 可视化模型关注的结构片段

---

## 4. 深度学习简介

### 4.1 神经网络结构
**感知机 (Perceptron)**：
```
y = σ(Σ wᵢxᵢ + b)
```
- σ：激活函数 (sigmoid, ReLU, tanh)
- w：权重，b：偏置

**多层感知机 (MLP)**：
```
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
```

**常用激活函数**：
```python
ReLU: f(x) = max(0, x)           # 最常用
Sigmoid: f(x) = 1/(1+e^(-x))     # 输出 0-1
Tanh: f(x) = (e^x-e^(-x))/(e^x+e^(-x))  # 输出 -1 到 1
Leaky ReLU: f(x) = max(0.01x, x) # 避免神经元死亡
```

---

### 4.2 反向传播算法 (Backpropagation)
**核心思想**：链式法则计算梯度

**前向传播**：
```
输入 → 计算每层输出 → 得到预测值 → 计算损失
```

**反向传播**：
```
损失 → 计算输出层梯度 → 逐层反向传播梯度 → 更新参数
```

**梯度下降更新**：
```
θ_new = θ_old - η · ∇L(θ)
```
- η：学习率 (learning rate)
- ∇L：损失函数梯度

**优化算法**：
- SGD (随机梯度下降)
- Momentum：考虑历史梯度
- Adam：自适应学习率 (最常用)
- AdamW：带权重衰减的 Adam

---

### 4.3 过拟合与泛化能力
**过拟合 (Overfitting)**：
- 现象：训练集表现好，测试集表现差
- 原因：模型过于复杂，记住了噪声

**欠拟合 (Underfitting)**：
- 现象：训练集和测试集都表现差
- 原因：模型过于简单，无法捕捉数据规律

**Bias-Variance Tradeoff**：
```
总误差 = Bias² + Variance + 不可约误差
```
- 高偏差：欠拟合
- 高方差：过拟合

**正则化技术**：
1. **L1/L2 正则化**：
   ```
   L_total = L_data + λ Σ|w|      # L1: Lasso
   L_total = L_data + λ Σw²       # L2: Ridge
   ```

2. **Dropout**：
   - 训练时随机失活神经元
   - 类似集成学习效果

3. **Early Stopping**：
   - 监控验证集误差
   - 停止于最优点

4. **数据增强 (Data Augmentation)**：
   - 图像：旋转、翻转、裁剪
   - MOF：结构扰动、配体替换

5. **Batch Normalization**：
   - 标准化每层输入
   - 加速训练，提升泛化

---

### 4.4 常见神经网络架构
**卷积神经网络 (CNN)**：
- 用途：图像、晶体结构
- 核心：卷积层、池化层
- 应用：MOF 晶体结构分类

**循环神经网络 (RNN/LSTM/GRU)**：
- 用途：序列数据
- 核心：隐藏状态记忆
- 应用：SMILES 序列生成

**图神经网络 (GNN)**：
- 用途：分子、晶体图
- 核心：消息传递机制
- 应用：MOF 性能预测 (SOTA)
- 典型模型：
  - GCN (图卷积网络)
  - GAT (图注意力网络)
  - SchNet, DimeNet (3D 几何信息)

**Transformer**：
- 用途：通用架构
- 核心：自注意力机制
- 应用：
  - 化学文献理解
  - 分子性质预测 (MolFormer)
  - 晶体生成 (Crystal Diffusion)

---

## 5. AI 在材料科学中的应用范式

### 5.1 正向预测 (Forward Prediction)
```
结构 → AI 模型 → 性能
```
- 给定 MOF 结构，预测吸附、催化性能
- 快速筛选候选材料

### 5.2 逆向设计 (Inverse Design)
```
目标性能 → AI 模型 → 生成结构
```
- 指定性能要求，生成满足条件的 MOF
- 生成模型 (GAN, VAE, Diffusion)

### 5.3 主动学习 (Active Learning)
```
初始模型 → 预测 → 选择最有信息量的样本 → 实验验证 → 更新模型 → 循环
```
- 减少实验次数
- 高效探索化学空间

### 5.4 多任务学习 (Multi-task Learning)
```
共享特征提取器 → 任务1 (吸附)
                  → 任务2 (稳定性)
                  → 任务3 (合成难度)
```
- 同时预测多个性能
- 迁移学习提升小数据性能

---

## 参考文献
1. Goodfellow et al. (2016). *Deep Learning*. MIT Press.
2. Murphy (2022). *Probabilistic Machine Learning: An Introduction*.
3. Butler et al. (2018). Machine learning for molecular and materials science. *Nature*, 559, 547-555.
4. Schütt et al. (2021). Quantum-chemical insights from interpretable atomistic neural networks. *Nature Communications*.

---

## 课后思考题
1. 监督学习与无监督学习在 MOF 研究中各有什么优势场景？
2. 为什么深度学习相比传统机器学习更适合处理 MOF 晶体结构数据？
3. 如何判断一个 MOF 性能预测模型是否过拟合？
4. 图神经网络如何表示 MOF 的拓扑结构？

---

**下一讲预告**：MOF 材料基础知识与数据库资源
