# 第三天理论：图神经网络(GNN)与 MOF 结构-性能建模

## 目录

1. [为什么需要图神经网络？](#1-为什么需要图神经网络)
2. [图的基本概念](#2-图的基本概念)
3. [MOF 的图表示](#3-mof-的图表示)
4. [图神经网络基础](#4-图神经网络基础)
5. [消息传递神经网络](#5-消息传递神经网络)
6. [主流 GNN 模型](#6-主流-gnn-模型)
7. [GNN 在 MOF 中的应用](#7-gnn-在-mof-中的应用)
8. [模型训练与优化](#8-模型训练与优化)
9. [案例研究](#9-案例研究)

---

## 1. 为什么需要图神经网络？

### 1.1 传统方法的局限

**第二天的方法（特征工程 + 机器学习）：**

```
MOF 结构 → [手工特征提取] → [特征向量] → [ML模型] → 性质预测
           ↑
    需要领域知识
    - 比表面积
    - 孔体积
    - 孔径
    - 元素组成
    ...
```

**局限性：**
- ❌ 依赖人工特征设计
- ❌ 可能遗漏重要的结构信息
- ❌ 无法捕捉原子间的局部相互作用
- ❌ 忽略了 3D 空间结构
- ❌ 特征计算可能耗时（如 Zeo++）

### 1.2 GNN 的优势

**图神经网络方法：**

```
MOF 结构 → [自动图构建] → [GNN 学习] → 性质预测
                           ↑
                    端到端学习
                    - 原子类型
                    - 键类型
                    - 空间位置
                    - 自动学习特征
```

**优势：**
- ✅ **端到端学习**：直接从结构学习，无需手工特征
- ✅ **排列不变性**：原子顺序不影响结果
- ✅ **局部性**：捕捉原子间相互作用
- ✅ **可迁移性**：学到的表示可用于不同任务
- ✅ **可解释性**：通过注意力机制理解模型关注点

### 1.3 对比示例

**任务：预测 MOF 的 CO₂ 吸附能**

| 方法 | 输入 | 特征数 | 性能 | 优点 | 缺点 |
|-----|------|-------|------|------|------|
| **XGBoost + 手工特征** | 几何/化学特征向量 | 15-50 | R² ≈ 0.85 | 训练快、可解释 | 需要特征工程 |
| **GNN (CGCNN)** | 原子坐标 + 晶格 | 自动学习 | R² ≈ 0.92 | 端到端、泛化好 | 训练慢、需GPU |
| **GNN (MEGNet)** | 原子坐标 + 键信息 | 自动学习 | R² ≈ 0.94 | 捕捉远程交互 | 数据需求大 |

---

## 2. 图的基本概念

### 2.1 什么是图？

图 G = (V, E) 由节点和边组成：

```
    节点（Vertices/Nodes）
         ○
        / \
       /   \
      ○─────○  边（Edges）
```

**数学定义：**

```
G = (V, E, X, A)

V = {v₁, v₂, ..., vₙ}      # 节点集合
E = {e₁, e₂, ..., eₘ}      # 边集合
X ∈ ℝⁿˣᵈ                    # 节点特征矩阵 (n个节点, d维特征)
A ∈ {0,1}ⁿˣⁿ                # 邻接矩阵
```

**邻接矩阵 A：**

```
     v₁  v₂  v₃
v₁ [ 0   1   1 ]    # v₁ 连接 v₂, v₃
v₂ [ 1   0   1 ]    # v₂ 连接 v₁, v₃
v₃ [ 1   1   0 ]    # v₃ 连接 v₁, v₂
```

### 2.2 图的类型

**1. 无向图 vs 有向图**

```
无向图:              有向图:
  ○───○              ○→○
   \ /                ↓
    ○                 ○
```

**2. 同质图 vs 异质图**

```
同质图 (所有节点类型相同):
  C─C─C

异质图 (节点有不同类型):
  C─O─Zn  (碳、氧、锌)
```

**3. 静态图 vs 动态图**

```
静态图: 结构固定
动态图: 结构随时间变化（本课程不涉及）
```

### 2.3 图的性质

**排列不变性 (Permutation Invariance)：**

```
图的表示不依赖于节点的顺序

原子序列 [C, H, O, N] 和 [H, C, N, O]
→ 应该产生相同的图表示
```

**局部性 (Locality)：**

```
节点的表示主要受邻居节点影响

  ○ ← 远程节点（影响小）
  │
  ○ ← 邻居（影响大）
  │
 [v] ← 目标节点
```

---

## 3. MOF 的图表示

### 3.1 将 MOF 转换为图

**MOF 晶体结构 → 图：**

```
CIF 文件:
  - 晶格参数 (a, b, c, α, β, γ)
  - 原子坐标
  - 空间群

        ↓ 转换

图表示:
  - 节点 = 原子
  - 边 = 化学键 / 空间邻近
  - 节点特征 = 原子属性
  - 边特征 = 键属性
```

**示例：简单 MOF 单元**

```
原子结构:
    O
   / \
  Zn  C
   \ /
    O

图表示:
节点: V = {Zn, C, O₁, O₂}
边:   E = {(Zn,O₁), (Zn,O₂), (C,O₁), (C,O₂)}
```

### 3.2 节点特征设计

**常用原子（节点）特征：**

| 特征类型 | 具体特征 | 维度 | 示例值 (C) |
|---------|---------|------|-----------|
| **基本属性** | 原子序数 | 1 | 6 |
| | 原子质量 | 1 | 12.01 |
| | 元素族 | 1 | 14 (IV A) |
| | 元素周期 | 1 | 2 |
| **电子性质** | 价电子数 | 1 | 4 |
| | 电负性 | 1 | 2.55 |
| | 第一电离能 | 1 | 11.26 eV |
| **几何性质** | 共价半径 | 1 | 0.77 Å |
| | 范德华半径 | 1 | 1.70 Å |
| **配位信息** | 配位数 | 1 | 4 |
| | 氧化态 | 1 | +4 |
| **One-Hot 编码** | 元素类型 | ~100 | [0,0,0,0,0,1,0,...] |

**特征向量示例（简化版）：**

```python
# 碳原子的特征向量
carbon_features = [
    6.0,        # 原子序数
    12.01,      # 原子质量
    2.55,       # 电负性
    4,          # 配位数
    0.77,       # 共价半径
    # ... 更多特征
]

# 或使用 One-Hot 编码
carbon_onehot = [0, 0, 0, 0, 0, 1, 0, 0, ..., 0]  # 第6位是1
```

### 3.3 边特征设计

**常用键（边）特征：**

| 特征类型 | 具体特征 | 说明 |
|---------|---------|------|
| **几何** | 键长 (Bond Length) | 原子间距离 |
| | 键角 (Bond Angle) | 三原子夹角 |
| | 方向向量 | (Δx, Δy, Δz) |
| **化学** | 键类型 | 单键/双键/配位键 |
| | 键级 (Bond Order) | 1, 2, 3, 1.5... |
| **扩展** | Gaussian 距离 | RBF(distance) |
| | 周期性 | 是否跨晶胞边界 |

**边定义策略：**

**策略 1：距离截断 (Radius Cutoff)**

```python
# 距离 < 阈值 → 创建边
cutoff_radius = 5.0  # Å

for atom_i, atom_j in all_pairs:
    distance = ||r_i - r_j||
    if distance < cutoff_radius:
        create_edge(i, j, features=[distance])
```

**策略 2：K-最近邻 (K-Nearest Neighbors)**

```python
# 每个原子连接最近的 k 个邻居
k = 12  # 配位数

for atom_i in atoms:
    neighbors = find_k_nearest(atom_i, k)
    for neighbor in neighbors:
        create_edge(atom_i, neighbor)
```

**策略 3：化学键识别**

```python
# 基于共价半径和
for atom_i, atom_j in all_pairs:
    distance = ||r_i - r_j||
    covalent_sum = r_cov(i) + r_cov(j)

    if distance < covalent_sum * 1.2:  # 容差
        create_edge(i, j)
```

### 3.4 晶体图的特殊性

**周期性边界条件：**

```
原始晶胞:        扩展到相邻晶胞:
┌─────┐         ┌─────┬─────┬─────┐
│  ●  │         │  ●  │  ●  │  ●  │
│     │    →    │  ╲  │  │  │  ╱  │
│  ●  │         │   ● │  ●  │ ●   │
└─────┘         └─────┴─────┴─────┘

中心原子可能与相邻晶胞中的原子成键
```

**处理方法：**

```python
# 创建超胞 (Supercell)
supercell = structure.repeat([2, 2, 2])

# 或使用周期性距离计算
def periodic_distance(pos1, pos2, cell_matrix):
    delta = pos2 - pos1
    # 映射到 [-0.5, 0.5] 范围
    delta_frac = np.dot(delta, np.linalg.inv(cell_matrix))
    delta_frac -= np.round(delta_frac)
    delta_cart = np.dot(delta_frac, cell_matrix)
    return np.linalg.norm(delta_cart)
```

---

## 4. 图神经网络基础

### 4.1 核心思想

**传统神经网络 vs 图神经网络：**

```
传统 CNN (图像):
输入: 固定大小的网格 (H × W × C)
操作: 卷积核滑动

输入 → [卷积] → [池化] → ... → 输出
       固定邻居


GNN (图):
输入: 可变大小的图 (N 个节点, M 条边)
操作: 消息传递

节点 → [聚合邻居] → [更新节点] → ... → 输出
       邻居数可变
```

### 4.2 GNN 的任务类型

**1. 节点级任务 (Node-Level)**

```
预测每个节点的属性

例子: 预测每个原子的电荷、磁矩
```

**2. 边级任务 (Edge-Level)**

```
预测边的属性

例子: 预测键能、键类型
```

**3. 图级任务 (Graph-Level)** ← **MOF 性质预测主要用这个**

```
预测整个图的属性

例子: 预测 MOF 的吸附能、带隙
```

### 4.3 GNN 的基本操作

**三个核心步骤：**

```
1. 消息生成 (Message)
   m_ij = φ(h_i, h_j, e_ij)

2. 消息聚合 (Aggregate)
   m_i = ⊕_{j∈N(i)} m_ij

3. 节点更新 (Update)
   h_i^{new} = ψ(h_i^{old}, m_i)
```

**可视化：**

```
时间步 t:

     h_j^t  h_k^t  h_l^t    ← 邻居节点状态
       ↓      ↓      ↓
     [消息生成 φ]
       ↓      ↓      ↓
     m_ij   m_ik   m_il    ← 消息
       ↓      ↓      ↓
        [聚合 ⊕]
            ↓
          m_i              ← 聚合消息
            ↓
      [更新 ψ]
            ↓
         h_i^{t+1}         ← 新节点状态
```

**聚合函数 ⊕ 的选择：**

```python
# 求和（保留邻居数量信息）
m_i = Σ m_ij

# 平均（归一化）
m_i = (1/|N(i)|) Σ m_ij

# 最大值（选择最重要的邻居）
m_i = max{m_ij}

# 注意力加权（学习邻居重要性）
m_i = Σ α_ij * m_ij
```

---

## 5. 消息传递神经网络 (MPNN)

### 5.1 MPNN 框架

**通用消息传递公式：**

```
消息传递层 (第 t 层):

m_i^{t+1} = Σ_{j∈N(i)} M_t(h_i^t, h_j^t, e_ij)

h_i^{t+1} = U_t(h_i^t, m_i^{t+1})

其中:
- M_t: 消息函数（通常是神经网络）
- U_t: 更新函数（通常是神经网络）
- h_i^t: 节点 i 在第 t 层的隐藏状态
- e_ij: 边特征
```

**读出函数 (Readout)：**

对于图级预测，需要聚合所有节点信息：

```
h_graph = R({h_i^T | i ∈ V})

常用方法:
- 求和: h_graph = Σ h_i
- 平均: h_graph = (1/N) Σ h_i
- 注意力池化: h_graph = Σ α_i * h_i
- Set2Set: 使用 LSTM 聚合
```

### 5.2 MPNN 的表达能力

**Weisfeiler-Lehman (WL) 测试：**

GNN 的表达能力与 WL 图同构测试相当。

**局限性：**
- 无法区分某些非同构图
- 难以捕捉长程依赖（需要多层）

**解决方案：**
- 更高阶的 GNN (k-GNN)
- 加入位置编码
- 使用图的全局特征

---

## 6. 主流 GNN 模型

### 6.1 CGCNN (Crystal Graph Convolutional Neural Network)

**文献：**
> Xie & Grossman, *Physical Review Letters*, 2018

**核心思想：**

专为晶体材料设计的 GNN，考虑周期性边界条件。

**架构：**

```
输入: 原子坐标 + 晶格参数

图构建:
- 节点: 原子
- 边: 距离 < 8 Å 的原子对
- 节点特征: 元素 One-Hot 编码
- 边特征: Gaussian expansion of distance

卷积层 (多层):
  z_i^{t+1} = h_i^{t} + Σ_{j∈N(i)} σ(z_j^{t} ⊙ g(z_j^{t}, r_ij))

  其中:
  - ⊙: element-wise 乘法
  - g: 门控函数
  - r_ij: 边特征（距离）

池化:
  h_graph = (1/N) Σ h_i^{final}

输出:
  property = MLP(h_graph)
```

**Gaussian 距离扩展：**

```python
# 将距离映射到高维
def gaussian_expansion(distance, centers, width):
    """
    centers = [0, 0.5, 1.0, 1.5, ..., 8.0]  # 均匀分布
    width = 0.5
    """
    expanded = exp(-((distance - centers)^2) / (2 * width^2))
    return expanded  # shape: (len(centers),)
```

**优点：**
- ✅ 专为晶体设计，处理周期性
- ✅ 门控机制，选择性传递信息
- ✅ 解释性好

**缺点：**
- ❌ 边特征仅使用距离（忽略键角等）
- ❌ 池化使用简单平均（信息损失）

### 6.2 MEGNet (MatErials Graph Network)

**文献：**
> Chen et al., *Chemistry of Materials*, 2019

**核心思想：**

三级图表示：原子 + 键 + 全局状态

**架构：**

```
图表示:
┌─────────────────────────────────┐
│  Atom (v) ←→ Bond (e) ←→ State (u)  │
│                                 │
│  - 原子特征    - 键特征    - 全局特征 │
│  - 坐标       - 距离       - 温度    │
│  - 元素       - 键角       - 压力    │
└─────────────────────────────────┘

MEGNet 块 (重复 N 次):

1. 边更新:
   e_ij' = φ_e(e_ij, v_i, v_j, u)

2. 节点更新:
   v_i' = φ_v(v_i, Σ e_ij', u)

3. 全局状态更新:
   u' = φ_u(u, Σ v_i', Σ e_ij')

输出:
  property = φ_out(u^{final})
```

**三级更新示例：**

```python
class MEGNetBlock(nn.Module):
    def forward(self, v, e, u, edge_index):
        # 1. Edge update
        v_i = v[edge_index[0]]  # 起始节点
        v_j = v[edge_index[1]]  # 结束节点
        e_new = self.edge_update(torch.cat([e, v_i, v_j, u], dim=-1))

        # 2. Node update
        e_agg = aggregate_edges(e_new, edge_index[0])  # 聚合到节点
        v_new = self.node_update(torch.cat([v, e_agg, u], dim=-1))

        # 3. Global update
        v_total = v_new.mean(dim=0)  # 所有节点平均
        e_total = e_new.mean(dim=0)  # 所有边平均
        u_new = self.global_update(torch.cat([u, v_total, e_total], dim=-1))

        return v_new, e_new, u_new
```

**优点：**
- ✅ 三级图，信息流更丰富
- ✅ 全局状态捕捉整体信息
- ✅ 性能强大，在多个数据集上表现优异

**缺点：**
- ❌ 计算复杂度高
- ❌ 需要更多训练数据

### 6.3 ALIGNN (Atomistic Line Graph Neural Network)

**文献：**
> Choudhary & DeCost, *npj Computational Materials*, 2021

**核心思想：**

使用 **线图 (Line Graph)** 捕捉键角信息。

**线图的概念：**

```
原图 (原子图):        线图 (键图):
    A                   e_AB ── e_AC
   / \                    │   ╱
  B   C                  e_BC

原图的边 → 线图的节点
原图的键角 → 线图的边
```

**架构：**

```
输入: 原子坐标

构建两个图:
1. 原子图 (Atom Graph)
   - 节点: 原子
   - 边: 键

2. 键图 (Bond Graph / Line Graph)
   - 节点: 键
   - 边: 键角

ALIGNN 层:
1. 更新键图节点（键）:
   e_ij' = UPDATE_BOND(e_ij, neighbors_in_line_graph)

2. 更新原子图节点（原子）:
   v_i' = UPDATE_ATOM(v_i, Σ e_ij')

池化:
  h_graph = Σ v_i^{final}

输出:
  property = MLP(h_graph)
```

**键角的重要性：**

```
CO₂ 分子:
O=C=O  (线性, 180°)  vs  O-C-O  (弯曲, 120°)

键角决定分子性质！
```

**优点：**
- ✅ 显式建模键角（三体相互作用）
- ✅ SOTA 性能（多个基准测试）
- ✅ JARVIS-DFT 数据集上表现最佳

**缺点：**
- ❌ 线图构建复杂
- ❌ 内存需求大（两个图）

### 6.4 SchNet

**文献：**
> Schütt et al., *NeurIPS*, 2017

**核心思想：**

连续滤波卷积 (Continuous-filter Convolution)

**架构：**

```
输入: 原子坐标（3D）

特征:
- 节点: 元素 One-Hot
- 边: 原子间距离

卷积:
  v_i^{t+1} = Σ_{j∈N(i)} v_j^{t} ⊙ W(||r_ij||)

  其中:
  - W: 连续滤波函数（神经网络）
  - ||r_ij||: 距离

输出:
  能量 = Σ E_i  （原子能量求和）
```

**旋转不变性：**

SchNet 仅使用距离（标量），天然满足旋转/平移不变性。

**优点：**
- ✅ 旋转/平移不变
- ✅ 适合分子能量预测
- ✅ 物理意义明确

**缺点：**
- ❌ 仅用距离，忽略角度
- ❌ 晶体周期性处理不如 CGCNN

### 6.5 DimeNet / DimeNet++

**文献：**
> Klicpera et al., *ICLR*, 2020

**核心思想：**

方向性消息传递 (Directional Message Passing)，显式建模键角和二面角。

**架构：**

```
特征:
- 节点: 原子
- 方向性边: 包含键角信息

消息传递:
  m_ij = f(v_i, v_j, d_ij, ∠(i,j,k))

  其中:
  - ∠(i,j,k): 键角

二面角信息:
  进一步考虑 四原子 i-j-k-l 的扭转角
```

**优点：**
- ✅ 方向性信息丰富
- ✅ 小分子性质预测SOTA

**缺点：**
- ❌ 计算成本极高
- ❌ 对大系统（如 MOF）较慢

### 6.6 模型对比总结

| 模型 | 年份 | 核心特点 | 节点特征 | 边特征 | 适用场景 | 性能 |
|-----|------|---------|---------|--------|---------|------|
| **CGCNN** | 2018 | 晶体专用 | One-Hot | Gaussian(距离) | 晶体材料 | ⭐⭐⭐ |
| **MEGNet** | 2019 | 三级图 | 原子属性 | 距离+键角 | 通用材料 | ⭐⭐⭐⭐ |
| **SchNet** | 2017 | 连续滤波 | One-Hot | 距离 | 分子 | ⭐⭐⭐ |
| **ALIGNN** | 2021 | 线图+键角 | 原子属性 | 键+键角 | 晶体+分子 | ⭐⭐⭐⭐⭐ |
| **DimeNet** | 2020 | 方向性 | One-Hot | 距离+角度 | 小分子 | ⭐⭐⭐⭐ |

**推荐：**
- **初学者**：CGCNN（简单、易懂）
- **性能优先**：ALIGNN 或 MEGNet
- **MOF 专用**：CGCNN 或 ALIGNN

---

## 7. GNN 在 MOF 中的应用

### 7.1 应用场景

**1. 性质预测**

```
输入: MOF 结构 (CIF)
输出: 连续值性质

任务:
- 气体吸附量 (CO₂, CH₄, H₂)
- 带隙 (Band Gap)
- 体积模量 (Bulk Modulus)
- 热导率
```

**2. 生成与设计**

```
逆向问题: 给定目标性质 → 生成 MOF 结构
（第四天课程内容）
```

**3. 结构筛选**

```
高通量筛选: 从数百万候选 MOF 中快速筛选
（第五天课程内容）
```

### 7.2 MOF-GNN 工作流

```
┌──────────────────────────────────────────────────┐
│                  MOF 数据集                       │
│         (CIF files + Properties)                 │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│            图构建 (Graph Construction)            │
│  • 读取 CIF → ASE / Pymatgen                     │
│  • 提取原子坐标、晶格参数                         │
│  • 构建邻接矩阵（距离截断/KNN）                   │
│  • 生成节点/边特征                                │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│            数据加载 (DataLoader)                  │
│  • PyTorch Geometric Data / DGL Graph            │
│  • 批处理 (Batching)                             │
│  • 数据增强（可选）                               │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│             GNN 模型训练                          │
│  • 前向传播: graph → GNN → prediction            │
│  • 损失函数: MSE / MAE                           │
│  • 优化器: Adam / AdamW                          │
│  • 学习率调度                                    │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│             模型评估                              │
│  • 测试集性能 (R², RMSE, MAE)                    │
│  • 注意力可视化                                  │
│  • 特征重要性分析                                │
└──────────────────────────────────────────────────┘
```

### 7.3 数据集

**常用 MOF 数据集：**

| 数据集 | 样本数 | 包含性质 | 来源 |
|-------|--------|---------|------|
| **QMOF** | 20,000+ | DFT 计算性质（带隙、磁矩等） | Northwestern |
| **CoRE MOF** | 14,000+ | 几何性质 | 文献收集 |
| **hMOF** | 137,000+ | 吸附等温线（模拟） | NIST |
| **ToBaCCo** | 13,000+ | 合成可行性 | Oregon State |

### 7.4 性能对比

**案例：QMOF 带隙预测**

| 方法 | MAE (eV) | R² | 训练时间 |
|-----|---------|-----|---------|
| Linear Regression + 手工特征 | 0.45 | 0.65 | 1 min |
| Random Forest + 手工特征 | 0.32 | 0.78 | 5 min |
| XGBoost + 手工特征 | 0.28 | 0.82 | 10 min |
| **CGCNN** | 0.24 | 0.86 | 30 min |
| **MEGNet** | 0.21 | 0.89 | 1 hour |
| **ALIGNN** | **0.18** | **0.92** | 2 hours |

**结论：**
- GNN 性能显著优于传统方法
- 训练时间较长，但预测快（毫秒级）
- 需要 GPU 加速

---

## 8. 模型训练与优化

### 8.1 损失函数

**回归任务（MOF 性质预测）：**

```python
# 均方误差（最常用）
loss = MSELoss()
L = (1/N) Σ (y_pred - y_true)²

# 平均绝对误差（对异常值鲁棒）
loss = L1Loss()
L = (1/N) Σ |y_pred - y_true|

# Huber Loss（结合 MSE 和 MAE）
loss = HuberLoss(delta=1.0)
L = { 0.5*(y_pred - y_true)²     if |error| ≤ delta
      delta*(|error| - 0.5*delta) otherwise
```

**多任务学习：**

```python
# 同时预测多个性质
loss = w1*MSE(y1_pred, y1_true) + w2*MSE(y2_pred, y2_true)

# 自动权重调整（Uncertainty Weighting）
loss = Σ (1/(2σ_i²))*MSE_i + log(σ_i)
```

### 8.2 优化器选择

```python
# Adam（最常用）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# AdamW（权重衰减）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# SGD with Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

### 8.3 学习率调度

```python
# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# ReduceLROnPlateau（性能停滞时降低）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# 线性 Warmup + 余弦衰减
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)
```

### 8.4 防止过拟合

**1. Dropout**

```python
class GNNLayer(nn.Module):
    def __init__(self, dropout=0.1):
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)  # 随机丢弃
        return x
```

**2. Early Stopping**

```python
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(epochs):
    train_loss = train()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

**3. 数据增强**

```python
# 旋转结构
rotated_structure = rotate_structure(structure, axis, angle)

# 添加噪声
noisy_positions = positions + np.random.normal(0, 0.01, positions.shape)

# 超胞扩展
supercell = structure.repeat([2, 2, 2])
```

### 8.5 超参数调优

**关键超参数：**

```python
hyperparameters = {
    # 模型架构
    'hidden_dim': [64, 128, 256],
    'num_layers': [3, 4, 5, 6],
    'num_heads': [4, 8],  # 注意力头数

    # 训练
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'dropout': [0.0, 0.1, 0.2],

    # 图构建
    'cutoff_radius': [5.0, 6.0, 8.0],
    'max_neighbors': [12, 20, 30],
}
```

**调优方法：**

```python
# 网格搜索
from sklearn.model_selection import ParameterGrid
for params in ParameterGrid(hyperparameters):
    model = create_model(**params)
    performance = train_and_evaluate(model)

# 贝叶斯优化
from optuna import create_study
def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    # ...
    return validation_loss

study = create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

## 9. 案例研究

### 9.1 案例 1：QMOF 带隙预测

**数据集：**
- 18,000+ MOF 结构
- DFT 计算的带隙 (0-6 eV)

**方法：CGCNN**

**结果：**

| 指标 | 值 |
|-----|---|
| R² | 0.86 |
| MAE | 0.24 eV |
| RMSE | 0.35 eV |

**关键发现：**
- 金属中心类型是最重要因素
- 配体共轭程度显著影响带隙
- GNN 捕捉到局部配位环境的影响

### 9.2 案例 2：CO₂ 吸附预测

**数据集：**
- CoRE MOF 2019
- GCMC 模拟的 CO₂ 吸附量（298K, 1bar）

**方法：MEGNet**

**结果：**

| 模型 | R² | MAE (mmol/g) |
|-----|-----|-------------|
| RF + 手工特征 | 0.82 | 0.45 |
| XGBoost + 手工特征 | 0.85 | 0.38 |
| **MEGNet** | **0.91** | **0.28** |

**可解释性分析：**

使用注意力机制发现：
- 开放金属位点（OMS）获得高注意力权重
- 极性官能团（-NH₂, -OH）贡献显著
- 孔径在 6-12 Å 的区域最重要

### 9.3 案例 3：高通量筛选

**任务：**
从 hMOF 数据库（137,000 个 MOF）中筛选 H₂ 存储性能优异的 MOF。

**流程：**

```
1. 使用 ALIGNN 预测所有 MOF 的 H₂ 吸附量
   - 推理速度: ~100 structures/second (GPU)
   - 总耗时: ~20 分钟

2. 排序并选择 Top 1000

3. 对 Top 1000 进行高精度 GCMC 模拟验证

4. 识别出 50 个高性能 MOF
```

**加速效果：**
- 传统方法：全部 GCMC 模拟需要 ~1 年（137,000 × 2 小时）
- GNN 筛选：20 分钟 + 2000 小时（Top 1000 验证）≈ 3 个月

**加速比：~4倍**

---

## 总结

### 关键要点

1. **GNN 的优势**：
   - 端到端学习，无需手工特征
   - 自然处理可变大小图
   - 捕捉局部相互作用

2. **主流模型**：
   - **CGCNN**：晶体专用，简单有效
   - **MEGNet**：三级图，性能强
   - **ALIGNN**：线图+键角，SOTA

3. **应用要点**：
   - 图构建：选择合适的截断半径/邻居数
   - 特征设计：原子属性 + 距离/角度
   - 训练技巧：学习率调度、Early Stopping

4. **挑战**：
   - 数据需求：通常需要 > 5000 样本
   - 计算成本：需要 GPU
   - 可解释性：黑盒模型

### 下一步

- **实操练习**：完成 [第三天实操教程](../../notebooks/day3_tutorial.ipynb)
- **深入学习**：阅读 CGCNN、MEGNet、ALIGNN 原始论文
- **项目实践**：在自己的 MOF 数据集上训练 GNN

---

## 参考资料

**核心论文：**

1. **CGCNN**
   Xie, T. & Grossman, J. C. *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.* Physical Review Letters (2018).

2. **MEGNet**
   Chen, C. et al. *Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals.* Chemistry of Materials (2019).

3. **SchNet**
   Schütt, K. T. et al. *SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions.* NeurIPS (2017).

4. **ALIGNN**
   Choudhary, K. & DeCost, B. *Atomistic Line Graph Neural Network for Improved Materials Property Predictions.* npj Computational Materials (2021).

5. **DimeNet**
   Klicpera, J. et al. *Directional Message Passing for Molecular Graphs.* ICLR (2020).

**综述文章：**

6. Reiser, P. et al. *Graph Neural Networks for Materials Science and Chemistry.* Communications Materials (2022).

7. Choudhary, K. et al. *Recent Advances and Applications of Deep Learning Methods in Materials Science.* npj Computational Materials (2022).

**在线资源：**

- PyTorch Geometric 教程: https://pytorch-geometric.readthedocs.io/
- DGL 教程: https://www.dgl.ai/
- JARVIS-Tools: https://jarvis-tools.readthedocs.io/ (包含 ALIGNN 实现)
- MatBench: https://matbench.materialsproject.org/ (基准测试)

---

*第三天理论结束 - 接下来进入实操环节*
