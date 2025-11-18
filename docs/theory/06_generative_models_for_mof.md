# 第四天理论：生成模型与 MOF 逆向设计

## 目录

1. [为什么需要生成模型？](#1-为什么需要生成模型)
2. [生成模型基础](#2-生成模型基础)
3. [变分自编码器 (VAE)](#3-变分自编码器-vae)
4. [生成对抗网络 (GAN)](#4-生成对抗网络-gan)
5. [扩散模型 (Diffusion Models)](#5-扩散模型-diffusion-models)
6. [MOF 生成模型](#6-mof-生成模型)
7. [逆向设计策略](#7-逆向设计策略)
8. [贝叶斯优化](#8-贝叶斯优化)
9. [可合成性评估](#9-可合成性评估)
10. [案例研究](#10-案例研究)

---

## 1. 为什么需要生成模型？

### 1.1 传统方法的局限

**前三天的方法：**

```
第一天：结构分析 → 理解已知 MOF
第二天：性质预测 → 给定结构 → 预测性质
第三天：GNN预测 → 更准确的性质预测

问题：只能预测，不能设计！
```

**逆向设计的挑战：**

```
传统方式：
设计师 → 猜测结构 → 合成 → 测试 → 失败 → 重复
         ↑__________________________________|

问题：
- MOF 设计空间巨大（10^6 - 10^18）
- 实验成本高（时间、材料、人力）
- 缺乏系统性设计方法
```

### 1.2 生成模型的价值

**生成式 AI 方法：**

```
目标性质 → [生成模型] → 新 MOF 结构 → [预测模型] → 验证性质
                ↑                              ↓
                |________反馈优化_______________|

优势：
✅ 自动生成候选结构
✅ 目标导向（性质→结构）
✅ 探索未知化学空间
✅ 加速材料发现
```

**应用场景：**

1. **De Novo 设计**：从零生成全新 MOF
2. **性质优化**：在已知 MOF 基础上改进
3. **多目标优化**：同时优化多个性质
4. **可合成性筛选**：生成易合成的结构

### 1.3 成功案例

**实例：CO₂ 捕集 MOF 设计**

传统方法：
- 筛选 10,000 个已知 MOF
- 找到 5 个高性能候选
- 耗时：6 个月

生成模型方法：
- 生成 1,000 个新 MOF
- 筛选出 20 个高性能候选
- 耗时：2 周
- **加速：10倍以上**

---

## 2. 生成模型基础

### 2.1 什么是生成模型？

**定义：**

学习数据分布 P(x)，并能够生成新样本。

```
训练数据: {x₁, x₂, ..., xₙ}  (已知 MOF 结构)
          ↓
    [学习分布 P(x)]
          ↓
   生成新样本: x_new ~ P(x)  (新 MOF 结构)
```

**vs 判别模型：**

| 模型类型 | 任务 | 学习目标 | 示例 |
|---------|------|---------|------|
| **判别模型** | 分类/回归 | P(y\|x) | GNN 预测性质 |
| **生成模型** | 生成新数据 | P(x) 或 P(x,y) | VAE 生成 MOF |

### 2.2 生成模型的类型

```
生成模型
├─ 显式密度模型
│  ├─ 易处理 (Tractable)
│  │  └─ 自回归模型 (Autoregressive)
│  └─ 近似推断 (Approximate)
│     ├─ 变分推断 (VAE) ⭐
│     └─ 马尔可夫链蒙特卡罗
│
└─ 隐式密度模型
   ├─ 生成对抗网络 (GAN) ⭐
   ├─ 扩散模型 (Diffusion) ⭐
   └─ 流模型 (Flow)
```

### 2.3 评估指标

**生成模型的评估：**

1. **生成质量**
   - 化学有效性：生成的结构是否合理？
   - 多样性：生成样本是否覆盖广泛空间？
   - 新颖性：是否生成训练集中没有的结构？

2. **性质匹配**
   - 生成样本的性质是否符合目标？
   - 分布是否与训练数据一致？

3. **可合成性**
   - 生成的 MOF 是否可以实验合成？

---

## 3. 变分自编码器 (VAE)

### 3.1 VAE 原理

**核心思想：**

将高维数据（MOF 结构）映射到低维潜空间（latent space），再从潜空间重构数据。

```
编码器:  x (MOF结构) → z (潜空间表示)
解码器:  z → x̂ (重构的MOF)
```

**数学框架：**

```
目标：最大化 P(x) = ∫ P(x|z) P(z) dz

问题：积分难以计算

解决：变分推断
- 引入 q(z|x) 近似 P(z|x)
- 最大化 ELBO (Evidence Lower Bound)
```

**ELBO (变分下界)：**

```
ELBO = 𝔼[log P(x|z)] - KL(q(z|x) || P(z))
       ↑                  ↑
   重构损失           正则化项

最大化 ELBO ⟺ 最小化损失:
Loss = Reconstruction Loss + KL Divergence
```

### 3.2 VAE 架构

```
┌─────────────────────────────────────────────────┐
│                    VAE                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  输入 x (MOF结构)                                │
│     ↓                                           │
│  ┌─────────┐                                    │
│  │ 编码器  │ → μ(x), σ(x)                       │
│  │Encoder │                                     │
│  └─────────┘                                    │
│      ↓                                          │
│  [重参数化技巧]                                  │
│  z = μ + σ ⊙ ε,  ε ~ N(0,I)                     │
│      ↓                                          │
│  ┌─────────┐                                    │
│  │ 解码器  │ → x̂ (重构MOF)                       │
│  │Decoder │                                     │
│  └─────────┘                                    │
│                                                 │
│  损失函数:                                       │
│  L = ||x - x̂||² + KL(q(z|x) || N(0,I))         │
│                                                 │
└─────────────────────────────────────────────────┘
```

**重参数化技巧 (Reparameterization Trick)：**

```python
# 问题：无法对随机采样求导
z ~ N(μ, σ²)  # 无法反向传播

# 解决：重参数化
ε ~ N(0, 1)
z = μ + σ * ε  # 可以对 μ, σ 求导
```

### 3.3 MOF-VAE

**MOF 表示：**

```
选项 1: 图表示
- 节点：原子
- 边：键
- 使用 GNN 编码器/解码器

选项 2: SMILES/InChI 字符串
- 将 MOF 结构转为字符串
- 使用 RNN/Transformer

选项 3: 晶格参数 + 原子坐标
- 固定长度向量
- 使用 CNN/MLP
```

**MOF-VAE 架构示例：**

```python
class MOFVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        # 编码器
        self.encoder = GNN(...)  # 图神经网络
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder = GNN(...)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_sigma(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
```

**损失函数：**

```python
def vae_loss(x, x_recon, mu, log_var):
    # 重构损失（MSE 或交叉熵）
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss  # beta: 权重
```

### 3.4 潜空间操作

**潜空间的性质：**

```
z₁ (MOF A) ─────→ z₂ (MOF B)
            插值

生成中间结构:
z_mid = α * z₁ + (1-α) * z₂,  α ∈ [0,1]
```

**性质导向生成：**

```
训练额外的性质预测器:
z → [MLP] → predicted_property

目标：
min_z ||property_predictor(z) - target_property||²
```

---

## 4. 生成对抗网络 (GAN)

### 4.1 GAN 原理

**核心思想：**

两个网络对抗训练：生成器 (Generator) 和判别器 (Discriminator)。

```
生成器 G:  噪声 z ~ N(0,I) → 假数据 x_fake
判别器 D:  数据 x → 概率 P(真)

目标：
- G 希望骗过 D（生成逼真数据）
- D 希望区分真假（判别准确）
```

**数学公式：**

```
min_G max_D V(D,G) = 𝔼[log D(x)] + 𝔼[log(1 - D(G(z)))]
                     ↑              ↑
                 真数据判为真     假数据判为假
```

### 4.2 GAN 训练

**训练循环：**

```python
for epoch in range(epochs):
    # 1. 训练判别器
    real_data = sample_from_dataset()
    fake_data = generator(random_noise())

    d_loss_real = -log(discriminator(real_data))
    d_loss_fake = -log(1 - discriminator(fake_data))
    d_loss = d_loss_real + d_loss_fake

    update_discriminator(d_loss)

    # 2. 训练生成器
    fake_data = generator(random_noise())
    g_loss = -log(discriminator(fake_data))  # 希望判别器认为是真的

    update_generator(g_loss)
```

**训练挑战：**

- 模式崩溃 (Mode Collapse)：生成器只生成少数几种样本
- 训练不稳定：G 和 D 难以平衡
- 梯度消失：判别器太强，生成器无法学习

**改进方法：**

- WGAN (Wasserstein GAN)：改进损失函数
- WGAN-GP：梯度惩罚
- StyleGAN：渐进式训练

### 4.3 MOF-GAN

**条件 GAN (cGAN) for MOF：**

```
输入：噪声 z + 目标性质 y
输出：MOF 结构 x

生成器: G(z, y) → x
判别器: D(x, y) → P(真 | x, y)

优势：可以指定目标性质生成 MOF
```

**挑战：**

- MOF 结构的离散性（原子类型、键）
- 化学有效性约束
- 需要大量训练数据

---

## 5. 扩散模型 (Diffusion Models)

### 5.1 扩散模型原理

**核心思想：**

通过逐步添加噪声（前向过程）破坏数据，然后学习逆向去噪过程生成数据。

```
前向扩散（固定）:
x₀ → x₁ → x₂ → ... → x_T ~ N(0,I)
干净数据        逐渐加噪        纯噪声

逆向去噪（学习）:
x_T → x_{T-1} → ... → x₁ → x₀
噪声              去噪步骤          生成数据
```

**数学框架：**

**前向过程：**

```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜI)

其中 βₜ: 噪声调度 (noise schedule)
```

**逆向过程（学习）：**

```
p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), Σ_θ(xₜ, t))

目标：学习 μ_θ 和 Σ_θ
```

### 5.2 去噪扩散概率模型 (DDPM)

**训练目标：**

```
预测噪声:
给定 xₜ 和 t，预测添加的噪声 ε

损失函数:
L = 𝔼[||ε - ε_θ(xₜ, t)||²]
```

**算法：**

**训练：**
```python
1. 从数据集采样 x₀
2. 随机选择时间步 t ~ Uniform(1, T)
3. 采样噪声 ε ~ N(0, I)
4. 计算 xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
5. 训练模型预测 ε: ε_θ(xₜ, t)
6. 更新参数最小化 ||ε - ε_θ(xₜ, t)||²
```

**采样（生成）：**
```python
1. 从纯噪声开始 x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
     ε_θ = model(xₜ, t)
     xₜ₋₁ = denoise(xₜ, ε_θ, t)
3. 返回 x₀
```

### 5.3 MOF 扩散模型

**E(3) 等变扩散模型：**

MOF 是 3D 结构，需要保持旋转/平移不变性。

```
E(3) Equivariant Diffusion:
- 节点特征（标量）: 旋转不变
- 坐标（向量）: 旋转等变

示例: EDM (E(3) Diffusion Model)
```

**条件生成：**

```
给定目标性质 y，生成 MOF x:

p(x | y) 通过条件扩散建模:
- 在每个去噪步骤中加入条件 y
- ε_θ(xₜ, t, y)
```

**优势：**

✅ 生成质量高
✅ 训练稳定（相比 GAN）
✅ 易于条件生成
✅ 支持多模态

**挑战：**

❌ 生成速度慢（需要多步去噪）
❌ 计算成本高

---

## 6. MOF 生成模型

### 6.1 现有 MOF 生成模型

**1. MOF-VAE (2020)**
- 使用 VAE 学习 MOF 拓扑的潜空间
- 输入：拓扑代码
- 输出：新拓扑

**2. MOFGen (2021)**
- 基于 GAN 的条件生成
- 输入：目标气体吸附量
- 输出：MOF 结构参数

**3. DiffCSP (2023)**
- 扩散模型用于晶体结构预测
- 生成晶格参数 + 原子坐标

**4. MatterGen (2024)**
- 多模态扩散模型
- 条件生成（性质、组成约束）

### 6.2 MOF 表示的挑战

**离散 + 连续混合：**

```
MOF 结构 = {
    晶格参数: 连续 (a, b, c, α, β, γ)
    原子类型: 离散 (C, H, O, N, Zn, ...)
    原子坐标: 连续 (x, y, z)
    拓扑: 离散 (pcu, dia, ...)
}

挑战：如何在一个模型中处理？
```

**解决方案：**

1. **分层生成**
   ```
   Step 1: 生成拓扑（离散）
   Step 2: 生成金属节点（离散）
   Step 3: 生成配体（离散/连续）
   Step 4: 优化原子坐标（连续）
   ```

2. **联合表示**
   ```
   使用连续潜空间 z 表示整个 MOF
   z → [Decoder] → (topology, nodes, linkers, coords)
   ```

### 6.3 化学有效性约束

**生成模型的问题：**

可能生成化学上不合理的结构：
- 原子重叠
- 不合理的键长/键角
- 违反化学价规则
- 不稳定的结构

**解决方案：**

1. **软约束（训练中）**
   ```
   Loss = Reconstruction Loss + α * Chemistry Loss

   Chemistry Loss:
   - 键长惩罚: Σ max(0, d_min - d_ij)
   - 价态惩罚: Σ |valence(atom) - expected|
   ```

2. **硬约束（后处理）**
   ```
   生成 → 检查化学有效性 → 拒绝/修正 → 输出
   ```

3. **结构优化**
   ```
   生成初始结构 → DFT/力场优化 → 最终结构
   ```

---

## 7. 逆向设计策略

### 7.1 目标导向生成

**问题定义：**

```
给定：目标性质 y* (如 CO₂ 吸附量 > 5 mmol/g)
求解：MOF 结构 x*，使得 f(x*) ≈ y*
```

**方法 1：潜空间优化**

```
1. 训练 VAE: x ⟷ z
2. 训练性质预测器: z → f(z)
3. 优化潜空间:
   z* = argmin_z ||f(z) - y*||²
4. 生成: x* = Decoder(z*)
```

**方法 2：条件生成**

```
训练条件 VAE/Diffusion:
p(x | y)

生成:
x* ~ p(x | y*)
```

**方法 3：强化学习**

```
智能体：生成器
环境：性质预测器
奖励：R = -||f(x) - y*||²

训练策略最大化期望奖励
```

### 7.2 多目标优化

**现实问题：**

```
同时优化多个性质：
- 高 CO₂ 吸附量
- 高 CO₂/CH₄ 选择性
- 低合成成本
- 高稳定性
```

**Pareto 优化：**

```
目标：找到 Pareto 前沿
- 无法同时改进所有目标
- 权衡不同目标

方法：
- 加权求和: L = w₁L₁ + w₂L₂ + ...
- Pareto GAN
- 多目标进化算法
```

### 7.3 约束优化

**约束条件：**

```
优化：max f(x)
约束：
  - 可合成性 > 阈值
  - 稳定性 > 阈值
  - 成本 < 预算
```

**方法：**

1. **惩罚法**
   ```
   L = -f(x) + λ Σ max(0, constraint_violation)
   ```

2. **投影梯度**
   ```
   更新 z → 投影到可行域
   ```

3. **拉格朗日乘数法**

---

## 8. 贝叶斯优化

### 8.1 原理

**问题：**

昂贵的黑盒优化
```
给定：目标函数 f(x)（昂贵评估，如 DFT 计算）
求解：x* = argmax f(x)
约束：评估次数有限
```

**贝叶斯优化思想：**

```
1. 建立 f(x) 的概率模型（高斯过程）
2. 使用采集函数决定下一个评估点
3. 评估 f(x)，更新模型
4. 重复直到收敛
```

### 8.2 高斯过程 (GP)

**定义：**

```
f(x) ~ GP(μ(x), k(x, x'))

μ(x): 均值函数
k(x, x'): 核函数（协方差）
```

**性质：**

给定观测 D = {(x₁, f₁), ..., (xₙ, fₙ)}，
在新点 x* 的预测：

```
f(x*) | D ~ N(μ*(x*), σ²*(x*))

μ*(x*): 预测均值
σ²*(x*): 预测不确定性
```

### 8.3 采集函数 (Acquisition Function)

**作用：**

平衡探索 (Exploration) 和利用 (Exploitation)

**常用采集函数：**

**1. Expected Improvement (EI)**

```
EI(x) = 𝔼[max(0, f(x) - f_best)]

      = (μ(x) - f_best) Φ(Z) + σ(x) φ(Z)

其中: Z = (μ(x) - f_best) / σ(x)
     Φ, φ: 标准正态分布的 CDF/PDF
```

**2. Upper Confidence Bound (UCB)**

```
UCB(x) = μ(x) + κ σ(x)

κ: 探索参数（越大越倾向探索）
```

**3. Probability of Improvement (PI)**

```
PI(x) = P(f(x) > f_best)
      = Φ((μ(x) - f_best) / σ(x))
```

### 8.4 MOF 设计中的贝叶斯优化

**应用：**

```
潜空间 z → [Decoder] → MOF x → [DFT/GCMC] → property y

目标：找到最优 z*

1. 初始采样：随机生成 n 个 z
2. 评估：计算对应的 MOF 性质
3. 建立 GP: z → y
4. 采集函数：选择下一个 z_next
5. 重复直到满足条件
```

**优势：**

- 样本高效（少量评估找到最优）
- 考虑不确定性
- 适合昂贵函数

---

## 9. 可合成性评估

### 9.1 为什么重要？

**问题：**

生成模型可能生成理论上高性能但实际无法合成的 MOF。

**挑战：**

- 热力学稳定性
- 动力学可达性
- 前驱体可获得性
- 合成条件可行性

### 9.2 可合成性指标

**1. 合成可及性分数 (SA Score)**

```
基于：
- 结构复杂度
- 官能团频率（在已知化合物中）
- 环系统复杂度

范围：1-10
- 1: 易合成
- 10: 难合成
```

**2. 逆合成分析**

```
目标 MOF → 逆向推断 → 前驱体 + 反应路径

检查：
- 前驱体是否商业可得？
- 反应路径是否已知？
```

**3. 稳定性预测**

```
使用 ML 模型预测：
- 热力学稳定性（形成能）
- 机械稳定性（体积模量）
- 化学稳定性（抗水解、抗氧化）
```

### 9.3 集成到生成流程

**方法 1：后筛选**

```
生成 → 可合成性评估 → 过滤 → 候选
```

**方法 2：约束优化**

```
优化：max f(x)
约束：SA_score(x) < 阈值
```

**方法 3：多任务学习**

```
同时优化：
- 目标性质
- 可合成性

Loss = w₁ L_property + w₂ L_synthesizability
```

---

## 10. 案例研究

### 10.1 案例：CO₂ 捕集 MOF 逆向设计

**目标：**

设计高性能 CO₂ 捕集 MOF：
- CO₂ 吸附量 > 6 mmol/g (298K, 1bar)
- CO₂/N₂ 选择性 > 50
- 可合成

**方法：VAE + 贝叶斯优化**

**流程：**

```
1. 数据准备
   - 收集 10,000 个已知 MOF + 性质数据

2. 训练 VAE
   - 输入：MOF 图表示
   - 潜空间维度：128
   - 训练 epoch：500

3. 训练性质预测器
   - 输入：潜空间 z
   - 输出：CO₂ 吸附量、选择性
   - 模型：MLP

4. 贝叶斯优化
   - 初始：随机采样 20 个 z
   - 采集函数：EI
   - 迭代：100 次
   - 每次评估：Decoder(z) → GCMC 模拟

5. 筛选与验证
   - 可合成性评估
   - DFT 优化
   - 实验验证（可选）
```

**结果：**

- 生成 50 个候选 MOF
- 其中 10 个满足所有约束
- 最佳 MOF：
  - CO₂ 吸附量：6.8 mmol/g ✅
  - 选择性：65 ✅
  - SA Score：4.2 ✅（可合成）

**vs 传统方法：**

| 指标 | 传统筛选 | VAE + BO |
|-----|---------|----------|
| 候选数 | 100 | 50 |
| 高性能 | 5 | 10 |
| 成功率 | 5% | 20% |
| 耗时 | 6 月 | 1 月 |

### 10.2 案例：多目标 MOF 优化

**目标：**

同时优化：
1. 甲烷工作容量 (最大化)
2. 密度 (最小化，便于运输)
3. 合成成本 (最小化)

**方法：Multi-objective GAN**

**Pareto 前沿：**

```
        甲烷工作容量
              ↑
              |    ● Pareto 最优点
              |  ●   ●
              | ●     ●
              |●       ●
              |_________→ 密度
```

生成的 MOF 分布在 Pareto 前沿上，提供多种权衡选择。

**结果：**

- 识别 15 个 Pareto 最优 MOF
- 用户可根据具体应用选择权衡点

---

## 总结

### 关键要点

1. **生成模型类型**
   - **VAE**：潜空间平滑，易于优化
   - **GAN**：生成质量高，但训练困难
   - **Diffusion**：最新方法，质量最佳

2. **MOF 生成的挑战**
   - 离散+连续混合表示
   - 化学有效性约束
   - 可合成性

3. **逆向设计策略**
   - 潜空间优化
   - 条件生成
   - 贝叶斯优化

4. **实用建议**
   - 从 VAE 开始（简单、稳定）
   - 使用贝叶斯优化节省计算
   - 必须考虑可合成性

### 下一步

- **实操练习**：完成第四天实操教程
- **深入学习**：阅读 VAE、扩散模型论文
- **项目实践**：在自己的 MOF 设计问题上应用

---

## 参考资料

**核心论文：**

1. **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
2. **GAN**: Goodfellow et al., "Generative Adversarial Nets" (2014)
3. **Diffusion**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
4. **MOF-VAE**: Yao et al., "Inverse Design of Nanoporous Materials" (2020)
5. **DiffCSP**: Jiao et al., "Crystal Structure Prediction by Diffusion" (2023)
6. **Bayesian Optimization**: Shahriari et al., "Taking the Human Out of the Loop" (2016)

**MOF 生成相关：**

7. MOFGen: Conditional generation of MOFs (2021)
8. MatterGen: Multimodal diffusion for materials (2024)

**在线资源：**

- Molecule.one: 逆合成分析工具
- BO libraries: GPyOpt, BoTorch, Ax

---

*第四天理论结束 - 接下来进入实操环节*
