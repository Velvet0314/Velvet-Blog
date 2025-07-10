---
title: PocketGen
createTime: 2025/05/22 20:12:32
tags:
  - 口袋生成
  - pLM
  - Transformer
permalink: /papers/分子生成/pocketgen
prev: /papers/分子生成/frag2seq
next: /papers/分子生成/ipdiff
outline: [2,5]
---

## **PocketGen 概述**

### **PocketGen 在干什么**

PocketGen是一种基于深度学习的生成模型，**专注于高效**==**生成**=={.danger}**与配体分子（如药物小分子）结合的全原子结构**==**蛋白质口袋**=={.danger}。其核心应用场景包括：

1. 药物发现：设计高亲和力的蛋白质口袋以加速先导化合物筛选，缩短药物研发周期

2. 酶工程：生成催化活性位点，优化酶的功能性以用于工业生物催化

3. 生物传感器开发：构建特异性结合口袋，用于检测环境或体内的特定分子

4. 蛋白质功能改造：重新设计现有蛋白质的结合位点，赋予其新的配体结合能力

### **PocketGen 解决的问题**

| 🚀 问题类别         | ❌ 传统方法的局限                                           | ✅ PocketGen 的解决方案                                                    |
| --------------- | --------------------------------------------------- | -------------------------------------------------------------------- |
| **序列-结构协同建模能力** | 🚨 基于物理的模拟或模板匹配难以同时优化蛋白质序列与三维结构，生成口袋缺乏生物学合理性        | ✨ 引入 **序列精炼模块**，结合 pLM 和结构适配器，通过交叉注意力机制实现结构信息向序列的注入，提升一致性            |
| **生成效率**        | 🚨 传统物理方法（如 PocketOpt）生成 100 个口袋需 >1000 秒，速度慢，效率低   | ✨ 采用 **迭代优化框架** 替代扩散模型，显著提升生成效率，仅为传统方法的 1/10，保持高质量                   |
| **多样性与亲和力平衡能力** | 🚨 生成高多样性口袋时结合亲和力常明显下降，难以兼顾                         | ✨ PocketGen 能保持高多样性的同时维持较高亲和力，**成功率达 95%**，实现多样性与功能性的良好平衡            |
| **多尺度相互作用建模**   | 🚨 传统模型难以统一建模原子级（如氢键）、残基级（侧链）与配体级（疏水作用）相互作用，导致结构不精准 | ✨ 设计 **双层图变换器（Bilevel Graph Transformer）**，通过分层注意力机制与跨层信息传递建模多尺度结构信息 |
| **空间变换鲁棒性**     | 🚨 模型对旋转、平移等坐标系变换敏感，导致预测结果依赖初始坐标系，结构稳定性差            | ✨ 引入 **E(3)-等变性建模框架**，确保结构对旋转、平移等几何变换具有不变性，提升预测结构的几何真实性与泛化能力         |

**⭐核心：**

1. ==**结构与序列的一致性**=={.note}
2. ==**准确模拟复杂口袋与配体间的相互作用**=={.note}

### **PocketGen 小总结**

:::: steps
1. PocketGen采用协同设计方式，基于 ==**配体分子与口袋周围的蛋白质结构**=={.important} 来预测 ==**口袋的氨基酸序列与口袋的结构**=={.important}

2. PocketGen不是将每个原子都视为图中的一个独立节点，而是 ==**将蛋白质-配体复合物表示为由“块 (blocks)”**=={.important} 组成的几何图。这里的“块”可以理解为比单个原子更高层次的结构单元

   - 对于蛋白质：一个“块”很可能代表一个 ==**氨基酸残基 (residue)**=={.tip}
   - 对于配体：一个“块”可能代表 ==**整个配体分子**=={.tip}，或者如果配体较大且具有柔性，也可能被划分为几个关键的子结构或药效团作为“块”

3. 几何图的选择确保了其仅包含连接关系，还包含了这些“块”在三维空间中的坐标信息，使得模型能够理解和处理空间结构

4. 双层注意力机制来捕捉多个粒度的相互作用：
     - 细粒度：==**原子和残基/配体层面**=={.tip}的相互作用
     - 粗粒度：包括 ==**蛋白质内和蛋白质-配体**=={.tip}相互作用

5. 在序列的更新中添加了一个 ==**结构适配器**=={.important}，实现结构与序列的对齐。在训练时，冻结其他层权重，只针对适配器进行训练
::::

## **PocketGen 的 Tips**

### ==**Tip 1：PocketGen 方法概述（Overview of PocketGen）**=={.note}

**⭐目标：**

**共同设计**蛋白质口袋的残基类型（序列）和三维（3D）结构，使其能够与靶配体分子结合

==**PocketGen 将口袋生成形式化为一个条件生成问题**=={.important}，其目标是学习一个条件生成模型：

$$
P(\mathcal{B} \mid \mathcal{A} \setminus \mathcal{B}, \mathcal{M})
$$

这表示已知：

- ==**口袋周围的蛋白质结构**（不包括口袋）：$\mathcal{A} \setminus \mathcal{B}$=={.tip}
- ==**配体的3D结构：** $\mathcal{M}$=={.tip}

去预测口袋区域 $\mathcal{B}$ 的概率分布

::: note “条件生成”的过程

具体来说，模型会根据两个主要的输入信息——即蛋白质分子上围绕着目标口袋的现有结构部分（口袋周围的蛋白质结构）以及我们希望这个口袋能够结合的特定小分子（称为“结合配体”）——来同时创造出这个口袋的氨基酸组成（序列）和其三维原子排布（结构）
:::

#### **1. 蛋白质和配体的基本定义**

- ==**整个蛋白质的残基序列 $\mathcal{A} = \mathbf{a}_1 \ldots \mathbf{a}_{N_s}$**=={.important}
  由 $N_s$ 个残基组成，其中每个残基 $\mathbf{a}_i$ 是一个氨基酸

- **蛋白质的3D结构表示为点云** $\{ {\mathbf{a}_{i,j}} \}_{1 \leq i \leq {N_s}, 1 \leq j \leq n_i}$
  
  ==${\mathbf{a}_{i,j}}$ 表示残基 $i$ 的第 $j$ 个原子=={.tip}
  
  ==每个残基 $\mathbf{a}_i$ 有 $n_i$ 个原子，由残基的类型决定=={.tip}
  
  每个原子的三维坐标为 $x(\mathbf{a}_{i,j}) \in \mathbb{R}^3$

- **前4个原子是主链原子 C$_\alpha$ [+note1], N, C, O，其他是侧链原子**

[+note1]:详见 [备忘录：分子生成术语](/notes/Memo/molecule.md) 

#### **2. 配体表示**

- 配体分子也被表示为3D点云 $\mathcal{M} = \{\mathbf{v}_k\}_{k=1}^{N_t}$，其中每个 $\mathbf{v}_k$ 是配体中的一个原子

- $x(\mathbf{v}_k)$ 表示第 $k$ 个原子的三维坐标


#### **3. 蛋白质口袋（pocket）的定义**

- **口袋 $\mathcal{B} = \mathbf{b}_1 \ldots \mathbf{b}_m$** 定义为 ==**最接近**=={.danger} 配体分子的那些残基

- 口袋也可由 ==**整个蛋白质的残基序列的一个子序列 $\mathcal{B} = \mathbf{a}_{e_1} \ldots \mathbf{a}_{e_m}$ 来表示**=={.important}

- **$\mathbf{e} = \{e_1, \ldots, e_m\}$** 表示 **属于口袋的蛋白质残基序列** 的索引集合

::: tip 为什么是"最接近"?
在三维空间中，**如果一个氨基酸残基的任何部分（任何原子）与结合配体分子的任何部分（任何原子）之间的距离在一个特定的阈值（例如$3.5 \text{Å}$）以内，那么这个氨基酸残基就被定义为属于该蛋白质口袋**

这个距离阈值（通常是根据典型的非键相互作用（如氢键、范德华力）的有效作用范围来选择的
:::

#### **4. 如何选出口袋的残基**

口袋的残基定义为那些至少有一个原子距离配体中某个原子的距离小于某个阈值 $\delta$ 的残基：

$$
\mathbf{e} = \left\{i \,\middle|\, \min_{1 \leq j \leq n_i, 1 \leq k \leq N_t} \|x(\mathbf{a}_{i,j}) - x(\mathbf{v}_k)\|_2 \leq \delta \right\}
$$

- $x(\mathbf{a}_{i,j})$：第 $i$ 个残基中第 $j$ 个原子的坐标
- $x(\mathbf{v}_k)$：配体中第 $k$ 个原子的坐标
- $\|\cdot\|_2$：L2 范数
- $\delta = 3.5 \text{Å}$：距离阈值，表示一个残基的任意原子距离配体中任意原子不超过 $3.5 \text{Å}$，就把这个残基认为在口袋中

### ==**Tip 2：等变双层图Transformer（Equivariant bilevel graph transformer）**=={.note}

为了表示的简洁性和计算的便利性，每个 ==**氨基酸残基（residue）**=={.important} 或 ==**整个配体分子（ligand）**=={.important} 被视为一个 ==**块 (block)**=={.important}。**一个块就是一组原子**

#### **符号定义**

蛋白质配体的复合物可以抽被象为一个由集合构成的几何图 $\mathscr{g}=(\upsilon,\varepsilon)$

**节点** $\upsilon$： 每个节点代表一个块 $i$。每个块 $i$ 包含两部分信息：$\upsilon = \{H_i,X_i \mid 1 \leq i \leq B\}$：

- **原子特征** $H_i \in R^{n_i × d_h}$：一个矩阵，==其中 $n_i$ 是块 $i$ 中的原子数量，$d_h$ 是原子特征的维度。$H_i[p]$ 是块 $i$ 中第 $p$ 个原子的可训练特征向量=={.tip}
    
- **原子坐标** $X_i \in R^{n_i × 3}$： 一个矩阵，$X_i[p]$ 是块 $i$ 中第 $p$ 个原子的三维坐标
    
- **特征初始化：** $H_i[p]$ 的初始特征由 **原子类型嵌入 (atom-type embedding)**、**残基/配体嵌入 (residue/ligand embeddings)** 和 **原子位置嵌入 (atom positional embeddings)** 拼接而成
    

**边** $\varepsilon$：

- 残基块间使用 ==**k-近邻图 (k-Nearest Neighbors graph, k-NN graph)**=={.important}
    
  对于蛋白质中的 **每一个残基块**：
      
    1. 计算它与其他所有残基块之间的距离（残基与残基之间两两计算距离）
    2. 找出与它距离最近的 $k$ 个其他残基
    3. ==**在这个残基与这 $k$ 个最近邻残基之间建立边**=={.tip}
    
- ==残基块与配体块之间建立边=={.tip}

- ==**每个块有一个自环（self-loop）**=={.tip}

#### **⭐原子级别注意力**

- 标准 Attention： 

$$
Q_i = H_iW_Q,\qquad K_J = H_jW_K,\qquad V_j = H_jW_v
$$

其中 $H_i\in\mathbb{R}^{n_i\times d_h}$ 为第 $i$ 号块内的原子特征矩阵，$W_Q,W_K,W_V$ 为可训练参数

- 定义 $X_{ij} \in \mathbb{R}^{n_i \times n_j \times 3}$ 为块 $i$ 和 块 $j$ 间的相对坐标，$D_{ij} \in \mathbb{R}^{n_i \times n_j}$ 为块 $i,j$ 间的距离

记 

$$
X_{ij}[p,q] = X_i[p] - X_j[q], D_{ij}[p,q] = \|X_{ij}[p,q]\|_2
$$

- ==原子间的注意力分数矩阵 $R_{ij} \in R^{n_i \times n_j}$ 中每一项：=={.important}

$$
R_{ij}[p,q] = \frac{1}{\sqrt{d_r}}(Q_i[p] ⋅ K_j[q]^T) + \sigma_D(RBF(D_{ij}[p,q]))
$$

这里 $d_r$ 是 $Q,K,V$向量的维度

- $\sigma_D(\mathrm{RBF}(D_{ij}[p,q]))$ 引入了距离偏置，使用 RBF (Radial Basis Functions) 将距离标量嵌入为向量

- 原子级注意力权重矩阵 $\alpha_{ij} \in R^{n_i \times n_j}$：

$$
\alpha_{ij} = \mathrm{Softmax}(R_{ij})
$$ 

对 $R_{ij}$ 的每一行进行 Softmax，得到块 $i$ 中每个原子对块 $j$ 中所有原子的注意力分布

**top-k 稀疏化**：为了计算效率和关注重点，$\alpha_{ij}$ 的每一行只保留值最大的 $k$ 个元素，其余设为零

#### **⭐残基-配体级别注意力**

- 块间整体关联性：

$$
r_{ij} = \frac{\mathbf{1}^TR_{ij}\mathbf{1}}{n_in_j}
$$

第一个 $\mathbf{1}^T \in 1 \times n_i$，第二个 $\mathbf{1} \in n_j \times 1$

这样就计算了 $R_{ij}$ 中块 $i$ 的每个原子对块 $j$ 中每个原子的注意力分数之和，在除以矩阵中元素数量得到块 $i$ 和块 $j$ 间的原子交互的**平均强度**

- 计算块 $j$ 对块 $i$ 的块级注意力权重 $\beta_{ij}$：

$$
\beta_{ij} = \frac{\exp(r_{ij})}{\sum_{j\in\mathcal N(i)}\exp(r_{ij})}
$$

其中 $\mathcal N(i)$表示 块 $i$ 的相邻块，做一个 softmax 的归一化，得到一个和为 1 的块间的注意力权重分布

::: note
==当 $i=j$ 时，就是 self-loop 针对块内的作用=={.note}
:::

#### **⭐更新公式**

- 更新公式：

原子 $p$ 在块 $i$ 中从块 $j$ 接收到的消息 $m_{ij,p}$：

$$
\begin{align*}

m_{ij,p} &= \beta_{ij} \bigl(\alpha_{ij}[p] \odot \phi_x(\ Q_i[p]\ ||\ K_j\ ||\ RBF(D_{ij}[p]))\bigr) \\[8pt]

H'_i[p] &= H_i[p] + \sum\limits_{j \in N(i)} \beta_{ij} \phi_h(\alpha_{ij}[p] ⋅ V_j) \\[8pt]

X_i'[p] &= X_i[p] +
\underbrace{
    
  \begin{cases}
  
  0, & \text{if } i \text{ belongs to other protein residues} \\[5pt]
  \sum_{j \in \mathcal{N}(i)} m_{ij, p} \cdot X_{ij}[p], & \text{if } i \text{ belongs to ligand or pocket residues} 
    
  \end{cases}
    
}_{\text{只考虑配体与口袋的残基}}
\end{align*}
$$

#### **等变FFN**

- 特征质心：$h_c = \mathrm{centroid}(H_i)$

- 几何质心：$x_c = \mathrm{centroid}(X_i)$

原子相对于块质心的相对坐标 $\Delta x_p$ 及其距离嵌入 $r_p$：

$$
\Delta x_p =X_i[p] - x_c,\qquad r_p=\mathrm{RBF}(\|\Delta x_p\|_2)
$$

FFN 更新公式：

$$
\begin{align*}
H'[p] &= H[p] + \sigma_h(H_i[p], h_c, r_p) \\[8pt]
X_i'[p] &= X_i[p] + \Delta x_p \, \sigma_x(H_i[p], h_c, r_p)
\end{align*}
$$

使用 LayerNorm 加速并维持训练的稳定性

关键作用：

- 引入非线性，学习更复杂的特征
    
- 质心特征学习
    
- 整个计算过程所用变量都能保持E(3)等变性

### **Tip 3：带有结构适配器的序列细化模块（Sequence refinement with pLMs and adapters）**


利用微调 pLM 来帮助改进蛋白质口袋序列 —— ==通过 adapter 向 pLM 注入先前网络得到的结构信息=={.important}

同时在训练时也只训练 adapter，冻结其他权重

#### **序列-结构交叉注意力机制（Sequence–structure cross-attention）**

- 结构特征 $H^{\mathrm{struct}} = \{ h^{\mathrm{struct}}_1,h^{\mathrm{struct}}_2 \cdots h^{\mathrm{struct}}_{N_s}\}$

==第 $i$ 个残基的结构表示：$h^{\mathrm{struct}}_i=\mathrm{mean\ pooling}(H_i)$=={.tip}

- 序列特征 $H^{\mathrm{seq}} = \{ h^{\mathrm{seq}}_1,h^{\mathrm{seq}}_2 \cdots h^{\mathrm{seq}}_{N_s}\}$

- 构建一个经典的 CrossAttention：

$$
\begin{align*}
  Q = H^{\mathrm{seq}}W_Q,\qquad K = H^{\mathrm{struct}}W_K, \qquad V = H^{\mathrm{struct}}W_V \\[8pt]
  \mathrm{CrossAttention}(Q_1, K_2, V_2) = \text{Softmax}\left( \frac{Q_1 K_2^\top}{\sqrt{d_r}} \right) V_2
\end{align*}
$$

这里下标表示来自两个不同的序列

#### **Bottleneck FFN**

学习非线性与抽象特征
