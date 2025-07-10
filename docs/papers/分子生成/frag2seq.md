---
title: Frag2Seq
createTime: 2025/07/04 19:30:27
tags:
  - 配体生成
  - LMs
  - Fragment 
permalink: /papers/分子生成/frag2seq
prev: {}
next: /papers/分子生成/pocketgen
outline: [2,5]
---

## **背景与动机**

### **SBDD 的传统方法**

高通量虚拟筛选和实验验证 —— 费钱费时费力

### **SBDD 对机器学习的挑战**

要求模型捕捉 ==**复杂的蛋白质-配体相互作用**=={.tip}，同时 ==**提高设计分子的药物相似性**=={.tip}

### **相关工作**

#### ==**自回归模型和扩散模型**=={.note}

缺点:
  - 只考虑 ==**原子生成**=={.tip}
  - 扩散模型通常需要数千步才能生成，导致 ==**生成效率低下**=={.tip}

#### ==**基于分子片段生成分子**=={.note}
缺点:
  - 需要一个涉及多个神经网络的 ==**复杂管道**=={.tip} 来选择和连接片段

#### **LMs 语言模型的优势**

- 处理 ==**大型数据集的效率高于基于扩散**=={.tip} 的方法
  
  利用 LMs 的上下文能力，**基于分子片段生成**，可以得到**更真实的分子子结构和减少步骤以提高效率**
- 从 **大量生物和化学文本** 中学习以完成各种 ==**潜在任务**=={.tip}，并生成分子和材料

#### ==**LMs 面临的挑战**=={.note}

- 需要使 LMs 适应处理分子数据的 ==**几何图结构**=={.important}
- 能否 ==**准确模拟分子的物理和化学性质**=={.important}
- 如何在 LMs 中编码蛋白质上下文信息以有效 ==**捕获蛋白质-配体相互作用**=={.important}，从而产生可以结合靶蛋白的分子
- 现有的 LMs 要么只考虑 2D 分子，要么需要考虑额外的设计以缓解对齐问题，==**在 SBDD 中没有现有的工作使用 LMs 直接生成 3D 配体**=={.important}

## **模型解析**

### 1. 问题定义 (Problem Definition)

*   **蛋白质口袋 (Protein Pocket):** $\mathcal{P}=\{(s_i,\boldsymbol{b}_i)\}^{n}_{i=1}$
    *   $s_i \in \mathbb{R^3}$：第 $i$ 个原子的 3D 笛卡尔坐标。
    *   $\boldsymbol{b}_i$：原子类型的 one-hot 向量。
*   **配体 (Ligand):** $\mathcal{M}=\{(v_j,z_j)\}^{m}_{i=1}$
    *   $v_j \in \mathbb{R^3}$：第 $j$ 个原子的 3D 笛卡尔坐标。
    *   $z_j$：原子类型。
*   **目标 (Objective):** 学习一个条件生成模型 $\text{Model}$，从训练蛋白配对中捕获条件概率分布 $p(\mathcal{M} \mid \mathcal{P})$。

---

### 2. 3D 分子切片 (3D Molecular Slicing)

*   **优点 (Advantages):**
    1.  **化学意义 (Chemical Significance):** 片段通常对应功能基团（如苯环、羧基），生成的分子更“合理”。
    2.  **效率更高 (Higher Efficiency):** 生成步骤从原子数量级减少到片段数量级，大大加快生成速度。
    3.  **结构更优 (Better Structure):** 直接生成刚性片段，更好地保持片段内部的键长和键角，避免生成不合理的局部结构。
*   **切片条件 (Slicing Criteria):** 切断满足以下三个条件的**可旋转化学键**，以在**保留重要功能基团完整性**的前提下最大化分解分子：
    1.  键不在环中。
    2.  键类型为单键。
    3.  键上的起始原子和末端原子的度数大于1（避免打断如 -OH 或 -COOH 这样的重要官能团）。
*   **切片结果 (Slicing Outcome):** 对于分子 $\mathcal{M}$，将其拆分为一组 3D 片段 $\{\mathscr{g}_i=(Z_i,V_i)\}_{i=1}^k$，其中 $Z_{i}$ 表示原子类型矩阵，$V_{i}$ 表示笛卡尔坐标。

---

### 3. 基于片段的三维分子词元化 (Fragment-based 3D Molecular Tokenization)

#### 3.1. 基于三维图同构的原子排序 (Atom Reordering based on 3D Graph Isomorphism)

*   **规范 SMILES (Canonical SMILES):**
    *   每个分子（片段）图使用**规范 SMILES**表示，其表示是**唯一**的。
    *   这允许对分子数据进行高效索引和检索，确保原子重新排序的最大不变性。
*   **定义 3.1. 3D 分子图的同构 (Definition 3.1. *[3D Molecular Graph Isomorphism]*)**
    *   设 $\mathcal{M}_1 = (V_1, Z_1)$ 和 $\mathcal{M}_2 = (V_2, Z_2)$ 是两个 3D 分子图，其中 $z_i$ 是节点类型向量，$v_i$ 是分子 $\mathcal{M}_i$ 的节点坐标。设 $ver(\cdot)$ 表示顶点集合，$attr(\cdot)$ 表示节点属性，且不存在边。
    *   设 $\mathcal{M}_1 \cong \mathcal{M}_2$ 表示两个属性图是同构的。
    *   如果存在一个双射 $b : ver(\mathcal{M}_1) \to ver(\mathcal{M}_2)$ 使得对于 $\mathcal{M}_1$ 中的每个原子索引 $i$，有 $z_i^{\mathcal{M}_1} = z_{b(i)}^{\mathcal{M}_2}$，并且存在一个 3D 变换 $\tau \in SE(3)$ 使得 $v_i^{\mathcal{M}_1} = \tau(v_{b(i)}^{\mathcal{M}_2})$，则称两个 3D 分子 $\mathcal{M}_1$ 和 $\mathcal{M}_2$ 是 **3D 同构的**，记作 $\mathcal{M}_1 \cong_{3D} \mathcal{M}_2$。
    *   如果允许一个小的误差 $\varepsilon$ 使得 $|v_i^{\mathcal{M}_1} - \tau(v_{b(i)}^{\mathcal{M}_2})| \leq \varepsilon$，我们称这两个 3D 图是 $\varepsilon$-**约束 3D 同构**的。
*   **引理 3.2. 3D 分子图同构的规范序 (Lemma 3.2. *[Canonical Ordering for 3D Molecular Graph Isomorphism]*)**
    *   设 $\mathcal{M}_1 = (V_1, Z_1)$ 和 $\mathcal{M}_2 = (V_2, Z_2)$ 是两个遵循定义 3.1 的 3D 分子图。设 $\mathbf{L} : M \to \mathcal{L}$ 是一个函数，它将分子 $\mathcal{M} \in M$（所有有限 3D 分子图的集合）映射到其规范序 $L(\mathcal{M}) \in \mathcal{L}$（所有可能规范序的集合），这个规范序是由规范 SMILES 产生的。那么以下等价关系成立：
        $$L(\mathcal{M}_1) = L(\mathcal{M}_2) \Leftrightarrow \mathcal{M}_1 \cong_{3D} \mathcal{M}_2$$
    *   其中 $\mathcal{M}_1 \cong_{3D} \mathcal{M}_2$ 表示 $\mathcal{M}_1$ 和 $\mathcal{M}_2$ 是 3D 同构的。

#### 3.2. SE(3)-等变分子和片段的局部坐标系的构建 (Construction of SE(3)-Equivariant Local Coordinate Frames for Molecules and Fragments)

*   **构建分子局部坐标系 (Molecule Local Coordinate Frame):**
    *   给定一个原子类型为 $Z$、原子坐标为 $V$ 的 3D 分子 $\mathcal{M}$ 作为输入，构建其分子局部坐标系 $\mathfrak{m} = (x,y,z)$。
    1.  将切分的 3D 的片段按规范序进行排序（具体排序方式参考代码）。
    2.  坐标系基于规范顺序中的**前三个非共线碎片中心**构建。
        *   设 $\ell_{1},\ell_{2},\ell_{m}$ 为这三个片段中心的索引，分子局部坐标系 $\mathfrak{m} = (x,y,z)$ 计算如下：
            $$x = \text{normalize}(v_{\ell_2} - v_{\ell_1}), \quad y = \text{normalize}((v_{\ell_m} - v_{\ell_1}) \times x), \quad z = x \times y \tag{1}$$
            *   $\text{normalize}(\cdot)$ 是归一化函数，将向量长度变为 1。
            *   $v$ 表示世界坐标系中的坐标。
        *   经过归一化后得到分子局部坐标系的基向量 $\mathfrak{m} = (x,y,z)$。
    *   **解释:**
        1.  **定原点 (Define Origin):** 选择第一个片段的中心 $v_{\ell_1}$ 作为这个**局部坐标系的原点**。
        2.  **定 $x$ 轴 (Define x-axis):** 从原点 $v_{\ell_1}$ 指向第二个片段中心 $v_{\ell_2}$ 的方向，定义为 $x$ 轴。
        3.  **定 $y$ 轴 (Define y-axis):** 第三个片段中心 $v_{\ell_m}$ 与 $x$ 轴定义了一个平面。$y$ 轴垂直于这个平面（通过叉乘实现）。
        4.  **定 $z$ 轴 (Define z-axis):** 根据右手定则，通过 $x$ 和 $y$ 的叉乘得到 $z$ 轴。

*   **构建分子局部坐标系 $\mathfrak{m}$ 和世界坐标系 $\mathfrak{w}$ 之间的变换，记作 $\mathfrak{m} \to \mathfrak{w}$:**
    *   变换包含：
        *   旋转矩阵 $R_{\mathfrak{m} \to \mathfrak{w}} \in \mathbb{R}^{3 \times 3}$，满足 $|R_{\mathfrak{m} \to \mathfrak{w}}| = 1$。
        *   平移向量 $\mathbf{t}_{\mathfrak{m} \to \mathfrak{w}} \in \mathbb{R}^3$。
    *   **旋转矩阵构建 (Rotation Matrix Construction):**
        *   由于基向量 $(x, y, z)$ 已经归一化且相互正交，我们可以直接将它们堆叠形成一个旋转矩阵：$R_{\mathfrak{m} \to \mathfrak{w}} = [x^T, y^T, z^T]$。
    *   **平移向量设置 (Translation Vector Setting):**
        *   平移向量可以设置为 $t_{\mathfrak{m} \to \mathfrak{w}} = v_{\ell_1}$，表示两个坐标系原点之间的位移。

*   **构建片段局部坐标系 $\mathfrak{g}$ 和变换 $\mathfrak{g} \to \mathfrak{w}$:**
    *   **构建过程 (Construction Process):** 类似于分子局部坐标系的构建过程，使用片段中前三个非共线原子来构建**片段局部坐标系**，记作 $\mathfrak{g}$。
    *   **等价变换 (Equivariant Transformation):** 片段局部坐标系也等价于输入分子的旋转和平移变换，可以获得旋转矩阵 $R_{\mathfrak{g} \to \mathfrak{w}}$ 和平移向量 $t_{\mathfrak{g} \to \mathfrak{w}}$，分别表示片段局部坐标系与世界坐标系之间的方向和位移。
    *   在片段局部坐标系下，可以获得原子的局部坐标 $V_{\mathscr{g}_i} \in \mathbb{R}^{q \times 3}$，其中 $i$ 表示第 $i$ 个片段，$q$ 表示片段中的原子数量。

*   **数据存储 (Data Storage):**
    *   将训练集中的每个切分片段保存到字典中。
        *   **键 (Key):** 每个片段的**规范 SMILES**。
        *   **值 (Value):** 原子类型和在相关片段局部坐标系下的原子局部坐标。

*   **推导变换 $\mathfrak{g} \to \mathfrak{m}$ (Derivation of Transformation $\mathfrak{g} \to \mathfrak{m}$):**
    *   将局部坐标系映射到世界坐标系：
        $$T_{\mathfrak{m} \to \mathfrak{w}} = \begin{bmatrix} R_{\mathfrak{m} \to \mathfrak{w}} & t_{\mathfrak{m} \to \mathfrak{w}} \\ \mathbf{0} & 1 \end{bmatrix}, \quad T_{\mathfrak{g} \to \mathfrak{w}} = \begin{bmatrix} R_{\mathfrak{g} \to \mathfrak{w}} & t_{\mathfrak{g} \to \mathfrak{w}} \\ \mathbf{0} & 1 \end{bmatrix} \tag{2}$$
        *   其中 $T \in \mathbb{R}^{4 \times 4}$ 表示齐次变换矩阵，$\mathbf{0} \in \mathbb{R}^{3 \times 1}$ 是零向量。
    *   使用坐标系变换的链式法则得到齐次变换：
        $$\begin{align*} T_{\mathfrak{g} \to \mathfrak{m}} &= T_{\mathfrak{m} \to \mathfrak{w}}^{-1} T_{\mathfrak{g} \to \mathfrak{w}} \\ &= \begin{bmatrix} R_{\mathfrak{m} \to \mathfrak{w}}^T & -R_{\mathfrak{m} \to \mathfrak{w}}^T t_{\mathfrak{m} \to \mathfrak{w}} \\ \mathbf{0} & 1 \end{bmatrix} \begin{bmatrix} R_{\mathfrak{g} \to \mathfrak{w}} & t_{\mathfrak{g} \to \mathfrak{w}} \\ \mathbf{0} & 1 \end{bmatrix} \tag{3} \\ &= \begin{bmatrix} R_{\mathfrak{m} \to \mathfrak{w}}^T R_{\mathfrak{g} \to \mathfrak{w}} & R_{\mathfrak{m} \to \mathfrak{w}}^T (t_{\mathfrak{g} \to \mathfrak{w}} - t_{\mathfrak{m} \to \mathfrak{w}}) \\ \mathbf{0} & 1 \end{bmatrix} \end{align*}$$
    *   提取旋转矩阵和平移向量：
        $$R_{\mathfrak{g} \to \mathfrak{m}} = R_{\mathfrak{m} \to \mathfrak{w}}^T R_{\mathfrak{g} \to \mathfrak{w}}, \quad \mathbf{t}_{\mathfrak{g} \to \mathfrak{m}} = R_{\mathfrak{m} \to \mathfrak{w}}^T (t_{\mathfrak{g} \to \mathfrak{w}} - t_{\mathfrak{m} \to \mathfrak{w}}) \tag{4}$$
    *   保存片段局部坐标系原点到片段原子中心的位移 $t_{\mathfrak{g} \to c(\mathscr{g})}$（在将原子局部坐标从片段局部坐标系转换回世界坐标系时使用）：
        $$t_{\mathfrak{g} \to c(\mathscr{g})} = V_{c(\mathscr{g})}^{\mathfrak{m}} - t_{\mathfrak{g} \to \mathfrak{m}} \tag{5}$$
        *   其中，$c(\mathscr{g})$ 表示任何片段 $\mathscr{g}$ 的中心，$V_{c(\mathscr{g})}^{\mathfrak{m}}$ 是片段中心在分子局部坐标系下的坐标。

#### 3.3. SE(3) 不变的分子片段局部表示 (SE(3)-Invariant Local Representation of Molecular Fragments): 获得片段的**相对位置**和**相对朝向**。

*   **球面坐标表示 (Spherical Coordinate Representation):** 利用函数 $f(\cdot)$ 将片段中心的坐标转换为分子局部坐标系下的球坐标。
    *   对每个片段中心 $\ell_i$（在世界坐标系下坐标为 $v_{\ell_i}$），定义其球坐标：
        $$ \begin{align*} d_i &= \|v_{\ell_i} - v_{\ell_1}\|_2, \\ \theta_i &= \arccos\left(\frac{(v_{\ell_i} - v_{\ell_1}) \cdot z}{d_{\ell_i}}\right), \tag{6} \\ \phi_i &= \operatorname{atan2}((v_{\ell_i} - v_{\ell_1}) \cdot y, (v_{\ell_i} - v_{\ell_1}) \cdot x) \end{align*} $$
        *   其中，$(x,y,z)$ 是分子局部坐标系的基向量，$v_{\ell_1}$ 是分子局部坐标系的原点（第一个片段中心）。
*   **旋转向量表示 (Rotation Vector Representation):** 利用函数 $g(\cdot,\cdot)$。
    *   **原因 (Reason):** 欧几里德空间中的 3D 旋转只有三个自由度，旋转矩阵表示具有冗余信息（3x3矩阵有9个数字，但实际只需三个数字表示三个自由度），并且可能不必要地增加 LMs 的上下文长度。
    *   已有旋转矩阵 $R_{\mathfrak{g} \to \mathfrak{m}}$，提取其旋转角度 $\psi$ 与单位旋转轴 $\mathbf{a} = (a_x, a_y, a_z)$，定义旋转向量：
        $$\mathbf{m} = \psi \mathbf{a} = (m_x, m_y, m_z) \in \mathbb{R}^3$$
        *   **紧凑度 (Compactness):** 3个数值代替9个矩阵元素，同样具备 SE(3) 不变性。
*   **逆映射 (Inverse Mapping):** 从相对表示中重建原始分子。
    *   使用坐标变换将原子坐标从片段局部坐标系转换到世界坐标系（片段 $\to$ 分子 $\to$ 世界）：
        $$t_{\mathfrak{g} \to \mathfrak{m}} = V_c^{\mathfrak{m}}(G) - t_{\mathfrak{g} \to c(G)}, \quad V^{\mathfrak{m}} = V^{\mathfrak{g}} R_{\mathfrak{g} \to \mathfrak{m}}^T + t_{\mathfrak{g} \to \mathfrak{m}}, \quad V^{\mathfrak{w}} = V^{\mathfrak{m}} R_{\mathfrak{m} \to \mathfrak{w}}^T + t_{\mathfrak{m} \to \mathfrak{w}}. \quad (7)$$
    *   其中：
        *   $t_{\mathfrak{g} \to \mathfrak{m}}$：片段局部坐标系到分子局部坐标系的平移向量。
        *   $V^{\mathfrak{g}}$：片段局部坐标系下的原子坐标。
        *   $V^{\mathfrak{m}}$：分子局部坐标系下的原子坐标。
        *   $V^{\mathfrak{w}}$：世界坐标系下的原子坐标。
        *   $R_{\mathfrak{g} \to \mathfrak{m}}^T$：片段局部坐标系到分子局部坐标系的旋转矩阵转置。
        *   $R_{\mathfrak{m} \to \mathfrak{w}}^T$：分子局部坐标系到世界坐标系的旋转矩阵转置。
*   **引理 3.3. 3D 分子图同构的规范序 (Lemma 3.3. *[Canonical Ordering for 3D Molecular Graph Isomorphism]*)**
    *   设 $M = (V, Z)$ 是一个具有节点类型矩阵 $Z$ 和节点坐标矩阵 $V$ 的 3D 分子图。设 $\mathfrak{m}$ 是基于 $L(M)$ 中前三个非共线片段中心构建的 $M$ 的等变局部坐标系，$\mathfrak{g}$ 是基于 $L(G_i)$ 中前三个非共线原子构建的任意片段 $G_i$ 的等变局部坐标系。
    *   $f(\cdot)$ 是我们的函数，它将分子 $M$ 的 3D 坐标矩阵 $V$ 在分子局部坐标系 $\mathfrak{m}$ 下映射到球面表示 $S$。$g(\cdot, \cdot)$ 是将分子局部坐标系 $\mathfrak{m}$ 和片段局部坐标系 $\mathfrak{g}$ 映射到旋转向量 $\mathbf{m}$ 的函数。那么对于任意 3D 变换 $\tau \in SE(3)$，我们有：
        $$f(V) = f(\tau(V)), \quad g(\mathfrak{m}, \mathfrak{g}) = g(\tau(\mathfrak{m}, \mathfrak{g}))$$
    *   给定球面表示 $S = f(V)$ 和旋转向量 $\mathbf{m} = g(\mathfrak{m}, \mathfrak{g})$，存在一个变换 $\tau \in SE(3)$ 使得：
        $$f^{-1}(S) = \tau(V), \quad g^{-1}(\mathbf{m}) = \tau(\mathfrak{m}, \mathfrak{g})$$
    *   这个引理说明了所构建的表示具有 SE(3) 等变性质：
        1.  无论分子如何在 3D 空间中旋转或平移，其球面表示和旋转向量都保持不变。
        2.  从这些表示可以唯一地重构出原始的几何结构（经过某个 SE(3) 变换）。

#### 3.4. Frag2Seq: 片段和几何感知词元化 (Frag2Seq: Fragment and Geometry-Aware Tokenization): 3D 分子片段到 1D 序列的可逆变换。

*   **片段-位置向量 (Fragment-Position Vector):**
    *   给定一个具有 $k$ 个片段的分子 $\mathcal{M}$，我们将每个片段 $\mathscr{g}_{i}$ 的片段向量表示为：
        $$\mathbf{x}_i = [s_i, d_i, \theta_i, \phi_i, m_{xi}, m_{yi}, m_{zi}]$$
        *   其中 $s_i$ 是 $\mathscr{g}_{i}$ 的规范 SMILES 字符串。
    *   按照其 $\ell_{1},\cdots,\ell_{k}$ 连接成一个 1D 序列。
*   **序列化 (Serialization):**
    *   按照片段的规范序 $\ell_1, \ldots, \ell_k$ 连接：
        $$\text{Frag2Seq}(M) = \text{concat}(x_{\ell_1}^*, \ldots, x_{\ell_k}^*)$$
*   **定理 3.4. 双射映射 (Theorem 3.4. *[Bijective Mapping]*)**
    *   对任意两分子 $\mathcal{M}_1, \mathcal{M}_2$，有：
        $$\text{Frag2Seq}(M_1) = \text{Frag2Seq}(M_2) \Longleftrightarrow M_1 \cong_{3D} M_2$$
    *   当坐标值舍入化到小数点后 $b$ 位时，仍保持 $(10^{-b}/2)$ 约束下等价。

---

### 4. 条件训练与生成 (Conditional Training and Generation)

*   **训练目标 (Training Objective):**
    $$\mathcal{L}(U;\mathcal{P}) = \sum_{i} \log p_{\theta}(u_i | u_{i-1}, \dots, u_1;\mathcal{P}) \tag{8}$$
    *   其中：
        *   $U = \{u_1, ..., u_n\}$: 整个分子的 token 序列（注意这里的 $u_i$ 是离散化后的 token，而不是原始的数值向量 $x_i$）。
        *   $\mathcal{P}$: 蛋白质口袋信息。
        *   $p_{\theta}$: 我们的语言模型，参数为 $\theta$。
*   **条件注入 (Conditional Injection):**
    *   使用了预训练的 **ESM-IF1** 模型来提取蛋白质的节点嵌入（特征）。
    *   在每个注意力块中的多头自我注意之后，都在蛋白质节点嵌入和配体 token 嵌入之间添加交叉注意力：其中 $Q$ 来自配体 token 嵌入，$K$ 和 $V$ 来自蛋白质节点嵌入。
*   **生成过程 (Generation Process):**
    *   从一个起始 token 开始，模型自回归地一个一个预测 token，直到生成结束符。
    *   然后，利用 Frag2Seq 的可逆性，将生成的 token 序列解码回 3D 分子结构。

## **数学推导**

:::card
@[pdf height="800px" zoom="80"](https://docs.velvet-notes.org/Frag2Seq-Math.pdf)
:::