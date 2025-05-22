---
title: IPDiff
createTime: 2025/03/17 20:19:32
tags:
  - 配体生成
  - 扩散模型
  - 先验网络
permalink: /projects/分子生成/ipdiff
prev: /projects/分子生成/pocketgen
next: /projects/分子生成/targetdiff
---

## **项目进度**

李青阳：IPDiff 论文阅读一遍

1. 训练流程梳理
2. 采样流程梳理
3. IPNet 网络结构梳理

## **后期规划**

- [x] IPDiff 的代码阅读和实验
- [ ] 基础知识总结 —— 笔记整理 + 思维导图绘制
	- [x] 扩散模型
	- [ ] Transformers
	- [ ] SE3-等变图神经网络
	- [ ] 论文的数学证明

## **IPDiff 的创新点**

1. 在扩散模型的 ==<b>前向扩散</b>=={.note} 和 ==<b>反向去噪</b>=={.note} 两个过程中都明确考虑了<b>口袋配体相互作用</b>
	- 前向扩散：
		提出了**先验偏移（prior-shifting）**：根据口袋结合位点和相应配体分子之间的相互作用来改变正向过程的扩散轨迹
	- 反向去噪：
		设计了**先验条件结合（prior-conditioning）**：通过在先前估计的蛋白质-配体相互作用上条件化配体分子的去噪来增强反向过程
2. IPNet：一个预训练的网络，用于提供上面两点的先验知识

## **IPDiff 理论**

### **IPDiff 原理简述**

### **IPDiff 训练算法流程**

以下用 $\mathcal{M}$ 代表配体分子（molecule），用 $\mathcal{P}$ 代表蛋白质（protein），用 $\mathbf{F}^\mathcal{M}$ 代表配体侧提取的交互特征，用 $\mathbf{F}^\mathcal{P}$ 代表蛋白侧提取的交互特征

:::: steps
1. 输入：
	- **Protein-ligand binding dataset** $\{\mathcal{P}, \mathcal{M}\}_{i=1}^N$：即含有蛋白-配体对以及真实结合构象的信息
	- **可学习的扩散去噪模型** $\phi_{\theta_1}$（负责在扩散的反向去噪过程里对配体进行去噪）
	- **可学习的神经网络** $\psi_{\theta_2}$（负责生成<b>“先验偏移”（prior-shifting）</b>所需的位移向量）
	- **预训练好的“蛋白-配体交互先验网络” IPNet**（负责提取蛋白-配体之间的交互特征，用于在正向与反向过程里融入先验知识）
2. 循环条件：当 $\phi_{\theta_1}$  和 $\psi_{\theta_2}$  还未收敛时，重复以下步骤：

3. 行 2：从训练集 $\{\mathcal{P}, \mathcal{M}\}_{i=1}^N$ 中随机采样一个蛋白-配体对，并将其标记为 $\{\mathbf{X}_0^\mathcal{M}, \mathbf{X}_0^\mathcal{P}\}, \{\mathbf{V}_0^\mathcal{M}, \mathbf{V}_0^\mathcal{P}\}$  
	- $\mathbf{X}_0^\mathcal{M}\in \mathbb{R}^{N_M\times 3}$ 表示配体分子的所有原子的三维坐标 $(x,y,z)$，$\mathbf{V}_0^\mathcal{M}$
	    - $N_M$ 是配体分子的原子数    
	    - $\mathbf{V}_0^\mathcal{M}$ 表示配体分子中所有原子的 **类型信息**（如原子种类、价态、部分电荷等 one-hot 编码形式）
    - $\mathbf{X}_0^\mathcal{P}\in \mathbb{R}^{N_P\times 3}$  表示蛋白质分子的所有原子的三维坐标 $(x,y,z)$，$\mathbf{V}_0^\mathcal{P}$    
	    - $N_P$ 是蛋白质分子的原子数 
	    - $\mathbf{V}_0^\mathcal{P}$ 表示蛋白质分子中所有原子的 **类型信息**（如氨基酸类别、元素类型等）
        
4. 行 3：从扩散时间步的范围 $\{0,1,\dots,T\}$ 中随机采样一个时间步 $t$这样可以在训练时随机地对不同时刻的数据进行去噪学习

5. 行 4：将整个蛋白-配体复合物的 **质心（Center of Mass, CoM）** 平移到原点，保证后续网络处理时的坐标中心一致
	- ==一般在 3D 分子扩散中，为了简化对平移不变性的建模，常把体系的几何中心或质心对齐到原点=={.note}
6. ==行 5：调用预训练的 IPNet 网络，输入 $[[\mathbf{X}_0^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]]$ 和 $[[\mathbf{V}_0^\mathcal{M}, \mathbf{V}_0^\mathcal{P}]]$，得到交互特征 $[[\mathbf{F}_0^\mathcal{M}, \mathbf{F}_0^\mathcal{P}]]$=={.important} 
	- **IPNet** 内部会对蛋白质与配体分别利用 **SE(3)-等变层网络** 或者 **图注意力层网络**，然后再通过蛋白-配体的 cross-attention（或其它融合方式）得到它们之间的交互表示
7. ==行 6：对配体坐标 $\mathbf{X}_0^\mathcal{M}$ 进行 **噪声扰动** 并加入由交互先验得到的坐标平移量 $\mathbf{S}_t^\mathcal{M}$=={.important}
	- 首先从高斯分布 $\mathcal{N}(0,\mathit{\mathbf{I}})$ 采样噪声 $\epsilon$
	- 然后用 $\psi_{\theta_2}(\mathbf{F}_0^\mathcal{M}, t)$ 计算“交互驱动”的位置偏移  $\mathbf{S}_t^\mathcal{M}$ ，其公式是：
	- $$\mathbf{S}_t^\mathcal{M} = \eta \cdot k_t \cdot \psi_{\theta_2}\bigl(\mathbf{F}_0^\mathcal{M}, t\bigr)$$
	- 其中：$\eta$ 是整体缩放系数，$k_t$ 是时间步相关的缩放因子，用于让正向过程在早期或晚期产生不同幅度的偏移
	- 在已有的正向扩散公式中，新的位置就是：
	- $$\mathbf{X}_t^\mathcal{M} = \sqrt{\bar{\alpha}_{\!t}}\,\mathbf{X}_0^\mathcal{M} + \mathbf{S}_t^\mathcal{M} \;+\; \sqrt{\,1-\bar{\alpha}_t\,}\,\epsilon$$
	- 这一步表明，在原本只加高斯噪声的基础上，又额外加上了与蛋白-配体交互相关的 **位置偏移** $\mathbf{S}_t^\mathcal{M}$，使得正向扩散能够“感知”蛋白口袋的特性
8. 行 7：对配体的原子类型信息 $\mathbf{V}_0^\mathcal{M}$ 进行 **Gumbel 噪声扰动** 并得到扰动后的类型信息 $\mathbf{V}_t^\mathcal{M}$ 
	- 这里用到 $g \sim \text{Gumbel}(0,1)$ 以及 $\text{onehot}(\cdot)$ 函数，将连续的信息转成 one-hot 离散类型  
	- 公式里 $\log c^M = \log(\bar{\alpha}_{t} V_0^M + (1 - \bar{\alpha}_{t}/K))$ 表示在原本离散扩散的公式中加入了 Gumbel 采样（与传统的离散扩散略有不同实现细节），以产生离散原子类型的扰动
	- 最后得到新的原子类型信息：
    - $$\mathbf{V}_t^\mathcal{M} = onehot(\text{argmax}_i(g_i + logc_i^\mathcal{M}))$$
9.  行 8：将离散化后的 $\mathbf{V}_t^\mathcal{M}$ 嵌入到向量表征 $\widetilde{\mathbf{H}}_t^{\mathcal{M},0}$；同理，把蛋白原子类型 $\mathbf{V}_0^\mathcal{P}$ 嵌入到 $\widetilde{\mathbf{H}}_t^{\mathcal{P},0}$
	- 这一步是为后续神经网络做输入准备，把离散的原子类型信息转换成可学习的向量表示
	- $\widetilde{\mathbf{H}}_0^{\mathcal{P},0} = \ldots = \widetilde{\mathbf{H}}_T^{\mathcal{P},0}$
10. ==行 9：在将要输入扩散模型去噪网络 $\phi_{\theta_1}$ 之前，将前面提取的蛋白-配体交互特征 $[[\mathbf{F}_0^\mathcal{M}, \mathbf{F}_0^\mathcal{P}]]$ 与刚才的蛋白质-配体的向量表示 $[[\widetilde{\mathbf{H}}_t^{\mathcal{M}}, \widetilde{\mathbf{H}}_t^{\mathcal{P}}]]$ 拼接，得到完整的条件特征  $[[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]]$：=={.important}
    - $$[[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]] \;=\; \mathrm{concat}\left([[\widetilde{\mathbf{H}}_t^{\mathcal{M}}, \widetilde{\mathbf{H}}_t^{\mathcal{P}}]], [[\mathbf{F}_0^\mathcal{M}, \mathbf{F}_0^\mathcal{P}]]\right)$$
11. 行 10：用去噪网络 $\phi_{\theta_1}$ 来预测去噪结果 $\bigl(\hat{\mathbf{X}}_{0|t}^\mathcal{M},\;\hat{\mathbf{V}}_{0|t}^\mathcal{M}\bigr)$，即从噪声状态 $t$ 反推到无噪声状态 0 的配体：
	- $\hat{\mathbf{X}}_{0|t}^\mathcal{M},\;\hat{\mathbf{V}}_{0|t}^\mathcal{M} =\phi_{\theta1}([[\mathbf{X}_t^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]], [[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]])$
	- 其中 $\phi_{\theta_1}$ 是一个包含若干层 SE(3)-等变消息传递的神经网络（应该是与 TargetDiff 相同的，通过 SE(3)-等变层来预测原子的坐标），输入包括 $[[\mathbf{X}_t^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]]$ 和拼接后的蛋白-配体表征 $[[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]]$
12. 行 11：根据网络输出的 $\bigl(\hat{\mathbf{X}}_{0|t}^\mathcal{M},\;\hat{\mathbf{V}}_{0|t}^\mathcal{M}\bigr)$ 与真实 $\bigl(\mathbf{X}_0^\mathcal{M}, \mathbf{V}_0^\mathcal{M}\bigr)$ 计算损失函数 $L$。损失函数的具体形式可参见论文，通常包含位置与类型的去噪误差
13. 行 12：同时对 $\theta_1$ 和 $\theta_2$ 执行反向传播与梯度更新，最小化上一步得到的损失 $L$
	- 注意，**IPNet** 的参数在训练阶段是 **冻结（Freeze Parameters）** 的，只对 $\phi_{\theta_1}$ 与 $\psi_{\theta_2}$ 两部分进行更新
14. 行 13：如果尚未收敛，则回到第 1 步继续迭代；若收敛则停止训练
::::

总结来说，**Algorithm 1** 中最核心的改进点在于：  

- 在 **正向扩散**（行 6）时就已经把蛋白-配体交互特征 $\mathbf{F}_0^\mathcal{M}$ 融入，借由 $\psi_{\theta_2}$ 产生额外的 **位置偏移（shifting）** $\mathbf{S}_t^\mathcal{M}$，使得训练时的正向过程能够感知蛋白质口袋 （也即 **prior-shifting**） 
- 在 **反向去噪**（行 9）时通过拼接 $[[\widetilde{\mathbf{H}}_t^{\mathcal{M}}, \widetilde{\mathbf{H}}_t^{\mathcal{P}}]], [[\mathbf{F}_0^\mathcal{M}, \mathbf{F}_0^\mathcal{P}]]$（也即 **prior-conditioning**）来告诉去噪网络在每个时刻如何利用蛋白-配体先验信息

<ImageCard
	image="https://s21.ax1x.com/2025/03/19/pEweBZR.png"
	width=85%
	center=true
/>

### **IPDiff 采样算法流程**

#### **输入：**
  - 目标蛋白结合位点 $\mathcal{P}$（包括其 ==**所有原子的 3D 坐标**=={.note} 和 ==**原子的类型信息**=={.note}）
  - 训练好的扩散去噪模型 $\phi_{\theta_1}$，网络 $\psi_{\theta_2}$ 和 预训练好的提供交互先验知识的 IPNet

#### **输出：**
  - 生成的 3D 配体分子 $\mathcal{M}$，使其能与给定蛋白质口袋 $\mathcal{P}$ 紧密结合

#### **主要步骤：**
:::: steps
1. 行 1：采样出生成的配体分子中原子的数量 $N_M$
	- 确定后续采样时需要生成多少个原子坐标和类型信息
2. 行 2：和训练阶段相同，将蛋白的质心移动到坐标原点

3. 行 3：先随机初始化配体的 3D 坐标 $\mathbf{X}_T^\mathcal{M}$ 和类型信息 $\mathbf{V}_T^\mathcal{M}$
	- 相当于在最初时刻 $T$ 拥有了“全噪声”状态的配体
	- 也可以在采样时让坐标或类型的噪声分布更大，以保证生成时的多样性
4. 行 4：初始化交互先验相关的向量 $[[\mathbf{F}_{0|T+1}^\mathcal{M}, \mathbf{F}_{0|T+1}^P]] = \mathbf{O}$ 以及初始位移量 $\mathbf{S}_T^\mathcal{M} = \mathbf{O}$
	- 因为在最开始我们对生成的配体没有任何确定信息
5. 行 5：进入一个从 $t = T$ 到 $1$ 的循环，每次迭代将噪声状态 $t$ 逐步去噪为 $t -1$
	- 整个过程与训练时的反向过程类似，但要注意我们现在是**真正地**在逐步生成分子，而不是对已知分子进行扰动
	- $\textbf{for}\ t\ \text{in}\ T, \ldots, 1\ \textbf{do}$
6. 行 6：将配体的原子类型信息 $\mathbf{V}_t^\mathcal{M}$ 嵌入到向量表征 $\widetilde{\mathbf{H}}_t^{\mathcal{M},0}$；同理，把蛋白质的原子类型信息 $\mathbf{V}_0^\mathcal{P}$ 嵌入到 $\widetilde{\mathbf{H}}_t^{\mathcal{P},0}$
   - $\widetilde{\mathbf{H}}_0^{\mathcal{P},0} = \ldots = \widetilde{\mathbf{H}}_T^{\mathcal{P},0}$
7. ==行 7：得到先验条件特征  $[[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]]$：=={.important}
	- $[[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]] = \mathrm{concat}\bigl([[\widetilde{\mathbf{H}}_t^\mathcal{M}, \widetilde{\mathbf{H}}_t^\mathcal{P}]], [[\mathbf{F}_{0|t+1}^\mathcal{M}, \mathbf{F}_{0|t+1}^\mathcal{P}]]\bigr)$
	- 这一步与训练时行 9 中的 **prior-conditioning** 对应
	- 将上一时间步估计的蛋白-配体先验交互表示与当前时间步的基本信息拼接，得到当前时间步的具有先验知识的信息
8. 行 8：调用扩散去噪网络 $\phi_{\theta_1}$ 预测 $\bigl(\hat{\mathbf{X}}_{0|t}^\mathcal{M}, \hat{\mathbf{V}}_{0|t}^\mathcal{M}\bigr)$：
	- $\hat{\mathbf{X}}_{0|t}^\mathcal{M}, \hat{\mathbf{V}}_{0|t}^\mathcal{M} = \phi_{\theta1}([[\mathbf{X}_t^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]], [[\mathbf{H}_t^{\mathcal{M},0}, \mathbf{H}_t^{\mathcal{P},0}]])$
9.  ==行 9：从位移后的后验概率 $p_{\theta_1}(\mathbf{X}_{t-1}^\mathcal{M} \mid \mathbf{X}_t^\mathcal{M}, \mathbf{X}_0^\mathcal{P}, \mathbf{F}_{0|t+1}^\mathcal{M})$ 中采样，得到 $\mathbf{X}_{t-1}^\mathcal{M}$=={.important}
	- $z \sim \mathcal{N}(0, \boldsymbol{I})$
	- $\mathbf{S}_{t-1}^{\mathcal{M}} = \eta \cdot k_{t-1} \cdot \psi_{\theta 2}(\mathbf{F}_{0|t+1}^{\mathcal{M}}, t-1)$ 是 $t - 1$ 时刻的位移量
	- $\mathbf{X}_{t-1}^{\mathcal{M}} = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{\mathbf{X}}_{0|t}^{\mathcal{M}} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}(\mathbf{X}_t^{\mathcal{M}} - \mathbf{S}_t^{\mathcal{M}}) + \mathbf{S}_{t-1}^{\mathcal{M}} + \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}\beta_t z$
10. 行 10：从后验概率 $q(\mathbf{V}_{t-1}^{\mathcal{M}} \mid \hat{\mathbf{V}}_{0|t}^{\mathcal{M}}, \mathbf{V}_{t}^{\mathcal{M}}, \mathbf{V}_{0}^{\mathcal{P}})$ 中采样，得到 $\mathbf{V}_{t-1}^\mathcal{M}$

11. ==行 11：再调用 IPNet，输入当前去噪后结果 $[[\hat{\mathbf{X}}_{0|t}^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]]$，$[[\hat{\mathbf{V}}_{0|t}^\mathcal{M}, \mathbf{V}_0^\mathcal{P}]]$，得到新的交互特征，更新给下一轮迭代使用=={.important}
	- $[[\mathbf{F}_{0|t}^\mathcal{M}, \mathbf{F}_{0|t}^\mathcal{P}]] =\text{IPN}\text{\scriptsize{ET}}([[\hat{\mathbf{X}}_{0|t}^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]], [[\hat{\mathbf{V}}_{0|t}^\mathcal{M}, \mathbf{V}_0^\mathcal{P}]])$

12. 行 12：循环直到 $t=1$ 结束，输出最终去噪完成的 $[[\mathbf{X}_0^\mathcal{M}, \mathbf{V}_0^\mathcal{M}]]$，即生成的配体分子
	- $\textbf{end for}$
::::

<ImageCard
	image="https://s21.ax1x.com/2025/03/19/pEweDd1.png"
	width=85%
	center=true
/>

## **IPDiff 代码**

### **代码解读：**[Velvet0314/IPDiff at 4LearnOnly](https://github.com/Velvet0314/IPDiff/tree/4LearnOnly)

### **环境安装 Tips**

- 运行环境同 TargetDiff

### **训练流程**

主要代码在 `train.py`和`molopt_score_model.py`中，大致流程与 TargetDiff 相同

训练算法核心在 `molopt_score_model.py` 的函数 `get_diffusion_loss` 中

:::: steps
1. 新定义了 IPNet 先验知识辅助后的配体和蛋白质
   ```python
   hbap_ligand = None
   hbap_protein = None
   ```
2. line 5：从 IPNet 中提取特征 —— 得到了添加了先验知识的蛋白质和配体
   ```python
   # line 5
   hbap_ligand, hbap_protein = net_cond.extract_features(gt_ligand_pos, gt_protein_pos, gt_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
   ```
3. line 6：先验知识生成**偏移（shift）**，在原本的扩散公式上加上偏移，得到扰动后的配体位置
	- 注意网络 $\psi_{\theta2}$
   ```python
   # line 6
   shift_cond_t = torch.cat([hbap_ligand, time_step[batch_ligand].unsqueeze(-1)], -1)
   # 注意 shift_t_mlp_pos 这个网络 —— ψ_θ_2
   # shift_t_mlp_pos 是一个全连接网络，输入是 128 维的向量，输出是 3 维的向量
   # BAPNet 提取的特征(hbap_ligand)是高维的语义特征(128维)
   # 但分子构象需要的是 3D 空间中的几何指导
   # 线性层将高维特征压缩映射到具体的 xyz 坐标偏移量
   shift_cond_t = self.shift_t_mlp_pos(shift_cond_t)
   # 配体在 t 时刻经过 先验偏移 和 扰动 后的三维坐标位置
   ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise + k_t_pos * shift_cond_t
   ```
4. line 9：将扩散模型原本的特征与 IPNet 提取的特征进行结合
   ```python
   # line 9
   if hbap_protein is None:
   	hbap_protein = torch.zeros([h_protein.shape[0], self.cond_dim]).to(h_protein.device)
   if hbap_ligand is None:
   	hbap_ligand = torch.zeros([init_ligand_h.shape[0], self.cond_dim]).to(init_ligand_h.device)

   # 将扩散模型原本的特征与 IPNet 提取的特征进行结合
   h_protein = self.emb_mlp(torch.cat([h_protein, hbap_protein], dim=1))
   init_ligand_h = self.emb_mlp(torch.cat([init_ligand_h, hbap_ligand], dim=1))

   # 这段代码给蛋白质和配体特征向量添加了节点类型指示器（node indicator）
   # 目的是明确区分蛋白质原子和配体原子
   # 节点类型识别：通过添加一个额外的特征维度（蛋白质为0，配体为1），模型可以明确区分两种不同类型的原子
   if self.config.node_indicator:
   	h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
   	init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

   h_all, pos_all, batch_all, mask_ligand, _ =compose_context(
   	h_protein=h_protein,
   	h_ligand=init_ligand_h,
   	pos_protein=protein_pos,
   	pos_ligand=init_ligand_pos,
   	batch_protein=batch_protein,
   	batch_ligand=batch_ligand,
   )
   ```
::::

### **采样流程**

主要代码在 `sample_split.py`和 `molopt_score_model.py` 中，大致流程与 TargetDiff 相同

采样算法核心在 `molopt_score_model.py` 的函数 `sample_diffusion` 中

:::: steps
1. line 7：信息嵌入与先验知识结合都在函数 `forward` 中，也就是 line 8 的预测

2. line 8：调用前向传播进行**预测（predict）**
   ```python
   # 从 t = T 开始，调用前向传播进行去噪，预测 t-1 时刻的配体3D坐标位置和类型信息
   # 因为预测，所以要调用 forward
   # line 8
   preds = self(
   	protein_pos=protein_pos,
   	protein_v=protein_v,
   	batch_protein=batch_protein,
   	init_ligand_pos=ligand_pos,
   	init_ligand_v=ligand_v,
   	batch_ligand=batch_ligand,
   	time_step=t,
   	hbap_protein=hbap_protein,
   	hbap_ligand=hbap_ligand
   )
   ```
3. line 9：从位移后的后验概率 $p_{\theta_1}(\mathbf{X}_{t-1}^\mathcal{M} \mid \mathbf{X}_t^\mathcal{M}, \mathbf{X}_0^\mathcal{P}, \mathbf{F}_{0|t+1}^\mathcal{M})$ 中采样，得到 $\mathbf{X}_{t-1}^\mathcal{M}$
   ```python
   # line 9
   pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, t_minus1=t_minus1, batch=batch_ligand, shift=shift_cond_t, shift_minus1=shift_cond_t_minus1)
   pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
   nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

   ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
   ligand_pos = ligand_pos_next

   gt_protein_pos = protein_pos.detach()
   gt_protein_v = protein_v.detach()
   gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
   gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)
   pred_ligand_pos = pos0_from_e.detach()
   pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)
   ```
4. line 11：再调用 IPNet，输入当前去噪后结果 $[[\hat{\mathbf{X}}_{0|t}^\mathcal{M}, \mathbf{X}_0^\mathcal{P}]]$，$[[\hat{\mathbf{V}}_{0|t}^\mathcal{M}, \mathbf{V}_0^\mathcal{P}]]$，得到新的交互特征，更新给下一轮迭代使用
   ```python
   # line 11
   hbap_ligand, hbap_protein = net_cond.extract_features(pred_ligand_pos, gt_protein_pos, pred_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
   hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()
   ```
::::

### **IPNet**

详解参见仓库中的 `bapnet.py`

## **IPDiff 数学推导**

## **一些疑问**

1. 什么是 SBDD？
	基于结构的药物设计 (Structure‐based drug design)
2. 代码
3. 数学推导