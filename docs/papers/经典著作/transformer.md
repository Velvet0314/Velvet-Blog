---
title: Attention Is All You Need
createTime: 2025/05/08 16:54:34
tags:
  - Transformer
  - Self-Attention
permalink: /papers/经典著作/transformer
prev: /papers/经典著作/resnet
next: /papers/经典著作/ddpm
---

## **Transformer 概述**

### **Transformer 解决的问题**

| 🚀 问题类别     | ❌ 传统方法的局限                                 | ✅ Transformer 的解决方案                 |
| -------------- | -------------------------------- | ----------------------------------- |
| **长距离依赖建模** | 🚨 RNN 难以捕捉长距离上下文，存在梯度消失/爆炸问题             | ✨ 采用自注意力机制，直接建立所有词之间的依赖关系           |
| **并行化效率**   | 🚨 RNN 的时序依赖性导致难以并行，训练时间较长                | ✨ 全注意力架构支持高度并行，显著提升训练速度             |
| **模型结构复杂度** | 🚨 复杂的 RNN 结构堆叠（如 LSTM+Attention）增加了计算和设计难度 | ✨ 采用统一的注意力模块（Self-Attention），简化模型架构 |
| **全局建模能力**  | 🚨 注意力机制作为辅助模块，受限于 RNN/CNN 框架，无法充分利用全局信息  | ✨ 纯注意力结构全局化信息建模，灵活捕捉各词之间的相互依赖       |

### **Transformer 的 Tips**

Transformer 的模型结构如下：

<ImageCard
	image="https://image.velvet-notes.org/blog/transformer_structure.png"
	width=65%
	center=true
/>

#### ==**Tip 1：编码器-解码器架构堆叠（Encoder and Decoder Stacks）**=={.note}

##### **Encoder**

**⭐整体结构：**

由 **6层（N=6）完全相同的层堆叠** 而成，每一层都有两个 **子层（sub-layer）**：

==1. **多头自注意力机制（Multi-Head Self-Attention）：**=={.note}
- 输入序列中的每个词（Token）可以与其他词建立依赖关系
- 通过多个注意力头并行计算，可以捕捉==不同的依赖模式（多头学习多表征）=={.important}

==2. **位置编码的全连接前馈网络（Position-wise Fully Connected Feed-Forward Network, FFN）：**=={.note}
- 对每个位置（Token）单独应用相同的前馈神经网络，==通过两个线性层映射到需要的语义空间中=={.important}，其中在输入 Embedding 后进行 **位置编码（Positional Encoding）**

==Encoder 的输入是 **原始序列**=={.important}

**⭐残差连接与层归一化（Layer Normalization）：**

**残差连接（Residual Connection）** —— 每个子层输出都会与输入直接相加，同时对每个子层的输出进行归一化，稳定训练过程：

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

这种设计缓解了梯度消失问题，并使梯度更稳定

**⭐输出维度：** 每一层的输出都保持相同的维度 $d_{model} = 512$，这确保了模型结构的简洁性

##### **Decoder**

**⭐整体结构：**

也由 **6层（N=6）完全相同的层堆叠** 而成，每层有三个子层：

==1. **掩码多头自注意力机制（Masked Multi-Head Self-Attention）：**=={.note}

与Encoder的自注意力类似，但引入了 <b>"掩码"（Mask）</b> 机制

掩码确保了位置 $i$ 的词只能看到它之前 $1 \to i-1$ 的词，避免在训练过程中泄漏未来信息

::: note 关于 Mask
==Self-Attention 是能够看到全部的输入（所有的 $\mathrm{Key}, \mathrm{Value}$），为了保持训练与预测的一致（自回归模型，在时刻 $t$ 只用已生成（或已知）的前序信息去预测下一个词），使用 Mask 只允许其看到 $t$ 时刻之前的信息=={.important}

**⭐Mask 的操作原理：**

由于在自注意力中，$\mathrm{Query}$ 会与每个 $\mathrm{Key}$ 进行计算，但是为了保持训练与预测的一致性，要求当前的 $\mathrm{Key}_t$ 只能取 $\mathrm{Key}1 \to \mathrm{Key}_{t-1}$

Mask 通过对 $\mathrm{Key}_t \cdots$ 以后的 $\mathrm{Key}$ 取一个非常大的负数，使得经 $\mathrm{softmax}$ 后这些结果都是 0
:::

==2. **多头非自注意力（Multi-Head Attention）：**=={.note}
- 将编码器的输出作为 $\mathrm{Key}$ 和 $\mathrm{Value}$，而 $\mathrm{Query}$ 来自解码器的前一层的输出。==这让 Decoder 能获取 Encoder 编码的全局信息（源序列）作为监督=={.important}
- 将编码器输出作为键值，相当于为解码器提供静态的全局记忆（Static Memory），而查询向量则代表动态生成过程中的当前状态。这种"动态查询+静态记忆"的组合，既满足自回归的因果约束，又能充分挖掘源序列信息

==3. **Feed-Forward Network (FFN)：**=={.note}
* 与 Encoder 中的 FFN 相同

**残差连接和层归一化：**

和 Encoder 一致，每个子层都有 **残差连接** 和 **层归一化** ，确保训练稳定

- ==在训练中，Decoder 的输入是 **目标序列** 经过 **右移（shift right）** 的一个序列=={.important}

::: note 关于 Teacher Forcing
使用真实的 **目标序列（Ground Truth）** 作为输入，以加速收敛并稳定训练过程
:::

::: note 关于 Scheduled Sampling
有些任务需要一些 **随机性（noise）**，错误的样本会得到更好的训练效果（泛用性）

如果单纯使用 Teacher Forcing，由于训练和推断时 Decoder 输入不一致导致 **曝光偏差（Exposure Bias）**
:::

- ==在推理中，Decoder 的输入是 **前面生成的序列**=={.important}

::: tip 为什么要 "Shift Right"?
为了易于预测第一个 token（对于第一个 token 需要一个类似于占位符的存在）
:::

Transformer 的训练流程如下（with teacher forcing）：

<ImageCard
	image="https://image.velvet-notes.org/blog/transformer_process.png"
	width=60%
	center=true
/>