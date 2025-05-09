---
title: DDPM
createTime: 2025/04/06 11:20:15
tags:
  - DDPM
permalink: /projects/经典著作/ddpm
prev: /projects/经典著作/transformer
next: {}
---

## **DDPM 概述**

### **DDPM 解决的问题**

| 🚀 问题类别          | ❌ 传统方法的局限                                | ✅ DDPM 的解决方案                                |
|-------------------------|----------------------------------------------------|-----------------------------------------------------|
| **生成模型的质量与稳定性**  | 🚨 GANs 训练不稳定，易出现模式崩溃<br>🚨 VAEs 生成质量有限，样本模糊 | ✨​​ 非对抗性训练框架，避免 GANs 的不稳定性<br>✨​​ 逐步“加噪-去噪”过程，生成高质量样本 |
| **复杂数据分布的建模**      | 🚨 GANs、VAEs、Flow 难以精确建模复杂数据分布<br>🚨 高分辨率图像生成困难 | ✨​​ 前向过程：数据逐步扰动为高斯噪声（马尔可夫链）<br>✨​​ 反向过程：去噪恢复数据分布 |
| **生成过程的可控性与渐进性** | 🚨 常见生成模型生成结果一成不变，缺乏细节控制<br>🚨 很难逐步提升生成样本质量 | ✨​​ 采用渐进式“有损压缩”视角<br>✨​​ 类似自回归模型的解码，允许逐步细化生成结果 |


### **DDPM 的 Tips**

#### ==**Tip 1：重参数化**=={.note}

**🎯目的：**  
==将**不可微**的采样操作变为**可微**的确定性变换，从而支持**反向传播**=={.note}

**💡基本思想：**  
将随机变量的采样过程分解为：
- **确定性部分**：由模型参数决定的可微函数  
- **随机性部分**：从固定、简单分布（如标准正态）中采样的噪声变量

**高斯分布中的重参数化形式：**  
$$
X = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

**可微性分析：**  
$$
\frac{\partial X}{\partial \mu} = 1, \quad \frac{\partial X}{\partial \sigma} = \epsilon
$$  

进而支持对 $\mu$ 和 $\sigma$ 的梯度计算

| 项目 | **无重参数化** | **有重参数化** |
|------|----------------|----------------|
| **采样方式** | 直接从分布 $\mathcal{N}(\mu, \sigma^2)$ 中采样 $x$ | 从 $\epsilon \sim \mathcal{N}(0,1)$ 采样，再计算 $x = \mu + \sigma \cdot \epsilon$ |
| **随机变量是否可微** | ❌ 不可微（采样操作断开了计算图） | ✅ 可微（采样来自固定分布，变换是可微函数） |
| **是否支持反向传播** | ❌ 不支持，梯度无法传播到 $\mu$, $\sigma$ | ✅ 支持，梯度可传递到分布参数 |
| **是否能用于梯度优化** | ❌ 不可直接优化概率模型参数 | ✅ 可使用梯度下降等方法优化分布参数 |
| **常见用途** | 传统的随机采样、模拟等 | 变分自编码器（VAE）、可微分采样过程 |


#### ==**Tip 2：任意时刻的 $\mathbf{x}_t$ 可以由 $\mathbf{x}_0$ 与 $\beta$ 表示**=={.note}

在正向过程中，数据 $\mathbf{x}_0$ 被逐步加入噪声，形成 $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$

借由该特点（$\mathrm{Eq.(4)}$）得知：==**任意时刻的状态 $\mathbf{x}_t$**，可以不通过逐步采样，而是**直接**由原始数据 $\mathbf{x}_0$ 和噪声调节参数序列 $\{\beta_t\}$ 推出：=={.note}

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \cdot \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$

**🎯意义：**
- 避免逐步执行 $t$ 次采样（**即一行公式即可模拟 $t$ 时刻的加噪结果**）
- 有利于训练过程中的 **目标重构 loss 构造**

#### ==**Tip 3：简化的加权变分下界作为损失函数**=={.note}

将原始的变分下界简化为一个统一的 MSE 损失项：

::: center
==$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right]$=={.note}
:::

## **DDPM 原理简述**

**DDPM（Denoising Diffusion Probabilistic Models，后简称扩散模型）** 的工作流程如下图所示：

<ImageCard
	image="https://s21.ax1x.com/2025/04/06/pEckx4s.png"
	width=85%
	center=true
/>

简要地讲，扩散模型是一种参数化的 **马尔可夫链（parameterized Markov chain）**，通过 **变分推断（variational inference）** 进行训练，以在有限时间内生成与数据分布相匹配的样本

模型通过学习 **正向扩散过程**（即马尔可夫链从原始数据逐步加噪直至信号被破坏）的转移规则，实现对这一过程的**逆转**（得到反向扩散过程，sample）

如果在正向扩散过程中，每一步添加的噪声都是 **小量的高斯噪声**，那么反向的采样过程也可以被建模为 **条件高斯分布（Conditional Gaussian）**

### **DDPM 数学推导**

#### $\mathrm{Eq.}(1)$ **推导**

扩散模型是一个 **潜变量模型（latent variable models）**，其形式为：

$$
p_\theta(\mathbf{x}_0) := \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}
$$

其中，$\mathbf{x}_1, \ldots, \mathbf{x}_T$ 是与数据 $\mathbf{x}_0$ 维度相同的潜变量，$\mathbf{x}_0 \sim q(\mathbf{x}_0)$

联合分布 $p_\theta(\mathbf{x}_{0:T})$ 被称为 **反向过程（reverse process）**，它被定义为一个马尔可夫链，其高斯转移由模型学习得出，并从 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ 开始：

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod\limits_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t), \qquad p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1} ; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t)) \tag{1}
$$

::: tip 边缘概率密度
对于连续型随机变量 $(X,Y)$，设它的概率密度为 $f(x,y)$

其关于 $X$ 的边缘概率密度为：

$$
f_X(x)=\int_{-\infty}^\infty f(x,y)dy
$$

同理，关于 $Y$ 的边缘概率密度为：

$$
f_Y(y)=\int_{-\infty}^\infty f(x,y)dx
$$
:::

<Card title="推导过程">
⭐推导：

$$
p_\theta(\mathbf{x}_0) := \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}
$$

$\mathbf{x}_T$ 表示纯高斯噪声，$\mathbf{x}_0$ 表示生成的样本

$p_\theta(\mathbf{x}_0)$ 表示我们最终从扩散模型生成的数据 $\mathbf{x}_0$ 的概率分布

$\theta$ 代表的是**模型的可训练参数**，通常是用于参数化神经网络的权重

由于整个扩散过程拆分成了马尔科夫链，现有 $\mathbf{x}_0,\cdots,\mathbf{x}_T$ 共 $T+1$ 个随机变量，其联合概率密度为 $p_\theta(\mathbf{x}_0,\mathbf{x}_1,\cdots,\mathbf{x}_T)$，在论文中简写为 $p_\theta(\mathbf{x}_{0:T})$

$p_\theta(\mathbf{x}_0)$ 即为联合概率密度 $p_\theta(\mathbf{x}_{0:T})$ 中关于 $\mathbf{x}_0$ 的边缘概率密度

由边缘概率密度的定义：

$$
p_{\theta}(\mathbf{x}_0) = \int p_{\theta}(\mathbf{x}_{0:T}) d\mathbf{x}_1 d\mathbf{x}_2 \ldots d\mathbf{x}_T = \int p_{\theta}(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
$$
</Card>

::: tip 概率的乘法公式
一般，设 $A_1,A_2,\cdots,A_n$ 为 $n$ 个事件，$n \geq 2$，且 $P(A_1,A_2,\cdots,A_{n-1}) \gt 0$，则有:
 
$$
P(A_1A_2\cdots A_n) = P(A_n | A_1A_2\cdots A_{n-1}) P(A_{n-1} | A_1A_2\cdots A_{n-2}) \cdots P(A_2 | A_1) P(A_1)
$$
:::

::: note 马尔可夫链
随机过程 $\{X_n, n = 0, 1, 2, \dots\}$ 称为马尔可夫链，若随机过程在某一时刻的随机变量 $X_n$ 只取有限或可列个值（比如非负整数集合，若不另作说明，以集合 $S$ 表示），并且对于任意的 $n \geq 0$，及任意状态 $i, j, i_0, i_1, \dots, i_{n-1} \in S$，有： 

$$
P(X_{n+1} = j | X_0 = i_0, X_1 = i_1, \dots, X_n = i) = P(X_{n+1} = j | X_n = i)
$$

其中，$X_n = i$ 表示过程在时刻 $n$ 处于状态 $i$；$S$ 为该过程的状态空间
:::

<Card title="推导过程">
⭐推导：

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod\limits_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)
$$

联合分布 $p_\theta(\mathbf{x}_{0:T})$ 被定义为一个马尔可夫链

$$
\begin{align*}
p_\theta(\mathbf{x}_{0:T}) &= p(\mathbf{x}_T) p_\theta(\mathbf{x}_{T-1}|\mathbf{x}_T) p_\theta(\mathbf{x}_{T-2}|\mathbf{x}_{T}\mathbf{x}_{T-1}) \cdots p_\theta(\mathbf{x}_0|\mathbf{x}_{1:T-1}) \quad (\text{乘法公式反向分解}) \\[5pt]
&= p(\mathbf{x}_T) p_\theta(\mathbf{x}_{T-1}|\mathbf{x}_T) p_\theta(\mathbf{x}_{T-2}|\mathbf{x}_{T-1}) \cdots p_\theta(\mathbf{x}_0|\mathbf{x}_1) \quad (\text{马尔可夫链定义}) \\[5pt]
&= p(\mathbf{x}_T) \prod\limits_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)
\end{align*}
$$

$p_\theta(\mathbf{x}_{0:T})$ 无 $\theta$ 角标，是因为它代表扩散过程的 **固定初始噪声分布**，不涉及可学习参数。模型的参数化仅作用于反向步骤 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$

转移概率的概率密度函数 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1} ; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t))$ 是关于 $\mathbf{x}_{t-1}$ 的一个高斯分布，其均值 $\mu_{\theta}(\mathbf{x}_t, t)$ 与 $\Sigma_{\theta}(\mathbf{x}_t, t)$ 是关于 $\mathbf{x}_t,t$ 的函数，其值通过学习得到

---
⭐解释：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1} ; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t))
$$

为什么 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 是一个高斯分布？ 

由参考文献 $[53]$，大多数扩散模型的正向过程与反向过程可由同一个函数形式描述

正向过程是由人为逐步添加小量的高斯噪声得到的一个高斯分布，故反向过程应与正向过程同为高斯分布
</Card>

#### $\mathrm{Eq.}(2)$ **推导**

扩散模型与其他潜变量模型的区别是：

近似后验分布 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ ，即 **正向过程（forward process）**（或是 **扩散过程（diffusion process）**），被固定为了一个马尔可夫链

该过程通过**逐步添加高斯噪声**，使得数据从真实分布逐渐扩散到一个标准正态分布

噪声的方差由一个**预定义的调度参数** $\beta_1,\cdots,\beta_T$ 控制：

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod\limits_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t - 1}), \qquad q(\mathbf{\mathbf{x}}_t | \mathbf{\mathbf{x}}_{t - 1}) = \mathcal{N}(\mathbf{\mathbf{x}}_{t} ; \sqrt{1-\beta_t}\mathbf{\mathbf{x}}_{t-1}, \beta_t \mathbf{I}) \tag{2}
$$

::: tip 条件概率公式
$$
P(B | A) = \frac{P(AB)}{P(A)}
$$
:::

<Card title="推导过程">
⭐推导：

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod\limits_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t - 1})
$$

$$
\begin{align*}
q(\mathbf{x}_{1:T} | \mathbf{x}_0) &= \dfrac{q(\mathbf{x}_{0:T})}{q(\mathbf{x}_0)} \\[8pt]
&= \frac{q(\mathbf{x}_0)q(\mathbf{x}_1 | \mathbf{x}_0)q(\mathbf{x}_2 | \mathbf{x}_1 \mathbf{x}_0)\cdots q(\mathbf{x}_T | \mathbf{x}_{0:T - 1})}{q(\mathbf{x}_0)} \\[5pt]
&= \frac{q(\mathbf{x}_0)q(\mathbf{x}_1 | \mathbf{x}_0)q(\mathbf{x}_2 | \mathbf{x}_1)\cdots q(\mathbf{x}_T | \mathbf{x}_{T - 1})}{q(\mathbf{x}_0)} \\[5pt]
&= q(\mathbf{x}_1 | \mathbf{x}_0)q(\mathbf{x}_2 | \mathbf{x}_1)\cdots q(\mathbf{x}_T | \mathbf{x}_{T - 1}) \\
&= \prod\limits_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t - 1})
\end{align*}
$$

---
⭐解释：

$$
q(\mathbf{\mathbf{x}}_t | \mathbf{\mathbf{x}}_{t - 1}) = \mathcal{N}(\mathbf{\mathbf{x}}_{t} ; \sqrt{1-\beta_t}\mathbf{\mathbf{x}}_{t-1}, \beta_t \mathbf{I})
$$

正向过程的分布是人为确定为上述形式的
</Card>

#### $\mathrm{Eq.}(3)$ **推导**

训练是通过优化负对数似然的常规 **变分下界（Evidence Lower Bound, ELBO）** 来进行的：

$$
\mathbb{E}[-\log p_\theta(\mathbf{x}_0)] \leq \mathbb{E}_q\left[-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] = L \tag{3}
$$

::: tip Jensen 不等式
Jensen 不等式适用于 **凹函数** $f(x)$，即对于任意随机变量 $X$ 和其概率分布 $p(x)$：

$$
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
$$

当 $f(x)$ 是 **凸函数** 时，方向相反
:::

::: note 期望
设连续型随机变量 $X$ 的概率密度为 $f(x)$，若积分  

$$
\int_{-\infty}^{+\infty} x f(x)dx
$$ 

绝对收敛，则称积分 $\int_{-\infty}^{+\infty} x f(x) dx$ 的值为随机变量 $X$ 的数学期望，记为 $E(X)$，即 

$$
E(X) = \int_{-\infty}^{+\infty} x f(x) dx
$$
:::

<Card title="推导过程">
⭐推导：

$$
\mathbb{E}[-\log p_\theta(\mathbf{x}_0)] \leq \mathbb{E}_q\left[-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right]
$$

==目标函数：$\mathbb{E} [- \log p_\theta(\mathbf{x}_0)]$=={.note}

最小化 **负对数似然（Negative Log Likelihood, NLL）**，即 **最大化数据的对数似然** $\mathbb{E} [\log p_\theta(\mathbf{x}_0)]$

- $p_\theta (\mathbf{x}_0)$ 表示我们最终从扩散模型生成的数据 $\mathbf{x}_0$ 的概率分布     
- 直接最小化它通常是不可行的，因为计算 $p_\theta(\mathbf{x}_0)$ 需要求解复杂的积分
- 这里是最大似然估计的概念补充

==优化策略：**变分推断**（使用变分推断引入上界来优化）=={.note}

引入一个辅助分布 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$（即近似后验分布，正向扩散过程）进行变分推断来帮助估计对数似然

由 $\mathrm{Eq.(1)}$ : $\log p_\theta(\mathbf{x}_0) = \log \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$，借由辅助分布 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 得到：

$$
\log p_\theta(\mathbf{x}_0) = \log \int q(\mathbf{x}_{1:T} | \mathbf{x}_0) \dfrac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \, d\mathbf{x}_{1:T}
$$

我们设：

$$
X = \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}
$$

那么原式就变成：

$$
\log p_\theta(\mathbf{x}_0) = \log \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ X \right]
$$

::: tip 注
这个期望的下标 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 表示关于随机变量 $\mathbf{x}_{1:T}$ 的期望，其中这些变量的分布由 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 给出 

$\mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}$ 是相对于 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$
计算的期望，定义如下：
 
$$
\mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} [f(\mathbf{x}_{1:T})] = \int q(\mathbf{x}_{1:T} | \mathbf{x}_0) f(\mathbf{x}_{1:T}) d\mathbf{x}_{1:T}
$$

这里，我们的函数是：

$$
f(\mathbf{x}_{1:T}) = \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}
$$

所以：

$$
\mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right] = \int q(\mathbf{x}_{1:T} | \mathbf{x}_0) \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \, d\mathbf{x}_{1:T}
$$
:::

因为 $\log(x)$ 是凹函数，我们可以对上式使用 $\mathrm{Jensen}$ 不等式：

$$
\log \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ X \right] \geq \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ \log X \right]
$$

::: note
因为对数函数 $\log(x)$ 是凹函数，由 Jensen 不等式：

$$
\log \mathbb{E}[X] \geq \mathbb{E}[\log X]
$$

这个结论很关键，它意味着：如果我们有一个期望的形式 $\mathbb{E}[X]$，对它取对数后总是大于等于对数的期望
:::

代入 $X = \dfrac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}$，我们得到：

$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right]
$$

这就是 **ELBO（变分下界）**，也是变分推断的核心结论

由 $\mathrm{Eq.(1)(2)}$ 展开期望中的对数项：

$$
\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}= \log p(\mathbf{x}_T) + \sum\limits_{t\geq1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}
$$

最后得到需要优化的 $L$
</Card>

#### $\mathrm{Eq.}(4)$ **推导**

正向过程的方差 $\beta_t$ 可以通过 **重参数化（reparameterization）** 进行学习，也可以作为超参数保持不变。而反向过程的表达能力部分通过在 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 中选择高斯条件分布来确保，因为当 $\beta_t$ 较小时，两个过程具有相同的函数形式。正向过程的一个显著特点是，它允许在任意时间步 $t$ 对 $\mathbf{x}_t$ 进行封闭形式的采样：记 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，我们有：

$$
q(\mathbf{\mathbf{x}}_t | \mathbf{\mathbf{x}}_0) = \mathcal{N}(\mathbf{\mathbf{x}}_{t} ; \sqrt{\bar{\alpha_t}}\mathbf{\mathbf{x}}_0, (1-\bar{\alpha_t}) \mathbf{I}) \tag{4}
$$

::: tip 重参数化

**重参数化（Reparameterization）** 是一种数学技巧，用于将一个随机变量的采样过程分解为：

1. 确定性部分（可微的参数化变换）

2. 随机性部分（来自一个固定、简单的分布）

其核心目的是让随机变量的生成过程对参数可微，从而支持基于梯度的优化（如深度学习中的反向传播）

假设我们有一个随机变量 $\mathbf{x}$，它服从某个参数化分布（如高斯分布 $\mathbf{x} \sim \mathcal{N}(\mu, \sigma^2)$）

直接采样 $\mathbf{x}$ 是不可微的（因为采样是一个随机操作，无法计算梯度）

重参数化的思路：  

- 将采样过程重新表述为一个由噪声变量和模型参数决定的确定性函数。我们不再直接从分布中采样，而是从一个简单且固定的分布（如标准正态分布）中采样一个噪声变量，然后通过一个确定性的变换函数来计算采样值

- 然后通过一个**确定性变换** $g(\theta, \epsilon)$ 生成 $X$，使得 $X$ 仍然服从目标分布，但梯度可以计算

在高斯分布的重参数化中，确定性变换 $g(\theta, \epsilon)$是将参数 $\theta = (\mu, \sigma)$ 和基础噪声 $\epsilon$ 映射到目标随机变量 $X$ 的数学表达式。具体形式为：

$$
X = g(\theta, \epsilon) = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

高斯分布的性质决定了其重参数化可以通过简单的线性变换实现：  

- 平移（$+\mu$）：调整均值  

- 缩放（$\times \sigma$）：调整方差  

- 变换后的 $X$ 仍严格服从 $\mathcal{N}(\mu, \sigma^2)$，因为：  

$$
\mathbb{E}[X] = \mu + \sigma \cdot \mathbb{E}[\epsilon] = \mu, \quad \mathrm{Var}(X) = \sigma^2 \cdot \mathrm{Var}(\epsilon) = \sigma^2
$$

重参数化的核心优势是可微性。对 $\theta = (\mu, \sigma)$ 的梯度为：  

$$
\frac{\partial X}{\partial \mu} = 1, \quad \frac{\partial X}{\partial \sigma} = \epsilon
$$  

梯度可通过反向传播计算，而 $\epsilon$ 被视为常量（因其来自固定分布）
:::

<Card title="推导过程">
⭐推导：

$$
q(\mathbf{\mathbf{x}}_t | \mathbf{\mathbf{x}}_0) = \mathcal{N}(\mathbf{\mathbf{x}}_{t} ; \sqrt{\bar{\alpha_t}}\mathbf{\mathbf{x}}_0, (1-\bar{\alpha_t}) \mathbf{I})
$$

由 $\mathrm{Eq}.(2)$：$q(\mathbf{\mathbf{x}}_t | \mathbf{\mathbf{x}}_{t - 1}) = \mathcal{N}(\mathbf{\mathbf{x}}_{t} ; \sqrt{1-\beta_t}\mathbf{\mathbf{x}}_{t-1}, \beta_t \mathbf{I})$，利用重参数化展开得到：

$$
\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \quad \mathrm{where} \quad \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})
$$

其中记 $\alpha_t = 1-\beta_t$

对 $\mathbf{\mathbf{x}}_{t-1}$ 继续递推展开：

$$
\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\beta_{t-1}}\epsilon_{t-2}
$$

将 $\mathbf{\mathbf{x}}_{t-1}$带入 $\mathbf{\mathbf{x}}_t$ 的表达式中：

$$
\mathbf{x}_t = \sqrt{\alpha_t}\,\Bigl(\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{\beta_{t-1}}\,\epsilon_{t-2}\Bigr)+ \sqrt{\beta_t}\,\epsilon_{t-1}
= \sqrt{\alpha_t\,\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{\alpha_t}\,\sqrt{\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\,\epsilon_{t-1}
$$

对 $\mathbf{\mathbf{x}}_{t-2}$ 继续递推展开：

$$
\mathbf{x}_{t-2} = \sqrt{\alpha_{t-2}}\mathbf{x}_{t-3} + \sqrt{\beta_{t-2}}\epsilon_{t-3}
$$

将 $\mathbf{\mathbf{x}}_{t-2}$带入 $\mathbf{\mathbf{x}}_{t-1}$ 的表达式中：

$$
\mathbf{x}_t = \sqrt{\alpha_t\,\alpha_{t-1}\,\alpha_{t-2}} \mathbf{x}_{t-3}+\sqrt{\alpha_t\,\alpha_{t-1}}\sqrt{\beta_{t-2}}\epsilon_{t-3}+\sqrt{\alpha_t}\,\sqrt{\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\,\epsilon_{t-1}
$$

最终得到 $\mathbf{x}_t$ 的表达式：

$$
\boxed{\mathbf{x}_t =\underbrace{\Bigl(\prod_{s=1}^{t}\sqrt{\alpha_s}\Bigr)}_{\sqrt{\bar{\alpha}_t}}\,x_0+\sum_{i=1}^{t}\underbrace{\Bigl(\sqrt{\beta_i}\prod_{j=i+1}^{t}\sqrt{\alpha_j}\Bigr)}_{\text{噪声系数}}\epsilon_{i-1}}
$$

其中记 $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$

</Card>

<Card title="推导过程（续）">

下面利用数学归纳发证明 $\sum_{i=1}^{t} \left( \sqrt{\beta_i} \prod_{j=i+1}^{t} \sqrt{\alpha_j} \right) \epsilon_{i-1} \sim \mathcal{N}\left(0, 1 - \bar{\alpha}_t\right)$

因为 $\epsilon_{i-1} \sim \mathcal{N}(0, \mathbf{I})$ 独立同分布，有：

$$
\mathrm{Var}\left( \sum_{i=1}^{t} \left( \sqrt{\beta_i} \prod_{j=i+1}^{t} \sqrt{\alpha_j} \right) \epsilon_{i-1} \right) = \sum_{i=1}^{t} \left( \sqrt{\beta_i} \prod_{j=i+1}^{t} \sqrt{\alpha_j} \right)^2 \cdot \mathrm{Var}(\epsilon_{i-1})
$$

::: tip 注

下面是有关方差的一些性质：

- **缩放**：若 $X$ 是均值为 $0$ 的随机变量，则  

$$ 
\mathrm{Var}(c\,X) = c^2\,\mathrm{Var}(X)
$$

- **独立和**：若 $X$ 与 $Y$ 独立，则  

$$
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)
$$

将这两条性质结合起来，就得到：
$$
\mathrm{Var}\bigl(a_i\,\epsilon_{i-1}\bigr) = a_i^2 \,\mathrm{Var}(\epsilon_{i-1}),\quad \mathrm{Var}\Bigl(\sum_i X_i\Bigr) = \sum_i \mathrm{Var}(X_i)\quad(\text{当 }X_i\text{ 相互独立})
$$
:::

而 $\mathrm{Var}(\epsilon_{i-1}) = 1$，所以：

$$
\mathrm{Var} = \sum_{i=1}^{t} \left( \sqrt{\beta_i} \prod_{j=i+1}^{t} \sqrt{\alpha_j} \right)^2 = \sum_{i=1}^{t} \left(\beta_i \prod_{j=i+1}^{t} \alpha_j\right)
$$

即要证明，对于所有正整数 $t$ 有：

$$
\sum_{i=1}^{t} \left(\beta_i \prod_{j=i+1}^{t} \alpha_j \right) = 1 - \bar{\alpha}_t
$$

其中

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s, \quad \beta_i = 1 - \alpha_i
$$

:::: steps
1. **基础情况：**$t = 1$

   当 $t=1$ 时，左边的求和只有一项：

   $$
   \sum_{i=1}^{1} \left(\beta_i \prod_{j=i+1}^{1} \alpha_j \right) = \beta_1 \cdot 1 = \beta_1
   $$

   右边为：

   $$
   1 - \bar{\alpha}_1 = 1 - \alpha_1
   $$

   因为 $\beta_1 = 1 - \alpha_1$，所以等式成立

2. **归纳假设：假设对于** $t = k$ **成立**

   $$
   \sum_{i=1}^{k} \left(\beta_i \prod_{j=i+1}^{k} \alpha_j \right) = 1 - \bar{\alpha}_k
   $$

3. **归纳步骤：证明** $t = k+1$ **时也成立**

   我们需要证明：

   $$
   \sum_{i=1}^{k+1} \left(\beta_i \prod_{j=i+1}^{k+1} \alpha_j \right) = 1 - \bar{\alpha}_{k+1}
   $$

   其中 $\bar{\alpha}_{k+1} = \bar{\alpha}_k \cdot \alpha_{k+1}$

   将求和拆分为前 $k$ 项和最后一项：

   $$
   \sum_{i=1}^{k+1} \left(\beta_i \prod_{j=i+1}^{k+1} \alpha_j \right) = \sum_{i=1}^{k} \left(\beta_i \prod_{j=i+1}^{k+1} \alpha_j \right) + \beta_{k+1}
   $$

   注意到对于 $1 \le i \le k$，有：

   $$
   \prod_{j=i+1}^{k+1} \alpha_j = \left( \prod_{j=i+1}^{k} \alpha_j \right) \cdot \alpha_{k+1}
   $$

   因此，

   $$
   \sum_{i=1}^{k} \left(\beta_i \prod_{j=i+1}^{k+1} \alpha_j \right) = \alpha_{k+1} \sum_{i=1}^{k} \left(\beta_i \prod_{j=i+1}^{k} \alpha_j \right)
   $$

   根据归纳假设，

   $$
   \sum_{i=1}^{k} \left(\beta_i \prod_{j=i+1}^{k} \alpha_j \right) = 1 - \bar{\alpha}_k
   $$

   所以上式变为：

   $$
   \alpha_{k+1}(1 - \bar{\alpha}_k)
   $$

   再加上最后一项 $\beta_{k+1} = 1 - \alpha_{k+1}$：

   $$
   \alpha_{k+1}(1 - \bar{\alpha}_k) + (1 - \alpha_{k+1}) = 1 - \alpha_{k+1} \bar{\alpha}_k
   $$

   而

   $$
   \bar{\alpha}_{k+1} = \bar{\alpha}_k \cdot \alpha_{k+1}
   $$

   所以我们得到：

   $$
   \sum_{i=1}^{k+1} \left(\beta_i \prod_{j=i+1}^{k+1} \alpha_j \right) = 1 - \bar{\alpha}_{k+1}
   $$

   证毕
::::
</Card>

#### $\mathrm{Eq.}(5)$ **推导**

至此，对损失函数 $L$ 的随机项进行随机梯度下降来实现高效训练。进一步地，通过将 $L$ 重写为以下形式以减小方差：

$$
\begin{align*}
L &= \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] \\[7pt]
&\Rightarrow \mathbb{E}_q\Bigl[\underbrace{D_{\mathrm{KL}}\bigl(q(x_T|x_0)\,\|\,p(x_T)\bigr)}_{L_T}
+\sum_{t=2}^T\underbrace{D_{\mathrm{KL}}\bigl(q(\mathbf{x}_{t-1}|\mathbf{x}_t,x_0)\,\|\,p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\bigr)}_{L_{t-1}}
-\underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Bigr] \tag{5}
\end{align*}
$$

<Card title="推导过程">
⭐推导：

$$
L \Rightarrow \mathbb{E}_q\Bigl[\underbrace{D_{\mathrm{KL}}\bigl(q(x_T|x_0)\,\|\,p(x_T)\bigr)}_{L_T}
+\sum_{t=2}^T\underbrace{D_{\mathrm{KL}}\bigl(q(\mathbf{x}_{t-1}|\mathbf{x}_t,x_0)\,\|\,p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\bigr)}_{L_{t-1}}
-\underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Bigr]
$$

$$
\begin{aligned}
L 
&= \mathbb{E}_{q}\Bigl[-\log p(x_T)-\sum_{t=1}^T \log\frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}{q(\mathbf{x}_t\mid \mathbf{x}_{t-1})}\Bigr]\\[8pt]
&\overset{t=T}{\Rightarrow}\mathbb{E}_{q}\Bigl[-\log p(x_T) \boxed{+ \log q(x_T\mid x_0)}\ \Bigr] 
+\mathbb{E}_{q}\Bigl[-\sum_{t=1}^{T} \log\frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}{q(\mathbf{x}_t\mid \mathbf{x}_{t-1})} 
   \boxed{- \log q(x_T\mid x_0)}\ \Bigr] \\[6pt]
&= \underbrace{D_{\rm KL}\bigl(q(x_T\mid x_0)\,\|\,p(x_T)\bigr)}_{L_T}
+\mathbb{E}_{q}\Bigl[-\sum_{t=1}^{T} \log p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)
    +\sum_{t=1}^{T-1}\log q(\mathbf{x}_t\mid \mathbf{x}_{t-1},x_0)\Bigr] \\[6pt]
&\overset{1<t\le T}{\Rightarrow}L_T
+\sum_{t=2}^T\underbrace{\mathbb{E}_{q}\Bigl[-\log p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)
    +\log q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,x_0)\Bigr]}_{D_{\rm KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,x_0)\|p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) =L_{t-1}}
-\underbrace{\mathbb{E}_{q}\bigl[\log p_\theta(x_0\mid x_1)\bigr]}_{L_0} \\[6pt]
\end{aligned}
$$
</Card>

#### $\mathrm{Eq.}(6)$ & $\mathrm{Eq.}(7)$ **推导**

$\mathrm{Eq.}(5)$ 使用 KL 散度直接比较 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 与正向过程的后验分布，当以$\mathbf{x}_0$ 为条件时，该后验分布具有解析解：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t \mathbf{I}) \tag{6}
$$

其中

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := 
\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + 
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
\quad \mathrm{and} \quad 
\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \tag{7}
$$

因此，$\mathrm{Eq.}(5)$ 中的所有 KL 散度项均为高斯分布间的比较，可通过 Rao-Blackwell 化方法直接计算其闭式解，从而避免高方差的 Monte Carlo 估计

<Card title="推导过程">
⭐推导：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t \mathbf{I})
$$

其中

$$
\boldsymbol{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) := 
\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + 
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
\quad \mathrm{and} \quad 
\beta_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

:::: steps
1. **正向过程**  
   $$
   q(\mathbf{x}_t\mid \mathbf{x}_{t-1})
   = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t I\bigr),
   \quad \alpha_t = 1-\beta_t
   $$

2. **累积保留率**  
   $$
   \bar\alpha_t \;=\;\prod_{s=1}^t \alpha_s,
   \quad
   \text{则有}
   \;q(\mathbf{x}_t\mid x_0)=\mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\bigr)
   $$

3. **联合高斯分布（二元变量）**  
   $$
   \begin{pmatrix}\mathbf{x}_{t-1}\\\mathbf{x}_t\end{pmatrix}
   \;\bigg|\;x_0
   \sim
   \mathcal{N}\!\Bigl(
     \underbrace{\begin{pmatrix}\sqrt{\bar\alpha_{t-1}}\,x_0\\[2pt]\sqrt{\bar\alpha_t}\,x_0\end{pmatrix}}_{\mu},
     \underbrace{\begin{pmatrix}
       (1-\bar\alpha_{t-1})\,I & \sqrt{\alpha_t}(1-\bar\alpha_{t-1})\,I\\[3pt]
       \sqrt{\alpha_t}(1-\bar\alpha_{t-1})\,I & (1-\bar\alpha_t)\,I
     \end{pmatrix}}_{\Sigma}
   \Bigr)
   $$

4. **条件高斯公式**  
   对于

   $$
     \begin{pmatrix}u\\v\end{pmatrix}\!\sim\!
     \mathcal{N}\!\bigl(\!(\mu_u,\mu_v),(\Sigma_{uu},\Sigma_{uv};\Sigma_{vu},\Sigma_{vv})\bigr)
   $$

   有

   $$
   p(u\mid v)
   = \mathcal{N}\bigl(u;\,\mu_u + \Sigma_{uv}\Sigma_{vv}^{-1}(v-\mu_v),\;
                     \Sigma_{uu}-\Sigma_{uv}\Sigma_{vv}^{-1}\Sigma_{vu}\bigr)
   $$

5. **计算后验均值**  
   - 令 $u=\mathbf{x}_{t-1},\,v=\mathbf{x}_t$。  
   - $\Sigma_{uv}=\sqrt{\alpha_t}(1-\bar\alpha_{t-1})I,\;\Sigma_{vv}=(1-\bar\alpha_t)I$。  
   $$
   \begin{aligned}
   \mu_t(\mathbf{x}_t,x_0)
   &= \mu_u + \Sigma_{uv}\Sigma_{vv}^{-1}(\mathbf{x}_t-\mu_v) \\[3pt]
   &= \sqrt{\bar\alpha_{t-1}}\,x_0
      + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
        \bigl(\mathbf{x}_t - \sqrt{\bar\alpha_t}\,x_0\bigr) \\[4pt]
   &= \frac{\sqrt{\bar\alpha_{t-1}}\,(1-\alpha_t)}{1-\bar\alpha_t}\,x_0
      + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t \\[2pt]
   &= \frac{\sqrt{\bar\alpha_{t-1}}\;\beta_t}{1-\bar\alpha_t}\,x_0
      + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t
   \end{aligned}
   $$

6. **计算后验方差**  
   $$
   \begin{aligned}
   \beta_t\,I
   &= \Sigma_{uu} - \Sigma_{uv}\Sigma_{vv}^{-1}\Sigma_{vu} \\[3pt]
   &= (1-\bar\alpha_{t-1})\,I
      - \frac{\bigl(\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\bigr)^2}{1-\bar\alpha_t}\,I \\[4pt]
   &= \frac{(1-\bar\alpha_{t-1})\,\beta_t}{1-\bar\alpha_t}\,I
   \end{aligned}
   $$

7. **最终后验分布（$\mathrm{Eq.}(6)$ & $\mathrm{Eq.}(7)$）**  
   $$
   \boxed{
   q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,x_0)
   = \mathcal{N}\bigl(\mathbf{x}_{t-1};\,\tilde\mu_t(\mathbf{x}_t,x_0),\,\tilde\beta_t\,I\bigr)}\ ,
   $$
   
   $$
   \mu_t(\mathbf{x}_t,x_0)= \frac{\sqrt{\bar\alpha_{t-1}}\;\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t,
   \quad
   \beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
   $$
::::
</Card>

### **DDPM 训练原理**

#### **正向过程与** $L_T$

==我们暂不考虑通过重参数化学习 $\beta_t$ 的可能性，而是简单地将其设定为常数=={.note}。在论文的实现中，后验分布 $q$ 并不含可学习参数，因此 $L_T$ 在训练过程中为常数项，可在损失函数中忽略不计

#### **反向过程与** $L_{1:T-1}$

我们接下来分析反向过程中的分布 $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$，该分布被建模为高斯形式：
$$
p(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
$$
对任意 $1 \leq t \leq T$ 均成立

首先，对于协方差 $\Sigma_\theta(\mathbf{x}_t, t)$，我们设置为时间相关的固定常数，即 $\Sigma_\theta(\mathbf{x}_t, t) = \sigma_t^2 I$，且不参与训练。在实验中，$\sigma_t^2 = \beta_t$ 与 $\sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ 均取得了相似的效果。前者适用于 $x_0 \sim \mathcal{N}(0, I)$ 的情况，后者适用于将 $x_0$ 映射为固定值的情形。这两个选择在反向过程的熵约束中，分别对应于单位方差数据的上下界

接下来，我们引入对均值项 $\mu_\theta(\mathbf{x}_t, t)$ 的一种特殊参数化形式，该形式的灵感来自对损失项 $L_t$ 的分析

由 $\mathrm{Eq.}(4)$ 可知：

$$
p(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 I)
$$
其对应的损失函数为：
$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \| \hat{\mu}_t(\mathbf{x}_t, x_0) - \mu_\theta(\mathbf{x}_t, t) \|^2 \right] + C \tag{8}
$$
其中 $C$ 为与 $x_0$ 无关的常数。因此，最直接的方式是令模型直接拟合 $\hat{\mu}_t$，即正向过程后验分布的均值

我们可以进一步重参数化该表达式，设 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$，代入 $\mathrm{Eq.}(7)$ 得：

$$
L_{t-1} - C = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \hat{\mu}_t\left(\mathbf{x}_t(x_0, \epsilon), \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t(x_0, \epsilon) - \sqrt{1 - \bar{\alpha}_t} \epsilon) \right) - \mu_\theta(\mathbf{x}_t(x_0, \epsilon), t) \right\|^2 \right] \tag{9}
$$

最终可化简为：

$$
= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - \mu_\theta(\mathbf{x}_t(x_0, \epsilon), t) \right\|^2 \right] \tag{10}
$$

算法流程如下：

<CardGrid>
   <ImageCard
   	image="https://s21.ax1x.com/2025/04/24/pEokVNq.png"
   	width=100%
   />
   <ImageCard
	   image="https://s21.ax1x.com/2025/04/24/pEokZ40.png"
	   width=100%
   />
</CardGrid>


$\mathrm{Eq.}(10)$ 表明，在给定 $\mathbf{x}_t$ 的条件下，$\mu_\theta$ 能够重现表达式 $\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon)$。既然 $\mathbf{x}_t$ 是模型输入，我们采用以下参数化方式：

$$
\mu_\theta(\mathbf{x}_t, t) = \hat{\mu}_t \left( \mathbf{x}_t, \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(\mathbf{x}_t)) \right) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) \tag{11}
$$

其中，$\epsilon_\theta$ 是一个函数近似器，用于从输入 $\mathbf{x}_t$ 中预测噪声项 $\epsilon$。为了从 $\mathbf{x}_t$ 得到 $\mathbf{x}_{t-1}$，有：

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

这一完整的采样过程（即 $\mathrm{\bold{Algorithm\ 2}}$），在形式上类似于基于得分函数学习的 Langevin 动力学。此外，采用参数化形式 $\mathrm{Eq.}(11)$ 时，$\mathrm{Eq.}(10)$ 可进一步简化为：

$$
\mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right] \tag{12}
$$

这一无偏的多尺度噪声匹配损失正如文献 $[55]$ 所述，可在不同噪声水平下联合训练。根据 $\mathrm{Eq.}(12)$，该目标等价于对 Langevin 型反向过程 $\mathrm{Eq.}(11)$ 所构造的变分下界进行优化

综上所述，我们通过将反向过程的均值函数 $\mu_\theta$ 重新参数化为 $\hat{\mu}_t$，建立了一个能够预测噪声项 $\epsilon$ 的模型框架

#### **数据缩放、反向过程解码器和 $L_0$**

假设图像数据由整数（取值范围为 $\{0, 1, \ldots, 255\}$）组成，并且这些整数被线性缩放到 $[-1, 1]$。这确保了神经网络的反向过程始终在一致缩放的输入上运行，起始于标准正态先验 $p(x_T)$

为了获得离散数据的对数似然，我们将反向过程的最后一项设置为一个独立的离散解码器，其从高斯分布 $\mathcal{N}(x_0; \mu_0(x_1, 1), \sigma_0^2 I)$ 中生成：

$$
p_0(x_0 | x_1) = \prod_{i=1}^D \int_{\delta_{-}(x_i)}^{\delta_{+}(x_i)} \mathcal{N}(x_i; \mu_{0,i}(x_1, 1), \sigma_0^2) \, dx \tag{13}
$$

其中：

- $D$ 表示数据的维度，$i$ 上标表示提取特定坐标
- $\delta_{+}(x)$ 和 $\delta_{-}(x)$ 定义如下：

$$
\delta_+(x) = 
\begin{cases} 
\infty & x = 255 \\ 
x + \frac{1}{255} & x < 255 
\end{cases}, \quad 
\delta_-(x) = 
\begin{cases} 
-\infty & x = 0 \\ 
x - \frac{1}{255} & x > 0 
\end{cases}
$$

在 VAE 解码器和自回归模型中，离散分布通常通过离散化连续分布来实现。我们的方法确保变分下界是离散数据的**无损编码（lossless coding）**，无需在数据中添加噪声或在对数似然中引入缩放操作的**雅可比行列式（Jacobian）**。在采样结束时，在无噪声下展示 $\mu_\theta(\mathbf{x}_1, 1)$


#### **简化的训练目标**

在上面定义的反向过程和解码器的基础上，变分下界（由 $\mathrm{Eq.}(12)$和 $\mathrm{Eq.}(13)$ 推导而来）显然可以微分并用于训练

然而，我们发现使用以下变种形式的变分下界在采样质量和实现上均表现地更好：

$$
\boxed{
L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right]} \tag{14}
$$

其中，$t$ 在 $1$ 和 $T$ 之间均匀分布。$t = 1$ 的情况对应于 $L_0$，此时在离散解码器定义 $\mathrm{Eq.}(13)$ 中的积分被高斯概率密度函数乘以区间宽度来近似，忽略了 $\sigma_1^2$ 和边缘效应。对于 $t > 1$ 的情况，它们对应于 $\mathrm{Eq.}(12)$ 的一个 **无权重版本**，类似于 NCSN 去噪分数匹配模型 $[55]$ 使用的损失加权方式（$L_T$ 没有出现，因为正向过程的方差 $\beta_t$ 是固定的）。$\mathbf{Algorithm 1}$ 展示了使用该简化目标的完整训练过程。

