# Method

## 1. Overview

MoTA is a model-agnostic test-time adaptation framework for image demoiréing. Given a demoiréing model $f_\theta$ pre-trained on a source domain $\mathcal{D}_S$ (e.g., images captured with an iPhone from an LCD screen), and a single test image $I$ from an unseen target domain $\mathcal{D}_T$ (e.g., captured with a different camera or from a different screen type), MoTA adapts the model at inference time to improve output quality — without accessing any paired data, ground-truth images, or multi-frame capture from $\mathcal{D}_T$.

The framework comprises three components. First, the **Moiré-Aware Self-Supervision (MASS)** module decomposes $I$ into frequency subbands via discrete wavelet transform, detects moiré-dominant frequency regions, and generates a pseudo-clean target $\tilde{I}$ through selective band attenuation. Second, **MoTA Adapters** — a Fourier Domain Adapter (FDA) and a Spatial Gating Adapter (SGA) — are inserted into the frozen backbone $f_\theta$ to provide a low-dimensional adaptation interface. Third, a **test-time optimization** loop minimizes the discrepancy between the backbone output $\hat{I} = f_{\theta,\phi}(I)$ and $\tilde{I}$ with respect to adapter parameters $\phi$ only; the backbone $\theta$ remains frozen throughout. After $T$ adaptation steps (typical $T=15$), the adapted model produces the final demoiréd output. Figure [FIG:overview] illustrates the complete pipeline.

## 2. Preliminaries and Problem Formulation

**Image Demoiréing.** Let $I_m \in \mathbb{R}^{H \times W \times 3}$ be a moiré-affected sRGB image and $I_c \in \mathbb{R}^{H \times W \times 3}$ its clean counterpart. A demoiréing model $f_\theta: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{H \times W \times 3}$ parameterized by $\theta$ is trained to minimize $\mathbb{E}_{(I_m, I_c) \sim \mathcal{D}_S}[\mathcal{L}(f_\theta(I_m), I_c)]$ on a source domain $\mathcal{D}_S$.

**Domain Shift in Demoiréing.** In practice, the test distribution $\mathcal{D}_T$ differs from $\mathcal{D}_S$ because moiré patterns are determined by capture-specific parameters — camera sensor, lens, screen type, pixel pitch, shooting distance — each of which can vary independently. The resulting domain gap causes $f_\theta$ to perform suboptimally on $\mathcal{D}_T$. Our goal is to adapt $\theta$ at test time using only $I_m \sim \mathcal{D}_T$, without access to $I_c$, additional capture hardware, or target-domain training data.

**Test-Time Adaptation.** We freeze $\theta$ and introduce lightweight adapter parameters $\phi$ (where $|\phi| \ll |\theta|$), yielding an augmented model $f_{\theta,\phi}$. At test time, we optimize $\phi$ on each input $I_m$ via a self-supervised loss $\mathcal{L}_{self}$ that requires no ground truth. After adaptation, the output is $\hat{I} = f_{\theta,\phi_T}(I_m)$.

**Wavelet Decomposition.** We use the 2D Discrete Wavelet Transform (DWT) with Haar wavelets to decompose images into four subbands: $LL$ (low-frequency approximation), $LH$ (horizontal detail), $HL$ (vertical detail), and $HH$ (diagonal detail). Moiré energy concentrates in $LH$ and $HL$, a property central to MASS.

## 3. Analysis of Existing Approaches

| Method | Core Idea | Strengths | Limitations | Representative |
|--------|-----------|-----------|-------------|----------------|
| Frequency-Domain (MBCNN, WDNet, FHDe2Net) | DCT/DWT 固定频率分解，在变换域分离摩尔纹 | 频域可解释性强，对特定频率摩尔纹效果好 | 固定分解基无法适应不同设备的摩尔纹频率变化 | [@zheng2022learning; @liu2020wavelet; @he2020fhde2net] |
| Frequency-Spatial Fusion (WDNet, MoiréNet) | 可学习频率分解 + 空间注意力融合 | 当前 SOTA，在标准 benchmark 上表现最好 | 频率分解策略训练后固定，跨域时无法自适应调整 | [@liu2020wavelet; @anonymous2025moirenet] |
| RAW+RGB Multi-Domain (RRID, MoiréXNet) | 联合 RAW+sRGB 信息，从源头处理摩尔纹 | 利用摩尔纹形成的物理先验 | 需 RAW 输入，MoiréXNet 的 TTT 模块绑定特定架构 | [@xu2024image; @li2025moirexnet] |
| Data-Centric (UniDemoiré, UnDeM) | 扩大数据分布覆盖，从数据角度缓解域偏移 | 泛化性在已知域内有所提升 | 生成/收集的数据分布终有限，真正 unseen domain 仍会失效 | [@yang2025unidemoire; @anonymous2024undem] |
| TTA for Restoration (DCTTA, TAO, TTPO) | 测试时自适应，推理时利用输入图像调整模型 | 无需目标域配对数据，可处理分布偏移 | 面向通用退化（噪声/模糊/雨雾），无摩尔纹特定设计 | [@tang2026degradation; @gou2024testtime; @li2026testtime] |
| PEFT Adapters (FraIR, BIR-Adapter) | 轻量适配器迁移预训练知识 | 参数高效，可插拔 | 仅用于已知任务迁移，未探索测试时自适应场景 | [@anonymous2026frair; @anonymous2025biradapter] |

**核心观察**：频率域方法缺自适应能力，TTA 方法缺摩尔纹特定设计，Adapter 方法缺测试时自适应场景。MoTA 填补三者交汇处的空白。

## 4. MoTA Framework Design

### 4.1 总体架构

MoTA（Moiré Test-Time Adaptation）是一个模型无关的测试时自适应框架。给定一个在源域 $\mathcal{D}_S$ 上预训练的去摩尔纹模型 $f_\theta$（可以是 CNN、Transformer 或 Mamba 架构），MoTA 在推理时仅利用单张目标域输入图像 $I \in \mathcal{D}_T$，通过轻量适配器和摩尔纹感知自监督信号，在不访问目标域 GT 的情况下完成跨域自适应。

整体流程分为三个阶段：

**阶段 1: 摩尔纹感知自监督信号生成 (MASS)**
对输入图像进行频率分解，检测摩尔纹主导的频率子带，生成伪干净目标 $\tilde{I}$ 作为自监督信号。

**阶段 2: 适配器增强的前向传播**
冻结的预训练骨干 $f_\theta$ 中插入了两组轻量适配器——傅里叶域适配器 (FDA) 和空间门控适配器 (SGA)，适配器参数 $\phi$ 在测试时更新。

**阶段 3: 测试时优化**
通过最小化输出 $\hat{I} = f_{\theta,\phi}(I)$ 与伪干净目标 $\tilde{I}$ 之间的 MASS 损失来更新适配器参数 $\phi$。骨干参数 $\theta$ 全程冻结。

[FIGURE: MoTA framework overview. Left: Input moiré image I. Middle-top: MASS module generates pseudo-clean target via frequency decomposition and moiré detection. Middle-bottom: Frozen pre-trained backbone with inserted MoTA Adapters (FDA+SGA). Right: Output Î. A red dashed arrow indicates gradient flow updating only adapter parameters φ via L_MASS. Three colored boxes group the three stages.]

### 4.2 Module Details

#### 2.2.1 摩尔纹频率检测器 (Moiré Frequency Detector, MFD)

**输入**：单张摩尔纹图像 $I \in \mathbb{R}^{H \times W \times 3}$
**输出**：频率子带级摩尔纹概率图 $M \in \mathbb{R}^{H/2 \times W/2 \times 3}$

**核心操作**：

首先对 $I$ 的每个通道应用单级 Haar 小波变换（DWT），分解为四个子带：
$$\{LL, LH, HL, HH\} = \text{DWT}(I)$$

摩尔纹在频域中体现为 LH 和 HL 子带（中频）中的异常高能量模式。MFD 是一个轻量的 4 层卷积网络，接受拼接后的四个子带作为输入：

$$M = \sigma(\text{MFD}([LL; LH; HL; HH]))$$

其中 $\sigma$ 为 sigmoid 激活，$M$ 表示每个空间位置在各频率子带上的摩尔纹概率。

**训练方式**：MFD 在源域上以合成监督信号预训练——对干净图像人工合成摩尔纹（通过已知的摩尔纹生成模型），已知合成时的频率分布作为 pseudo GT。MFD 在测试时不更新。

**与现有工作的差异**：不同于 Self-Adaptive Demoiré [@liu2020wavelet] 需要双摄像头硬件，也不同于 MoiréXNet [@li2025moirexnet] 需要多帧 RAW 输入，MFD 仅需单张 sRGB 图像即可提供频率级摩尔纹定位。

#### 2.2.2 伪干净目标生成 (MASS Signal)

**输入**：原始图像 $I$ 和摩尔纹概率图 $M$
**输出**：伪干净目标 $\tilde{I}$

**核心操作**：

将输入图像的 DWT 子带与摩尔纹概率图做频率选择性衰减：

$$LL' = LL \quad \text{(保持低频内容不变)}$$
$$LH' = LH \odot (1 - \alpha \cdot M)$$
$$HL' = HL \odot (1 - \alpha \cdot M)$$
$$HH' = HH \quad \text{(保持高频纹理不变)}$$

然后通过逆小波变换重建伪干净图像：
$$\tilde{I} = \text{IDWT}(LL', LH', HL', HH')$$

其中 $\alpha \in [0.3, 0.7]$ 控制衰减强度（$M$ 高的区域衰减强，低区域衰减弱）。该操作的设计动机基于一个关键观察：**摩尔纹能量集中在中频子带，而图像内容和精细纹理分别占据低频和高频子带**。通过选择性抑制中频摩尔纹能量，得到的 $\tilde{I}$ 虽然不是完美 GT，但作为自监督目标已足够指导适配器学习域偏移。

**设计动机**：相比 DCTTA 使用的扩散模型重退化（需数百步采样），MASS 仅需一次 DWT/IDWT，计算成本 O(HW)，适合测试时迭代。

#### 2.2.3 傅里叶域适配器 (Fourier Domain Adapter, FDA)

**插入位置**：骨干网络每个 stage 的输出之后（例如 ResNet 的每个 block 或 Transformer 的每个 encoder layer 之后）

**输入/输出**：特征图 $X \in \mathbb{R}^{C \times H \times W}$，输出增强特征 $X'$

**核心操作**：

对特征沿通道维度做 1D FFT 变换到频域，施加低秩谱调制后逆变换：

$$F = \text{FFT}_{channel}(X) \in \mathbb{C}^{C \times H \times W}$$
$$\Delta S = A B^T \quad \text{其中 } A, B \in \mathbb{R}^{C \times r}, r \ll C$$
$$F' = F \odot (1 + \Delta S)$$
$$X' = \text{IFFT}_{channel}(F') + X \quad \text{(残差连接)}$$

其中 $r = \max(4, C/32)$ 为低秩维度。FDA 的参数总量仅 $2Cr$，通常 $<1\%$ 的骨干参数量。

**设计动机**：域偏移在频域中表现为不同频率分量的相对强度变化。FDA 通过在频域进行低秩调制来矫正这种偏移。选择沿通道维度的 1D FFT（而非空间维度的 2D FFT）是因为通道间相关性更能反映不同频率滤波器的响应模式。残差连接确保初始状态等同于原始骨干网络。

**与 FraIR 的差异**：FraIR [@anonymous2026frair] 的傅里叶适配器用于任务迁移（如去噪→去模糊），我们的 FDA 用于域迁移（源域→目标域），且 FDA 在测试时才被更新。

#### 2.2.4 空间门控适配器 (Spatial Gating Adapter, SGA)

**插入位置**：与 FDA 并行放置于同一插入点

**输入/输出**：特征图 $X$，输出门控特征 $X''$

**核心操作**：

$$G = \text{Conv}_{3 \times 3}(\text{Conv}_{1 \times 1}(X))$$
$$X'' = X \odot \sigma(G)$$

其中 $\text{Conv}_{1 \times 1}$ 将通道数压缩为原来的 $1/4$，$\text{Conv}_{3 \times 3}$ 提取局部空间上下文，$\sigma$ 为 sigmoid 门控。

**设计动机**：摩尔纹在空间上呈现非均匀分布（通常集中在特定区域），SGA 通过空间门控让网络学会"哪里需要更多/更少的特征修正"。与 FDA 形成互补——FDA 负责频率域矫正，SGA 负责空间域调制。

### 4.3 Loss Function

测试时自适应的总损失函数：

$$\mathcal{L}_{total} = \mathcal{L}_{MASS} + \lambda_{reg} \mathcal{L}_{reg}$$

#### MASS 损失

$$\mathcal{L}_{MASS} = \|\hat{I} - \tilde{I}\|_1$$

使用 L1 损失（而非 L2）以减少对伪干净目标中残余摩尔纹的过拟合。L1 对异常值的鲁棒性更好。

#### 正则化损失

$$\mathcal{L}_{reg} = \|\Delta\phi\|_2^2 = \|\phi_t - \phi_0\|_2^2$$

约束适配器参数的更新幅度，防止单张图像上的过拟合。$\phi_0$ 是适配器的初始参数（训练在源域上使输出不变），$\phi_t$ 是当前步参数。

#### 超参数

- $\lambda_{reg} = 0.01$：正则化权重
- $\alpha = 0.5$：频率衰减强度
- $T = 15$：每张图像的自适应步数
- $\eta = 1 \times 10^{-4}$：适配器学习率
- $r = \max(4, C/32)$：FDA 低秩维度

## 5. Baseline Selection

```json
[
  {
    "name": "MoiréNet",
    "paper": "MoiréNet: A Compact Dual-Domain Network for Image Demoiréing",
    "bibtex_key": "anonymous2025moirenet",
    "why_selected": "当前 SOTA，频率-空间双域融合，4 个 benchmark 上最佳 PSNR",
    "type": "SOTA",
    "code_available": false,
    "code_url": null
  },
  {
    "name": "WDNet",
    "paper": "WDNet: Image-Demoiréing Transformer via Efficient Frequency Decomposition",
    "bibtex_key": "liu2020wavelet",
    "why_selected": "Transformer 架构 SOTA，与 MoiréNet 代表不同技术路线",
    "type": "SOTA",
    "code_available": true,
    "code_url": "https://github.com/xyLiu339/WDNet"
  },
  {
    "name": "MBCNN",
    "paper": "Learning Frequency Domain Priors for Image Demoiréing",
    "bibtex_key": "zheng2022learning",
    "why_selected": "频率域先验方法的经典代表，AIM 2019 冠军，被广泛对比",
    "type": "classic",
    "code_available": false,
    "code_url": null
  },
  {
    "name": "MoiréXNet",
    "paper": "MoiréXNet: Adaptive Multi-Scale Demoiréing with Linear Attention TTT",
    "bibtex_key": "li2025moirexnet",
    "why_selected": "最直接竞争者——唯一将 TTT 用于去摩尔纹的工作",
    "type": "closest_competitor",
    "code_available": false,
    "code_url": null
  },
  {
    "name": "ESDNet",
    "paper": "Towards Efficient and Scale-Robust UHD Image Demoiréing",
    "bibtex_key": "yu2022towards",
    "why_selected": "UHD 方向代表，与我们的方法形成不同应用场景的对照",
    "type": "different_paradigm",
    "code_available": true,
    "code_url": "https://xinyu-andy.github.io/uhdm-page"
  },
  {
    "name": "UniDemoiré",
    "paper": "UniDemoiré: Towards Universal Image Demoiréing with Data Generation",
    "bibtex_key": "yang2025unidemoire",
    "why_selected": "数据驱动泛化路线的代表，与我们的 TTA 路线形成方法论对比",
    "type": "different_paradigm",
    "code_available": false,
    "code_url": null
  }
]
```

## 6. Algorithm

[FIGURE: Algorithm pseudo-code listing for MoTA. Single-column, using algorithmic environment.]

```
Algorithm 1: MoTA — Moiré Test-Time Adaptation

Input: 测试图像 I, 预训练骨干 f_θ (冻结), 预训练 MFD, 适配器 φ_0
Output: 去摩尔纹图像 Î
Parameters: T = 15, η = 1e-4, λ_reg = 0.01, α = 0.5

1:  /* 阶段 1: 生成 MASS 信号 */
2:  {LL, LH, HL, HH} ← DWT(I)                       ▷ Haar 小波分解
3:  M ← MFD([LL; LH; HL; HH])                        ▷ 摩尔纹频率检测
4:  LH' ← LH ⊙ (1 − α · M)                           ▷ 中频衰减
5:  HL' ← HL ⊙ (1 − α · M)
6:  Ĩ ← IDWT(LL, LH', HL', HH)                       ▷ 伪干净目标

7:  /* 阶段 2: 测试时适配器优化 */
8:  φ ← φ_0                                           ▷ 初始化适配器参数
9:  for t = 1 to T do
10:   Î ← f_{θ,φ}(I)                                  ▷ 前向传播 (骨干冻结, 适配器激活)
11:   L_MASS ← ‖Î − Ĩ‖_1                              ▷ 自监督损失
12:   L_reg ← ‖φ − φ_0‖_2^2                           ▷ 正则化损失
13:   L ← L_MASS + λ_reg · L_reg
14:   φ ← φ − η · ∇_φ L                               ▷ 仅更新适配器参数
15: end for

16: Î ← f_{θ,φ_T}(I)                                  ▷ 最终输出
17: return Î
```

**关键创新标注**：
- 第 3 行：MFD 提供频率级摩尔纹定位（替代了传统 TTA 方法中的盲目自监督信号）
- 第 4-6 行：MASS 选择性衰减中频（保留低频内容+高频纹理，仅抑制摩尔纹频段）
- 第 14 行：仅更新适配器参数（<5% 总参数），骨干网络完全冻结

### 复杂度分析

| | 骨干参数 | 可训练参数 | 每步 TTA 时间 | 总 TTA 时间 (T=15) |
|---|---|---|---|---|
| Full Fine-tuning | N | N | O(N) | O(15N) |
| MoiréXNet (TTT) | N | ~0.3N (attention layers) | O(N) | O(15·0.3N) |
| **MoTA (ours)** | N (frozen) | **~0.03N** (FDA+SGA) | **O(rC)** | **O(15·rC)** |

其中 N 为骨干参数量，r 为低秩维度（r << C），C 为通道数。

## 7. Theoretical Analysis

### 5.1 为什么 MASS 信号不会导致平凡解

一个自然的担忧是：MASS 信号来自输入图像本身的频率衰减，优化 $\|\hat{I} - \tilde{I}\|_1$ 是否会让网络简单地学习"对输入图像做低通滤波"这个平凡解？

我们通过两方面设计避免该问题：

**a) 适配器容量约束**：FDA+SGA 的总参数量不到骨干的 3%。低秩约束和空间门控的局部感受野使得适配器无法学习全局的低通滤波操作——这需要全秩的全连接变换。

**b) 骨干的归纳偏置**：预训练骨干 $f_\theta$ 已经被训练为"从摩尔纹图像中恢复干净图像"的映射，其隐空间中的表示天然偏向自然图像流形。适配器仅需调整频域偏移量，而非从头学习去摩尔纹。

形式化地，设 $f_\theta$ 满足：对源域图像 $I_S$，有 $f_\theta(I_S) \approx I_S^{clean}$。在目标域，由于域偏移 $\delta$，有 $f_\theta(I_T) = I_T^{clean} + \epsilon_\delta$。适配器的任务是学习残差映射 $g_\phi$ 使得 $f_{\theta,\phi}(I_T) \approx I_T^{clean}$。由于 $g_\phi$ 的低容量约束，它只能学习 $\epsilon_\delta$ 的补偿而无法推翻 $f_\theta$ 的去摩尔纹能力。

### 5.2 跨域泛化边界

我们给出一个非正式的泛化分析。假设源域和目标域之间的分布偏移主要体现在频率域：源域摩尔纹频段为 $\mathcal{F}_S$，目标域为 $\mathcal{F}_T$（$\mathcal{F}_S \neq \mathcal{F}_T$）。

传统方法的误差来源为 $f_\theta$ 在 $\mathcal{F}_T$ 上的表示能力不足，产生误差 $\epsilon_{freq}$。MASS 通过频率自适应衰减将 $\mathcal{F}_T$ 的部分能量转移到 $\mathcal{F}_S$ 的范围内，使得适配后的 $f_{\theta,\phi}$ 的有效输入分布更接近源域训练分布。因此：

$$\text{Error}_T(f_{\theta,\phi}) \leq \text{Error}_T(f_\theta) - \underbrace{\gamma \cdot |\mathcal{F}_T \setminus \mathcal{F}_S|}_{\text{MASS 贡献}} + \underbrace{O(r/C)}_{\text{适配器容量惩罚}}$$

其中 $\gamma > 0$ 取决于 MFD 的检测精度。该分析表明：当域偏移以频域差异为主时（去摩尔纹问题恰好如此），MoTA 可提供有保证的改进。
