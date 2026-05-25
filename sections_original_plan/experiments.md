# Experiments

## 1. Datasets and Evaluation Protocol

### 1.1 Datasets

我们使用 4 个公开数据集，覆盖不同分辨率、设备和摩尔纹类型：

| Dataset | Resolution | Type | Train/Test Split | Source Device |
|---------|-----------|------|------------------|---------------|
| **TIP2018** [@sun2018moire] | ~400×400 | Real (screen-captured) | 135,000 / 1,500 pairs | iPhone + LCD |
| **FHDMi** [@he2020fhde2net] | 1920×1080 | Real (screen-captured) | 10,481 / 1,624 pairs | Canon 6D + LCD |
| **LCDMoire** [@zheng2022learning] | 1024×1024 | Synthetic (image+text) | 9,000 / 1,200 pairs | Generated |
| **UHDM** [@yu2022towards] | 4328×3248 (4K) | Real (screen-captured) | 4,500 / 500 pairs | Sony α7 III + LCD |

### 1.2 Cross-Domain Evaluation Protocol

这是本工作的核心评测设计。现有去摩尔纹方法几乎全部在同域（in-domain）评测，掩盖了域偏移问题。我们定义两种评测设定：

**In-Domain (ID)**：在 TIP2018 上做标准 train/test split 评测（与现有工作一致，作为 sanity check）。

**Cross-Domain (CD)**：所有方法在 **TIP2018 训练集** 上训练，在 **FHDMi / LCDMoire / UHDM 测试集** 上评测。这模拟了真实场景：模型用某个设备采集的数据训练后，被部署到完全不同的设备上。

对于我们的 MoTA 框架，TTA 过程仅在目标域测试图像上进行（无需目标域训练数据）。对于 MoiréXNet，其内置的 TTT 模块同样在推理时生效。对于 UniDemoiré，我们按原论文设置了其数据增强管线（使用其收集的摩尔纹模式库做训练数据增强），因为这是其方法的核心贡献。

### 1.3 Domain Gap Quantification

为量化域间差异，我们计算各数据集间的 FID（Fréchet Inception Distance）：

| | TIP2018 | FHDMi | LCDMoire | UHDM |
|---|---|---|---|---|
| TIP2018 | — | 42.3 | 38.7 | 51.6 |
| FHDMi | 42.3 | — | 35.1 | 29.8 |
| LCDMoire | 38.7 | 35.1 | — | 43.2 |
| UHDM | 51.6 | 29.8 | 43.2 | — |

*注：表中数值待填充真实 FID 计算结果。FID 在摩尔纹图像上计算，越高表示域差异越大。*

## 2. Evaluation Metrics

| Metric | Formula / Description | Range | Higher Better? | Type |
|--------|----------------------|-------|----------------|------|
| **PSNR** | $10 \log_{10}(MAX_I^2 / MSE)$ | $[0, \infty)$ dB | Yes | Primary |
| **SSIM** | $[l(I,\hat{I})]^\alpha \cdot [c(I,\hat{I})]^\beta \cdot [s(I,\hat{I})]^\gamma$ | $[-1, 1]$ | Yes | Primary |
| **LPIPS** | Perceptual distance via pre-trained AlexNet features | $[0, 1]$ | No | Secondary |
| **CD-Gain** | $\Delta$PSNR 在跨域 vs 同域的下降幅度：$PSNR_{ID} - PSNR_{CD}$ | $[0, \infty)$ | **No** | Proposed |

**CD-Gain** 是我们提出的评测指标，量化方法的跨域鲁棒性。CD-Gain 越低，方法跨域性能下降越少，泛化能力越好。

*注：PSNR 和 SSIM 是去摩尔纹领域标准指标，LPIPS 补充感知质量评估（如 MoiréNet、WDNet 等 SOTA 方法均使用）。*

## 3. Implementation Details

### 3.1 Training (Pre-training on Source Domain)

所有方法（包括我们的骨干网络）在 TIP2018 训练集上以相同设定预训练：

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) |
| Learning rate | $1 \times 10^{-4}$ (cosine annealing to $1 \times 10^{-6}$) |
| Batch size | 16 (TIP2018), 4 (FHDMi/UHDM, due to resolution) |
| Epochs | 200 (early stopping on val PSNR, patience=20) |
| Input size | 256×256 random crop (training), full resolution (testing) |
| Data augmentation | Random horizontal flip, rotation (±5°) |
| Loss | L1 + 0.1 × SSIM loss (standard for demoiréing) |
| Hardware | 4× NVIDIA A100 (80GB) |
| Random seeds | 3 runs (seed=42, 123, 2026), report mean ± std |

### 3.2 MoTA-Specific Settings

**MFD 预训练**：MFD 在 TIP2018 训练集上用合成监督信号预训练——对干净图像用已知摩尔纹模式合成摩尔纹，已知合成参数作为频率检测 pseudo GT。MFD 在 TTA 阶段冻结。

**TTA 设定**：

| Hyperparameter | Value |
|---------------|-------|
| Adaptation steps $T$ | 15 |
| Adapter learning rate $\eta$ | $1 \times 10^{-4}$ |
| Regularization weight $\lambda_{reg}$ | 0.01 |
| Frequency attenuation $\alpha$ | 0.5 |
| FDA low-rank dimension $r$ | $\max(4, C/32)$ |
| Adapter initialization | $\phi_0$ 训练为使输出不变 (identity-mapping initialization) |

**骨干网络配置**：我们分别以 MoiréNet [@anonymous2025moirenet] 和 WDNet [@liu2020wavelet] 作为骨干，验证 MoTA 的模型无关性。默认骨干为 MoiréNet。

### 3.3 Baseline 复现说明

| Baseline | 实现来源 | 备注 |
|----------|---------|------|
| MBCNN [@zheng2022learning] | 第三方复现 (基于论文描述) | 无官方代码 |
| WDNet [@liu2020wavelet] | 官方代码 | github.com/laulampaul/WDNet_demoire |
| ESDNet [@yu2022towards] | 官方代码 | github.com/xinyu-andy/uhdm-page |
| MoiréNet [@anonymous2025moirenet] | 基于论文复现 | 无官方代码，严格按照论文描述实现 |
| WDNet [@liu2020wavelet] | 官方代码 | github.com/laulampaul/WDNet_demoire |
| MoiréXNet [@li2025moirexnet] | 基于论文复现 | 无官方代码 |
| UniDemoiré [@yang2025unidemoire] | 基于论文复现 | 使用其数据增强管线；无官方代码 |

所有 Baseline 使用相同的训练数据和训练设定（3.1 节），确保公平对比。

## 4. Main Results

### 4.1 In-Domain Results (TIP2018)

*表 1: TIP2018 test set 同域评测结果。所有方法在 TIP2018 train set 上训练。最优加粗，次优加下划线。*

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params (M) |
|--------|-------------|--------|---------|------------|
| MBCNN [@zheng2022learning] | - | - | - | - |
| WDNet [@liu2020wavelet] | - | - | - | - |
| ESDNet [@yu2022towards] | - | - | - | - |
| MoiréNet [@anonymous2025moirenet] | - | - | - | 5.5 |
| WDNet [@liu2020wavelet] | - | - | - | - |
| MoiréXNet [@li2025moirexnet] | - | - | - | - |
| UniDemoiré [@yang2025unidemoire] | - | - | - | - |
| **MoTA (MoiréNet backbone)** | **-** | **-** | **-** | 5.5 + 0.17 |
| **MoTA (WDNet backbone)** | **-** | **-** | **-** | - |

*表中数值待填充真实实验结果。同域设定下 TTA 不期望大幅提升（因为没有域偏移），我们的方法应与骨干网络持平。*

### 4.2 Cross-Domain Results (Main Claim)

*表 2: 跨域泛化结果。所有方法在 TIP2018 上训练，在目标域测试集上评测（无目标域微调）。CD-Gain 以 TIP2018 → FHDMi 的 PSNR 下降计算。*

**TIP2018 → FHDMi**

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | CD-Gain ↓ |
|--------|-------------|--------|---------|-----------|
| MBCNN | - | - | - | - |
| WDNet | - | - | - | - |
| ESDNet | - | - | - | - |
| MoiréNet | - | - | - | - |
| WDNet | - | - | - | - |
| MoiréXNet | - | - | - | - |
| UniDemoiré | - | - | - | - |
| **MoTA (MoiréNet backbone)** | **-** | **-** | **-** | **-** |
| **MoTA (WDNet backbone)** | **-** | **-** | **-** | **-** |

**TIP2018 → LCDMoire**

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-------------|--------|---------|
| (同上表结构) | - | - | - |

**TIP2018 → UHDM**

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-------------|--------|---------|
| (同上表结构) | - | - | - |

*预期：在跨域设定下，传统方法（MBCNN, WDNet, ESDNet, MoiréNet, WDNet）PSNR 大幅下降（域偏移）；UniDemoiré 因数据增强泛化稍好但仍下降；MoiréXNet 因内置 TTT 有一定抗偏移能力；MoTA 因模型无关 TTA 框架在三个目标域上均获得最小 CD-Gain。*

## 5. Ablation Studies

### 5.1 Component Ablation

*表 3: MoTA 各组件消融实验（TIP2018 → FHDMi，MoiréNet 骨干）。*

| Variant | PSNR (dB) ↑ | SSIM ↑ | Δ PSNR vs Full |
|---------|-------------|--------|-----------------|
| **Full MoTA** | **-** | **-** | **0.00** |
| w/o FDA (仅 SGA) | - | - | - |
| w/o SGA (仅 FDA) | - | - | - |
| w/o MASS (用 L1 reconstruction on input 替代) | - | - | - |
| w/o $\mathcal{L}_{reg}$ | - | - | - |
| w/o TTA (frozen backbone, no adaptation) | - | - | - |
| Full fine-tuning (更新全部骨干参数) | - | - | - |

*预期：FDA 和 SGA 分别贡献互补增益；MASS 是最关键组件（移除后性能大幅下降）；$\mathcal{L}_{reg}$ 防止过拟合；Full fine-tuning 可能因过拟合单张图像而性能反而下降。*

### 5.2 TTA Steps Sensitivity

*表 4: 自适应步数 $T$ 的影响（TIP2018 → FHDMi）。*

| $T$ | 0 (no TTA) | 1 | 5 | 10 | 15 (default) | 20 | 30 |
|-----|------------|---|---|----|--------------|----|----|
| PSNR (dB) | - | - | - | - | - | - | - |

*预期：$T=0$ 等于无 TTA；$T=5-10$ 快速收敛；$T=15$ 接近饱和；$T>20$ 收益递减。*

### 5.3 Frequency Attenuation Strength $\alpha$

*表 5: 频率衰减强度 $\alpha$ 的影响。*

| $\alpha$ | 0.0 | 0.3 | 0.5 (default) | 0.7 | 1.0 |
|----------|-----|-----|---------------|-----|-----|
| PSNR (dB) | - | - | - | - | - |

*预期：$\alpha=0$ 等于无 MASS 信号 → 无改善；$\alpha=1.0$ 过度衰减导致纹理丢失 → 性能下降；最优在 0.3-0.7。*

### 5.4 Backbone Compatibility

*表 6: MoTA 在不同骨干网络上的效果（TIP2018 → FHDMi）。*

| Backbone | w/o MoTA (frozen) | w/ MoTA | Δ PSNR |
|----------|-------------------|---------|--------|
| MoiréNet (CNN, 5.5M) | - | - | - |
| WDNet (Transformer) | - | - | - |
| WDNet (Wavelet CNN) | - | - | - |
| ESDNet (UHD CNN) | - | - | - |

*预期：MoTA 在所有骨干上均获得正向 Δ PSNR，验证模型无关性。*

### 5.5 Adapter Parameter Efficiency

*表 7: 不同适配策略的参数量与性能对比（TIP2018 → FHDMi，MoiréNet 骨干）。*

| Strategy | Trainable Params (M) | % of Backbone | PSNR (dB) |
|----------|---------------------|---------------|-----------|
| No adaptation | 0 | 0% | - |
| Full fine-tuning | 5.51 | 100% | - |
| LoRA (rank=4, all layers) | ~0.25 | ~4.5% | - |
| MoiréXNet TTT (attention layers) | ~1.65 | ~30% | - |
| **MoTA (FDA+SGA)** | **~0.17** | **~3.1%** | **-** |

*预期：MoTA 以最少可训练参数获得最佳跨域性能。*

## 6. Supplementary Experiments

### 6.1 Efficiency Analysis

*表 8: 推理效率对比（TIP2018 → FHDMi，单张 FHD 图像，A100 GPU）。*

| Method | Inference Time (ms) | TTA Overhead (ms) | Total Time (ms) | GPU Memory (GB) |
|--------|--------------------|--------------------|------------------|-----------------|
| MoiréNet | - | 0 | - | - |
| WDNet | - | 0 | - | - |
| MoiréXNet | - | - | - | - |
| **MoTA (MoiréNet)** | - | - | - | - |

*TTA Overhead = MASS 生成时间 + T 步适配器更新总时间。*

### 6.2 Qualitative Results

*[FIGURE: Qualitative comparison of demoiréing results on cross-domain samples. Left to right: Input moiré image (FHDMi sample), MoiréNet output (over-smoothed, residual moiré patterns visible), MoiréXNet output (better but still artifacts), MoTA output (ours, closest to GT). Zoomed-in patches highlight key differences: MoTA better preserves fine texture details while removing moiré patterns.]*

*[FIGURE: Heatmap visualization of MASS signal. Input image, MFD output M (moiré probability map overlaid), frequency-band visualization showing LH/HL suppression. Demonstrates that MASS accurately localizes moiré-dominant frequency regions.]*

### 6.3 Cross-Domain Generalization Visualization

*[FIGURE: t-SNE visualization of feature representations before and after MoTA adaptation. Two clusters: source domain (TIP2018, blue) and target domain (FHDMi, red). Left: before adaptation, clusters are separated (domain gap). Right: after MoTA adaptation, target domain features align with source domain distribution.]*

## 7. Reproducibility Checklist

- [x] 所有数据集为公开数据集
- [x] 代码将开源（含 MoTA 框架 + 预训练 MFD + 所有配置）
- [x] 随机种子固定 (42, 123, 2026)，结果报告均值 ± 标准差
- [x] Baseline 使用官方代码或详细复现说明
- [x] 所有超参数在 3.2 节列出

---

*注：所有表格中 "-" 表示待填充真实实验结果。实验方案设计完成，等待实际运行后填入数据。*
