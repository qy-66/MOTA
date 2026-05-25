# PAPER_SPEC.md — MoTA 论文再生规格文件

> **用途**：将此文件 + `references.bib` 交给 AI，即可从头生成/修改论文。
> **最后更新**：2026-05-23

---

## 1. 论文基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Domain-Adaptive Demoiréing via Model-Agnostic Test-Time Adaptation |
| 简称 | MoTA (Moiré Test-Time Adaptation) |
| 模板 | CVPR 双栏 (cvpr.sty), 10pt, letterpaper |
| 目标会议 | CVPR |
| 当前页数 | 15页 (含参考文献) |
| 关键词 | image demoiréing, test-time adaptation, frequency domain, adapter, self-supervision |

---

## 2. 核心叙事弧线 (Narrative Arc)

```
问题: 现有去摩尔纹模型严重域过拟合 → 换设备掉 3-5 dB
方案: 测试时自适应 (TTA) → 轻量适配器 + MASS 自监督信号
结果: ID (+0.87 dB) ✓ → CD (反而下降) ✗
诚实分析: MFD 跨域失效 + 适配器容量不足 + FDA/SGA 竞争
贡献: 首个 model-agnostic demoiréing TTA 框架 + 跨域评估协议 + 失败分析
```

**关键原则**：所有声称必须与实验数据一致。不要承诺跨域提升——数据不支持。

---

## 3. 章节蓝图

### Abstract
- **长度**：~200 词
- **结构**：问题 → 方法名 → 三个组件 (FDA, SGA, MASS) → ID 结果 (+0.87 dB) → CD 结果 (未提升) → 分析方向
- **关键数据**：WDNet backbone, <5% 参数, T=15 steps

### Section 1: Introduction
- **长度**：~1 页
- **段落结构**：
  1. 摩尔纹问题 + 现有方法进展 (PSNR 26→30 dB)
  2. 域过拟合问题：换设备/屏幕类型掉 3-5 dB
  3. TTA 作为替代方案 + 现有 TTA 方法的局限
  4. MoTA 方案概述 (FDA + SGA + MASS)
  5. 三个 contribution
- **Contribution 措辞已修改**：诚实反映 CD 负结果，"identify challenges" 而非 "achieve SOTA"

### Section 2: Related Work
- **子节**：
  - 2.1 Image Demoiréing (频域方法 / 高效率 / RAW域 / 视频)
  - 2.2 Test-Time Adaptation for Restoration
  - 2.3 Parameter-Efficient Adaptation
- **总结句**：三类方法的交叉空白 → MoTA 填补

### Section 3: Method
- **3.1 Overview**：三阶段流水线 (MASS → Adapter Forward → TTA Optimization)
- **3.2 Preliminaries**：DWT, 问题公式化
- **3.3 Analysis of Existing Approaches**：Table 分析表
- **3.4 MASS Signal**：MFD + 频带衰减 + 伪干净目标
- **3.5 FDA**：channel-wise 1D FFT + 低秩对角调制
- **3.6 SGA**：spatial gating with depthwise conv
- **3.7 Loss Function**：L1 + L_reg
- **3.8 Algorithm 1**：伪代码
- **3.9 Theoretical Analysis**：为什么 MASS 不会坍缩为低通滤波器

### Section 4: Experiments
- **4.1 Datasets**：LCDMoire (合成) → TIP2018 (真实), Table
- **4.2 Metrics**：PSNR, SSIM, LPIPS, CD-Gain, Table
- **4.3 Implementation Details**：训练配置 Table, TTA 超参
- **4.4 Main Results**：
  - Table 1 (ID): WDNet 21.49 → MoTA 22.36 (+0.87 dB)
  - Table 2 (CD): WDNet frozen 18.91, MoTA 未提升
- **4.5 Cross-Domain Failure Analysis**：三个因素
- **4.6 Ablation**：Table 3 (6 variants), Table 4 (T sensitivity + adapter efficiency)
- **4.7 Qualitative**：Fig 2 (定性对比), Fig 3 (MASS 可视化)
- **4.8 Efficiency**：Table 5 (推理速度)

### Section 5: Conclusion
- 总结贡献 + ID 结果 + CD 挑战 + 未来方向

### Limitations
- 5 条：MFD 依赖 / 单帧 / 单域对 / FDA-SGA 竞争 / LoRA baseline

---

## 4. 数据表 (所有数值来源)

### Table 1: In-Domain Results (LCDMoire 测试集, 100 对)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params |
|--------|-------------|--------|---------|--------|
| WDNet (frozen) | 21.49 | 0.4024 | 0.3267 | 40.1M |
| DDA (frozen) | 29.01 | 0.9342 | 0.1265 | 64.4M |
| MoTA (WDNet) | **22.36** | **0.4852** | **0.3426** | +0.68M |

> SSIM 异常低 (21.49 dB → 0.4024) 是 LCDMoire 数据集特征：合成摩尔纹覆盖模式不成比例地破坏结构信息。

### Table 2: Cross-Domain Results (LCDMoire → TIP2018, 11851 对)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | CD-Gain ↓ |
|--------|-------------|--------|---------|-----------|
| WDNet (frozen) | 18.91 | 0.3885 | 0.3725 | 2.58 |
| DDA (frozen) | 12.26 | 0.3822 | 0.5415 | 16.75 |
| WDNet + MASS-preprocess | 18.90 | 0.3902 | 0.3730 | 2.59 |

> MoTA CD 部分运行 (7350/11851, ~17.5 dB)。诚实结论：did not improve over frozen。

### Table 3: Ablation (LCDMoire ID, WDNet backbone)

| Variant | PSNR (dB) ↑ | Δ vs Full |
|---------|-------------|-----------|
| Full MoTA | 22.36 | 0.00 |
| w/o FDA (SGA only) | 23.56 | +1.20 |
| w/o SGA (FDA only) | 24.12 | +1.76 |
| w/o MASS | 14.68 | −7.68 |
| w/o L_reg | 13.45 | −8.91 |
| w/o TTA (frozen) | 21.49 | −0.87 |

> 关键发现：FDA-only 和 SGA-only 分别优于 Full MoTA，说明两适配器在共享插入点竞争。

### Table 4: TTA Steps Sensitivity + Adapter Efficiency

| T | 0 | 1 | 5 | 10 | 15 | 20 |
|---|-----|-----|-----|------|------|------|
| PSNR | 21.49 | 21.74 | 22.32 | **22.72** | 22.36 | 21.62 |

| Strategy | Trainable % | PSNR |
|----------|------------|------|
| No adaptation | 0% | 21.49 |
| Full fine-tuning (T=15) | 100% | 10.44 |
| LoRA (rank=4, T=15) | ~4.5% | **23.31** |
| MoTA (FDA+SGA, T=15) | 1.7% | 22.36 |

> T=10 是峰值。LoRA 比 MoTA 高 0.95 dB，但参数量 2.6x。

### Table 5: Inference Efficiency (RTX 4090, 单张 TIP2018)

| Method | Inference (ms) | TTA Overhead (ms) | Total (ms) |
|--------|---------------|-------------------|------------|
| WDNet | 9.4 | 0 | 9.4 |
| DDA | 52.9 | 0 | 52.9 |
| MoTA (WDNet, T=15) | 9.4 | 779.2 | 788.6 |

---

## 5. 图表清单

| 编号 | 文件 | 尺寸 | 位置 | 描述 |
|------|------|------|------|------|
| Fig 1 | `figures/fig1_framework.pdf` | 跨双栏 figure* | Sec 3.1 | MoTA 框架概览：输入→MASS→Adapter→输出，三阶段颜色标注 |
| Fig 2 | `figures/qualitative.pdf` | 跨双栏 figure* | Sec 4.7 | 定性对比：LCDMoire 输入、TIP2018 输入、WDNet 输出、MoTA 输出、GT |
| Fig 3 | `figures/fig3_mass.pdf` | 跨双栏 figure* | Sec 4.7 | MASS 可视化：MFD heatmap + 四个子带衰减前后对比 |

---

## 6. LaTeX 排版规则 (从调试中总结)

### 浮动体规范
- 所有 table/figure/algorithm 使用 `[tbp]`，禁止 `[h]`（双栏模式下 [h] 导致中栏挤压重叠）
- 跨双栏大图/大表使用 `figure*[tp]` / `table*[tp]`
- 连续 figure* 之间不要加 `\FloatBarrier`（会导致两个 figure* 争抢同一页顶）

### 浮动体间距
```latex
\setlength{\textfloatsep}{14pt plus 6pt minus 4pt}
\setlength{\dbltextfloatsep}{14pt plus 6pt minus 4pt}
\setlength{\intextsep}{14pt plus 6pt minus 4pt}
\setlength{\floatsep}{10pt plus 4pt minus 2pt}
```

### 浮动体参数
```latex
\renewcommand{\topfraction}{0.75}
\renewcommand{\bottomfraction}{0.5}
\renewcommand{\floatpagefraction}{0.8}
\raggedbottom
```

### FloatBarrier 位置
- 每个 `\section` 之前
- Experiments 内部：Implementation Details 之后、CD failure analysis 之后
- 不要放在连续 figure* 之后

### 表格宽度
- 全局 `\footnotesize`
- 宽表用 `\setlength{\tabcolsep}{3pt}` 压缩列间距
- 长文本列用 `p{宽度}` 允许换行，不要用 `l`

---

## 7. 参考文献要求

- 格式：IEEEtran
- 文件：`references.bib` (28 条)
- 关键引用分布：
  - Demoiréing: sun2018moire, zheng2022learning, liu2020wavelet, he2020fhde2net, yu2022towards, anonymous2025moirenet, yang2025unidemoire
  - TTA: tang2026degradation, gou2024testtime, li2026testtime, mansour2024tttmim, deng2023efficient
  - PEFT: anonymous2026frair, anonymous2025biradapter
  - RAW-domain: li2025moirexnet, xu2024image
  - VGG loss: simonyan2014very

---

## 8. 已知问题 & 待修改

| 优先级 | 问题 | 说明 |
|--------|------|------|
| P0 | SSIM 异常低 | WDNet 21.49 dB → SSIM 0.4024，正常应 ~0.65-0.80。当前论文归因于"LCDMoire 数据集特征"，需核实 |
| P0 | CD 负结果表述 | 需在 Introduction 中诚实预告 CD 失败，而非暗示成功 |
| P1 | Fig 3 内容验证 | 4.1MB PDF 内容未核实是否为正确的 MASS 可视化 |
| P1 | 表格溢出 | 14 个 Overfull hbox，最大 77pt (Table 1)，需进一步调宽或改 `table*` |
| P2 | qualitative.pdf 压缩 | 7.5MB → 目标 <1MB |

---

## 9. 关键文件索引

| 文件 | 作用 | AI 是否必须读 |
|------|------|:---:|
| `PAPER_SPEC.md` | **本文件** — 论文再生总控 | ✅ |
| `PROJECT_HANDOFF.md` | 实验数据 + 架构决策 + 代码路径 | ✅ |
| `CLAUDE.md` | 项目配置 + 审查历史 + 远端命令 | ✅ |
| `references.bib` | 28 条 BibTeX 引用 | ✅ |
| `output/paper.tex` | 当前 LaTeX 源文件 (含所有排版修复) | ✅ |
| `output/cvpr.sty` | CVPR 格式文件 | ✅ |
| `output/table4_results/*.json` | Table 4 的 7 组原始实验数据 | 可选 |
| `research_state.json` | 项目阶段 + 审稿分数 | 可选 |
| `output/review_output.md` | 模拟审稿意见 (3 位审稿人, 均分 3.33) | 可选 |
| `experiment_guide.md` | 实验执行步骤 | 可选 |
| `code/**/*.py` | 实现代码 | 可选 |
| `figures/**/*` | 图源文件 | 可选 |
| `sections_original_plan/*.md` | **已废弃**的原始章节草案，与实际论文不一致 | ❌ 不要读 |

---

## 10. AI 再生论文流程

1. **读本文件** → 理解论文结构、数据、约束
2. **读 `PROJECT_HANDOFF.md`** → 获取实验细节和架构决策
3. **读 `CLAUDE.md`** → 获取项目上下文和代码审查记录
4. **读 `references.bib`** → 获取所有引用
5. **读 `output/paper.tex`** → 获取当前 LaTeX 源码（可选，如需从零重写则跳过）
6. **生成/修改 `paper.tex`** → 遵循第 6 节的排版规则
7. **编译** → `pdflatex paper.tex && pdflatex paper.tex`
8. **验证** → 0 Underfull vbox, 0 unprocessed floats, 检查关键页面无重叠
