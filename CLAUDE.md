# MoTA 论文实验项目

## 每次对话必做

1. **Skill("using-superpowers")** — 确认该用哪个 skill 再动手
2. **读本文件** — 了解当前项目状态
3. **改完代码 → Skill("code-review-expert")** — 审查 SOLID/安全/边界
4. **说"完成"之前 → Skill("verification-before-completion")** — 跑验证

**远端调试铁律：**
- 整个 `code/` 目录覆盖上传，不逐文件传
- 传完 `grep` 关键行确认版本
- 先 5 张图 dry-run，确认正确再跑全量
- 跑前确认数据路径

## 项目目标

完成论文 "Domain-Adaptive Demoiréing via Architecture-Flexible Test-Time Adaptation"。

- Paper: `output/paper.tex` → `output/paper.pdf` (**16 页，0 警告**)
- 目标会议: CVPR 2026
- 实验指南: `experiment_guide.md`

## 目录结构

```
/root/autodl-tmp/
├── LCDMoire/              ← 合成训练/验证数据
├── Tip-2018/testdata/     ← 跨域测试 (~11851 对)
├── WDNet_demoire-master/  ← WDNet 代码
├── DDA-main/              ← DDA 代码
└── demoireing-paper/code/ ← 本项目核心代码
```

## 核心代码文件

| 文件 | 作用 |
|------|------|
| `code/eval.py` | 评测主入口。--mode id/cd/ablation, --use_mota, --adapter_variant full/fda_only/sga_only, --T N, --tta_mode, --tag, --n_runs N |
| `code/train.py` | MFD训练 + 适配器初始化 |
| `code/unified_dataloader.py` | 统一数据加载 (LCDMoire + TIP2018) |
| `code/run_multi_seed.sh` | 多 seed 重复评测 (但已确认 deterministic, std=0) |
| `code/mota/tta.py` | TTA 循环: mota_adapt / full_ft_adapt / lora_adapt / no_mass_adapt |
| `code/mota/mfd.py` | MFD 检测器 + DWT/IDWT |
| `code/mota/mass.py` | MASS 伪干净信号生成 |
| `code/mota/adapters.py` | FDA (rFFT + low-rank) + SGA (spatial gate) + insert_mota_adapters |
| `code/mota/utils.py` | to_01 / to_11 / set_seed / InputRangeAdapter |
| `code/diagnose_cd.py` | 跨域诊断 (MASS quality, MFD activation, adapter drift, CD ablation) |
| `code/visualize_mass.py` | MASS 信号可视化 |
| `code/visualize_qualitative.py` | 定性对比可视化 |
| `code/benchmark_speed.py` | 推理速度计时 |

## 论文当前数据 (2026-05-25 最终版)

**论文配置**: FDA-only = 推荐配置 (SGA 是 explored variant, 与 FDA 并行竞争导致退化)

| Table | 关键数据 |
|-------|------|
| tab:main_results (ID+CD 合并) | ID: WDNet 21.49 / DDA 29.01 / **MoTA+FDA 24.12** (+2.63 dB). CD: WDNet 18.91 / DDA 12.26 / MASS-pre 18.90 / MoTA+FDA 17.50 |
| tab:ablation (Table 3) | Full (FDA+SGA) 22.36 / FDA-only **24.12** (+2.63) / SGA-only 23.56 / w/o MASS 14.68 / w/o L_reg 13.45 / Frozen 21.49 |
| T sensitivity (FDA-only) | T=0:21.49 T=1:21.88 T=5:22.67 T=10:23.48 **T=15:24.12** T=20:23.92 — 峰值在 T=15 |
| Adaptation efficiency | Frozen 21.49 / Full FT 10.44 (崩塌) / LoRA (rank=4, 4.5%) 23.31 / FDA+SGA (1.7%) 22.36 / **FDA-only (1.1%) 24.12** |
| Inference speed | WDNet 9.4ms / DDA 52.9ms / MoTA+FDA 788.6ms (T=15) |
| CD diagnostics | MASS pseudo-clean 17.57 dB (2.76 dB BELOW frozen 20.33 on 100-image subset) |
| Complexity | MoTA+FDA: ~0.011N trainable params, O(rC) per step |

**论文叙事**：
- FDA-only 同域 +2.63 dB 有效，超 LoRA (4.1× params)
- 跨域不涨 (MoTA+FDA 17.50 < frozen 18.91) — MASS pseudo-clean 质量是瓶颈
- FDA+SGA 并行竞争意外发现 (单独用更好)
- MASS + L_reg 是关键组件 (ID 下移除各掉 7-9 dB)
- LoRA 4.1× params 仍低于 FDA-only (24.12 vs 23.31)

## 论文编译

```bash
cd output
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
# 验证: grep -c Warning paper.log 必须为 0
# 验证: grep "Output written" paper.log 确认 16 pages
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
```

**编译关键规则 (禁止违反):**
- **四步编译不可省略**: pdflatex → bibtex → pdflatex → pdflatex，缺一不可
- **编译间隙禁止删除 .aux**: 中间产物必须保留到全部编译完成
- **禁止用 \clearpage 或其他强制分页**: 会涨到 17+ 页
- **浮动体用 `[tbp]` 或 `[H]`**: `[H]` 仅用于 Algorithm 和 Complexity table
- **Table 8+9 已合并为 tab:main_results**: 不能拆回两张表
- **FloatBarrier 保留在**: L97(Fig1前), L317(Algorithm前), L369(Complexity后), L381(Experiments前), L461(Main Results前), L641(Limitations前)
- **日期格式**: `\date{May 2026}` (仅年月)
- **表格不重叠的根源**: 合并表减少浮动体数量 + FloatBarrier 队列管理

## eval.py 当前接口

```
--mode {id, cd, ablation}
--use_mota                    # 启用 MoTA 适配
--adapter_variant {full, fda_only, sga_only}  # 适配器变体 (id/cd 模式)
--tta_mode {mota, full_ft, lora, no_mass}
--mass_preprocess             # 仅 MASS 预处理后 frozen 推理
--T N                         # 自适应步数 (ablation 模式也生效)
--checkpoint PATH
--mfd_checkpoint PATH
--adapter_checkpoint PATH
--tag TAG                     # 输出文件后缀
--seed N                      # 随机种子 (default 42)
--n_runs N                    # 多 seed 重复 (已确认 deterministic, std=0)
```

**常用命令:**

```bash
# ID 评测
python eval.py --mode id --tag wdnet

# CD 评测 (frozen)
python eval.py --mode cd --tag wdnet

# CD 评测 (MoTA FDA-only)
python eval.py --mode cd --use_mota --adapter_variant fda_only --T 15 --tag fda_cd

# Ablation (T=15, 全部 6 组)
python eval.py --mode ablation --T 15 --tag ab_T15

# T 敏感度 sweep
for T in 0 1 5 10 15 20; do
    python eval.py --mode ablation --T $T --tag ab_T${T}
done

# 跨域诊断
python diagnose_cd.py --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --n_images 100 --output results/diagnostics
```

## 关键约定

- 所有长时间命令必须用 tmux
- T=0 时 eval.py 自动退化为 frozen 推理 (不进入 TTA 循环)
- adapter_variant 仅对 id/cd 模式生效；ablation 模式自动跑全部 variant
- ablation 的 `w/o SGA` = FDA-only, `w/o FDA` = SGA-only
- unified_dataloader 输出 [-1,1]；DDA 训练用 [0,1]，已通过 InputRangeAdapter 处理
- TTA 函数内部自动缩放 pseudo_clean 匹配模型输出分辨率
- WDNet wavelet 需输入尺寸为 4 的倍数
- TIP2018 实际路径: `/root/autodl-tmp/Tip-2018/` (注意大小写)
- eval.py 默认 cudnn.deterministic=True (所有评测 std=0)
- 不要编造数据: 所有数值必须有对应实验结果或代码支撑

## 已确认的复现数据 (2026-05-25)

| 实验 | T | 结果 | 状态 |
|------|:---:|------|:---:|
| ID frozen | 0 | 21.49 | ✅ 可复现 |
| ID FDA-only | 15 | 24.12 | ✅ 可复现 |
| ID FDA+SGA | 15 | 22.36 | ✅ 可复现 |
| ID SGA-only | 15 | 23.56 | ✅ 可复现 |
| CD frozen | 0 | 18.91 | ✅ |
| T sweep FDA-only | 0/1/5/10/15/20 | 21.49/21.88/22.67/23.48/24.12/23.92 | ✅ |

## 修复历史 (2026-05-25 关键轮次)

**数据修正:**
- 复杂度表 MoTA ~0.03N → ~0.011N (FDA-only 实际参数量)
- LoRA/FDA 参数量比率 2.6× → 4.1× (4.5%/1.1% = 4.09)
- Algorithm 加 M 下采样步 (合并到单行，不增行数)
- FDA 公式 FFT → rFFT，加 δ 截断说明
- 图说明 "closest to clean GT" → "showing improved detail retention on this example"
- Table 8+9 合并为 tab:main_results (减少浮动体冲突)

**排版修正:**
- Algorithm [H] 非浮动 + 注释内嵌
- Complexity table: center 环境 + 标签缩短 (MoiréXNet (TTT) → MoiréXNet)
- Table 3 底部注释 \small → \footnotesize
- FloatBarrier 6 处关键位置防止重叠

**CVPR 规范:**
- IEEEtran ------ 不是错误 (重名作者标准格式)
- 匿名引用缺失字段不改 (arXiv 预印本真实状态)
- 不编造缺失数据
