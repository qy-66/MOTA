# 实验执行指南

Paper 需要填满 **5 张表 + 1 张诊断表**，WDNet backbone 全量实验 + DDA frozen baseline。

> **2026-05-25 最终版**：FDA-only 为推荐配置 (24.12 dB, +2.63 over frozen)。SGA 降为 explored variant (FDA+SGA 并行竞争导致退化)。Table 8+9 合并为 tab:main_results。eval.py 新增 --adapter_variant 参数。T 敏感度表改为 FDA-only 数据。

> **2026-05-22 重要修复**：适配器 hook 位置过滤、ablation 改用 ID 数据集、_IdentityAdapter 占位、SGA channels 属性。

---

## 实验矩阵总览

| 表号 | 名称 | 数据集 | 行数 | 说明 |
|------|------|--------|------|------|
| Table 1 | In-Domain | LCDMoire test | 3 | WDNet frozen, DDA frozen, **MoTA (WDNet+FDA-only)** |
| Table 2 | Cross-Domain | TIP2018 test | 4 | WDNet frozen, DDA frozen, WDNet MASS-pre, MoTA (WDNet) |
| Table 3 | Ablation | LCDMoire test | 6 | WDNet backbone, 6 变体 |
| Table 4a | T sensitivity | LCDMoire test | 6 | WDNet backbone, `bash fill_table4.sh` |
| Table 4b | Efficiency | LCDMoire test | 4 | WDNet backbone, `bash fill_table4.sh` |
| Table 5 | Speed | 单张 TIP2018 | 3 | WDNet, DDA, MoTA (WDNet) |
| Table 6 | CD Diagnostics | TIP2018 subset | 5 | **新增**: MASS质量 / MFD行为 / 适配器漂移 / CD消融 |

> ⚠️ **2026-05-23 叙事变化**：FDA-only (+2.63 dB) > FDA+SGA (+0.87 dB)，因此 **Table 1/3/4b 中 MoTA 以 FDA-only 为推荐配置**。SGA 在论文中作为 "explored variant" 描述。

> ⚠️ **DDA (MBCNN) 限制**：DDA 内部有 4× 下采样（256×256 patch 训练），1024×1024 输入时输出 256×256。MoTA 的 pseudo_clean 缩放 + bilinear upsampling 导致梯度太弱，TTA 优化无法收敛。因此 DDA 仅在 Table 1/2 中报 frozen baseline 数字。

---

## 提前准备

| 物品 | 说明 |
|------|------|
| AutoDL | RTX 4090 / A100，PyTorch ≥ 1.13 |
| LCDMoire 数据集 | `/root/autodl-tmp/LCDMoire/` |
| TIP2018 测试集 | `/root/autodl-tmp/Tip-2018/testdata/`（注意大小写） |
| WDNet 代码 | `/root/autodl-tmp/WDNet_demoire-master/` |
| DDA 代码（已修改） | `/root/autodl-tmp/DDA-main/` |
| 我的代码 | `/root/autodl-tmp/demoireing-paper/code/` |

### autodl 最终目录结构

```
/root/autodl-tmp/
├── LCDMoire/
│   ├── train/
│   │   ├── moire/          # 10000 张摩尔纹 (AIM 2019 官方)
│   │   └── clean/          # 10000 张干净图
│   ├── val_moire/          # 100 张（AIM 2019 官方 val，Table 1 同域评测用）
│   └── val_clean/          # 100 张
│
├── Tip-2018/               # ⚠️ 注意大小写和连字符
│   └── testdata/
│       ├── source/          # 摩尔纹 ~11851 张（Table 2 跨域评测用）
│       └── target/          # 干净图 ~11851 张
│
├── WDNet_demoire-master/
├── DDA-main/                # 已修改（dataset.py + train.py + test.py）
└── demoireing-paper/
    └── code/
        ├── train_wdnet_lcdmoire.py   # WDNet 训练
        ├── train.py                   # MFD + 适配器初始化
        ├── eval.py                    # 评测主脚本（自动判断 WDNet/DDA 加载）
        ├── benchmark_speed.py         # Table 5 推理速度计时
        ├── diagnose_cd.py             # 🆕 跨域诊断（MASS质量/MFD行为/漂移/CD消融）
        ├── run_multi_seed.sh          # 多 seed 重复评测 (mean±std)
        ├── fill_table4.sh            # Table 4 批量实验
        ├── unified_dataloader.py
        ├── requirements.txt
        └── mota/
            ├── __init__.py
            ├── adapters.py            # FDA + SGA
            ├── mfd.py                 # MFD 检测器 + DWT/IDWT
            ├── mass.py                # MASS 伪干净信号
            ├── tta.py                 # TTA 循环 (4 种模式)
            ├── wdnet_loader.py        # WDNet 加载工具
            └── utils.py               # 共享工具 (to_01/to_11/set_seed)
```

### 前置：确认代码中的路径

`train_wdnet_lcdmoire.py` 和 `unified_dataloader.py` 中硬编码了 `/root/autodl-tmp/`，确认与你 autodl 实际目录一致。

**代码修改已完成清单**（直接上传覆盖即可）：

| 文件 | 改了什么 |
|------|---------|
| `DDA-main/dataset/dataset.py` | 新增 `_is_image()` 过滤非图片；`clear` → `clean` 适配 LCDMoire |
| `DDA-main/train.py` | 去掉 class1/2/3 硬编码，三个 DataLoader 共享同一份数据 |
| `DDA-main/test.py` | SSIM `multichannel=True` → `channel_axis=-1` |
| `code/eval.py` | 自动加载 WDNet/DDA；支持 `--mass_preprocess` / `--tta_mode`；完整 ablation 6 变体 |
| `code/mota/tta.py` | 新增 `full_ft_adapt()` 和 `lora_adapt()` 用于 Table 4b |
| `code/mota/mass.py` | 内置 `generate_mass_signal()` MASS 信号生成 |

---

## tmux 使用指南（所有长时间训练必须用）

作为代码小白，你只需要记住 4 个操作：

| 操作 | 命令 | 什么时候用 |
|------|------|-----------|
| 创建 session | `tmux new -s 名字` | 开始训练前 |
| **断开（最关键）** | 先按 `Ctrl+B`，松开，再按 `D` | 想关终端走人 |
| 重新连接 | `tmux attach -t 名字` | 下次登录想回来看进度 |
| 查看所有 session | `tmux ls` | 忘了自己开了哪些 |

流程演示：

```bash
# 1. SSH 登录 autodl
ssh root@xxx

# 2. 创建 tmux session
tmux new -s dda

# 3. 在 tmux 里正常跑训练
cd /root/autodl-tmp/DDA-main
python main.py ...

# 4. 训练跑起来了，想关电脑走人
#    按 Ctrl+B，松手，再按 D
#    看到 [detached] 说明已断开

# 5. 关闭终端/关机，训练继续跑

# 6. 第二天重新登录
ssh root@xxx
tmux attach -t dda    # 回到训练界面，看到实时进度
```

### 命名约定（避免搞混）

| 训练任务 | tmux 名字 |
|---------|----------|
| WDNet 训练 | `wdnet` |
| DDA 训练 | `dda` |
| MFD 训练 | `mfd` |
| 适配器初始化 | `adapter` |
| Table 1 评测 | `eval1` |
| Table 2 评测 | `eval2` |
| Ablation | `ablation` |
| T 敏感度 + 效率 | `table4` |

### 注意事项

- **Autodl 实例重启会丢失所有 tmux session**，但磁盘上的 checkpoint 不受影响，只需重新跑未完成的步骤即可
- 退出了 tmux 不知道怎么回来？先 `tmux ls` 看看有哪些 session，然后用 `tmux attach -t 名字` 连回去
- tmux 里报错想退出？先 `Ctrl+C` 停掉命令，然后输入 `exit` 退出 tmux
- 评测命令（几分钟到一小时）可以不用 tmux，但用了更安全

---

## 第一步：装环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-image lpips piq tqdm opencv-python matplotlib scipy colour
pip install torchnet 2>/dev/null || pip install git+https://github.com/pytorch/tnt.git
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 第二步：确认数据就绪

```bash
ls /root/autodl-tmp/LCDMoire/train/moire/ | head -3   # 应 ~10000 张
ls /root/autodl-tmp/LCDMoire/train/clean/ | head -3   # 应 ~10000 张
ls /root/autodl-tmp/LCDMoire/val_moire/ | wc -l       # 应 100 张 (AIM 2019 官方 val)
ls /root/autodl-tmp/LCDMoire/val_clean/ | wc -l       # 应 100 张
ls /root/autodl-tmp/Tip-2018/testdata/source/ | wc -l # 应 ~11851 张
ls /root/autodl-tmp/Tip-2018/testdata/target/ | wc -l # 应 ~11851 张
```

> ⚠️ **路径注意**：`unified_dataloader.py` 中 `DATASET_ROOTS` 的路径必须与实际一致。如 TIP2018 实际路径为 `/root/autodl-tmp/Tip-2018/`（大小写/连字符差异），需要对应修改。

---

## 第三步：训练 WDNet Baseline（约 12h）

> ⚠️ **2026-05-20 修复**：`train_wdnet_lcdmoire.py` 中 VGGPerceptualLoss 的 forward 存在数据流 bug（slice1/slice2/slice3 独立作用在原始输入而非串联），已于 2026-05-20 修复。如果之前训过 WDNet，需要用修复后的代码重新训练。

### 执行

```bash
# 1. 获取 wavelet 参数（不用 tmux，就几秒）
cd /root/autodl-tmp/WDNet_demoire-master
rm -rf WaveletSRNet  # 如果之前克隆失败，删掉重来
git clone https://ghproxy.com/https://github.com/hhb072/WaveletSRNet.git
cp WaveletSRNet/wavelet_weights_c2.pkl ./
ls -lh wavelet_weights_c2.pkl

# 2. 创建 tmux session 并开始训练
tmux new -s wdnet
cd /root/autodl-tmp/demoireing-paper/code
PYTHONIOENCODING=utf-8 python train_wdnet_lcdmoire.py \
    --epochs 200 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 1e-4

# 3. 训练跑起来后，Ctrl+B D 断开，关终端走人
# 4. 回来看进度：tmux attach -t wdnet
```

产物：`checkpoints/wdnet_best.pth`

### 📊 自动记录 & 👀 你需要观察的

| 数据 | 来源 | 如何记录 |
|------|------|---------|
| 每 10 epoch 的 val PSNR | 终端输出 | **你注意看**：PSNR 是否持续上升，收敛到多少 |
| Early stopping 触发时 epoch | 终端输出 | **你记下来**：最终停在哪个 epoch |
| `wdnet_best.pth` | 磁盘文件 | 自动保存，不要动 |
| 最终 val PSNR | 终端最后一行 | **你记下来** → 后续填入 Table 1 WDNet 行 |

**正常期望**：val PSNR 应在 21-28 dB 范围（AIM 2019 官方 val 集 100 张）。不同训练 run 之间可能有 ~2 dB 波动。

---

## 第四步：训练 DDA Baseline（约 2-3 天，100 epoch；实际早停+LR衰减会提前退出）

> ✅ **已完成 (2026-05-20)**：65 epoch 收敛，best PSNR ~38 dB，实际耗时 19h。产物：`DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth`

### 前置：准备小验证集（避免验证9000张太慢）

```bash
mkdir -p /root/autodl-tmp/LCDMoire/val
ln -sf /root/autodl-tmp/LCDMoire/val_moire /root/autodl-tmp/LCDMoire/val/moire
ln -sf /root/autodl-tmp/LCDMoire/val_clean /root/autodl-tmp/LCDMoire/val/clean
```

### 执行

```bash
tmux new -s dda
cd /root/autodl-tmp/DDA-main

python main.py \
    --dataset aim \
    --traindata_path /root/autodl-tmp/LCDMoire/train/ \
    --testdata_path /root/autodl-tmp/LCDMoire/val/ \
    --arch MBCNN \
    --batchsize 4 \
    --max_epoch 100 \
    --lr 1e-4 \
    --patch_size 256 \
    --save_prefix /root/autodl-tmp/DDA-main/result \
    --save_every 5

# Ctrl+B D 断开，回来看：tmux attach -t dda
```

产物：
- `result/MBCNN_aim/1pth_folder/ckpt_best.pth`
- `result/MBCNN_aim/1pth_folder/ckpt_last.pth`（续跑用）

### 📊 自动记录 & 👀 你需要观察的

| 数据 | 来源 | 如何记录 |
|------|------|---------|
| 每 10 epoch val PSNR | `result/MBCNN_aim/log.txt` | 自动写日志，**你看终端也行** |
| 训练 PSNR 趋势（iter 级别） | 终端输出 | 确认 PSNR 从 ~7 dB 涨到 > 20 dB |
| `ckpt_best.pth` | 磁盘 | 自动保存，不要动 |
| `log.txt` | 磁盘 | 自动记录，保留备查 |

**正常期望**：训练 PSNR 应快速从 ~7 dB 涨到 ~20+ dB，最终 val PSNR 参考值 28+ dB。

---

## 第五步：训练 MoTA 组件（约 4h）

### 5.1 预训练 MFD（约 3h）

```bash
tmux new -s mfd
cd /root/autodl-tmp/demoireing-paper/code
python train.py --train_mfd --mfd_epochs 50
# 可调整 --mfd_epochs 控制训练轮数，默认 50
# Ctrl+B D 断开
```

产物：`checkpoints/mfd.pth`

### 👀 你需要观察的

| 数据 | 如何记录 |
|------|---------|
| MFD 训练 loss | 每 10 epoch 打印一次，应持续下降 |
| 最终 loss | 应该在 0.01 以下 |

### 5.2 初始化适配器（约 30min）

上面的命令直接跑即可——`train.py` 已内置 WDNet 加载代码。

产物：`checkpoints/adapters_init.pth`

**WDNet 适配器跑完后，需要给 DDA 再跑一次**。打开 `train.py`，搜索 `加载 WDNet backbone`，把那一整段（约 30 行，从 `WDNET_DIR = ...` 到 `print("WDNet backbone loaded")`）全部替换为：

```python
        # ====== 加载 DDA backbone ======
        sys.path.insert(0, "/root/autodl-tmp/DDA-main")
        from Net.MBCNN import MBCNN
        backbone = MBCNN(64)

        state = torch.load("/root/autodl-tmp/DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth",
                          map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        backbone.load_state_dict(state)
        backbone.eval()
        print("DDA backbone loaded")
```

然后把下面一行的保存路径从 `adapters_init.pth` 改为 `adapters_init_dda.pth`，保存后执行：

```bash
tmux new -s adapter
cd /root/autodl-tmp/demoireing-paper/code
python train.py --init_adapters
# Ctrl+B D 断开
```

产物：`checkpoints/adapters_init_dda.pth`

### 👀 你需要观察的

| 数据 | 如何记录 |
|------|---------|
| Adapter init loss | 10 epoch 应全部下降到 < 0.001（理想时 < 0.0001） |

---

## 第六步：Table 1 · 同域评测 (LCDMoire)

**含义**：在 LCDMoire 官方验证集（100 对，AIM 2019 标准划分）上评估。MoTA 使用 FDA-only 配置（推荐配置，+2.63 dB over frozen），FDA+SGA 组合因 representation competition 仅 +0.87 dB。

### 执行

```bash
tmux new -s eval1
cd /root/autodl-tmp/demoireing-paper/code

# 1. WDNet frozen
python eval.py --mode id --checkpoint checkpoints/wdnet_best.pth --tag wdnet

# 2. DDA frozen (MoTA 不适用于 DDA，仅报 frozen baseline)
python eval.py --mode id \
    --checkpoint /root/autodl-tmp/DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth \
    --tag dda

# 3. MoTA (FDA-only) on WDNet  ← 推荐配置
python eval.py --mode ablation \
    --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --tag wdnet
# → 从 ablation.json 取 "w/o SGA (FDA only)" 的 PSNR = 24.12 dB 作为 Table 1 MoTA 值

# 跑完后 Ctrl+B D 断开，或 exit 退出
```

### 📊 自动记录 & 👀 你需要观察的

| 数据 | 自动保存位置 | 你要做什么 |
|------|------------|-----------|
| WDNet frozen PSNR/SSIM/LPIPS | `results/id_frozen_wdnet.json` | 填入 Table 1 WDNet frozen 行 |
| DDA frozen PSNR/SSIM/LPIPS | `results/id_frozen_dda.json` | 填入 Table 1 DDA frozen 行 |
| WDNet MoTA PSNR/SSIM/LPIPS | `results/id_mota_wdnet.json` | 填入 Table 1 MoTA(WDNet) 行 |
| DDA MoTA PSNR/SSIM/LPIPS | `results/id_mota_dda.json` | 填入 Table 1 MoTA(DDA) 行 |
| 每 50 张图的进度打印 | 终端 | 扫一眼确认 PSNR 稳定 |

### 填入 Table 1

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params (M) |
|--------|-------------|--------|---------|------------|
| WDNet | 21.49 | 0.4024 | 0.3267 | 40.1M |
| DDA | 29.01 | 0.9342 | 0.1265 | 64.4M |
| **MoTA (WDNet+FDA)** | **24.12** | --- | --- | 40.1M + 0.43M |

> DDA 仅报 frozen baseline（架构限制，见 Table 2 说明）。MoTA 以 FDA-only 为推荐配置。SSIM/LPIPS 待补（ablation 仅算 PSNR）。

> **Params(M)**：在 Python 里跑 `sum(p.numel() for p in model.parameters()) / 1e6`。WDNet 约 40M，DDA MBCNN 约 64M。FDA-only 适配器约 0.43M。

### 👀 关键判断

FDA-only MoTA PSNR 应提升 **+1.5 到 +3.0 dB** over frozen。若 w/o SGA (FDA-only) 在 ablation 中显著高于 Full MoTA → confirm representation competition。

---

## 第七步：Table 2 · 跨域评测 (LCDMoire → TIP2018)

**核心实验**：在 TIP2018 真实屏幕拍摄数据上评估泛化。从没见过的设备 → 真实场景。

### 执行

```bash
tmux new -s eval2
cd /root/autodl-tmp/demoireing-paper/code

# ---- frozen baseline ----
python eval.py --mode cd --checkpoint checkpoints/wdnet_best.pth --tag wdnet
python eval.py --mode cd \
    --checkpoint /root/autodl-tmp/DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth \
    --tag dda

# ---- MASS as pre-processing ----
python eval.py --mode cd --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth --mass_preprocess --tag wdnet

# ---- MoTA FDA-only (推荐) ----
python eval.py --mode cd --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --use_mota --adapter_variant fda_only --T 15 --tag fda_cd

# ---- MoTA FDA+SGA (用于消融对比) ----
python eval.py --mode cd --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --use_mota --T 15 --tag full_cd

# Ctrl+B D 断开
```

### 📊 自动记录 & 👀 你需要观察的

| 数据 | 自动保存位置 | 你要做什么 |
|------|------------|-----------|
| WDNet frozen PSNR/SSIM/LPIPS | `results/cd_frozen_wdnet.json` | 填入 Table 2 WDNet frozen 行 |
| DDA frozen PSNR/SSIM/LPIPS | `results/cd_frozen_dda.json` | 填入 Table 2 DDA frozen 行 |
| WDNet MASS-pre PSNR/SSIM/LPIPS | `results/cd_mass_pre_wdnet.json` | 填入 Table 2 WDNet MASS-pre 行 |
| DDA MASS-pre PSNR/SSIM/LPIPS | `results/cd_mass_pre_dda.json` | 填入 Table 2 DDA MASS-pre 行 |
| WDNet MoTA PSNR/SSIM/LPIPS | `results/cd_mota_wdnet.json` | 填入 Table 2 MoTA(WDNet) 行 |
| DDA MoTA PSNR/SSIM/LPIPS | `results/cd_mota_dda.json` | 填入 Table 2 MoTA(DDA) 行 |

### 填入 Table 2

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | CD-Gain ↓ |
|--------|-------------|--------|---------|-----------|
| WDNet (frozen) | 18.91 | 0.3885 | 0.3725 | 2.58 |
| DDA (frozen) | 12.26 | 0.3822 | 0.5415 | 16.75 |
| WDNet + MASS-preprocess | 18.90 | 0.3902 | 0.3730 | 2.59 |

> DDA CD 12.26 dB 因 MBCNN 在 256×256 patch 训练，全分辨率推理时内部 4× 下采样导致严重退化。论文 footnote 已说明。

> MoTA CD 结果（~17.50 dB）不在表格中单列，而在 4.4.1 诊断分析中讨论。

> **CD-Gain** = PSNR(Table 1) − PSNR(Table 2)。越小越好 = 跨域退化越少。**eval.py 现在自动计算**：跑完 CD 评测后若同域 `id_frozen_{tag}.json` 存在，自动打印 CD-Gain。

### 👀 关键判断

| 信号 | 含义 |
|------|------|
| Frozen CD PSNR 比 ID PSNR 低 ≥ 2 dB | 域偏移确实存在（预期内） |
| **MoTA CD PSNR < Frozen CD PSNR** | CD TTA 失败（预期内） — **诚实报告** |
| MASS-pre ≈ Frozen | MASS 预处理单独无效，需 TTA 循环 |
| MASS pseudo-clean PSNR < 19 dB | CD 失败根因（见诊断实验） |

> ⚠️ **当前结论**：MoTA CD 不涨（~17.50 dB vs frozen 18.91 dB）。论文诚实报告这一负结果，并通过诊断实验分析根因。**不要尝试"调参"来强行让 CD 涨** — exploratory paper 的价值在于诚实分析失败原因。

---

## 第八步：Table 3 · Ablation（消融实验）

### 执行

```bash
tmux new -s ablation
cd /root/autodl-tmp/demoireing-paper/code

python eval.py --mode ablation \
    --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --tag wdnet

# Ctrl+B D 断开
```

自动跑完 6 组变体：Full MoTA (FDA+SGA)、w/o FDA (SGA only)、w/o SGA (**FDA only — 推荐配置**)、w/o MASS（以输入自身为 pseudo-clean target，真实 TTA 循环）、w/o L_reg、w/o TTA。

### 📊 自动记录

| 数据 | 保存位置 |
|------|---------|
| 6 组 PSNR/SSIM | `results/ablation.json` |

### 👀 你需要手动算的

| 变体 | Δ vs Full 计算方式 |
|------|-------------------|
| Full MoTA | 0.00（基准） |
| 其他 5 组 | `PSNR(变体) - PSNR(Full MoTA)` — 应为**负数** |

### 填入 Table 3

| Variant | PSNR (dB) ↑ | SSIM ↑ | Δ vs Full |
|---------|-------------|--------|-----------|
| Full MoTA (FDA+SGA) | 22.36 | 0.4852 | 0.00 |
| w/o FDA (SGA only) | 23.56 | --- | **+1.20** |
| **w/o SGA (FDA only) ← 推荐** | **24.12** | --- | **+1.76** |
| w/o MASS (L1 on input) | 14.68 | --- | −7.68 |
| w/o L_reg | 13.45 | --- | −8.91 |
| w/o TTA (frozen) | 21.49 | 0.4024 | −0.87 |

> ⚠️ **Δ vs Full 列**：正值 = 优于 Full MoTA (FDA+SGA)。FDA-only 和 SGA-only 均为正值 → 两个适配器组合后**互相竞争而非互补**。论文据此推荐 FDA-only 作为 MoTA 配置。

### 👀 关键判断

| 信号 | 含义 |
|------|------|
| w/o MASS 降最多 | MASS 是最关键组件 |
| **w/o SGA (FDA-only) > Full** | FDA 单独工作优于 FDA+SGA 组合 → **representation competition** |
| w/o FDA (SGA-only) 也 > Full | SGA 单独也优于组合 → 两个适配器互相竞争而非互补 |
| w/o TTA = frozen | 消融的下界 |

> ⚠️ **重要结论**：FDA-only (+2.63 dB) > SGA-only (+2.07 dB) ≫ FDA+SGA (+0.87 dB)。**论文以 FDA-only 为 MoTA 推荐配置**，SGA 在 Section 3.4.5 中以 "explored variant" 描述。

---

## 第八步½：Table 6 · 跨域诊断实验 🆕

**含义**：定量诊断 CD TTA 失败根因（审稿人最关心的数据点）。4 项子实验。

### 执行

```bash
cd /root/autodl-tmp/demoireing-paper/code

# 快速诊断（跳过慢速的 drift + CD ablation）
python diagnose_cd.py \
    --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --n_images 100 \
    --skip_drift \
    --output results/diagnostics

# 完整诊断（含适配器漂移 + CD 消融，约 2-3h）
python diagnose_cd.py \
    --checkpoint checkpoints/wdnet_best.pth \
    --mfd_checkpoint checkpoints/mfd.pth \
    --adapter_checkpoint checkpoints/adapters_init.pth \
    --n_images 100 \
    --n_drift_images 10 \
    --output results/diagnostics
```

### 📊 自动保存

| 文件 | 内容 |
|------|------|
| `results/diagnostics/diag1_mass_quality.json` | MASS pseudo-clean PSNR/SSIM vs GT |
| `results/diagnostics/diag2_mfd_behavior.json` | MFD 激活统计 (ID vs CD) |
| `results/diagnostics/diag3_adapter_drift.json` | ||φ_t − φ_0|| 轨迹 (ID vs CD) |
| `results/diagnostics/diag4_cd_ablation.json` | CD w/o MASS / w/o L_reg PSNR |
| `results/diagnostics/diagnostic_summary.json` | 汇总 |

### 👀 关键判断

| 诊断 | 期望 | 如果异常 |
|------|------|---------|
| MASS pseudo-clean PSNR | < 19 dB → CD TTA 失败根因确认 | > 20 dB → MASS 不是瓶颈，问题在别处 |
| MFD CD vs ID 激活比 | < 0.7× → MFD 跨域退化 | > 0.9× → MFD 跨域尚可 |
| 适配器 CD vs ID 漂移比 | < 0.5× 或 > 2× → 梯度过弱/过拟合 | ~1× → 适配器更新正常 |
| CD w/o MASS ΔPSNR | 接近 0 (不像 ID 的 −7.68) → MASS 在 CD 失效 | < −3 dB → MASS 在 CD 仍有用 |

### 填入 Table 6 (paper.tex `tab:cd_diagnostics`)

诊断数据跑完后，打开 `output/paper.tex`，搜索 `tab:cd_diagnostics`，将 `---` 替换为实际数值。

---

## 第九步：Table 4a + 4b · TTA 步数敏感度 + 适配器效率

**含义**：T 敏感度 (T=1,5,10,15,20) + Full FT + LoRA 基线，共 7 组实验。使用 LCDMoire ID (100 张)。

> ⚠️ **2026-05-22 更新**：Table 4 改用 ID 数据集（100 张，~15min/组），非 CD（11851 张，太慢）。新脚本 `fill_table4.sh` 自动跑完 7 组 + 汇总。

### 执行

```bash
tmux new -s table4
cd /root/autodl-tmp/demoireing-paper/code
bash fill_table4.sh
# Ctrl+B D 断开，回来：tmux attach -t table4
```

预计耗时：~110min（7 组 × ~15min）

### 📊 自动记录

| 模式 | 保存位置 |
|------|---------|
| T=1 | `results/table4/id_mota_wdnet_T1.json` |
| T=5 | `results/table4/id_mota_wdnet_T5.json` |
| T=10 | `results/table4/id_mota_wdnet_T10.json` |
| T=15 | `results/table4/id_mota_wdnet_T15.json` |
| T=20 | `results/table4/id_mota_wdnet_T20.json` |
| Full FT | `results/table4/id_full_ft_wdnet_fullft.json` |
| LoRA | `results/table4/id_lora_wdnet_lora.json` |

脚本跑完自动打印汇总。下载 JSON 到本地后运行 `output/collect_table4.py` 更新 paper.tex Table 4。

### ✅ 实际结果 (2026-05-22)

**Table 4a — T sensitivity:**
| T | 0 | 1 | 5 | 10 | 15 | 20 |
|---|----|---|---|----|----|----|
| PSNR (dB) | 21.49 | 21.74 | 22.32 | **22.72** | 22.36 | 21.62 |

峰值在 T=10（22.72 dB），T=20 过拟合降至 frozen 以下。

**Table 4b — Adapter efficiency:**
| Strategy | Trainable % | PSNR (dB) |
|----------|------------|-----------|
| No adaptation | 0% | 21.49 |
| Full fine-tuning | 100% | 10.44 |
| LoRA (rank=4) | ~4.5% | 23.31 |
| MoTA (FDA+SGA, T=15) | 1.7% | 22.36 |
| **MoTA (FDA-only, T=15)** | **1.1%** | **24.12** |

Full FT 单图崩塌（10.44 dB）；**FDA-only MoTA +2.63 dB 超越 LoRA +1.82 dB，且参数量仅 1/4**。FDA+SGA 组合因 representation competition 仅 +0.87 dB。

### 👀 关键判断

| 信号 | 含义 |
|------|------|
| T=10 已获大部分增益 | TTA 收敛快（针对 FDA+SGA 组合；FDA-only T sensitivity 待跑） |
| Full FT << Frozen | 单图全量微调灾难性遗忘 |
| FDA-only > LoRA | FDA 的频率选择性设计比通用低秩适配更适合去摩尔纹 |
| FDA-only > FDA+SGA | Representation competition — 论文核心发现之一 |

---

## 第十一步：Table 5 · 推理速度

### 执行

脚本已创建在 `code/benchmark_speed.py`，直接运行：

```bash
cd /root/autodl-tmp/demoireing-paper/code
python benchmark_speed.py
```

### 📊 你需要记录

| Method | Inference (ms) | TTA Overhead (ms) | Total (ms) |
|--------|---------------|--------------------|------------|
| WDNet | 终端打印 | 0 | = Inference |
| MoTA (T=15) | = WDNet Inference | = Total - Inference | 终端打印 |

> MoiréNet 和 MoiréXNet 代码不可用（paper 标记 "—"），跳过不跑。

---

## 执行顺序汇总

```
步骤      内容                    产物                    填入
─────────────────────────────────────────────────────────────
Step 3    训 WDNet               wdnet_best.pth           基础
Step 4    训 DDA                 ckpt_best.pth            基础
Step 5    训 MFD + 适配器         mfd.pth, adapters_init.pth  基础
Step 6    同域评测               3 组 PSNR/SSIM/LPIPS     Table 1
Step 7    跨域评测               4 组 PSNR/SSIM/LPIPS     Table 2
Step 8    Ablation              6 组 PSNR                Table 3
Step 8½   跨域诊断              5 组诊断数据              Table 6 🆕
Step 9    fill_table4.sh         7 组 PSNR                Table 4a+4b
Step 10   速度测试              ms                        Table 5
```

---

## 数据记录速查表

### 计算机自动保存（`results/` 目录下）

| 文件 | 内容 | 对应表 |
|------|------|--------|
| `id_frozen_wdnet.json` | WDNet frozen（LCDMoire） | Table 1 |
| `id_frozen_dda.json` | DDA frozen（LCDMoire） | Table 1 |
| `id_mota_wdnet.json` | MoTA(WDNet)（LCDMoire） | Table 1 |
| `cd_frozen_wdnet.json` | WDNet frozen（TIP2018） | Table 2 |
| `cd_frozen_dda.json` | DDA frozen（TIP2018） | Table 2 |
| `cd_mass_pre_wdnet.json` | WDNet MASS-pre | Table 2 |
| `cd_mota_wdnet.json` | MoTA(WDNet) | Table 2 |
| `results/table4/id_mota_wdnet_T*.json` | T 敏感度 (1,5,10,15,20) | Table 4a |
| `results/table4/id_full_ft_wdnet_fullft.json` | Full FT | Table 4b |
| `results/table4/id_lora_wdnet_lora.json` | LoRA | Table 4b |
| `ablation.json` | 6 组 PSNR/SSIM | Table 3 |
| `diagnostics/diag1_mass_quality.json` | MASS pseudo-clean PSNR/SSIM | Table 6 🆕 |
| `diagnostics/diag2_mfd_behavior.json` | MFD 激活统计 (ID vs CD) | Table 6 🆕 |
| `diagnostics/diag3_adapter_drift.json` | 适配器漂移轨迹 | Table 6 🆕 |
| `diagnostics/diag4_cd_ablation.json` | CD 消融 PSNR | Table 6 🆕 |

### 你需要手动记录/计算的

| 数据 | 怎么得到 |
|------|---------|
| Params(M) | `python -c "import torch; ...; print(sum(p.numel()..."` |
| CD-Gain | eval.py 自动打印（需先跑 Table 1） |
| Δ vs Full (Table 3) | PSNR(变体) − PSNR(Full MoTA) |
| T 敏感度 PSNR (Table 4a) | 终端打印，6 个数字 |
| Trainable % (Table 4b) | 脚本打印可训参数占比 |
| 推理速度 (Table 5) | `benchmark_speed.py` 终端打印 |

---

## 期望结论

```
                          WDNet backbone    DDA backbone
────────────────────────────────────────────────────────
Frozen (baseline)          21.49             29.01
MoTA (FDA-only, ours)      24.12 (+2.63)     N/A (架构限制)
LoRA (rank=4)              23.31 (+1.82)     —
FDA+SGA (explored)         22.36 (+0.87)     —
```

**核心发现**：
1. **FDA-only (+2.63 dB) > LoRA (+1.82 dB)** — 频率选择性适配器比通用低秩适配更适合此任务
2. **FDA-only > FDA+SGA** — parallel adapters compete, not complement（representation competition）
3. **MASS 是最关键组件** — w/o MASS: −7.68 dB
4. **CD TTA 失败** — MASS pseudo-clean 在 real captures 上质量不足，MFD 跨域退化
5. DDA 因架构限制仅报 frozen baseline；"architecture-flexible" 声明在论文中限定为 WDNet full TTA + DDA frozen comparison

---

## 常见问题

**Q: 显存爆了？**
```bash
python train_wdnet_lcdmoire.py --batch_size 2 --grad_accum 8
```

**Q: wavelet_weights_c2.pkl 下载失败？**
```bash
git clone https://gitclone.com/github.com/hhb072/WaveletSRNet.git
git clone https://kgithub.com/hhb072/WaveletSRNet.git
```

**Q: 训练中断了怎么办？**
- WDNet: 暂无断点续跑，重头来
- DDA: `--resume /path/to/ckpt_last.pth`

**Q: LPIPS 报错？**
```bash
pip install lpips   # 首次运行自动下载 AlexNet 权重
```

**Q: `results/` 下的 JSON 文件怎么读？**
```bash
python -c "import json; d=json.load(open('results/cd_frozen.json')); print(d['avg'])"
```

**Q: tmux session 断开后忘了名字？**
```bash
tmux ls    # 列出所有 session 名字
```

**Q: 退出 tmux 后训练还在跑吗？**
在的。只要你不输入 `exit` 或 `Ctrl+C` 停掉训练进程，训练继续跑。`Ctrl+B D` 只是断开你的界面连接。

## eval.py 2026-05-25 更新

**新增参数:**
- `--adapter_variant {full, fda_only, sga_only}` — 仅在 id/cd 模式下生效，ablation 模式忽略
- `--T` 现在对 ablation 模式也生效 (之前硬编码 T=15)

**T=0 处理:** eval.py 自动退化为 frozen 推理 (不进入 TTA 循环)

**T 敏感度 sweep:**
```bash
for T in 0 1 5 10 15 20; do
    python eval.py --mode ablation --T $T --tag ab_T${T} \
        --checkpoint checkpoints/wdnet_best.pth \
        --mfd_checkpoint checkpoints/mfd.pth \
        --adapter_checkpoint checkpoints/adapters_init.pth
done
echo "提取 FDA-only (w/o SGA) PSNR 填入 T sensitivity 表"
```

**Q: Autodl 实例重启了怎么办？**
tmux session 会丢失，但磁盘上的 checkpoint 不受影响。重新创建 tmux session，从对应步骤重跑未完成的部分。DDA 可以用 `--resume` 续跑。

**Q: 我想跑一个新命令但正在 tmux 里？**
先 `Ctrl+B D` 断开回到普通终端，再开一个新的 tmux session 跑新命令。多个 tmux session 互不干扰。
