#!/bin/bash
# ============================================================
# MoTA 实验管线 — 一键运行脚本
# 在 AutoDL RTX 4090 实例上执行:
#   chmod +x run_pipeline.sh
#   bash run_pipeline.sh
# ============================================================
set -e

# ---- 配置 (改这里) ----
DATA_DIR="/root/autodl-tmp"
RESULTS_DIR="./results"
CKPT_DIR="./checkpoints"

# ---- 0. 环境检查 ----
echo "========================================="
echo "Step 0: 环境检查"
echo "========================================="
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# 检查数据目录
echo ""
echo "检查数据集..."
for ds in LCDMoire TIP2018; do
    if [ -d "$DATA_DIR/$ds" ]; then
        count=$(find "$DATA_DIR/$ds" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
        echo "  $ds: OK ($count images)"
    else
        echo "  $ds: NOT FOUND — 请先上传数据到 $DATA_DIR/$ds"
        exit 1
    fi
done

# 检查 WDNet 代码
if [ ! -d "$DATA_DIR/WDNet" ]; then
    echo "  WDNet 代码未找到，正在 clone..."
    git clone https://github.com/xyLiu339/WDNet.git "$DATA_DIR/WDNet" || {
        echo "  Clone 失败，请手动下载: git clone https://github.com/xyLiu339/WDNet.git"
    }
fi

mkdir -p $RESULTS_DIR $CKPT_DIR

# ---- 1. 训练 WDNet Baseline ----
echo ""
echo "========================================="
echo "Step 1: 训练 WDNet baseline (约 12h)"
echo "========================================="
# 注意: 需要先修改 unified_dataloader.py 中的 DATASET_ROOTS 为你的实际路径
python train.py 2>&1 | tee $RESULTS_DIR/train_baseline.log

# ---- 2. 预训练 MFD ----
echo ""
echo "========================================="
echo "Step 2: 预训练 MFD (约 3h)"
echo "========================================="
python train.py --train_mfd 2>&1 | tee $RESULTS_DIR/train_mfd.log

# ---- 3. 初始化 MoTA 适配器 ----
echo ""
echo "========================================="
echo "Step 3: 初始化适配器 (约 30min)"
echo "========================================="
python train.py --init_adapters 2>&1 | tee $RESULTS_DIR/init_adapters.log

# ---- 4. 同域评测 ----
echo ""
echo "========================================="
echo "Step 4: 同域评测 (LCDMoire)"
echo "========================================="
python eval.py --mode id --checkpoint $CKPT_DIR/best_baseline.pth 2>&1 | tee $RESULTS_DIR/eval_id.log

# ---- 5. 跨域评测 (frozen) ----
echo ""
echo "========================================="
echo "Step 5: 跨域评测 (frozen, 无 TTA)"
echo "========================================="
python eval.py --mode cd --checkpoint $CKPT_DIR/best_baseline.pth 2>&1 | tee $RESULTS_DIR/eval_cd_frozen.log

# ---- 6. 跨域评测 (MoTA) ----
echo ""
echo "========================================="
echo "Step 6: 跨域评测 (MoTA, T=15)"
echo "========================================="
python eval.py --mode cd --checkpoint $CKPT_DIR/best_baseline.pth \
    --mfd_checkpoint $CKPT_DIR/mfd.pth \
    --adapter_checkpoint $CKPT_DIR/adapters_init.pth \
    --use_mota --T 15 2>&1 | tee $RESULTS_DIR/eval_cd_mota.log

# ---- 7. Ablation ----
echo ""
echo "========================================="
echo "Step 7: Ablation study"
echo "========================================="
python eval.py --mode ablation --checkpoint $CKPT_DIR/best_baseline.pth \
    --mfd_checkpoint $CKPT_DIR/mfd.pth \
    --adapter_checkpoint $CKPT_DIR/adapters_init.pth 2>&1 | tee $RESULTS_DIR/ablation.log

# ---- 8. 汇总结果 ----
echo ""
echo "========================================="
echo "Step 8: 汇总结果"
echo "========================================="
python - <<'PYEOF'
import json, os, glob

print("\n" + "="*60)
print("实验管线完成! 结果汇总:")
print("="*60)

for f in sorted(glob.glob("results/*.json")):
    name = os.path.basename(f)
    data = json.load(open(f))
    if "avg" in data:
        avg = data["avg"]
        print(f"\n{name}:")
        print(f"  PSNR:  {avg.get('psnr_mean', 'N/A')} dB")
        print(f"  SSIM:  {avg.get('ssim_mean', 'N/A')}")
        print(f"  LPIPS: {avg.get('lpips_mean', 'N/A')}")
    elif "Full MoTA" in data:
        print(f"\n{name} (Ablation):")
        for k, v in data.items():
            print(f"  {k:20s}  PSNR={v.get('psnr_mean','N/A')} dB")

print("\n把上面 ↑ 的 PSNR/SSIM/LPIPS 值填入 paper.md 的 '-' 位置。")
PYEOF

echo ""
echo "========================================="
echo "全部完成!"
echo "========================================="
