#!/bin/bash
# 多 seed 评测脚本 — 论文 Table 1-4 的 mean±std 需要 3 次独立训练 + 3 次评测。
# 如果只做了单次训练，本脚本对同一次训练的 checkpoint 重复评测 3 次
# （捕获运行时随机性，但主要方差来源是训练随机性）。
#
# 用法:
#   bash run_multi_seed.sh --mode cd --checkpoint checkpoints/wdnet_best.pth --use_mota --tag wdnet
#
# 产物: results/{mode}_{suffix}_{tag}_multi.json (含 mean/std)

SEEDS=(42 123 2026)
ALL_JSONS=""

# 提取用户传入的 --tag 值（支持 --tag value 和 --tag=value 两种写法）
USER_TAG=""
next_is_tag=0
for arg in "$@"; do
    case "$arg" in
        --tag=*) USER_TAG="${arg#*=}" ;;
        --tag)   next_is_tag=1 ;;
        *)       if [ "$next_is_tag" = "1" ]; then
                     USER_TAG="$arg"
                     next_is_tag=0
                 fi ;;
    esac
done

for seed in "${SEEDS[@]}"; do
    TAG="${USER_TAG:+${USER_TAG}_}seed${seed}"
    echo "=== Run seed=$seed tag=$TAG ==="
    python eval.py "$@" --seed "$seed" --tag "$TAG"
    # 找到本次运行产出的 JSON 文件（最新）
    LATEST=$(ls -t results/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        ALL_JSONS="$ALL_JSONS $LATEST"
    fi
done

echo ""
echo "=== 多 seed 汇总 ==="
python -c "
import json, sys, os

files = '$ALL_JSONS'.strip().split()
psnr_vals, ssim_vals, lpips_vals = [], [], []

for f in files:
    if not os.path.exists(f): continue
    d = json.load(open(f))
    avg = d.get('avg', d)  # ablation.json 无 'avg' 键
    if 'psnr_mean' in avg:
        psnr_vals.append(avg['psnr_mean'])
        ssim_vals.append(avg['ssim_mean'])
        lpips_vals.append(avg.get('lpips_mean', 0))

if psnr_vals:
    import statistics as st
    print(f'PSNR:  {st.mean(psnr_vals):.2f} ± {st.stdev(psnr_vals):.2f} dB  (n={len(psnr_vals)})')
    print(f'SSIM:  {st.mean(ssim_vals):.4f} ± {st.stdev(ssim_vals):.4f}')
    print(f'LPIPS: {st.mean(lpips_vals):.4f} ± {st.stdev(lpips_vals):.4f}')
else:
    print('No results found. Check that eval.py completed successfully.')
"
