"""
评测脚本: 在同域 (LCDMoire) 和跨域 (LCDMoire→TIP2018) 上评测。

用法:
  python eval.py --mode id               # 同域评测
  python eval.py --mode cd               # 跨域评测
  python eval.py --mode cd --use_mota     # 跨域 + MoTA TTA
  python eval.py --mode ablation          # Ablation study
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataloader import get_dataloader
from mota.mfd import MFD
from mota.adapters import insert_mota_adapters, FDA, SGA
from mota.tta import mota_adapt, full_ft_adapt, lora_adapt, no_mass_adapt
from mota.wdnet_loader import load_wdnet_backbone, load_dda_backbone, _strip_module_prefix, psnr_from_mse

# 模块级 LPIPS 加载（避免每次 compute_metrics 重复构造模型）
_LPIPS_FN = None
try:
    import lpips as _lpips
    _LPIPS_FN = _lpips.LPIPS(net='alex')
except ImportError:
    pass

# 模块级 SSIM 可用性
_SSIM_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _SSIM_AVAILABLE = True
except ImportError:
    pass


def compute_metrics(out, tgt):
    """计算 PSNR, SSIM, LPIPS (需要 pip install lpips)。"""
    mse = nn.functional.mse_loss(out, tgt)
    psnr = psnr_from_mse(mse)

    # SSIM
    ssim_val = 0.0
    if _SSIM_AVAILABLE:
        out_np = (out.squeeze(0).cpu() + 1.0) / 2.0
        tgt_np = (tgt.squeeze(0).cpu() + 1.0) / 2.0
        out_img = (out_np.permute(1,2,0).numpy() * 255).astype('uint8')
        tgt_img = (tgt_np.permute(1,2,0).numpy() * 255).astype('uint8')
        ssim_val = _ssim_fn(out_img, tgt_img, channel_axis=2, data_range=255)

    # LPIPS
    lpips_val = 0.0
    if _LPIPS_FN is not None:
        lpips_val = _LPIPS_FN.to(out.device)(out, tgt).item()

    return {
        "psnr": round(psnr.item(), 2),
        "ssim": round(ssim_val, 4),
        "lpips": round(lpips_val, 4),
    }


def evaluate(model, dataloader, device="cuda", use_mota=False,
             adapters=None, mfd=None, T=15, mass_preprocess=False,
             lambda_reg=1e-6, tta_mode="mota"):
    """评测循环。"""
    model = model.to(device).eval()

    if use_mota and T == 0:
        use_mota = False  # T=0 退化为 frozen 推理

    if use_mota and tta_mode == "mota" and adapters is not None:
        adapters = adapters.to(device).train()

    from mota.mass import generate_mass_signal

    all_metrics = []
    for i, (inp, tgt, path) in enumerate(dataloader):
        inp = inp.to(device)
        tgt = tgt.to(device)

        if mass_preprocess and mfd is not None:
            inp = generate_mass_signal(inp, mfd)
            with torch.no_grad():
                out = model(inp)
        elif use_mota:
            if tta_mode == "no_mass" and adapters is not None:
                out = no_mass_adapt(inp, model, adapters, T=T, lambda_reg=lambda_reg)
            elif tta_mode == "full_ft" and mfd is not None:
                out = full_ft_adapt(inp, model, mfd, T=T, lambda_reg=lambda_reg)
            elif tta_mode == "lora" and mfd is not None:
                out = lora_adapt(inp, model, mfd, T=T, lambda_reg=lambda_reg)
            elif mfd is not None and adapters is not None:
                out = mota_adapt(inp, model, adapters, mfd, T=T, lambda_reg=lambda_reg)
            else:
                with torch.no_grad():
                    out = model(inp)
        else:
            with torch.no_grad():
                out = model(inp)

        if out.shape[-2:] != tgt.shape[-2:]:
            out = nn.functional.interpolate(out, size=tgt.shape[-2:], mode='bilinear', align_corners=False)

        metrics = compute_metrics(out, tgt)
        metrics["path"] = path[0] if isinstance(path, (list, tuple)) else str(path)
        all_metrics.append(metrics)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataloader)}] "
                  f"PSNR={metrics['psnr']:.1f}, SSIM={metrics['ssim']:.4f}")

    # 汇总
    avg = {
        "psnr_mean": round(sum(m["psnr"] for m in all_metrics) / len(all_metrics), 2),
        "ssim_mean": round(sum(m["ssim"] for m in all_metrics) / len(all_metrics), 4),
        "lpips_mean": round(sum(m["lpips"] for m in all_metrics) / len(all_metrics), 4),
        "num_samples": len(all_metrics),
    }
    return avg, all_metrics


def run_ablation(model, dataloader, adapters, mfd, T=15, device="cuda"):
    """跑组件消融实验。T 控制所有变体的适配步数。"""
    results = {}

    print(f"\n--- Full MoTA (T={T}) ---")
    avg, _ = evaluate(model, dataloader, device, use_mota=True,
                      adapters=adapters, mfd=mfd, T=T)
    results["Full MoTA"] = avg

    class _IdentityAdapter(nn.Module):
        """恒等占位符，用于 ablation：不做任何变换"""
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
        def forward(self, x):
            return x

    print(f"\n--- w/o FDA (SGA only, T={T}) ---")
    adapters_sga = nn.ModuleList()
    for i in range(0, len(adapters), 2):
        adapters_sga.append(_IdentityAdapter(adapters[i].channels))
        adapters_sga.append(adapters[i + 1])  # SGA
    avg, _ = evaluate(model, dataloader, device, use_mota=True,
                      adapters=adapters_sga, mfd=mfd, T=T)
    results["w/o FDA"] = avg

    print(f"\n--- w/o SGA (FDA only, T={T}) ---")
    adapters_fda = nn.ModuleList()
    for i in range(0, len(adapters), 2):
        adapters_fda.append(adapters[i])  # FDA
        adapters_fda.append(_IdentityAdapter(adapters[i].channels))
    avg, _ = evaluate(model, dataloader, device, use_mota=True,
                      adapters=adapters_fda, mfd=mfd, T=T)
    results["w/o SGA"] = avg

    print("\n--- w/o TTA (frozen) ---")
    frozen_avg, _ = evaluate(model, dataloader, device, use_mota=False)
    results["w/o TTA"] = frozen_avg

    print(f"\n--- w/o MASS (L1 on input, T={T}) ---")
    avg, _ = evaluate(model, dataloader, device, use_mota=True,
                      tta_mode="no_mass", adapters=adapters, T=T)
    results["w/o MASS"] = avg

    print(f"\n--- w/o L_reg (lambda_reg=0, T={T}) ---")
    avg, _ = evaluate(model, dataloader, device, use_mota=True,
                      adapters=adapters, mfd=mfd, T=T, lambda_reg=0.0)
    results["w/o L_reg"] = avg

    return results


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="id",
                        choices=["id", "cd", "ablation"])
    parser.add_argument("--use_mota", action="store_true")
    parser.add_argument("--tta_mode", type=str, default="mota",
                        choices=["mota", "full_ft", "lora", "no_mass"],
                        help="TTA 模式: mota=适配器, full_ft=全量微调, lora=LoRA基线")
    parser.add_argument("--mass_preprocess", action="store_true",
                        help="仅做 MASS 预处理后 frozen 推理，不迭代 TTA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wdnet_best.pth")
    parser.add_argument("--mfd_checkpoint", type=str, default="checkpoints/mfd.pth")
    parser.add_argument("--adapter_checkpoint", type=str, default="checkpoints/adapters_init.pth")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--tag", type=str, default="",
                        help="输出文件区分标签，如 'wdnet'/'dda'。为空则不追加。")
    parser.add_argument("--T", type=int, default=15)
    parser.add_argument("--adapter_variant", type=str, default="full",
                        choices=["full", "fda_only", "sga_only"],
                        help="适配器变体: full=FDA+SGA, fda_only=FDA, sga_only=SGA (仅 id/cd 模式)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="重复评测次数（>1 时用不同 seed 重复评测并报告 mean±std）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 输出目录转为绝对路径（避免相对路径问题）
    args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mode: {args.mode} | MoTA: {args.use_mota} | TTA: {args.tta_mode} | MASS-pre: {args.mass_preprocess}")

    # LPIPS 模型一次性迁移到目标设备
    if _LPIPS_FN is not None:
        _LPIPS_FN = _LPIPS_FN.to(device)

    # ---- 选择数据集 ----
    if args.mode in ("id", "ablation"):
        loader = get_dataloader("LCDMoire", split="test", batch_size=1)
        print("Evaluation: LCDMoire (In-Domain)")
    elif args.mode == "cd":
        loader = get_dataloader("TIP2018", split="test", batch_size=1)
        print("Evaluation: TIP2018 (Cross-Domain)")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ---- 加载模型 ----
    if "wdnet" in args.checkpoint.lower():
        model = load_wdnet_backbone(args.checkpoint, device)
        print("WDNet pipeline loaded")

    else:
        from mota.wdnet_loader import load_dda_backbone
        model = load_dda_backbone(args.checkpoint, device=device)
        print("DDA MBCNN loaded")

    # ---- 加载 MFD ----
    mfd = MFD()
    if os.path.exists(args.mfd_checkpoint):
        mfd.load_state_dict(torch.load(args.mfd_checkpoint))
        mfd.to(device).eval()
        print("MFD loaded")
    elif args.use_mota or args.mass_preprocess or args.mode == "ablation":
        raise FileNotFoundError(
            f"MFD checkpoint not found at {args.mfd_checkpoint}. "
            "Run 'python train.py --train_mfd' first.")

    # ---- 加载适配器 ----
    adapters = None
    if (args.use_mota or args.mode == "ablation") and model is not None:
        # 支持新旧两种 adapter checkpoint 格式
        ckpt = torch.load(args.adapter_checkpoint) if os.path.exists(args.adapter_checkpoint) else None
        if ckpt is not None and isinstance(ckpt, dict) and "channels" in ckpt:
            adapters, channels = insert_mota_adapters(model, channels_per_stage=ckpt["channels"])
            adapters.load_state_dict(ckpt["state_dict"])
            print(f"Adapters loaded ({len(channels)} stages)")
        elif ckpt is not None:
            adapters, channels = insert_mota_adapters(model)
            adapters.load_state_dict(ckpt)
            print(f"Adapters loaded (auto-detected {len(channels)} stages)")
        else:
            adapters, channels = insert_mota_adapters(model)
            print(f"  [WARN] adapter checkpoint 未找到，使用随机初始化适配器")

    if args.use_mota and args.tta_mode == "mota" and adapters is None and model is not None:
        print("  [WARN] --use_mota 已设置但适配器为 None，将退化为 frozen 推理")

    # ---- 适配器变体过滤 (id/cd 模式) ----
    if args.adapter_variant != "full" and adapters is not None and args.mode != "ablation":
        class _IdentityAdapter(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.channels = channels
            def forward(self, x):
                return x

        variant_adapters = nn.ModuleList()
        for i in range(0, len(adapters), 2):
            fda_adapter = adapters[i]
            sga_adapter = adapters[i + 1]
            if args.adapter_variant == "fda_only":
                variant_adapters.append(fda_adapter)
                variant_adapters.append(_IdentityAdapter(fda_adapter.channels))
            elif args.adapter_variant == "sga_only":
                variant_adapters.append(_IdentityAdapter(sga_adapter.channels))
                variant_adapters.append(sga_adapter)
        adapters = variant_adapters
        # Count actual (non-identity) adapters
        n_active = sum(1 for a in adapters if not isinstance(a, _IdentityAdapter))
        print(f"Adapter variant: {args.adapter_variant} ({n_active} active adapters)")

    if not _SSIM_AVAILABLE:
        print("  [WARN] skimage 未安装，SSIM 将全为 0")
    if _LPIPS_FN is None:
        print("  [WARN] lpips 未安装，LPIPS 将全为 0")

    # ---- 跑评测（支持多 seed 重复评测报告 mean±std） ----
    # 论文声明的 3-seed 标准集合
    _STANDARD_SEEDS = [42, 123, 2026]
    run_seeds = _STANDARD_SEEDS[:args.n_runs] if args.n_runs <= 3 else [args.seed + i for i in range(args.n_runs)]

    def _aggregate_runs(all_runs, keys_override=None):
        """跨 run 聚合: 对每个 variant 的 psnr_mean/ssim_mean/lpips_mean 求 mean±std。"""
        if not all_runs:
            return {}
        keys = keys_override if keys_override is not None else all_runs[0].keys()
        agg = {}
        for key in keys:
            psnrs = [r[key]['psnr_mean'] for r in all_runs]
            ssims = [r[key]['ssim_mean'] for r in all_runs]
            lpipss = [r[key]['lpips_mean'] for r in all_runs]
            agg[key] = {
                'psnr_mean': round(float(np.mean(psnrs)), 2),
                'psnr_std':  round(float(np.std(psnrs, ddof=1) if len(psnrs) > 1 else 0), 2),
                'ssim_mean': round(float(np.mean(ssims)), 4),
                'ssim_std':  round(float(np.std(ssims, ddof=1) if len(ssims) > 1 else 0), 4),
                'lpips_mean': round(float(np.mean(lpipss)), 4),
                'lpips_std':  round(float(np.std(lpipss, ddof=1) if len(lpipss) > 1 else 0), 4),
            }
        return agg

    if args.mode == "ablation" and model is not None and adapters is not None:
        all_run_results = []
        for run_idx, run_seed in enumerate(run_seeds):
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            print(f"\n{'='*50}\nRun {run_idx+1}/{len(run_seeds)} (seed={run_seed})\n{'='*50}")
            results = run_ablation(model, loader, adapters, mfd, T=args.T, device=device)
            all_run_results.append(results)

        aggregated = _aggregate_runs(all_run_results)
        fname = f"{args.output}/ablation_{len(run_seeds)}seeds.json"
        json.dump({'aggregated': aggregated, 'per_run': all_run_results, 'seeds': run_seeds},
                  open(fname, "w"), indent=2)
        print(f"\nAblation results ({len(run_seeds)} seeds) → {fname}")
        for k, v in aggregated.items():
            print(f"  {k}: PSNR={v['psnr_mean']}±{v['psnr_std']} dB, "
                  f"SSIM={v['ssim_mean']}±{v['ssim_std']}")

    elif model is not None:
        if args.use_mota:
            suffix = args.tta_mode
        elif args.mass_preprocess:
            suffix = "mass_pre"
        else:
            suffix = "frozen"

        all_avgs = []
        for run_idx, run_seed in enumerate(run_seeds):
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            print(f"\n{'='*50}\nRun {run_idx+1}/{len(run_seeds)} (seed={run_seed})\n{'='*50}")
            avg, detail = evaluate(model, loader, device,
                                   use_mota=args.use_mota,
                                   mass_preprocess=args.mass_preprocess,
                                   tta_mode=args.tta_mode,
                                   adapters=adapters, mfd=mfd, T=args.T)
            avg['_seed'] = run_seed
            all_avgs.append(avg)

        # 将平铺的 avg dict 包装为单 variant 格式，复用 _aggregate_runs
        single_runs = [{'_single': a} for a in all_avgs]
        aggregated = _aggregate_runs(single_runs)

        tag_part = f"_{args.tag}" if args.tag else ""
        fname = f"{args.output}/{args.mode}_{suffix}{tag_part}_{len(run_seeds)}seeds.json"
        json.dump({
            'aggregated': aggregated,
            'per_run': all_avgs,
            'seeds': run_seeds,
            'num_samples': all_avgs[0]['num_samples'] if all_avgs else 0
        }, open(fname, "w"), indent=2)

        print(f"\n{args.mode.upper()} results ({len(run_seeds)} seeds) → {fname}")
        v = aggregated['_single']
        print(f"  PSNR:  {v['psnr_mean']}±{v['psnr_std']} dB")
        print(f"  SSIM:  {v['ssim_mean']}±{v['ssim_std']}")
        print(f"  LPIPS: {v['lpips_mean']}±{v['lpips_std']}")

        # CD-Gain auto-computation
        if args.mode == "cd":
            id_fname = f"{args.output}/id_frozen{tag_part}_1seeds.json"
            if os.path.exists(id_fname):
                id_data = json.load(open(id_fname))
                id_psnr = (id_data.get('aggregated', {})
                           .get('_single', id_data.get('avg', {}))
                           .get('psnr_mean', 0))
                cd_gain = round(id_psnr - v['psnr_mean'], 2)
                print(f"  CD-Gain: {cd_gain} dB  (ID={id_psnr} − CD={v['psnr_mean']})")
            else:
                print(f"  CD-Gain: N/A (run ID frozen eval first)")
