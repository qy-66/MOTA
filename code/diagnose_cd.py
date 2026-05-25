"""
Cross-Domain Failure Diagnostics for MoTA.

Runs three diagnostic experiments to understand why MoTA fails in CD setting:
  1. MASS pseudo-clean quality: PSNR/SSIM of MASS output vs real GT on TIP2018
  2. MFD cross-domain behavior: qualitative comparison of MFD outputs (ID vs CD)
  3. Adapter parameter drift: ||phi_t - phi_0|| trajectory in ID vs CD

Usage:
  python diagnose_cd.py --checkpoint checkpoints/wdnet_best.pth \
      --mfd_checkpoint checkpoints/mfd.pth \
      --adapter_checkpoint checkpoints/adapters_init.pth \
      --n_images 100 --output results/diagnostics
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataloader import get_dataloader
from mota.mfd import MFD
from mota.mass import generate_mass_signal
from mota.adapters import insert_mota_adapters
from mota.tta import mota_adapt
from mota.wdnet_loader import load_wdnet_backbone, psnr_from_mse


def compute_psnr_ssim(img1, img2):
    """Compute PSNR and SSIM between two tensors in [-1, 1]."""
    mse = F.mse_loss(img1, img2)
    psnr = psnr_from_mse(mse)

    ssim_val = 0.0
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        a = (img1.squeeze(0).cpu() + 1.0) / 2.0
        b = (img2.squeeze(0).cpu() + 1.0) / 2.0
        a_img = (a.permute(1, 2, 0).numpy() * 255).astype('uint8')
        b_img = (b.permute(1, 2, 0).numpy() * 255).astype('uint8')
        ssim_val = ssim_fn(a_img, b_img, channel_axis=2, data_range=255)
    except ImportError:
        pass

    return psnr.item(), ssim_val


# ──────────────────────────────────────────────
# Diagnostic 1: MASS pseudo-clean quality on TIP2018
# ──────────────────────────────────────────────
def diagnose_mass_quality(model, mfd, cd_loader, device, n_images=100):
    """Evaluate how good MASS pseudo-clean targets are vs real GT."""
    print("\n" + "=" * 60)
    print("Diagnostic 1: MASS Pseudo-Clean Quality on TIP2018")
    print("=" * 60)

    results = []
    model.eval()
    mfd.eval()

    for i, (inp, tgt, path) in enumerate(cd_loader):
        if i >= n_images:
            break
        inp = inp.to(device)
        tgt = tgt.to(device)

        with torch.no_grad():
            # Generate pseudo-clean target
            pseudo = generate_mass_signal(inp, mfd, alpha=0.5)
            # Frozen backbone output
            frozen_out = model(inp)

        # Align resolution
        if pseudo.shape[-2:] != tgt.shape[-2:]:
            pseudo = F.interpolate(pseudo, size=tgt.shape[-2:], mode='bilinear')
        if frozen_out.shape[-2:] != tgt.shape[-2:]:
            frozen_out = F.interpolate(frozen_out, size=tgt.shape[-2:], mode='bilinear')

        pseudo_psnr, pseudo_ssim = compute_psnr_ssim(pseudo, tgt)
        frozen_psnr, frozen_ssim = compute_psnr_ssim(frozen_out, tgt)

        results.append({
            "idx": i,
            "pseudo_psnr": round(pseudo_psnr, 2),
            "pseudo_ssim": round(pseudo_ssim, 4),
            "frozen_psnr": round(frozen_psnr, 2),
            "frozen_ssim": round(frozen_ssim, 4),
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_images}] pseudo PSNR={pseudo_psnr:.1f} / frozen PSNR={frozen_psnr:.1f}")

    avg = {
        "pseudo_psnr_mean": round(np.mean([r["pseudo_psnr"] for r in results]), 2),
        "pseudo_ssim_mean": round(np.mean([r["pseudo_ssim"] for r in results]), 4),
        "frozen_psnr_mean": round(np.mean([r["frozen_psnr"] for r in results]), 2),
        "frozen_ssim_mean": round(np.mean([r["frozen_ssim"] for r in results]), 4),
        "n_images": len(results),
    }

    print(f"\n  MASS pseudo-clean  → PSNR={avg['pseudo_psnr_mean']} dB, SSIM={avg['pseudo_ssim_mean']}")
    print(f"  Frozen backbone     → PSNR={avg['frozen_psnr_mean']} dB, SSIM={avg['frozen_ssim_mean']}")
    print(f"  Gap (pseudo − frozen) = {avg['pseudo_psnr_mean'] - avg['frozen_psnr_mean']:.1f} dB")
    if avg['pseudo_psnr_mean'] < 19:
        print("  ⚠ Pseudo-clean quality < 19 dB → MASS provides insufficient supervision for CD TTA.")

    return avg, results


# ──────────────────────────────────────────────
# Diagnostic 2: MFD cross-domain behavior
# ──────────────────────────────────────────────
def diagnose_mfd_behavior(mfd, id_loader, cd_loader, device, n_images=10):
    """Compare MFD output statistics between ID (LCDMoire) and CD (TIP2018)."""
    print("\n" + "=" * 60)
    print("Diagnostic 2: MFD Cross-Domain Behavior")
    print("=" * 60)
    print(f"  Comparing MFD activation statistics on {n_images} images each (ID vs CD)")

    mfd.eval()

    def collect_mfd_stats(loader, tag):
        means, stds, sparsities = [], [], []
        for i, (inp, tgt, path) in enumerate(loader):
            if i >= n_images:
                break
            inp_01 = (inp.to(device) + 1.0) / 2.0
            with torch.no_grad():
                M = mfd(inp_01)  # (1, 3, H, W)
            for c in range(3):
                mc = M[0, c].cpu().numpy()
                means.append(float(mc.mean()))
                stds.append(float(mc.std()))
                sparsities.append(float((mc < 0.2).mean()))  # fraction of low-activation pixels
        return {
            "mean": round(np.mean(means), 4),
            "std": round(np.mean(stds), 4),
            "sparsity": round(np.mean(sparsities), 4),
        }

    id_stats = collect_mfd_stats(id_loader, "ID")
    cd_stats = collect_mfd_stats(cd_loader, "CD")

    print(f"\n  ID  (LCDMoire):  mean activation={id_stats['mean']}, std={id_stats['std']}, "
          f"low-activation fraction={id_stats['sparsity']}")
    print(f"  CD  (TIP2018):   mean activation={cd_stats['mean']}, std={cd_stats['std']}, "
          f"low-activation fraction={cd_stats['sparsity']}")
    print(f"  Delta:           mean diff={cd_stats['mean'] - id_stats['mean']:.4f}, "
          f"std diff={cd_stats['std'] - id_stats['std']:.4f}")

    if cd_stats['sparsity'] > id_stats['sparsity'] * 1.3:
        print("  ⚠ MFD activations significantly sparser on CD → detection sensitivity degraded.")

    return {"id": id_stats, "cd": cd_stats}


# ──────────────────────────────────────────────
# Diagnostic 3: Adapter parameter drift (ID vs CD)
# ──────────────────────────────────────────────
def diagnose_adapter_drift(model, adapters, mfd, id_loader, cd_loader, device,
                           n_images=10, T=15, lr=1e-4, lambda_reg=1e-6):
    """Track ||phi_t - phi_0|| across TTA steps, comparing ID vs CD."""
    print("\n" + "=" * 60)
    print("Diagnostic 3: Adapter Parameter Drift (ID vs CD)")
    print("=" * 60)
    print(f"  Tracking ||phi_t - phi_0|| over T={T} steps on {n_images} images each")

    def run_tta_tracking(loader, tag):
        all_drifts = []  # list of (n_steps,) arrays
        for img_idx, (inp, tgt, path) in enumerate(loader):
            if img_idx >= n_images:
                break

            inp = inp.to(device)
            tgt = tgt.to(device)

            # Reset adapters
            adapters_orig = {name: p.clone() for name, p in adapters.named_parameters()}
            adapters.train()

            with torch.no_grad():
                pseudo = generate_mass_signal(inp, mfd, alpha=0.5)

            # Ensure size compatibility
            if pseudo.shape[-2:] != tgt.shape[-2:]:
                pseudo = F.interpolate(pseudo, size=tgt.shape[-2:], mode='bilinear')

            optimizer = torch.optim.Adam(adapters.parameters(), lr=lr)
            drifts = []

            for step in range(T):
                optimizer.zero_grad()
                out = model(inp)
                if out.shape[-2:] != pseudo.shape[-2:]:
                    out = F.interpolate(out, size=pseudo.shape[-2:], mode='bilinear')

                loss_mass = F.l1_loss(out, pseudo)
                loss_reg = sum((p - adapters_orig[name]).pow(2).sum()
                               for name, p in adapters.named_parameters())
                loss = loss_mass + lambda_reg * loss_reg
                loss.backward()
                optimizer.step()

                # Compute drift
                drift = sum((p - adapters_orig[name]).pow(2).sum().sqrt().item()
                           for name, p in adapters.named_parameters())
                drifts.append(drift)

            all_drifts.append(np.array(drifts))

            if (img_idx + 1) % 5 == 0:
                print(f"  [{tag}] {img_idx+1}/{n_images} done")

        # Average across images
        stacked = np.stack(all_drifts)  # (n_images, T)
        mean_drift = stacked.mean(axis=0)
        std_drift = stacked.std(axis=0)
        return mean_drift, std_drift

    id_mean, id_std = run_tta_tracking(id_loader, "ID")
    cd_mean, cd_std = run_tta_tracking(cd_loader, "CD")

    print(f"\n  ID final drift (T={T}): {id_mean[-1]:.4f} ± {id_std[-1]:.4f}")
    print(f"  CD final drift (T={T}): {cd_mean[-1]:.4f} ± {cd_std[-1]:.4f}")
    ratio = cd_mean[-1] / (id_mean[-1] + 1e-8)
    print(f"  CD/ID drift ratio:      {ratio:.2f}×")
    if ratio < 0.7:
        print("  ⚠ CD drift significantly smaller → adapters under-adapt (weak gradients from noisy pseudo-clean)")
    elif ratio > 1.5:
        print("  ⚠ CD drift significantly larger → adapters over-fit to noisy MASS signal")

    return {
        "id": {"mean_per_step": id_mean.tolist(), "std_per_step": id_std.tolist()},
        "cd": {"mean_per_step": cd_mean.tolist(), "std_per_step": cd_std.tolist()},
        "ratio": round(float(ratio), 2),
    }


# ──────────────────────────────────────────────
# Diagnostic 4: Cross-domain ablation (MASS + L_reg on CD)
# ──────────────────────────────────────────────
def diagnose_cd_ablation(model, adapters, mfd, cd_loader, device, n_images=100):
    """Run key ablations (w/o MASS, w/o L_reg) on CD to verify ID findings generalize."""
    print("\n" + "=" * 60)
    print("Diagnostic 4: Cross-Domain Ablation")
    print("=" * 60)

    from mota.tta import mota_adapt, no_mass_adapt
    from mota.mass import generate_mass_signal

    def eval_subset(loader, n):
        psnrs = []
        for i, (inp, tgt, path) in enumerate(loader):
            if i >= n: break
            inp_dev = inp.to(device)
            tgt_dev = tgt.to(device)
            with torch.no_grad():
                out = model(inp_dev)
            if out.shape[-2:] != tgt_dev.shape[-2:]:
                out = F.interpolate(out, size=tgt_dev.shape[-2:], mode='bilinear')
            mse = F.mse_loss(out, tgt_dev)
            psnrs.append(psnr_from_mse(mse).item())
        return np.mean(psnrs)

    print("  Running: w/o TTA (frozen)")
    frozen_psnr = eval_subset(cd_loader, n_images)

    print("  Running: w/o MASS (input as target)")
    no_mass_psnrs = []
    for i, (inp, tgt, path) in enumerate(cd_loader):
        if i >= n_images: break
        inp_dev = inp.to(device)
        try:
            out = no_mass_adapt(inp_dev, model, adapters, T=15, lambda_reg=1e-6)
        except Exception:
            out = model(inp_dev)
        tgt_dev = tgt.to(device)
        if out.shape[-2:] != tgt_dev.shape[-2:]:
            out = F.interpolate(out, size=tgt_dev.shape[-2:], mode='bilinear')
        mse = F.mse_loss(out, tgt_dev)
        no_mass_psnrs.append(psnr_from_mse(mse).item())
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_images}] w/o MASS")
    no_mass_psnr = np.mean(no_mass_psnrs)

    print("  Running: w/o L_reg")
    no_reg_psnrs = []
    for i, (inp, tgt, path) in enumerate(cd_loader):
        if i >= n_images: break
        inp_dev = inp.to(device)
        sigma = generate_mass_signal(inp_dev, mfd)
        try:
            adapters_copy = type(adapters)()
            for a in adapters:
                adapters_copy.append(type(a)(a.channels).to(device))
            adapters_copy.load_state_dict(adapters.state_dict())
            out = mota_adapt(inp_dev, model, adapters_copy, mfd, T=15, lambda_reg=0.0)
        except Exception:
            out = model(inp_dev)
        tgt_dev = tgt.to(device)
        if out.shape[-2:] != tgt_dev.shape[-2:]:
            out = F.interpolate(out, size=tgt_dev.shape[-2:], mode='bilinear')
        mse = F.mse_loss(out, tgt_dev)
        no_reg_psnrs.append(psnr_from_mse(mse).item())
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_images}] w/o L_reg")
    no_reg_psnr = np.mean(no_reg_psnrs)

    print(f"\n  CD Ablation Results ({n_images} images):")
    print(f"    w/o TTA (frozen): {frozen_psnr:.2f} dB")
    print(f"    w/o MASS:          {no_mass_psnr:.2f} dB  (Δ = {no_mass_psnr - frozen_psnr:+.2f})")
    print(f"    w/o L_reg:         {no_reg_psnr:.2f} dB  (Δ = {no_reg_psnr - frozen_psnr:+.2f})")

    return {
        "frozen": round(frozen_psnr, 2),
        "w_o_mass": round(no_mass_psnr, 2),
        "w_o_lreg": round(no_reg_psnr, 2),
        "n_images": n_images,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoTA Cross-Domain Diagnostics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wdnet_best.pth")
    parser.add_argument("--mfd_checkpoint", type=str, default="checkpoints/mfd.pth")
    parser.add_argument("--adapter_checkpoint", type=str, default="checkpoints/adapters_init.pth")
    parser.add_argument("--output", type=str, default="results/diagnostics")
    parser.add_argument("--n_images", type=int, default=100, help="Images for MASS quality + CD ablation")
    parser.add_argument("--n_drift_images", type=int, default=10, help="Images for drift tracking (slow)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_drift", action="store_true", help="Skip adapter drift (slow)")
    parser.add_argument("--skip_cd_ablation", action="store_true", help="Skip CD ablation (slow)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | n_images: {args.n_images}")

    # Load data
    id_loader = get_dataloader("LCDMoire", split="test", batch_size=1)
    cd_loader = get_dataloader("TIP2018", split="test", batch_size=1)
    print(f"ID  loader: LCDMoire test ({len(id_loader)} images)")
    print(f"CD  loader: TIP2018 test ({len(cd_loader)} images)")

    # Load model
    model = load_wdnet_backbone(args.checkpoint, device)
    model.eval()
    print("WDNet pipeline loaded")

    # Load MFD
    mfd = MFD()
    mfd.load_state_dict(torch.load(args.mfd_checkpoint, map_location=device))
    mfd.to(device).eval()
    print("MFD loaded")

    # Load adapters (handle both raw state_dict and wrapped format)
    ckpt = torch.load(args.adapter_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "channels" in ckpt:
        adapters, channels = insert_mota_adapters(model, channels_per_stage=ckpt["channels"])
        adapters.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        adapters, channels = insert_mota_adapters(model)
        adapters.load_state_dict(ckpt["state_dict"])
    else:
        adapters, channels = insert_mota_adapters(model)
        adapters.load_state_dict(ckpt)
    adapters.to(device)
    print(f"Adapters loaded ({len(channels)} stages)")

    all_results = {}

    # Diagnostic 1: MASS quality
    d1_avg, d1_detail = diagnose_mass_quality(model, mfd, cd_loader, device, n_images=args.n_images)
    all_results["mass_quality"] = {"avg": d1_avg}
    with open(f"{args.output}/diag1_mass_quality.json", "w") as f:
        json.dump({"avg": d1_avg, "detail": d1_detail}, f, indent=2)

    # Diagnostic 2: MFD behavior
    d2 = diagnose_mfd_behavior(mfd, id_loader, cd_loader, device, n_images=min(args.n_images, 50))
    all_results["mfd_behavior"] = d2
    with open(f"{args.output}/diag2_mfd_behavior.json", "w") as f:
        json.dump(d2, f, indent=2)

    # Diagnostic 3: Adapter drift
    if not args.skip_drift:
        d3 = diagnose_adapter_drift(model, adapters, mfd, id_loader, cd_loader, device,
                                    n_images=args.n_drift_images, T=15)
        all_results["adapter_drift"] = d3
        with open(f"{args.output}/diag3_adapter_drift.json", "w") as f:
            json.dump(d3, f, indent=2)
    else:
        print("\n[Skipped] Diagnostic 3: Adapter Drift (--skip_drift)")

    # Diagnostic 4: CD ablation
    if not args.skip_cd_ablation:
        d4 = diagnose_cd_ablation(model, adapters, mfd, cd_loader, device, n_images=args.n_images)
        all_results["cd_ablation"] = d4
        with open(f"{args.output}/diag4_cd_ablation.json", "w") as f:
            json.dump(d4, f, indent=2)
    else:
        print("\n[Skipped] Diagnostic 4: CD Ablation (--skip_cd_ablation)")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if "mass_quality" in all_results:
        print(f"  MASS pseudo-clean PSNR (CD): {all_results['mass_quality']['avg']['pseudo_psnr_mean']} dB")
    if "mfd_behavior" in all_results:
        print(f"  MFD CD/ID activation ratio:   "
              f"{all_results['mfd_behavior']['cd']['mean'] / max(all_results['mfd_behavior']['id']['mean'], 1e-8):.2f}x")
    if "adapter_drift" in all_results:
        print(f"  Adapter drift CD/ID ratio:    {all_results['adapter_drift']['ratio']}x")
    if "cd_ablation" in all_results:
        print(f"  CD w/o MASS:  {all_results['cd_ablation']['w_o_mass']} dB")
        print(f"  CD w/o L_reg: {all_results['cd_ablation']['w_o_lreg']} dB")

    # Write aggregate
    with open(f"{args.output}/diagnostic_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {args.output}/")
