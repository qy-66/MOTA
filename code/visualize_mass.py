"""
MASS signal visualization for Figure 3.
Shows: input moire, MFD heatmap overlay, DWT subbands before/after attenuation,
and pseudo-clean output. Demonstrates selective mid-frequency suppression.

Usage:
  python visualize_mass.py    # uses first TIP2018 test sample
  python visualize_mass.py --index 42
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataloader import get_dataloader
from mota.mfd import MFD, dwt_haar, idwt_haar
from mota.utils import set_seed


def tensor_to_np(t):
    """[-1,1] or [0,1] tensor -> [0,255] uint8 numpy (H,W,3)."""
    t = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    if t.min() < -0.5:
        t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)
    return (t * 255).numpy().astype(np.uint8)


def subband_to_np(s):
    """Single-channel subband (1,C,H,W) -> uint8 numpy (H,W)."""
    s = s.detach().cpu().squeeze()
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return (s.numpy() * 255).astype(np.uint8)


def heatmap_overlay(img_rgb, heatmap, alpha=0.5):
    """Overlay colormap heatmap on RGB image.
    img_rgb: uint8 (H,W,3), heatmap: float (H,W) in [0,1]."""
    from matplotlib.cm import jet
    h = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    cmap = jet(h)[:, :, :3]  # (H,W,3) float in [0,1]
    cmap_img = (cmap * 255).astype(np.uint8)
    return (img_rgb * (1 - alpha) + cmap_img * alpha).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0,
                        help="TIP2018 test sample index")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="results/fig3_mass.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_dataloader("TIP2018", split="test", batch_size=1)
    inp, tgt, _ = loader.dataset[args.index]
    inp = inp.unsqueeze(0).to(device)
    tgt = tgt.unsqueeze(0).to(device)

    # MFD
    mfd = MFD().to(device).eval()
    ckpt = torch.load("checkpoints/mfd.pth", map_location=device)
    mfd.load_state_dict(ckpt.get("state_dict", ckpt))

    # Convert to [0,1] for DWT and MFD
    inp_01 = (inp + 1) / 2

    # MASS signal generation (same logic as mass.py)
    with torch.no_grad():
        M = mfd(inp_01)  # (1, 3, H, W)

        # Pad to even
        _, _, H, W = inp_01.shape
        pad_h = 0 if H % 2 == 0 else 1
        pad_w = 0 if W % 2 == 0 else 1
        inp_pad = F.pad(inp_01, (0, pad_w, 0, pad_h)) if pad_h or pad_w else inp_01
        M_pad = F.pad(M, (0, pad_w, 0, pad_h)) if pad_h or pad_w else M

        # DWT on gray
        inp_gray = inp_pad.mean(dim=1, keepdim=True)
        LL, LH, HL, HH = dwt_haar(inp_gray)

        # M downsampled to subband resolution
        M_ds = F.interpolate(M_pad, size=LH.shape[2:], mode='bilinear')

        alpha = 0.5
        LH_att = LH * (1.0 - alpha * M_ds[:, 0:1, :, :])
        HL_att = HL * (1.0 - alpha * M_ds[:, 1:2, :, :])
        HH_att = HH  # preserve high-frequency texture

        pseudo_gray = idwt_haar(LL, LH_att, HL_att, HH_att)
        correction = pseudo_gray - inp_gray
        pseudo_clean = inp_pad + correction
        pseudo_clean = torch.clamp(pseudo_clean, 0.0, 1.0)

        # Unpad
        if pad_h:
            pseudo_clean = pseudo_clean[:, :, :-1, :]
            inp_01 = inp_01[:, :, :-1, :]
            M = M[:, :, :-1, :]
        if pad_w:
            pseudo_clean = pseudo_clean[:, :, :, :-1]
            inp_01 = inp_01[:, :, :, :-1]
            M = M[:, :, :, :-1]

    # Convert to numpy
    img_inp = tensor_to_np(inp)  # original [-1,1] input
    img_tgt = tensor_to_np(tgt)
    img_pseudo = tensor_to_np(pseudo_clean * 2 - 1)  # back to [-1,1] for display

    # MFD heatmap: average over 3 channels, upsample to input size
    mfd_heatmap = M.squeeze(0).mean(dim=0).cpu()  # (H, W)
    mfd_heatmap_up = F.interpolate(
        mfd_heatmap.unsqueeze(0).unsqueeze(0),
        size=img_inp.shape[:2], mode='bilinear'
    ).squeeze().numpy()

    img_overlay = heatmap_overlay(img_inp, mfd_heatmap_up)

    # Subbands
    sb_LL = subband_to_np(LL)
    sb_LH = subband_to_np(LH)
    sb_HL = subband_to_np(HL)
    sb_HH = subband_to_np(HH)
    sb_LH_att = subband_to_np(LH_att)
    sb_HL_att = subband_to_np(HL_att)

    # --- Plot ---
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fontsize = 10

    # Row 1: Input | MFD Overlay | Pseudo-Clean | Ground Truth
    titles_r1 = ["(a) Input moire", "(b) MFD detection", "(c) Pseudo-clean (MASS)", "(d) Ground truth"]
    imgs_r1 = [img_inp, img_overlay, img_pseudo, img_tgt]
    for j in range(4):
        axes[0, j].imshow(imgs_r1[j])
        axes[0, j].set_title(titles_r1[j], fontsize=fontsize, fontweight='bold')

    # Row 2: Subbands BEFORE attenuation
    titles_r2 = ["(e) LL (low-freq)", "(f) LH (horizontal)", "(g) HL (vertical)", "(h) HH (high-freq)"]
    imgs_r2 = [sb_LL, sb_LH, sb_HL, sb_HH]
    for j in range(4):
        axes[1, j].imshow(imgs_r2[j], cmap='gray')
        axes[1, j].set_title(titles_r2[j], fontsize=fontsize, fontweight='bold')

    # Row 3: Subbands AFTER attenuation
    titles_r3 = ["(i) LL' (unchanged)", "(j) LH' (attenuated)", "(k) HL' (attenuated)", "(l) HH' (unchanged)"]
    imgs_r3 = [sb_LL, sb_LH_att, sb_HL_att, sb_HH]  # LL and HH unchanged
    for j in range(4):
        axes[2, j].imshow(imgs_r3[j], cmap='gray')
        axes[2, j].set_title(titles_r3[j], fontsize=fontsize, fontweight='bold')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle("Figure 3: MASS Signal Visualization — Selective Mid-Frequency Suppression",
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(pad=1.5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
