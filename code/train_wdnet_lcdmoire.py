"""
在 LCDMoire 上训练 WDNet baseline。
WDNet 工作在小波域：RGB → wavelet_dec(3→48ch) → WDNet → wavelet_rec(48→3ch) → +input
"""

import sys
import io
import os

# ---- 必须在所有 import 之前：修复 WDNet 源码中文字符编码报错 ----
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mota.wdnet_loader import psnr_from_mse
from mota.utils import to_01

# ---- 路径 ----
CODE_DIR = "/root/autodl-tmp/demoireing-paper/code"
sys.path.insert(0, CODE_DIR)

from unified_dataloader import get_dataloader
from mota.wdnet_loader import (
    WDNetPipeline, _Latin1Pickle, _FakeMoxContext, WDNET_DIR, WAVELET_PATH
)


# ============================================================
# 1. 加载 WDNet + 小波变换（完整的 wavelet 域 pipeline）
# ============================================================
def build_wdnet_pipeline(device):
    """构建完整的 WDNet pipeline: wavelet_dec → WDNet → wavelet_rec + input"""
    saved_path = sys.path.copy()
    sys.path.insert(0, WDNET_DIR)
    try:
        with _Latin1Pickle(), _FakeMoxContext():
            from model_dense import WDNet, WaveletTransform
            wavelet_dec = WaveletTransform(scale=2, dec=True, params_path=WAVELET_PATH)
            wavelet_rec = WaveletTransform(scale=2, dec=False, params_path=WAVELET_PATH)
        print("  Wavelet weights loaded")
        wdnet = WDNet(in_channel=3)
        model = WDNetPipeline(wavelet_dec, wdnet, wavelet_rec).to(device)
    finally:
        sys.path = saved_path
    return model


# ============================================================
# 2. 小波域纹理损失 (参考 WDNet 原始实现)
# ============================================================
def wavelet_texture_loss(x, y, alpha=1.2):
    """
    小波域高频纹理一致性损失。
    x, y: (B, C, H, W) 小波高频子带
    防止去摩尔纹过程中丢失纹理细节。
    """
    nc = 3
    C = x.shape[1]
    if C % nc != 0:
        return torch.tensor(0.0, device=x.device)
    xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
    yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
    xi2 = torch.sum(xi * xi, dim=2)
    yi2 = torch.sum(yi * yi, dim=2)
    return torch.mean(F.relu(yi2.mul(alpha) - xi2))


# ============================================================
# 3. VGG 感知损失
# ============================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)])
        for p in self.parameters():
            p.requires_grad = False

    def to(self, device):
        super().to(device)
        self.slice1.to(device)
        self.slice2.to(device)
        self.slice3.to(device)
        return self

    def forward(self, x, y):
        x = to_01(x)
        y = to_01(y)
        x1, y1 = self.slice1(x), self.slice1(y)
        x2, y2 = self.slice2(x1), self.slice2(y1)
        x3, y3 = self.slice3(x2), self.slice3(y2)
        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2) + F.l1_loss(x3, y3)
        return loss


# ============================================================
# 4. 训练循环
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader = get_dataloader("LCDMoire", split="train", batch_size=args.batch_size)
    val_loader   = get_dataloader("LCDMoire", split="test",  batch_size=1)
    print(f"  Train: {len(train_loader.dataset)} pairs")
    print(f"  Val:   {len(val_loader.dataset)} pairs")

    print("\nBuilding model...")
    model = build_wdnet_pipeline(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_psnr = 0.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        accum = 0
        print(f"Epoch {epoch+1}/{args.epochs} starting...", flush=True)

        for batch_idx, (inp, tgt, _) in enumerate(train_loader):
            inp = inp.to(device)
            tgt = tgt.to(device)

            out = model(inp)

            loss_l1  = criterion_l1(out, tgt)
            loss_vgg = criterion_vgg(out, tgt) if epoch >= 5 else 0.0

            w_out = model.dec(out)
            w_tgt = model.dec(tgt)
            loss_sr = F.l1_loss(w_out[:, 3:, :, :], w_tgt[:, 3:, :, :])
            loss_lr = F.l1_loss(w_out[:, :3, :, :], w_tgt[:, :3, :, :])
            loss_tex = wavelet_texture_loss(w_out[:, 3:, :, :], w_tgt[:, 3:, :, :])

            loss = (loss_l1 + 0.1 * loss_vgg
                    + 100 * loss_sr + 10 * loss_lr + 5 * loss_tex) / args.grad_accum
            loss.backward()

            accum += 1
            if accum >= args.grad_accum:
                optimizer.step()
                optimizer.zero_grad()
                accum = 0

            total_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % 500 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  batch {batch_idx+1}/{len(train_loader)}, avg loss={avg:.4f}", flush=True)

        scheduler.step()

        if (epoch + 1) % 10 != 0:
            continue

        model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for inp, tgt, _ in val_loader:
                out = model(inp.to(device))
                mse = F.mse_loss(out, tgt.to(device))
                total_psnr += psnr_from_mse(mse).item()

        avg_psnr = total_psnr / len(val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | val PSNR={avg_psnr:.2f} dB")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "wdnet_best.pth"))
            print(f"  → Saved (PSNR={best_psnr:.2f})")
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(args)
