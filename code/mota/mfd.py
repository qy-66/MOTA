"""
MFD: Moiré Frequency Detector.
轻量 4 层 CNN，输入 DWT 的 4 个子带，输出摩尔纹概率图。
在源域预训练后冻结，TTA 阶段不再更新。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_haar(x: torch.Tensor):
    """
    Haar 离散小波变换 (单级)。
    输入: (B, C, H, W) — H, W 必须为偶数
    返回: LL, LH, HL, HH 各 (B, C, H/2, W/2)
    """
    # 行变换
    x_even = x[:, :, :, 0::2]   # (B, C, H, W/2)
    x_odd  = x[:, :, :, 1::2]   # (B, C, H, W/2)
    L = (x_even + x_odd) / 2.0
    H = (x_even - x_odd) / 2.0

    # 列变换
    L_even = L[:, :, 0::2, :]   # (B, C, H/2, W/2)
    L_odd  = L[:, :, 1::2, :]
    H_even = H[:, :, 0::2, :]
    H_odd  = H[:, :, 1::2, :]

    LL = (L_even + L_odd) / 2.0
    LH = (L_even - L_odd) / 2.0
    HL = (H_even + H_odd) / 2.0
    HH = (H_even - H_odd) / 2.0

    return LL, LH, HL, HH


def idwt_haar(LL, LH, HL, HH):
    """
    Haar 逆小波变换。
    输入: 4 个子带各 (B, C, H/2, W/2)
    返回: (B, C, H, W)
    """
    B = LL.size(0)
    C = LL.size(1)
    H_half = LL.size(2)
    W_half = LL.size(3)
    if hasattr(H_half, 'item'):
        H_half, W_half = H_half.item(), W_half.item()
    H, W = H_half * 2, W_half * 2

    # 逆列变换
    L_even = LL + LH
    L_odd  = LL - LH
    H_even = HL + HH
    H_odd  = HL - HH

    L = LL.new_zeros((LL.size(0), LL.size(1), LL.size(2) * 2, LL.size(3)))
    H = LL.new_zeros((LL.size(0), LL.size(1), LL.size(2) * 2, LL.size(3)))
    L[:, :, 0::2, :] = L_even
    L[:, :, 1::2, :] = L_odd
    H[:, :, 0::2, :] = H_even
    H[:, :, 1::2, :] = H_odd

    # 逆行变换
    x_even = L + H
    x_odd  = L - H

    x = LL.new_zeros((LL.size(0), LL.size(1), LL.size(2) * 2, LL.size(3) * 2))
    x[:, :, :, 0::2] = x_even
    x[:, :, :, 1::2] = x_odd

    return x


class MFD(nn.Module):
    """
    Moiré Frequency Detector.
    输入 12 通道 (3 RGB × 4 DWT 子带), 输出 3 通道摩尔纹概率图。
    """

    def __init__(self):
        super().__init__()
        # 轻量 4 层 CNN
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)   # 下采样 2×
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) — 单张 RGB 摩尔纹图像
        返回: M (B, 3, H, W) — 每像素每子带的摩尔纹概率 [0, 1]

        内部会先做 DWT, 所以输入是 RGB 图像即可。
        输出的 3 通道对应 LH, HL, HH 的摩尔纹概率
        (LL 通道无摩尔纹, 不需要概率图)。
        """
        # 确保 H, W 为偶数
        if x.shape[2] % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        if x.shape[3] % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))

        # DWT 分解
        LL, LH, HL, HH = [], [], [], []
        for c in range(3):  # 每个 RGB 通道分别做
            ll, lh, hl, hh = dwt_haar(x[:, c:c+1, :, :])
            LL.append(ll)
            LH.append(lh)
            HL.append(hl)
            HH.append(hh)

        # 拼接: (B, 12, H/2, W/2)
        subbands = torch.cat([
            torch.cat(LL, dim=1),
            torch.cat(LH, dim=1),
            torch.cat(HL, dim=1),
            torch.cat(HH, dim=1),
        ], dim=1)

        # CNN 处理
        feat = F.relu(self.conv1(subbands))
        feat = F.relu(self.conv2(feat))
        feat = F.relu(self.conv3(feat))
        out  = self.conv4(feat)
        out  = self.upsample(out)           # 回到原分辨率

        return torch.sigmoid(out)


def train_mfd(mfd, train_loader, epochs=50, lr=1e-4, device="cuda"):
    """
    在源域 (LCDMoire) 上预训练 MFD。
    用已知的合成摩尔纹模式生成 pseudo GT — 这里用简化版:
    MFD 的 GT 是 DWT 子带中摩尔纹的实际位置 (从合成过程可知)。
    """
    mfd = mfd.to(device)
    opt = torch.optim.Adam(mfd.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        for inp, tgt, _ in train_loader:
            inp = inp.to(device)  # (B, 3, H, W) 摩尔纹图像
            tgt = tgt.to(device)  # (B, 3, H, W) 干净图像
            # dataloader 输出 [-1,1]，MFD 期望 [0,1]（与推理时 mass.py 一致）
            inp = (inp + 1.0) / 2.0
            tgt = (tgt + 1.0) / 2.0

            # Pseudo GT: 输入和目标的频域差异就是摩尔纹的"位置"
            with torch.no_grad():
                # 摩尔纹的频域差异主要在中频 (LH, HL)
                inp_ll, inp_lh, inp_hl, inp_hh = dwt_haar(inp.mean(dim=1, keepdim=True))
                tgt_ll, tgt_lh, tgt_hl, tgt_hh = dwt_haar(tgt.mean(dim=1, keepdim=True))
                # 中频差异大 → 摩尔纹概率高
                pseudo_gt = torch.cat([
                    torch.abs(inp_lh - tgt_lh),
                    torch.abs(inp_hl - tgt_hl),
                    torch.abs(inp_hh - tgt_hh),
                ], dim=1)  # (B, 3, H/2, W/2)
                # 归一化到 [0, 1]
                pseudo_gt = pseudo_gt / (pseudo_gt.max() + 1e-8)

            pred = mfd(inp)  # (B, 3, H, W)

            # 将 pseudo_gt 上采样到 pred 的分辨率
            pseudo_gt_up = F.interpolate(pseudo_gt, size=pred.shape[2:], mode='bilinear')

            loss = F.mse_loss(pred, pseudo_gt_up)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  MFD epoch {epoch+1}/{epochs}, loss={total_loss/len(train_loader):.6f}")
    return mfd
