"""
MASS: Moiré-Aware Self-Supervision 信号生成。
给定输入图像和 MFD 输出的摩尔纹概率图, 生成伪干净目标。
"""
import torch
import torch.nn.functional as F
from .mfd import dwt_haar, idwt_haar


def generate_mass_signal(
    inp: torch.Tensor,
    mfd: torch.nn.Module,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    从单张输入图像生成 MASS 伪干净目标。

    参数:
      inp:   (1, 3, H, W) or (B, 3, H, W)  摩尔纹图像, 值域 [-1, 1]
      mfd:   预训练好的 MFD 模型 (eval 模式)
      alpha: 频率衰减强度, 默认 0.5

    返回:
      pseudo_clean: 同 shape, 值域 [-1, 1]
    """
    was_training = mfd.training
    mfd.eval()

    with torch.no_grad():
        # 将 inp 从 [-1,1] 转到 [0,1] (DWT 需要)
        inp_01 = (inp + 1.0) / 2.0

        # 获得摩尔纹概率图
        M = mfd(inp_01)  # (B, 3, H, W)

        # 确保尺寸为偶数
        _, _, H, W = inp_01.shape
        pad_h = 0 if H % 2 == 0 else 1
        pad_w = 0 if W % 2 == 0 else 1
        if pad_h or pad_w:
            inp_01 = F.pad(inp_01, (0, pad_w, 0, pad_h))
            M = F.pad(M, (0, pad_w, 0, pad_h))

        # DWT 分解 — 逐通道平均（"luminance-channel shortcut"）
        # 使用灰度图做 DWT (取 RGB 均值), 应用到每个通道, 降低计算量 3×
        # TODO: MFD 内部也做 per-RGB-channel DWT (12ch 输入), 两个 DWT 路径有冗余 —
        # MFD 的 per-channel DWT 结果可复用于 MASS signal generation, 减少重复计算
        inp_gray = inp_01.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        LL, LH, HL, HH = dwt_haar(inp_gray)

        # M 的 3 通道语义映射:
        #   channel 0 → LH (horizontal detail, moiré probability)
        #   channel 1 → HL (vertical detail, moiré probability)
        #   channel 2 → HH (diagonal detail, trained but unused at inference)
        M_ds = F.interpolate(M, size=LH.shape[2:], mode='bilinear')  # (B, 3, H/2, W/2)

        # 中频衰减（HH 保留高频纹理，不对其做衰减）
        LH_attenuated = LH * (1.0 - alpha * M_ds[:, 0:1, :, :])
        HL_attenuated = HL * (1.0 - alpha * M_ds[:, 1:2, :, :])
        HH_attenuated = HH

        # 重建灰度图
        pseudo_gray = idwt_haar(LL, LH_attenuated, HL_attenuated, HH_attenuated)

        # 将灰度修正量应用到 RGB 各通道
        correction = pseudo_gray - inp_gray
        pseudo_clean = inp_01 + correction

        # 裁剪到 [0, 1]
        pseudo_clean = torch.clamp(pseudo_clean, 0.0, 1.0)

        # 转回 [-1, 1]
        pseudo_clean = pseudo_clean * 2.0 - 1.0

        # 去掉 padding
        if pad_h:
            pseudo_clean = pseudo_clean[:, :, :-1, :]
        if pad_w:
            pseudo_clean = pseudo_clean[:, :, :, :-1]

    if was_training:
        mfd.train()

    return pseudo_clean
