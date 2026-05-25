"""WDNet 骨干网络加载工具 — 消除 eval.py / train.py / benchmark_speed.py 间的重复代码。"""

import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


WDNET_DIR = "/root/autodl-tmp/WDNet_demoire-master"
WAVELET_PATH = os.path.join(WDNET_DIR, "wavelet_weights_c2.pkl")


class _Latin1Pickle:
    """上下文管理器：临时将 pickle.load 改为 latin1 编码（兼容旧权重文件）。"""

    def __enter__(self):
        self._orig = pickle.load
        def _patched(f, **kw):
            kw.setdefault('encoding', 'latin1')
            return self._orig(f, **kw)
        pickle.load = _patched
        return self

    def __exit__(self, *args):
        pickle.load = self._orig


class _FakeMoxContext:
    """上下文管理器：注入假 moxing 模块（WDNet 源码 import 时需要）。"""

    def __enter__(self):
        class FakeMox:
            class file:
                @staticmethod
                def shift(*a, **kw): pass
                @staticmethod
                def read(path):
                    return open(path.replace('s3://bucket-8280/liulin/ddwnet_2021/', ''), 'rb')
        sys.modules["moxing"] = FakeMox
        return self

    def __exit__(self, *args):
        del sys.modules["moxing"]


class WDNetPipeline(nn.Module):
    """WDNet + 小波变换的完整 pipeline：wavelet_dec → WDNet → wavelet_rec + 残差。"""

    def __init__(self, wavelet_dec, wdnet, wavelet_rec):
        super().__init__()
        self.dec = wavelet_dec
        self.net = wdnet
        self.rec = wavelet_rec

    def forward(self, x):
        # pad 到 4 的倍数（2 级 DWT 需要）
        _, _, H, W = x.shape
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x
        w = self.dec(x_padded)
        out = self.rec(self.net(w))
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out + x


def load_wdnet_backbone(checkpoint_path: str, device: str = "cuda"):
    """
    加载完整的 WDNet pipeline（含小波变换）。

    参数:
      checkpoint_path: wdnet_best.pth 的路径
      device: "cuda" 或 "cpu"

    返回:
      model: WDNetPipeline (eval 模式)
    """
    sys.path.insert(0, WDNET_DIR)

    with _Latin1Pickle(), _FakeMoxContext():
        from model_dense import WDNet, WaveletTransform
        wavelet_dec = WaveletTransform(scale=2, dec=True, params_path=WAVELET_PATH)
        wavelet_rec = WaveletTransform(scale=2, dec=False, params_path=WAVELET_PATH)

    wdnet = WDNet(in_channel=3)
    model = WDNetPipeline(wavelet_dec, wdnet, wavelet_rec)

    state = torch.load(checkpoint_path, map_location=device)
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


DDA_DIR = "/root/autodl-tmp/DDA-main"
DDA_CKPT = "/root/autodl-tmp/DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth"


def load_dda_backbone(checkpoint_path: str = None, device: str = "cuda"):
    """
    加载 DDA MBCNN backbone + InputRangeAdapter。

    参数:
      checkpoint_path: ckpt_best.pth 路径，默认使用项目标准路径
      device: "cuda" 或 "cpu"

    返回:
      model: InputRangeAdapter(MBCNN(64)) (eval 模式)
    """
    if checkpoint_path is None:
        checkpoint_path = DDA_CKPT

    sys.path.insert(0, DDA_DIR)
    from Net.MBCNN import MBCNN
    from .utils import InputRangeAdapter

    model = MBCNN(64)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model = InputRangeAdapter(model)
    model = _DDAOutputWrapper(model)
    model.to(device).eval()
    return model


class _DDAOutputWrapper(nn.Module):
    """将 MBCNN [0,1] 输出转回 [-1,1]，与 dataloader 目标值域一致。"""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out * 2.0 - 1.0


def _strip_module_prefix(state):
    """去除 DataParallel 包装产生的 'module.' 前缀。"""
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def psnr_from_mse(mse, max_val=2.0):
    """PSNR = 10 * log10(max_val^2 / MSE)。默认 max_val=2.0（[-1,1] 范围）。"""
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
