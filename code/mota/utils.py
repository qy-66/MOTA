"""共享工具函数——避免跨文件重复代码。"""
import random
import numpy as np
import torch


def to_01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (x + 1.0) / 2.0


def to_11(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-1, 1]"""
    return x * 2.0 - 1.0


def set_seed(seed: int) -> None:
    """固定随机种子 + cuDNN 确定性模式。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InputRangeAdapter(torch.nn.Module):
    """包装模型：将 [-1,1] 输入转为模型期望的 [0,1]，并解包 list/tuple 输出。

    DDA 系列模型训练用 [0,1] 像素范围，但 unified_dataloader 统一输出 [-1,1]。
    此外部分模型（如 MBCNN）forward 返回 list/tuple 而非单个 tensor。
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad 到 32 的倍数（MBCNN 多层 stride-2 conv + concat 要求）
        _, _, H, W = x.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        x_in = x * 0.5 + 0.5
        if pad_h or pad_w:
            x_in = torch.nn.functional.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')
        out = self.inner(x_in)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out
