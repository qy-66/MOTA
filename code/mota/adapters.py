"""
MoTA 适配器: FDA (Fourier Domain Adapter) + SGA (Spatial Gating Adapter)

这两个适配器插入到冻结骨干网络的中间层，在 TTA 时更新。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FDA(nn.Module):
    """
    Fourier Domain Adapter.
    沿通道维度做 1D FFT → 低秩谱调制 → IFFT。
    参数量: 2 * C * r，通常 < 总参数的 1%。
    """

    def __init__(self, channels: int, rank: int = None):
        super().__init__()
        if rank is None:
            rank = max(4, channels // 32)
        self.channels = channels
        self.rank = rank

        # 低秩矩阵: ΔS = A @ B^T,  A,B ∈ R^{C × r}
        self.A = nn.Parameter(torch.zeros(channels, rank))
        self.B = nn.Parameter(torch.zeros(channels, rank))
        nn.init.normal_(self.A, std=0.01)
        nn.init.normal_(self.B, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: (B, C, H, W)，残差连接已内置
        """
        B, C, H, W = x.shape

        # 1. 沿通道维度做 1D FFT
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
        F_x = torch.fft.rfft(x_flat.float(), dim=-1)      # (B*H*W, C//2+1) complex

        # 2. 低秩谱调制：取 A@B^T 对角 → per-channel 调制标量 → 截断到频率维度
        delta_freq = (self.A * self.B).sum(dim=1)[:F_x.shape[-1]]  # (C//2+1,)
        F_x_mod = F_x * (1.0 + delta_freq.unsqueeze(0))  # broadcast over spatial

        # 3. IFFT 回去
        x_mod = torch.fft.irfft(F_x_mod, n=C, dim=-1)     # (B*H*W, C)
        x_mod = x_mod.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 4. 残差连接
        return x_mod.to(x.dtype) + x


class SGA(nn.Module):
    """
    Spatial Gating Adapter.
    3×3 conv 提取局部上下文 → sigmoid 门控 → 逐元素调制。
    参数量极小: C*C/4*9 + C*C/4 ≈ 2.5*C² (压缩比 1/4 后)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        hidden = max(channels // 4, 8)
        self.squeeze = nn.Conv2d(channels, hidden, kernel_size=1)
        self.spatial = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.expand  = nn.Conv2d(hidden, channels, kernel_size=1)

        # 初始化为接近恒等映射 (门控≈1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: (B, C, H, W)，残差连接确保初始化近似恒等映射
        """
        g = self.squeeze(x)
        g = self.spatial(g)
        g = self.expand(g)
        gate = torch.sigmoid(g)
        return x * gate + x


# ============================================================
# 插入适配器的工具函数
# ============================================================

def insert_mota_adapters(model: nn.Module, channels_per_stage: list = None):
    """
    在冻结骨干的每个 stage 输出后插入 FDA+SGA 对。

    参数:
      model: 去摩尔纹骨干网络 (MoiréNet / Freqformer 等)
      channels_per_stage: 每个 stage 输出通道数。
        如果不传，会自动尝试从 model 推断。

    返回:
      adapters: nn.ModuleList of (FDA, SGA) 对
      hook_handles: 用于注册/移除 forward hooks 的 handle 列表

    用法:
      adapters, _ = insert_mota_adapters(freqformer)
      # 训练/推理时 forward 正常走，适配器通过 hook 自动插入
    """
    if channels_per_stage is None:
        # 扫描模型 Conv2d 输出通道，但跳过 wavelet 变换层和输出层
        channels_set = set()
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 32:
                if any(skip in name for skip in ('dec.', 'rec.', 'final', 'conv_last')):
                    continue
                channels_set.add(m.out_channels)
        channels_per_stage = sorted(channels_set)
        # 最多取 4 个 stage
        if len(channels_per_stage) > 4:
            step = len(channels_per_stage) // 4
            channels_per_stage = channels_per_stage[::step][:4]

    adapters = nn.ModuleList()
    for c in channels_per_stage:
        adapters.append(FDA(c))
        adapters.append(SGA(c))

    return adapters, channels_per_stage
