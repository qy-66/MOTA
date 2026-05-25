"""
TTA 循环: 在测试时更新适配器参数, 骨干冻结。
这是 MoTA 的核心推理流程。
"""
import torch
import torch.nn as nn
from .adapters import FDA, SGA
from .mass import generate_mass_signal


def mota_adapt(
    inp: torch.Tensor,
    backbone: nn.Module,
    adapters: nn.ModuleList,
    mfd: nn.Module,
    T: int = 15,
    lr: float = 1e-4,
    lambda_reg: float = 1e-6,
    alpha: float = 0.5,
    verbose: bool = False,
) -> torch.Tensor:
    """
    对单张图像执行 MoTA 测试时自适应。

    参数:
      inp:        (1, 3, H, W)  摩尔纹图像, [-1, 1]
      backbone:   冻结的预训练骨干
      adapters:   nn.ModuleList of FDA/SGA pairs
      mfd:        预训练的 MFD (冻结)
      T:          自适应步数
      lr:         适配器学习率
      lambda_reg: 正则化权重
      alpha:      MASS 衰减强度
      verbose:    是否打印每步 loss

    返回:
      out: (1, 3, H, W) 去摩尔纹结果, [-1, 1]
    """
    device = inp.device

    # 保存初始适配器参数（用 param-list 缓存避免每次循环 named_parameters()）
    phi_0 = [(p, p.data.clone()) for p in adapters.parameters()]

    # 生成 MASS 伪干净目标 (只需一次)
    pseudo_clean = generate_mass_signal(inp, mfd, alpha=alpha)

    # 若模型有内部下采样（如 DDA MBCNN 1024→256），缩放 pseudo_clean 匹配输出分辨率
    with torch.no_grad():
        probe = forward_with_adapters(inp, backbone, adapters)
    if probe.shape[-2:] != pseudo_clean.shape[-2:]:
        pseudo_clean = nn.functional.interpolate(pseudo_clean, size=probe.shape[-2:],
                                                 mode='bilinear', align_corners=False)

    # 适配器优化器
    opt = torch.optim.Adam(adapters.parameters(), lr=lr)

    best_out = None
    best_loss = float('inf')

    for t in range(T):
        opt.zero_grad()

        out = forward_with_adapters(inp, backbone, adapters)
        loss_mass = torch.abs(out - pseudo_clean).mean()

        loss_reg = sum(((p - p0) ** 2).sum() for p, p0 in phi_0)

        loss = loss_mass + lambda_reg * loss_reg
        loss.backward()
        opt.step()

        if verbose:
            print(f"  TTA step {t+1}/{T}: L_MASS={loss_mass.item():.6f}, "
                  f"L_reg={loss_reg.item():.6f}, total={loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out.detach().clone()

    # 返回最低 loss 时的输出 (而非最后一步, 防止过拟合)
    if best_out is None:
        best_out = out.detach()

    return best_out


def full_ft_adapt(
    inp: torch.Tensor,
    backbone: nn.Module,
    mfd: nn.Module,
    T: int = 15,
    lr: float = 1e-4,
    lambda_reg: float = 1e-6,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Full Fine-Tuning baseline: TTA 时解冻全部 backbone 参数做 Adam 优化。
    仅用于 Table 4b 对比。
    """
    device = inp.device
    backbone = backbone.to(device).train()

    phi_0 = [(p, p.data.clone()) for p in backbone.parameters()]
    pseudo_clean = generate_mass_signal(inp, mfd, alpha=alpha)
    with torch.no_grad():
        backbone.eval()
        probe = backbone(inp)
        backbone.train()
    if probe.shape[-2:] != pseudo_clean.shape[-2:]:
        pseudo_clean = nn.functional.interpolate(pseudo_clean, size=probe.shape[-2:],
                                                 mode='bilinear', align_corners=False)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)

    best_out = None
    best_loss = float('inf')

    for t in range(T):
        opt.zero_grad()
        out = backbone(inp)
        loss_mass = torch.abs(out - pseudo_clean).mean()
        loss_reg = sum(((p - p0) ** 2).sum() for p, p0 in phi_0)
        loss = loss_mass + lambda_reg * loss_reg
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out.detach().clone()

    backbone.eval()
    return best_out if best_out is not None else out.detach()


class LoRAConv2d(nn.Module):
    """给 Conv2d 添加低秩并行路径的 LoRA 包装器。"""

    def __init__(self, conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.conv = conv  # 冻结的原始卷积
        for p in self.conv.parameters():
            p.requires_grad = False

        in_ch = conv.in_channels
        out_ch = conv.out_channels
        ks = conv.kernel_size[0]
        padding = conv.padding[0]
        stride = conv.stride[0]
        groups = conv.groups

        self.lora_down = nn.Conv2d(in_ch, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_ch, kernel_size=ks,
                                 padding=padding, stride=stride,
                                 groups=1 if groups == 1 else min(rank, groups),
                                 bias=False)
        nn.init.kaiming_normal_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_up(self.lora_down(x))


def apply_lora_to_conv2d(model: nn.Module, rank: int = 4):
    """
    把 backbone 中所有 Conv2d 包装为 LoRAConv2d。
    返回 (修改后的 model, lora 参数列表)。
    """
    lora_params = []
    _replace_conv_with_lora(model, rank, lora_params)
    return model, lora_params


def _replace_conv_with_lora(module, rank, lora_params, prefix=''):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.groups == 1:
            lora_wrapper = LoRAConv2d(child, rank=rank)
            setattr(module, name, lora_wrapper)
            lora_params.extend([lora_wrapper.lora_down.weight,
                                lora_wrapper.lora_up.weight])
        else:
            _replace_conv_with_lora(child, rank, lora_params, f'{prefix}{name}.')
    return module


def _restore_conv_from_lora(module):
    """Unwrap LoRAConv2d wrappers, restoring original Conv2d layers in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, child.conv)
        else:
            _restore_conv_from_lora(child)


def lora_adapt(
    inp: torch.Tensor,
    backbone: nn.Module,
    mfd: nn.Module,
    T: int = 15,
    lr: float = 1e-4,
    lambda_reg: float = 1e-6,
    rank: int = 4,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    LoRA baseline: 给 backbone 的 Conv2d 包装 LoRA，TTA 时只更新 LoRA 参数。
    仅用于 Table 4b 对比。
    """
    device = inp.device
    backbone = backbone.to(device).eval()

    backbone, lora_params = apply_lora_to_conv2d(backbone, rank=rank)
    backbone = backbone.to(device)

    lora_phi_0 = [p.data.clone() for p in lora_params]
    pseudo_clean = generate_mass_signal(inp, mfd, alpha=alpha)
    with torch.no_grad():
        probe = backbone(inp)
    if probe.shape[-2:] != pseudo_clean.shape[-2:]:
        pseudo_clean = nn.functional.interpolate(pseudo_clean, size=probe.shape[-2:],
                                                 mode='bilinear', align_corners=False)
    opt = torch.optim.Adam(lora_params, lr=lr)

    best_out = None
    best_loss = float('inf')

    for t in range(T):
        opt.zero_grad()
        out = backbone(inp)
        loss_mass = torch.abs(out - pseudo_clean).mean()
        loss_reg = sum(((p - p0) ** 2).sum()
                       for p, p0 in zip(lora_params, lora_phi_0))
        loss = loss_mass + lambda_reg * loss_reg
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out.detach().clone()

    _restore_conv_from_lora(backbone)
    return best_out if best_out is not None else out.detach()


def no_mass_adapt(
    inp: torch.Tensor,
    backbone: nn.Module,
    adapters: nn.ModuleList,
    T: int = 15,
    lr: float = 1e-4,
    lambda_reg: float = 1e-6,
) -> torch.Tensor:
    """
    w/o MASS 消融变体：将输入自身作为 pseudo-clean target（L1 重建）。
    TTA 试图让输出逼近输入 → 测试适配器在没有 MASS 信号时的表现。
    """
    phi_0 = [(p, p.data.clone()) for p in adapters.parameters()]

    # 若模型有下采样，缩放 target（input 自身）匹配输出分辨率
    pseudo_target = inp
    with torch.no_grad():
        probe = forward_with_adapters(inp, backbone, adapters)
    if probe.shape[-2:] != pseudo_target.shape[-2:]:
        pseudo_target = nn.functional.interpolate(pseudo_target, size=probe.shape[-2:],
                                                   mode='bilinear', align_corners=False)

    opt = torch.optim.Adam(adapters.parameters(), lr=lr)
    best_out = None
    best_loss = float('inf')

    for t in range(T):
        opt.zero_grad()
        out = forward_with_adapters(inp, backbone, adapters)
        loss_mass = torch.abs(out - pseudo_target).mean()

        loss_reg = sum(((p - p0) ** 2).sum() for p, p0 in phi_0)

        loss = loss_mass + lambda_reg * loss_reg
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out.detach().clone()

    return best_out if best_out is not None else out.detach()


def forward_with_adapters(
    x: torch.Tensor,
    backbone: nn.Module,
    adapters: nn.ModuleList,
) -> torch.Tensor:
    """
    通过冻结骨干 + 适配器做前向传播。

    优先尝试输入/输出端插入（适配器通道数 == 输入通道数时）。
    若都不匹配（适配器通道数对应内部 Conv2d 层），则通过 forward hook
    插入到中间层，确保适配器参数参与计算图并产生梯度。
    """

    # ---- 检查是否有适配器匹配输入/输出通道数 ----
    in_ch = x.shape[1]
    use_hooks = True
    for i in range(0, len(adapters), 2):
        if adapters[i].channels == in_ch:
            use_hooks = False
            break

    if not use_hooks:
        # 输入/输出端插入
        x_in = x
        for i in range(0, len(adapters), 2):
            if adapters[i].channels == x_in.shape[1]:
                x_in = adapters[i](x_in)
                x_in = adapters[i+1](x_in)
        with torch.no_grad():
            out = backbone(x_in)
        x_out = out
        for i in range(0, len(adapters), 2):
            if adapters[i].channels == x_out.shape[1]:
                x_out = adapters[i](x_out)
                x_out = adapters[i+1](x_out)
        return x_out
    else:
        # 中间层 hook 插入
        hooks = register_adapter_hooks(backbone, adapters)
        if not hooks:
            import warnings
            warnings.warn("forward_with_adapters: no adapter matched any Conv2d layer — "
                          "output will be identical to frozen backbone")
        # 注意：backbone(x) 不在 torch.no_grad() 下执行，冻结 backbone 的中间激活
        # 会被保留用于反向传播，增加 GPU 内存开销。以 batch_size=4 + 24GB 显存可接受。
        out = backbone(x)
        for h in hooks:
            h.remove()
        return out


# ============================================================
# 更精细的中间层适配 (通过 forward hook)
# ============================================================

class AdapterHook:
    """在骨干的中间层插入适配器的 forward hook。"""
    def __init__(self, adapter_fda, adapter_sga):
        self.fda = adapter_fda
        self.sga = adapter_sga

    def __call__(self, module, input, output):
        # output: (B, C, H, W) 特征图
        x = self.fda(output)
        x = self.sga(x)
        return x


def register_adapter_hooks(backbone, adapters, target_layer_types=(nn.Conv2d,)):
    """
    在骨干的指定层类型后注册适配器 hook。
    按通道数匹配：每个 adapter pair 只挂到与其 channels 一致的第一个 Conv2d 层。
    同一通道数的后续层不会被挂钩（一个 stage 只插一对适配器）。
    返回 hook handles 列表（可能为空）。
    """
    hooks = []
    used = set()

    for _, module in backbone.named_modules():
        if not isinstance(module, target_layer_types):
            continue
        ch = module.out_channels
        for i in range(0, len(adapters), 2):
            if i in used:
                continue
            if adapters[i].channels == ch:
                hook = AdapterHook(adapters[i], adapters[i + 1])
                handle = module.register_forward_hook(hook)
                hooks.append(handle)
                used.add(i)
                break

    return hooks
