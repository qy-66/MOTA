"""
训练脚本: 在 LCDMoire 上训练 WDNet baseline + MoTA 组件。

用法:
  python train.py                    # 训练 WDNet baseline
  python train.py --train_mfd        # 预训练 MFD
  python train.py --init_adapters     # 初始化 MoTA 适配器 (identity mapping)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# 添加代码路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataloader import get_dataloader
from mota.mfd import MFD, train_mfd
from mota.wdnet_loader import load_wdnet_backbone, _strip_module_prefix
from mota.tta import forward_with_adapters
from mota.utils import InputRangeAdapter


def train_baseline(model, train_loader, val_loader, epochs=200, device="cuda"):
    """
    训练去摩尔纹 baseline (例如 WDNet)。

    由于 WDNet 是外部仓库, 这里提供一个通用训练循环模板。
    你需要:
      1. from WDNet.models import WDNet  # 替换为实际导入
      2. model = WDNet()  # 或任何其他 backbone
      3. train_baseline(model, train_loader, val_loader)
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_psnr = 0.0
    patience_counter = 0

    accumulation_steps = 4  # 梯度累积

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        opt.zero_grad()
        for i, (inp, tgt, _) in enumerate(train_loader):
            inp = inp.to(device)
            tgt = tgt.to(device)

            out = model(inp)

            loss_l1 = nn.functional.l1_loss(out, tgt)
            loss = loss_l1 / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad()

        scheduler.step()

        # ---- 验证 ----
        model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for inp, tgt, _ in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                out = model(inp)
                mse = nn.functional.mse_loss(out, tgt)
                total_psnr += 10 * torch.log10(4.0 / (mse + 1e-8)).item()

        avg_psnr = total_psnr / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | val PSNR: {avg_psnr:.2f} dB")

        # Early stopping
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_baseline.pth")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model


def init_adapters_identity(adapters, backbone, train_loader, device="cuda"):
    """
    初始化适配器为恒等映射:
    训练适配器使得 backbone(adapter(x)) ≈ backbone(x),
    这样 TTA 开始时适配器不影响原始骨干的输出。
    """
    backbone = backbone.to(device).eval()
    adapters = adapters.to(device).train()

    opt = torch.optim.Adam(adapters.parameters(), lr=1e-4)

    for epoch in range(10):  # 适配器只用 256×256 crop 训练，loss < 0.001 即收敛
        total_loss = 0.0
        for inp, _, _ in train_loader:
            inp = inp.to(device)
            with torch.no_grad():
                target_out = backbone(inp)

            # 插入适配器后前向
            adapted_out = forward_with_adapters(inp, backbone, adapters)

            loss = nn.functional.mse_loss(adapted_out, target_out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"  Adapter init epoch {epoch+1}/10, loss={total_loss/len(train_loader):.6f}")

    return adapters


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mfd", action="store_true", help="预训练 MFD")
    parser.add_argument("--init_adapters", action="store_true", help="初始化适配器")
    parser.add_argument("--backbone", type=str, default="wdnet", choices=["wdnet", "dda"],
                        help="适配器初始化的 backbone (wdnet/dda)")
    parser.add_argument("--mfd_epochs", type=int, default=50, help="MFD 训练 epoch 数")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 数据加载 ----
    # 注意: 需要提前修改 unified_dataloader.py 中的 DATASET_ROOTS 为你的实际路径
    train_loader = get_dataloader("LCDMoire", split="train", batch_size=4)
    val_loader = get_dataloader("LCDMoire", split="test", batch_size=1)  # 适配器初始化用全分辨率

    if args.train_mfd:
        print("\n" + "="*50)
        print("Training MFD...")
        print("="*50)
        mfd = MFD()
        mfd = train_mfd(mfd, train_loader, epochs=args.mfd_epochs, device=device)
        torch.save(mfd.state_dict(), "checkpoints/mfd.pth")
        print("MFD saved to checkpoints/mfd.pth")

    elif args.init_adapters:
        print("\n" + "="*50)
        print("Initializing MoTA adapters to identity mapping...")
        print("="*50)

        # ====== 加载 backbone ======
        if args.backbone == "dda":
            sys.path.insert(0, "/root/autodl-tmp/DDA-main")
            from Net.MBCNN import MBCNN
            backbone = MBCNN(64)
            state = torch.load(
                "/root/autodl-tmp/DDA-main/result/MBCNN_aim/1pth_folder/ckpt_best.pth",
                map_location=device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            state = _strip_module_prefix(state)
            backbone.load_state_dict(state)
            backbone.eval()
            backbone = InputRangeAdapter(backbone)
            save_path = "checkpoints/adapters_init_dda.pth"
            print("DDA backbone loaded")
        else:
            backbone = load_wdnet_backbone("checkpoints/wdnet_best.pth", device)
            save_path = "checkpoints/adapters_init.pth"
            print("WDNet backbone loaded")

        # 创建适配器
        from mota.adapters import insert_mota_adapters
        adapters, channels = insert_mota_adapters(backbone)
        adapters = init_adapters_identity(adapters, backbone, train_loader, device)
        torch.save({"state_dict": adapters.state_dict(), "channels": channels}, save_path)
        print(f"Adapters saved to {save_path}")

    else:
        print("\n" + "="*50)
        print("Training baseline...")
        print("="*50)
        print("请在此处导入你的 backbone 模型 (如 WDNet), ")
        print("然后调用 train_baseline() 函数。")
        print("="*50)
