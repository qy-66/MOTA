"""
定性对比可视化：从 TIP2018 随机选取样本，对比 frozen vs MoTA 输出。
生成论文 Figure 定性对比图 (Sec 4.6)。

用法:
  python visualize_qualitative.py --tag wdnet     # WDNet backbone
  python visualize_qualitative.py --tag dda       # DDA backbone
  python visualize_qualitative.py --tag wdnet --n_samples 8
"""
import os, sys, argparse, random
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataloader import get_dataloader
from mota.wdnet_loader import load_wdnet_backbone
from mota.mfd import MFD
from mota.adapters import insert_mota_adapters
from mota.tta import mota_adapt
from mota.utils import set_seed


def tensor_to_np(t):
    """[-1,1] tensor → [0,255] uint8 numpy (H,W,3)"""
    t = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    t = torch.clamp(t, -1, 1)
    t = (t + 1) / 2 * 255
    return t.numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="wdnet",
                        choices=["wdnet", "dda"])
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="results/qualitative.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_dataloader("TIP2018", split="test", batch_size=1)
    indices = random.sample(range(len(loader.dataset)), min(args.n_samples, len(loader.dataset)))

    if args.tag == "wdnet":
        model = load_wdnet_backbone("checkpoints/wdnet_best.pth", device)
        adapter_ckpt = "checkpoints/adapters_init.pth"
    else:
        from mota.wdnet_loader import load_dda_backbone
        model = load_dda_backbone(device=device)
        adapter_ckpt = "checkpoints/adapters_init_dda.pth"
    model.eval()
    print(f"{args.tag} backbone loaded")

    mfd = MFD().to(device).eval()
    ckpt_mfd = torch.load("checkpoints/mfd.pth", map_location=device)
    mfd.load_state_dict(ckpt_mfd.get("state_dict", ckpt_mfd))

    adapters, _ = insert_mota_adapters(model)
    adapters = adapters.to(device)
    ckpt = torch.load(adapter_ckpt, map_location=device)
    init_state = ckpt.get("state_dict", ckpt)
    adapters.load_state_dict(init_state)

    fig, axes = plt.subplots(args.n_samples, 4, figsize=(16, 4 * args.n_samples))
    if args.n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        inp, tgt, _ = loader.dataset[idx]
        inp = inp.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)

        # frozen
        with torch.no_grad():
            frozen = model(inp)
            if isinstance(frozen, (list, tuple)):
                frozen = frozen[0]

        # Reset adapters to initial state per image to prevent cross-image drift
        adapters.load_state_dict(init_state)
        mota_out = mota_adapt(inp, model, adapters, mfd, T=15)

        axes[row, 0].imshow(tensor_to_np(inp))
        axes[row, 0].set_title("Input (moiré)")
        axes[row, 1].imshow(tensor_to_np(frozen))
        axes[row, 1].set_title(f"Frozen {args.tag}")
        axes[row, 2].imshow(tensor_to_np(mota_out))
        axes[row, 2].set_title(f"MoTA ({args.tag})")
        axes[row, 3].imshow(tensor_to_np(tgt))
        axes[row, 3].set_title("Ground Truth")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
