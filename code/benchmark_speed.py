"""推理速度计时：WDNet / DDA / MoTA — Table 5"""
import sys, os, torch, time, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 从 TIP2018 加载一张真实图像 ----
from unified_dataloader import get_dataloader
loader = get_dataloader("TIP2018", split="test", batch_size=1)
real_img, _, _ = next(iter(loader))


def benchmark(name, model, inp, do_mota=False, adapters=None, mfd=None):
    """计时：warmup 10 次 + 测 50 次取均值"""
    model = model.cuda().eval()
    x = inp.cuda()

    if do_mota:
        from mota.tta import mota_adapt
        for _ in range(3):
            _ = mota_adapt(x, model, adapters, mfd, T=1)
            torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.synchronize()
        _ = mota_adapt(x, model, adapters, mfd, T=15)
        torch.cuda.synchronize()
        return (time.time() - t0) * 1000

    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()

    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return np.mean(times) * 1000


# ====== WDNet ======
from mota.wdnet_loader import load_wdnet_backbone
wdnet = load_wdnet_backbone("checkpoints/wdnet_best.pth", "cuda")
wdnet_infer = benchmark("WDNet", wdnet, real_img)
print(f"WDNet inference: {wdnet_infer:.1f} ms")

# ====== DDA ======
from mota.wdnet_loader import load_dda_backbone
dda = load_dda_backbone(device="cuda")
dda_infer = benchmark("DDA", dda, real_img)
print(f"DDA  inference: {dda_infer:.1f} ms")

# ====== MoTA (WDNet, T=15) ======
from mota.tta import mota_adapt
from mota.adapters import insert_mota_adapters
from mota.mfd import MFD

mfd = MFD().cuda().eval()
ckpt_mfd = torch.load("checkpoints/mfd.pth")
mfd.load_state_dict(ckpt_mfd.get("state_dict", ckpt_mfd))

adapters_w, _ = insert_mota_adapters(wdnet)
ckpt_aw = torch.load("checkpoints/adapters_init.pth")
adapters_w.load_state_dict(ckpt_aw.get("state_dict", ckpt_aw))
adapters_w.to("cuda")

mota_total = benchmark("MoTA-WDNet", wdnet, real_img, do_mota=True,
                       adapters=adapters_w, mfd=mfd)
print(f"MoTA  total (WDNet, T=15): {mota_total:.1f} ms")
print(f"  TTA overhead: {mota_total - wdnet_infer:.1f} ms")

# ====== MoTA (DDA, T=15) ======
adapters_d, _ = insert_mota_adapters(dda)
ckpt_ad = torch.load("checkpoints/adapters_init_dda.pth")
adapters_d.load_state_dict(ckpt_ad.get("state_dict", ckpt_ad))
adapters_d.to("cuda")

mota_dda_total = benchmark("MoTA-DDA", dda, real_img, do_mota=True,
                           adapters=adapters_d, mfd=mfd)
print(f"MoTA  total (DDA, T=15): {mota_dda_total:.1f} ms")
print(f"  TTA overhead: {mota_dda_total - dda_infer:.1f} ms")
