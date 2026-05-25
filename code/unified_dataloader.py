"""
统一数据加载器 — 所有 baseline 和 MoTA 共用同一个预处理管线。

支持的数据集:
  - LCDMoire:  1024×1024 合成图像对, 9K train / 1.2K test
  - TIP2018:   约 400×400 真实屏幕拍摄, 135K train / 1.5K test

用法:
  train_loader = get_dataloader("LCDMoire", split="train", batch_size=4)
  test_loader  = get_dataloader("TIP2018", split="test", batch_size=1)
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ============================================================
# 配置：修改这里的路径为你 AutoDL 上的实际路径
# ============================================================
# LCDMoire 的实际结构:
#   LCDMoire/
#     train/moire/*.png   (摩尔纹, 9000张)
#     train/clean/*.png   (干净图, 9000张)
#     val_moire/*.png     (摩尔纹, 1200张)
#     val_clean/*.png     (干净图, 1200张)
#
# TIP2018 的实际结构:
#   TIP2018/
#     testdata/source/*.png   (摩尔纹)
#     testdata/target/*.png   (干净图)
DATASET_ROOTS = {
    "LCDMoire": "/root/autodl-tmp/LCDMoire",
    "TIP2018":  "/root/autodl-tmp/Tip-2018",
}

# 各数据集的自定义子目录结构
# 格式: { split: {"input": "相对路径", "target": "相对路径"} }
# 路径相对于 DATASET_ROOTS[数据集名]
CUSTOM_SPLIT_DIRS = {
    "LCDMoire": {
        "train":  {"input": "train/moire", "target": "train/clean"},
        "test":   {"input": "val_moire",   "target": "val_clean"},
    },
    "TIP2018": {
        "train": None,  # TIP2018 训练集目录结构待确认，暂不支持
        "test":   {"input": "testdata/source", "target": "testdata/target"},
    },
}

# ============================================================
# 统一的预处理参数
# ============================================================
PATCH_SIZE  = 256          # 训练时随机 crop 的大小
MEAN = [0.5, 0.5, 0.5]     # 归一化均值 (RGB)
STD  = [0.5, 0.5, 0.5]     # 归一化标准差


class DemoireDataset(Dataset):
    """
    通用去摩尔纹数据集。
    假设目录结构:
      root/
        train/
          input/    # 摩尔纹图像
          target/   # 干净图像
        test/
          input/
          target/
    如果实际结构不同，修改 _scan_files 方法。
    """

    def __init__(self, root: str, split: str = "train", is_train: bool = True,
                 dataset_name: str = ""):
        self.root = Path(root)
        self.split = split
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.pairs = self._scan_files()

        # 训练增强 (所有 baseline 统一)
        if is_train:
            self.geo_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ])
        else:
            self.geo_transform = None

    def _scan_files(self):
        """扫描数据集目录, 返回 (input_path, target_path) 列表。"""
        split_dir = self.root / self.split

        # ---- 自定义结构（在 CUSTOM_SPLIT_DIRS 中配置的数据集）----
        custom = CUSTOM_SPLIT_DIRS.get(self.dataset_name, {})
        if self.split in custom:
            dirs = custom[self.split]
            if dirs is None:
                raise ValueError(
                    f"数据集 {self.dataset_name} 不支持 split='{self.split}'。"
                    f"支持的 split: {[k for k, v in custom.items() if v is not None]}")
            input_dir  = self.root / dirs["input"]
            target_dir = self.root / dirs["target"]
            input_files  = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
            target_files = sorted(list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg")))
            if len(input_files) == len(target_files):
                print(f"[{self.dataset_name}/{self.split}] "
                      f"input={len(input_files)}, target={len(target_files)} → {len(input_files)} pairs")
                return list(zip(input_files, target_files))
            else:
                raise RuntimeError(
                    f"{self.dataset_name} {self.split}: input={len(input_files)}, target={len(target_files)}, "
                    f"数量不匹配！\n  input:  {input_dir}\n  target: {target_dir}"
                )

        # ---- 通用逻辑 (用于 TIP2018 等) ----
        # 尝试匹配常见的目录命名约定
        input_dirs  = ["input", "moire", "moire_images", "blur", "hazy", "degraded"]
        target_dirs = ["target", "clean", "clean_images", "sharp", "gt", "ground_truth", "clear"]

        input_dir = None
        target_dir = None
        for d in input_dirs:
            if (split_dir / d).exists():
                input_dir = split_dir / d
                break
        for d in target_dirs:
            if (split_dir / d).exists():
                target_dir = split_dir / d
                break

        # 如果上面的都不匹配，尝试直接列出所有图像文件
        if input_dir is None or target_dir is None:
            all_images = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
            if len(all_images) == 0:
                # 递归搜索
                all_images = sorted(split_dir.rglob("*.png")) + sorted(split_dir.rglob("*.jpg"))
            # 假设前半是 input, 后半是 target
            n = len(all_images) // 2
            input_paths  = all_images[:n]
            target_paths = all_images[n:]
            return list(zip(input_paths, target_paths))

        input_files  = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
        target_files = sorted(list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg")))

        # 按文件名匹配
        pairs = []
        for inp in input_files:
            stem = inp.stem
            # 尝试找到同名 target — 某些数据集的 target 文件名可能有后缀
            for tgt in target_files:
                tgt_stem = tgt.stem
                # 去掉可能的 target 后缀 (如 _clean, _gt, _target, _c)
                for suffix in ["_clean", "_gt", "_target", "_c", "_clear"]:
                    if tgt_stem.endswith(suffix):
                        tgt_stem = tgt_stem[:-len(suffix)]
                        break
                if stem == tgt_stem:
                    pairs.append((inp, tgt))
                    break

        if len(pairs) == 0:
            # 最后的 fallback：按排序位置匹配
            if len(input_files) == len(target_files):
                pairs = list(zip(input_files, target_files))

        print(f"[{self.root.name}/{self.split}] found {len(pairs)} pairs")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]

        # 读取图像 (RGB)
        inp_img = Image.open(inp_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        # 统一 resize: 确保短边 >= PATCH_SIZE (用于后续 crop)
        w, h = inp_img.size
        if min(w, h) < PATCH_SIZE:
            scale = PATCH_SIZE / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            inp_img = inp_img.resize((new_w, new_h), Image.BICUBIC)
            tgt_img = tgt_img.resize((new_w, new_h), Image.BICUBIC)

        # 转为 Tensor (C, H, W), [0, 1]
        inp = TF.to_tensor(inp_img)
        tgt = TF.to_tensor(tgt_img)

        # 训练时: 随机 crop + 几何增强
        if self.is_train:
            # 先做几何增强 (输入和目标做相同的变换)
            c, h, w = inp.shape
            # 拼接做 joint transform
            stacked = torch.cat([inp, tgt], dim=0)  # (6, H, W)
            stacked = self.geo_transform(stacked)
            inp, tgt = stacked[:3], stacked[3:]

            # 随机 256×256 crop (同一个位置)
            _, h, w = inp.shape
            if h > PATCH_SIZE and w > PATCH_SIZE:
                top  = random.randint(0, h - PATCH_SIZE)
                left = random.randint(0, w - PATCH_SIZE)
                inp = inp[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                tgt = tgt[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]

        # 归一化到 [-1, 1]
        inp = (inp - 0.5) * 2.0
        tgt = (tgt - 0.5) * 2.0

        return inp, tgt, str(inp_path)


def get_dataloader(dataset_name: str, split: str = "train",
                   batch_size: int = 4, num_workers: int = 4):
    """获取指定数据集的 DataLoader。"""
    root = DATASET_ROOTS.get(dataset_name)
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_ROOTS.keys())}")

    is_train = (split == "train")
    dataset = DemoireDataset(root=root, split=split, is_train=is_train,
                             dataset_name=dataset_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,  # 训练时丢弃不完整 batch, 避免 BN 出问题
    )
    return loader


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    for ds_name in ["LCDMoire", "TIP2018"]:
        root = DATASET_ROOTS[ds_name]
        if os.path.exists(root):
            print(f"\n{'='*50}")
            print(f"Testing {ds_name} @ {root}")
            loader = get_dataloader(ds_name, split="train", batch_size=2)
            inp, tgt, path = next(iter(loader))
            print(f"  input  shape: {inp.shape}  range: [{inp.min():.2f}, {inp.max():.2f}]")
            print(f"  target shape: {tgt.shape}  range: [{tgt.min():.2f}, {tgt.max():.2f}]")
        else:
            print(f"\n[SKIP] {ds_name} not found at {root} — 请下载后修改 DATASET_ROOTS")
