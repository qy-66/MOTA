# MoTA: Domain-Adaptive Demoiréing via Adapter-Based Test-Time Adaptation

Yu Qiu, Hunan University

## Overview

MoTA is a test-time adaptation framework for image demoiréing. It inserts a lightweight Fourier Domain Adapter (FDA, <2% backbone parameters) into a frozen pre-trained demoiréing model, guided by a Moiré-Aware Self-Supervision (MASS) signal. Only the adapter is updated at inference time — no paired target-domain data required.

**Key results:**
- In-domain (LCDMoire): +2.63 dB PSNR over frozen WDNet
- Cross-domain (LCDMoire→TIP2018): performance below frozen baseline (17.50 vs 18.91 dB)
- Honest analysis of cross-domain failure with quantitative diagnostics

## Code Structure

```
code/
├── eval.py                    # Main evaluation entry
├── train.py                   # MFD training + adapter initialization
├── unified_dataloader.py      # Data loading (LCDMoire + TIP2018)
├── diagnose_cd.py             # Cross-domain diagnostics
├── visualize_mass.py          # MASS signal visualization
├── visualize_qualitative.py   # Qualitative comparison
├── benchmark_speed.py         # Inference speed measurement
├── mota/
│   ├── tta.py                 # TTA loop (mota/full_ft/lora/no_mass)
│   ├── adapters.py            # FDA + SGA adapters
│   ├── mass.py                # MASS pseudo-clean signal
│   ├── mfd.py                 # Moiré Frequency Detector + DWT/IDWT
│   ├── wdnet_loader.py        # WDNet backbone loader
│   └── utils.py               # Shared utilities
└── requirements.txt           # Python dependencies
```

## Paper

See `output/paper.pdf` for the full manuscript.

## Datasets

- **LCDMoire**: Synthetic moiré dataset (AIM 2019)
- **TIP2018**: Real screen-captured moiré images

## Citation

If you use this work, please cite:
```
@article{qiu2026mota,
  title={Domain-Adaptive Demoiréing via Adapter-Based Test-Time Adaptation},
  author={Qiu, Yu},
  year={2026}
}
```
