# sonar-flux-lora-augmentation

Code and documentation for the paper:

**Diffusion-Model-Based Data Augmentation for Target Detection in Side-Scan Sonar Images**

This repository complements the public Kaggle dataset:

https://www.kaggle.com/datasets/yangyuanxu/sonar-flux-synthetic

The Kaggle dataset stores synthetic/generated side-scan sonar images and YOLO-format annotations. This GitHub repository stores reproducibility code, configuration templates, result tables, plotting scripts, and documentation for the manuscript revision experiments.

## Important Data Note

The original real side-scan sonar images are **not redistributed** due to data-licensing restrictions. The released Kaggle dataset contains only synthetic/generated images and YOLO-format annotations.

Released synthetic subsets:

- `flux_lora_16_auto_screened`: 1780 synthetic images and labels.
- `flux_lora_16_manual_600`: 600 synthetic images and labels.

Class order:

| ID | Class |
|---:|---|
| 0 | body |
| 1 | plane |
| 2 | ship |

## Repository Contents

- `configs/`: YOLO dataset configuration templates.
- `prompts/`: sonar-oriented prompt templates.
- `scripts/`: utility scripts for downloading, auditing, subset construction, training, evaluation, image-quality calculation, and plotting.
- `results/`: manuscript-ready CSV result summaries.
- `figures/`: selected manuscript result figures.
- `docs/`: dataset, screening, and reproduction notes.

## Quick Start

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset:

```bash
python scripts/download_kaggle_dataset.py --output data/kaggle
```

Audit YOLO labels:

```bash
python scripts/label_audit.py --dataset data/kaggle/flux_lora_16_auto_screened --out audit_auto_screened.csv
python scripts/label_audit.py --dataset data/kaggle/flux_lora_16_manual_600 --out audit_manual_600.csv
```

Plot manuscript figures from the included CSV tables:

```bash
python scripts/plot_figure3_main_comparison.py
python scripts/plot_figure6_ablation.py
python scripts/plot_figure7_lora_rank.py
python scripts/plot_figure8_backbone.py
```

Optional YOLOv8n training requires user-provided real train/validation/test data:

```bash
python scripts/train_yolov8n.py --data configs/detection_flux_lora_auto_screened.yaml --seed 42
```

Exact manuscript detection metrics require the original real validation/test images, which are not redistributed.

## Main Result Summary

Fixed YOLOv8n, averaged over seeds 42, 43, and 44:

| Method | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Original YOLOv8n | 0.7400 ± 0.0132 | 0.3994 ± 0.0187 |
| FLUX+LoRA auto-screened | 0.8582 ± 0.0328 | 0.5115 ± 0.0164 |

## License

Repository code is released under the MIT License. The Kaggle dataset is released separately under **CC BY-NC-SA 4.0**.

## Citation

Please cite the paper and dataset if you use this repository or the associated synthetic side-scan sonar dataset. See `CITATION.cff`.

