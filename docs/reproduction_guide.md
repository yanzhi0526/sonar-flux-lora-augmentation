# Reproduction Guide

This repository provides scripts and configuration templates for reproducing the analysis workflow around the released synthetic dataset.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download the Kaggle Dataset

Configure Kaggle CLI with your own credentials, then run:

```bash
python scripts/download_kaggle_dataset.py --output data/kaggle --unzip
```

## 3. Audit Labels

```bash
python scripts/label_audit.py --dataset data/kaggle/flux_lora_16_auto_screened --out audit_auto_screened.csv
python scripts/label_audit.py --dataset data/kaggle/flux_lora_16_manual_600 --out audit_manual_600.csv
```

## 4. Prepare Real Validation/Test Data

The manuscript uses real-only validation and test splits. These real images are not redistributed. To reproduce exact detection metrics, place the original real train/validation/test images in your own local dataset structure and update the YAML templates under `configs/`.

## 5. Train YOLOv8n

```bash
python scripts/train_yolov8n.py --data configs/detection_flux_lora_auto_screened.yaml --seed 42 --name flux_lora_seed42
```

The manuscript used YOLOv8n, image size 640, 100 epochs, batch size 24, SGD optimizer, initial learning rate 0.01, and disabled YOLO built-in random augmentation.

## 6. Evaluate

```bash
python scripts/eval_yolov8n.py --weights runs/fixed_yolov8n/flux_lora_seed42/weights/best.pt --data configs/detection_flux_lora_auto_screened.yaml --split test
```

## 7. Plot Result Figures

```bash
python scripts/plot_figure3_main_comparison.py
python scripts/plot_figure6_ablation.py
python scripts/plot_figure7_lora_rank.py
python scripts/plot_figure8_backbone.py
```

The included CSV files reproduce manuscript summary plots. Exact detection metrics require the non-redistributed real validation/test split.

