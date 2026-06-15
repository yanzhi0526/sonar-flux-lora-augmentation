# Kaggle Dataset

Dataset URL:

https://www.kaggle.com/datasets/yangyuanxu/sonar-flux-synthetic

The Kaggle release contains only synthetic/generated side-scan sonar images and YOLO-format annotations. It does not redistribute the original real sonar train/validation/test images.

## Structure

```text
flux_lora_16_auto_screened/images
flux_lora_16_auto_screened/labels
flux_lora_16_manual_600/images
flux_lora_16_manual_600/labels
metadata/
```

## Subsets

- `flux_lora_16_auto_screened`: 1780 generated images selected through label audit and quality screening.
- `flux_lora_16_manual_600`: 600 regenerated FLUX+LoRA images with manual annotations.

Both subsets use the canonical YOLO class order: `0 body`, `1 plane`, `2 ship`.

