# Dataset Description

The associated Kaggle dataset is available at:

https://www.kaggle.com/datasets/yangyuanxu/sonar-flux-synthetic

It contains only synthetic/generated side-scan sonar images and YOLO-format labels. The original real side-scan sonar images used for detector validation and testing in the manuscript are not redistributed because of data-licensing restrictions.

## Released Subsets

```text
flux_lora_16_auto_screened/
  images/
  labels/
flux_lora_16_manual_600/
  images/
  labels/
metadata/
```

- `flux_lora_16_auto_screened`: 1780 selected synthetic FLUX+LoRA rank-16 images and labels.
- `flux_lora_16_manual_600`: 600 regenerated and manually annotated FLUX+LoRA rank-16 images and labels.

## Class Order

| ID | Class |
|---:|---|
| 0 | body |
| 1 | plane |
| 2 | ship |

Object-level class distributions may differ from sample-level screening statistics because some images contain more than one annotated object.

