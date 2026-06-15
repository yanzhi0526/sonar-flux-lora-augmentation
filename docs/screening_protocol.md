# Screening Protocol

The auto-screened FLUX+LoRA subset was produced from generated candidates through label and quality checks. The screening procedure did not use validation or test detection results.

## Label Auditing

The label-audit stage checked:

- missing label files;
- empty label files;
- invalid class IDs;
- invalid YOLO bounding-box format;
- normalized bounding-box coordinates;
- positive box width and height.

## Quality Screening Criteria

Generated samples were retained when they showed:

- clear target structure;
- plausible target-highlight and acoustic-shadow relationship;
- consistent seabed background appearance;
- reliable bounding-box annotation.

## Screening Statistics

- Raw generated images: 2317
- Valid candidates after label audit: 2079
- Final screened samples: 1780
- Acceptance rate from raw generation: 76.82%
- Acceptance rate from valid candidates: 85.62%

The released `flux_lora_16_auto_screened` subset contains 1780 synthetic images. Object-level class distributions may differ from sample-level screening statistics because some retained samples contain multiple annotated objects.

