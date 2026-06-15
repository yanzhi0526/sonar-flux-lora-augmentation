"""Audit a YOLO-format image/label dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_dirs(dataset: Path) -> tuple[Path, Path]:
    images = dataset / "images"
    labels = dataset / "labels"
    if not images.exists() or not labels.exists():
        raise FileNotFoundError("Expected dataset/images and dataset/labels.")
    return images, labels


def audit_label(path: Path) -> tuple[int, Counter, list[str]]:
    errors: list[str] = []
    counts: Counter = Counter()
    rows = 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, counts, ["empty_label"]
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        rows += 1
        if len(parts) != 5:
            errors.append(f"line_{line_no}:expected_5_columns")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line_{line_no}:non_numeric")
            continue
        if cls not in {0, 1, 2}:
            errors.append(f"line_{line_no}:invalid_class_{cls}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            errors.append(f"line_{line_no}:bbox_not_normalized")
        if w <= 0 or h <= 0:
            errors.append(f"line_{line_no}:non_positive_bbox")
        counts[cls] += 1
    return rows, counts, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset folder containing images/ and labels/.")
    parser.add_argument("--out", default="label_audit.csv", help="CSV output path.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    images_dir, labels_dir = find_dirs(dataset)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    label_files = sorted(labels_dir.glob("*.txt"))
    label_map = {p.stem: p for p in label_files}

    class_counts: Counter = Counter()
    image_class_counts: Counter = Counter()
    missing = empty = invalid_rows = 0
    per_file = []

    for img in images:
        label = label_map.get(img.stem)
        if label is None:
            missing += 1
            per_file.append([img.name, "", 0, "missing_label"])
            continue
        rows, counts, errors = audit_label(label)
        if "empty_label" in errors:
            empty += 1
        if errors and errors != ["empty_label"]:
            invalid_rows += len(errors)
        class_counts.update(counts)
        for cls in counts:
            image_class_counts[cls] += 1
        per_file.append([img.name, label.name, rows, ";".join(errors)])

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset", str(dataset)])
        writer.writerow(["images", len(images)])
        writer.writerow(["label_files", len(label_files)])
        writer.writerow(["missing_labels", missing])
        writer.writerow(["empty_labels", empty])
        writer.writerow(["invalid_label_rows", invalid_rows])
        for cls, name in [(0, "body"), (1, "plane"), (2, "ship")]:
            writer.writerow([f"objects_{name}", class_counts[cls]])
            writer.writerow([f"images_with_{name}", image_class_counts[cls]])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

