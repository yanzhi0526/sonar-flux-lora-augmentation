"""Copy selected YOLO images and labels into a new subset folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--source-images", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--filename-column", default="filename")
    args = parser.parse_args()

    selected = pd.read_csv(args.selected_csv)
    src_img = Path(args.source_images)
    src_lbl = Path(args.source_labels)
    target = Path(args.target)
    out_img = target / "images"
    out_lbl = target / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in selected[args.filename_column].astype(str):
        img = src_img / name
        if not img.exists():
            matches = list(src_img.glob(Path(name).stem + ".*"))
            if not matches:
                raise FileNotFoundError(f"Missing image for {name}")
            img = matches[0]
        label = src_lbl / (img.stem + ".txt")
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {img.name}")
        shutil.copy2(img, out_img / img.name)
        shutil.copy2(label, out_lbl / label.name)
        copied += 1
    print(f"Copied {copied} image/label pairs to {target}")


if __name__ == "__main__":
    main()

