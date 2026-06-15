"""Plot controlled diffusion-backbone comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "table6_backbone_results.csv")
    labels = ["FLUX-base", "SD1.5+LoRA", "FLUX+LoRA"]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=150)
    width = 0.34
    axes[0].bar(x - width / 2, df["map50_mean"], width, yerr=df["map50_std"], label="mAP@0.5", edgecolor="black")
    axes[0].bar(x + width / 2, df["map5095_mean"], width, yerr=df["map5095_std"], label="mAP@0.5:0.95", edgecolor="black", hatch="//")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("(a) Detection", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, df["fid"], edgecolor="black")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("FID")
    axes[1].set_title("(b) FID", loc="left", fontsize=10)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, df["kid"], edgecolor="black")
    axes[2].set_xticks(x, labels, rotation=15, ha="right")
    axes[2].set_ylabel("KID")
    axes[2].set_title("(c) KID", loc="left", fontsize=10)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = ROOT / "figures" / "figure8_backbone.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(out)


if __name__ == "__main__":
    main()

