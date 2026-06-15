"""Plot main fixed-YOLOv8n comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "table3_main_detection_results.csv")
    labels = ["Original", "Trad.-target", "Trad.-all", "FLUX+LoRA"]
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150)
    ax.bar(x - width / 2, df["map50_mean"], width, yerr=df["map50_std"], label="mAP@0.5", edgecolor="black")
    ax.bar(x + width / 2, df["map5095_mean"], width, yerr=df["map5095_std"], label="mAP@0.5:0.95", edgecolor="black", hatch="//")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = ROOT / "figures" / "figure3_main_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(out)


if __name__ == "__main__":
    main()

