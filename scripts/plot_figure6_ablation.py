"""Plot FLUX sample-amount and screening ablations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "table4_ablation_results.csv")
    amount = df[df["ablation_type"] == "Amount"]
    screening = df[df["ablation_type"] == "Screening"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=150)

    x = np.arange(len(amount))
    axes[0].errorbar(x, amount["map50_mean"], yerr=amount["map50_std"], marker="o", label="mAP@0.5")
    axes[0].errorbar(x, amount["map5095_mean"], yerr=amount["map5095_std"], marker="s", label="mAP@0.5:0.95")
    axes[0].set_xticks(x, amount["setting"])
    axes[0].set_ylabel("mAP")
    axes[0].set_title("(a) Amount", loc="left", fontsize=10)
    axes[0].grid(axis="y", alpha=0.25)

    x2 = np.arange(len(screening))
    width = 0.34
    axes[1].bar(x2 - width / 2, screening["map50_mean"], width, yerr=screening["map50_std"], label="mAP@0.5", edgecolor="black")
    axes[1].bar(x2 + width / 2, screening["map5095_mean"], width, yerr=screening["map5095_std"], label="mAP@0.5:0.95", edgecolor="black", hatch="//")
    axes[1].set_xticks(x2, screening["setting"])
    axes[1].set_title("(b) Screening", loc="left", fontsize=10)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = ROOT / "figures" / "figure6_ablation.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(out)


if __name__ == "__main__":
    main()

