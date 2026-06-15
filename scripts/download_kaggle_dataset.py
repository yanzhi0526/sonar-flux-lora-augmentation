"""Download the public synthetic sonar dataset from Kaggle.

This script does not include or manage Kaggle credentials. Configure Kaggle CLI
with your own account first, then run this helper.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DATASET = "yangyuanxu/sonar-flux-synthetic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/kaggle", help="Download/extract directory.")
    parser.add_argument("--unzip", action="store_true", help="Unzip after download.")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(out)]
    if args.unzip:
        cmd.append("--unzip")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

