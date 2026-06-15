"""Compute FID and KID between two image folders.

The implementation uses torchvision InceptionV3 DEFAULT weights and 2048-D
features, matching the protocol described in the manuscript. For large folders,
run on a machine with sufficient GPU/CPU memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageFolder(Dataset):
    def __init__(self, folder: Path):
        self.files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        self.tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as im:
            return self.tf(im.convert("RGB"))


@torch.no_grad()
def extract_features(folder: Path, batch: int, device: str) -> np.ndarray:
    ds = ImageFolder(folder)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    feats = []
    for x in loader:
        y = model(x.to(device))
        feats.append(y.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def fid(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))


def kid(a: np.ndarray, b: np.ndarray, subsets: int = 50, subset_size: int = 100) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    m = min(subset_size, len(a), len(b))
    scores = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        k_xx = polynomial_kernel(x, x, degree=3, gamma=1.0 / x.shape[1], coef0=1)
        k_yy = polynomial_kernel(y, y, degree=3, gamma=1.0 / y.shape[1], coef0=1)
        k_xy = polynomial_kernel(x, y, degree=3, gamma=1.0 / x.shape[1], coef0=1)
        scores.append(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    real = extract_features(Path(args.real), args.batch, args.device)
    gen = extract_features(Path(args.generated), args.batch, args.device)
    kid_mean, kid_std = kid(real, gen)
    print({"fid": fid(real, gen), "kid_mean": kid_mean, "kid_std": kid_std})


if __name__ == "__main__":
    main()

