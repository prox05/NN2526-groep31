#!/usr/bin/env python3
"""Generate assignment submission CSV from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def find_image_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if any(root.glob(ext)):
            return root

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        matches = list(root.rglob(ext))
        if matches:
            return matches[0].parent

    raise FileNotFoundError(f"No images found under {root}")


def read_sample_csv(sample_csv: Path) -> list[str]:
    lines = sample_csv.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"Empty CSV: {sample_csv}")

    header = lines[0].strip().split(",")
    if header != ["img_name", "label"]:
        raise ValueError(f"Expected header 'img_name,label' in {sample_csv}, got: {header}")

    names: list[str] = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Malformed line {i} in {sample_csv}: {line!r}")
        names.append(parts[0])
    return names


class FoodTestDataset(Dataset):
    def __init__(self, image_dir: Path, image_names: list[str], tfm: transforms.Compose):
        self.image_dir = image_dir
        self.image_names = image_names
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img_path = self.image_dir / name
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x = self.tfm(img)
        return x, name


def load_model(checkpoint_path: Path, num_classes: int) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_classes = int(ckpt.get("num_classes", num_classes))
    image_size = int(ckpt.get("image_size", 224))

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, ckpt_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, image_size


def predict(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    sample_csv = data_dir / "sample.csv"
    test_root = data_dir / "test_set"
    test_image_dir = find_image_dir(test_root)

    image_names = read_sample_csv(sample_csv)
    model, image_size = load_model(Path(args.checkpoint), args.num_classes)

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = FoodTestDataset(test_image_dir, image_names, tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: list[tuple[str, int]] = []
    with torch.no_grad():
        for x, names in loader:
            x = x.to(device)
            logits = model(x)
            batch_preds = logits.argmax(dim=1).cpu().tolist()
            for n, p in zip(names, batch_preds):
                preds.append((n, p + 1))  # convert back to 1-based labels

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("img_name,label\n")
        for name, label in preds:
            f.write(f"{name},{label}\n")

    print(f"Wrote submission with {len(preds)} rows to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and generate submission CSV")
    parser.add_argument("--data-dir", type=str, default="Data")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--output-csv", type=str, default="submissions/submission.csv")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
