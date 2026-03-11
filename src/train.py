#!/usr/bin/env python3
"""Train a baseline classifier for the food recognition assignment."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


@dataclass
class Record:
    img_name: str
    label: int  # 0-based


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def read_labels_csv(csv_path: Path) -> list[Record]:
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"Empty CSV: {csv_path}")

    header = lines[0].strip().split(",")
    if header != ["img_name", "label"]:
        raise ValueError(f"Expected header 'img_name,label' in {csv_path}, got: {header}")

    records: list[Record] = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Malformed line {i} in {csv_path}: {line!r}")
        img_name, label_str = parts
        label = int(label_str)
        records.append(Record(img_name=img_name, label=label - 1))
    return records


class FoodTrainDataset(Dataset):
    def __init__(self, image_dir: Path, records: list[Record], tfm: transforms.Compose):
        self.image_dir = image_dir
        self.records = records
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = self.image_dir / rec.img_name
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x = self.tfm(img)
        y = torch.tensor(rec.label, dtype=torch.long)
        return x, y


def split_records(records: list[Record], val_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    idx = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_size = int(len(records) * val_ratio)
    val_idx = set(idx[:val_size])

    train_records = [records[i] for i in range(len(records)) if i not in val_idx]
    val_records = [records[i] for i in range(len(records)) if i in val_idx]
    return train_records, val_records


def build_model(num_classes: int, try_pretrained: bool) -> nn.Module:
    if try_pretrained and ResNet18_Weights is not None:
        try:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            print(f"Could not load pretrained weights ({exc}); using random init.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    labels_csv = data_dir / "train_labels.csv"
    train_root = data_dir / "train_set"
    image_dir = find_image_dir(train_root)

    all_records = read_labels_csv(labels_csv)
    train_records, val_records = split_records(all_records, args.val_ratio, args.seed)

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = FoodTrainDataset(image_dir, train_records, train_tfm)
    val_ds = FoodTrainDataset(image_dir, val_records, eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=args.num_classes, try_pretrained=args.pretrained)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    meta_path = out_dir / "train_meta.json"

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    print(f"Training on {device} with {len(train_records)} train / {len(val_records)} val images")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if val_acc > (best_val_acc + args.min_delta):
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping triggered: "
                f"no val_acc improvement > {args.min_delta} for {args.early_stopping_patience} epochs."
            )
            break

    meta = {
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "num_classes": args.num_classes,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "checkpoint": str(best_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint to: {best_path}")
    print(f"Saved metadata to: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline food classifier")
    parser.add_argument("--data-dir", type=str, default="Data", help="Directory containing train_set/ and CSV files")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Where checkpoints are stored")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop if val_acc does not improve by at least --min-delta for this many epochs (0 disables)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum required val_acc improvement to reset early stopping counter",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Try loading ImageNet pretrained weights (falls back to random init if unavailable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
