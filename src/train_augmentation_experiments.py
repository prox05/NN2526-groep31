#!/usr/bin/env python3
"""
Augmentation Ablation Study – NN2526 Groep 31
==============================================
Runs 5 augmentation configurations sequentially and saves results per config.

Usage in Colab:
    !python train_augmentation_experiments.py --data-dir . --epochs 5 --pretrained

Results are saved to:  augmentation_results/
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from torchvision.models import ResNet50_Weights
except Exception:
    ResNet50_Weights = None

# ── Pretty printing ────────────────────────────────────────────────────────────

SEP  = "=" * 70
SEP2 = "-" * 70

def pprint(msg: str, color: str = ""):
    colors = {"green": "\033[92m", "yellow": "\033[93m",
              "cyan": "\033[96m", "red": "\033[91m", "bold": "\033[1m", "": ""}
    reset = "\033[0m" if color else ""
    print(f"{colors[color]}{msg}{reset}", flush=True)

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Record:
    img_name: str
    label: int

@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float

@dataclass
class ExperimentResult:
    config_name: str
    description: str
    epochs: list[EpochResult] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    total_time_sec: float = 0.0

# ── Augmentation configs ───────────────────────────────────────────────────────

def get_augmentation_configs(image_size: int) -> dict[str, dict]:
    """
    5 augmentation strategies to compare.

    Scientific references per config are printed during training.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    configs = {

        # ── A: No augmentation (baseline) ────────────────────────────────────
        "A_no_augmentation": {
            "description": "Baseline – resize only, no augmentation",
            "paper_refs": [
                "He et al. (2016) – Deep Residual Learning (ResNet). CVPR.",
            ],
            "why": (
                "Baseline to measure the raw improvement from augmentation. "
                "Without augmentation, the model can only learn from the exact "
                "training pixels, making it highly prone to overfitting on small datasets."
            ),
            "train_tfm": transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]),
        },

        # ── B: Flip + Crop (geometric only) ──────────────────────────────────
        "B_geometric_only": {
            "description": "Geometric aug: RandomResizedCrop + HorizontalFlip",
            "paper_refs": [
                "Krizhevsky et al. (2012) – AlexNet: random crops + flips first used in deep learning. NeurIPS.",
                "Shorten & Khoshgoftaar (2019) – A survey on Image Data Augmentation. J. Big Data.",
            ],
            "why": (
                "Horizontal flips preserve food class identity (a pizza flipped is still pizza). "
                "Random crops force the model to be location-invariant and simulate "
                "different camera framings, a common real-world variation."
            ),
            "train_tfm": transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        },

        # ── C: Geometric + ColorJitter ────────────────────────────────────────
        "C_geometric_plus_color": {
            "description": "Geometric aug + ColorJitter (brightness/contrast/saturation)",
            "paper_refs": [
                "Shorten & Khoshgoftaar (2019) – A survey on Image Data Augmentation. J. Big Data.",
                "Howard (2013) – Some improvements on deep convolutional neural network based image classification. arXiv.",
            ],
            "why": (
                "Food photography varies heavily in lighting conditions (restaurant lighting, "
                "daylight, flash). ColorJitter simulates this variation, preventing the model "
                "from relying on absolute colour values rather than food texture/shape."
            ),
            "train_tfm": transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)],
                    p=0.6
                ),
                transforms.ToTensor(),
                normalize,
            ]),
        },

        # ── D: Full pipeline (current best) ───────────────────────────────────
        "D_full_pipeline": {
            "description": "Full aug: Crop + Flip + ColorJitter + Rotation (current config)",
            "paper_refs": [
                "Shorten & Khoshgoftaar (2019) – A survey on Image Data Augmentation. J. Big Data.",
                "Cubuk et al. (2019) – AutoAugment: Learning augmentation policies from data. CVPR.",
            ],
            "why": (
                "RandomRotation adds rotational invariance – handy since plated food "
                "can appear at any angle in photos. Combines all geometric and colour "
                "augmentations for maximum variety."
            ),
            "train_tfm": transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
                    p=0.4
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ]),
        },

        # ── E: Aggressive augmentation (RandAugment) ─────────────────────────
        "E_randaugment": {
            "description": "Aggressive aug: Full pipeline + RandAugment + RandomErasing",
            "paper_refs": [
                "Cubuk et al. (2020) – RandAugment: Practical automated data augmentation. NeurIPS.",
                "Zhong et al. (2020) – Random Erasing Data Augmentation. AAAI.",
            ],
            "why": (
                "RandAugment searches over a large augmentation space (shear, posterize, solarize, etc.) "
                "automatically. RandomErasing simulates occlusion (e.g. a hand covering part of the dish), "
                "forcing the model to use global context rather than local features."
            ),
            "train_tfm": transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            ]),
        },
    }

    return configs

# ── Utility functions ──────────────────────────────────────────────────────────

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
    header = lines[0].strip().split(",")
    if header != ["img_name", "label"]:
        raise ValueError(f"Bad header in {csv_path}: {header}")
    records = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad line {i}: {line!r}")
        records.append(Record(img_name=parts[0], label=int(parts[1]) - 1))
    return records

def split_records(records, val_ratio, seed):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    val_size = int(len(records) * val_ratio)
    val_idx = set(idx[:val_size])
    train_r = [records[i] for i in range(len(records)) if i not in val_idx]
    val_r   = [records[i] for i in range(len(records)) if i in val_idx]
    return train_r, val_r

class FoodDataset(Dataset):
    def __init__(self, image_dir, records, tfm):
        self.image_dir = image_dir
        self.records   = records
        self.tfm       = tfm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with Image.open(self.image_dir / rec.img_name) as img:
            x = self.tfm(img.convert("RGB"))
        return x, torch.tensor(rec.label, dtype=torch.long)

def build_model(num_classes, pretrained):
    if pretrained and ResNet50_Weights is not None:
        try:
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as e:
            pprint(f"Pretrained load failed ({e}), using random init.", "yellow")
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

# ── Single experiment runner ───────────────────────────────────────────────────

def run_experiment(
    config_name: str,
    config: dict,
    records: list[Record],
    image_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> ExperimentResult:

    pprint(f"\n{SEP}", "cyan")
    pprint(f"  EXPERIMENT {config_name}", "bold")
    pprint(f"  {config['description']}", "cyan")
    pprint(SEP, "cyan")

    pprint("\n  WHY this augmentation?", "yellow")
    print(f"  {config['why']}\n")

    pprint("  Scientific references:", "yellow")
    for ref in config["paper_refs"]:
        print(f"    • {ref}")
    print()

    set_seed(args.seed)

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    eval_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FoodDataset(image_dir, train_records, config["train_tfm"])
    val_ds   = FoodDataset(image_dir, val_records,   eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.num_classes, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss()

    result = ExperimentResult(config_name=config_name, description=config["description"])
    t0 = time.time()

    pprint(f"  Training {len(train_records)} train / {len(val_records)} val  |  device: {device}", "")
    print(f"  {'Epoch':<8} {'TrainLoss':<12} {'TrainAcc':<12} {'ValLoss':<12} {'ValAcc':<12} {'LR'}")
    print(f"  {SEP2}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            run_loss    += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total   += y.size(0)

        train_loss = run_loss / max(run_total, 1)
        train_acc  = run_correct / max(run_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        # ── Print epoch row ──
        improved = val_acc > result.best_val_acc
        marker = " ✓ BEST" if improved else ""
        color  = "green" if improved else ""
        pprint(
            f"  {epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {lr:.2e}{marker}",
            color
        )

        if improved:
            result.best_val_acc = val_acc
            result.best_epoch   = epoch
            # save checkpoint
            ckpt_path = out_dir / f"{config_name}_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "image_size": args.image_size,
                "config_name": config_name,
                "val_acc": val_acc,
            }, ckpt_path)

        result.epochs.append(EpochResult(epoch, train_loss, train_acc, val_loss, val_acc, lr))
        scheduler.step()

    result.total_time_sec = time.time() - t0
    pprint(f"\n  DONE – best val_acc = {result.best_val_acc:.4f} (epoch {result.best_epoch})"
           f"  |  time: {result.total_time_sec:.0f}s", "green")

    return result

# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(results: list[ExperimentResult]) -> None:
    pprint(f"\n\n{'#'*70}", "bold")
    pprint("  FINAL COMPARISON – All Experiments", "bold")
    pprint(f"{'#'*70}\n", "bold")
    print(f"  {'Config':<30} {'Description':<42} {'BestValAcc':>10} {'BestEpoch':>10}")
    print(f"  {'-'*95}")
    best_acc = max(r.best_val_acc for r in results)
    for r in sorted(results, key=lambda x: -x.best_val_acc):
        marker = " ← WINNER" if r.best_val_acc == best_acc else ""
        color  = "green" if r.best_val_acc == best_acc else ""
        pprint(
            f"  {r.config_name:<30} {r.description:<42} {r.best_val_acc:>10.4f} {r.best_epoch:>10}{marker}",
            color
        )

# ── Save results to JSON ──────────────────────────────────────────────────────

def save_results(results: list[ExperimentResult], out_dir: Path) -> None:
    summary = []
    for r in results:
        summary.append({
            "config_name":    r.config_name,
            "description":    r.description,
            "best_val_acc":   r.best_val_acc,
            "best_epoch":     r.best_epoch,
            "total_time_sec": r.total_time_sec,
            "epochs": [
                {
                    "epoch":      e.epoch,
                    "train_loss": e.train_loss,
                    "train_acc":  e.train_acc,
                    "val_loss":   e.val_loss,
                    "val_acc":    e.val_acc,
                    "lr":         e.lr,
                }
                for e in r.epochs
            ],
        })

    out_path = out_dir / "all_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pprint(f"\n  Results saved to: {out_path}", "cyan")

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmentation ablation study – NN2526 Groep 31")
    p.add_argument("--data-dir",     type=str,   default=".",     help="Dir with train_labels.csv and train_set/")
    p.add_argument("--output-dir",   type=str,   default="augmentation_results")
    p.add_argument("--num-classes",  type=int,   default=80)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio",    type=float, default=0.1)
    p.add_argument("--image-size",   type=int,   default=224)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--pretrained",   action="store_true",
                   help="Use ImageNet pretrained weights (strongly recommended)")
    p.add_argument("--only",         type=str,   default=None,
                   help="Run only one config by name, e.g. --only C_geometric_plus_color")
    return p.parse_args()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pprint(f"\n{SEP}", "bold")
    pprint("  NN2526 Groep 31 – Augmentation Ablation Study", "bold")
    pprint(SEP, "bold")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
          f"  pretrained={args.pretrained}  seed={args.seed}")

    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = data_dir / "train_labels.csv"
    train_root = data_dir / "train_set"
    image_dir  = find_image_dir(train_root)
    all_records = read_labels_csv(labels_csv)

    pprint(f"\n  Found {len(all_records)} labelled images in {image_dir}", "cyan")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_augmentation_configs(args.image_size)

    # Optionally run only one config
    if args.only:
        if args.only not in configs:
            raise ValueError(f"Unknown config '{args.only}'. Choose from: {list(configs.keys())}")
        configs = {args.only: configs[args.only]}

    results: list[ExperimentResult] = []
    for name, cfg in configs.items():
        result = run_experiment(name, cfg, all_records, image_dir, args, device, out_dir)
        results.append(result)

    if len(results) > 1:
        print_summary(results)

    save_results(results, out_dir)

    pprint("\n  Checkpoints (.pt) and results (all_results.json) saved to:", "cyan")
    pprint(f"  {out_dir.resolve()}\n", "bold")


if __name__ == "__main__":
    main()