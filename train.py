"""Train a sea-creature image classifier.

Example:
    python -m src.train --data-dir data/sea_animals --epochs 15 --arch resnet18

Saves the best-by-validation checkpoint to outputs/best_model.pt and a
training-curves figure to outputs/training_curves.png.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import build_dataloaders
from .model import build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch. If optimizer is given, train; otherwise evaluate."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += images.size(0)

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser(description="Train a sea-creature classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to ImageFolder root.")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device}")

    data = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    classes = data["classes"]
    print(f"Found {len(classes)} classes: {classes}")

    model = build_model(
        num_classes=len(classes),
        arch=args.arch,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # Class-weighted loss counters class imbalance (rarer creatures matter more).
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        counts = data["class_counts"].astype(np.float64)
        weights = counts.sum() / (len(counts) * np.clip(counts, 1, None))
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, data["train_loader"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, data["val_loader"], criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train loss {tr_loss:.3f} acc {tr_acc:.3f} "
            f"| val loss {va_loss:.3f} acc {va_acc:.3f} "
            f"| {time.time() - start:.0f}s"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": classes, "arch": args.arch,
                 "img_size": args.img_size},
                ckpt_path,
            )
            print(f"  saved new best (val acc {va_acc:.3f}) to {ckpt_path}")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, os.path.join(args.out_dir, "training_curves.png"))
    print(f"Best validation accuracy: {best_val_acc:.3f}")


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
