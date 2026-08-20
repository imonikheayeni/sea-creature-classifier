"""Evaluate a trained checkpoint on the held-out test split.

Example:
    python -m src.evaluate --data-dir data/sea_animals --ckpt outputs/best_model.pt

Prints a per-class precision/recall/F1 report and writes a confusion-matrix
figure to outputs/confusion_matrix.png.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from .data import build_dataloaders
from .model import build_model
from .train import pick_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a sea-creature classifier.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt", default="outputs/best_model.pt")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    # Rebuild the identical split (same seed) so the test set is truly held out.
    data = build_dataloaders(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if data["classes"] != classes:
        raise RuntimeError("Class list on disk does not match the checkpoint.")

    model = build_model(len(classes), arch=ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    y_true, y_pred = collect_predictions(model, data["test_loader"], device)

    print("\nClassification report (test split):\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
    ax.set_title("Confusion matrix (test split)")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
