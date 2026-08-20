"""Predict the sea creature in a single image.

Example:
    python -m src.predict --ckpt outputs/best_model.pt --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .model import build_model
from .train import pick_device


def load_bundle(ckpt_path, device):
    """Load a checkpoint and return a ready-to-use model plus metadata."""
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    model = build_model(len(classes), arch=ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    transform = build_transforms(img_size, train=False)
    return model, classes, transform


@torch.no_grad()
def predict_image(model, classes, transform, image_path, device, topk=3):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = F.softmax(model(tensor), dim=1).squeeze(0)
    k = min(topk, len(classes))
    top_probs, top_idx = probs.topk(k)
    return [(classes[i], float(p)) for p, i in zip(top_probs, top_idx)]


def main():
    parser = argparse.ArgumentParser(description="Predict a single image.")
    parser.add_argument("--ckpt", default="outputs/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = pick_device()
    model, classes, transform = load_bundle(args.ckpt, device)
    results = predict_image(model, classes, transform, args.image, device, args.topk)

    print(f"\nPredictions for {args.image}:")
    for label, prob in results:
        print(f"  {label:<16} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
