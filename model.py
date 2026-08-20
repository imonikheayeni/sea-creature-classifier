"""Model definitions.

We fine-tune an ImageNet-pretrained backbone and swap the final classifier
layer for one sized to our number of sea-creature classes. ResNet18 is a solid
default: small enough to train on a modest GPU, strong enough to score well.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = {"resnet18", "resnet34", "resnet50"}


def build_model(
    num_classes: int,
    arch: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Return a backbone with its head replaced for `num_classes`.

    freeze_backbone=True trains only the new head (fast, good on small data).
    freeze_backbone=False fine-tunes the whole network (usually higher accuracy).
    """
    if arch not in SUPPORTED:
        raise ValueError(f"arch must be one of {sorted(SUPPORTED)}, got {arch}")

    weight_map = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
    }
    ctor = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }[arch]

    weights = weight_map[arch].DEFAULT if pretrained else None
    model = ctor(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # new head, always trainable
    return model
