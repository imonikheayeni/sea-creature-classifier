"""Data loading, augmentation, and stratified train/val/test splitting.

The dataset is expected in torchvision ImageFolder layout:

    data/sea_animals/
        Dolphin/*.jpg
        Jelly Fish/*.jpg
        Sharks/*.jpg
        ...

Each subfolder name becomes a class label.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ImageNet statistics, used because we fine-tune ImageNet-pretrained backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True):
    """Return a torchvision transform pipeline.

    Training uses light augmentation; validation/test uses only resize + normalize.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformSubset(Dataset):
    """Wrap a subset of an ImageFolder so each split gets its own transform.

    ImageFolder returns PIL images when no transform is set; we apply the
    per-split transform here so training augmentation never leaks into
    validation or test.
    """

    def __init__(self, base: datasets.ImageFolder, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]  # image is a PIL image
        return self.transform(image), label


def stratified_split(targets, val_size: float, test_size: float, seed: int):
    """Split indices into train/val/test while preserving class proportions."""
    idx = np.arange(len(targets))
    holdout = val_size + test_size
    train_idx, temp_idx = train_test_split(
        idx, test_size=holdout, stratify=targets, random_state=seed
    )
    temp_targets = [targets[i] for i in temp_idx]
    rel_test = test_size / holdout
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, stratify=temp_targets, random_state=seed
    )
    return train_idx, val_idx, test_idx


def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
):
    """Build train/val/test DataLoaders plus metadata.

    Returns a dict with loaders, class names, and per-class training counts
    (useful for class-weighted loss on an imbalanced dataset).
    """
    base = datasets.ImageFolder(data_dir)  # no transform yet, returns PIL
    targets = base.targets
    classes = base.classes

    train_idx, val_idx, test_idx = stratified_split(targets, val_size, test_size, seed)

    train_ds = TransformSubset(base, train_idx, build_transforms(img_size, train=True))
    val_ds = TransformSubset(base, val_idx, build_transforms(img_size, train=False))
    test_ds = TransformSubset(base, test_idx, build_transforms(img_size, train=False))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Per-class counts over the training split, ordered by class index.
    train_targets = [targets[i] for i in train_idx]
    counts = np.bincount(train_targets, minlength=len(classes))

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_counts": counts,
    }
