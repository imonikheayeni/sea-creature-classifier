"""Download the Sea Animals Image Dataset from Kaggle.

Requires a (free) Kaggle account. kagglehub reads credentials from
~/.kaggle/kaggle.json or the KAGGLE_USERNAME / KAGGLE_KEY environment variables.
Get your token from https://www.kaggle.com/settings > "Create New Token".

Example:
    python -m src.download_data

By default this pulls all 23 classes. Pass --classes to keep only a subset
(the 9-class version matches a common course setup).
"""

from __future__ import annotations


import argparse
import os
import shutil

import kagglehub

DATASET = "vencerlanz09/sea-animals-image-dataste"


# The 9-class subset used by many intro courses.
COURSE_SUBSET = [
    "Dolphin", "Jelly Fish", "Octopus", "Puffers", "Sea Rays",
    "Sea Urchins", "Sharks", "Turtle_Tortoise", "Whale",
]



def main():
    parser = argparse.ArgumentParser(description="Download the sea animals dataset.")
    parser.add_argument("--dest", default="data/sea_animals", help="Where to copy images.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of class folder names to keep. Omit to keep all.",
    )
    parser.add_argument("--course-subset", action="store_true", help="Keep the 9 course classes.")
    args = parser.parse_args()

    print("Downloading via kagglehub (this can take a few minutes)...")
    cache_path = kagglehub.dataset_download(DATASET)
    print(f"Cached at: {cache_path}")

    # The archive contains a single folder holding all class subdirectories.
    source_root = _find_class_root(cache_path)
    keep = args.classes
    if args.course_subset:
        keep = COURSE_SUBSET

    os.makedirs(args.dest, exist_ok=True)
    copied = 0
    for name in sorted(os.listdir(source_root)):
        src = os.path.join(source_root, name)
        if not os.path.isdir(src):
            continue
        if keep is not None and name not in keep:
            continue
        dst = os.path.join(args.dest, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += 1
    print(f"Copied {copied} class folders into {args.dest}")


def _find_class_root(cache_path: str) -> str:
    """Locate the directory whose children are the class folders."""
    # Prefer a directory that directly contains many subfolders of images.
    best, best_count = cache_path, _count_subdirs(cache_path)
    for root, dirs, _ in os.walk(cache_path):
        count = sum(os.path.isdir(os.path.join(root, d)) for d in dirs)
        if count > best_count:
            best, best_count = root, count
    return best


def _count_subdirs(path: str) -> int:
    return sum(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))


if __name__ == "__main__":
    main()
