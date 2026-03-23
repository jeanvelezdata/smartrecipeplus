"""
Download and prepare the Food-101 dataset for DINO pretraining.

Two dataset backends are provided:

  HuggingFace (default, no local setup required):
      get_food101_images(split="all")

  Local directory (fast — use this when images are already on disk):
      get_local_images(data_dir="/content/food101_images")

      Expects data_dir to contain JPEG/PNG files in any nested structure,
      e.g. the standard Food-101 layout:
          <data_dir>/
              apple_pie/
                  *.jpg
              baby_back_ribs/
                  *.jpg
              ...
      or a flat directory of images.  Labels are ignored; only PIL images
      are returned.

Usage:
    python data/prepare_food101.py --verify
    python data/prepare_food101.py --verify --data-dir /content/food101_images
"""

import argparse
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class Food101ImageDataset(Dataset):
    """PyTorch Dataset that wraps a HuggingFace Food-101 split and yields only PIL images."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["image"].convert("RGB")


def get_food101_images(split="train"):
    """
    Return a PyTorch Dataset yielding raw PIL images (no labels) from Food-101
    via the HuggingFace datasets library.

    Passing split="all" combines train and validation splits (~101K images).

    Args:
        split: "train", "validation", or "all"

    Returns:
        A PyTorch Dataset of PIL images.
    """
    from datasets import load_dataset

    if split == "all":
        train_ds = load_dataset("ethz/food101", split="train")
        val_ds = load_dataset("ethz/food101", split="validation")
        return ConcatDataset([
            Food101ImageDataset(train_ds),
            Food101ImageDataset(val_ds),
        ])

    hf_ds = load_dataset("ethz/food101", split=split)
    return Food101ImageDataset(hf_ds)


# ---------------------------------------------------------------------------
# Local directory backend (fast path)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class LocalImageDataset(Dataset):
    """
    Glob-based Dataset that reads images from a local directory tree.

    Walks data_dir recursively and collects all files with a recognised
    image extension.  Labels are discarded — only PIL images are returned,
    making this a drop-in replacement for Food101ImageDataset.

    Args:
        data_dir: Root directory to search (str or Path).
    """

    def __init__(self, data_dir):
        self.paths = sorted(
            p for p in Path(data_dir).rglob("*")
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No image files found under '{data_dir}'. "
                f"Supported extensions: {_IMAGE_EXTENSIONS}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        return Image.open(self.paths[idx]).convert("RGB")


def get_local_images(data_dir):
    """
    Return a LocalImageDataset for a directory of food images already on disk.

    This is significantly faster than the HuggingFace backend because images
    are read directly from the filesystem with multiple DataLoader workers.

    Args:
        data_dir: Path to a directory containing JPEG/PNG images (nested
                  class subdirectories are fine; labels are ignored).

    Returns:
        A PyTorch Dataset of PIL images.
    """
    dataset = LocalImageDataset(data_dir)
    print(f"[LocalImageDataset] Found {len(dataset):,} images under '{data_dir}'")
    return dataset


def _verify():
    """Download Food-101 and display a grid of 16 sample images."""
    import math
    import random
    import matplotlib.pyplot as plt

    print("Downloading / loading Food-101 (train + validation)...")
    dataset = get_food101_images(split="all")
    print(f"Total images: {len(dataset)}")

    indices = random.sample(range(len(dataset)), 16)
    images = [dataset[i] for i in indices]

    cols = 4
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle("Food-101 sample images", y=1.01)
    plt.savefig("food101_samples.png", bbox_inches="tight")
    print("Saved food101_samples.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Food-101 dataset")
    parser.add_argument("--verify", action="store_true",
                        help="Download and display 16 sample images")
    args = parser.parse_args()

    if args.verify:
        _verify()
    else:
        parser.print_help()
