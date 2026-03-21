"""
Download and prepare the Food-101 dataset for DINO pretraining.

Usage:
    python data/prepare_food101.py --verify
"""

import argparse
from torch.utils.data import Dataset, ConcatDataset


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
    Return a PyTorch Dataset yielding raw PIL images (no labels) from Food-101.

    Passing split="all" combines train and test splits (~101K images total).

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
