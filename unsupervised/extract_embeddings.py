"""
Extract and save fixed feature embeddings from a DINO pretrained backbone.

Outputs one .npz file per split containing:
    embeddings  — float32 array, shape [N, embed_dim]
                  (2048 for ResNet-50, 384 for ViT-S/16)
    labels      — int32 array,   shape [N]  (Food-101 class index, 0–100)
    indices     — int32 array,   shape [N]  (original dataset index)
    cluster_ids — int32 array,   shape [N]  (k-means cluster, only if --kmeans > 0)

The teacher backbone is used by default because its EMA-smoothed weights
produce more stable representations than the still-training student.

Usage:
    python extract_embeddings.py \\
        --checkpoint checkpoints/resnet50/epoch_0030.pth \\
        --config configs/resnet50.yaml

    # Both splits + 101-way k-means cluster assignments
    python extract_embeddings.py \\
        --checkpoint checkpoints/resnet50/epoch_0030.pth \\
        --config configs/resnet50.yaml \\
        --splits train val \\
        --kmeans 101 \\
        --output-dir embeddings/

Loading the output (no PyTorch required):
    import numpy as np
    data = np.load("embeddings/resnet50_train_embeddings.npz")
    X = data["embeddings"]   # [75750, 2048]
    y = data["labels"]       # [75750]
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbones import get_backbone


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Food101LabeledDataset(Dataset):
    """HuggingFace Food-101 split yielding (PIL image, class index)."""

    def __init__(self, split, transform=None):
        from datasets import load_dataset
        hf_split = "validation" if split == "val" else split
        self.dataset = load_dataset("ethz/food101", split=hf_split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_eval_transform():
    """Standard ImageNet normalisation with no augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_teacher_backbone(checkpoint_path, backbone_name):
    """
    Load the teacher backbone from a DINO checkpoint.

    The teacher is preferred over the student because its EMA-smoothed weights
    are more stable and produce higher-quality representations at inference time.

    Falls back to ImageNet-pretrained weights when checkpoint_path is None.

    Args:
        checkpoint_path: Path to the .pth checkpoint, or None.
        backbone_name:   Config backbone string (e.g. "resnet50").

    Returns:
        (backbone model in eval mode, embed_dim)
    """
    backbone, embed_dim = get_backbone(backbone_name, pretrained=True)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "teacher_backbone" in ckpt:
            backbone.load_state_dict(ckpt["teacher_backbone"])
            epoch = ckpt.get("epoch", "?")
            print(f"Loaded teacher backbone from {checkpoint_path}  (epoch {epoch})")
        else:
            print("Warning: 'teacher_backbone' key not found — using ImageNet weights.")

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone, embed_dim


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract(backbone, dataset, device, batch_size, num_workers):
    """
    Run every image in dataset through backbone and collect embeddings.

    Args:
        backbone:    Frozen backbone in eval mode.
        dataset:     Dataset yielding (image_tensor, label).
        device:      Torch device.
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        embeddings — float32 ndarray, shape [N, D]
        labels     — int32   ndarray, shape [N]
        indices    — int32   ndarray, shape [N]  (0-based dataset positions)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_embeddings, all_labels = [], []
    offset = 0

    backbone = backbone.to(device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            feats = backbone(images.to(device))
            all_embeddings.append(feats.cpu().float().numpy())
            all_labels.append(labels.numpy())

            n = images.size(0)
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
                processed = offset + n
                print(f"  {processed}/{len(dataset)} images processed")
            offset += n

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    indices = np.arange(len(dataset), dtype=np.int32)

    return embeddings, labels, indices


# ---------------------------------------------------------------------------
# Optional k-means clustering
# ---------------------------------------------------------------------------

def run_kmeans(embeddings, k):
    """
    Fit k-means on L2-normalised embeddings and return cluster assignments.

    Normalisation ensures cosine-like distance behaviour, which is appropriate
    for DINO features (trained with L2-normalised bottleneck representations).

    Args:
        embeddings: float32 array, shape [N, D].
        k:          Number of clusters.

    Returns:
        cluster_ids — int32 array, shape [N].
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    print(f"  Fitting k-means (k={k}) on {len(embeddings)} L2-normalised embeddings...")
    normed = normalize(embeddings, norm="l2")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=4096)
    cluster_ids = km.fit_predict(normed).astype(np.int32)
    print(f"  k-means inertia: {km.inertia_:.2f}")
    return cluster_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fixed DINO feature embeddings for downstream use"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to DINO .pth checkpoint file",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config used for pretraining",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to extract (default: train val)",
    )
    parser.add_argument(
        "--output-dir", default="embeddings",
        help="Directory to write .npz files (default: embeddings/)",
    )
    parser.add_argument(
        "--kmeans", type=int, default=0, metavar="K",
        help="If K > 0, fit k-means with K clusters and save cluster_ids "
             "(default: 0 = disabled)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Inference batch size (default: 256)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backbone_name = cfg["backbone"]
    embed_dim = cfg["embed_dim"]
    num_workers = cfg.get("num_workers", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:   {device}")
    print(f"Backbone: {backbone_name}  (embed_dim={embed_dim})")
    print(f"Splits:   {args.splits}")

    backbone, _ = load_teacher_backbone(args.checkpoint, backbone_name)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n{'='*56}")
        print(f"  Extracting embeddings — {split} split")
        print(f"{'='*56}")

        dataset = Food101LabeledDataset(split=split, transform=get_eval_transform())
        print(f"  Dataset size: {len(dataset)} images")

        embeddings, labels, indices = extract(
            backbone, dataset, device, args.batch_size, num_workers
        )
        print(f"  Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")

        save_dict = {
            "embeddings": embeddings,
            "labels": labels,
            "indices": indices,
        }

        if args.kmeans > 0:
            cluster_ids = run_kmeans(embeddings, args.kmeans)
            save_dict["cluster_ids"] = cluster_ids

        out_path = os.path.join(args.output_dir, f"{backbone_name}_{split}_embeddings.npz")
        np.savez_compressed(out_path, **save_dict)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")

    print("\nDone.")
    print("\nTo load embeddings in downstream models:")
    print("    import numpy as np")
    for split in args.splits:
        varname = split
        print(f"    {varname} = np.load('{args.output_dir}/{backbone_name}_{split}_embeddings.npz')")
    print(f"    X_train = train['embeddings']  # float32, shape [N, {embed_dim}]")
    print(f"    y_train = train['labels']       # int32,   shape [N]")


if __name__ == "__main__":
    main()
