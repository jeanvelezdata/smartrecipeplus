"""
Evaluation script for DINO pretrained encoders.

Evaluates encoder quality via:
    - linear_probe:   Freeze backbone, train linear head on Food-101, report top-1/top-5 accuracy.
    - tsne:           Extract features from 5,000 test images, run t-SNE, save coloured scatter plot.
    - attention_map:  (ViT only) Extract CLS attention maps from last block, save heatmap grid.
    - all:            Run all applicable evaluations.

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.pth --config configs/resnet50.yaml --eval-type linear_probe
    python evaluate.py --checkpoint path/to/checkpoint.pth --config configs/resnet50.yaml --eval-type tsne
    python evaluate.py --checkpoint path/to/checkpoint.pth --config configs/resnet50.yaml --eval-type attention_map
    python evaluate.py --checkpoint path/to/checkpoint.pth --config configs/resnet50.yaml --eval-type all
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbones import get_backbone


# Data helpers

class Food101LabeledDataset(Dataset):
    """Wraps HuggingFace Food-101 and yields (PIL image, class index)."""

    def __init__(self, split="train", transform=None):
        from datasets import load_dataset
        hf_split = "validation" if split == "validation" else split
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
    """Standard ImageNet normalisation for evaluation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# Backbone loading

def load_backbone_from_checkpoint(checkpoint_path, backbone_name, pretrained_fallback=True):
    """
    Load student backbone weights from a DINO checkpoint.

    Falls back to ImageNet pretrained weights if checkpoint_path is None.

    Args:
        checkpoint_path:    Path to the .pth checkpoint file.
        backbone_name:      Backbone string (e.g. "resnet50").
        pretrained_fallback: If True, initialise with ImageNet weights before
                             loading checkpoint (safe to keep True; checkpoint
                             will override anyway).

    Returns:
        (backbone model, embed_dim)
    """
    backbone, embed_dim = get_backbone(backbone_name, pretrained=pretrained_fallback)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "student_backbone" in ckpt:
            backbone.load_state_dict(ckpt["student_backbone"])
            print(f"Loaded student backbone weights from {checkpoint_path}")
        else:
            print("Warning: 'student_backbone' key not found in checkpoint — using ImageNet weights.")
    return backbone, embed_dim


# Linear Probe

def _run_linear_probe_on_backbone(backbone, embed_dim, device, num_epochs=50,
                                   batch_size=256, num_workers=2):
    """
    Freeze backbone and train a single linear head on Food-101 train split,
    then evaluate on the validation split.

    Args:
        backbone:    Pretrained backbone (will be frozen).
        embed_dim:   Output feature dimension of the backbone.
        device:      Torch device.
        num_epochs:  Number of epochs to train the linear head.
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        (top1_acc, top5_acc) as percentages (0–100 floats).
    """
    tfm = get_eval_transform()
    train_ds = Food101LabeledDataset(split="train", transform=tfm)
    val_ds = Food101LabeledDataset(split="validation", transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Freeze backbone entirely
    backbone = backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Linear classification head
    head = nn.Linear(embed_dim, 101).to(device)

    optimizer = torch.optim.SGD(head.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        head.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = backbone(images)
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"    Epoch {epoch+1:3d}/{num_epochs}  loss: {avg_loss:.4f}"
                  f"  lr: {scheduler.get_last_lr()[0]:.6f}")

    # Evaluation
    head.eval()
    top1_correct = top5_correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            logits = head(features)

            _, top1_pred = logits.max(dim=1)
            top1_correct += (top1_pred == labels).sum().item()

            _, top5_pred = logits.topk(5, dim=1)
            top5_correct += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    return top1, top5


def run_linear_probe(checkpoint_path, cfg, device):
    """
    Run linear probe comparison: food-DINO pretrained vs. ImageNet-only baseline.

    Prints a formatted comparison table.

    Args:
        checkpoint_path: Path to the DINO checkpoint file.
        cfg:             Config dict loaded from the YAML file.
        device:          Torch device.
    """
    backbone_name = cfg["backbone"]
    embed_dim = cfg["embed_dim"]
    num_workers = cfg.get("num_workers", 2)

    print("\n" + "=" * 56)
    print("  Linear Probe Evaluation")
    print("=" * 56)

    print("\n[1/2] Training linear probe — Food-DINO pretrained encoder...")
    backbone_dino, _ = load_backbone_from_checkpoint(checkpoint_path, backbone_name)
    top1_dino, top5_dino = _run_linear_probe_on_backbone(
        backbone_dino, embed_dim, device, num_workers=num_workers
    )
    print(f"      → Top-1: {top1_dino:.1f}%   Top-5: {top5_dino:.1f}%")

    print("\n[2/2] Training linear probe — vanilla ImageNet encoder (baseline)...")
    backbone_base, _ = get_backbone(backbone_name, pretrained=True)
    top1_base, top5_base = _run_linear_probe_on_backbone(
        backbone_base, embed_dim, device, num_workers=num_workers
    )
    print(f"      → Top-1: {top1_base:.1f}%   Top-5: {top5_base:.1f}%")

    # Comparison table
    sep = "-" * 54
    print(f"\n{sep}")
    print(f"| {'Encoder':<22} | {'Top-1 Acc':>9} | {'Top-5 Acc':>9} |")
    print(f"|{'-'*24}|{'-'*11}|{'-'*11}|")
    print(f"| {'ImageNet baseline':<22} | {top1_base:>7.1f}%  | {top5_base:>7.1f}%  |")
    print(f"| {'Food-DINO pretrained':<22} | {top1_dino:>7.1f}%  | {top5_dino:>7.1f}%  |")
    print(f"{sep}\n")


# t-SNE / UMAP Visualisation

def run_tsne(checkpoint_path, cfg, device, n_samples=5000):
    """
    Extract features from n_samples Food-101 validation images, run t-SNE
    (perplexity=30), colour points by class label, and save the plot.

    Output: tsne_visualization.png

    Args:
        checkpoint_path: Path to the DINO checkpoint file.
        cfg:             Config dict.
        device:          Torch device.
        n_samples:       Number of images to embed (default 5,000).
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("\n" + "=" * 56)
    print("  t-SNE Visualisation")
    print("=" * 56)

    backbone_name = cfg["backbone"]
    num_workers = cfg.get("num_workers", 2)

    backbone, _ = load_backbone_from_checkpoint(checkpoint_path, backbone_name)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    tfm = get_eval_transform()
    val_ds = Food101LabeledDataset(split="validation", transform=tfm)

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(val_ds), min(n_samples, len(val_ds)), replace=False)
    subset = torch.utils.data.Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_features, all_labels = [], []
    print(f"Extracting features from {len(subset)} images...")

    with torch.no_grad():
        for images, labels in loader:
            feats = backbone(images.to(device))
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Running t-SNE (perplexity=30) on {len(features)} "
          f"× {features.shape[1]}-dim feature vectors...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(14, 12))
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap="tab20", s=4, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Food-101 class index")
    ax.set_title(
        f"t-SNE of {backbone_name} features on Food-101 ({len(features)} images)",
        fontsize=13,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.axis("off")
    plt.tight_layout()
    out_path = "tsne_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}\n")
    plt.close()


# Attention Map Visualisation (ViT only)

def run_attention_maps(checkpoint_path, cfg, device, n_images=8):
    """
    Extract the CLS-token self-attention from the last ViT block and overlay
    an averaged attention heatmap on each input image.

    Saves a 2-column grid (original | attention overlay) as attention_maps.png.
    Only runs for ViT-based backbones; skips silently for ResNet.

    For a 224×224 input with patch size 16 the spatial grid is 14×14 = 196 patches.
    Attention shape at the last block: [B, num_heads, N, N] where N = 197 (CLS + 196 patches).
    Row 0 of the attention matrix (CLS token) is averaged across heads and reshaped to 14×14.

    Args:
        checkpoint_path: Path to the DINO checkpoint file.
        cfg:             Config dict.
        device:          Torch device.
        n_images:        Number of example images to visualise.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image as PILImage

    backbone_name = cfg["backbone"]
    if "vit" not in backbone_name.lower():
        print(f"\nAttention map visualisation requires a ViT backbone "
              f"(got '{backbone_name}'). Skipping.\n")
        return

    print("\n" + "=" * 56)
    print("  Attention Map Visualisation (ViT)")
    print("=" * 56)

    backbone, _ = load_backbone_from_checkpoint(checkpoint_path, backbone_name)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    attention_store = {}

    def _capture_attn(module, inp, _out):
        attention_store["attn"] = inp[0].detach() 

    hook = backbone.blocks[-1].attn.attn_drop.register_forward_hook(_capture_attn)

    # Transforms
    tfm_tensor = get_eval_transform()
    tfm_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    val_ds_raw = Food101LabeledDataset(split="validation", transform=None)
    rng = np.random.default_rng(seed=7)
    indices = rng.choice(len(val_ds_raw), n_images, replace=False)

    fig, axes = plt.subplots(n_images, 2, figsize=(6, n_images * 3))

    for row, idx in enumerate(indices):
        pil_img, _label = val_ds_raw[int(idx)]
        pil_cropped = tfm_crop(pil_img)           
        tensor_img = tfm_tensor(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = backbone(tensor_img)

        attn = attention_store["attn"]        
        # CLS token attending to all patch tokens
        attn_cls = attn[0, :, 0, 1:]          
        attn_map = attn_cls.mean(0)            
        attn_map = attn_map.reshape(14, 14).cpu().numpy()

        # Normalise to [0, 1]
        lo, hi = attn_map.min(), attn_map.max()
        attn_map = (attn_map - lo) / (hi - lo + 1e-8)

        # Upsample to 224×224
        attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (224, 224), PILImage.BILINEAR
        )
        attn_upsampled = np.array(attn_pil) / 255.0        

        # Heatmap overlay
        heatmap_rgb = cm.jet(attn_upsampled)[..., :3]       
        img_np = np.array(pil_cropped).astype(float) / 255.0
        overlay = np.clip(0.5 * img_np + 0.5 * heatmap_rgb, 0.0, 1.0)

        axes[row, 0].imshow(pil_cropped)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)

        axes[row, 1].imshow(overlay)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Attention overlay", fontsize=9)

    hook.remove()
    plt.suptitle(f"DINO Self-Attention Maps — {backbone_name}", fontsize=12, y=1.002)
    plt.tight_layout()
    out_path = "attention_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}\n")
    plt.close()

# Entry point

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a DINO pretrained encoder (linear probe, t-SNE, attention maps)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to DINO .pth checkpoint file",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config file used for pretraining",
    )
    parser.add_argument(
        "--eval-type",
        choices=["linear_probe", "tsne", "attention_map", "all"],
        default="all",
        help="Which evaluation(s) to run (default: all)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {cfg['backbone']}  |  eval-type: {args.eval_type}")

    if args.eval_type in ("linear_probe", "all"):
        run_linear_probe(args.checkpoint, cfg, device)

    if args.eval_type in ("tsne", "all"):
        run_tsne(args.checkpoint, cfg, device)

    if args.eval_type in ("attention_map", "all"):
        run_attention_maps(args.checkpoint, cfg, device)


if __name__ == "__main__":
    main()
