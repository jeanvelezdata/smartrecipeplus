"""
Export clean backbone weights from a DINO checkpoint.

Strips the projection head, teacher, optimizer state, and scaler, leaving
only the student backbone state dict.  The exported file loads directly into
a vanilla torchvision / timm model via load_state_dict(..., strict=False).

Usage:
    python export_weights.py \\
        --checkpoint path/to/checkpoint.pth \\
        --config configs/resnet50.yaml \\
        --output resnet50_food_dino.pth

Loading the exported weights (3 lines):
    # ResNet-50
    import torchvision, torch
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load("resnet50_food_dino.pth"), strict=False)

    # ViT-S/16
    import timm, torch
    model = timm.create_model("vit_small_patch16_224")
    model.load_state_dict(torch.load("vit_small16_food_dino.pth"), strict=False)
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export student backbone weights from a DINO checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the full DINO .pth checkpoint",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config file used for pretraining",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the exported backbone weights .pth file",
    )
    return parser.parse_args()


def count_parameters(state_dict):
    """Return the total number of parameters in a state dict."""
    return sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))


def main():
    """Load a DINO checkpoint and export only the student backbone weights."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    backbone_name = cfg["backbone"]

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Validate expected keys
    if "student_backbone" not in ckpt:
        available = list(ckpt.keys())
        raise KeyError(
            f"'student_backbone' not found in checkpoint. "
            f"Available keys: {available}"
        )

    backbone_state_dict = ckpt["student_backbone"]

    # Report
    num_params = count_parameters(backbone_state_dict)
    num_keys = len(backbone_state_dict)
    print(f"Backbone:           {backbone_name}")
    print(f"Number of keys:     {num_keys}")
    print(f"Number of params:   {num_params:,}")

    # Keys stripped from the checkpoint
    stripped_keys = [k for k in ckpt if k != "student_backbone"]
    print(f"Stripped keys:      {stripped_keys}")

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(backbone_state_dict, args.output)
    print(f"\nExported backbone weights saved to: {args.output}")

    # Print usage snippet
    if "resnet" in backbone_name.lower():
        print("\n--- Load with torchvision ---")
        print("import torchvision, torch")
        print(f"model = torchvision.models.resnet50()")
        print(f'model.load_state_dict(torch.load("{args.output}"), strict=False)')
    else:
        print("\n--- Load with timm ---")
        print("import timm, torch")
        print(f'model = timm.create_model("{backbone_name}")')
        print(f'model.load_state_dict(torch.load("{args.output}"), strict=False)')


if __name__ == "__main__":
    main()
