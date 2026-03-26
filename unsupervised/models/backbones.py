"""
Backbone wrappers for DINO pretraining.

Supported backbones:
    - resnet50: torchvision ResNet-50, output dim 2048
    - vit_small_patch16_224: timm ViT-Small/16, output dim 384
"""

import torch
import torch.nn as nn


def get_resnet50(pretrained=True):
    """
    Load a ResNet-50 backbone with the final FC layer removed.

    The model outputs a 2048-dim feature vector (after global average pooling).

    Args:
        pretrained: If True, initialise with ImageNet weights.

    Returns:
        (model, embed_dim) where embed_dim == 2048.
    """
    import torchvision.models as tvm
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    base = tvm.resnet50(weights=weights)

    base.fc = nn.Identity()
    return base, 2048


def get_vit_small(pretrained=True):
    """
    Load a ViT-Small/16 backbone with the classification head removed.

    The model outputs the CLS token embedding of dim 384.

    Args:
        pretrained: If True, initialise with ImageNet weights.

    Returns:
        (model, embed_dim) where embed_dim == 384.
    """
    import timm

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=0,
        dynamic_img_size=True,   # allows local crops (96×96) via positional embedding interpolation
    )
    return model, 384


def get_backbone(name, pretrained=True):
    """
    Dispatcher that maps a config backbone string to the correct loader.

    Args:
        name: One of "resnet50" or "vit_small_patch16_224".
        pretrained: Whether to load pretrained ImageNet weights.

    Returns:
        (model, embed_dim) tuple.

    Raises:
        ValueError: If name is not recognised.
    """
    _registry = {
        "resnet50": get_resnet50,
        "vit_small_patch16_224": get_vit_small,
    }
    if name not in _registry:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from: {list(_registry.keys())}"
        )
    return _registry[name](pretrained=pretrained)
