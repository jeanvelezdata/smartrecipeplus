"""
DINO multi-crop augmentation pipeline.

Produces a list of tensors [global_crop_1, global_crop_2, local_crop_1, ..., local_crop_N]
from a single PIL image, plus a collate function that batches each crop position independently.
"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageOps


# ImageNet normalization constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    """Apply Gaussian blur with random radius drawn from [radius_min, radius_max]."""

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class Solarization:
    """Apply PIL solarization with probability p."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


def _color_jitter():
    return T.RandomApply(
        [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    )


def _build_global_transform(crop_size, blur_p, solarize_p):
    """Build the transform pipeline for one global crop."""
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        _color_jitter(),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=blur_p),
        Solarization(p=solarize_p),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _build_local_transform(crop_size):
    """Build the transform pipeline for one local crop."""
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.05, 0.4), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        _color_jitter(),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class DINOMultiCropTransform:
    """
    DINO multi-crop augmentation.

    Takes a PIL image and returns a list of tensors:
        [global_crop_1, global_crop_2, local_crop_1, ..., local_crop_N]

    Global crop 1: Gaussian blur always applied (p=1.0), no solarization.
    Global crop 2: Gaussian blur rarely applied (p=0.1), solarization (p=0.2).
    Local crops: Gaussian blur (p=0.5), no solarization.

    Args:
        global_crop_size: Spatial size for global crops (default 224).
        local_crop_size:  Spatial size for local crops (default 96).
        num_local_crops:  Number of local crops to generate (default 6; use 4 for low memory).
    """

    def __init__(self, global_crop_size=224, local_crop_size=96, num_local_crops=6):
        self.global_transforms = [
            _build_global_transform(global_crop_size, blur_p=1.0, solarize_p=0.0),
            _build_global_transform(global_crop_size, blur_p=0.1, solarize_p=0.2),
        ]
        self.local_transform = _build_local_transform(local_crop_size)
        self.num_local_crops = num_local_crops

    def __call__(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            List of tensors with length 2 + num_local_crops.
        """
        crops = [t(img) for t in self.global_transforms]
        crops += [self.local_transform(img) for _ in range(self.num_local_crops)]
        return crops


def dino_collate_fn(batch):
    """
    Custom collate function for DINO multi-crop batches.

    Args:
        batch: List of samples, where each sample is a list of tensors
               [global_1, global_2, local_1, ..., local_N].

    Returns:
        List of tensors, one per crop position, each shaped [B, 3, H, W].
    """
    num_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch]) for i in range(num_crops)]
