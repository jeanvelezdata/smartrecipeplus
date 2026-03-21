"""
Checkpoint utilities for DINO pretraining.

Handles saving, loading, and optional Google Drive persistence (Colab).
"""

import os
import shutil
import torch

_COLAB_DRIVE_ROOT = "/content/drive"


def _is_drive_mounted():
    """Return True if Google Drive is mounted at the expected Colab path."""
    return os.path.isdir(_COLAB_DRIVE_ROOT)


def save_checkpoint(state_dict, epoch, config, path):
    """
    Save a DINO training checkpoint.

    Args:
        state_dict: Dict containing at minimum:
            - "student": student model state dict
            - "teacher": teacher model state dict
            - "optimizer": optimizer state dict
            - "scaler": GradScaler state dict
            - "center": center buffer tensor
            - "loss": current loss value
        epoch: Current epoch number (int).
        config: Config dict (will be saved alongside weights for reproducibility).
        path: Destination file path (str), e.g. "./checkpoints/resnet50/epoch_010.pth".
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {**state_dict, "epoch": epoch, "config": config}
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    """
    Load a DINO checkpoint from disk.

    Args:
        path: Path to the .pth checkpoint file.

    Returns:
        The checkpoint dict as saved by save_checkpoint.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    print(f"Checkpoint loaded: {path}  (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def save_to_drive(local_path, drive_path):
    """
    Copy a checkpoint file to Google Drive (Colab).

    Args:
        local_path: Source file path on the local filesystem.
        drive_path: Destination path under the Drive mount, e.g.
                    "/content/drive/MyDrive/dino_checkpoints/epoch_010.pth".
    """
    if not _is_drive_mounted():
        print(
            f"Warning: Google Drive does not appear to be mounted at "
            f"'{_COLAB_DRIVE_ROOT}'. Skipping Drive save."
        )
        return

    os.makedirs(os.path.dirname(os.path.abspath(drive_path)), exist_ok=True)
    shutil.copy2(local_path, drive_path)
    print(f"Checkpoint copied to Drive: {drive_path}")
