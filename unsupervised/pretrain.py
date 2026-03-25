"""
DINO pretraining script.

Usage:
    python pretrain.py --config configs/resnet50.yaml
    python pretrain.py --config configs/resnet50.yaml --resume checkpoints/resnet50/epoch_010.pth
    python pretrain.py --config configs/resnet50.yaml --low-memory
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Allow imports from the unsupervised/ root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.prepare_food101 import get_food101_images, get_local_images
from models.dino import DINOLoss, DINOModel
from utils.augmentations import DINOMultiCropTransform, dino_collate_fn
from utils.checkpoint import load_checkpoint, save_checkpoint, save_to_drive


# Helpers 

def load_config(path):
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_schedule(start, end, step, max_steps):
    """Cosine interpolation from start → end over max_steps steps."""
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * step / max_steps))


def get_lr(base_lr, warmup_epochs, epoch, total_epochs, min_lr=1e-6):
    """
    Linear warmup then cosine decay.

    Returns the learning rate for the given epoch (0-indexed).
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def set_weight_decay(optimizer, wd):
    for group in optimizer.param_groups:
        if group.get("apply_wd", True):
            group["weight_decay"] = wd


def student_parameters(model):
    """Return only student backbone + head parameters (teacher excluded)."""
    return list(model.student_backbone.parameters()) + list(model.student_head.parameters())


# Training loop

def train(args):
    cfg = load_config(args.config)

    # Low-memory overrides
    if args.low_memory:
        cfg["batch_size"] = 32
        cfg["num_local_crops"] = 4
        grad_accum_steps = 2
        print("[low-memory] batch_size=32, num_local_crops=4, grad_accum=2")
    else:
        grad_accum_steps = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & DataLoader
    transform = DINOMultiCropTransform(
        global_crop_size=cfg["global_crop_size"],
        local_crop_size=cfg["local_crop_size"],
        num_local_crops=cfg["num_local_crops"],
    )
    if args.data_dir:
        dataset = get_local_images(args.data_dir)
    else:
        dataset = get_food101_images(split="all")

    class _WrappedDataset(torch.utils.data.Dataset):
        """Apply DINOMultiCropTransform to a PIL-image dataset."""
        def __init__(self, base, tfm):
            self.base = base
            self.tfm = tfm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            return self.tfm(self.base[idx])

    wrapped = _WrappedDataset(dataset, transform)
    loader = DataLoader(
        wrapped,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=True,             
        prefetch_factor=3, 
        drop_last=True,
        collate_fn=dino_collate_fn,
    )

    # Model 
    model = DINOModel(
        backbone_name=cfg["backbone"],
        pretrained=cfg.get("use_imagenet_init", True),
        proj_hidden_dim=cfg["proj_hidden_dim"],
        proj_bottleneck_dim=cfg["proj_bottleneck_dim"],
        proj_output_dim=cfg["proj_output_dim"],
    ).to(device)

    criterion = DINOLoss(
        teacher_temp=cfg["teacher_temp"],
        student_temp=cfg["student_temp"],
    )

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "apply_wd": True},
            {"params": no_decay_params, "weight_decay": 0.0, "apply_wd": False},
        ],
        lr=cfg["base_lr"],
        weight_decay=cfg["weight_decay"],
    )


    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.student_backbone.load_state_dict(ckpt["student_backbone"])
        model.student_head.load_state_dict(ckpt["student_head"])
        model.teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
        model.teacher_head.load_state_dict(ckpt["teacher_head"])
        model.center.copy_(ckpt["center"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Logging setup 
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.csv")
    log_file_existed = os.path.isfile(log_path)
    log_fh = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(
        log_fh, fieldnames=["epoch", "batch", "loss", "lr", "momentum", "timestamp"]
    )
    if not log_file_existed:
        log_writer.writeheader()

    # Training 
    total_epochs = cfg["epochs"]
    num_batches = len(loader)
    num_global = cfg["num_global_crops"]

    for epoch in range(start_epoch, total_epochs):
        model.train()

        # Per-epoch LR and weight decay
        lr = get_lr(cfg["base_lr"], cfg["warmup_epochs"], epoch, total_epochs)
        wd = cosine_schedule(
            cfg["weight_decay"], cfg["weight_decay_end"], epoch, total_epochs
        )
        set_lr(optimizer, lr)
        set_weight_decay(optimizer, wd)

        # Teacher momentum: cosine schedule 0.996 → 1.0
        teacher_momentum = cosine_schedule(
            cfg["teacher_momentum_start"],
            cfg["teacher_momentum_end"],
            epoch,
            total_epochs,
        )

        optimizer.zero_grad()
        running_loss = 0.0

        for batch_idx, crops in enumerate(loader):
            crops = [c.to(device, non_blocking=True) for c in crops]
            global_crops = crops[:num_global]
            is_accum_step = (batch_idx + 1) % grad_accum_steps != 0
            is_last_batch = (batch_idx + 1) == num_batches

            # Forward 
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                student_out = model.forward_student(crops)
                teacher_out = model.forward_teacher(global_crops)
                loss = criterion(student_out, teacher_out, model.center)
                # Scale loss for gradient accumulation
                loss_scaled = loss / grad_accum_steps

            # Backward
            loss_scaled.backward()

            if not is_accum_step or is_last_batch:
                nn.utils.clip_grad_norm_(student_parameters(model), max_norm=3.0)

                optimizer.step()
                optimizer.zero_grad()

                # Teacher EMA & center update
                model.update_teacher(teacher_momentum)
                model.update_center(teacher_out, cfg["center_momentum"])

            running_loss = 0.9 * running_loss + 0.1 * loss.item()

            # Stdout logging
            print(
                f"[Epoch {epoch+1}/{total_epochs}] "
                f"[Batch {batch_idx+1}/{num_batches}] "
                f"loss: {loss.item():.4f}  "
                f"lr: {lr:.6f}  "
                f"momentum: {teacher_momentum:.4f}"
            )

            # CSV logging
            log_writer.writerow({
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr:.8f}",
                "momentum": f"{teacher_momentum:.6f}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            log_fh.flush()

        # Checkpoint 
        if (epoch + 1) % cfg["checkpoint_freq"] == 0 or (epoch + 1) == total_epochs:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch+1:04d}.pth")
            state = {
                "student_backbone": model.student_backbone.state_dict(),
                "student_head": model.student_head.state_dict(),
                "teacher_backbone": model.teacher_backbone.state_dict(),
                "teacher_head": model.teacher_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "center": model.center,
                "loss": running_loss,
            }
            save_checkpoint(state, epoch, cfg, ckpt_path)

            drive_path = os.path.join(
                "/content/drive/MyDrive/dino_checkpoints",
                cfg["backbone"],
                os.path.basename(ckpt_path),
            )
            save_to_drive(ckpt_path, drive_path)

    log_fh.close()
    print("Training complete.")


# Entry point 

def parse_args():
    parser = argparse.ArgumentParser(description="DINO pretraining")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to a local directory of food images (fast path). "
             "If omitted, downloads Food-101 via HuggingFace datasets.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode: batch_size=32, num_local_crops=4, grad_accum=2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
