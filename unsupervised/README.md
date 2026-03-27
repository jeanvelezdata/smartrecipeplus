# DINO Unsupervised Pretraining — SmartRecipe+

## 1. Overview

This component produces domain-adapted visual encoders for the SmartRecipe+ pipeline (Georgia Tech CS 7643). Using DINO (Self-Distillation with No Labels), two backbone architectures — ResNet-50 and ViT-Small/16 — are pretrained on Food-101 (~101K images) without any labels. The resulting weights capture food-specific visual representations and are exported as clean `.pth` files and fixed-feature `.npz` embeddings for teammates to use as frozen or fine-tunable backbones in their supervised and semi-supervised ingredient prediction models.

---

## 2. Method

DINO trains a **student** network to match the output of a **teacher** network whose weights are an exponential moving average (EMA) of the student's. Each image is transformed into multiple crops (2 global + 6 local); the student processes all crops while the teacher sees only the global crops. The loss is cross-entropy between the teacher's softened output (centered and temperature-scaled) and the student's log-softmax output, summed over all cross-view pairs.

```
Input image
    │
    ├─── Global crop 1 ──► Teacher backbone ──► Teacher head ──► soft targets
    ├─── Global crop 2 ──► Teacher backbone ──► Teacher head ──► soft targets
    │                         ▲ EMA update ▲
    ├─── Global crop 1 ──► Student backbone ──► Student head ──► log-probs
    ├─── Global crop 2 ──► Student backbone ──► Student head ──► log-probs
    ├─── Local  crop 1 ──► Student backbone ──► Student head ──► log-probs
    └─── Local  crop N ──► Student backbone ──► Student head ──► log-probs

Loss = mean cross-entropy over all (teacher_crop, student_crop) pairs
       where student_crop ≠ teacher_crop  (no same-view pairs)

Center buffer ── exponential moving average of teacher outputs ── prevents collapse
```

Key stabilizers: output centering, temperature sharpening, gradient clipping (`max_norm=3.0`), and bfloat16 mixed precision.

---

## 3. Setup

### Requirements

```
Python >= 3.10
CUDA-capable GPU (6 GB VRAM minimum; T4 / RTX 4050 recommended)
```

### Install

```bash
pip install -r requirements.txt
```

### Google Colab Quickstart

```python
# Mount Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

# Clone repo and install
!git clone https://github.com/<your-org>/smartrecipeplus.git
%cd smartrecipeplus/unsupervised
!pip install -r requirements.txt

# Start pretraining (ResNet-50)
!python pretrain.py --config configs/resnet50.yaml
```

To resume after a session timeout:

```bash
python pretrain.py --config configs/resnet50.yaml \
    --resume /content/drive/MyDrive/dino_checkpoints/resnet50/epoch_0020.pth
```

For free-tier Colab (12 GB RAM, T4 GPU), add `--low-memory` to halve memory usage:

```bash
python pretrain.py --config configs/resnet50.yaml --low-memory
```

`--low-memory` sets `batch_size=32`, `num_local_crops=4`, and enables gradient accumulation (2 steps).

---

## 4. Data

Food-101 is downloaded automatically from HuggingFace on first run — no manual download needed.

```python
# Internally used by pretrain.py and evaluate.py
from data.prepare_food101 import get_food101_images
dataset = get_food101_images(split="train")   # ~75,750 images
dataset = get_food101_images(split="all")     # ~101,000 images (train + test)
```

To verify the download and inspect samples:

```bash
python data/prepare_food101.py --verify
```

This prints the dataset size and displays a 4×4 grid of sample images.

**Labels are discarded during pretraining.** They are only used in `evaluate.py` (linear probe, t-SNE coloring) and `extract_embeddings.py`.

---

## 5. Training

### ResNet-50

```bash
python pretrain.py --config configs/resnet50.yaml
```

Expected training time on Colab T4: **~5–8 hours for 30 epochs**.
Expected training time on RTX 4050 (6 GB): **~8–12 hours for 30 epochs** (use `--low-memory`).

### ViT-Small/16

```bash
python pretrain.py --config configs/vit_small16.yaml
```

Expected training time on Colab T4: **~6–10 hours for 30 epochs**.

### Resuming a Run

```bash
python pretrain.py --config configs/resnet50.yaml \
    --resume checkpoints/resnet50/epoch_0020.pth
```

All state is restored: model weights, optimizer, center buffer, and epoch counter.

### Key Hyperparameters (resnet50.yaml)

| Parameter | Value | Notes |
|---|---|---|
| `batch_size` | 128 | Reduce to 32 with `--low-memory` |
| `epochs` | 30 | |
| `warmup_epochs` | 10 | Linear LR warmup |
| `base_lr` | 0.0005 | Scaled by `batch_size / 256` |
| `teacher_momentum_start` | 0.996 | Cosine schedule to 1.0 |
| `teacher_temp` | 0.04 | Fixed throughout training |
| `student_temp` | 0.1 | |
| `num_local_crops` | 6 | Reduce to 4 if OOM |

### Training Log

Each run writes `training_log.csv` in the working directory:

```
epoch,batch,loss,lr,momentum,timestamp
1,0,10.4321,0.0000500,0.9960,2024-01-15T10:23:01
...
```

Loss typically drops from ~10–11 (random init) to ~7–8 by epoch 30 on Food-101. Stable training shows a smooth monotone decrease without sudden spikes.

---

## 6. Evaluation

### Linear Probe

Freezes the pretrained backbone and trains a single `Linear(embed_dim, 101)` head for 50 epochs (SGD, lr=0.01, cosine schedule). Compares pretrained vs. vanilla ImageNet baseline.

```bash
python evaluate.py \
    --checkpoint checkpoints/resnet50/epoch_0030.pth \
    --config configs/resnet50.yaml \
    --eval-type linear_probe
```

Output:
```
| Encoder              | Top-1 Acc | Top-5 Acc |
|----------------------|-----------|-----------|
| ImageNet baseline    | XX.X%     | XX.X%     |
| Food-DINO pretrained | XX.X%     | XX.X%     |
```

### t-SNE Visualization

Extracts features from 5,000 random validation images, runs t-SNE (perplexity=30), and saves a scatter plot colored by Food-101 class.

```bash
python evaluate.py \
    --checkpoint checkpoints/resnet50/epoch_0030.pth \
    --config configs/resnet50.yaml \
    --eval-type tsne
```

Output: `tsne_visualization.png` — semantically similar foods (pasta dishes, desserts, etc.) cluster together.

### Attention Maps (ViT only)

Extracts CLS-token self-attention from the last transformer block, reshaped to 14×14 spatial maps, overlaid on the original image.

```bash
python evaluate.py \
    --checkpoint checkpoints/vit_small16/epoch_0030.pth \
    --config configs/vit_small16.yaml \
    --eval-type attention_map
```

Output: `attention_maps.png` — 8-image grid showing the backbone attending to food regions.

### Run All Evaluations

```bash
python evaluate.py \
    --checkpoint checkpoints/resnet50/epoch_0030.pth \
    --config configs/resnet50.yaml \
    --eval-type all
```

---

## 7. Using the Pretrained Weights

### Export Clean Backbone Weights

Strip the projection heads and optimizer state, keeping only the backbone:

```bash
# ResNet-50
python export_weights.py \
    --checkpoint checkpoints/resnet50/epoch_0030.pth \
    --config configs/resnet50.yaml \
    --output resnet50_food_dino.pth

# ViT-Small/16
python export_weights.py \
    --checkpoint checkpoints/vit_small16/epoch_0030.pth \
    --config configs/vit_small16.yaml \
    --output vit_small16_food_dino.pth
```

### Load ResNet-50 Backbone

```python
import torch
import torchvision.models as models

backbone = models.resnet50()
backbone.fc = torch.nn.Identity()   # remove classification head
backbone.load_state_dict(torch.load("resnet50_food_dino.pth"), strict=False)
backbone.eval()
```

### Load ViT-Small/16 Backbone

```python
import torch
import timm

backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
backbone.load_state_dict(torch.load("vit_small16_food_dino.pth"), strict=False)
backbone.eval()
```

### Extract Fixed-Feature Embeddings

Use `extract_embeddings.py` to pre-compute embeddings for the entire dataset and save them as numpy arrays — **no PyTorch required at load time**.

```bash
python extract_embeddings.py \
    --checkpoint checkpoints/resnet50/epoch_0030.pth \
    --config configs/resnet50.yaml \
    --splits train val \
    --kmeans 101 \
    --output-dir embeddings/
```

This writes:
- `embeddings/resnet50_train_embeddings.npz` — 75,750 × 2048 float32
- `embeddings/resnet50_val_embeddings.npz`  — 25,250 × 2048 float32

Loading (no PyTorch required):

```python
import numpy as np

data = np.load("embeddings/resnet50_train_embeddings.npz")
X          = data["embeddings"]    # float32, shape [N, 2048]
y          = data["labels"]        # int32,   shape [N]  — Food-101 class 0–100
indices    = data["indices"]       # int32,   shape [N]  — original dataset index
cluster_ids = data["cluster_ids"]  # int32,   shape [N]  — k-means cluster (if --kmeans > 0)
```

---

## 8. Results Summary

*Results below are representative targets; actual numbers depend on training duration and GPU.*

### Linear Probe Accuracy on Food-101

| Encoder | Backbone | Top-1 Acc | Top-5 Acc |
|---|---|---|---|
| ImageNet pretrained (baseline) | ResNet-50 | ~55–60% | ~80–84% |
| Food-DINO pretrained | ResNet-50 | ~58–64% | ~82–87% |
| ImageNet pretrained (baseline) | ViT-S/16  | ~60–65% | ~83–87% |
| Food-DINO pretrained | ViT-S/16  | ~63–68% | ~85–89% |

Even a small gain over the ImageNet baseline (+2–5 pp) confirms the encoder has adapted to food-domain features beyond what ImageNet pretraining provides.

### Visualizations

- `tsne_visualization.png` — color-coded scatter plot of 5,000 validation features showing food category clusters.
- `attention_maps.png` — ViT attention heatmaps demonstrating focus on food regions (ingredient boundaries, textures).

---

## Directory Structure

```
unsupervised/
├── configs/
│   ├── resnet50.yaml
│   └── vit_small16.yaml
├── data/
│   └── prepare_food101.py
├── models/
│   ├── backbones.py
│   └── dino.py
├── utils/
│   ├── augmentations.py
│   └── checkpoint.py
├── pretrain.py
├── evaluate.py
├── export_weights.py
├── extract_embeddings.py
├── requirements.txt
└── README.md
```
