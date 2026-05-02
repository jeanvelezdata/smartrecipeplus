# Cuisine-Aware Vision-Language Learning for Ingredient and Recipe Prediction

**CS 7643 — Deep Learning | Georgia Institute of Technology**

Alejandra Ordaz Lopez · Zhangcao Luk · Abdoulreza Ghotbi · Jean C. Vélez

---

## Overview

This project builds a food understanding system that combines contrastive vision-language pretraining with a hierarchical ingredient prediction framework. The core insight is that visually similar dishes (e.g., a bowl of noodles in broth) can belong to completely different culinary traditions and require distinct ingredient vocabularies — injecting cuisine context resolves this ambiguity.

The pipeline has three main stages:
1. **CLIP fine-tuning + FAISS retrieval** — align food images with recipe text; retrieve top-*k* recipes at inference.
2. **Cuisine pseudo-labeling** — a TF-IDF + LinearSVC teacher trained on *What's Cooking* generates cuisine labels for the unlabeled food image dataset.
3. **Dual-head ingredient prediction** — a cuisine classifier and a cuisine-aware ingredient predictor that fuses visual features with soft cuisine posteriors.

A preliminary DINO-based unsupervised pretraining stage (ResNet-50 and ViT-Small/16 on Food-101) is also included; it was superseded by the supervised CLIP pipeline for the end task but the code remains in [`unsupervised/`](unsupervised/).

---

## Results

### CLIP + FAISS Retrieval

| Metric | Value |
|---|---|
| Top-1 Accuracy | ~1.0000 |
| Top-5 Accuracy | 0.9994 |
| MRR@5 | 0.9835 |
| Jaccard Similarity (Top-1) | 0.9719 |

### Cuisine-Aware vs. Baseline Ingredient Prediction (Experiment D — last 4 layers)

| Model | Validation Loss |
|---|---|
| Baseline (image only) | 0.1656 |
| Cuisine-Aware | 0.1625 |

Largest per-cuisine F1 gains over the baseline:

| Cuisine | Baseline F1 | Cuisine-Aware F1 | ΔF1 |
|---|---|---|---|
| Jamaican | 0.2196 | 0.2931 | +7.35% |
| Vietnamese | — | — | +4.97% |
| Chinese | 0.2511 | 0.2977 | +4.66% |
| Korean | — | — | +4.58% |
| Brazilian | 0.1025 | 0.1228 | +20.0% (rel.) |

Cuisine classifier validation accuracy: ~38–40%.

### DINO Pretraining (Food-101 Linear Probe)

| Encoder | Backbone | Top-1 | Top-5 |
|---|---|---|---|
| ImageNet baseline | ResNet-50 | ~68.7% | ~90.4% |
| Food-DINO | ResNet-50 | ~86.7% | ~97.4% |
| ImageNet baseline | ViT-S/16 | ~79.4% | ~94.9% |
| Food-DINO | ViT-S/16 | ~88.1% | ~97.5% |

---

## Repository Structure

```
.
├── Experiments-recipe_cuisine_aware_merged.ipynb   # Main pipeline notebook
├── train_dino_food.ipynb                            # DINO pretraining notebook
├── report.tex                                       # Final project report
├── unsupervised/                                    # DINO code and documentation
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── pretrain.py
│   ├── evaluate.py
│   ├── export_weights.py
│   ├── extract_embeddings.py
│   └── README.md
└── DINO_UNSUPERVISED_PRETRAINING_ARCHITECTURE.md    # DINO design document
```

---

## Datasets

| Dataset | Use | Source |
|---|---|---|
| Food Ingredients and Recipes Dataset with Images | CLIP fine-tuning, retrieval, ingredient prediction | [Kaggle](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) |
| What's Cooking | Cuisine teacher model (39,774 recipes, 20 cuisines) | [Kaggle](https://www.kaggle.com/c/whats-cooking) |
| Food-101 | DINO unsupervised pretraining (~101K images) | HuggingFace `ethz/food101` |

The recipes dataset is downloaded via KaggleHub. Entries with missing values or invalid image references are filtered out. Each recipe is represented as a unified text string combining the title and the first six ingredients.

---

## Pipeline

### Stage 1 — CLIP Fine-tuning

Base model: `openai/clip-vit-base-patch32`. Selective fine-tuning unfreezes a configurable subset of layers while freezing the rest to prevent catastrophic forgetting.

**Ablation configurations:**

| Config | Unfrozen layers | CLIP val loss | Downstream ingredient F1 |
|---|---|---|---|
| A — projection only | projection heads only | 2.3065 | weakest |
| B — last 1 | last 1 encoder layer + projection | — | — |
| C — last 2 | last 2 encoder layers + projection | 1.7649 (best) | strong |
| D — last 4 | last 4 encoder layers + projection | higher overfitting | best downstream |

Training uses symmetric contrastive loss (temperature 0.07), AdamW at lr=1e-5, batch size 16, 3 epochs.

### Stage 2 — FAISS Index & Retrieval

Image embeddings are L2-normalized and indexed with FAISS (inner product = cosine similarity). At inference, the top-*k* most similar recipes are retrieved with their title, ingredients, and similarity score.

### Stage 3 — Cuisine Expert Teacher

A TF-IDF vectorizer + LinearSVC trained on *What's Cooking* (85/15 stratified split) generates pseudo-cuisine labels for the food image dataset. This transfers cuisine supervision without any manual labeling.

### Stage 4 — Dual-Head Ingredient Prediction

**Cuisine Classifier** — lightweight MLP operating on 512-dim CLIP image embeddings, trained with cross-entropy loss.

**Baseline Ingredient Predictor** — MLP operating on image embeddings alone, trained with binary cross-entropy against multi-hot ingredient labels (vocabulary size: 300).

**Cuisine-Aware Ingredient Predictor** — extends the baseline by fusing image embeddings with a 32-dim cuisine embedding. At inference, `forward_soft` passes Softmax cuisine probabilities (rather than a hard label) through the fusion layer, propagating cuisine uncertainty.

**Key hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Ingredient vocabulary size | 300 |
| Cuisine embedding size | 32 |
| Downstream learning rate | 1e-3 |
| Downstream epochs | 15–20 |
| Ingredient threshold | 0.30 |
| Optimizer | AdamW |

---

## Running the Main Notebook

Open `Experiments-recipe_cuisine_aware_merged.ipynb` in Google Colab or a local Jupyter environment. The notebook is self-contained and runs all stages in order:

1. Imports & dataset download
2. Data loading & augmentation
3. CLIP fine-tuning
4. FAISS index & recipe search
5. Baseline evaluation (Top-K, MRR, Jaccard)
6. t-SNE visualization
7. Cuisine expert teacher model
8. Cuisine-aware extension (dual-head training, evaluation, comparison)

**Recommended runtime:** Colab L4 GPU or equivalent. The CLIP fine-tuning stage takes the longest (~hours depending on dataset size and configuration).

---

## Limitations

- Cuisine classification accuracy plateaued at ~38–40%; noisy cuisine priors occasionally hurt prediction (e.g., Cajun Creole ΔF1 = −0.0396).
- The dataset is heavily skewed toward Italian and Southern US cuisines; minority cuisines like Filipino (23 samples) and Brazilian (34 samples) yield less stable evaluations.
- CLIP fine-tuning exhibits consistent overfitting, with train–validation loss gaps of 1.25–1.57 loss units.
- Retrieval metrics are near-saturated (Top-5 ≈ 0.9994), making them ineffective for distinguishing between fine-tuning configurations.

---

## Unsupervised Pretraining (DINO)

See [`unsupervised/README.md`](unsupervised/README.md) for full documentation on the DINO pretraining component (setup, training commands, evaluation, pretrained weight export).

---

## Team Contributions

| Member | Role |
|---|---|
| Abdoulreza Ghotbi | Dataset preprocessing, CLIP fine-tuning, FAISS retrieval system, evaluation (Top-K, MRR, Jaccard) |
| Zhangcao Luk | Cuisine-aware dual-head architecture, cuisine classifier, results analysis |
| Alejandra Ordaz Lopez | CLIP ablation experiments (A–D), threshold tuning, early stopping experiments, quantitative & qualitative analysis |
| Jean C. Vélez | DINO self-supervised pretraining (ResNet-50 & ViT-S/16), Food-101 linear probe evaluation |
