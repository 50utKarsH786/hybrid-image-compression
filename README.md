# Hybrid Deep Learning Architecture for Scalable and High-Quality Image Compression

> Implementation of Al-Khafaji & Ramaha (2025) — *Scientific Reports* 15, 22926  
> DOI: [10.1038/s41598-025-06481-0](https://doi.org/10.1038/s41598-025-06481-0)

![CI](https://github.com/YOUR_USERNAME/hybrid-image-compression/actions/workflows/ci.yml/badge.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/hybrid-image-compression/blob/main/colab_notebook.ipynb)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Demo](#demo)
- [Results](#results)
- [Git Workflow (Group Collaboration)](#git-workflow-group-collaboration)
  - [Branch Strategy](#branch-strategy)
  - [Commit Message Convention](#commit-message-convention)
  - [Step-by-Step for Each Member](#step-by-step-for-each-member)
- [Running Tests](#running-tests)
- [Team Members & Responsibilities](#team-members--responsibilities)

---

## Overview

This project implements a **hybrid deep learning-based image compression framework** combining:

| Component | Role |
|-----------|------|
| **GLCM** (Gray-Level Co-occurrence Matrix) | Texture-aware spatial feature extraction |
| **K-Means Clustering** | Adaptive region-based compression (no manual ROI needed) |
| **SWT** (Stationary Wavelet Transform) | Shift-invariant multiresolution decomposition |
| **SDAE** (Stacked Denoising Autoencoder) | Deep feature compression with residual connection |
| **MSE + SSIM Loss** | Perceptual quality-aware training objective |

### Key Results (from paper)

| Dataset | PSNR (dB) | SSIM | MS-SSIM | BPP |
|---------|-----------|------|---------|-----|
| NIH Chest X-ray | 50.06 | 0.9963 | 1.0000 | 0.832 |
| INBreast | 48.85 | 0.9866 | 0.9997 | 0.443 |
| Camelyon16 | 49.75 | 0.9997 | 1.0000 | 0.851 |
| DIV2K | 51.85 | 1.0000 | 1.0000 | 0.598 |
| Kodak | 51.30 | 0.9993 | 1.0000 | 0.612 |

Encoding–decoding time: **0.065 s** (fastest among all compared methods).

---

## Architecture

```
Input Image
    │
    ▼
Pre-Processing (resize 256×256, normalise [0,1])
    │
    ▼
GLCM Feature Extraction  ──►  K-Means Clustering
    │                               │
    └───────────────────────────────┘
                    │
                    ▼ (per region/cluster)
          Stationary Wavelet Transform (SWT)
                    │
          ┌─────────┴──────────┐
          │                    │
    Approx. Coeffs       Detail Coeffs
          └─────────┬──────────┘
                    │
                    ▼
         SDAE Encoder → Bottleneck (quantised)
                    │
                    ▼  (stored / transmitted)
         SDAE Decoder
                    │
                    ▼
         Inverse SWT (ISWT)
                    │
                    ▼
     Residual Add (+ input)  → Reconstructed Image
```

---

## Project Structure

```
hybrid-image-compression/
├── src/
│   ├── __init__.py
│   ├── model.py          # GLCM, K-Means, SWT, SDAE, HybridCompressor
│   ├── metrics.py        # PSNR, SSIM, MS-SSIM, BPP, PRD, MSE
│   └── data_utils.py     # Dataset loaders, preprocessing, augmentation
├── tests/
│   ├── __init__.py
│   └── test_model.py     # Unit tests for all components
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI pipeline
├── train.py              # Training entry-point
├── evaluate.py           # Evaluation / inference script
├── demo.py               # Interactive single-image demo
├── colab_notebook.ipynb  # Google Colab guided notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

Organise datasets under `data/` as follows:

```
data/
├── DIV2K/
│   ├── train/          ← training images (.png)
│   └── valid/
├── NIH_Chest_Xray/
│   └── images/
├── INBreast/
│   └── images/
├── Camelyon16/
│   └── patches/
└── Kodak/              ← 24 test images
```

> **Download links**  
> - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  
> - [NIH Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC)  
> - [INBreast](https://www.kaggle.com/datasets/martholi/inbreast)  
> - [Camelyon16](https://camelyon16.grand-challenge.org/Data/)  
> - [Kodak](https://r0k.us/graphics/kodak/)

---

## Usage

### Training

```bash
python train.py \
  --data_dir   data/DIV2K/train \
  --epochs     50 \
  --batch_size 16 \
  --bottleneck 64 \
  --n_clusters 4 \
  --output_dir outputs/

# Quick test run (100 images, 5 epochs):
python train.py --data_dir data/DIV2K/train --max_images 100 --epochs 5
```

Outputs saved to `outputs/`:
- `best_weights.h5` — best validation checkpoint
- `final_weights.h5` — final epoch weights
- `training_curves.png` — loss curves
- `training_log.csv` — per-epoch metrics
- `config.json` — full training configuration

---

### Evaluation

```bash
# Evaluate on the Kodak benchmark:
python evaluate.py \
  --data_dir   data/Kodak \
  --weights    outputs/best_weights.h5 \
  --output_dir results/kodak/ \
  --save_images

# Evaluate on NIH Chest X-ray (first 100):
python evaluate.py \
  --data_dir   data/NIH_Chest_Xray/images \
  --weights    outputs/best_weights.h5 \
  --max_images 100 \
  --output_dir results/nih/
```

Outputs: `results.json`, `metrics_bar.png`, per-image comparisons.

---

### Demo

```bash
# Run on a single image with full visualisation:
python demo.py \
  --image   data/Kodak/kodim01.png \
  --weights outputs/best_weights.h5

# Run without trained weights (shows architecture outputs):
python demo.py --image samples/test.png --no_weights
```

Outputs in `results/demo/`:
- `original.png`, `reconstructed.png`
- `full_comparison.png`
- `glcm_map.png` — GLCM energy heatmap
- `cluster_map.png` — K-Means region clusters
- `swt_subbands.png` — SWT sub-band visualisation

---

## Results

After training, run evaluation and compare your results against Table 5 of the paper:

| Method | PSNR (dB) | MS-SSIM | BPP |
|--------|-----------|---------|-----|
| Li et al. 2025 | 49.1 | 0.9987 | 0.68 |
| Finder et al. 2022 | 47.0 | 0.997 | 0.80 |
| Amina et al. 2024 | 46.78 | 0.990 | 1.50 |
| **Ours (paper)** | **50.36** | **0.9999** | **0.6677** |

---

## Git Workflow (Group Collaboration)

### Branch Strategy

```
main           ← production-ready, protected (no direct pushes)
  └── develop  ← integration branch (all features merge here first)
        ├── feature/data-and-preprocessing   (Member 1)
        ├── feature/glcm-kmeans              (Member 2)
        ├── feature/swt-sdae-model           (Member 3)
        ├── feature/evaluation-and-demo      (Member 4)
        └── bugfix/fix-<short-description>
```

### Commit Message Convention

Use the **Conventional Commits** format:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | A new feature or function |
| `fix` | A bug fix |
| `refactor` | Code restructuring (no feature/bug change) |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `chore` | Dependency updates, config changes |
| `style` | Formatting, whitespace (no logic change) |
| `perf` | Performance improvements |

**Examples:**

```bash
git commit -m "feat(glcm): implement patch-wise GLCM feature extraction"
git commit -m "feat(sdae): add residual connection to decoder output"
git commit -m "fix(swt): handle odd-dimension images with zero-padding"
git commit -m "test(metrics): add PSNR/SSIM unit tests for identical images"
git commit -m "docs(readme): add dataset download links and setup instructions"
git commit -m "refactor(model): extract SWT utility into separate functions"
git commit -m "chore(deps): pin tensorflow to 2.12 for stability"
git commit -m "perf(kmeans): reduce GLCM sample size to 200 for faster fitting"
```

---

### Step-by-Step for Each Member

#### Initial setup (once per member)


## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test class
pytest tests/test_model.py::TestSDAE -v
```

CI runs automatically on every push and pull request via GitHub Actions.

---

## Team Members & Responsibilities

| Member | Branch | Responsibility |
|--------|--------|----------------|
| Member 1 | `feature/data-and-preprocessing` | Dataset loaders, preprocessing, augmentation (`src/data_utils.py`) |
| Member 2 | `feature/glcm-kmeans` | GLCM texture features + K-Means clustering (`src/model.py`) |
| Member 3 | `feature/swt-sdae-model` | SWT pipeline + SDAE architecture + HybridCompressor (`src/model.py`) |
| Member 4 | `feature/evaluation-and-demo` | Metrics, evaluation scripts, demo, Colab notebook |



---

## Citation

```bibtex
@article{alkhafaji2025hybrid,
  title   = {Hybrid deep learning architecture for scalable and high-quality image compression},
  author  = {Al-Khafaji, Mustafa and Ramaha, Nehad T. A.},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {22926},
  year    = {2025},
  doi     = {10.1038/s41598-025-06481-0}
}
```

---

## License

This project is released under the MIT License.
