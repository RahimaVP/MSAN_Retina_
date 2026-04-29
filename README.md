# MSAN_Retina

PyTorch implementation of **"Multi-Modal Retinal Image Classification with Modality-Specific Attention Network"**, published in *IEEE Transactions on Medical Imaging*, 40(6): 1591–1602, 2021.

> If you use this code, please cite:
> X. He, Y. Deng, L. Fang and Q. Peng, "Multi-Modal Retinal Image Classification with Modality-Specific Attention Network," *IEEE Transactions on Medical Imaging*, 40 (6): 1591–1602, 2021.

Contact: [leyuan_fang@hnu.edu.cn](mailto:leyuan_fang@hnu.edu.cn) or [xx_h@hnu.edu.cn](mailto:xx_h@hnu.edu.cn)

---

## Overview

MSAN is a dual-branch deep learning framework for classifying retinal diseases from two complementary imaging modalities:

- **Fundus images** — color fundus photographs
- **OCT B-scans** — Optical Coherence Tomography cross-sections with automatically generated ROI masks

The two branches extract modality-specific features independently, then fuse them for a final classification decision. The model supports two clinical tasks:

| Task | Classes |
|------|---------|
| Macula | acute CSR, chronic CSR, ci-DME, geographic AMD, Healthy, neovascular AMD |
| Optic Disc (OD) | Glaucoma, Healthy |

---

## Architecture

```
Fundus Image ──► ResNet18 + PAM (Position Attention) ──► 512-dim features ──┐
                                                                              ├──► Fusion FC ──► Prediction
OCT B-scan ───► ResNet18 (4-ch: RGB + ROI mask) ──────► 512-dim features ──┘
```

### Branches

**Fundus Branch (`model/MSA_subnet.py`)**
- ResNet18 backbone
- Position Attention Module (PAM) applied after the first residual block
- PAM uses query/key/value self-attention to capture long-range spatial dependencies

**OCT Branch (`model/RGA_subnet.py`)**
- ResNet18 backbone pre-trained on ImageNet
- Modified first conv layer accepts 4-channel input: 3-channel OCT + 1-channel ROI mask
- ROI mask weights initialized to zero; RGB weights copied from pre-trained model

**Fusion Classifier (`model/msan.py`)**
- Concatenates both 512-dim feature vectors → 1024-dim
- FC(1024 → 512) → ReLU → Dropout(0.5) → FC(512 → num_classes)

### Loss Function

Combined loss from all three outputs:

```
Loss = 0.5 × Loss_fundus + 0.5 × Loss_oct + Loss_fusion
```

Each term uses **Focal Loss** (γ=2, α=0.25) to handle class imbalance.

### Attention Modules (`utils/attention.py`)

- **PAM_Module** — Position attention via non-local self-attention (spatial)
- **CAM_Module** — Channel attention using max-energy aggregation
- **semanticModule** — U-Net style encoder-decoder for semantic feature compression

---

## Project Structure

```
MSAN_Retina/
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── best_model.pth            # Saved best model checkpoint
├── confusion_matrix.png      # Output: confusion matrix
│
├── model/
│   ├── msan.py               # Top-level MSAN model
│   ├── MSA_subnet.py         # Fundus branch (ResNet18 + PAM)
│   ├── RGA_subnet.py         # OCT branch (4-channel ResNet18)
│   └── ResNet.py             # ResNet building blocks
│
├── data/
│   ├── dataset.py            # MultiModalDataset (PyTorch Dataset)
│   ├── prepare_data.py       # Data preparation and CSV generation
│   ├── macula_train.csv      # Macula training split
│   ├── macula_val.csv        # Macula validation split
│   ├── macula_test.csv       # Macula test split
│   ├── od_train.csv          # OD training split
│   ├── od_val.csv            # OD validation split
│   └── od_test.csv           # OD test split
│
├── utils/
│   ├── FocalLoss.py          # Focal loss implementation
│   ├── attention.py          # PAM, CAM, semanticModule
│   ├── utils.py              # Confusion matrix, model utilities
│   ├── classerrormeter.py    # AUC meter for multi-class ROC
│   └── logger.py             # File logging utility
│
└── latest/
    └── Dataset/Dataset/
        ├── Macula/           # Raw macula images (6 classes)
        ├── OD/               # Raw OD images (2 classes)
        └── generated_roi_masks/  # Auto-generated Otsu ROI masks
```

---

## Dataset

### Raw Data Layout

The dataset follows a hierarchical structure:

```
Dataset/
├── Macula/
│   ├── acute CSR/
│   │   ├── P_1/
│   │   │   ├── Left Eye/
│   │   │   │   ├── *_Color_*.jpg      ← Fundus image
│   │   │   │   ├── *_B-scan_*.jpg     ← OCT B-scan
│   │   │   │   └── VolumeFrames/
│   │   │   └── Right Eye/
│   │   └── P_2/ ...
│   ├── chronic CSR/
│   ├── ci-DME/
│   ├── geographic_AMD/
│   ├── Healthy/
│   └── neovascular_AMD/
└── OD/
    ├── Glaucoma/
    └── Healthy/
```

### CSV Format

Each CSV file (train/val/test) contains:

| Column | Description |
|--------|-------------|
| `fundus_path` | Absolute path to fundus color image |
| `oct_path` | Absolute path to OCT B-scan image |
| `roi_path` | Absolute path to generated ROI mask |
| `label` | Integer class index |
| `class_name` | Human-readable class name |
| `patient_id` | Patient identifier (used for group-aware splitting) |

### Data Splits

Splits are performed at the **patient level** using `GroupShuffleSplit` to prevent data leakage:

| Split | Proportion |
|-------|-----------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

---

## Installation

```bash
git clone https://github.com/your-repo/MSAN_Retina.git
cd MSAN_Retina
pip install -r requirements.txt
```

### Requirements

```
torch==2.9.0
torchvision==0.24.0
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.7.2
opencv-python==4.12.0.88
tqdm==4.66.5
pillow==12.0.0
```

---

## Usage

### Step 1 — Prepare Data

Run this once to scan the raw dataset, generate ROI masks via Otsu thresholding, and produce the train/val/test CSV splits.

```bash
python data/prepare_data.py
```

Edit the `DATASET_ROOT` variable inside `prepare_data.py` to point to your local dataset path before running.

This will:
1. Scan all patient/eye directories for fundus and OCT image pairs
2. Generate binary ROI masks using Gaussian blur + Otsu thresholding + median blur
3. Save masks to `latest/Dataset/Dataset/generated_roi_masks/`
4. Output `data/macula_train.csv`, `data/macula_val.csv`, `data/macula_test.csv` (and OD equivalents)

### Step 2 — Train

```bash
python train.py \
  --train_csv data/macula_train.csv \
  --val_csv data/macula_val.csv \
  --num_classes 6 \
  --image_size 300 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --epochs 50 \
  --model_path best_model.pth
```

For the OD (Glaucoma) task:

```bash
python train.py \
  --train_csv data/od_train.csv \
  --val_csv data/od_val.csv \
  --num_classes 2
```

Training outputs:
- `best_model.pth` — checkpoint with best validation accuracy
- `convergence_graph.png` — loss and accuracy curves

### Step 3 — Evaluate

```bash
python test.py
```

Evaluation outputs:
- Accuracy, Precision, Recall, F1-score (weighted)
- Per-class AUC scores
- `roc_curves.png` — one-vs-rest ROC curves for all classes
- `confusion_matrix.png` — normalized confusion matrix heatmap

---

## Training Configuration

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `image_size` | 300 | Input image resolution (H × W) |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 0.001 | Initial Adam learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `epochs` | 50 | Total training epochs |
| `scheduler` | StepLR | step_size=30, gamma=0.1 |
| `focal_gamma` | 2 | Focal loss focusing parameter |
| `focal_alpha` | 0.25 | Focal loss weighting factor |

### Data Augmentation (Training Only)

| Transform | Parameters |
|-----------|-----------|
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | ±10° |
| ColorJitter | brightness=0.1, contrast=0.1 |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

Validation and test sets use resize + normalize only (no augmentation).

---

## ROI Mask Generation

ROI masks are generated automatically from OCT B-scans using classical image processing:

1. Read OCT image as grayscale
2. Apply Gaussian blur (5×5 kernel)
3. Apply Otsu's thresholding → binary mask
4. Apply median blur (5×5) to remove noise
5. Save as PNG alongside the dataset

These masks are fed as a 4th channel into the OCT branch, guiding the network to focus on clinically relevant retinal regions.

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best model weights (state dict) |
| `convergence_graph.png` | Train/val loss and accuracy over epochs |
| `roc_curves.png` | Per-class ROC curves with AUC values |
| `confusion_matrix.png` | Normalized confusion matrix |
| `data/*_master.csv` | Full dataset metadata before splitting |
| `data/*_train/val/test.csv` | Patient-level train/val/test splits |

---

## Citation

```bibtex
@article{he2021msan,
  author  = {He, Xingxin and Deng, Yuhao and Fang, Leyuan and Peng, Qiang},
  title   = {Multi-Modal Retinal Image Classification with Modality-Specific Attention Network},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {40},
  number  = {6},
  pages   = {1591--1602},
  year    = {2021}
}
```
