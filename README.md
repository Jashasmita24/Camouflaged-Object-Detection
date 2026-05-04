# 🦎 Camouflaged Object Detection (CODNet + UNet + YOLOv8)

**Deep Learning based Camouflaged Object Detection from Multispectral Images**

> PhD Registration Seminar Prototype — OUTR Bhubaneswar  
> **Researcher:** Jashasmita Pal (Roll No: 24520007)  
> **Supervisors:** Dr. Jibitesh Mishra, Dr. Asimananda Khandual  
> **School:** School of Computer Science

---

## 📌 Project Overview

This project implements a **triple-model** deep learning system for detecting camouflaged objects (animals, insects, etc.) that blend seamlessly into their backgrounds:

| Model | Type | Key Features |
|-------|------|--------------|
| **CODNet** | Segmentation | ResNet50 backbone + CSIM + SGFL + Boundary-Guided Decoder |
| **UNet** | Segmentation | Encoder-Decoder with skip connections (ResNet50 or custom encoder) |
| **YOLOv8** | Object Detection | Real-time bounding-box detection via Ultralytics |

A fully interactive **Streamlit web demo** is included for real-time inference, gallery browsing, and batch processing with all three models.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the CAMO dataset from Kaggle
python download_dataset.py

# 3. Launch the Streamlit demo (no training needed for demo)
streamlit run app.py

# 4. (Optional) Train CODNet from scratch
python train.py --model_type codnet --epochs 100 --batch_size 8

# 5. (Optional) Train UNet from scratch
python train.py --model_type unet --epochs 100 --batch_size 8

# 6. (Optional) Fine-tune YOLOv8
python train_yolo.py

# 7. (Optional) Run inference on images
python inference.py --image path/to/image.jpg --checkpoint checkpoints/codnet_best.pth
python inference.py --model unet --image path/to/image.jpg
python inference.py --model yolo --image path/to/image.jpg

# 8. (Optional) Evaluate on test set
python test.py --checkpoint checkpoints/codnet_best.pth
python test.py --checkpoint checkpoints/unet/unet_best.pth --model_type unet
```

---

## 🏗️ Model Architectures

### CODNet (Segmentation)

```
Input Image (RGB, 352×352)
        ↓
  ResNet50 Backbone      → C2 (256ch), C3 (512ch), C4 (1024ch), C5 (2048ch)
        ↓
  CSIM (Cross-Scale Interaction Module)   → 4 fused features (64ch each)
        ↓
  SGFL (Semantic Guided Feature Learning) → 4 refined features + 4 boundary maps
        ↓
  Boundary-Guided Decoder                 → Main prediction + 4 coarse predictions
        ↓
  Output Segmentation Mask (H×W, binary)
```

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | **ResNet50 Backbone** | Multi-scale feature extraction (4 scale levels) |
| 2 | **CSIM** | Cross-scale interaction with channel + spatial attention |
| 3 | **SGFL** | Foreground enhancement, background suppression, boundary extraction |
| 4 | **Decoder** | Progressive boundary-guided upsampling with deep supervision |

### UNet (Segmentation)

```
Input Image (RGB, 352×352)
        ↓
  Encoder (ResNet50 or Custom)   → 4 skip connections + bottleneck
        ↓
  Decoder (UpBlocks + Skip Concat) → Progressive upsampling
        ↓
  1×1 Conv Head
        ↓
  Output Segmentation Mask (H×W, binary)
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Encoder | **ResNet50 / Custom** | 4-stage downsampling with skip connections |
| Bottleneck | **ConvBlock** | Deepest feature map processing |
| Decoder | **UpBlocks** | Transposed conv + skip concat + double conv |
| Head | **1×1 Conv** | Single-channel segmentation logits |

### YOLOv8 (Object Detection)

| Variant | Params | Speed | mAP50 |
|---------|--------|-------|-------|
| **yolov8n** | 3.2 M | Fastest | 37.3 |
| **yolov8s** | 11.2 M | Fast | 44.9 |
| **yolov8m** | 25.9 M | Medium | 50.2 |
| **yolov8l** | 43.7 M | Slow | 52.9 |
| **yolov8x** | 68.2 M | Slowest | 53.9 |

---

## 📊 Evaluation Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **MAE** | Mean Absolute Error | ↓ Lower is better |
| **S-measure** | Structural similarity (object + region) | ↑ Higher is better |
| **E-measure** | Enhanced alignment measure | ↑ Higher is better |
| **wF-measure** | Weighted F-measure (precision/recall) | ↑ Higher is better |

---

## 📁 Project Structure & File Descriptions

```
Camouflaged-Object-Detection-main/
├── app.py                   # Streamlit web demo (CODNet + UNet + YOLOv8)
├── train.py                 # Model training script (CODNet / UNet)
├── train_yolo.py            # YOLOv8 fine-tuning script
├── inference.py             # Single/batch inference (CODNet / UNet / YOLOv8)
├── test.py                  # Model evaluation script (CODNet / UNet)
├── download_dataset.py      # Kaggle dataset downloader
├── start_training.py        # Smart training launcher (auto-detects dataset)
├── setup_train_split.py     # Creates train/test splits from COD10K-v3
├── requirements.txt         # Python dependencies
├── kaggle.json              # Kaggle API credentials (user-supplied)
├── flowchart.jpeg           # Architecture flowchart diagram
├── download_log.txt         # Log file from dataset download
├── .gitignore               # Git ignore rules
│
├── config/
│   ├── __init__.py          # Package init
│   └── config.py            # Central config (MODEL_TYPE, hyperparameters, paths)
│
├── models/
│   ├── __init__.py          # Package init (exports CODNet, UNet, YOLODetector)
│   ├── backbone.py          # ResNet50 / Res2Net50 feature extractor
│   ├── cod_net.py           # CODNet model (Backbone → CSIM → SGFL → Decoder)
│   ├── unet.py              # UNet model (Encoder-Decoder with skip connections)
│   ├── csim.py              # Cross-Scale Interaction Module
│   ├── sgfl.py              # Semantic Guided Feature Learning Module
│   ├── decoder.py           # Boundary-Guided Progressive Decoder
│   └── yolo_detector.py     # YOLOv8 wrapper (Ultralytics)
│
├── datasets/
│   ├── __init__.py          # Package init
│   ├── cod_dataset.py       # COD10K / CAMO / CHAMELEON dataset loader
│   ├── animals_dataset.py   # CAMO Kaggle animals dataset loader
│   └── transforms.py        # Training & test augmentation pipelines
│
└── utils/
    ├── __init__.py          # Package init
    ├── losses.py            # CODLoss + UNetLoss (BCE + IoU + Boundary)
    └── metrics.py           # MAE, S-measure, E-measure, wF-measure calculators
```

---

## 📄 Detailed File Descriptions

### Root Scripts

#### `app.py`
The main **Streamlit web application** providing an interactive UI for camouflaged object detection with **triple-model support**. It features:
- **Three detection modes**: CODNet (Segmentation), UNet (Segmentation), and YOLOv8 (Object Detection) — switchable via sidebar radio.
- **Upload & Detect tab**: Upload any animal image and get real-time detection with heatmap, binary mask, and overlay visualizations.
- **Animal Gallery tab**: Browse and run detection on images from the downloaded CAMO dataset.
- **Batch Process tab**: Upload up to 20 images at once and receive a summary results table.
- **About tab**: Triple-model architecture details, evaluation metrics, and project information.
- Sidebar controls for input resolution, detection threshold, model-specific settings (YOLOv8 variant/confidence/IoU), and CUDA/CPU device info.

**Run with:** `streamlit run app.py`

---

#### `train.py`
The **full model training script** supporting both CODNet and UNet. Key features:
- `--model_type` argument selects between `codnet` and `unet`.
- Parses CLI arguments for backbone, epochs, batch size, learning rate, and image size.
- Supports **mixed precision (AMP)** training via `torch.cuda.amp` for faster training on CUDA GPUs.
- Uses **Adam optimizer** with **Cosine Annealing LR** scheduler.
- Trains with `CODLoss` (CODNet) or `UNetLoss` (UNet) — both use BCE + IoU + Boundary.
- Validates after every epoch, reports MAE, S-measure, E-measure, and wF-measure.
- Saves periodic and best-MAE checkpoints to `checkpoints/` (CODNet) or `checkpoints/unet/` (UNet).
- Supports **TensorBoard** logging to `runs/` for real-time monitoring.
- Supports checkpoint resuming with `--resume`.

**Run with:**
```bash
python train.py --model_type codnet --epochs 50 --batch_size 4
python train.py --model_type unet --epochs 50 --batch_size 4
```

---

#### `inference.py`
The **command-line inference and visualization script** supporting all three models. Key features:
- `--model` argument selects between `codnet`, `unet`, and `yolo`.
- Runs detection on a **single image** (`--image`) or an **entire folder** (`--image_dir`).
- Saves prediction masks (`*_mask.png`) and side-by-side result visualizations (`*_result.png`).
- CODNet/UNet: 3-panel figure (Input | Heatmap | Overlay). YOLOv8: 2-panel figure (Input | Detections).
- Supports configurable binary threshold (`--threshold`) and optional interactive display (`--show`).

**Run with:**
```bash
python inference.py --image img.jpg --checkpoint checkpoints/codnet_best.pth
python inference.py --model unet --image img.jpg
python inference.py --model yolo --image img.jpg --conf 0.25
```

---

#### `test.py`
The **model evaluation script** supporting both CODNet and UNet. Key features:
- Loads a trained checkpoint and evaluates it over the full test split.
- Auto-detects model type from checkpoint metadata, or accepts `--model_type` CLI argument.
- Computes and prints MAE, S-measure, E-measure, and wF-measure.
- Optionally saves all prediction masks to `results/predictions/` with `--save_results`.

**Run with:**
```bash
python test.py --checkpoint checkpoints/codnet_best.pth
python test.py --checkpoint checkpoints/unet/unet_best.pth --model_type unet
```

---

#### `train_yolo.py`
**YOLOv8 fine-tuning script** for camouflage detection. Key features:
- Wraps Ultralytics YOLOv8 training API for custom dataset fine-tuning.
- Configurable model variant, epochs, batch size, and image size.

**Run with:** `python train_yolo.py`

---

#### `download_dataset.py`
**Downloads the CAMO camouflaged animals dataset from Kaggle** (`tankz/camo-dataset`). Key features:
- Verifies that Kaggle API credentials (`~/.kaggle/kaggle.json`) are present before downloading.
- Downloads and extracts ~1,250 animal images (8 categories, ~130 MB) to `data/animals/`.
- Prints a directory tree of downloaded files after completion.

**Run with:** `python download_dataset.py`

---

#### `start_training.py`
A **smart training launcher** that auto-detects the COD10K dataset folder structure and starts training. Key features:
- Scans `data/COD10K/` recursively to find Train/Test image and GT directories (handles nested or flat layouts).
- Automatically patches `config/config.py` with the detected dataset paths.
- Launches `train.py` with recommended parameters (100 epochs, batch size 8, ResNet50 backbone).

**Run with:** `python start_training.py`

---

#### `setup_train_split.py`
Creates a **train/test data split** from the COD10K-v3 test data. Useful when the full training set is not available. Key features:
- Reads images and ground-truth masks from `data/COD10K/COD10K-v3/Test/`.
- Shuffles and splits data 80% train / 20% test (seed=42 for reproducibility).
- Copies split files into the standard `data/COD10K/Train/` and `data/COD10K/Test/` directory structure expected by `CODDataset`.

**Run with:** `python setup_train_split.py`

---

#### `requirements.txt`
Lists all **Python package dependencies** needed to run the project:

| Package | Purpose |
|---------|---------|
| `torch >= 2.0.0` | Core deep learning framework |
| `torchvision >= 0.15.0` | ResNet50 backbone, image transforms |
| `timm >= 0.9.0` | Optional Res2Net50 backbone |
| `albumentations >= 1.3.0` | Data augmentation pipeline |
| `opencv-python-headless` | Image processing utilities |
| `Pillow >= 10.0.0` | Image loading and saving |
| `scipy >= 1.11.0` | Morphological operations (boundary contours) |
| `tqdm >= 4.65.0` | Progress bars |
| `tensorboard >= 2.14.0` | Training metrics visualization |
| `numpy >= 1.24.0` | Array operations |
| `matplotlib >= 3.7.0` | Visualization and heatmap plots |
| `kaggle >= 1.6.0` | Kaggle dataset API |
| `streamlit >= 1.28.0` | Web demo UI framework |
| `pandas >= 2.0.0` | Batch results summary table |

---

#### `kaggle.json`
Kaggle API credentials file. Must be placed at `~/.kaggle/kaggle.json` (user home directory) before running `download_dataset.py`. Obtain it from [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token.

---

#### `flowchart.jpeg`
Architecture flowchart diagram illustrating the end-to-end CODNet pipeline visually.

---

#### `download_log.txt`
Log file generated during dataset download, recording which files were downloaded and any warnings encountered.

---

### `config/` — Configuration

#### `config/config.py`
**Central configuration file** — all hyperparameters and directory paths are defined here. Key settings:

| Category | Key Parameters |
|----------|---------------|
| **Paths** | `DATA_DIR`, `CHECKPOINT_DIR`, `RESULT_DIR`, `LOG_DIR` |
| **Dataset** | `TRAIN_IMAGE_DIR`, `TRAIN_GT_DIR`, `TEST_IMAGE_DIR`, `TEST_GT_DIR` |
| **Model** | `MODEL_TYPE = "codnet"`, `BACKBONE = "resnet50"`, `CHANNEL_DIM = 64` |
| **UNet** | `UNET_ENCODER = "resnet50"`, `UNET_CHECKPOINT_DIR` |
| **YOLOv8** | `YOLO_MODEL = "yolov8n"`, `YOLO_CONFIDENCE`, `YOLO_IOU` |
| **Training** | `IMAGE_SIZE = 352`, `BATCH_SIZE = 8`, `EPOCHS = 100`, `LEARNING_RATE = 1e-4` |
| **Loss** | `BCE_WEIGHT = 1.0`, `IOU_WEIGHT = 1.0`, `BOUNDARY_WEIGHT = 0.5` |
| **Misc** | `SEED = 42`, `SAVE_EVERY = 10`, `USE_AMP = True` |

Also auto-creates `data/`, `checkpoints/`, `checkpoints/yolo/`, `checkpoints/unet/`, `results/`, and `runs/` directories on import.

---

### `models/` — Neural Network Modules

#### `models/backbone.py`
**Feature extraction backbone** wrapping pretrained ResNet50 (default) or Res2Net50 (optional via `timm`). Extracts **4-scale feature maps** from the input image:

| Feature | Resolution | Channels |
|---------|-----------|---------|
| C2 | H/4 × W/4 | 256 |
| C3 | H/8 × W/8 | 512 |
| C4 | H/16 × W/16 | 1024 |
| C5 | H/32 × W/32 | 2048 |

Loaded with ImageNet pretrained weights by default. The `get_backbone(name, pretrained)` factory function selects between ResNet50 and Res2Net50.

---

#### `models/cod_net.py`
**Main CODNet model class** that assembles all modules into a complete end-to-end pipeline. Takes an RGB image tensor `[B, 3, H, W]` and returns:
- `main_pred`: Final segmentation map `[B, 1, H, W]`
- `coarse_preds`: List of 4 deep supervision predictions
- `boundary_maps`: List of 4 boundary maps from SGFL

Also provides a `build_model(config)` factory function for instantiation from config.

---

#### `models/csim.py`
**Cross-Scale Interaction Module (CSIM)** — performs feature alignment and fusion across adjacent backbone scales. Contains:
- `ChannelAttention`: Global average + max pooling squeeze-and-excitation block.
- `SpatialAttention`: Channel-wise mean + max pooling spatial gate.
- `CrossScaleInteractionBlock`: Upsamples lower-resolution feature to match higher-resolution, reduces both to 64 channels, concatenates, fuses, then applies channel & spatial attention.
- `CSIM`: Applies cross-scale interaction to all 4 backbone feature levels (C2↔C3, C3↔C4, C4↔C5, C5→reduce), producing 4 uniform 64-channel features.

---

#### `models/sgfl.py`
**Semantic Guided Feature Learning Module (SGFL)** — refines cross-scale features using semantic guidance and boundary awareness. Contains:
- `GlobalContextBlock`: GAP-based scene-level semantic weighting.
- `ForegroundEnhancementBlock`: Multi-scale dilated convolutions (dilation 1, 3, 5) with residual connection to amplify foreground regions.
- `BoundaryExtractionBlock`: Learned edge detector that outputs both refined boundary features and a 1-channel boundary prediction map.
- `SGFLBlock`: Combines all three sub-blocks per feature level.
- `SGFL`: Applies `SGFLBlock` independently to all 4 feature levels, returning refined features and boundary maps.

---

#### `models/unet.py`
**U-Net segmentation model** with two encoder options:

| Encoder | Skip Channels | Bottleneck | Pretrained |
|---------|--------------|------------|------------|
| **ResNet50** | [64, 256, 512, 1024] | 2048 | ImageNet |
| **Custom** | [64, 128, 256, 512] | 1024 | No |

Contains:
- `ConvBlock`: Two 3×3 Conv-BN-ReLU layers (standard U-Net block).
- `UpBlock`: Transposed convolution + skip-connection concatenation + ConvBlock.
- `ResNet50Encoder`: Pretrained ResNet50 repurposed as encoder with 4 skip taps.
- `CustomEncoder`: Lightweight 4-stage ConvBlock encoder (no pretraining).
- `UNet`: Full encoder-decoder model. `forward()` returns `(main_pred, [], [])` for pipeline compatibility.
- `build_unet(config)`: Factory function for config-driven instantiation.

---

#### `models/yolo_detector.py`
**YOLOv8 wrapper** around the Ultralytics library for bounding-box object detection. Provides:
- `YOLODetector`: Wraps model loading, inference, detection drawing, and coverage computation.
- `load_yolo_model()`: Factory function for loading a YOLO model variant.

---

#### `models/decoder.py`
**Boundary-Guided Progressive Decoder** — progressively upsamples from the deepest feature (1/32 resolution) back to full resolution using skip connections. Contains:
- `DecoderBlock`: Upsamples deeper features 2×, fuses with same-level skip feature via additive fusion, then applies two conv-BN-ReLU refinement layers.
- `BoundaryGuidedDecoder`: Chains 3 `DecoderBlock`s (f5→d4→d3→d2), produces 4 coarse prediction maps at intermediate resolutions for **deep supervision**, then outputs the final full-resolution segmentation map via a `final_conv` head.

---

### `datasets/` — Data Loading

#### `datasets/cod_dataset.py`
**Primary dataset loader** for standard COD benchmark datasets (COD10K, CAMO, CHAMELEON, NC4K). Expects the folder structure:
```
dataset_root/
  Train/ Images/ *.jpg   GT/ *.png
  Test/  Images/ *.jpg   GT/ *.png
```
Applies training augmentations or test-time-only transforms, pairs each image with its ground-truth mask, and returns normalized PyTorch tensors. Also exposes `get_image_path(idx)` for saving results by filename.

---

#### `datasets/animals_dataset.py`
**Extended dataset loader** for the CAMO Kaggle animals dataset. Supports two flexible layouts:
- **Layout A** (CAMO-style): `animals/Image/` + `animals/GT/`
- **Layout B** (flat): `animals/*.jpg` (image-only, no masks)

Automatically detects which layout is present and operates in paired (image+mask) or image-only mode accordingly. Used by `app.py` for the gallery tab.

---

#### `datasets/transforms.py`
**Data augmentation pipelines** using [albumentations](https://albumentations.ai/). Defines:
- `get_train_transforms(image_size)`: Resize → HorizontalFlip → VerticalFlip → RandomRotate90 → ColorJitter → GaussianBlur → Normalize → ToTensorV2
- `get_test_transforms(image_size)`: Resize → Normalize → ToTensorV2
- `BasicTransform`: Fallback class (resize + horizontal flip only) used when `albumentations` is not installed.

---

### `utils/` — Utilities

#### `utils/losses.py`
**Loss functions** for training CODNet and UNet. Implements:
- `BCEWithLogitsLoss`: Standard binary cross-entropy loss on raw logits.
- `IoULoss`: Soft intersection-over-union loss penalizing mask overlap errors.
- `BoundaryLoss`: Boundary-aware weighted BCE — applies 5× higher loss weight on boundary pixels (extracted via Laplacian kernel + dilation).
- `CODLoss`: Combined loss = `BCE_weight × BCE + IoU_weight × IoU + Boundary_weight × BoundaryLoss`, plus 0.5× deep supervision on coarse predictions and 0.3× boundary map supervision.
- `UNetLoss`: Simplified combined loss (BCE + IoU + Boundary) without deep supervision — designed for UNet which outputs only a single segmentation map.

---

#### `utils/metrics.py`
**Evaluation metric calculator** implementing all standard COD benchmark metrics:
- `MAE`: Mean absolute pixel-wise error between predicted probability map and ground-truth binary mask.
- `S-measure`: Structure similarity combining object-level and region-level scores (Fan et al., 2017).
- `E-measure`: Enhanced alignment measure using adaptive thresholding (Fan et al., 2018).
- `wF-measure`: Weighted F-measure with adaptive thresholding (precision/recall balance).

`MetricCalculator` accumulates per-sample results via `.update(pred, gt)` and returns averaged scores via `.get_results()`.

---

## 📖 References

This work builds on state-of-the-art COD methods including:
- **SINet** (Fan et al., 2020) — Camouflaged Object Detection
- **BGNet** — Boundary-Guided Network for COD
- **CAMO-UNet** — U-Net adapted for camouflage
- **CAMFNet** — Context-aware multi-scale fusion

See `report.pdf` for the full literature review and methodology.

---

## 🛠️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA GPU (6GB VRAM) | NVIDIA RTX 4050 or better |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB (code + checkpoints) | 5 GB (with dataset) |
| Python | 3.9+ | 3.10+ |
| PyTorch | 2.0+ | 2.6+ |

> CPU inference is supported but significantly slower. AMP (automatic mixed precision) is automatically enabled on CUDA for faster training.

---

*Built with PyTorch 2.6 • ResNet50 • U-Net • YOLOv8 (Ultralytics) • Streamlit 1.55 • OUTR Bhubaneswar PhD Research*
