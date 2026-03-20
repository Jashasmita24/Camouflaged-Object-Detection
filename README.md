# Deep Learning based Camouflaged Object Detection (CODNet)

![Camouflaged Object](https://img.shields.io/badge/Task-Camouflaged_Object_Detection-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red) ![License](https://img.shields.io/badge/License-MIT-green)

An advanced deep learning pipeline designed to detect and segment objects that are visually hidden or blended into their backgrounds. This state-of-the-art framework is built with PyTorch and features a custom neural network architecture (**CODNet**) optimized with boundary-guided mapping, cross-scale interaction techniques, and auto-mixed precision (AMP) training.

Developed by **Jashasmita Pal**.

---

## 📖 Abstract / Overview

Camouflaged Object Detection (COD) is a highly challenging computer vision task because the target objects share extremely similar color, texture, and structural patterns with their background surroundings. Unlike generic salient object detection (SOD), COD requires the model to mine subtle boundary disruptions and deep semantic cues.

This project introduces **CODNet**, a novel architecture that mimics the human visual system's "search and identify" mechanism. It first locates the global semantic positioning of the hidden object, and then iteratively refines the local boundary details using boundary-aware loss functions.

---

## 🚀 Architectural Details

The CODNet architecture revolves around a pre-trained **ResNet50** backbone, integrated tightly with custom-engineered modules to tackle the difficult challenge of separating camouflaged foregrounds from matching backgrounds:

### 1. Backbone Extractor (ResNet50)
The standard ResNet50 framework (via `timm` and `torchvision`) is used to extract multi-level representations of the input image. It produces four hierarchical feature maps representing low-level spatial edges up to high-level abstract semantics.

### 2. CSIM (Cross-Scale Interaction Module)
In typical encoder-decoder networks, low-level features are simply concatenated with high-level features. CSIM significantly improves this by actively fusing and multiplying multi-scale features, allowing the network to cross-reference global structural context with high-resolution texture details. It prevents the dilution of boundary gradients in deeper layers.

### 3. SGFL (Semantic Guided Feature Learning)
Deeper layers in the ResNet backbone naturally contain noise from the complex backgrounds. SGFL leverages deep, high-confidence semantic maps to guide the learning of shallower layers. It acts as an attention mechanism, selectively amplifying object-specific features while suppressing the camouflage background noise.

### 4. Boundary-Guided Decoder
Unlike traditional U-Net style decoders that rely solely on Binary Cross-Entropy (BCE) over the entire image region, this custom decoder mathematically isolates the edges of the prediction. By calculating an explicit **Boundary Loss** via Laplacian kernels, it aggressively penalizes structural blurring and forces the network to draw sharp, crisp delineations around the hidden object.

---

## 🧠 Loss Functions & Training Infrastructure

The model is supervised using a joint-loss architecture computed in `utils/losses.py`:
- **Weighted BCE Loss (`BCEWithLogitsLoss`)**: Focuses on pixel-level classification accuracy.
- **IoU Loss (Intersection over Union)**: Focuses on the global shape structure and region completeness.
- **Boundary Loss**: Focuses specifically on the pixels along the object's contour using Max Pooling and Laplacian edge detection.

**Hyperparameters Setup (`config.py`):**
- Epochs: `100` (Default)
- Batch Size: `8`
- Initial Learning Rate: `1e-4`
- Optimizer: `AdamW` with `CosineAnnealingLR` scheduler.
- Augmentations: Albumentations (Color Jitter, Gaussian Blur, Random Flip, Rotate).

---

## 📊 Evaluation Metrics

Instead of simple accuracy percentages, the framework natively evaluates its topological and boundary predictions using sophisticated structural mathematics embedded in `utils/metrics.py`:
1. **MAE (Mean Absolute Error)**: Calculates raw pixel deviations between prediction and ground truth. (Lower is better, perfect is `0.0`).
2. **S-measure (Structure-measure)**: Evaluates region-aware and object-aware structural similarity. Measures global fidelity. (Higher is better, perfect is `1.0`).
3. **E-measure (Enhanced-alignment measure)**: Computes local pixel-matching simultaneously with image-level statistics. (Higher is better, perfect is `1.0`).
4. **wF-measure (Weighted F-measure)**: Strictly penalizes false negatives on the hard boundary lines of hidden objects, correcting conventional F-measure interpolation flaws. (Higher is better, perfect is `1.0`).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jashasmita24/Camouflaged-Object-Detection.git
   cd Camouflaged-Object-Detection
   ```

2. **Install dependencies:**
   Make sure you have Python 3.9+ and `pip` installed.
   ```bash
   pip install -r requirements.txt
   ```
   *Core Dependencies: PyTorch (>=2.0), Torchvision, Timm, Albumentations, OpenCV, Streamlit.*

3. **Download the Dataset:**
   The primary dataset used is **COD10K** (from Kaggle), consisting of highly curated natural camouflaged images sourced from photography databases.
   ```bash
   python download_dataset.py
   ```
   *Make sure you have your `kaggle.json` authentication token placed in your workspace.*

---

## 🏃‍♂️ Usage Guide

### 1. Training the Model
We provide a smart launcher that automatically parses the downloaded dataset, updates your configuration paths, and initiates an Auto-Mixed Precision (AMP) training loop.

```bash
python start_training.py
```
*Alternatively, you can manually run `python train.py --epochs 100 --batch_size 8 --backbone resnet50`.*

**Monitor Training** using TensorBoard (tracks live Loss curves, boundary metrics, and learning rate):
```bash
tensorboard --logdir runs/
```

### 2. Testing and Evaluation
To score your newly trained model on your test dataset using the S, E, and wF measures metrics seamlessly:

```bash
python test.py --checkpoint checkpoints/codnet_best.pth
```

### 3. CLI Inference (Predictions)
To run predictions visually over an entire directory of testing images and extract the explicit foreground masks into an output folder:

```bash
python inference.py --image_dir data/COD10K/Test/Images/ --checkpoint checkpoints/codnet_best.pth
```

### 4. Interactive Web Application (Dashboard)
This project includes a fully integrated **Streamlit** dashboard! You can upload your own custom images directly from your browser and watch the AI neural-network extract the camouflaged object in real-time.

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```bash
├── app.py                  # Streamlit Dashboard UI
├── download_dataset.py     # Automates Kaggle COD10K installation
├── start_training.py       # Training bootstrapper and config injector
├── train.py                # Main PyTorch AMP Training loop
├── test.py                 # Evaluation script (S-measure, E-measure, MAE)
├── inference.py            # CLI visual segmentation script
├── config/
│   └── config.py           # Hyperparameters and folder paths
├── datasets/
│   ├── cod_dataset.py      # Custom Dataset Loader definition
│   └── transforms.py       # Albumentations data-augmentation pipeline
├── models/
│   ├── backbone.py         # ResNet50 wrapper
│   ├── csim.py             # Cross-Scale Interaction Module layer
│   ├── sgfl.py             # Semantic Guided Feature layer
│   ├── decoder.py          # Boundary-focused upsampling decoder
│   └── cod_net.py          # Final unified CODNet Module interface
├── utils/
│   ├── losses.py           # Custom BCE + Boundary Loss algorithms
│   └── metrics.py          # Structural Evaluation math functions
└── runs/                   # TensorBoard logging outputs
```

## 🏷️ Credits & Acknowledgment
This work is inspired by modern advancements in Camouflaged Object Detection networks from prestigious computer vision conferences (CVPR/ICCV). Datasets generated and aggregated for scientific study.
