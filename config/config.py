"""
Configuration for Camouflaged Object Detection (COD) Prototype.
All hyperparameters and paths are centralized here.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "runs")

# Dataset paths (update after downloading)
TRAIN_IMAGE_DIR = r"C:\Users\Asutosh\TARUN\Deep Learning based Camouflaged\data\COD10K\Train\Images"
TRAIN_GT_DIR = r"C:\Users\Asutosh\TARUN\Deep Learning based Camouflaged\data\COD10K\Train\GT"
TEST_IMAGE_DIR = r"C:\Users\Asutosh\TARUN\Deep Learning based Camouflaged\data\COD10K\Test\Images"
TEST_GT_DIR = r"C:\Users\Asutosh\TARUN\Deep Learning based Camouflaged\data\COD10K\Test\GT"

# ─── Model ────────────────────────────────────────────────────────────────────
BACKBONE = "resnet50"         # Options: "resnet50", "res2net50"
PRETRAINED = True             # Use ImageNet pretrained backbone
IN_CHANNELS = 3               # RGB input (extend to more for multispectral)
NUM_CLASSES = 1               # Binary segmentation (camouflaged vs background)
CHANNEL_DIM = 64              # Base channel dimension for decoder

# ─── Training ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 352              # Standard COD input size
BATCH_SIZE = 8                # Fits in 6GB VRAM with AMP
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6                # For cosine annealing

# Loss weights
BCE_WEIGHT = 1.0
IOU_WEIGHT = 1.0
BOUNDARY_WEIGHT = 0.5

# ─── Mixed Precision ─────────────────────────────────────────────────────────
USE_AMP = True                # Automatic Mixed Precision for memory savings

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED = 42
SAVE_EVERY = 10               # Save checkpoint every N epochs
PRINT_FREQ = 20               # Print loss every N iterations

# Create directories
for d in [DATA_DIR, CHECKPOINT_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
