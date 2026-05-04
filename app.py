"""
=============================================================================
 Camouflaged Object Detection — Streamlit Demo App  (Triple-Model: CODNet + UNet + YOLOv8)
 Deep Learning based COD from Multispectral Images
 OUTR Bhubaneswar – Jashasmita Pal (Roll: 24520007)
=============================================================================

Run:  streamlit run app.py
"""
import os
import sys
import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms.functional as TF

import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COD – Camouflaged Object Detection",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e0e0ff; }

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card h2 { font-size: 2.2rem; margin: 0; color: #a78bfa; }
.metric-card p  { font-size: 0.9rem; color: #94a3b8; margin: 4px 0 0; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #c4b5fd;
    border-left: 4px solid #7c3aed; padding-left: 12px; margin: 20px 0 10px;
}

/* Prediction overlay badge */
.badge-high { background: #16a34a; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.85rem; }
.badge-med  { background: #d97706; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.85rem; }
.badge-low  { background: #64748b; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.85rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(10,10,30,0.85) !important; border-right: 1px solid rgba(255,255,255,0.08); }

/* Upload area */
[data-testid="stFileUploader"] label { color: #c4b5fd !important; }

/* YOLOv8 detection table */
.yolo-badge {
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── Path Setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "codnet_best.pth")
DEMO_DIR = os.path.join(ROOT, "data", "animals")
RESULTS_DIR = os.path.join(ROOT, "results", "streamlit")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── CODNet Model Loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CODNet model…")
def load_codnet():
    """Load CODNet into cache (runs once per session)."""
    from models.cod_net import CODNet
    import config.config as cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CODNet(
        backbone_name=cfg.BACKBONE,
        pretrained=not os.path.exists(CHECKPOINT_PATH),
        channel_dim=cfg.CHANNEL_DIM,
    )

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        trained = True
        epoch = ckpt.get("epoch", 0) + 1
    else:
        trained = False
        epoch = 0

    model.to(device).eval()
    return model, device, trained, epoch


# ─── YOLOv8 Model Loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_yolo(model_variant: str = "yolov8n", confidence: float = 0.25, iou: float = 0.45):
    """Load YOLOv8 detector into cache."""
    from models.yolo_detector import YOLODetector
    return YOLODetector(model_name=model_variant, confidence=confidence, iou_threshold=iou)


# ─── UNet Model Loading ───────────────────────────────────────────────────────
UNET_CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "unet", "unet_best.pth")

@st.cache_resource(show_spinner="Loading UNet model…")
def load_unet():
    """Load UNet into cache (runs once per session)."""
    from models.unet import UNet
    import config.config as cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = getattr(cfg, "UNET_ENCODER", "resnet50")
    model = UNet(
        encoder=encoder,
        pretrained=not os.path.exists(UNET_CHECKPOINT_PATH),
    )

    if os.path.exists(UNET_CHECKPOINT_PATH):
        ckpt = torch.load(UNET_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        trained = True
        epoch = ckpt.get("epoch", 0) + 1
    else:
        trained = False
        epoch = 0

    model.to(device).eval()
    return model, device, trained, epoch


# ─── CODNet Inference ─────────────────────────────────────────────────────────
def preprocess(pil_img, image_size=352):
    """Preprocess PIL Image → model input tensor."""
    img_resized = pil_img.resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(np.array(img_resized))
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


@torch.no_grad()
def run_codnet_inference(model, device, pil_img, image_size=352, threshold=0.5):
    """Run CODNet inference and return (mask_prob, binary_mask, overlay, ms, coverage)."""
    tensor = preprocess(pil_img, image_size).to(device)

    t0 = time.time()
    main_pred, _, _ = model(tensor)
    inference_ms = (time.time() - t0) * 1000

    prob = torch.sigmoid(main_pred).squeeze().cpu().numpy()
    prob_pil = Image.fromarray((prob * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    prob_map = np.array(prob_pil).astype(np.float32) / 255.0

    binary = (prob_map > threshold).astype(np.uint8)

    orig = np.array(pil_img.convert("RGB"))
    overlay = orig.copy()
    green = np.zeros_like(orig)
    green[:, :, 1] = 255
    mask_bool = binary.astype(bool)
    overlay[mask_bool] = (0.45 * green[mask_bool] + 0.55 * overlay[mask_bool]).astype(np.uint8)

    from scipy.ndimage import binary_dilation, binary_erosion
    try:
        dilated = binary_dilation(mask_bool, iterations=2)
        eroded  = binary_erosion(mask_bool, iterations=2)
        contour = dilated & ~eroded
        overlay[contour] = [255, 60, 60]
    except Exception:
        pass

    coverage = float(binary.mean()) * 100
    return prob_map, binary, Image.fromarray(overlay), inference_ms, coverage


def colorize_heatmap(prob_map):
    """Convert probability map to a hot colormap PIL image."""
    import matplotlib.cm as cm
    colored = (cm.hot(prob_map)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


# ─── YOLOv8 Inference ────────────────────────────────────────────────────────
def run_yolo_inference(detector, pil_img):
    """Run YOLOv8 inference. Returns (annotated_img, detections, inference_ms, coverage)."""
    detections, inference_ms = detector.detect(pil_img)
    annotated = detector.draw_detections(pil_img, detections)
    coverage = detector.compute_coverage(pil_img, detections)
    return annotated, detections, inference_ms, coverage


def pil_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦎 COD Settings")
    st.markdown("---")

    # ── Detection mode selector ───────────────────────────────────────────────
    st.markdown("### 🔬 Detection Mode")
    detection_mode = st.radio(
        "Select model",
        options=["CODNet (Segmentation)", "UNet (Segmentation)", "YOLOv8 (Object Detection)"],
        index=0,
        label_visibility="collapsed",
    )
    use_yolo = (detection_mode == "YOLOv8 (Object Detection)")
    use_unet = (detection_mode == "UNet (Segmentation)")

    st.markdown("---")

    # ── Model-specific controls ───────────────────────────────────────────────
    if use_yolo:
        st.markdown("### ⚡ YOLOv8 Settings")
        yolo_variant = st.selectbox(
            "Model Variant",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            index=0,
            help="n=nano(fastest), s=small, m=medium, l=large, x=extra-large",
        )
        yolo_conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
        yolo_iou  = st.slider("NMS IoU Threshold",    0.10, 0.95, 0.45, 0.05)

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        try:
            yolo_detector = load_yolo(yolo_variant, yolo_conf, yolo_iou)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            num_classes = len(yolo_detector.class_names)
            st.success("YOLOv8 loaded ✓")
            st.markdown(f"- **Variant:** {yolo_variant}")
            st.markdown(f"- **Classes:** {num_classes}")
            st.markdown(f"- **Device:** {gpu_name}")
            model_loaded = True
        except Exception as e:
            st.error(f"YOLOv8 error: {e}")
            model_loaded = False
            yolo_detector = None

        # Shared display options
        image_size = 640
        threshold = yolo_conf

    elif use_unet:
        st.markdown("### 🔬 UNet Settings")
        image_size = st.selectbox("Input Resolution", [352, 256, 448], index=0)
        threshold  = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        show_overlay = st.checkbox("Show Overlay", value=True)

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        try:
            model, device, trained, epoch = load_unet()
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            params = sum(p.numel() for p in model.parameters())
            st.success("UNet loaded ✓")
            st.markdown(f"- **Encoder:** ResNet50")
            st.markdown(f"- **Params:** {params/1e6:.1f}M")
            st.markdown(f"- **Device:** {gpu_name}")
            if trained:
                st.markdown(f"- **Trained:** Epoch {epoch}")
            else:
                st.warning("No UNet checkpoint found.\nUsing pretrained encoder only.\nTrain first: `python train.py --model_type unet`")
            model_loaded = True
            yolo_detector = None
        except Exception as e:
            st.error(f"UNet error: {e}")
            model_loaded = False
            yolo_detector = None

    else:
        st.markdown("### 🧠 CODNet Settings")
        image_size = st.selectbox("Input Resolution", [352, 256, 448], index=0)
        threshold  = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        show_overlay = st.checkbox("Show Overlay", value=True)

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        try:
            model, device, trained, epoch = load_codnet()
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            params = sum(p.numel() for p in model.parameters())
            st.success("CODNet loaded ✓")
            st.markdown(f"- **Backbone:** ResNet50")
            st.markdown(f"- **Params:** {params/1e6:.1f}M")
            st.markdown(f"- **Device:** {gpu_name}")
            if trained:
                st.markdown(f"- **Trained:** Epoch {epoch}")
            else:
                st.warning("No checkpoint found.\nUsing pretrained backbone only.\nTrain first: `python train.py`")
            model_loaded = True
            yolo_detector = None
        except Exception as e:
            st.error(f"Model error: {e}")
            model_loaded = False
            yolo_detector = None

    st.markdown("---")
    st.markdown("### 📥 Dataset")
    st.markdown("Download CAMO animal dataset:")
    if st.button("▶ Run Downloader", use_container_width=True):
        st.info("Run in terminal:\n```\npython download_dataset.py\n```")

    animal_imgs = []
    if os.path.isdir(DEMO_DIR):
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            import glob as _glob
            animal_imgs += _glob.glob(os.path.join(DEMO_DIR, "**", ext), recursive=True)
    if animal_imgs:
        st.success(f"{len(animal_imgs)} animal images available")
    else:
        st.info("No animal images yet.\nRun `python download_dataset.py`")

    st.markdown("---")
    st.caption("OUTR Bhubaneswar • PhD Research\nJashasmita Pal — Roll: 24520007")


# ─── Main Header ─────────────────────────────────────────────────────────────
mode_badge = (
    '<span style="background:linear-gradient(90deg,#f59e0b,#ef4444);color:white;'
    'padding:4px 14px;border-radius:20px;font-size:1rem;font-weight:700;">⚡ YOLOv8</span>'
    if use_yolo else
    ('<span style="background:linear-gradient(90deg,#059669,#06b6d4);color:white;'
     'padding:4px 14px;border-radius:20px;font-size:1rem;font-weight:700;">🔬 UNet</span>'
     if use_unet else
     '<span style="background:linear-gradient(90deg,#7c3aed,#06b6d4);color:white;'
     'padding:4px 14px;border-radius:20px;font-size:1rem;font-weight:700;">🧠 CODNet</span>')
)

st.markdown(f"""
<div style='text-align:center; padding: 20px 0 10px;'>
  <h1 style='font-size:2.6rem; font-weight:800;
             background: linear-gradient(90deg, #a78bfa, #60a5fa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🦎 Camouflaged Object Detection
  </h1>
  <p style='color:#94a3b8; font-size:1.05rem;'>
    Active Model: {mode_badge} &bull; CODNet &bull; UNet &bull; YOLOv8 &bull; ResNet50 &bull; CSIM &bull; SGFL
  </p>
  <p style='color:#7c3aed; font-size:0.9rem;'>
    Jashasmita Pal &nbsp;|&nbsp; OUTR Bhubaneswar &nbsp;|&nbsp; PhD Research
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_upload, tab_demo, tab_batch, tab_about = st.tabs([
    "📤 Upload & Detect", "🖼️ Animal Gallery", "📁 Batch Process", "ℹ️ About"
])


# ─── Shared: detection dispatcher ────────────────────────────────────────────
def _run_detection(pil_img):
    """Run the currently selected model on pil_img. Returns a result dict."""
    if use_yolo and yolo_detector:
        annotated, detections, inf_ms, coverage = run_yolo_inference(yolo_detector, pil_img)
        return {"mode": "yolo", "annotated": annotated, "detections": detections,
                "inf_ms": inf_ms, "coverage": coverage}
    else:
        # Both CODNet and UNet use the same segmentation inference path
        mode_label = "unet" if use_unet else "codnet"
        prob_map, binary, overlay, inf_ms, coverage = run_codnet_inference(
            model, device, pil_img, image_size=image_size, threshold=threshold)
        heatmap = colorize_heatmap(prob_map)
        return {"mode": mode_label, "prob_map": prob_map, "binary": binary,
                "overlay": overlay, "heatmap": heatmap, "inf_ms": inf_ms, "coverage": coverage}


def _render_results(pil_img, res, key_prefix=""):
    """Render detection results (works for both CODNet and YOLOv8)."""
    import pandas as pd

    inf_ms   = res["inf_ms"]
    coverage = res["coverage"]

    # Metrics row
    st.markdown("### 📊 Detection Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{inf_ms:.0f}ms</h2><p>Inference Time</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h2>{coverage:.1f}%</h2><p>Area Coverage</p></div>""", unsafe_allow_html=True)
    with col3:
        level = "High" if coverage > 15 else ("Medium" if coverage > 5 else "Low")
        badge = "badge-high" if coverage > 15 else ("badge-med" if coverage > 5 else "badge-low")
        st.markdown(f"""<div class="metric-card">
            <h2><span class="{badge}">{level}</span></h2>
            <p>Confidence Level</p></div>""", unsafe_allow_html=True)
    with col4:
        model_label = res["mode"].upper()
        st.markdown(f"""<div class="metric-card">
            <h2 style="font-size:1.4rem">{model_label}</h2><p>Active Model</p></div>""",
            unsafe_allow_html=True)

    st.markdown("")

    if res["mode"] == "yolo":
        # YOLOv8 layout: original + annotated
        c1, c2 = st.columns(2)
        c1.image(pil_img, caption="Original Image", use_container_width=True)
        c2.image(res["annotated"], caption=f"YOLOv8 Detections ({len(res['detections'])} found)",
                 use_container_width=True)

        # Detection table
        if res["detections"]:
            st.markdown("### 🎯 Detected Objects")
            rows = [{
                "Class":      d["class_name"],
                "Confidence": f"{d['confidence']:.3f}",
                "BBox (x1,y1,x2,y2)": str(d["bbox"]),
            } for d in res["detections"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")

        # Download
        st.markdown("---")
        st.markdown("### 💾 Download")
        st.download_button("⬇ Download Annotated Image",
                           data=pil_to_bytes(res["annotated"]),
                           file_name=f"{key_prefix}_yolo_result.png",
                           mime="image/png", use_container_width=True)

    else:
        # CODNet layout: 4 columns
        col_orig, col_hmap, col_mask, col_overlay = st.columns(4)
        with col_orig:
            st.image(pil_img, caption="Original Image", use_container_width=True)
        with col_hmap:
            if show_heatmap:
                st.image(res["heatmap"], caption="Detection Heatmap (Hot)", use_container_width=True)
        with col_mask:
            st.image(
                Image.fromarray((res["binary"] * 255).astype(np.uint8)),
                caption=f"Binary Mask (t={threshold})", use_container_width=True
            )
        with col_overlay:
            if show_overlay:
                st.image(res["overlay"], caption="Overlay (Green = Detected)", use_container_width=True)

        # Download results
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button("⬇ Download Mask",
                               data=pil_to_bytes(Image.fromarray((res["binary"] * 255).astype(np.uint8))),
                               file_name=f"{key_prefix}_mask.png", mime="image/png",
                               use_container_width=True)
        with dl_col2:
            st.download_button("⬇ Download Heatmap",
                               data=pil_to_bytes(res["heatmap"]),
                               file_name=f"{key_prefix}_heatmap.png", mime="image/png",
                               use_container_width=True)
        with dl_col3:
            st.download_button("⬇ Download Overlay",
                               data=pil_to_bytes(res["overlay"]),
                               file_name=f"{key_prefix}_overlay.png", mime="image/png",
                               use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Detect
# ════════════════════════════════════════════════════════════════════════
with tab_upload:
    mode_str = "YOLOv8 Object Detection" if use_yolo else "CODNet Segmentation"
    st.markdown(f'<div class="section-header">Upload an Image — {mode_str}</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded and model_loaded:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner(f"🔍 Running {mode_str}…"):
            res = _run_detection(pil_img)
        _render_results(pil_img, res, key_prefix=uploaded.name.split('.')[0])

    elif not model_loaded:
        st.error("Model failed to load. Check the sidebar for errors.")
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px; color:#64748b;
                    border:2px dashed rgba(124,58,237,0.4); border-radius:16px; margin-top:20px;'>
          <div style='font-size:4rem;'>🦎</div>
          <h3 style='color:#a78bfa;'>Upload an animal image to detect camouflage</h3>
          <p>Try images of lizards, insects, frogs, or military targets!</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Animal Gallery
# ════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<div class="section-header">Camouflaged Animal Examples (Kaggle CAMO Dataset)</div>',
                unsafe_allow_html=True)

    import glob as _glob

    all_imgs = []
    if os.path.isdir(DEMO_DIR):
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_imgs += _glob.glob(os.path.join(DEMO_DIR, "**", ext), recursive=True)

    if not all_imgs:
        st.info("""
        ### No animal images found yet.
        Download the CAMO dataset to see animal examples here.

        **Steps:**
        1. Set up your Kaggle credentials (see sidebar)
        2. Run in terminal:
           ```
           .\\venv\\Scripts\\activate
           python download_dataset.py
           ```
        """)
    else:
        gal_col1, gal_col2 = st.columns([2, 1])
        with gal_col1:
            max_show = st.slider("Number of images to show", 4, min(50, len(all_imgs)), 12, 4)
        with gal_col2:
            selected_img = st.selectbox(
                "Detect a specific image",
                options=["— select —"] + [os.path.basename(p) for p in all_imgs[:50]],
            )

        import random
        sample = random.sample(all_imgs, min(max_show, len(all_imgs)))

        cols_per_row = 4
        for row_start in range(0, len(sample), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, img_path in enumerate(sample[row_start:row_start + cols_per_row]):
                with cols[idx]:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        st.image(img, caption=os.path.basename(img_path)[:20],
                                 use_container_width=True)
                        if st.button(f"Detect", key=f"det_{row_start}_{idx}"):
                            st.session_state["gallery_image"] = img_path
                    except Exception:
                        pass

        if "gallery_image" in st.session_state or selected_img != "— select —":
            img_to_detect = st.session_state.get("gallery_image")
            if selected_img != "— select —":
                for p in all_imgs:
                    if os.path.basename(p) == selected_img:
                        img_to_detect = p
                        break

            if img_to_detect and model_loaded:
                st.markdown("---")
                st.markdown("### Detection Result")
                pil = Image.open(img_to_detect).convert("RGB")
                mode_str = "YOLOv8" if use_yolo else "CODNet"
                with st.spinner(f"Running {mode_str} detection…"):
                    res = _run_detection(pil)
                _render_results(pil, res, key_prefix=os.path.splitext(os.path.basename(img_to_detect))[0])


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Processing
# ════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-header">Batch Process Multiple Images</div>',
                unsafe_allow_html=True)

    uploaded_batch = st.file_uploader(
        "Upload multiple images (max 20)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_batch and model_loaded:
        if len(uploaded_batch) > 20:
            st.warning("Max 20 images for batch processing. Using first 20.")
            uploaded_batch = uploaded_batch[:20]

        mode_str = "YOLOv8" if use_yolo else ("UNet" if use_unet else "CODNet")
        if st.button(f"🚀 Run Batch Detection ({mode_str})", type="primary", use_container_width=True):
            progress = st.progress(0, text="Processing images…")
            results = []

            for i, upl in enumerate(uploaded_batch):
                pil = Image.open(upl).convert("RGB")
                res = _run_detection(pil)
                n_det = len(res.get("detections", [])) if use_yolo else "—"
                results.append((upl.name, pil, res, n_det))
                progress.progress(
                    (i + 1) / len(uploaded_batch),
                    text=f"Processing {i+1}/{len(uploaded_batch)}: {upl.name}"
                )

            progress.empty()
            st.success(f"Processed {len(results)} images!")

            # Summary table
            st.markdown("### Results Summary")
            import pandas as pd
            if use_yolo:
                df = pd.DataFrame([
                    {"Image": n, "Detections": nd, "Coverage (%)": round(r["coverage"], 2),
                     "Inference (ms)": round(r["inf_ms"], 1)}
                    for n, _, r, nd in results
                ])
            else:
                df = pd.DataFrame([
                    {"Image": n, "Coverage (%)": round(r["coverage"], 2),
                     "Inference (ms)": round(r["inf_ms"], 1)}
                    for n, _, r, nd in results
                ])
            st.dataframe(df, use_container_width=True)

            # Thumbnails
            st.markdown("### Detections")
            for name, orig, res, n_det in results:
                det_label = f"{n_det} objects" if use_yolo else f"Coverage: {res['coverage']:.1f}%"
                with st.expander(f"{name} — {det_label}"):
                    c1, c2 = st.columns(2)
                    c1.image(orig, "Original", use_container_width=True)
                    if use_yolo:
                        c2.image(res["annotated"], "YOLOv8 Detections", use_container_width=True)
                    else:
                        overlay_label = "UNet Overlay" if use_unet else "CODNet Overlay"
                        c2.image(res["overlay"], overlay_label, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:50px; color:#64748b;
                    border:2px dashed rgba(124,58,237,0.4); border-radius:16px;'>
          <div style='font-size:3rem;'>📂</div>
          <p style='color:#a78bfa;'>Upload multiple images to batch process all at once</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## About This Project

    This is a PhD registration seminar prototype for **Deep Learning based Camouflaged Object
    Detection from Multispectral Images** at **OUTR Bhubaneswar**, Odisha.

    **Researcher:** Jashasmita Pal (Roll No: 24520007)
    **Supervisors:** Dr. Jibitesh Mishra, Dr. Asimananda Khandual
    **School:** School of Computer Science

    ---

    ### Triple-Model Architecture
    """)

    arch_col1, arch_col2, arch_col3 = st.columns(3)

    with arch_col1:
        st.markdown("#### 🧠 CODNet (Segmentation)")
        st.markdown("""
        | Stage | Module | Purpose |
        |-------|--------|---------|
        | 1 | **ResNet50 Backbone** | Multi-scale feature extraction |
        | 2 | **CSIM** | Cross-Scale Interaction Module |
        | 3 | **SGFL** | Semantic Guided Feature Learning |
        | 4 | **Decoder** | Boundary-Guided Progressive Decoding |

        **Output:** pixel-level segmentation mask + heatmap
        """)

    with arch_col2:
        st.markdown("#### 🔬 UNet (Segmentation)")
        st.markdown("""
        | Stage | Module | Purpose |
        |-------|--------|---------|
        | Encoder | **ResNet50 / Custom** | 4-stage downsampling |
        | Bottleneck | **Conv Block** | Deepest feature map |
        | Decoder | **UpBlocks + Skip** | 4-stage upsampling |
        | Head | **1×1 Conv** | Single-channel output |

        **Output:** pixel-level segmentation mask + heatmap
        """)

    with arch_col3:
        st.markdown("#### ⚡ YOLOv8 (Object Detection)")
        st.markdown("""
        | Variant | Params | Speed | mAP50 |
        |---------|--------|-------|-------|
        | **yolov8n** | 3.2 M | Fastest | 37.3 |
        | **yolov8s** | 11.2 M | Fast | 44.9 |
        | **yolov8m** | 25.9 M | Medium | 50.2 |
        | **yolov8l** | 43.7 M | Slow | 52.9 |
        | **yolov8x** | 68.2 M | Slowest | 53.9 |

        **Output:** bounding boxes with class labels & confidence
        """)

    st.markdown("---")
    st.markdown("""
    ### Evaluation Metrics

    | Metric | CODNet | UNet | YOLOv8 |
    |--------|--------|------|--------|
    | **MAE** ↓ | Segmentation error | Segmentation error | — |
    | **S-measure** ↑ | Structure similarity | Structure similarity | — |
    | **E-measure** ↑ | Enhanced alignment | Enhanced alignment | — |
    | **mAP50** ↑ | — | — | Detection accuracy |
    | **Inference (ms)** ↓ | Varies by GPU | Varies by GPU | ~10–100 ms |

    ---
    ### References
    Builds on SINet, BGNet, CAMO-UNet, CAMFNet, boundary-guided detection networks,
    and Ultralytics YOLOv8. See `report.pdf` for the full literature review.

    ---
    <div style='text-align:center; color:#64748b; font-size:0.85rem;'>
      Built with PyTorch 2.6 • ResNet50 • U-Net • YOLOv8 (Ultralytics) • Streamlit 1.55
    </div>
    """, unsafe_allow_html=True)
