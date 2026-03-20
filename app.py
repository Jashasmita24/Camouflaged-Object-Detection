"""
=============================================================================
 Camouflaged Object Detection — Streamlit Demo App
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
</style>
""", unsafe_allow_html=True)

# ─── Path Setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "codnet_best.pth")
DEMO_DIR = os.path.join(ROOT, "data", "animals")
RESULTS_DIR = os.path.join(ROOT, "results", "streamlit")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CODNet model…")
def load_model():
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


# ─── Inference ────────────────────────────────────────────────────────────────
def preprocess(pil_img, image_size=352):
    """Preprocess PIL Image → model input tensor."""
    img_resized = pil_img.resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(np.array(img_resized))
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


@torch.no_grad()
def run_inference(model, device, pil_img, image_size=352, threshold=0.5):
    """Run model inference and return (mask_prob, binary_mask, overlay)."""
    tensor = preprocess(pil_img, image_size).to(device)

    t0 = time.time()
    main_pred, _, _ = model(tensor)
    inference_ms = (time.time() - t0) * 1000

    # Prob map at original size
    prob = torch.sigmoid(main_pred).squeeze().cpu().numpy()
    prob_pil = Image.fromarray((prob * 255).astype(np.uint8)).resize(
        pil_img.size, Image.BILINEAR
    )
    prob_map = np.array(prob_pil).astype(np.float32) / 255.0

    # Binary mask
    binary = (prob_map > threshold).astype(np.uint8)

    # Overlay (green highlight)
    orig = np.array(pil_img.convert("RGB"))
    overlay = orig.copy()
    green = np.zeros_like(orig)
    green[:, :, 1] = 255
    mask_bool = binary.astype(bool)
    overlay[mask_bool] = (0.45 * green[mask_bool] + 0.55 * overlay[mask_bool]).astype(np.uint8)

    # Red contour
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


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦎 COD Settings")
    st.markdown("---")

    image_size = st.selectbox("Input Resolution", [352, 256, 448], index=0)
    threshold  = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    show_heatmap = st.checkbox("Show Heatmap", value=True)
    show_overlay = st.checkbox("Show Overlay", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    try:
        model, device, trained, epoch = load_model()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        params = sum(p.numel() for p in model.parameters())
        st.success(f"Model loaded ✓")
        st.markdown(f"- **Backbone:** ResNet50")
        st.markdown(f"- **Params:** {params/1e6:.1f}M")
        st.markdown(f"- **Device:** {gpu_name}")
        if trained:
            st.markdown(f"- **Trained:** Epoch {epoch}")
        else:
            st.warning("No checkpoint found.\nUsing pretrained backbone only.\nTrain first: `python train.py`")
        model_loaded = True
    except Exception as e:
        st.error(f"Model error: {e}")
        model_loaded = False

    st.markdown("---")
    st.markdown("### 📥 Dataset")
    st.markdown("Download CAMO animal dataset:")
    if st.button("▶ Run Downloader", use_container_width=True):
        st.info("Run in terminal:\n```\npython download_dataset.py\n```")

    # Count available demo images
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
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px;'>
  <h1 style='font-size:2.6rem; font-weight:800;
             background: linear-gradient(90deg, #a78bfa, #60a5fa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🦎 Camouflaged Object Detection
  </h1>
  <p style='color:#94a3b8; font-size:1.05rem;'>
    Deep Learning based COD from Animals Dataset &bull;
    Backbone: ResNet50 &bull; CSIM &bull; SGFL &bull; Boundary-Guided Decoder
  </p>
  <p style='color:#7c3aed; font-size:0.9rem;'>
    Jashasmita Pal &nbsp;|&nbsp; OUTR Bhubaneswar &nbsp;|&nbsp; PhD Research
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_demo, tab_batch, tab_about = st.tabs([
    "📤 Upload & Detect", "🖼️ Animal Gallery", "📁 Batch Process", "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Detect
# ════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown('<div class="section-header">Upload a Camouflaged Animal Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded and model_loaded:
        pil_img = Image.open(uploaded).convert("RGB")

        with st.spinner("🔍 Detecting camouflaged object…"):
            prob_map, binary_mask, overlay_img, inf_ms, coverage = run_inference(
                model, device, pil_img, image_size=image_size, threshold=threshold
            )
            heatmap_img = colorize_heatmap(prob_map)

        # Metrics row
        st.markdown("### 📊 Detection Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h2>{inf_ms:.0f}ms</h2><p>Inference Time</p></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h2>{coverage:.1f}%</h2><p>Region Coverage</p></div>""",
                unsafe_allow_html=True)
        with col3:
            level = "High" if coverage > 15 else ("Medium" if coverage > 5 else "Low")
            badge = "badge-high" if coverage > 15 else ("badge-med" if coverage > 5 else "badge-low")
            st.markdown(f"""<div class="metric-card">
                <h2><span class="{badge}">{level}</span></h2>
                <p>Camouflage Confidence</p></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <h2>{image_size}px</h2><p>Processing Size</p></div>""",
                unsafe_allow_html=True)

        st.markdown("")

        # Image grid
        col_orig, col_hmap, col_mask, col_overlay = st.columns(4)
        with col_orig:
            st.image(pil_img, caption="Original Image", use_container_width=True)
        with col_hmap:
            if show_heatmap:
                st.image(heatmap_img, caption="Detection Heatmap (Hot)", use_container_width=True)
        with col_mask:
            st.image(
                Image.fromarray((binary_mask * 255).astype(np.uint8)),
                caption=f"Binary Mask (t={threshold})", use_container_width=True
            )
        with col_overlay:
            if show_overlay:
                st.image(overlay_img, caption="Overlay (Green = Detected)", use_container_width=True)

        # Download results
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        def pil_to_bytes(img, fmt="PNG"):
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            return buf.getvalue()

        with dl_col1:
            st.download_button(
                "⬇ Download Mask",
                data=pil_to_bytes(Image.fromarray((binary_mask * 255).astype(np.uint8))),
                file_name=f"{uploaded.name.split('.')[0]}_mask.png",
                mime="image/png", use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "⬇ Download Heatmap",
                data=pil_to_bytes(heatmap_img),
                file_name=f"{uploaded.name.split('.')[0]}_heatmap.png",
                mime="image/png", use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                "⬇ Download Overlay",
                data=pil_to_bytes(overlay_img),
                file_name=f"{uploaded.name.split('.')[0]}_overlay.png",
                mime="image/png", use_container_width=True,
            )

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
# TAB 2 — Animal Gallery (from Kaggle dataset)
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
        # Gallery controls
        gal_col1, gal_col2 = st.columns([2, 1])
        with gal_col1:
            max_show = st.slider("Number of images to show", 4, min(50, len(all_imgs)), 12, 4)
        with gal_col2:
            selected_img = st.selectbox(
                "Detect a specific image",
                options=["— select —"] + [os.path.basename(p) for p in all_imgs[:50]],
            )

        # Show gallery
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

        # Run detection on selected gallery image
        if "gallery_image" in st.session_state or (
            selected_img != "— select —"
        ):
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
                with st.spinner("Running detection…"):
                    prob, binary, overlay, inf_ms, cov = run_inference(
                        model, device, pil, image_size=image_size, threshold=threshold
                    )
                    hmap = colorize_heatmap(prob)

                r1, r2, r3, r4 = st.columns(4)
                r1.image(pil, caption="Original", use_container_width=True)
                r2.image(hmap, caption="Heatmap", use_container_width=True)
                r3.image(Image.fromarray((binary*255).astype(np.uint8)),
                         caption="Binary Mask", use_container_width=True)
                r4.image(overlay, caption="Overlay", use_container_width=True)
                st.caption(f"Inference: {inf_ms:.0f}ms | Coverage: {cov:.1f}%")


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

        if st.button("🚀 Run Batch Detection", type="primary", use_container_width=True):
            progress = st.progress(0, text="Processing images…")
            results = []

            for i, upl in enumerate(uploaded_batch):
                pil = Image.open(upl).convert("RGB")
                prob, binary, overlay, inf_ms, cov = run_inference(
                    model, device, pil, image_size=image_size, threshold=threshold
                )
                results.append((upl.name, pil, overlay, cov, inf_ms))
                progress.progress((i + 1) / len(uploaded_batch),
                                  text=f"Processing {i+1}/{len(uploaded_batch)}: {upl.name}")

            progress.empty()
            st.success(f"Processed {len(results)} images!")

            # Show summary table
            st.markdown("### Results Summary")
            import pandas as pd
            df = pd.DataFrame([
                {"Image": n, "Coverage (%)": round(c, 2), "Inference (ms)": round(t, 1)}
                for n, _, _, c, t in results
            ])
            st.dataframe(df, use_container_width=True)

            # Show thumbnails
            st.markdown("### Detections")
            for name, orig, overlay, cov, inf_ms in results:
                with st.expander(f"{name} — Coverage: {cov:.1f}%"):
                    c1, c2 = st.columns(2)
                    c1.image(orig, "Original", use_container_width=True)
                    c2.image(overlay, "Detected Overlay", use_container_width=True)
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

    ### Model Architecture (CNN-based)
    """)

    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        | Stage | Module | Purpose |
        |-------|--------|---------|
        | 1 | **ResNet50 Backbone** | Multi-scale feature extraction |
        | 2 | **CSIM** | Cross-Scale Interaction Module |
        | 3 | **SGFL** | Semantic Guided Feature Learning |
        | 4 | **Decoder** | Boundary-Guided Progressive Decoding |
        """)

    with arch_col2:
        st.markdown("""
        **Key Features:**
        - 25.6M parameters (efficient)
        - Deep supervision at 5 scales
        - Boundary-aware loss function
        - Mixed precision (AMP) training
        - Cosine annealing LR schedule

        **Training Dataset:** CAMO (Kaggle)
        animals: frogs, lizards, insects, fish…
        """)

    st.markdown("---")
    st.markdown("""
    ### Evaluation Metrics
    Standard COD benchmark metrics used in the research community:

    | Metric | Description | Better |
    |--------|-------------|--------|
    | **MAE** | Mean Absolute Error (lower = better) | ↓ |
    | **S-measure** | Structure similarity | ↑ |
    | **E-measure** | Enhanced alignment | ↑ |
    | **wF-measure** | Weighted F-measure | ↑ |

    ---
    ### References
    The work builds on state-of-the-art COD methods including SINet, BGNet, CAMO-UNet,
    CAMFNet, and boundary-guided detection networks. See `report.pdf` for the full literature review.

    ---
    <div style='text-align:center; color:#64748b; font-size:0.85rem;'>
      Built with PyTorch 2.6 • ResNet50 • Streamlit 1.55 • RTX 4050 GPU
    </div>
    """, unsafe_allow_html=True)
