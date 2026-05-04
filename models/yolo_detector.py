"""
YOLOv8 Detector Wrapper for Camouflaged Object Detection Project.

Wraps the Ultralytics YOLOv8 model to provide a consistent API
that integrates with the existing COD pipeline (app.py, inference.py).

Supported models:
  yolov8n  – nano  (~3.2M params, fastest)
  yolov8s  – small (~11.2M params)
  yolov8m  – medium (~25.9M params)
  yolov8l  – large  (~43.7M params)
  yolov8x  – extra-large (~68.2M params)

Usage:
    from models.yolo_detector import YOLODetector
    detector = YOLODetector(model_name="yolov8n", confidence=0.25)
    detections = detector.detect(pil_image)
    annotated  = detector.draw_detections(pil_image, detections)
"""
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch


# ─── Colour palette for class boxes ──────────────────────────────────────────
_PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112,  31),
    (255, 178, 29),  (207, 210,   49), (72,  249,  10),
    (146, 204,  23), (61,  219,  134), (26,  147,  52),
    (0,  212, 187),  (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100,  115, 255), (0,   24,  236),
    (132,  56, 255), (82,   0, 133),   (203,  56, 255),
    (255,  149, 200),(255,  55, 199),
]

def _class_colour(cls_id: int):
    return _PALETTE[int(cls_id) % len(_PALETTE)]


# ─── Detection dataclass (plain dict for simplicity) ─────────────────────────
def _make_detection(x1, y1, x2, y2, confidence, class_id, class_name):
    return {
        "bbox": (int(x1), int(y1), int(x2), int(y2)),   # pixel coords
        "confidence": float(confidence),
        "class_id": int(class_id),
        "class_name": str(class_name),
    }


# ─── Wrapper class ────────────────────────────────────────────────────────────
class YOLODetector:
    """
    YOLOv8 object detector wrapper.

    Args:
        model_name (str): One of 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l',
                          'yolov8x', or a path to a custom .pt checkpoint.
        confidence  (float): Minimum prediction confidence [0, 1].
        iou_threshold (float): NMS IoU threshold [0, 1].
        image_size  (int): Input image size for inference (default 640).
        device      (str|None): 'cpu', 'cuda', or None for auto-detect.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics>=8.0.0"
            ) from err

        self.model_name   = model_name
        self.confidence   = confidence
        self.iou_threshold = iou_threshold
        self.image_size   = image_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve model path: if not a local file, use model name (auto-download)
        model_path = model_name if os.path.isfile(model_name) else f"{model_name}.pt"
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Class name list from model
        self.class_names = self.model.names  # dict {id: name}

    # ── Core detection ────────────────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, pil_image: Image.Image) -> tuple[list[dict], float]:
        """
        Run YOLOv8 detection on a PIL image.

        Returns:
            detections (list[dict]): Each dict has keys:
                'bbox'       : (x1, y1, x2, y2) in pixels
                'confidence' : float
                'class_id'   : int
                'class_name' : str
            inference_ms (float): Inference latency in milliseconds.
        """
        # Convert PIL → numpy (RGB)
        img_np = np.array(pil_image.convert("RGB"))

        t0 = time.time()
        results = self.model.predict(
            source=img_np,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, str(cls_id))
                detections.append(_make_detection(x1, y1, x2, y2, conf, cls_id, cls_name))

        return detections, inference_ms

    # ── Annotated image ───────────────────────────────────────────────────────
    def draw_detections(
        self,
        pil_image: Image.Image,
        detections: list[dict],
        line_width: int = 3,
        font_size: int = 16,
    ) -> Image.Image:
        """
        Draw bounding boxes + labels on a copy of pil_image.

        Returns:
            annotated PIL image (RGB).
        """
        annotated = pil_image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)

        # Try to load a nicer font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            colour = _class_colour(det["class_id"])
            label  = f'{det["class_name"]} {det["confidence"]:.2f}'

            # Box
            for i in range(line_width):
                draw.rectangle(
                    [x1 - i, y1 - i, x2 + i, y2 + i],
                    outline=colour,
                )

            # Label background
            try:
                bbox_text = font.getbbox(label)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)

            label_top = y1 - th - 6
            if label_top < 0:
                label_top = y1 + 2

            draw.rectangle(
                [x1, label_top, x1 + tw + 6, label_top + th + 4],
                fill=colour,
            )
            draw.text(
                (x1 + 3, label_top + 2),
                label,
                fill="white",
                font=font,
            )

        return annotated

    # ─── Coverage estimate (fraction of image covered by boxes) ──────────────
    def compute_coverage(self, pil_image: Image.Image, detections: list[dict]) -> float:
        """Return % of image area covered by detected bounding boxes (rough)."""
        W, H = pil_image.size
        mask = np.zeros((H, W), dtype=np.uint8)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
        return float(mask.mean()) * 100

    # ─── Repr ─────────────────────────────────────────────────────────────────
    def __repr__(self):
        return (
            f"YOLODetector(model={self.model_name!r}, "
            f"conf={self.confidence}, iou={self.iou_threshold}, "
            f"device={self.device!r})"
        )


# ─── Convenience loader (for @st.cache_resource use) ─────────────────────────
def load_yolo_model(
    model_name: str = "yolov8n",
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
) -> YOLODetector:
    """
    Load and return a YOLODetector instance.
    Intended for use with @st.cache_resource in the Streamlit app.
    """
    return YOLODetector(
        model_name=model_name,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )
