"""
visualization_utils.py — Drawing, plotting, and feature map visualization.

Used by:
  - 00_quick_pretrained_yolo_baseline.ipynb
  - 02_bdd100k_preparation.ipynb
  - 04_inference_yolo26.ipynb
  - 05_extract_backbone_features.ipynb
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── Colour palette for BDD100K classes ───────────────────────────────────────
CLASS_COLORS = {
    "person":        (255,  80,  80),   # red
    "rider":         (255, 160,  60),   # orange
    "car":           ( 60, 180, 255),   # blue
    "truck":         (100, 100, 255),   # purple-ish
    "bus":           (255, 220,  60),   # yellow
    "train":         (180, 100, 255),   # violet
    "motorcycle":    (100, 255, 100),   # green
    "bicycle":       (255, 100, 200),   # pink
    "traffic light": (  0, 255, 200),   # cyan
    "traffic sign":  (200, 200,   0),   # olive
}

DEFAULT_COLOR = (200, 200, 200)


def draw_bboxes(
    image: np.ndarray,
    boxes: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image:       BGR numpy array (OpenCV format).
        boxes:       List of dicts with keys: 'class_id', 'x1', 'y1', 'x2', 'y2',
                     and optionally 'confidence', 'class_name'.
        class_names: Optional mapping from class_id → name.
        thickness:   Line thickness.
        font_scale:  Text font scale.

    Returns:
        Image with drawn bounding boxes.
    """
    img = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        # Determine class name
        name = box.get("class_name", "")
        if not name and class_names and "class_id" in box:
            name = class_names.get(box["class_id"], f"cls_{box['class_id']}")

        # Colour
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label text
        conf = box.get("confidence", None)
        label = name
        if conf is not None:
            label += f" {conf:.2f}"

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return img


def visualize_predictions(
    image_path: str,
    results,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 8),
) -> np.ndarray:
    """
    Overlay Ultralytics YOLO prediction results on an image.

    Args:
        image_path: Path to the original image.
        results:    Ultralytics Results object (single image).
        save_path:  If set, save annotated image here.
        show:       Whether to display with matplotlib.
        figsize:    Figure size.

    Returns:
        Annotated image as BGR numpy array.
    """
    # Get the plotted result from Ultralytics
    result = results[0] if isinstance(results, list) else results
    annotated = result.plot()  # BGR numpy array

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"💾 Saved: {save_path}")

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(os.path.basename(image_path))
        plt.tight_layout()
        plt.show()

    return annotated


def visualize_yolo_labels(
    image_path: str,
    label_path: str,
    class_names: Dict[int, str],
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Draw YOLO-format labels (normalised) on the corresponding image.

    Args:
        image_path:  Path to image file.
        label_path:  Path to YOLO .txt label file.
        class_names: Dict mapping class_id → name.
        figsize:     Figure size.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    h, w = img.shape[:2]

    boxes = []
    if os.path.isfile(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = [float(v) for v in parts[1:5]]

                # Convert YOLO normalised → pixel coords
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                boxes.append({
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, f"cls_{cls_id}"),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

    annotated = draw_bboxes(img, boxes, class_names)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{os.path.basename(image_path)} — {len(boxes)} boxes")
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(
    features: Dict[str, "torch.Tensor"],
    max_channels: int = 16,
    figsize_per_map: Tuple[int, int] = (12, 3),
) -> None:
    """
    Visualize a grid of feature map channels for each captured layer.

    Args:
        features:        OrderedDict of {name: tensor} from FeatureExtractor.
        max_channels:    Maximum number of channels to display per layer.
        figsize_per_map: Figure size per feature map row.
    """
    import torch

    for name, feat in features.items():
        if feat.dim() < 4:
            print(f"  Skipping {name}: shape {tuple(feat.shape)} (not a spatial feature map)")
            continue

        # Take first batch element
        feat_2d = feat[0].cpu()  # (C, H, W)
        n_channels = min(feat_2d.shape[0], max_channels)

        cols = min(8, n_channels)
        rows = (n_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_map[0], figsize_per_map[1] * rows))
        fig.suptitle(f"{name}  —  shape: {tuple(feat.shape)}", fontsize=11)

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for i in range(rows):
            for j in range(cols):
                ch_idx = i * cols + j
                if ch_idx < n_channels:
                    axes[i, j].imshow(feat_2d[ch_idx].numpy(), cmap="viridis")
                    axes[i, j].set_title(f"ch {ch_idx}", fontsize=8)
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()


def plot_training_results(results_dir: str, figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Display training curves from Ultralytics' results.csv.

    Args:
        results_dir: Path to the training run directory (contains results.csv).
        figsize:     Figure size.
    """
    import pandas as pd

    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"❌ results.csv not found in {results_dir}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Box loss
    if "train/box_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train/box_loss"], label="train")
        if "val/box_loss" in df.columns:
            axes[0].plot(df["epoch"], df["val/box_loss"], label="val")
        axes[0].set_title("Box Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

    # Class loss
    if "train/cls_loss" in df.columns:
        axes[1].plot(df["epoch"], df["train/cls_loss"], label="train")
        if "val/cls_loss" in df.columns:
            axes[1].plot(df["epoch"], df["val/cls_loss"], label="val")
        axes[1].set_title("Cls Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    # mAP
    if "metrics/mAP50(B)" in df.columns:
        axes[2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    if "metrics/mAP50-95(B)" in df.columns:
        axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[2].set_title("mAP")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle("Training Results", fontsize=13)
    plt.tight_layout()
    plt.show()


def save_side_by_side(
    img_original: np.ndarray,
    img_predicted: np.ndarray,
    save_path: str,
    title: str = "",
) -> None:
    """Save a side-by-side comparison of original and predicted images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB))
    ax2.set_title("Predictions")
    ax2.axis("off")
    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: {save_path}")
