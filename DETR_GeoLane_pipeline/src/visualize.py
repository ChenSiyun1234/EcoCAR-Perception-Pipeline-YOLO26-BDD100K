"""
Visualization utilities for dual-path model outputs.

Draws:
  - vehicle detection boxes with class labels
  - lane polylines as colored curves on the image
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional

from .config import VEHICLE_CLASSES, EXPANDED_CLASSES, LANE_TRAIN_CATS
from .losses import box_cxcywh_to_xyxy

# Color palette for vehicle classes
DET_COLORS = [
    (60, 180, 255),   # car - blue
    (100, 100, 255),  # truck - purple
    (255, 220, 60),   # bus - yellow
    (100, 255, 100),  # motorcycle - green
    (255, 100, 200),  # bicycle - pink
    (255, 180, 60),   # train - orange
    (180, 255, 180),  # rider - light green
]

# Color palette for lane types
LANE_COLORS = [
    (255, 255, 255),  # single white
    (255, 255, 0),    # single yellow
    (200, 200, 200),  # single other
    (255, 200, 200),  # double white
    (255, 200, 0),    # double yellow
    (150, 150, 150),  # double other
    (0, 200, 200),    # road curb
]


def draw_detections(img: np.ndarray, pred_logits: torch.Tensor,
                    pred_boxes: torch.Tensor, conf_thresh: float = 0.3,
                    use_expanded: bool = False) -> np.ndarray:
    """Draw detection boxes on image.

    Args:
        img: (H, W, 3) RGB uint8
        pred_logits: (Q, C+1) raw logits
        pred_boxes: (Q, 4) normalized cxcywh
        conf_thresh: minimum confidence
    """
    vis = img.copy()
    h, w = vis.shape[:2]
    classes = EXPANDED_CLASSES if use_expanded else VEHICLE_CLASSES

    scores, labels = pred_logits[:, :-1].max(dim=-1)
    scores = scores.sigmoid()
    keep = scores > conf_thresh

    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[keep]).cpu().numpy()
    scores_k = scores[keep].cpu().numpy()
    labels_k = labels[keep].cpu().numpy()

    for box, score, cls_id in zip(boxes_xyxy, scores_k, labels_k):
        x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
        color = DET_COLORS[cls_id % len(DET_COLORS)]
        label = f"{classes[cls_id]} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_lanes(img: np.ndarray, pred_points: torch.Tensor,
               pred_exist: torch.Tensor, pred_types: Optional[torch.Tensor] = None,
               exist_thresh: float = 0.5, thickness: int = 3) -> np.ndarray:
    """Draw predicted lane polylines on image.

    Args:
        img: (H, W, 3) RGB uint8
        pred_points: (Q, N, 2) normalized coords
        pred_exist: (Q, 1) logits
        pred_types: (Q, T) logits, optional
        exist_thresh: confidence threshold
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    exist_prob = pred_exist[:, 0].sigmoid()
    keep = exist_prob > exist_thresh

    points = pred_points[keep].cpu().numpy()
    if pred_types is not None:
        types = pred_types[keep].argmax(dim=-1).cpu().numpy()
    else:
        types = [0] * len(points)

    for pts, lt in zip(points, types):
        # Scale to image coordinates
        px = (pts[:, 0] * w).astype(int)
        py = (pts[:, 1] * h).astype(int)
        coords = np.stack([px, py], axis=-1)

        color = LANE_COLORS[lt % len(LANE_COLORS)]
        for i in range(len(coords) - 1):
            cv2.line(vis, tuple(coords[i]), tuple(coords[i + 1]), color, thickness)

    return vis


def draw_all(img: np.ndarray, outputs: dict, conf_thresh: float = 0.3,
             lane_thresh: float = 0.5, use_expanded: bool = False) -> np.ndarray:
    """Draw both detections and lanes on image."""
    vis = draw_detections(
        img, outputs["det_pred_logits"][0], outputs["det_pred_boxes"][0],
        conf_thresh, use_expanded)
    vis = draw_lanes(
        vis, outputs["lane_pred_points"][0], outputs["lane_exist_logits"][0],
        outputs.get("lane_type_logits", [None])[0] if "lane_type_logits" in outputs else None,
        lane_thresh)
    return vis
