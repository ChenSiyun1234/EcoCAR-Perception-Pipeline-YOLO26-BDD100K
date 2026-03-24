"""
lane_utils.py — BDD100K lane annotation processing: polyline → binary mask.

BDD100K lane annotations use `poly2d` with vertices and Bezier control points.
This module rasterises them into binary segmentation masks for training.

Lane categories (all merged to foreground for binary segmentation):
  lane/crosswalk, lane/double other, lane/double white, lane/double yellow,
  lane/road curb, lane/single other, lane/single white, lane/single yellow

Used by:
  - 07_prepare_lane_masks.ipynb
  - 08_joint_training.ipynb (via JointBDDDataset)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── BDD100K lane categories ─────────────────────────────────────────────────
BDD_LANE_CATEGORIES = [
    "lane/crosswalk",
    "lane/double other",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
]

# For binary segmentation, all lane categories → class 1 (foreground)
LANE_CAT_TO_ID = {cat: 1 for cat in BDD_LANE_CATEGORIES}


def _bezier_curve(p0, p1, p2, p3, num_points: int = 30) -> List[Tuple[int, int]]:
    """
    Compute points along a cubic Bezier curve defined by 4 control points.

    Args:
        p0, p1, p2, p3: Control points as (x, y) tuples/lists.
        num_points:      Number of points to sample along the curve.

    Returns:
        List of (x, y) integer points.
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        points.append((int(round(x)), int(round(y))))
    return points


def _poly2d_to_points(
    vertices: List[List[float]],
    types: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Convert a BDD100K poly2d annotation (with Bezier control points) to a
    list of (x, y) pixel coordinates.

    Args:
        vertices: List of [x, y] coordinates.
        types:    String of 'L' (line) and 'C' (Bezier control point) characters.
        scale_x:  Horizontal scaling factor (for resizing masks).
        scale_y:  Vertical scaling factor.

    Returns:
        List of (x, y) integer points forming the polyline.
    """
    points = []
    i = 0
    n = len(vertices)

    while i < n:
        vx = vertices[i][0] * scale_x
        vy = vertices[i][1] * scale_y

        if i + 2 < n and i + 1 < len(types) and types[i + 1] == "C":
            # Cubic Bezier: current point + 2 control points + end point
            if i + 3 < n:
                p0 = (vx, vy)
                p1 = (vertices[i + 1][0] * scale_x, vertices[i + 1][1] * scale_y)
                p2 = (vertices[i + 2][0] * scale_x, vertices[i + 2][1] * scale_y)
                p3 = (vertices[i + 3][0] * scale_x, vertices[i + 3][1] * scale_y)
                curve_pts = _bezier_curve(p0, p1, p2, p3)
                points.extend(curve_pts)
                i += 4
            else:
                points.append((int(round(vx)), int(round(vy))))
                i += 1
        else:
            points.append((int(round(vx)), int(round(vy))))
            i += 1

    return points


def render_lane_mask(
    labels: List[Dict],
    mask_width: int = 320,
    mask_height: int = 180,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Render BDD100K lane labels for a single image into a binary mask.

    Args:
        labels:         List of label dicts from BDD100K JSON (for one frame).
        mask_width:     Output mask width (default: 1/4 of BDD image).
        mask_height:    Output mask height.
        img_width:      Original BDD image width.
        img_height:     Original BDD image height.
        line_thickness: Polyline drawing thickness in pixels.

    Returns:
        Binary mask of shape (mask_height, mask_width), dtype uint8, values {0, 255}.
    """
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    scale_x = mask_width / img_width
    scale_y = mask_height / img_height

    for label in labels:
        category = label.get("category", "")
        if category not in LANE_CAT_TO_ID:
            continue

        poly2d_data = label.get("poly2d")
        if not poly2d_data or not isinstance(poly2d_data, list):
            continue

        polygons = []
        if len(poly2d_data) > 0:
            if isinstance(poly2d_data[0], dict):
                # Standard Scalabel: list of dicts
                polygons = poly2d_data
            elif isinstance(poly2d_data[0], list):
                if len(poly2d_data[0]) >= 2 and isinstance(poly2d_data[0][0], (int, float)):
                    # List of points (a single polygon)
                    polygons = [{"vertices": poly2d_data}]
                elif len(poly2d_data[0]) > 0 and isinstance(poly2d_data[0][0], list):
                    # List of lists of points (multiple polygons)
                    polygons = [{"vertices": p} for p in poly2d_data if isinstance(p, list)]

        for poly in polygons:
            vertices = poly.get("vertices", [])
            types = poly.get("types", "L" * len(vertices))

            if len(vertices) < 2:
                continue

            points = _poly2d_to_points(vertices, types, scale_x, scale_y)

            if len(points) >= 2:
                pts_array = np.array(points, dtype=np.int32)
                cv2.polylines(mask, [pts_array], isClosed=False,
                              color=255, thickness=line_thickness)

    return mask


def convert_bdd_lanes_to_masks(
    json_path: str,
    output_mask_dir: str,
    mask_width: int = 320,
    mask_height: int = 180,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 2,
    debug_limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Batch convert BDD100K lane JSON annotations to per-image binary mask PNGs.

    Args:
        json_path:       Path to BDD100K lane annotation JSON.
        output_mask_dir: Directory to write mask PNGs.
        mask_width:      Output mask width.
        mask_height:     Output mask height.
        img_width:       Original image width.
        img_height:      Original image height.
        line_thickness:  Polyline thickness.
        debug_limit:     Process only N images if set.

    Returns:
        Dict with stats: total_images, images_with_lanes, total_lane_annotations.
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if debug_limit is not None:
        data = data[:debug_limit]

    stats = {
        "total_images": len(data),
        "images_with_lanes": 0,
        "total_lane_annotations": 0,
    }

    for frame in tqdm(data, desc="Rendering lane masks"):
        img_name = frame.get("name", "")
        mask_name = Path(img_name).stem + ".png"
        mask_path = os.path.join(output_mask_dir, mask_name)

        labels = frame.get("labels", [])
        if labels is None:
            labels = []

        # Filter to lane labels only
        lane_labels = [l for l in labels if l.get("category", "") in LANE_CAT_TO_ID]

        if lane_labels:
            stats["images_with_lanes"] += 1
            stats["total_lane_annotations"] += len(lane_labels)

        mask = render_lane_mask(
            lane_labels,
            mask_width=mask_width,
            mask_height=mask_height,
            img_width=img_width,
            img_height=img_height,
            line_thickness=line_thickness,
        )

        cv2.imwrite(mask_path, mask)

    return stats


def print_lane_stats(stats: Dict[str, int]) -> None:
    """Pretty-print lane mask conversion statistics."""
    print(f"\n{'='*40}")
    print(f" Lane Mask Statistics")
    print(f"{'='*40}")
    print(f" Total images:          {stats['total_images']:,}")
    print(f" Images with lanes:     {stats['images_with_lanes']:,}")
    pct = (stats['images_with_lanes'] / max(1, stats['total_images'])) * 100
    print(f" Lane coverage:         {pct:.1f}%")
    print(f" Total lane annotations:{stats['total_lane_annotations']:,}")
    avg = stats['total_lane_annotations'] / max(1, stats['images_with_lanes'])
    print(f" Avg lanes per image:   {avg:.1f}")
    print(f"{'='*40}")
