"""
Lane mask rendering utilities.
Migrated from yolo26_pipeline/src/lane_utils.py.
Renders BDD100K poly2d lane annotations into binary segmentation masks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


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


def _poly2d_to_polygon_dicts(poly2d_data):
    polygons = []
    if poly2d_data is None:
        return polygons
    if isinstance(poly2d_data, dict):
        return [poly2d_data]
    if not isinstance(poly2d_data, list) or len(poly2d_data) == 0:
        return polygons

    first = poly2d_data[0]
    if isinstance(first, dict):
        return poly2d_data
    if isinstance(first, (list, tuple)):
        if len(first) >= 2 and isinstance(first[0], (int, float)):
            return [{"vertices": poly2d_data}]
        if len(first) > 0 and isinstance(first[0], (list, tuple)):
            return [{"vertices": p} for p in poly2d_data if isinstance(p, list) and len(p) > 0]
    return polygons


def _normalize_vertices_and_types(poly):
    if isinstance(poly, dict):
        verts = poly.get("vertices", []) or []
        types = poly.get("types", "") or ("L" * len(verts))
    elif isinstance(poly, list):
        verts = poly
        types = "L" * len(verts)
    else:
        return [], ""

    out_verts = []
    out_types = []
    for v in verts:
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            x = float(v[0]); y = float(v[1])
        except Exception:
            continue
        out_verts.append([x, y])
        if len(v) >= 3 and isinstance(v[2], str) and len(v[2]) > 0:
            out_types.append(v[2][0])
        else:
            idx = len(out_verts) - 1
            if idx < len(types) and isinstance(types[idx], str) and len(types[idx]) > 0:
                out_types.append(types[idx][0])
            else:
                out_types.append("L")
    return out_verts, "".join(out_types)


def render_lane_mask(
    labels: List[Dict],
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
) -> np.ndarray:
    """Render lane polylines as a binary mask."""
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    sx = mask_width / img_width
    sy = mask_height / img_height

    for label in labels:
        cat = label.get("category", "")
        if not isinstance(cat, str) or not cat.startswith("lane/"):
            continue

        for poly in _poly2d_to_polygon_dicts(label.get("poly2d")):
            verts, _types = _normalize_vertices_and_types(poly)
            pts = []
            for v in verts:
                x = int(round(v[0] * sx))
                y = int(round(v[1] * sy))
                x = min(max(x, 0), mask_width - 1)
                y = min(max(y, 0), mask_height - 1)
                pts.append((x, y))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.array(pts, dtype=np.int32)], False, 255, line_thickness)

    return mask


def convert_bdd_lanes_to_masks(
    json_path: str,
    output_mask_dir: str,
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
    debug_limit: Optional[int] = None,
) -> Dict[str, int]:
    """Convert all BDD100K lane labels from a JSON file to binary mask PNGs."""
    os.makedirs(output_mask_dir, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)
    if debug_limit is not None:
        data = data[:debug_limit]

    stats = {"total_images": len(data), "images_with_lanes": 0, "total_lane_annotations": 0}
    for frame in tqdm(data, desc="Rendering lane masks"):
        img_name = frame.get("name", "")
        mask_name = Path(img_name).stem + ".png"
        mask_path = os.path.join(output_mask_dir, mask_name)

        labels = frame.get("labels") or []
        lane_labels = [l for l in labels if isinstance(l.get("category", ""), str) and l.get("category", "").startswith("lane/")]

        if lane_labels:
            stats["images_with_lanes"] += 1
            stats["total_lane_annotations"] += len(lane_labels)

        mask = render_lane_mask(lane_labels, mask_width, mask_height, img_width, img_height, line_thickness)
        cv2.imwrite(mask_path, mask)
    return stats


def print_lane_stats(stats: Dict[str, int]) -> None:
    print(f"\n{'='*40}")
    print(" Lane Mask Statistics")
    print(f"{'='*40}")
    print(f" Total images:          {stats['total_images']:,}")
    print(f" Images with lanes:     {stats['images_with_lanes']:,}")
    pct = (stats['images_with_lanes'] / max(1, stats['total_images'])) * 100
    print(f" Lane coverage:         {pct:.1f}%")
    print(f" Total lane annotations:{stats['total_lane_annotations']:,}")
    avg = stats['total_lane_annotations'] / max(1, stats['images_with_lanes'])
    print(f" Avg lanes per image:   {avg:.1f}")
    print(f"{'='*40}")
