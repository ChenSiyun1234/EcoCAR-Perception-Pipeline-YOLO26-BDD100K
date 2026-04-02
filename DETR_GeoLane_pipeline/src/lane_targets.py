"""
BDD100K poly2d → structured lane targets.

Parses the original BDD100K lane annotations (poly2d format) and converts
them into fixed-length ordered point sequences for query-based lane training.

Reference: https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md

Target format per lane:
  - existence: 1.0 (lane present) or 0.0 (padding)
  - points: (N, 2) array of normalized (x, y) coordinates, ordered top→bottom
  - visibility: (N,) boolean mask (1 = visible point)
  - lane_type: integer category ID

Per image:
  - max_lanes x {existence, points, visibility, lane_type}
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import (
    BDD_IMG_W, BDD_IMG_H, LANE_TRAIN_CATS, LANE_CAT_TO_ID,
)


def parse_poly2d(poly2d_field) -> List[np.ndarray]:
    """Extract polyline vertex arrays from BDD100K poly2d field.

    The poly2d field can be:
      - list of dicts: [{"vertices": [[x,y], ...], "types": "LL..."}, ...]
      - list of lists: [[[x,y], [x,y], ...], ...]
      - None or empty
    """
    if poly2d_field is None:
        return []
    polylines = []
    for item in poly2d_field:
        if isinstance(item, dict):
            verts = item.get("vertices", [])
        elif isinstance(item, (list, tuple)):
            if len(item) > 0 and isinstance(item[0], (list, tuple)):
                verts = item
            else:
                continue
        else:
            continue
        if len(verts) >= 2:
            polylines.append(np.array(verts, dtype=np.float64))
    return polylines


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to exactly n evenly-spaced points.

    Points are ordered top-to-bottom (ascending y in image coords).

    Returns:
        points: (n, 2) array
        visibility: (n,) boolean — all True for now (future: clip to image)
    """
    # Order by y ascending (top to bottom)
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()

    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]

    if total < 1e-6:
        # Degenerate polyline
        out = np.tile(pts[0], (n, 1))
        return out, np.ones(n, dtype=bool)

    # Sample at uniform arc-length intervals
    sample_dists = np.linspace(0.0, total, n)
    resampled = np.zeros((n, 2), dtype=np.float64)
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_len, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cum_len[idx]
        seg_len = seg_lens[idx]
        if seg_len < 1e-9:
            resampled[i] = pts[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t

    visibility = np.ones(n, dtype=bool)
    return resampled, visibility


def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    """Convert one frame's BDD100K labels to structured lane targets.

    Args:
        labels: list of label dicts from BDD100K JSON
        max_lanes: maximum number of lane slots
        num_points: points per lane
        img_w, img_h: original image dimensions for normalization

    Returns:
        dict with arrays:
          existence: (max_lanes,) float32
          points:    (max_lanes, num_points, 2) float32, normalized [0,1]
          visibility: (max_lanes, num_points) float32
          lane_type: (max_lanes,) int64
    """
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    lane_idx = 0
    for label in labels:
        if lane_idx >= max_lanes:
            break

        cat = label.get("category", "")
        if cat not in LANE_CAT_TO_ID:
            continue

        poly2d = label.get("poly2d")
        polylines = parse_poly2d(poly2d)
        if not polylines:
            continue

        for pl in polylines:
            if lane_idx >= max_lanes:
                break
            if len(pl) < 2:
                continue

            resampled, vis = resample_polyline(pl, num_points)

            # Normalize to [0, 1]
            resampled[:, 0] /= img_w
            resampled[:, 1] /= img_h
            resampled = np.clip(resampled, 0.0, 1.0)

            existence[lane_idx] = 1.0
            points[lane_idx] = resampled.astype(np.float32)
            visibility[lane_idx] = vis.astype(np.float32)
            lane_type[lane_idx] = LANE_CAT_TO_ID[cat]
            lane_idx += 1

    return {
        "existence": existence,
        "points": points,
        "visibility": visibility,
        "lane_type": lane_type,
    }


class LaneLabelCache:
    """Loads and caches BDD100K lane annotations from consolidated JSON.

    The BDD100K consolidated label file is a JSON list of frame dicts:
    [
      {
        "name": "xxx.jpg",
        "labels": [
          {"category": "lane/single white", "poly2d": [...], ...},
          ...
        ]
      },
      ...
    ]
    """

    def __init__(self, json_path: str, max_lanes: int = 10,
                 num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}

        if json_path and os.path.isfile(json_path):
            print(f"Loading lane labels from {json_path} ...")
            with open(json_path, "r") as f:
                data = json.load(f)
            for frame in data:
                name = frame.get("name", "")
                labels = frame.get("labels") or []
                # Only cache frames that have lane annotations
                lane_labels = [l for l in labels
                               if l.get("category", "").startswith("lane/")]
                if lane_labels:
                    self._cache[name] = lane_labels
            print(f"  Cached lane labels for {len(self._cache)} frames")
        else:
            print(f"  No lane labels found at: {json_path}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        """Get structured lane targets for an image, or None if no lanes."""
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(
            labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
