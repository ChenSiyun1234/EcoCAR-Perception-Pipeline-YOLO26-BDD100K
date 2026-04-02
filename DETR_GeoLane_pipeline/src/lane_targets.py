
"""
BDD100K poly2d → structured lane targets.

Robust parser for official BDD100K lane annotations. Supports:
- per-image JSON files under 100k/train|val|test/*.json
- consolidated JSON files (list of frame dicts)
- old format with frames/objects
- official lane category format: category == "lane" with attributes.laneType(s)
- legacy custom category format: category == "lane/single white", etc.
- poly2d with straight-line and cubic/quadratic Bezier control points via types string
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID


def _normalize_lane_category(label: dict) -> Optional[str]:
    """Return canonical training category like 'lane/single white'."""
    cat = str(label.get("category", "") or "").strip().lower()
    attrs = label.get("attributes") or {}

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)]

    lane_types = []
    for key in ["laneType", "laneTypes", "type", "types"]:
        lane_types.extend(_as_list(attrs.get(key)))
    lane_types = [t.strip().lower() for t in lane_types if t and str(t).strip()]

    if cat.startswith("lane/"):
        norm = cat
    elif cat == "lane":
        if not lane_types:
            return None
        norm = f"lane/{lane_types[0]}"
    else:
        # tolerate labels where category itself is just the lane subtype
        if cat in {"crosswalk", "double other", "double white", "double yellow",
                   "road curb", "single other", "single white", "single yellow"}:
            norm = f"lane/{cat}"
        else:
            return None

    return norm if norm in LANE_CAT_TO_ID else None


def _iter_record_labels(record: dict) -> Iterable[dict]:
    """Yield all label/object dicts from new or old official BDD100K schemas."""
    if not isinstance(record, dict):
        return []
    if isinstance(record.get("labels"), list):
        return record.get("labels") or []
    frames = record.get("frames")
    if isinstance(frames, list):
        out = []
        for fr in frames:
            if isinstance(fr, dict):
                objs = fr.get("objects") or fr.get("labels") or []
                if isinstance(objs, list):
                    out.extend(objs)
        return out
    if isinstance(record.get("objects"), list):
        return record.get("objects") or []
    return []


def _record_image_name(record: dict, fallback_json_path: Optional[str] = None) -> str:
    name = record.get("name") or record.get("image") or record.get("imageName") or ""
    if isinstance(name, str) and name.strip():
        return os.path.basename(name)
    if fallback_json_path:
        return Path(fallback_json_path).stem + ".jpg"
    return ""


def _normalize_poly2d_items(poly2d_field) -> List[dict]:
    if poly2d_field is None:
        return []
    if isinstance(poly2d_field, dict):
        return [poly2d_field]
    if not isinstance(poly2d_field, list):
        return []
    items = []
    for item in poly2d_field:
        if isinstance(item, dict):
            items.append(item)
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
            items.append({"vertices": item, "types": "L" * len(item), "closed": False})
    return items


def _coerce_vertices_and_types(poly: dict) -> Tuple[np.ndarray, str, bool]:
    verts = poly.get("vertices") or []
    types = poly.get("types") or ""
    closed = bool(poly.get("closed", False))
    out = []
    for v in verts:
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            x = float(v[0]); y = float(v[1])
        except Exception:
            continue
        out.append([x, y])
    pts = np.asarray(out, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64), "", closed
    if not isinstance(types, str) or len(types) != len(pts):
        types = "L" * len(pts)
    return pts, types, closed


def _sample_quad(p0, p1, p2, n=16):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _sample_cubic(p0, p1, p2, p3, n=20):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 +
            3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _poly2d_segment_to_points(poly: dict, bezier_samples: int = 20) -> np.ndarray:
    """Convert one official poly2d segment into dense points.

    types string semantics follow BDD100K official docs:
    - 'L': anchor/line vertex
    - 'C': Bezier control point
    Typical cubic pattern: L C C L
    We also tolerate quadratic: L C L
    """
    verts, types, closed = _coerce_vertices_and_types(poly)
    if len(verts) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    dense = [verts[0].copy()]
    i = 1
    while i < len(verts):
        cur = dense[-1]
        t = types[i] if i < len(types) else 'L'
        if t == 'L':
            dense.append(verts[i].copy())
            i += 1
            continue

        # Try cubic: current -> C -> C -> L
        if (i + 2 < len(verts) and types[i] == 'C' and types[i + 1] == 'C'
                and types[i + 2] == 'L'):
            pts = _sample_cubic(cur, verts[i], verts[i + 1], verts[i + 2], bezier_samples)
            dense.extend(pts[1:])
            i += 3
            continue

        # Try quadratic: current -> C -> L
        if i + 1 < len(verts) and types[i] == 'C' and types[i + 1] == 'L':
            pts = _sample_quad(cur, verts[i], verts[i + 1], max(8, bezier_samples // 2))
            dense.extend(pts[1:])
            i += 2
            continue

        # Fallback: malformed control point sequence, connect straight
        dense.append(verts[i].copy())
        i += 1

    if closed and len(dense) > 2:
        if np.linalg.norm(np.asarray(dense[0]) - np.asarray(dense[-1])) > 1e-6:
            dense.append(np.asarray(dense[0]).copy())
    return np.asarray(dense, dtype=np.float64)


def parse_poly2d(poly2d_field, bezier_samples: int = 20) -> List[np.ndarray]:
    polylines: List[np.ndarray] = []
    for poly in _normalize_poly2d_items(poly2d_field):
        pts = _poly2d_segment_to_points(poly, bezier_samples=bezier_samples)
        if len(pts) >= 2:
            # drop exact duplicates while preserving order
            keep = [0]
            for i in range(1, len(pts)):
                if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
                    keep.append(i)
            pts = pts[keep]
            if len(pts) >= 2:
                polylines.append(pts)
    return polylines


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # Re-orient top->bottom for consistency
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]

    if total < 1e-6:
        out = np.tile(pts[0], (n, 1))
        return out, np.zeros(n, dtype=bool)

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

    visibility = ((resampled[:, 0] >= 0.0) & (resampled[:, 0] <= BDD_IMG_W - 1) &
                  (resampled[:, 1] >= 0.0) & (resampled[:, 1] <= BDD_IMG_H - 1))
    return resampled, visibility


def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    lane_items: List[Tuple[str, np.ndarray]] = []
    for label in labels:
        norm_cat = _normalize_lane_category(label)
        if norm_cat is None:
            continue
        for pl in parse_poly2d(label.get("poly2d")):
            if len(pl) < 2:
                continue
            # reject tiny or degenerate shapes
            span_y = float(pl[:, 1].max() - pl[:, 1].min())
            span_x = float(pl[:, 0].max() - pl[:, 0].min())
            arc = float(np.sqrt(((np.diff(pl, axis=0)) ** 2).sum(axis=1)).sum())
            if arc < 2.0:
                continue
            lane_items.append((norm_cat, pl))

    # Keep most informative lanes first (long vertical extent, then arc length)
    lane_items.sort(key=lambda x: ((x[1][:, 1].max() - x[1][:, 1].min()),
                                   np.sqrt(((np.diff(x[1], axis=0)) ** 2).sum(axis=1)).sum()), reverse=True)

    lane_idx = 0
    for norm_cat, pl in lane_items:
        if lane_idx >= max_lanes:
            break
        resampled, vis = resample_polyline(pl, num_points)
        # Normalize to [0,1] but preserve visibility information.
        norm = resampled.copy()
        norm[:, 0] /= float(img_w)
        norm[:, 1] /= float(img_h)
        existence[lane_idx] = 1.0
        points[lane_idx] = np.clip(norm, 0.0, 1.0).astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[norm_cat]
        lane_idx += 1

    return {
        "existence": existence,
        "points": points,
        "visibility": visibility,
        "lane_type": lane_type,
    }


class LaneLabelCache:
    """Loads lane annotations from a directory of per-image JSONs or from a JSON file."""

    def __init__(self, source_path: str, max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}
        self._debug_samples: List[dict] = []

        if not source_path:
            print("  No lane label source path provided")
            return
        if os.path.isdir(source_path):
            self._load_from_directory(source_path)
        elif os.path.isfile(source_path):
            self._load_from_file(source_path)
        else:
            print(f"  No lane labels found at: {source_path}")

    def _cache_record(self, record: dict, fallback_path: Optional[str] = None):
        labels = list(_iter_record_labels(record))
        name = _record_image_name(record, fallback_path)
        if not name or not labels:
            return
        targets = frame_to_lane_targets(labels, self.max_lanes, self.num_points)
        if float(targets["existence"].sum()) > 0:
            self._cache[name] = labels
            if len(self._debug_samples) < 3:
                self._debug_samples.append({
                    "name": name,
                    "n_labels": len(labels),
                    "existence_sum": float(targets["existence"].sum()),
                    "first_label": labels[0],
                })

    def _load_from_directory(self, dir_path: str):
        print(f"Loading lane labels from per-image directory {dir_path} ...")
        json_files = sorted([str(p) for p in Path(dir_path).glob("*.json")])
        for jpath in json_files:
            try:
                with open(jpath, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        self._cache_record(rec, jpath)
            elif isinstance(data, dict):
                self._cache_record(data, jpath)
        print(f"  Cached lane labels for {len(self._cache)} frames from directory")
        if len(self._cache) == 0 and json_files:
            self._print_debug_sample(json_files[0])

    def _load_from_file(self, json_path: str):
        print(f"Loading lane labels from {json_path} ...")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to load lane labels: {e}")
            return
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    self._cache_record(rec, json_path)
        elif isinstance(data, dict):
            self._cache_record(data, json_path)
        print(f"  Cached lane labels for {len(self._cache)} frames")
        if len(self._cache) == 0:
            self._print_debug_payload(data, json_path)

    def _print_debug_sample(self, sample_path: str):
        try:
            with open(sample_path, "r") as f:
                data = json.load(f)
            self._print_debug_payload(data, sample_path)
        except Exception as e:
            print(f"  Could not inspect sample JSON {sample_path}: {e}")

    def _print_debug_payload(self, data, src: str):
        print(f"  Debugging lane source: {src}")
        print(f"  top-level type: {type(data)}")
        rec = None
        if isinstance(data, list) and data:
            rec = data[0]
        elif isinstance(data, dict):
            rec = data
        if isinstance(rec, dict):
            print(f"  record keys: {list(rec.keys())[:20]}")
            labels = list(_iter_record_labels(rec))
            print(f"  extracted label count: {len(labels)}")
            if labels:
                lab = labels[0]
                print(f"  first label category: {lab.get('category')}")
                print(f"  first label attr keys: {list((lab.get('attributes') or {}).keys())}")
                print(f"  has poly2d: {'poly2d' in lab}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
