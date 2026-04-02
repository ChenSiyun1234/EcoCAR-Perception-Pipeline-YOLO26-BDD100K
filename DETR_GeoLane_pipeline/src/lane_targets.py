"""
BDD100K poly2d -> structured lane targets.

This parser is intentionally defensive:
- supports per-image JSON directories like 100k/train/*.json
- supports consolidated JSON files
- supports official `labels`, old `frames/objects`, and unknown nested layouts
- recursively searches for lane-like dicts that contain poly2d
- expands BDD poly2d segments (L/C types) into dense point sequences
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID

KNOWN_LANE_SUBTYPES = {
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow",
}

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]

def _iter_possible_labels(record: Any) -> Iterable[dict]:
    if isinstance(record, dict):
        if isinstance(record.get("labels"), list):
            for x in record["labels"]:
                if isinstance(x, dict):
                    yield x
        if isinstance(record.get("objects"), list):
            for x in record["objects"]:
                if isinstance(x, dict):
                    yield x
        frames = record.get("frames")
        if isinstance(frames, list):
            for fr in frames:
                if isinstance(fr, dict):
                    for x in _iter_possible_labels(fr):
                        yield x

def _record_image_name(record: dict, fallback_json_path: Optional[str] = None) -> str:
    for key in ["name", "image", "imageName", "filename", "id"]:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v)
            if "." not in base:
                base += ".jpg"
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + ".jpg"
    return ""

def _normalize_lane_category(label: dict) -> Optional[str]:
    cat = str(label.get("category", "") or "").strip().lower()
    attrs = label.get("attributes") or {}

    lane_types = []
    for key in ["laneType", "laneTypes", "type", "types", "style", "styles", "lane_type"]:
        lane_types.extend(_as_list(attrs.get(key)))
    if "subtype" in label:
        lane_types.extend(_as_list(label.get("subtype")))
    lane_types = [t.strip().lower() for t in lane_types if str(t).strip()]

    norm = None
    if cat.startswith("lane/"):
        norm = cat
    elif cat == "lane":
        if lane_types:
            norm = f"lane/{lane_types[0]}"
    elif cat in KNOWN_LANE_SUBTYPES:
        norm = f"lane/{cat}"

    if norm is None:
        text_bits = [cat]
        for _, v in attrs.items():
            text_bits.extend(_as_list(v))
        joined = " ".join(x.lower() for x in text_bits)
        for subtype in KNOWN_LANE_SUBTYPES:
            if subtype in joined:
                norm = f"lane/{subtype}"
                break

    return norm if norm in LANE_CAT_TO_ID else None

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
    out_types = []
    for i, v in enumerate(verts):
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            x = float(v[0]); y = float(v[1])
        except Exception:
            continue
        out.append([x, y])
        typ = None
        if len(v) >= 3 and isinstance(v[2], str) and v[2]:
            typ = v[2][0]
        elif isinstance(types, str) and i < len(types) and types[i]:
            typ = types[i]
        out_types.append(typ or "L")
    pts = np.asarray(out, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64), "", closed
    return pts, "".join(out_types), closed

def _sample_quad(p0, p1, p2, n=16):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def _sample_cubic(p0, p1, p2, p3, n=20):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 +
            3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)

def _poly2d_segment_to_points(poly: dict, bezier_samples: int = 20) -> np.ndarray:
    verts, types, closed = _coerce_vertices_and_types(poly)
    if len(verts) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    dense = [verts[0].copy()]
    i = 1
    while i < len(verts):
        cur = dense[-1]
        t = types[i] if i < len(types) else "L"
        if t == "L":
            dense.append(verts[i].copy())
            i += 1
            continue
        if i + 2 < len(verts) and types[i] == "C" and types[i + 1] == "C" and types[i + 2] == "L":
            pts = _sample_cubic(cur, verts[i], verts[i + 1], verts[i + 2], bezier_samples)
            dense.extend(pts[1:])
            i += 3
            continue
        if i + 1 < len(verts) and types[i] == "C" and types[i + 1] == "L":
            pts = _sample_quad(cur, verts[i], verts[i + 1], max(8, bezier_samples // 2))
            dense.extend(pts[1:])
            i += 2
            continue
        dense.append(verts[i].copy())
        i += 1
    if closed and len(dense) > 2 and np.linalg.norm(np.asarray(dense[0]) - np.asarray(dense[-1])) > 1e-6:
        dense.append(np.asarray(dense[0]).copy())
    pts = np.asarray(dense, dtype=np.float64)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    return pts[keep]

def parse_poly2d(poly2d_field, bezier_samples: int = 20) -> List[np.ndarray]:
    out = []
    for poly in _normalize_poly2d_items(poly2d_field):
        pts = _poly2d_segment_to_points(poly, bezier_samples=bezier_samples)
        if len(pts) >= 2:
            out.append(pts)
    return out

def _walk(obj, path="root"):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")

def extract_lane_labels_any(record: dict) -> List[dict]:
    labels = []
    for lab in _iter_possible_labels(record):
        if isinstance(lab, dict) and lab.get("poly2d") is not None:
            labels.append(lab)
    if not labels:
        for _, node in _walk(record):
            if isinstance(node, dict) and node.get("poly2d") is not None:
                labels.append(node)
    return [lab for lab in labels if _normalize_lane_category(lab) is not None]

def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
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
            span_y = float(pl[:, 1].max() - pl[:, 1].min())
            arc = float(np.sqrt(((np.diff(pl, axis=0)) ** 2).sum(axis=1)).sum())
            if arc < 2.0 or span_y < 1.0:
                continue
            lane_items.append((norm_cat, pl))
    lane_items.sort(key=lambda x: ((x[1][:, 1].max() - x[1][:, 1].min()), np.sqrt(((np.diff(x[1], axis=0)) ** 2).sum(axis=1)).sum()), reverse=True)
    lane_idx = 0
    for norm_cat, pl in lane_items:
        if lane_idx >= max_lanes:
            break
        resampled, vis = resample_polyline(pl, num_points)
        norm = resampled.copy()
        norm[:, 0] /= float(img_w)
        norm[:, 1] /= float(img_h)
        existence[lane_idx] = 1.0
        points[lane_idx] = np.clip(norm, 0.0, 1.0).astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[norm_cat]
        lane_idx += 1
    return {"existence": existence, "points": points, "visibility": visibility, "lane_type": lane_type}

def inspect_json_for_lanes(source_path: str, limit: int = 3) -> List[dict]:
    samples = []
    paths = sorted(str(p) for p in Path(source_path).glob("*.json"))[:limit] if os.path.isdir(source_path) else [source_path]
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception as e:
            samples.append({"path": p, "error": str(e)})
            continue
        rec = data[0] if isinstance(data, list) and data else data
        labels = extract_lane_labels_any(rec) if isinstance(rec, dict) else []
        sample = {"path": p, "top_type": type(data).__name__, "record_keys": list(rec.keys())[:20] if isinstance(rec, dict) else None, "num_lane_like": len(labels)}
        if labels:
            lab = labels[0]
            sample["first_category"] = lab.get("category")
            sample["first_attr_keys"] = list((lab.get("attributes") or {}).keys())
            polys = _normalize_poly2d_items(lab.get("poly2d"))
            if polys:
                verts, types, closed = _coerce_vertices_and_types(polys[0])
                sample["first_vertices_head"] = verts[:5].tolist() if len(verts) else []
                sample["first_types"] = types[:20]
                sample["closed"] = closed
        samples.append(sample)
    return samples

class LaneLabelCache:
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
        labels = extract_lane_labels_any(record)
        name = _record_image_name(record, fallback_path)
        if not name or not labels:
            return
        targets = frame_to_lane_targets(labels, self.max_lanes, self.num_points)
        if float(targets["existence"].sum()) > 0:
            self._cache[name] = labels
            if len(self._debug_samples) < 3:
                first = labels[0]
                polys = _normalize_poly2d_items(first.get("poly2d"))
                verts, types, _ = _coerce_vertices_and_types(polys[0]) if polys else (np.zeros((0, 2)), "", False)
                self._debug_samples.append({
                    "name": name,
                    "n_labels": len(labels),
                    "existence_sum": float(targets["existence"].sum()),
                    "first_category": first.get("category"),
                    "first_vertices_head": verts[:5].tolist() if len(verts) else [],
                    "first_types": types[:20],
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
        if self._debug_samples:
            print(f"  Debug sample: {self._debug_samples[0]}")
        elif json_files:
            print("  No usable lane polylines extracted. JSON inspection:")
            for s in inspect_json_for_lanes(dir_path, limit=2):
                print(f"   {s}")

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
        if self._debug_samples:
            print(f"  Debug sample: {self._debug_samples[0]}")
        else:
            print(f"  No usable lane polylines extracted. JSON inspection: {inspect_json_for_lanes(json_path, limit=1)}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
