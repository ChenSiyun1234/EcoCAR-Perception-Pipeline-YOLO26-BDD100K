"""
Lane mask rendering utilities.
Migrated from yolo26_pipeline/src/lane_utils.py.
Renders BDD100K poly2d lane annotations into binary segmentation masks.
Supports both consolidated JSON files and the older per-image JSON directory
layout used by the working DETR_GeoLane pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .lane_targets import extract_lane_labels_any, parse_poly2d


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


def _record_image_name(record, fallback_json_path=None):
    if not isinstance(record, dict):
        if fallback_json_path:
            return Path(fallback_json_path).stem + '.jpg'
        return ''
    for key in ['name', 'image', 'imageName', 'filename', 'id']:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            base = os.path.basename(value)
            if '.' not in base:
                base += '.jpg'
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + '.jpg'
    return ''


def _iter_records_from_source(json_path: str):
    if os.path.isdir(json_path):
        json_files = sorted(str(p) for p in Path(json_path).glob('*.json'))
        for jpath in json_files:
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as exc:
                yield ('__error__', jpath, exc)
                continue
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        yield (rec, jpath, None)
            elif isinstance(data, dict):
                yield (data, jpath, None)
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict):
                yield (rec, json_path, None)
    elif isinstance(data, dict):
        yield (data, json_path, None)


def _resample_polyline_uniform(pts: np.ndarray, n: int = 32) -> np.ndarray:
    """Resample a polyline to `n` points evenly spaced along arc length.
    `pts` is (K, 2) in the image frame.
    """
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0) if len(pts) else np.zeros((n, 2))
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    for d, target in enumerate(t):
        idx = int(np.searchsorted(cum, target, side='right') - 1)
        idx = max(0, min(idx, len(pts) - 2))
        denom = max(cum[idx + 1] - cum[idx], 1e-6)
        alpha = (target - cum[idx]) / denom
        out[d] = pts[idx] * (1.0 - alpha) + pts[idx + 1] * alpha
    return out


def _polyline_y_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of vertical-extent overlap between two polylines (0..1)."""
    ya0, ya1 = float(a[:, 1].min()), float(a[:, 1].max())
    yb0, yb1 = float(b[:, 1].min()), float(b[:, 1].max())
    inter = max(0.0, min(ya1, yb1) - max(ya0, yb0))
    union = max(1e-6, max(ya1, yb1) - min(ya0, yb0))
    return inter / union


def _mean_lateral_distance(a_rs: np.ndarray, b_rs: np.ndarray) -> float:
    """Mean pairwise distance between two resampled polylines of the
    same length. Used as a pairing cost.
    """
    return float(np.linalg.norm(a_rs - b_rs, axis=1).mean())


def _pair_lane_boundaries(
    polys: List[np.ndarray],
    max_dist: float,
    min_y_overlap: float = 0.5,
    resample_n: int = 32,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Greedy pairing of lane boundary polylines.

    [INFERRED] BDD100K does not link the two markings of a single
    driving lane. We treat every pair of polylines whose
      - vertical extent overlap ≥ `min_y_overlap`, and
      - mean lateral distance (after arc-length resampling) ≤ `max_dist`
    as candidate pairs, and greedily assign closest pairs first. Any
    polyline that doesn't match a partner (typical for road curbs and
    isolated markings) stays unpaired and is drawn as-is.

    Returns `(pairs, singles)` — index lists into `polys`.
    """
    n = len(polys)
    if n == 0:
        return [], []
    rs = [_resample_polyline_uniform(p, resample_n) for p in polys]

    # All candidate pairs sorted by lateral distance.
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = _polyline_y_overlap(polys[i], polys[j])
            if ov < min_y_overlap:
                continue
            d = _mean_lateral_distance(rs[i], rs[j])
            if d > max_dist:
                continue
            cand.append((d, i, j))
    cand.sort()

    used = [False] * n
    pairs = []
    for _, i, j in cand:
        if used[i] or used[j]:
            continue
        pairs.append((i, j))
        used[i] = used[j] = True
    singles = [i for i in range(n) if not used[i]]
    return pairs, singles


def _centerline(a: np.ndarray, b: np.ndarray, n: int = 48) -> np.ndarray:
    """Midline between two polylines (arc-length resampled to `n`)."""
    a_rs = _resample_polyline_uniform(a, n)
    b_rs = _resample_polyline_uniform(b, n)
    # Align orientation — if endpoints are flipped, reverse b.
    if np.linalg.norm(a_rs[0] - b_rs[0]) > np.linalg.norm(a_rs[0] - b_rs[-1]):
        b_rs = b_rs[::-1]
    return 0.5 * (a_rs + b_rs)


def render_lane_mask(
    labels: List[Dict],
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
    pair_centerlines: bool = True,
    pair_max_dist_ratio: float = 0.12,
) -> np.ndarray:
    """Render BDD100K lane polylines as a binary mask.

    YOLOPv2 paper §3: training mask width 8, test mask width 2, drawn
    on the **centerline between the two annotated lines** of each lane.

    BDD100K annotates each lane marking as an independent `poly2d`. We
    reconstruct the centerline with a greedy pairing heuristic:
      * resample every polyline to 32 equal-arc-length points;
      * pair (i, j) whose vertical-extent overlap ≥ 50 % and whose mean
        lateral distance ≤ `pair_max_dist_ratio × img_width`
        (default 0.12 × 1280 ≈ 154 px — comfortably covers normal lane
        widths);
      * greedy assignment: closest unused pair first;
      * unpaired polylines (road curbs, single shoulder markings) are
        drawn as-is. These are the "centerline" of themselves.

    Set `pair_centerlines=False` to get the old "draw every poly2d
    directly" behavior.

    [INFERRED] The paper does not publish its pairing code; the
    heuristic above is the closest reproducible approximation.
    """
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    sx = mask_width / float(img_width)
    sy = mask_height / float(img_height)

    # ── Extract all dense polylines (image-frame pixel coords) ────────
    polys: List[np.ndarray] = []
    for label in labels:
        geom_field = label.get('poly2d')
        if geom_field is None and label.get('seg2d') is not None:
            geom_field = label.get('seg2d')
        for dense_pts in parse_poly2d(geom_field):
            if len(dense_pts) >= 2:
                polys.append(np.asarray(dense_pts, dtype=np.float64))

    # ── Pair boundary polylines into lanes ───────────────────────────
    pairs, singles = ([], list(range(len(polys))))
    if pair_centerlines and len(polys) >= 2:
        max_dist_px = pair_max_dist_ratio * float(img_width)
        pairs, singles = _pair_lane_boundaries(polys, max_dist=max_dist_px)

    def _draw(poly_img: np.ndarray):
        pts = np.empty((len(poly_img), 2), dtype=np.int32)
        pts[:, 0] = np.clip(np.round(poly_img[:, 0] * sx), 0, mask_width - 1)
        pts[:, 1] = np.clip(np.round(poly_img[:, 1] * sy), 0, mask_height - 1)
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], False, 255, line_thickness)

    for i, j in pairs:
        _draw(_centerline(polys[i], polys[j]))
    for i in singles:
        _draw(polys[i])

    return mask


def convert_bdd_lanes_to_masks(
    json_path: str,
    output_mask_dir: Optional[str] = None,
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
    debug_limit: Optional[int] = None,
    overwrite: bool = True,
    pair_centerlines: bool = True,
    pair_max_dist_ratio: float = 0.12,
    **legacy_kwargs,
) -> Dict[str, int]:
    """Convert BDD100K lane labels to binary mask PNGs.

    Accepts either:
    - a consolidated JSON file (lane_train.json / lane_val.json)
    - an old-style per-image JSON directory (100k/train / 100k/val)

    Backward-compatible aliases are accepted so older notebooks copied from
    earlier experiments still run:
      output_dir   -> output_mask_dir
      img_w / img_h -> img_width / img_height
      mask_w / mask_h -> mask_width / mask_height
    """
    if output_mask_dir is None:
        output_mask_dir = legacy_kwargs.pop('output_dir', None)
    if output_mask_dir is None:
        raise TypeError('convert_bdd_lanes_to_masks() missing required argument: output_mask_dir')

    if 'img_w' in legacy_kwargs:
        img_width = legacy_kwargs.pop('img_w')
    if 'img_h' in legacy_kwargs:
        img_height = legacy_kwargs.pop('img_h')
    if 'mask_w' in legacy_kwargs:
        mask_width = legacy_kwargs.pop('mask_w')
    if 'mask_h' in legacy_kwargs:
        mask_height = legacy_kwargs.pop('mask_h')
    if legacy_kwargs:
        raise TypeError(f'Unexpected keyword arguments: {sorted(legacy_kwargs.keys())}')

    os.makedirs(output_mask_dir, exist_ok=True)

    stats = {
        'total_records_seen': 0,
        'total_images': 0,
        'images_with_lanes': 0,
        'total_lane_annotations': 0,
        'json_errors': 0,
        'written_masks': 0,
        'skipped_existing': 0,
    }

    debug_examples = []
    iterator = _iter_records_from_source(json_path)
    desc = 'Rendering lane masks from directory' if os.path.isdir(json_path) else 'Rendering lane masks from file'

    for item, source_path, error in tqdm(iterator, desc=desc):
        if item == '__error__':
            stats['json_errors'] += 1
            if len(debug_examples) < 3:
                debug_examples.append({'source': source_path, 'error': str(error)})
            continue

        stats['total_records_seen'] += 1
        if debug_limit is not None and stats['total_records_seen'] > int(debug_limit):
            break

        record = item
        image_name = _record_image_name(record, source_path)
        if not image_name:
            continue

        lane_labels = extract_lane_labels_any(record)
        mask_name = Path(image_name).stem + '.png'
        mask_path = os.path.join(output_mask_dir, mask_name)

        stats['total_images'] += 1
        if lane_labels:
            stats['images_with_lanes'] += 1
            stats['total_lane_annotations'] += len(lane_labels)

        if (not overwrite) and os.path.isfile(mask_path):
            stats['skipped_existing'] += 1
            continue

        mask = render_lane_mask(
            lane_labels,
            mask_width=mask_width,
            mask_height=mask_height,
            img_width=img_width,
            img_height=img_height,
            line_thickness=line_thickness,
            pair_centerlines=pair_centerlines,
            pair_max_dist_ratio=pair_max_dist_ratio,
        )
        cv2.imwrite(mask_path, mask)
        stats['written_masks'] += 1

        if len(debug_examples) < 3:
            debug_examples.append({
                'image_name': image_name,
                'source': source_path,
                'n_lane_labels': len(lane_labels),
                'mask_path': mask_path,
                'mask_has_pixels': int(mask.sum() > 0),
            })

    if debug_examples:
        print('Lane render debug examples:')
        for ex in debug_examples:
            print(' ', ex)
    return stats


def print_lane_stats(stats: Dict[str, int]) -> None:
    print(f"\n{'='*40}")
    print(' Lane Mask Statistics')
    print(f"{'='*40}")
    print(f" Total records seen:     {stats.get('total_records_seen', 0):,}")
    print(f" Total images:          {stats['total_images']:,}")
    print(f" Images with lanes:     {stats['images_with_lanes']:,}")
    pct = (stats['images_with_lanes'] / max(1, stats['total_images'])) * 100
    print(f" Lane coverage:         {pct:.1f}%")
    print(f" Total lane annotations:{stats['total_lane_annotations']:,}")
    avg = stats['total_lane_annotations'] / max(1, stats['images_with_lanes'])
    print(f" Avg lanes per image:   {avg:.1f}")
    print(f" Written masks:         {stats.get('written_masks', 0):,}")
    print(f" Skipped existing:      {stats.get('skipped_existing', 0):,}")
    print(f" JSON errors:           {stats.get('json_errors', 0):,}")
    print(f"{'='*40}")
