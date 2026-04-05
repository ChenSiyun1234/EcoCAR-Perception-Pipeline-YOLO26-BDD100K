
"""
Utilities to rebuild the current DualPathNet dataset directly from the original
BDD100K zip files stored in EcoCAR/downloads.

This module intentionally avoids depending on legacy notebook2 logic.
It supports:
- extracting bdd100k_labels.zip, bdd100k_images_100k.zip, bdd100k_seg_maps.zip
- locating detection JSON files in the extracted tree
- creating a clean vehicle-only YOLO dataset with classes:
  car, truck, bus, motorcycle, bicycle
- wiring paths_config.yaml so the lane parser can discover raw labels later
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from .config import ECOCAR_ROOT, DOWNLOADS_DIR, RAW_DATASET_ROOT, PATHS_CONFIG

VEHICLE_NAME_TO_ID = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
}
VEHICLE_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle"]

DETECTION_JSON_CANDIDATES = {
    "train": [
        "labels/bdd100k_labels_images_train.json",
        "bdd100k/labels/bdd100k_labels_images_train.json",
        "bdd100k_labels_images_train.json",
    ],
    "val": [
        "labels/bdd100k_labels_images_val.json",
        "bdd100k/labels/bdd100k_labels_images_val.json",
        "bdd100k_labels_images_val.json",
    ],
}

IMAGE_DIR_CANDIDATES = {
    "train": [
        "images/100k/train",
        "bdd100k/images/100k/train",
        "100k/train",
    ],
    "val": [
        "images/100k/val",
        "bdd100k/images/100k/val",
        "100k/val",
    ],
}

SEG_DIR_CANDIDATES = {
    "train": [
        "seg/images/train",
        "labels/sem_seg/masks/train",
        "bdd100k/seg/images/train",
        "bdd100k/labels/sem_seg/masks/train",
    ],
    "val": [
        "seg/images/val",
        "labels/sem_seg/masks/val",
        "bdd100k/seg/images/val",
        "bdd100k/labels/sem_seg/masks/val",
    ],
}

@dataclass
class RebuildSummary:
    raw_root: str
    output_root: str
    train_json: str
    val_json: str
    train_images: str
    val_images: str
    train_seg: str = ""
    val_seg: str = ""
    counts_train: Dict[str, int] = None
    counts_val: Dict[str, int] = None


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def ensure_zip_extracted(zip_path: str | Path, extract_root: str | Path, stamp_name: Optional[str] = None, force: bool = False) -> str:
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing zip: {zip_path}")
    stamp_name = stamp_name or (zip_path.stem + '.extracted')
    stamp = extract_root / stamp_name
    if force or not stamp.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_root)
        _touch(stamp)
    return str(extract_root)


def _resolve_first(root: str | Path, rel_candidates: Iterable[str]) -> Optional[str]:
    root = Path(root)
    for rel in rel_candidates:
        p = root / rel
        if p.exists():
            return str(p)
    return None


def locate_detection_jsons(raw_root: str | Path) -> Tuple[str, str]:
    train_json = _resolve_first(raw_root, DETECTION_JSON_CANDIDATES['train'])
    val_json = _resolve_first(raw_root, DETECTION_JSON_CANDIDATES['val'])
    if not train_json or not val_json:
        raise FileNotFoundError(
            f"Could not locate train/val detection JSONs under {raw_root}. "
            f"Expected names like bdd100k_labels_images_train.json and _val.json"
        )
    return train_json, val_json


def locate_image_dirs(raw_root: str | Path) -> Tuple[str, str]:
    train_dir = _resolve_first(raw_root, IMAGE_DIR_CANDIDATES['train'])
    val_dir = _resolve_first(raw_root, IMAGE_DIR_CANDIDATES['val'])
    if not train_dir or not val_dir:
        raise FileNotFoundError(f"Could not locate train/val image directories under {raw_root}")
    return train_dir, val_dir


def locate_seg_dirs(raw_root: str | Path) -> Tuple[str, str]:
    train_dir = _resolve_first(raw_root, SEG_DIR_CANDIDATES['train']) or ""
    val_dir = _resolve_first(raw_root, SEG_DIR_CANDIDATES['val']) or ""
    return train_dir, val_dir


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _box2d_to_yolo(box2d: dict, img_w: float = 1280.0, img_h: float = 720.0) -> Optional[Tuple[float, float, float, float]]:
    try:
        x1 = float(box2d['x1'])
        y1 = float(box2d['y1'])
        x2 = float(box2d['x2'])
        y2 = float(box2d['y2'])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) * 0.5) / img_w
    yc = ((y1 + y2) * 0.5) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    xc, yc, w, h = map(_clip01, (xc, yc, w, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return xc, yc, w, h


def _iter_detection_labels(record: dict) -> Iterable[Tuple[int, Tuple[float, float, float, float]]]:
    for lab in record.get('labels', []) or []:
        if not isinstance(lab, dict):
            continue
        category = str(lab.get('category', '') or '').strip().lower()
        if category not in VEHICLE_NAME_TO_ID:
            continue
        box2d = lab.get('box2d') or {}
        yolo = _box2d_to_yolo(box2d)
        if yolo is None:
            continue
        yield VEHICLE_NAME_TO_ID[category], yolo


def convert_detection_json_to_vehicle_yolo(json_path: str | Path, labels_out_dir: str | Path) -> Dict[str, int]:
    json_path = Path(json_path)
    labels_out_dir = Path(labels_out_dir)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    counts = Counter()
    for rec in records:
        name = rec.get('name')
        if not isinstance(name, str) or not name:
            continue
        stem = Path(name).stem
        rows = []
        for cls_id, (xc, yc, w, h) in _iter_detection_labels(rec):
            rows.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[VEHICLE_NAMES[cls_id]] += 1
        (labels_out_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return dict(counts)


def _link_or_copy_tree(src: str | Path, dst: str | Path, mode: str = 'symlink'):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def write_vehicle_yaml(output_root: str | Path):
    output_root = Path(output_root)
    yaml_path = output_root / 'bdd100k_vehicle5.yaml'
    data = {
        'path': str(output_root),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 5,
        'names': {i: n for i, n in enumerate(VEHICLE_NAMES)},
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(yaml_path)


def write_paths_config(raw_root: str | Path, dataset_root: str | Path, extra: Optional[dict] = None):
    raw_root = str(raw_root)
    dataset_root = str(dataset_root)
    payload = {
        'bdd_raw_dir': raw_root,
        'bdd100k_raw': raw_root,
        'bdd_root': raw_root,
        'bdd100k_root': raw_root,
        'raw_bdd100k_dir': raw_root,
        'dataset_root': dataset_root,
    }
    if extra:
        payload.update(extra)
    Path(PATHS_CONFIG).parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS_CONFIG, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return PATHS_CONFIG


def summarize_label_ids(labels_dir: str | Path, max_files: int = 2000) -> Dict[int, int]:
    labels_dir = Path(labels_dir)
    counts = Counter()
    files = sorted(labels_dir.glob('*.txt'))[:max_files]
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    counts[int(float(parts[0]))] += 1
                except Exception:
                    continue
    return dict(sorted(counts.items()))


def rebuild_dualpath_dataset(
    downloads_dir: str | Path = DOWNLOADS_DIR,
    raw_root: str | Path = RAW_DATASET_ROOT,
    output_root: str | Path = os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k_yolo'),
    image_mode: str = 'symlink',
    force_extract: bool = False,
) -> RebuildSummary:
    downloads_dir = Path(downloads_dir)
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    labels_zip = downloads_dir / 'bdd100k_labels.zip'
    images_zip = downloads_dir / 'bdd100k_images_100k.zip'
    seg_zip = downloads_dir / 'bdd100k_seg_maps.zip'

    ensure_zip_extracted(labels_zip, raw_root, stamp_name='.labels_extracted', force=force_extract)
    ensure_zip_extracted(images_zip, raw_root, stamp_name='.images_extracted', force=force_extract)
    if seg_zip.is_file():
        ensure_zip_extracted(seg_zip, raw_root, stamp_name='.seg_extracted', force=force_extract)

    train_json, val_json = locate_detection_jsons(raw_root)
    train_images, val_images = locate_image_dirs(raw_root)
    train_seg, val_seg = locate_seg_dirs(raw_root)

    (output_root / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels').mkdir(parents=True, exist_ok=True)

    _link_or_copy_tree(train_images, output_root / 'images' / 'train', mode=image_mode)
    _link_or_copy_tree(val_images, output_root / 'images' / 'val', mode=image_mode)
    if train_seg:
        _link_or_copy_tree(train_seg, output_root / 'seg_maps' / 'train', mode=image_mode)
    if val_seg:
        _link_or_copy_tree(val_seg, output_root / 'seg_maps' / 'val', mode=image_mode)

    counts_train = convert_detection_json_to_vehicle_yolo(train_json, output_root / 'labels' / 'train')
    counts_val = convert_detection_json_to_vehicle_yolo(val_json, output_root / 'labels' / 'val')

    write_vehicle_yaml(output_root)
    write_paths_config(raw_root, output_root, extra={
        'train_detection_json': train_json,
        'val_detection_json': val_json,
        'train_images_dir': train_images,
        'val_images_dir': val_images,
        'train_seg_dir': train_seg,
        'val_seg_dir': val_seg,
    })

    return RebuildSummary(
        raw_root=str(raw_root),
        output_root=str(output_root),
        train_json=train_json,
        val_json=val_json,
        train_images=train_images,
        val_images=val_images,
        train_seg=train_seg,
        val_seg=val_seg,
        counts_train=counts_train,
        counts_val=counts_val,
    )
