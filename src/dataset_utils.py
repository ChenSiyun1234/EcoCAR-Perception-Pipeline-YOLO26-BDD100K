"""
dataset_utils.py — BDD100K ↔ YOLO format conversion and dataset helpers.

Used by:
  - 02_bdd100k_preparation.ipynb  (convert + build dataset)
  - 03_train_yolo26_on_bdd.ipynb  (load dataset YAML)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# ── BDD100K detection class mapping (alphabetical, canonical order) ──────────
BDD_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(BDD_CLASSES)}


def get_bdd_class_mapping() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Return (class_list, name→id, id→name) for BDD100K detection."""
    return BDD_CLASSES, CLASS_TO_ID, ID_TO_CLASS


# ── BDD100K JSON → YOLO txt conversion ──────────────────────────────────────

def convert_bdd100k_to_yolo(
    json_path: str,
    output_label_dir: str,
    img_width: int = 1280,
    img_height: int = 720,
    debug_limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Convert a BDD100K detection JSON file to YOLO label .txt files.

    Each output line: <class_id> <x_center> <y_center> <width> <height>
    All values normalised to [0, 1].

    Args:
        json_path:        Path to BDD100K det JSON (e.g. det_train.json).
        output_label_dir: Directory to write per-image .txt label files.
        img_width:        Image width (BDD100K default 1280).
        img_height:       Image height (BDD100K default 720).
        debug_limit:      If set, process only this many images (for testing).

    Returns:
        Dict mapping class_name → count of annotations converted.
    """
    os.makedirs(output_label_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if debug_limit is not None:
        data = data[:debug_limit]

    class_counts: Dict[str, int] = {c: 0 for c in BDD_CLASSES}
    skipped = 0

    for frame in tqdm(data, desc="Converting annotations"):
        img_name = frame["name"]
        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(output_label_dir, label_name)

        lines = []
        labels = frame.get("labels", [])
        if labels is None:
            labels = []

        for label in labels:
            category = label.get("category", "")
            if category not in CLASS_TO_ID:
                skipped += 1
                continue

            box = label.get("box2d")
            if box is None:
                continue

            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            # Convert to YOLO format (normalised x_center, y_center, w, h)
            x_center = ((x1 + x2) / 2.0) / img_width
            y_center = ((y1 + y2) / 2.0) / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            class_id = CLASS_TO_ID[category]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            class_counts[category] += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    if skipped:
        print(f"⚠ Skipped {skipped} annotations with unknown categories")

    return class_counts


# ── Dataset YAML generation ─────────────────────────────────────────────────

def create_dataset_yaml(
    dataset_root: str,
    output_path: str,
    train_images: str = "images/train",
    val_images: str = "images/val",
) -> str:
    """
    Generate a YOLO-compatible dataset YAML file.

    Args:
        dataset_root:  Absolute path to the dataset root directory.
        output_path:   Where to write the YAML file.
        train_images:  Relative path from dataset_root to training images.
        val_images:    Relative path from dataset_root to validation images.

    Returns:
        The absolute path to the written YAML file.
    """
    config = {
        "path": dataset_root,
        "train": train_images,
        "val": val_images,
        "nc": len(BDD_CLASSES),
        "names": {i: name for i, name in enumerate(BDD_CLASSES)},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Dataset YAML written to: {output_path}")
    return os.path.abspath(output_path)


# ── Dataset structure verification ───────────────────────────────────────────

def verify_dataset_structure(dataset_root: str) -> bool:
    """
    Check that the YOLO dataset directory has the expected layout:
      dataset_root/
        images/train/  (non-empty)
        images/val/    (non-empty)
        labels/train/  (non-empty)
        labels/val/    (non-empty)

    Returns True if all checks pass, False otherwise.
    """
    required = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]
    all_ok = True
    for subdir in required:
        full = os.path.join(dataset_root, subdir)
        if not os.path.isdir(full):
            print(f"❌ Missing directory: {full}")
            all_ok = False
        else:
            count = len(os.listdir(full))
            if count == 0:
                print(f"⚠ Empty directory: {full}")
                all_ok = False
            else:
                print(f"✅ {subdir}: {count} files")

    return all_ok


# ── Symlink / copy helper (Colab-safe) ──────────────────────────────────────

def link_or_copy_images(
    src_dir: str,
    dst_dir: str,
    use_symlinks: bool = True,
    debug_limit: Optional[int] = None,
) -> int:
    """
    Populate dst_dir with images from src_dir (symlink on Linux, copy on Windows).

    Args:
        src_dir:      Source image directory.
        dst_dir:      Destination image directory.
        use_symlinks: Try symlinks first (fast, saves disk on Colab).
        debug_limit:  If set, only link/copy this many images.

    Returns:
        Number of images linked/copied.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(os.listdir(src_dir))
    if debug_limit:
        files = files[:debug_limit]

    count = 0
    for fname in tqdm(files, desc=f"Linking images → {os.path.basename(dst_dir)}"):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            count += 1
            continue
        try:
            if use_symlinks:
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            count += 1
        except OSError:
            # Fallback to copy if symlinks not supported
            shutil.copy2(src, dst)
            count += 1

    return count


# ── Print class distribution ─────────────────────────────────────────────────

def print_class_distribution(class_counts: Dict[str, int]) -> None:
    """Pretty-print the per-class annotation counts."""
    total = sum(class_counts.values())
    print(f"\n{'Class':<20} {'Count':>8} {'Pct':>7}")
    print("─" * 37)
    for name in BDD_CLASSES:
        c = class_counts.get(name, 0)
        pct = (c / total * 100) if total > 0 else 0
        print(f"{name:<20} {c:>8,} {pct:>6.1f}%")
    print("─" * 37)
    print(f"{'TOTAL':<20} {total:>8,}")
