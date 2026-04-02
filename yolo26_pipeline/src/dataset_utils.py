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

# ── BDD100K full 10-class mapping (for reference / legacy) ───────────────────
BDD_CLASSES_FULL = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# ── Vehicle-only classes (new primary target) ────────────────────────────────
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

# BDD100K category name -> vehicle class ID (0-4)
VEHICLE_CLASS_TO_ID = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
}

# Legacy aliases for backward compatibility
BDD_CLASSES = VEHICLE_CLASSES
CLASS_TO_ID = VEHICLE_CLASS_TO_ID
ID_TO_CLASS = {idx: name for name, idx in VEHICLE_CLASS_TO_ID.items()}


def get_bdd_class_mapping() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Return (class_list, name→id, id→name) for vehicle-only detection."""
    return VEHICLE_CLASSES, VEHICLE_CLASS_TO_ID, ID_TO_CLASS


# ── BDD100K JSON → YOLO txt conversion ──────────────────────────────────────

def aggregate_per_frame_jsons(
    json_dir: str,
    output_json: str,
) -> int:
    """
    Aggregate per-frame Scalabel JSON files into a single consolidated JSON.

    The official BDD100K labels zip may contain per-frame JSONs (one per image)
    instead of a single consolidated file. This function merges them into the
    format expected by convert_bdd100k_to_yolo():
        [{"name": "img.jpg", "labels": [...]}, ...]

    Per-frame JSONs can be either:
      - A single dict: {"name": "...", "labels": [...]}
      - A list with one entry: [{"name": "...", "labels": [...]}]
      - Scalabel format with "frames" key: {"frames": [{"name": "...", ...}]}

    Args:
        json_dir:    Directory containing per-frame .json files.
        output_json: Path to write the consolidated JSON file.

    Returns:
        Number of frames aggregated.
    """
    json_files = sorted([
        f for f in os.listdir(json_dir)
        if f.lower().endswith('.json')
    ])

    if not json_files:
        print(f"⚠ No JSON files found in {json_dir}")
        return 0

    all_frames = []
    for jf in tqdm(json_files, desc=f"Aggregating {os.path.basename(json_dir)}"):
        fpath = os.path.join(json_dir, jf)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if isinstance(data, list):
            all_frames.extend(data)
        elif isinstance(data, dict):
            if 'frames' in data:
                all_frames.extend(data['frames'])
            elif 'name' in data:
                all_frames.append(data)
            else:
                # Try using the filename stem as the image name
                stem = os.path.splitext(jf)[0]
                data['name'] = stem + '.jpg'
                all_frames.append(data)

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(all_frames, f)

    print(f"✅ Aggregated {len(all_frames)} frames → {output_json}")
    return len(all_frames)


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
        # Handle name field that may lack .jpg extension
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
        "nc": len(VEHICLE_CLASSES),
        "names": {i: name for i, name in enumerate(VEHICLE_CLASSES)},
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
    image_list: Optional[List[str]] = None,
) -> int:
    """
    Populate dst_dir with images from src_dir (symlink on Linux, copy on Windows).

    Args:
        src_dir:      Source image directory.
        dst_dir:      Destination image directory.
        use_symlinks: Try symlinks first (fast, saves disk on Colab).
        debug_limit:  If set, only link/copy this many images.
        image_list:   Provide exact list of filenames to bypass FUSE os.listdir.

    Returns:
        Number of images linked/copied.
    """
    os.makedirs(dst_dir, exist_ok=True)
    if image_list is not None:
        files = image_list
    else:
        try:
            files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except OSError:
            print(f"⚠ OSError listing {src_dir}. Provide image_list manually to bypass.")
            return 0
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


# ── Find expected images (robust, FUSE-safe) ─────────────────────────────────

def find_expected_images(
    label_dir: str,
    image_src_dirs,
) -> Tuple[List[str], str]:
    """
    Find images matching label files, searching multiple candidate directories.

    Designed to work on Google Drive FUSE where:
      - os.listdir() crashes on dirs with ~40K files
      - Per-file os.path.exists() × 20K files causes FUSE timeout

    Strategy:
      1. Read label stems from the LOCAL (fast) label directory
      2. For each candidate source dir, probe a SMALL sample (10 files)
         to test if the directory is responsive and contains matches
      3. Once a working dir is found, do the full match with os.path.exists()
      4. Return the full list of matched image filenames

    Symlink directories are deprioritized since FUSE is unreliable with them.

    Args:
        label_dir:      Directory containing YOLO .txt label files.
        image_src_dirs: A single path (str) or list of paths to search.

    Returns:
        (list_of_image_filenames, source_directory)
    """
    if isinstance(image_src_dirs, str):
        image_src_dirs = [image_src_dirs]

    # Deprioritize symlinks (FUSE errors)
    real_dirs = [d for d in image_src_dirs if os.path.isdir(d) and not os.path.islink(d)]
    link_dirs = [d for d in image_src_dirs if os.path.isdir(d) and os.path.islink(d)]
    ordered_dirs = real_dirs + link_dirs

    if not ordered_dirs:
        print(f"⚠ No valid image source directories found among: {image_src_dirs}")
        return [], ""

    # Get label stems from LOCAL label dir (fast Colab SSD, no FUSE issue)
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    label_stems = [os.path.splitext(f)[0] for f in label_files]
    print(f"  📝 {len(label_stems)} label files to match")

    if not label_stems:
        return [], ""

    # For each candidate dir, PROBE a small sample to find a working directory
    PROBE_SIZE = 10
    probe_stems = label_stems[:PROBE_SIZE]

    chosen_dir = ""
    for d in ordered_dirs:
        hits = 0
        for stem in probe_stems:
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(d, stem + ext)):
                    hits += 1
                    break
        if hits > 0:
            print(f"  ✅ Probe: {hits}/{PROBE_SIZE} found in {d}")
            chosen_dir = d
            break
        else:
            print(f"  ❌ Probe: 0/{PROBE_SIZE} found in {d}")

    if not chosen_dir:
        print("⚠ No directory responded to probe. Returning empty list.")
        return [], ""

    # Full match against the chosen directory
    print(f"  🔍 Full matching against: {chosen_dir}")
    expected = []
    for stem in tqdm(label_stems, desc="Matching"):
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = stem + ext
            if os.path.exists(os.path.join(chosen_dir, candidate)):
                expected.append(candidate)
                break

    return expected, chosen_dir


# ── Print class distribution ─────────────────────────────────────────────────

def print_class_distribution(class_counts: Dict[str, int]) -> None:
    """Pretty-print the per-class annotation counts."""
    total = sum(class_counts.values())
    print(f"\n{'Class':<20} {'Count':>8} {'Pct':>7}")
    print("─" * 37)
    for name in VEHICLE_CLASSES:
        c = class_counts.get(name, 0)
        pct = (c / total * 100) if total > 0 else 0
        print(f"{name:<20} {c:>8,} {pct:>6.1f}%")
    # Also show non-vehicle counts if present
    non_vehicle = {k: v for k, v in class_counts.items() if k not in VEHICLE_CLASS_TO_ID}
    if non_vehicle:
        skipped_total = sum(non_vehicle.values())
        print(f"{'(non-vehicle)':<20} {skipped_total:>8,} {'skip':>7}")
    print("─" * 37)
    print(f"{'TOTAL (vehicle)':<20} {total:>8,}")
