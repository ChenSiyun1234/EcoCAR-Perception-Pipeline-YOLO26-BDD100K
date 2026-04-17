# Dataset / Path / Lane-Parsing Handoff for Claude Code

This note summarizes **all current dataset usage methods, structures, and pitfalls** for the `yolop_vehicle_lane` project so you can debug without re-learning the path traps.

---

## 1. Canonical project root

Use this as the main project root:

```python
PROJECT_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
```

Do **not** use the old path under:

```python
/content/drive/MyDrive/EcoCAR/EcoCAR-Perception-Pipeline-YOLO26-BDD100K/yolop_vehicle_lane
```

All notebook outputs that need to survive runtime resets must be saved back under `PROJECT_ROOT`.

---

## 2. Persistence rule: `/content` is not persistent

This is one of the most important rules.

### Never assume:
- a processed dataset already exists in `/content/...`
- notebook N can reuse notebook M outputs from local SSD
- extracted zips or tar files still exist in the next runtime

### Always assume:
- every notebook must recover what it needs
- if a notebook produces reusable artifacts, it must save them back to Drive
- downstream notebooks should read from Drive-backed outputs, not hope `/content` still has them

---

## 3. Two dataset forms exist in this workflow

There are **two distinct dataset forms**:

### A. Raw BDD-style data
Used as the source of truth for images / old JSON labels / lane poly2d.

Possible roots include:
- `/content/bdd100k_raw`
- `/content/bdd100k`
- `/content/drive/MyDrive/EcoCAR/bdd100k_raw`
- `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_raw`
- roots discovered via `paths_config.yaml`

But the internal layout is **not guaranteed to be only one shape**.

Typical valid sublayouts include:

#### Images
```text
RAW_BDD_ROOT/images/100k/train/*.jpg
RAW_BDD_ROOT/images/100k/val/*.jpg
```

or

```text
RAW_BDD_ROOT/bdd100k/images/100k/train/*.jpg
RAW_BDD_ROOT/bdd100k/images/100k/val/*.jpg
```

#### Detection JSON labels
```text
RAW_BDD_ROOT/labels/100k/train/*.json
RAW_BDD_ROOT/labels/100k/val/*.json
```

or

```text
RAW_BDD_ROOT/bdd100k/labels/100k/train/*.json
RAW_BDD_ROOT/bdd100k/labels/100k/val/*.json
```

#### Old lane JSON source (important)
This project often uses the **old per-image JSON directory layout**:

```text
RAW_BDD_ROOT/100k/train/*.json
RAW_BDD_ROOT/100k/val/*.json
```

These files contain lane annotations embedded per image.

This is why `BDD_LANE_JSON_TRAIN` or `BDD_LANE_JSON_VAL` may actually be a **directory**, not a single `.json` file.

### B. Packaged project dataset root
This is the project-ready dataset used by training notebooks.

Canonical packaged root name:

```text
bdd100k_vehicle5
```

Expected structure:

```text
DATASET_ROOT/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt or processed labels used by the training path
    val/*.txt or processed labels used by the training path
  masks/
    train/*.png
    val/*.png
```

Typical location after persistence:

```text
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/datasets/bdd100k_vehicle5
```

And tar archive:

```text
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/datasets/bdd100k_vehicle5.tar
```

---

## 4. What notebook00 is supposed to do

`00_rebuild_dataset_and_lane_cache.ipynb` is responsible for:

1. resolving the current project root
2. recovering or extracting the packaged dataset root
3. locating the raw BDD root
4. locating lane JSON source(s)
5. rendering lane masks into `DATASET_ROOT/masks/{train,val}`
6. verifying masks and spot-checking images
7. smoke-testing the dataset loader
8. saving the rebuilt dataset back to Drive for later notebooks

### notebook00 outputs that must be persisted

```text
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/datasets/bdd100k_vehicle5/masks/train/*.png
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/datasets/bdd100k_vehicle5/masks/val/*.png
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/datasets/bdd100k_vehicle5.tar
```

---

## 5. Path resolution strategy that should be used

### Project root
Always:

```python
PROJECT_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
```

### Packaged dataset root
Use helper logic similar to:
- check for already extracted local packaged dataset
- if missing, extract from Drive tar
- if tar missing, check Drive dataset dir directly

### Raw BDD root
Use helper logic that can:
- read `paths_config.yaml`
- search both project-local and shared EcoCAR locations
- if needed, auto-extract raw BDD zips from Drive into `/content/bdd100k_raw`

### Images / labels 100k directories
Do **not** assume:

```python
BDD_IMAGES = RAW_BDD_ROOT + '/images/100k'
BDD_LABELS = RAW_BDD_ROOT + '/labels/100k'
```

without validation.

Instead resolve explicitly via helper functions like:
- `resolve_bdd_images_100k_dir(raw_bdd_root)`
- `resolve_bdd_labels_100k_dir(raw_bdd_root)`

which should try both:
- `raw_root/images/100k`
- `raw_root/bdd100k/images/100k`
- same idea for labels

---

## 6. Important lane parsing facts

This is one of the biggest recurring failure points.

### The old lane source may be a directory, not a file
For this project, the working DETR-style line often uses:

```text
/content/bdd100k_raw/100k/train
/content/bdd100k_raw/100k/val
```

where each split is a directory of per-image JSONs.

So a variable named like `BDD_LANE_JSON_TRAIN` may actually be:
- a JSON file
- or a directory of many JSON files

### Therefore never do this blindly
```python
with open(json_path, 'r') as f:
    ...
```

unless you first confirm `os.path.isfile(json_path)`.

### Correct behavior
The lane rendering path must support **both**:
- consolidated JSON file input
- per-image JSON directory input

### Lane extraction logic
Use the same logic as the working DETR line as much as possible.

Preferred helpers:
- `extract_lane_labels_any(record)`
- `parse_poly2d(geom_field)`

These are better than inventing a new parser because they already handle:
- lane categories like `lane/single white`, `lane/road curb`, etc.
- old poly2d layouts
- point densification / geometry conversion

### Do not assume one lane schema only
Possible lane sources include:
- `labels/lane/polygons/lane_train.json` style
- old per-image JSON directories under `100k/train` and `100k/val`

Your code must detect which one it got.

---

## 7. Detection labels in raw BDD JSON

Raw detection labels under `labels/100k/{split}/*.json` are old BDD-style JSONs.

Typical access path in current dataset code:

```python
label['frames'][0]['objects']
```

So do not replace that parsing path casually unless you verify the raw schema in the actual extracted files.

---

## 8. Why the preview cell showed `image not found`

The masks were generated correctly, but the preview cell used a path assumption like:

```python
img_path = Path(BDD_IMAGES) / 'val' / (mask_path.stem + '.jpg')
```

This fails if:
- `BDD_IMAGES` was resolved to the wrong internal root
- the actual image lives under a nested `bdd100k/images/100k/...` path
- the file extension is not exactly `.jpg`
- packaged dataset images exist but raw path is wrong

### Correct preview strategy
Use a function like `find_preview_image(mask_stem, split)` that tries:

1. `DATASET_ROOT/images/{split}/{stem}.jpg|jpeg|png`
2. `BDD_IMAGES/{split}/{stem}.jpg|jpeg|png`
3. `RAW_BDD_ROOT/bdd100k/images/100k/{split}/{stem}.jpg|jpeg|png`
4. `RAW_BDD_ROOT/images/100k/{split}/{stem}.jpg|jpeg|png`

Only after all of those fail should you show `image not found`.

---

## 9. Common bugs already observed

### Bug A - using old project root
Wrong:
```python
/content/drive/MyDrive/EcoCAR/EcoCAR-Perception-Pipeline-YOLO26-BDD100K/yolop_vehicle_lane
```

Correct:
```python
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane
```

### Bug B - treating a lane JSON directory as a file
Example failure:
- `IsADirectoryError: ... '/content/bdd100k_raw/100k/train'`

### Bug C - lane render function argument mismatch
Example failure:
- notebook calls `output_dir=...`
- function expects `output_mask_dir=...`

The renderer should support backward-compatible aliases.

### Bug D - dataset root considered invalid too early
If notebook00 is preparing masks for the first time, the packaged dataset may already have:
- `images/train`
- `labels/train`

but **not yet** `masks/train`

The dataset-root resolver for notebook00 must not reject the packaged dataset just because masks do not exist yet.

### Bug E - preview image path too rigid
This caused the `image not found` symptom.

### Bug F - not saving notebook outputs back to Drive
If notebook00 writes masks only into local `/content/...`, later notebooks will lose them on the next runtime.

---

## 10. Minimum debugging checklist for notebook00

Before rendering masks, print all of these:

```python
print(PROJECT_ROOT)
print(DATASET_ROOT)
print(RAW_BDD_ROOT)
print(BDD_IMAGES)
print(BDD_LABELS)
print(BDD_LANE_JSON_TRAIN)
print(BDD_LANE_JSON_VAL)
print(os.path.isdir(os.path.join(BDD_IMAGES, 'train')))
print(os.path.isdir(os.path.join(BDD_IMAGES, 'val')))
```

Also print for each lane source:
- exists?
- is_file?
- is_dir?
- first few sample JSON file names if directory
- first few inspection results from `inspect_json_for_lanes(...)`

After rendering, print:
- number of masks in `train`
- number of masks in `val`
- non-empty preview paths

Before finishing notebook00, verify these exist on Drive:

```text
PROJECT_ROOT/datasets/bdd100k_vehicle5/masks/train
PROJECT_ROOT/datasets/bdd100k_vehicle5/masks/val
PROJECT_ROOT/datasets/bdd100k_vehicle5.tar
```

---

## 11. Current notebook organization expectations

Notebook outputs should be chained through Drive, not local SSD.

Expected stages:
- `00_...` rebuild dataset + lane masks + save back to Drive
- `01_...` augmentation verification
- `02_...` baseline train
- `03_...` eval / ablation
- `07_...` video profiling

If a notebook needs an artifact from an earlier notebook, that artifact must already exist under `PROJECT_ROOT/...` or `PROJECT_ROOT/datasets/...`.

---

## 12. Video profiling note

The target GPU for video validation/profiling is now:

```text
A5000
```

Not H100.

So any profiling notebook defaults, comments, report filenames, or MFU assumptions should be updated accordingly.

---

## 13. Practical instruction for Claude Code

When debugging, do **not** start by redesigning the architecture.

Start with the dataset chain:
1. verify project root
2. verify packaged dataset root
3. verify raw BDD root
4. verify image root
5. verify label root
6. verify lane source type (file vs dir)
7. verify mask generation counts
8. verify preview image lookup
9. verify save-back-to-Drive
10. only then debug training

Most current failures in this project have been caused by **path/layout mismatches**, not by the model itself.

