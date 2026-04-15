# Reconstruction audit: current state vs expected YOLOPv2

## Bottom line
The current rebuild is **not yet a faithful YOLOPv2 reproduction**. It is closer to a **YOLOP-based vehicle+lane variant** with some project-specific simplifications.

## What is still YOLOP-like
- `lib/models/yolop_vehicle_lane.py` still uses the YOLOP-style `MCnet` assembly pattern.
- The backbone/head wiring remains YOLOP-flavored rather than an explicit YOLOPv2 encoder-decoder redesign.
- The data pipeline still uses YOLOP-style online augmentation (`random_perspective`, HSV, flip).
- The loss stack is still essentially YOLOP-style unless `LOSS.FL_GAMMA > 0` is explicitly enabled.

## What is missing relative to the intended YOLOPv2 direction
- No clear **E-ELAN-style shared encoder** implementation.
- No explicit **group-convolution-heavy encoder redesign** matching the paper direction.
- No clear **task-specific multi-scale decoder redesign** for the lane branch.
- No implemented **Mosaic + MixUp training path** in the main dataset loader.
- No faithful **YOLOPv2 ablation structure** where only one factor is swapped at a time.
- No evidence of the stronger YOLOPv2-style lane-side decoder refinement discussed in the paper.

## What was altered beyond a clean reproduction
- The project removed drivable-area prediction entirely. This is intentional for the new task, but it means this is **not a strict reproduction**.
- The dataset code originally anchored on lane-mask files and assumed raw BDD JSON labels. That was too brittle for the DETR_GeoLane-style packaged dataset.
- The notebook path configuration was hard-coded to raw BDD folders, which conflicted with the packaged dataset workflow used elsewhere in your project.

## Dataset fix added in this patch
The loader now supports both:
1. Raw BDD roots
   - `/content/bdd100k/images/100k/{train,val}`
   - `/content/bdd100k/labels/100k/{train,val}`
   - `/content/bdd100k/lane_masks/{train,val}`
2. Packaged DETR_GeoLane-style roots
   - `/content/bdd100k_vehicle5/images/{train,val}`
   - `/content/bdd100k_vehicle5/labels/{train,val}`
   - `/content/bdd100k_vehicle5/masks/{train,val}` or `/lane_masks/{train,val}`

The loader now:
- anchors on images instead of lane masks
- accepts either YOLO `.txt` labels or raw per-image BDD `.json` labels
- auto-resolves `/masks` vs `/lane_masks`
- prints the resolved dataset layout at runtime for easier debugging

## Practical recommendation
Use this patch as a **stability fix**, not as proof that phase-1 YOLOPv2 reproduction is finished. The next step should be a more surgical audit/rebuild of:
- encoder
- neck
- lane decoder
- augmentation path
- loss configuration
so that phase 1 truly matches your intended YOLOPv2 baseline before moving to more advanced replacements.
