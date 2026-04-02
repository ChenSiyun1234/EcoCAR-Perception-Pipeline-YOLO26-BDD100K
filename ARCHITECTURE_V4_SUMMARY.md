# EcoCAR Perception Pipeline v4 — Architecture Summary

## Overview

Complete architectural overhaul of the joint detection + lane segmentation pipeline.
The v4 design eliminates the COCO-80 class space mismatch that was the root cause of
detection degradation in the joint model, and replaces unnecessary complexity with a
clean, minimal architecture.

---

## Root Cause Analysis

**Problem:** The v3 joint model suffered detection degradation when the lane head was added.

**Root cause:** The detection head kept COCO's nc=80 class space. BDD100K's 10 classes
were scattered across 80 COCO channels (e.g., car at index 2, truck at index 7), leaving
70 channels as dead weight. These unused channels contributed gradient noise during joint
training, and the DualTaskNeck / teacher distillation / staged weighting were all
workarounds that never addressed the fundamental mismatch.

**Fix:** Replace the COCO-80 head with a native nc=5 vehicle detection head. This
eliminates the mismatch entirely, making the compensatory complexity unnecessary.

---

## Architecture Changes

### Detection: COCO-80 -> Native nc=5

| | v3 (old) | v4 (new) |
|---|---|---|
| Detect head | nc=80 (COCO) | nc=5 (vehicle) |
| Classes | 10 BDD mapped to COCO indices | 5 vehicle: car, truck, bus, motorcycle, bicycle |
| Weight init | Full COCO weights | Per-channel transfer from COCO for matching classes |
| Dead channels | 70 unused | 0 |

**Dropped classes:**
- person, rider, traffic light, traffic sign (non-vehicle, irrelevant for EcoCAR)
- train (~50 annotations in 70K images, causes unstable metrics)

**Class mapping (BDD -> Vehicle):**
```
BDD 2 (car)        -> Vehicle 0
BDD 3 (truck)      -> Vehicle 1
BDD 4 (bus)        -> Vehicle 2
BDD 6 (motorcycle) -> Vehicle 3
BDD 7 (bicycle)    -> Vehicle 4
```

### Lane Segmentation: Transformer Head (LightMUSTER)

- Spatial Reduction Attention (SRA) keeps computation O(N) instead of O(N^2)
- Multi-scale feature fusion from neck outputs
- Config: embed_dim=128, num_heads=4, depth=2
- Output: 160x160 binary mask (square, matching 640x640 input)

### Removed Components

| Component | Why removed |
|---|---|
| DualTaskNeck | Compensated for class mismatch; unnecessary with native nc=5 |
| Teacher distillation | Detection preservation workaround; root cause fixed |
| Staged loss weighting | Was needed to prevent early lane gradients from disrupting mismatched detection head |
| PCGrad | Gradient conflict resolution for a problem that no longer exists |
| Detection preservation | L2/teacher constraints on detection head no longer needed |

### Simplified Loss

```
total_loss = 1.0 * det_loss + 0.3 * lane_loss
```

Fixed weights, no scheduling, no teacher, no gradient manipulation.

---

## File Changes

### Core Source

| File | Change |
|---|---|
| `src/utils/class_map.py` | **Rewritten.** Vehicle-only class definitions, BDD-to-vehicle remapping |
| `src/multitask_model.py` | **Rewritten (v4).** Native nc=5 head replacement, no DualTaskNeck, shared neck features |
| `src/models/multitask.py` | Thin re-export of `src.multitask_model` |
| `src/dataset_utils.py` | Updated: vehicle-only class defs, nc=5 YAML generation |
| `src/losses/det_loss.py` | Uses `remap_targets_bdd_to_vehicle` (was COCO remap) |
| `src/metrics/detection.py` | Simplified: all classes are vehicle, per-class AP, single mAP metric |

### Configs

| File | Change |
|---|---|
| `configs/default.yaml` | **Rewritten.** nc=5, 160x160 masks, transformer lane head, SGD lr=0.002, fixed loss weights |
| `configs/bdd100k_joint.yaml` | Updated: nc=5, vehicle names, 160x160 masks, transformer head |
| `configs/bdd100k_yolo.yaml` | Updated: nc=5, vehicle names |

### Entry Points

| File | Change |
|---|---|
| `train.py` | Updated imports, default 160x160 masks, removed old CLI args |
| `eval.py` | **Rewritten.** Vehicle-only metrics, per-class AP, no COCO remapping |
| `infer.py` | **Rewritten.** Vehicle classes, 5-color palette, direct class IDs |

### Notebooks

| Notebook | Change |
|---|---|
| `03_bdd100k_training.ipynb` | Vehicle-only detection training, SGD lr=0.002 |
| `08_joint_training.ipynb` | **Rewritten.** v4 joint training with inline config |
| `09_joint_inference.ipynb` | **Rewritten.** Vehicle detection + lane eval, per-class AP |
| `10_video_inference.ipynb` | **New.** End-to-end video pipeline from Drive/EcoCAR/video/ |
| `11_gpu_profiling.ipynb` | **New.** NVML-based per-second GPU utilization during video inference |

### Files No Longer Used (kept for reference)

- `src/dual_task_neck.py` — DualTaskNeck (removed from model)
- `src/task_interaction.py` — TaskInteractionBlock (removed from model)
- `src/models/dual_neck.py` — DualTaskNeck module
- `src/models/interaction.py` — TaskInteractionBlock module
- `src/losses/multitask.py` — StagedWeighting, PCGrad (config uses "fixed" strategy now)

---

## Model Architecture Diagram

```
Input Image (640x640)
        |
  YOLO26-S Backbone (COCO pretrained)
        |
  YOLO26-S Neck (FPN/PAN)
        |
   +----+----+
   |         |
Detect Head  Transformer Lane Head
(nc=5)       (LightMUSTER, SRA)
   |         |
5 vehicle    160x160 lane mask
classes      (binary)
```

---

## Training Configuration

```yaml
backbone: yolo26s (COCO pretrained)
optimizer: SGD, lr=0.002, momentum=0.937, backbone_lr_scale=0.1
scheduler: cosine, warmup=3 epochs, min_lr_ratio=0.01
loss: fixed weights (det=1.0, lane=0.3)
masks: 160x160 (square)
batch_size: 16
epochs: 40
patience: 8
```

---

## Known Remaining Issues

1. **Lane masks not yet generated**: BDD100K per-frame JSONs lack poly2d data.
   The consolidated label files (from the BDD100K GitHub release) are needed.
   This blocks NB07, NB08, and NB09 lane evaluation.

2. **Dataset size**: Only ~17K train images extracted vs 70K available in BDD100K.
   Full dataset extraction would improve training.
