# YOLOPX stage-1 baseline

This package adds a YOLOPX-derived baseline to the existing stage-1 vehicle + lane pipeline.

What was added or repaired:
1. `lib/models/yolopx_common.py` — YOLOPX/YOLOv7 ELAN building blocks copied into a separate namespace.
2. `lib/models/yolopx_head.py` — YOLOX-style decoupled anchor-free detection head.
3. `lib/models/yolopx_baseline.py` — two-task YOLOPX model: object detection + lane segmentation only. The lane output now uses the same `Conv(8, 2, 3, 1) + seg_head('sigmoid')` pattern as YOLOPX.
4. `lib/core/loss.py` — YOLOPX branch in `get_loss()`, using dynamic-k YOLOX detection loss and YOLOPX lane focal BCE + Tversky loss.
5. `stage1/configs/yolopx_vehicle_lane_baseline.yaml` — YOLOPX-aligned stage-1 config.
6. `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb` — training notebook with `CONFIG = 'YOLOPX'`.

Training defaults:
1. Tasks: object detection + lane segmentation.
2. Removed task: drivable-area segmentation.
3. Detection classes: one merged `vehicle` class, matching the stage-1 protocol.
4. Optimizer: AdamW.
5. lr0: 0.001.
6. Batch size: 32 by default. If RTX PRO 6000 memory becomes tight, set `BATCH_OVERRIDE = 16` or `24` in the notebook.
7. Warmup: 3 epochs.
8. YOLOPX loss: `0.02 * (5 * IoU + focal objectness) + 0.2 * lane focal BCE + 0.2 * lane Tversky`.
9. Augmentation follows the public YOLOPX config: flip, HSV, random perspective with scale 0.25, rotation 10, translation 0.1; mosaic is disabled by default and MixUp probability is left at 0.15 for parity with the config field.

Run from Colab:
1. Copy this project to `/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane`.
2. Run `stage1/notebooks/00_rebuild_dataset_and_lane_cache.ipynb` if the dataset cache is not ready.
3. Run `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb`.
