# RTX PRO 6000 batch-boost profile

This version keeps the YOLOPX-style architecture/loss path and raises the default training batch for RTX PRO 6000.

Changed:
1. `stage1/configs/yolopx_vehicle_lane_baseline.yaml`
   - `BATCH_SIZE_PER_GPU: 64`
   - `LR0: 0.002`
2. `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb`
   - `BATCH_OVERRIDE = 64`
   - `LR0_OVERRIDE = 0.002`
   - worker cap relaxed to 8-12

Fallback settings:
- If batch 64 is slower in samples/s or unstable, use:
  - `BATCH_OVERRIDE = 48`
  - `LR0_OVERRIDE = 0.0015`
- If still unstable, use:
  - `BATCH_OVERRIDE = 32`
  - `LR0_OVERRIDE = 0.001`

Do not judge speed from the first batch. Check after batch 20 or later.
