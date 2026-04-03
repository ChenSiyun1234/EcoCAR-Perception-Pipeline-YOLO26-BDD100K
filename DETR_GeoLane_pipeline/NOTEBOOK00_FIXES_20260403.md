# Notebook00 analysis and fixes (2026-04-03)

## Problems identified from the current run

1. **Detection is learning much more slowly than lane prediction.**
   - By epoch 14, lane mIoU reaches 42.1%, while detection mAP50 is only 16.4%.
   - This is a classic sign that the joint task balance is tilted toward the lane branch in the early stage.

2. **Validation is too expensive and is run every epoch.**
   - With 10,000 validation images, full validation every epoch adds a large fixed time cost.
   - The runtime staying around 2733–2801s/epoch suggests validation is a major contributor.

3. **Detection label schema may be inconsistent with the loader.**
   - In notebook00, class statistics show `motorcycle=0` and `bicycle=0`.
   - The old loader assumed original BDD100K IDs, but many YOLO exports already use remapped 0..4 class IDs.
   - If the loader interprets the wrong schema, some classes are silently ignored.

4. **Training can be interrupted, but resume behavior was missing.**
   - Checkpoints were being saved, but the trainer did not automatically restore the last checkpoint and continue.

5. **Validation metrics fluctuate after epoch 14.**
   - Example: lane mIoU 42.1 -> 42.3 -> 40.5, detection mAP50 16.4 -> 14.2 -> 10.8.
   - This suggests optimization instability and/or noisy joint supervision.

## Fixes added

### 1) Automatic checkpoint resume
- Trainer now checks for `weights/last.pt` first, then `weights/best_joint.pt`.
- It restores:
  - model state
  - optimizer state
  - AMP scaler state
  - best scores
  - history CSV
- If no checkpoint exists, training starts from scratch.

### 2) Detection label schema auto-detection
- Added automatic label schema inference:
  - `bdd_original`
  - `vehicle_remapped`
  - `expanded_remapped`
- This prevents silent class dropping when labels are already remapped.

### 3) Early task balancing warmup
- Added configurable early-stage task weights:
  - `det_task_warmup_weight`
  - `lane_task_warmup_weight`
  - `task_warmup_epochs`
- Notebook00 now uses a lighter lane weight in the first 8 epochs to help detection catch up.

### 4) Validation throttling support
- Added `val_interval` and `max_val_batches` config hooks.
- Notebook00 is updated to validate every 2 epochs by default.
- This reduces wasted time while keeping checkpointing and metric tracking intact.

### 5) Slight dataloader efficiency cleanup
- Enabled `persistent_workers` when `num_workers > 0`.
- Kept non-blocking transfers in trainer.

## Files changed
- `src/dataset.py`
- `src/trainer.py`
- `src/config.py`
- `configs/default.yaml`
- `notebooks/00_dualpath_pipeline.ipynb`

## Suggested next experiment

If you want the safest next run:
- keep the current architecture
- use the new auto-resume trainer
- use `val_interval=2`
- keep `lane_task_warmup_weight=0.35`
- continue from the latest checkpoint instead of restarting

