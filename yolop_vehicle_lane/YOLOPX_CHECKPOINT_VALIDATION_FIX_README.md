Checkpoint and validation fix for YOLOPX stage1:
1. `VAL_FREQ` is now set to 1 by default in `stage1/configs/yolopx_vehicle_lane_baseline.yaml`.
2. Validation is scheduled by completed epoch count: `(epoch + 1) % VAL_FREQ == 0`.
3. `latest.pth` is saved after every completed epoch, even if validation is skipped.
4. Visible epoch snapshots are saved for epochs 1, 2, 3 and then every 5 epochs as `epoch_XXXX.pth`.
5. Best checkpoints are still selected only from validation metrics: `best_det.pth`, `best_lane.pth`, `best_joint.pth`, and `best.pth`.
