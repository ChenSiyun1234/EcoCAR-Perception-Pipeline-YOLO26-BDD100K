# DETR_GeoLane v2 patch summary

## Main changes
- Reworked detection head toward RT-/RF-DETR-style iterative box refinement.
- Reworked lane head toward CLRerNet / MapTR style learned priors + iterative curve refinement.
- Replaced rigid point-to-point lane loss with curve-aware geometry losses:
  - bidirectional point-to-curve loss
  - tangent consistency
  - smoothness regularization
- Added temporal utilities for video:
  - cross-frame lane association
  - temporal curve smoothing
  - hook for future temporal consistency training
- Hardened dataset root and lane JSON path resolution to follow the old YOLO26 conventions.
- Fixed horizontal flip label synchronization bug.

## New files
- `src/temporal_utils.py`
- `tools/apply_zip_patch.py`
- `tools/apply_zip_patch.bat`
- `IMPROVEMENTS_V2.md`
