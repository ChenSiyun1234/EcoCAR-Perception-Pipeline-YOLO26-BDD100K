# DualPathNet upgrade notes (2026-04-02)

## What was wrong in notebook00

The original notebook00 result showed three immediate symptoms:

- epoch time was extremely long because training used the full 70k/10k split with batch size 8, so one epoch already contains 8750 training steps plus 1250 validation steps;
- the lane branch stayed close to zero because it relied almost entirely on brittle early-stage polyline matching;
- both task heads were simplified, older-style implementations rather than stronger modern variants.

## Main defects found in the original implementation

### Detection head
- learned queries only, without content-aware query initialization;
- no iterative box refinement;
- no auxiliary decoder supervision;
- weak positional modeling over flattened multiscale features.

### Lane head
- absolute point prediction from scratch for every query;
- no lane-anchor prior or structured query prior;
- no overlap-style supervision, so early training depends too much on exact geometric matching;
- evaluation focused on a curve-matching F1, which is unstable and hard to interpret early.

### Loss / metric design
- lane supervision was almost entirely geometric;
- there was no thick-line overlap objective matching the visual notion of lane agreement;
- the plotted training curves were too thin and the lane metric did not match the intended use case.

## What was changed

### Detection branch upgrade
The detection head was upgraded to a more RT-DETRv2-style design:
- multiscale 2D positional encoding;
- encoder-memory-based query selection;
- iterative box refinement;
- auxiliary intermediate outputs.

### Lane branch upgrade
The lane head was upgraded to a more MapTRv2-style design:
- multiscale positional encoding;
- explicit lane-anchor priors;
- iterative point refinement around priors;
- auxiliary intermediate outputs.

### New lane supervision
A hybrid lane loss is now used:
- original curve geometry loss;
- tangent and curvature regularization;
- thick-line soft raster overlap loss at low resolution;
- lane visibility BCE;
- lane type classification.

This means the model still predicts vectorized lanes, but optimization is guided by a mask-style overlap signal that is much easier to learn from.

### New lane evaluation
Lane evaluation now also computes:
- lane_mIoU
- lane_overlap_f1

These are derived from the same thick-polyline overlap representation, so the loss and metric are aligned.

## Important honesty note

This package does **not** contain a full direct import of entire external GitHub repositories such as official RT-DETRv2 or full MapTR/StreamMapNet/LaneSegNet training stacks. Those repos are much larger and require deeper dataset / framework integration.

What is included here is a **practical in-project transplant of the most useful modern ideas** so the current codebase can be upgraded without rewriting the whole project.

## Files changed
- `src/detection_head.py`
- `src/lane_head.py`
- `src/losses.py`
- `src/metrics.py`
- `src/model.py`
- `src/trainer.py`
- `src/config.py`
- `configs/default.yaml`
- `notebooks/00_dualpath_pipeline.ipynb`

## Next recommended step
Run notebook00 again and compare:
- epoch time
- det_map50
- lane_mIoU
- lane_overlap_f1
- stability of train/val curves

If lane learning still starts too slowly, the next escalation should be:
1. train detection branch first for a short warm start,
2. freeze part of detection for several epochs,
3. raise lane overlap weight slightly,
4. optionally add a small raster lane decoder only for auxiliary supervision.
