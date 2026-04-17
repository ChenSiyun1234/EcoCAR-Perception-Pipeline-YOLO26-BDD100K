# Repair Pass v2 — targeted YOLOPv2 alignment fixes

This pass addressed the five specific gaps called out against the
previous repair. No scope creep, no DETR/stage-2 changes.

## 1. Lane GT generation — real centerline reconstruction (§1)

**Before**: `lib/utils/lane_render.py::render_lane_mask` drew every
BDD `poly2d` independently at `line_thickness`. Docstring itself
admitted this was not the paper's "centerline between two annotated
lines" and that over-representation scaled with thickness.

**Fix** (`lib/utils/lane_render.py`):

- New helpers: `_resample_polyline_uniform`, `_polyline_y_overlap`,
  `_mean_lateral_distance`, `_pair_lane_boundaries`, `_centerline`.
- `render_lane_mask` now:
  1. Resamples every lane polyline to 32 equally-arc-length points.
  2. Builds candidate pairs whose vertical-extent overlap ≥ 50 % and
     whose mean lateral distance ≤ `pair_max_dist_ratio × img_width`
     (default 0.12 × 1280 ≈ 154 px).
  3. Greedy-assigns closest pairs first.
  4. Draws the **midline** of every matched pair.
  5. Falls back to direct-draw for unpaired polylines (road curbs,
     isolated shoulder markings — these *are* their own centerlines).
- `convert_bdd_lanes_to_masks` now forwards `pair_centerlines` and
  `pair_max_dist_ratio` parameters.
- `notebook 00` cell 5 calls the renderer with `PAIR_CENTERLINES=True`,
  `overwrite=True`, and the paper's per-split widths (train=8, val=2).
- The compressed archive persistence (`EcoCAR/datasets/bdd100k_vehicle5.tar.gz`)
  path in cell 8 is unchanged; downstream notebooks keep working.

`[INFERRED]`: the paper does not publish its pairing tool. The greedy
proximity + y-overlap heuristic is the closest reproducible
approximation from public BDD data.

## 2. Lane forward / loss pipeline — double-sigmoid bug removed (§2)

**Before**: YOLOP upstream code (and our inherited baselines) applied
`nn.Sigmoid()` to the lane branch inside `forward()` *and* used
`BCEWithLogitsLoss`, which sigmoids internally — a double-sigmoid.

**Fix**:

- `lib/models/yolopv2_baseline.py::MCnetV2.forward` returns raw logits.
- `lib/models/yolop_baseline.py::MCnet.forward` same treatment
  (documented as a deliberate deviation from YOLOP upstream to kill
  its bug).
- Both baselines gain a `.predict()` helper that returns sigmoid'd
  probabilities for inference / export.
- `lib/core/loss.py::MultiHeadLoss._forward_impl` docstring now
  explicitly states `predictions[1]` is logits. `BCEseg` is built as
  `BCEWithLogitsLoss` (optionally focal-wrapped via `LOSS.LL_FL_GAMMA`).
- Argmax paths in `lib/core/function.py::validate` are unaffected
  (monotonic transform preserves argmax).
- **Hybrid focal+dice** added: new `LOSS.LL_DICE_GAIN` knob; the
  MultiHeadLoss computes `1 - Dice(sigmoid(pred_fg), target_fg)` and
  adds it to the total. Default 0 (focal-only baseline); YOLOPv2 §3
  hybrid ablation = set > 0.

## 3. Val resolution is now truly 640×384 (§3)

**Before**: notebook 02 computed `val_size = max(cfg.TEST.IMAGE_SIZE)`
which collapsed `[640, 384]` to `640`. The dataset then called
`letterbox(new_shape=640, auto=True, stride=32)` which happened to
produce 640×384 for BDD's 1280×720 aspect *by accident* (stride
rounding). Any other input aspect would have broken.

**Fix**:

- `lib/dataset/AutoDriveDataset.py::__getitem__` now accepts either
  an `int` (square, auto=True letterbox — legacy path) or a tuple
  `(H, W)` (explicit shape, `auto=False`). When a tuple is passed the
  output is guaranteed to be exactly that shape.
- `notebook 02` cell 6 passes `val_size = (H, W) = (384, 640)` from
  `cfg.TEST.IMAGE_SIZE`. Prints the realized `C×H×W` of a sample so
  you can eyeball it.
- `notebook 03` cell 4 does the same for evaluation.

## 4. Scheduler — documented + SGDR ablation toggle (§4)

**Before**: iter-based linear warmup for `WARMUP_EPOCHS`, then
per-epoch cosine annealing. This IS a valid reading of the paper's
"cosine annealing with warm restart" (interpreting the "warm restart"
as the initial linear ramp from 0). Not wrong, but not documented.

**Fix**:

- `notebook 02` cell 7 now prints the scheduler policy string:
  `cosine-annealing + linear warmup (YOLOP default)` by default.
- New `TRAIN.SGDR` / `TRAIN.SGDR_T0` / `TRAIN.SGDR_TMULT` knobs: set
  `SGDR=True` to swap in `CosineAnnealingWarmRestarts`. We don't
  default to this because the paper publishes no `T_0`/`T_mult`; that
  variant is an ablation only.
- `AUDIT_YOLOPV2_ALIGNMENT.md` and the scheduler docstring both mark
  this as `[INFERRED]`.

## 5. Phase-1 still clean (§5)

No regression. Stage-2 DETR lane code remains quarantined under
`stage2/`. The phase-1 factory (`lib/models/__init__.py`,
`lib/core/__init__.py`, `lib/dataset/__init__.py`) imports zero
stage-2 symbols. Notebook list at the top of the phase-1 tree:

```
notebooks/00_rebuild_dataset_and_lane_cache.ipynb
notebooks/01_augmentation_lab.ipynb
notebooks/02_train_yolopv2_vehicle_lane_baseline.ipynb
notebooks/03_eval_and_backbone_ablation.ipynb
notebooks/07_a5000_video_profile.ipynb
```

Stage-2 notebooks live at `stage2/notebooks/` and are opt-in.

## What remains `[INFERRED]`

Labelled in code; update if/when YOLOPv2 source is released:

1. **E-ELAN `groups` value** — set to 2 at stride ≥ 16, param count
   lands in paper's 38-40 M band.
2. **Centerline pairing rule** — paper doesn't publish its tool; we
   use the greedy proximity heuristic above.
3. **Focal γ (both det and lane)** — paper says focal, not γ; we use
   1.5 (YOLOv5/YOLOP default).
4. **Scheduler warm-restart semantics** — `[INFERRED]` either way
   (single cosine vs SGDR); default is single cosine.
5. **Hybrid focal+dice weighting** — paper mentions ablation, doesn't
   disclose weight; we expose `LOSS.LL_DICE_GAIN` and default to 0.

## Acceptance check

- [x] Lane GT is reconstructed centerlines, not direct poly2d raster.
- [x] Lane loss is a single-sigmoid logits path (BCEWithLogits / focal
      on logits; optional dice on sigmoided probabilities).
- [x] Validation uses rectangular 640×384 letterbox with `auto=False`.
- [x] Scheduler policy is documented and optionally swappable.
- [x] Phase-1 baseline path unpolluted by stage-2 research code.
- [x] Notebook outputs are still persisted as a single `.tar.gz` under
      `EcoCAR/datasets/`.
- [x] A5000 is still the profiling target in notebook 07.
