# Stage1 audit + repair — v5

Scope: `yolop_vehicle_lane/stage1/` end-to-end, against YOLOP upstream
(`external_repos/YOLOP/`) and the YOLOPv2 paper (`external_repos/2208.11434v1.pdf`).
The public YOLOPv2 repo is inference-only; training-side choices are
labelled `[INFERRED]` wherever they are reconstructed rather than
source-recovered.

## 1. Current stage1 status

The stage1 run the user uploaded is **structurally broken**, not just slow:

| metric | epoch 0 | 1 | 2 | 3 | 12 |
|---|---:|---:|---:|---:|---:|
| mAP@0.5 | 0.0002 | 0.0005 | 0.0644 | → collapses | 0.0001 |
| mAP@[.5:.95] | 0.0000 | 0.0001 | 0.0151 | → collapses | 0.0000 |
| LL_IoU | **0.0061** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| LL_Acc | 0.5734 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Lane IoU hits zero at epoch 1 and never recovers. Detection mAP flutters
and eventually collapses too. This is consistent with two compounding
defects both in YOLOP upstream and still present in our stage1 code:
a non-differentiable lane IoU loss, and a 0-indexed vs. 1-indexed
warmup-iter mismatch that silently sets LR=0 for all weight groups during
epoch 0 and drives the bias groups around wildly.

## 2. Bugs found (`BUG — MUST FIX`)

| # | file : line | bug |
|---|---|---|
| B1 | `lib/core/function.py:60` | `num_iter = i + num_batch * (epoch - 1)` is YOLOP's 1-indexed form, but our loop is 0-indexed (`BEGIN_EPOCH=0`). At epoch 0, `num_iter` is **negative**; `np.interp` clamps LR to 0 for non-bias groups and to `WARMUP_BIASE_LR=0.1` for the bias group. Only biases trained during epoch 0. |
| B2 | `lib/core/loss.py:119-132` (old) | Lane IoU loss computed with `torch.max(...)` (argmax → no gradient) and `SegmentationMetric(...).cpu()` (graph cut). Contribution to backprop: **zero**. Inherited verbatim from YOLOP upstream. |
| B3 | `lib/utils/augmentations.py:162-163` (`letterbox`) | Lane mask resized with `cv2.INTER_LINEAR` — bilinear on a thin binary signal yields intermediate values that the downstream `threshold(>1)` binarizer truncates unpredictably. |
| B4 | `lib/utils/augmentations.py:253-254` (`load_mosaic`) | Same — lane mask used the image's interp choice (`INTER_AREA` or `INTER_LINEAR`). |
| B5 | `lib/utils/augmentations.py:67-71` (`random_perspective`) | `cv2.warpAffine` / `cv2.warpPerspective` default to `INTER_LINEAR`; no interp flag was passed for the lane mask. |
| B6 | `lib/dataset/AutoDriveDataset.py:271-273` | Lane mask resize uses `INTER_AREA` / `INTER_LINEAR` chosen by the image's scale factor. |
| B7 | `lib/core/function.py:355-357` (old) | Validation "Speed" line computed `t_inf / seen * 1e3` where `t_inf` was the **last batch's** wall clock, not the cumulative. The printed ms/img scaled inversely with validation-set size. |
| B8 | `lib/core/function.py:229-236` (old) | `validate()` plot branch upsamples 2-channel **logits** with `mode='bilinear'` and then argmax's them — mixes classes at boundaries. `scale_factor=int(1/ratio)` also truncates to an integer (OK at 1/2 = 2, wrong at 1/0.75 = 1). |
| B9 | `notebooks/02 cell 9` (old) | `is_best = map50 > best_map or ll_iou > best_ll_iou` saved a single `best.pth` that could be overwritten by a det-task regression that happened to improve lane IoU (or vice versa), breaking downstream comparisons. |

## 3. Fixes applied

| fix | file | substance |
|---|---|---|
| F-B1 | `lib/core/function.py` | `num_iter = i + num_batch * epoch` with an inline comment explaining the 0-indexed convention. |
| F-B2 | `lib/core/loss.py` | Replaced the CPU-argmax lane IoU block with a **differentiable soft-IoU** on `sigmoid(fg_logits)` vs. `targets[1][:, 1]`, with padding rows/cols excluded so padding does not bias the denominator. |
| F-B3…F-B6 | `lib/utils/augmentations.py`, `lib/dataset/AutoDriveDataset.py` | Every lane-mask resize / warp now uses `cv2.INTER_NEAREST` (or the `flags=cv2.INTER_NEAREST` warp kwarg). Image interp is unchanged. |
| F-B7 | `lib/core/function.py` | Validation speed now reports `T_inf.avg * 1e3` / `T_nms.avg * 1e3`, which are already per-image averages. |
| F-B8 | `lib/core/function.py` | In the plot branch we argmax FIRST, then upsample the int mask with `mode='nearest'` and `scale_factor=float(1/ratio)` (no truncation). |
| F-B9 | `stage1/notebooks/02 cell 9` | Explicit `best_det.pth` / `best_lane.pth` / `best_joint.pth` (`joint = 0.5*mAP50 + 0.5*ll_iou`). `best.pth` kept as alias of `best_joint.pth` for stage2/3 compatibility. |
| +resume guard | `stage1/notebooks/02 cell 8` | Raise a clear error when `latest.pth` shape-mismatches the current model instead of silently skipping weights. |
| +speed | `stage1/notebooks/02 cell 3` | TF32 on matmul + cudnn; `cudnn.benchmark = True`; DataLoader `persistent_workers=True`, `prefetch_factor=2`. |
| +run mode | `stage1/notebooks/02 cells 3+4+6+9` | `RUN_MODE in {'smoke','full'}`. Smoke mode → 16 train + 16 val samples, 2 epochs, plots on. Full mode → plots only at epoch 0. |
| +diagnostics | `stage1/notebooks/02b_stage1_diagnostics.ipynb` | New notebook: per-image fg ratio, sigmoid(fg_logit) stats, overfit-16-samples on lane-only, 4-sample image/GT/pred overlay grid. |

## 4. `PARTIAL / INFERRED` decisions (explicitly not "bugs")

| # | area | note |
|---|---|---|
| P1 | E-ELAN `groups=2` from stride ≥ 16 | `[INFERRED]` — paper says "ELAN with more efficient structure", does not publish group count. 2 lands us in the published 38-40M param band. |
| P2 | SGDR `T_0=100`, `T_mult=1` over 300 epochs | `[INFERRED]` — "cosine annealing with warm restart" is the paper's exact wording; actual period not published. |
| P3 | Focal γ on cls/obj/lane-seg = 1.5 | `[INFERRED]` — YOLOv5/YOLOP default. Paper says "focal" with no γ. |
| P4 | LL_DICE_GAIN = 0.5 | `[INFERRED]` — chosen to keep dice magnitude ≲ focal term. |
| P5 | Centerline pairing heuristic (y-overlap ≥ 0.5, lateral ≤ 0.12·W) | `[INFERRED]` — BDD100K does not link the two sides of a lane, and the paper does not release its pairing tool. See `lib/utils/lane_render.py`. |
| P6 | Mosaic + MixUp recipe | `[INFERRED]` — YOLOv7 hyp.scratch defaults; YOLOPv2 inherits YOLOv7 lineage. |
| P7 | `predict()` lane output contract (1-ch sigmoid at H/2) | `[INFERRED]` from `external_repos/YOLOPv2/utils/utils.py::lane_line_mask`, which calls `torch.round(sigmoid(ll)).squeeze(1)`. |

## 5. `MATCH` (faithful to upstream / paper)

- Backbone: Focus → Conv → E-ELAN stages to stride 32, SPP neck,
  FPN + PAN as `MCnet_0` — verified cell-by-cell against
  `external_repos/YOLOP/lib/models/YOLOP.py:MCnet_0`.
- Detection head: YOLOv5-style `Detect` with 3 anchor-free scales,
  `nc` wired from `cfg.MODEL.NC` at build time (fix applied in v4).
- Stage1 class taxonomy: `STAGE1_VEHICLE_MERGED` = {car, bus, truck,
  train → 0}; motorcycle/bicycle dropped. Matches paper's "vehicle"
  class.
- Mask widths: train 8, val 2 — paper §3.
- Train letterbox 640×640, val letterbox 640×384 (rectangular,
  `auto=False`). Paper §3.
- Train-time lane output contract: 2-channel raw logits at full
  input resolution → single `BCEWithLogitsLoss` call. Consistent with
  `MultiHeadLoss` and `validate()`.

## 6. Audit matrix (short form)

| dimension | verdict |
|---|---|
| model topology | MATCH |
| backbone / neck | MATCH (E-ELAN groups = `[INFERRED]`) |
| lane decoder (deconv) | MATCH |
| detection loss | MATCH |
| **lane seg BCE + focal** | MATCH |
| **lane seg dice (hybrid)** | MATCH |
| **lane seg IoU loss** | **BUG — fixed to soft-IoU** |
| optimizer / scheduler | PARTIAL — SGDR `T_0` `[INFERRED]` |
| **warmup iter indexing** | **BUG — fixed** |
| train vs eval output contract | MATCH (2-ch logits @ train, 1-ch sigmoid H/2 @ `predict()`) |
| class protocol | MATCH (stage1_vehicle_merged) |
| augmentation path | MATCH (Mosaic/MixUp `[INFERRED]`) |
| **lane mask interpolation** | **BUG — fixed to NEAREST** |
| dataset path resolution | MATCH |
| lane mask generation | PARTIAL — pairing heuristic `[INFERRED]` |
| **checkpoint selection** | **BUG — fixed to 3-way (det/lane/joint)** |
| notebook flow | REPAIRED |
| drive persistence | MATCH |
| **val speed reporting** | **BUG — fixed** |

## 7. Output summary for the broken run (reference)

> stage1 run logged with the pre-v5 code on
> `yolopv2_best_row.yaml`. After 12 epochs detection mAP@0.5 was
> 0.0001, lane IoU was exactly 0.0000. Two cumulative defects
> explain the collapse:
>
> 1. The iter-index formula was YOLOP's 1-indexed form applied to
>    our 0-indexed loop, so during epoch 0 every non-bias weight
>    saw LR=0 while biases saw a decaying pull toward
>    `WARMUP_BIASE_LR=0.1`. The model could not actually learn
>    from the first epoch.
> 2. The lane IoU loss was computed with `torch.max` + numpy
>    `SegmentationMetric`, severing the graph; its gradient
>    contribution was zero. The only surviving lane signal was the
>    BCE term, which is dominated by the ~92 % background and
>    collapses to an all-background prediction — exactly what we
>    see with LL_Acc=0 and LL_IoU=0.
>
> v5 replaces the iter index with `i + num_batch * epoch`, swaps
> the lane IoU for a differentiable soft-IoU, and switches every
> lane-mask resize path to NEAREST. No hyperparameter change.

## 8. Things not touched (by design)

- Taxonomy stayed 1-class merged vehicle.
- Project root stayed `/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane`.
- Stage boundaries: no stage2 ideas leaked into stage1 code.
- A5000 profile notebook unchanged (already stage-local).

## 9. Remaining uncertainties

- YOLOPv2 paper §3 says "focal + dice" for the best row without
  specifying weights. Our LL_DICE_GAIN=0.5 is a reasonable guess
  that keeps dice ≲ focal in scale; it has not been tuned.
- E-ELAN `groups` value is a best-fit against the paper's reported
  parameter count, not a source recovery.
- SGDR period (`T_0`, `T_mult`) is the biggest free variable; the
  paper just writes "cosine annealing with warm restart". `T_0=100`
  / `T_mult=1` gives two restarts in the 300-epoch budget, which
  is a defensible default but not source-recovered.
- `LANE_TRAIN_THICKNESS=8` / `LANE_TEST_THICKNESS=2` are faithful
  to paper §3, but the paper does not publish its **centerline
  pairing algorithm** — ours is a greedy y-overlap + lateral
  distance heuristic. Different pairing choices can shift lane IoU
  by a few points independent of model quality.
