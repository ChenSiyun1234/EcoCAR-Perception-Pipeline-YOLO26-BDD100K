# Phase-1 Reconstruction Audit — yolop_vehicle_lane

**Purpose:** audit the current reconstruction against the upstream originals
(`external_repos/YOLOP`, `external_repos/YOLOPv2`) and the YOLOP / YOLOPv2
papers, and document what was repaired so phase-1 is a defensible
YOLOPv2-style baseline rather than a relabelled YOLOP.

## TL;DR

Before repair, the reconstruction **was YOLOP with the drivable-area head
removed**, not YOLOPv2. The model block config was a near-verbatim copy of
YOLOP's `MCnet_0`, the decoder was YOLOP's progressive upsample, the
augmentation path lacked Mosaic / MixUp that the YOLOPv2 paper inherits from
its YOLOv7 lineage, and notebook 07 targeted H100 despite the deployment
GPU being A5000.

After repair, phase-1 has two explicit baselines side-by-side:

| baseline | file | honest label |
|---|---|---|
| `YOLOP-vehicle-lane` | `lib/models/yolop_baseline.py` | YOLOP minus DA |
| `YOLOPv2-vehicle-lane` (inferred) | `lib/models/yolopv2_baseline.py` | YOLOv7-ELAN reconstruction, decoder + detect output format matched to `external_repos/YOLOPv2/demo.py` |

Any YOLOPv2 component that could not be read directly from
`external_repos/YOLOPv2` (which ships `torch.jit`-scripted weights, no
Python architecture source) is **marked `# [INFERRED]` in the code and
listed at the bottom of this document**.

---

## 1. Upstream evidence used

### `external_repos/YOLOP/lib/models/YOLOP.py`
- Python source for the full `MCnet_0` / `MCnet_SPP` / `MCnet_share` block
  configs. Shared encoder + FPN + detection PAN + **two** progressive
  upsample decoders (DA + lane), each ending in `Conv(8, 2, 3, 1)` and
  squashed by a `Sigmoid` at forward time.
- `common.py` supplies `Focus`, `Conv`, `BottleneckCSP`, `SPP`, `Detect`,
  `Concat`, `SharpenConv`.

### `external_repos/YOLOP/lib/dataset/AutoDriveDataset.py`
- Augmentation pipeline: letterbox → `random_perspective` → HSV →
  horizontal flip. **No Mosaic. No MixUp.** No cutout at train time.
- Target tuple during training is `[labels_out, seg_label, lane_label]`,
  with both seg targets as 2-channel stacks (`bg`, `fg`).

### `external_repos/YOLOPv2/demo.py` + `utils/utils.py`
- The released repository is **inference-only**: `model = torch.jit.load(weights)`.
  There is no Python architecture source in the release. The *only*
  ground-truth architectural signal is the output shape contract:
  ```python
  [pred, anchor_grid], seg, ll = model(img)
  pred                          # 3 scales, [B, 3*(5+80), ny, nx]
  anchor_grid                   # 3 scales of anchor scales
  seg [B, 2, H/2, W/2]          # 2-ch softmax driving area
  ll  [B, 1, H/2, W/2]          # 1-ch sigmoid lane line — rounded in demo
  ```
- `split_for_trace_model` confirms detection uses 3 YOLOv5-style anchored
  scales with strides `[8, 16, 32]` and decodes `xy = (sig*2-0.5+grid)*stride`
  and `wh = (sig*2)**2 * anchor_grid`.
- `letterbox` default `new_shape=(640, 640)`, `auto=True`, `stride=32`.
- `driving_area_mask` and `lane_line_mask` both interpolate with
  `scale_factor=2` and crop `[:, :, 12:372, :]` — consistent with a seg
  head that emits half-resolution feature maps of a letterboxed 640×384
  input.

### YOLOPv2 paper (`arXiv:2208.11434`)
- Backbone: "more efficient ELAN structures" — E-ELAN from YOLOv7.
- Neck: SPPCSPC + PAN, per YOLOv7 lineage.
- Decoders: the paper explicitly names **deconvolution (transpose-conv)
  upsampling** in the segmentation decoders.
- Lane prediction is treated as a binary mask task.
- 38.9 M parameters total, 91 FPS V100 at 640.

### YOLOP paper
- YOLOP's lane decoder is progressive `Upsample + Conv + BottleneckCSP`.
  No deconvolution. Two-channel softmax.

---

## 2. File-by-file audit (state **before** repair)

| # | file | upstream origin | verdict | notes |
|---|------|----------------|---------|-------|
| M1 | `lib/models/yolop_vehicle_lane.py` | **YOLOP `MCnet_0`** | YOLOP, relabelled | `Focus + BottleneckCSP + SPP + Upsample neck + 2-ch lane`. No ELAN, no SPPCSPC, no transpose-conv. Only real changes vs YOLOP: DA branch deleted, `Detect` classes 1→5. Filename implies YOLOPv2 — misleading. |
| M2 | `lib/models/common.py` | YOLOP `common.py` | faithful | Direct copy with `BN_MOMENTUM` restored. Does **not** provide ELAN / SPPCSPC. |
| D1 | `lib/dataset/AutoDriveDataset.py` | YOLOP `AutoDriveDataset.py` | faithful-plus | 3-tuple→2-tuple adaptation (DA removed) is clean. Path-layout resolver is a genuine improvement over the upstream. Augmentation path is **identical to YOLOP** — still no Mosaic/MixUp. |
| D2 | `lib/dataset/bdd.py` | YOLOP `bdd.py` | faithful-plus | Vehicle-only `id_dict` (5 classes) and lane-mask-driven index are documented changes. |
| D3 | `lib/utils/augmentations.py` | YOLOP `augmentations.py` | faithful, trimmed | `random_perspective` / `letterbox` / `cutout` converted to 2-tuple. Still no `load_mosaic` / `load_mixup` helper. |
| L1 | `lib/core/loss.py` | YOLOP `loss.py` | faithful | Removes DA terms; keeps YOLOP's detection CIoU + BCE, lane BCE + IoU. Matches YOLOP losses, not YOLOPv2 — but YOLOPv2 paper does not disclose training losses explicitly, so parity with YOLOP is the conservative choice. |
| L2 | `lib/core/function.py` | YOLOP `function.py` | faithful | 3-head→2-head rewiring is correct. |
| N1 | `notebooks/02_train_yolopv2_vehicle_lane_baseline.ipynb` | n/a | **misleading filename** | First-cell markdown correctly says "YOLOP-style baseline" but filename says "yolopv2". Users reading only the filename get a false impression. |
| N2 | `notebooks/07_h100_video_profile.ipynb` | n/a | **wrong GPU target** | Uses H100 SKU + 989 TFLOPS peak. Deployment target is A5000. |
| N3 | `notebooks/04_modern_backbone_neck_upgrade.ipynb` | n/a | placeholder | Prints a TODO; does nothing real. Valid as a future-work stub; label as such. |
| N4 | `notebooks/05_lane_head_and_loss_upgrade.ipynb` | n/a | placeholder | Same. |

## 3. Specific concerns from the user's brief

**A. Model definition** — confirmed: before repair, `yolop_vehicle_lane.py`
was YOLOP with DA removed. After repair, a second model
`yolopv2_baseline.py` provides the closest defensible reconstruction.

**B. Augmentation path** — confirmed: Mosaic / MixUp were not in the
dataloader before repair. After repair, they are implemented behind
config flags `DATASET.MOSAIC` and `DATASET.MIXUP`, with probabilities that
default to YOLOv7's recipe. They are marked `[INFERRED]` because YOLOPv2
does not publish training code.

**C. Lane decoder + loss** — confirmed: YOLOP-style progressive upsample
before repair. After repair, the YOLOPv2 baseline uses transpose-conv
upsampling (paper text) and a 1-channel sigmoid output (demo contract).

**D. Training / eval reproducibility** — phase-1 runs are pinned via
`configs/yolopv2_vehicle_lane_baseline.yaml`. One YAML = one seed +
one input size + one LR schedule + one aug policy + one loss stack.

**E. Dataset flow** — the packaged / raw / lane-JSON fallbacks in
`AutoDriveDataset._resolve_split_paths` and `drive_dataset.py` are
preserved end-to-end.

**F. Paths** — audit found no stale `EcoCAR-Perception-Pipeline-*` refs
after the earlier restructuring; all current notebooks point at
`/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane`.

**G. A5000 profiling** — notebook 07's GPU dict now defaults to A5000.

## 4. Components that could NOT be reproduced verbatim (`[INFERRED]`)

These are decisions made from the YOLOPv2 paper + YOLOv7 lineage + demo.py
output contract, **not** from reading YOLOPv2 source. They are marked
`[INFERRED]` in code.

1. **E-ELAN block topology** — paper says "more efficient ELAN". The exact
   number of 3×3 branches and the expansion inside E-ELAN are taken from
   YOLOv7's `yolov7.yaml` ELAN: two parallel 1×1 branches, one passing
   through four 3×3 convs, concat of 4 intermediate features, 1×1 fuse.
2. **SPPCSPC** — taken from YOLOv7's implementation: CSP-wrapped SPP with
   `k=(5, 9, 13)`. YOLOPv2 paper does not reveal its SPP variant; SPPCSPC
   is YOLOv7's default and preserves the parameter budget the paper claims.
3. **Channel schedule** — chosen to land the backbone+neck+head around
   38–40 M params (YOLOPv2 reports 38.9 M). Exact channels are a guess; it
   is within the conservative reconstruction instruction.
4. **Deconvolution decoder** — paper mentions "deconvolution". We use
   `ConvTranspose2d(stride=2)` in three stages (64→32→16 channels), ending
   in `Conv(16, 1, 1, 1)` for the lane head. Driving-area head is kept in
   `lib/models/yolopv2_baseline.py` but disabled for this project — lane
   is the retained task.
5. **Training recipe** — Mosaic p=1.0 below 75% of epochs and MixUp p=0.15
   are YOLOv7 defaults. YOLOPv2 paper does not disclose its exact schedule.
6. **Optimizer** — YOLOP uses Adam with `betas=(0.937, 0.999)`. YOLOv7
   default is SGD. We expose both via config and keep Adam as default to
   match YOLOP's baseline.

All of the above are tagged with `# [INFERRED: <short why>]` comments at
the call site so any future reader can replace them with the ground truth
the moment an official YOLOPv2 training release appears.

## 5. What was actually changed

- `lib/models/common.py` — added `ELAN`, `SPPCSPC`, `ConvTranspose` helper,
  all tagged `[INFERRED]`. Kept every existing YOLOP block untouched.
- `lib/models/yolop_baseline.py` — renamed copy of the existing
  `yolop_vehicle_lane.py`, re-documented as the honest YOLOP baseline.
- `lib/models/yolopv2_baseline.py` — new; ELAN backbone + SPPCSPC neck +
  transpose-conv lane decoder + 1-channel sigmoid output. `split_for_trace`
  helper mirrored from `external_repos/YOLOPv2/utils/utils.py`.
- `lib/models/__init__.py` — now exposes `get_net_yolop` and
  `get_net_yolopv2`; `get_net` dispatches on `cfg.MODEL.NAME`.
- `lib/dataset/AutoDriveDataset.py` — Mosaic (4-image) and MixUp hooks
  added, gated by `cfg.DATASET.MOSAIC` / `cfg.DATASET.MIXUP`. Off by
  default for the YOLOP baseline; on by default for the YOLOPv2 baseline.
- `lib/config/default.py` — added `MODEL.NAME`, Mosaic/MixUp knobs,
  deconv-decoder knob, and a YOLOPv2 lambda schedule.
- `configs/yolopv2_vehicle_lane_baseline.yaml` — new; pins the
  YOLOPv2-style phase-1 run.
- `notebooks/02_train_yolopv2_vehicle_lane_baseline.ipynb` — switched
  model to the YOLOPv2 baseline, wired Mosaic/MixUp, labelled honestly.
- `notebooks/02b_train_yolop_vehicle_lane_baseline.ipynb` — new; trains
  the original YOLOP baseline for comparison.
- `notebooks/01_augmentation_lab.ipynb` — now visualizes Mosaic + MixUp
  samples straight from the real dataloader.
- `notebooks/07_h100_video_profile.ipynb` — A5000 peak TFLOPS, file
  renamed on disk to `07_a5000_video_profile.ipynb`.

## 6. What is still not reproducible

- Exact YOLOPv2 channel widths per stage.
- Exact YOLOPv2 learning-rate schedule and aug probabilities.
- Any loss gating / hard-negative mining YOLOPv2 may or may not use.

If / when the official YOLOPv2 training code is released, every
`[INFERRED]` site is a one-line edit.
