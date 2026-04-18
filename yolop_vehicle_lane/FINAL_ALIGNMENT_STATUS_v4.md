# Final alignment status vs YOLOPv2 best row

**Target row.** Paper arXiv 2208.11434 §3, final cumulative ablation row
(ELAN backbone + SPP + FPN + PAN + Mosaic + MixUp + focal cls/obj +
focal + dice on lane seg). **Primary YAML:**
`configs/yolopv2_best_row.yaml`.

**Legend.**
- `MATCH` — byte- or semantically equivalent to upstream / paper.
- `PARTIAL` — shape right, one or more sub-detail drifts; usually
  because the paper does not publish the exact value.
- `INTENTIONAL DEVIATION` — delta by design (only drivable area).
- `INFERRED` — value or algorithm cannot be recovered from the public
  YOLOPv2 release; we picked the closest defensible option and
  labelled it.

| # | dimension | status | notes |
|---|---|---|---|
|  1 | repo layout: phase-1 isolated from stage-2 | MATCH | stage-2 quarantined under `stage2/`; `lib/models/__init__.py` imports zero stage-2 symbols |
|  2 | drivable-area head | INTENTIONAL DEVIATION | removed; only sanctioned task delta |
|  3 | backbone family | PARTIAL + INFERRED | YOLOP `MCnet_0` with `BottleneckCSP` → `ELAN` (`groups=2` at stride ≥ 16). Paper says "ELAN with group conv"; exact `groups` and channel schedule not published |
|  4 | neck SPP (not SPPCSPC) | MATCH | reverted to YOLOP's `SPP(k=5,9,13)` per paper text |
|  5 | neck FPN + PAN | MATCH | layer-by-layer identical to `YOLOP/lib/models/YOLOP.py::MCnet_0` |
|  6 | detection head | MATCH | `Detect(nc=5, …)` with YOLOP BDD-kmeans anchors; `nc` bump is part of the DA deviation, not an extra change |
|  7 | lane head taps | MATCH | from layer 16 (post-FPN encoder out), same as YOLOP lane branch |
|  8 | lane head decoder | MATCH | 3 × `ConvTranspose2d` (stride 2) stages — paper §3 "deconvolution" |
|  9 | lane GT — per-split width | MATCH | train 8 / test 2, `notebook 00` passes these per split |
| 10 | lane GT — centerline vs raw poly2d | PARTIAL + INFERRED | greedy pairing (y-overlap + lateral distance) then midline; unpaired polylines drawn as-is. Paper's pairing code is not public |
| 11 | detection cls + obj loss | MATCH | `FocalLoss(BCEWithLogitsLoss, γ=1.5)`. γ [INFERRED] |
| 12 | lane seg loss (best row) | MATCH + INFERRED weights | focal + dice. `LL_FL_GAMMA=1.5`, `LL_DICE_GAIN=0.5`. Both values [INFERRED] |
| 13 | lane seg loss (focal-only row) | MATCH | secondary YAML sets `LL_DICE_GAIN=0`; every other knob identical to best row |
| 14 | Mosaic / MixUp | MATCH | wired in `BddDataset.__getitem__` behind `DATASET.MOSAIC / MIXUP`; best-row YAML turns both on |
| 15 | optimizer | MATCH | SGD + Nesterov + momentum 0.937 + wd 0.005 |
| 16 | initial LR | MATCH | LR0 = 0.01 |
| 17 | warmup | MATCH | iter-based linear warmup, `WARMUP_EPOCHS=3.0`, unchanged from YOLOP upstream — YOLOPv2 paper text uses the same budget |
| 18 | cosine / warm-restart schedule | PARTIAL + INFERRED | SGDR (`CosineAnnealingWarmRestarts`) default; `T_0=100`, `T_mult=1`. Paper says "warm restart" but does not publish periods |
| 19 | total epochs | MATCH | 300 |
| 20 | train image size | MATCH | 640 × 640 |
| 21 | test / eval image size | MATCH | 640 × 384 — `BddDataset.inputsize` accepts `(H, W)` tuple, `letterbox(auto=False)` ensures exact shape |
| 22 | anchors / autoanchor | MATCH | YOLOP BDD-kmeans; `NEED_AUTOANCHOR=False` |
| 23 | lane forward / train contract | MATCH (internal) | `forward` returns `(det_out, lane_logits_2ch_fullres)`, `MultiHeadLoss` expects logits |
| 24 | lane inference / demo contract | MATCH | `predict(x) → (det_out, lane_prob_1ch_halfres)`; exactly what `external_repos/YOLOPv2/demo.py::lane_line_mask` consumes |
| 25 | logits-vs-sigmoid double-bug | MATCH (fixed) | YOLOP-upstream double sigmoid removed; `BCEWithLogitsLoss` + optional `FocalLoss` + optional dice-on-probabilities now compose cleanly |
| 26 | notebook layout | MATCH | phase-1 notebooks 00 / 01 / 02 / 03 / 07, export in 06; stage-2 under `stage2/notebooks/` |
| 27 | Drive-persistent workflow | MATCH | single `.tar.gz` under `EcoCAR/datasets/`; all notebooks use `ensure_local_dataset_from_drive` |
| 28 | profiling GPU | MATCH (user spec) | A5000 defaults in notebook 07 |

## Per-config ablation matrix

| knob | best-row YAML | focal-only YAML | YOLOP YAML |
|---|---|---|---|
| `MODEL.NAME` | YOLOPv2 | YOLOPv2 | YOLOP |
| Backbone | ELAN/groups | ELAN/groups | CSP/Focus |
| Mosaic | ✅ | ✅ | ❌ |
| MixUp | ✅ | ✅ | ❌ |
| `FL_GAMMA` (det) | 1.5 | 1.5 | 0 |
| `LL_FL_GAMMA` (lane) | 1.5 | 1.5 | 0 |
| `LL_DICE_GAIN` | **0.5** | **0.0** | 0 |
| Optimizer | SGD | SGD | Adam |
| `LR0` | 0.01 | 0.01 | 0.001 |
| `WD` | 0.005 | 0.005 | 0.0005 |
| Epochs | 300 | 300 | 100 |
| Scheduler | SGDR | SGDR | cosine+warmup |
| Train size | 640×640 | 640×640 | 640×640 |
| Test size | 640×384 | 640×384 | 640×640 |

## Deliverable checklist

- [x] Single unambiguous primary target (`yolopv2_best_row.yaml`)
- [x] Secondary ablation YAML (`yolopv2_focal_only_ablation.yaml`)
- [x] Lane output contract locked (forward = logits full-res; predict = sigmoid H/2)
- [x] Scheduler default = SGDR (documented, toggleable)
- [x] Notebooks 02 / 03 / 06 / 07 point at the best-row checkpoint dir by default
- [x] `REPAIR_SUMMARY_v4.md` + this file
- [x] No regression on centerline GT, 8/2 width split, logits loss, rectangular eval, Drive tar, A5000 profiling
