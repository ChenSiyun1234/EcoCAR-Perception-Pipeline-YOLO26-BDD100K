# Repair pass v4 — YOLOPv2 "best row" alignment

**Targeted row.** YOLOPv2 paper (arXiv 2208.11434 §3) presents a
cumulative ablation:

| row # | added component (cumulative) |
|---|---|
| 1 | YOLOP baseline |
| 2 | + ELAN / group-conv backbone, SPP kept in neck |
| 3 | + Mosaic + MixUp (Bag of Freebies) |
| 4 | + focal loss on det cls + obj |
| 5 | + focal loss on lane seg |
| 6 | + **hybrid focal + dice on lane seg** ← this pass targets this row |

Row 6 is the final row where *every* improvement is active and is the
one that delivers the paper's best reported lane-seg numbers. That is
the primary target of this repo now. Drivable-area removal remains the
only intentional task deviation.

## Changes in this pass (v4)

### 1. Unambiguous primary / ablation split (user §1)

- `configs/yolopv2_vehicle_lane_baseline.yaml` **deleted** — it was
  ambiguous about focal-only vs focal+dice.
- **New**: `configs/yolopv2_best_row.yaml` — the primary target.
  `LL_DICE_GAIN = 0.5`, `SGDR = true`, every earlier improvement
  turned on.
- **New**: `configs/yolopv2_focal_only_ablation.yaml` — identical
  except `LL_DICE_GAIN = 0.0`. Used as the A/B reference.
- `configs/yolop_vehicle_lane_baseline.yaml` unchanged — honest YOLOP
  minus DA, kept as a baseline reference.

### 2. Lane output contract — no more ambiguity (user §2)

`lib/models/yolopv2_baseline.py` defines exactly two contracts:

| method | returns | who uses it |
|---|---|---|
| `forward(x)` | `(det_out, lane_logits_2ch_fullres)` | training, validate(), loss |
| `predict(x)` | `(det_out, lane_prob_1ch_halfres)`   | notebooks 06 (export) and 07 (profile) |

The paper's public demo (`external_repos/YOLOPv2/demo.py`) expects the
second shape; `predict()` returns exactly that. The double-sigmoid bug
is gone — forward() emits raw logits and `BCEWithLogitsLoss` consumes
them correctly, so the math is clean end-to-end.

Docstrings are verbose about which is which to prevent regressions.

### 3. Scheduler — SGDR is the default now (user §3)

- `TRAIN.SGDR = True` is the new default in both YAMLs and in
  `lib/config/default.py`.
- `TRAIN.SGDR_T0 = 100`, `TRAIN.SGDR_TMULT = 1` ⇒ restarts at epochs
  100 and 200 of the 300-epoch budget.
- The paper's phrase "cosine annealing with warm restart" is taken
  literally as SGDR. T_0 / T_mult are `[INFERRED]` because the paper
  does not publish them.
- `TRAIN.SGDR = False` still runs YOLOP-style cosine + linear-warmup
  (the previous default) for anyone who wants to ablate the scheduler.

### 4. Preserved, not regressed (user §4)

- Centerline-oriented lane GT (greedy boundary pairing + midline
  rasterization) — `lib/utils/lane_render.py`, untouched this pass.
- Train / eval width split 8 / 2 — untouched.
- Logits-consistent lane loss path — the fix from v2 is still in
  place; this pass only added the dice term as a pure-probability
  hybrid on top.
- True 640×384 eval — untouched; both YOLOPv2 YAMLs still set
  `TEST.IMAGE_SIZE = [640, 384]` and notebook 02 / 03 pass an
  explicit `(H, W)` tuple so `letterbox(auto=False)` is used.
- Drive-persistent `.tar.gz` workflow — untouched (notebook 00 cell 8).
- A5000 profiling defaults — untouched.

### 5. What is still `[INFERRED]`

Every site tagged with a code comment. Summary:

| item | where | why it cannot be made exact |
|---|---|---|
| E-ELAN `groups` | `yolopv2_baseline.py::YOLOPv2Cfg` | paper says "group conv" without publishing the number |
| Focal γ (det + lane) | both YAMLs, `LOSS.FL_GAMMA`, `LOSS.LL_FL_GAMMA` | paper says focal, γ not published (we use 1.5) |
| Dice gain for hybrid | best-row YAML, `LOSS.LL_DICE_GAIN` | paper mentions hybrid as ablation, weight not published (we use 0.5) |
| SGDR `T_0`, `T_mult` | both YAMLs, `TRAIN.SGDR_*` | paper says "warm restart", periods not published (we use 100, 1) |
| BDD centerline pairing | `lane_render.py::_pair_lane_boundaries` | paper's pairing code not released; we use greedy proximity + y-overlap |

## Acceptance criteria (user §Acceptance)

- [x] Exactly one unambiguous primary YAML — `yolopv2_best_row.yaml`.
- [x] Focal+dice vs focal-only: no longer a switch buried in comments;
      it's the only real knob difference between the two YOLOPv2
      configs, and the best row is the default everywhere.
- [x] Lane output contract locked: `forward` = logits full-res,
      `predict` = sigmoid H/2, both with matching docstrings and
      matching usage in notebooks 06 / 07.
- [x] Scheduler default: SGDR (explicit, documented, toggleable).
- [x] Docs (this file + `FINAL_ALIGNMENT_STATUS_v4.md`) call out
      MATCH / PARTIAL / INTENTIONAL DEVIATION / INFERRED per dimension.
