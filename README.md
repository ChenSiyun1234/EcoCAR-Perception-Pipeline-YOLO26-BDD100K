# EcoCAR Perception Pipeline — YOLO26 + BDD100K

A Colab-first, notebook-based autonomous driving perception pipeline using **YOLO26** (Ultralytics, Jan 2026) and **BDD100K** (Berkeley DeepDrive).

## 🚗 Project Overview

This project builds a perception pipeline for the EcoCAR autonomous driving project using a **shared YOLO26 backbone/neck** jointly trained for both **object detection** (10 BDD100K classes) and **lane segmentation** in a single training stage.

---

## 📋 Quick Start

### Phase 1 — Detection Baseline (Notebooks 00–06)

| # | Notebook | Purpose |
|---|---|---|
| 00 | Pretrained Baseline | Smoke test YOLO26 on COCO weights |
| 01 | Dataset Setup | Download BDD100K (images + detection labels + lane labels) |
| 02 | BDD Preparation | Convert JSON → YOLO format, build train/val folders |
| 03 | Train Detection | Fine-tune `yolo26s.pt` (detection-only, 10 epochs) |
| 04 | Inference | Run & visualise detection predictions |
| 05 | Feature Extraction | Hook backbone/neck layers, capture feature maps |
| 06 | Model Inspection | Print architecture, identify components |

### Phase 2 — Joint Training (Notebooks 07–09)

| # | Notebook | Purpose |
|---|---|---|
| **07** | **Prepare Lane Masks** | Rasterise BDD100K `poly2d` polylines → binary masks (320×180) |
| **08** | **Joint Training** | Train shared backbone + detection head + lane seg head |
| **09** | **Joint Inference** | Visualise detection boxes + lane mask overlay |

**Recommended order:**
```
00 → 01 → 02 → 07 → 08 → 09
```
(Skip 03–06 if going straight to joint training)

---

## 📁 Project Structure

```
EEC174/
├── notebooks/
│   ├── 00_quick_pretrained_yolo_baseline.ipynb
│   ├── 01_dataset_setup_and_download.ipynb
│   ├── 02_bdd100k_preparation.ipynb
│   ├── 03_train_yolo26_on_bdd.ipynb            # Detection-only training
│   ├── 04_inference_yolo26.ipynb
│   ├── 05_extract_backbone_features.ipynb
│   ├── 06_model_inspection.ipynb
│   ├── 07_prepare_lane_masks.ipynb              # ★ Lane mask rasterisation
│   ├── 08_joint_training.ipynb                  # ★ Joint det + lane training
│   └── 09_joint_inference.ipynb                 # ★ Joint inference
├── configs/
│   ├── bdd100k_yolo.yaml                        # Detection-only config
│   └── bdd100k_joint.yaml                       # Joint training config
├── src/
│   ├── dataset_utils.py                          # BDD→YOLO conversion
│   ├── feature_hooks.py                          # Hook-based feature extraction
│   ├── model_utils.py                            # Model loading & info
│   ├── visualization_utils.py                    # Drawing & plotting
│   ├── lane_utils.py                             # ★ Polyline→mask rasterisation
│   ├── multitask_model.py                        # ★ MultiTaskYOLO (det + lane)
│   ├── joint_dataset.py                          # ★ Joint dataset loader
│   └── joint_trainer.py                          # ★ Custom training loop
├── requirements.txt
└── README.md
```

---

## 🏗️ Joint Training Architecture

```
Input → YOLO26 Backbone → Neck (FPN/PAN) → ┬→ Detect Head (10-class boxes)
                                            └→ Lane Seg Head (binary mask)

Loss = α·det_loss + β·lane_loss    (default: α=1.0, β=0.5)
```

| Component | Detail |
|---|---|
| Backbone/Neck | Shared, pretrained `yolo26s.pt`, lower LR |
| Detection head | Native YOLO Detect module |
| Lane head | 3×(Conv+BN+ReLU) + upsample (~0.5M params) |
| Lane loss | BCE + Dice (handles thin lane class imbalance) |
| Lane mask size | 320×180 (¼ resolution) |

---

## 🗂️ BDD100K Dataset

| Item | Detail |
|---|---|
| **Source** | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) |
| **Images** | 100K driving images (1280×720) |
| **Detection classes (10)** | person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign |
| **Lane categories (8)** | single/double × white/yellow/other + crosswalk + road curb |

### What to download
- `bdd100k_images_100k.zip` — all images (~6.4 GB)
- `bdd100k_labels_release.zip` — detection + lane labels (~100 MB)

---

## 📂 Where Outputs Are Saved

| Output | Location |
|---|---|
| Baseline predictions | `EcoCAR/outputs/00_baseline/` |
| Lane masks | `EcoCAR/datasets/bdd100k_yolo/masks/{train,val}/` |
| Joint training runs | `EcoCAR/training_runs/joint_det_lane/` |
| Joint weights | `EcoCAR/weights/joint_det_lane_best.pt` |
| Joint inference | `EcoCAR/outputs/09_joint_inference/` |

---

## ⚡ Tips

- **Debug mode:** Set `DEBUG_LIMIT = 100` in notebooks 02/07/08 to test with a tiny subset
- **GPU memory issues:** Use `yolo26n.pt` instead of `yolo26s.pt`, or reduce batch size to 4
- **Colab timeouts:** Save weights to Drive frequently; use Colab Pro for longer sessions
- **Better results:** Train for 30–50 epochs with `yolo26m.pt`
