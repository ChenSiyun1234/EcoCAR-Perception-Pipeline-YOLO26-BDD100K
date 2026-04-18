# Stage1 YOLOPv2 realignment patch (2026-04-18)

This patch corrects the main paper-alignment drift that remained after the earlier repair package.

## Verified paper facts used for this patch

- YOLOPv2 keeps object detection, drivable-area segmentation, and lane segmentation in **separate heads**.
- The **lane branch is connected to the end of the FPN layer** and uses **deconvolution** in the decoder.
- For lane training, the paper preprocesses BDD100K lanes by computing a **centerline** and drawing a mask with **width 8 px for training** and **width 2 px for testing**.
- The lane loss is the paper's **hybrid Dice + Focal** formulation written for **C = 2 categories**.

## What was corrected here

1. Restored the Stage1 default notebook target from `YOLOPv2-paper-no-da` to `YOLOPv2-best-row`.
2. Restored the lane head to a **2-channel bg/fg output** instead of the earlier 1-channel foreground surrogate.
3. Kept the lane tap at the true **end-of-FPN feature** (layer 17 in this YOLOP-derived neck).
4. Reworked the lane loss path so the default YOLOPv2 route uses **2-class softmax focal + Dice**.
5. Kept backward compatibility for old 1-channel experiments so older checkpoints can still be loaded.
6. Switched the Stage1 best-row configs away from speculative SGDR restarts and back to plain **cosine annealing + 3-epoch warmup**, which is the safer reading of the paper text.

## Why this matters

The previous package was still mixing a paper-aligned decoder with a non-paper 1-channel training contract. This patch removes that mismatch.
