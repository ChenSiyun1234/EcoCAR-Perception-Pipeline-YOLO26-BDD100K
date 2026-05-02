YOLOPX stable training profile for stage1:
1. Reverted training batch to 32.
2. Reverted lr0 to 0.001.
3. Increased warmup to 5 epochs for smoother early training.
4. Kept validation frequency at every completed epoch.
5. Kept checkpoint fix if present in the source zip: latest.pth saves after every completed epoch.
6. Reduced MixUp probability to 0.05 and Mosaic probability to 0.5 for early stability.

Reason:
Batch 64 with lr0 0.002 can keep GPU memory low but may make early optimization less stable.
The goal of this version is not maximum GPU memory usage; it is a reliable baseline curve.
