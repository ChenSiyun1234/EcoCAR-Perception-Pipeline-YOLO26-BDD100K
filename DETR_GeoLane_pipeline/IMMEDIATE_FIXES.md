This patch focuses on the immediate blocker:
1. Search /content/drive/MyDrive/EcoCAR/downloads/bdd100k_labels.zip
2. Auto-extract it to /content/drive/MyDrive/EcoCAR/datasets/bdd100k_raw
3. Search official lane json paths like bdd100k/labels/lane/polygons/lane_train.json
4. Add notebook debug cells to print actual JSON structure
5. Fix dataset horizontal flip bug
