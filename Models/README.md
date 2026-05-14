# Trained Model Weights

This folder contains the trained model weights used for the document tampering detection project.

## Included Models

### `sidtd_yolov8n_baseline_best.pt`

Baseline YOLOv8n model trained on the SIDTD YOLO dataset.

Test performance:

- Precision: 0.826
- Recall: 0.865
- mAP@0.5: 0.888
- mAP@0.5:0.95: 0.708

### `sidtd_yolov8s_opt768_best.pt`

Final optimized YOLOv8s model selected for the project.

Test performance:

- Precision: 0.936
- Recall: 0.852
- mAP@0.5: 0.930
- mAP@0.5:0.95: 0.727

## Final Selected Model

The final selected model is:

```text
Models/sidtd_yolov8s_opt768_best.pt