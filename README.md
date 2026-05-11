## YOLO Pipeline

This branch contains the YOLO training, validation, inference, and dataset checking scripts.

### 1. Check the dataset

Before training, validate the dataset structure and YOLO labels:

```bash
python src/evaluation/check_dataset.py --dataset dataset
```

The expected dataset structure is:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
  metadata.csv
```

### 2. Train YOLO

For a quick mini-dataset test:

```bash
python src/training/train_yolo.py --data dataset/data.yaml --epochs 5 --imgsz 416 --batch 4 --device cpu --name mini_test
```

For the main baseline training:

```bash
python src/training/train_yolo.py --data dataset/data.yaml --epochs 50 --imgsz 640 --batch 8 --device cpu --name tamper_yolov8n_baseline
```

### 3. Validate YOLO

Validate on the validation split:

```bash
python src/evaluation/validate_yolo.py --weights runs/detect/tamper_yolov8n_baseline/weights/best.pt --data dataset/data.yaml --split val --device cpu
```

Validate on the test split:

```bash
python src/evaluation/validate_yolo.py --weights runs/detect/tamper_yolov8n_baseline/weights/best.pt --data dataset/data.yaml --split test --device cpu
```

### 4. Run inference

Run prediction on test images:

```bash
python src/inference/predict_yolo.py --weights runs/detect/tamper_yolov8n_baseline/weights/best.pt --source dataset/images/test --conf 0.25 --device cpu
```

Predicted images and YOLO prediction `.txt` files will be saved under:

```text
outputs/predictions/
```

## Current Model Scope

The model detects one class:

```text
0: tampered_region
```

A document is considered potentially forged if at least one tampered region is detected.

## Limitations

This system does not prove legal document forgery. It detects visual regions that resemble the synthetic tampering patterns used during training. Performance depends strongly on the diversity and realism of the generated tampering.