# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document tampering and forgery detection using YOLO-based object detection. The model localizes tampered regions in identity document images (passports, IDs). Input: document image. Output: bounding boxes with confidence scores over suspicious regions. A document is classified as forged if at least one `tampered_region` is detected above the confidence threshold.

The dataset is derived from **SIDTD** (Secure Identity Document Tampering Detection). Raw SIDTD templates live in `data/raw/templates/`. The processed YOLO-format dataset lives in `dataset/`.

## Environment Setup

Python 3.9.6 virtual environment at `.venv/`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The SIDTD library is installed in editable mode from `external/SIDTD_Dataset/`. To verify:

```bash
python -c "import SIDTD; print('SIDTD import OK')"
```

## Common Commands

All scripts must be run from the repo root (`document-tampering-forgery-detection/`).

**Regenerate the YOLO dataset from raw SIDTD templates:**
```bash
python scripts/export_sidtd_to_yolo.py
```

**Visualize YOLO labels (draws bounding boxes, saves to `outputs/visual_checks/yolo_export/`):**
```bash
python scripts/visualize_yolo_labels.py
```

**Inspect raw SIDTD annotation JSONs:**
```bash
python scripts/inspect_sidtd_annotations.py --root data/raw --max-files 10
```

**Train with YOLOv8 (requires `ultralytics` installed):**
```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=30 imgsz=640 batch=8
```

## Data Architecture

### Raw SIDTD layout (input to export script)
```
data/raw/templates/
  Images/
    reals/       # authentic document images (*.jpg)
    fakes/       # tampered document images (*.jpg)
  Annotations/
    reals/       # VIA-format annotation JSONs per document type (e.g. alb_id.json)
    fakes/       # one JSON per fake image with: src, field, ctype, shift, type_transformation
```

### Exported YOLO dataset (output, ready for training)
```
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/    # empty .txt = clean, one YOLO line per tampered region
  data.yaml
  metadata.csv
```

Only one class: `0: tampered_region`.

### How the export pipeline works (`scripts/export_sidtd_to_yolo.py`)

1. For each fake image: read `fake_meta["field"]` (the modified field name) and `fake_meta["src"]` (the source real image filename).
2. Load the real document-type annotation JSON from `Annotations/reals/{doc_type}.json` (VIA format).
3. Look up the field's polygon/rect region by matching `field_name` in `region_attributes`.
4. Convert the VIA shape to `x1,y1,x2,y2`, add a small padding (`max(8px, 0.5% of min dimension)`), clamp to image bounds.
5. Write YOLO-normalized box to the label file.
6. Clean images get empty `.txt` label files.

**Split assignment is deterministic** via MD5 hash of the source image stem (e.g. `alb_id_00`), ensuring all fake variants of the same source document land in the same split (no leakage). Ratios: 70% train / 15% val / 15% test.

### Dataset stats (current export)
- 2222 total images: 1000 clean, 1222 tampered
- train: 1557 | val: 339 | test: 326

## Key Implementation Notes

- `stable_split()` in the export script hashes the **source** document key, not the fake image name. This prevents train/test contamination from multiple fakes derived from the same real document.
- VIA annotation format stores field regions as either `polygon` (all_points_x/all_points_y) or `rect` (x/y/width/height); `region_to_bbox()` handles both.
- Field name matching uses a substring fallback (e.g. `nationality` matches `nationality_eng`) when exact match fails.
- `outputs/` contains visual QA artifacts; `outputs/visual_checks/manual_step8/` holds the dataset handoff QA sheets.
