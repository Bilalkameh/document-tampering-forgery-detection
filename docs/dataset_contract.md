# Dataset Contract

This document defines the required dataset format between the dataset generation side and the YOLO training side.

## Folder Structure

The dataset must be exported as:

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

## YOLO Labels

Every image must have a matching `.txt` file with the same base name.

Example:

```text
images/train/doc_001.jpg
labels/train/doc_001.txt
```

For tampered images, each label file contains one or more YOLO-format bounding boxes:

```text
class_id center_x center_y width height
```

All box values must be normalized between `0` and `1`.

The only class is:

```text
0: tampered_region
```

For clean/authentic images, the matching `.txt` label file must be empty.

## data.yaml

The `data.yaml` file must contain:

```yaml
path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: tampered_region
```

## metadata.csv

The metadata file should contain at least:

```text
image_id
split
image_path
label_path
is_tampered
tamper_type
x1
y1
x2
y2
source_image
```

For clean images:

```text
is_tampered = 0
tamper_type = clean
x1, y1, x2, y2 can be empty
```

For tampered images:

```text
is_tampered = 1
tamper_type = text_replacement / field_overwrite / copy_paste / blur_edit / noise_mismatch / etc.
x1, y1, x2, y2 should contain the bounding box in pixel coordinates
```

If an image has multiple tampered regions, the metadata can contain multiple rows for the same image.

## Mini Dataset First

Before the full dataset, a small test sample should be exported:

```text
20 train images
5 val images
5 test images
matching labels
data.yaml
metadata.csv
```

This mini dataset will be used to test the YOLO training pipeline before final training.

## Acceptance Checklist

Before training, the YOLO side will check:

- Every image has a matching `.txt` label file.
- Clean images have empty label files.
- Tampered images have at least one bounding box.
- All YOLO box values are normalized between `0` and `1`.
- The only class id used is `0`.
- `data.yaml` paths are correct.
- `metadata.csv` matches the dataset files.
- The dataset contains both clean and tampered examples.