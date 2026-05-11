# Dataset Specification

## Structure

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

## Naming convention

Images:
- doc_000001.jpg
- doc_000002.jpg
- ...

Labels:
- doc_000001.txt
- doc_000002.txt
- ...

Each image must have a matching label file with the same base name.

## YOLO label format

Each non-empty label file contains one or more lines:

class_id center_x center_y width height

- normalized between 0 and 1
- only one class:
  - 0 = tampered_region

## Clean images

Clean/authentic images must have:
- an empty `.txt` label file
- `is_tampered = 0`
- `tamper_type = clean`
- empty x1, y1, x2, y2 columns in metadata.csv

## Tampered images

Tampered images must have:
- one or more YOLO boxes in the `.txt` file
- `is_tampered = 1`
- tamper_type filled
- x1, y1, x2, y2 filled in metadata.csv

## Multiple tampered regions

If an image has multiple tampered regions:
- the label file contains multiple YOLO lines
- metadata.csv contains multiple rows for the same image_id

## Metadata columns

image_id,split,image_path,label_path,is_tampered,tamper_type,x1,y1,x2,y2,source_image
