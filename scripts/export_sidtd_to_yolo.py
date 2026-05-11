from pathlib import Path
from PIL import Image
import json
import csv
import shutil
import hashlib


SIDTD_ROOT = Path("data/raw/templates")
DATASET_DIR = Path("dataset")

CLASS_ID = 0
CLASS_NAME = "tampered_region"

# Set to None to use everything.
# For first test, you can set MAX_FAKES = 100.
MAX_FAKES = None
MAX_REALS = None

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def stable_split(key: str) -> str:
    """
    Deterministic split based on source image key.
    This avoids putting fake variants of the same source document in different splits.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF

    if value < SPLIT_RATIOS["train"]:
        return "train"
    if value < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def reset_dataset_dirs():
    for split in ["train", "val", "test"]:
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        if image_dir.exists():
            shutil.rmtree(image_dir)
        if label_dir.exists():
            shutil.rmtree(label_dir)

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    content = f"""path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    (DATASET_DIR / "data.yaml").write_text(content)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_doc_type_from_source(src_name: str) -> str:
    """
    Example:
    alb_id_00.jpg -> alb_id
    aze_passport_23.jpg -> aze_passport
    rus_internalpassport_98.jpg -> rus_internalpassport
    """
    stem = Path(src_name).stem
    parts = stem.split("_")

    if parts[-1].isdigit():
        return "_".join(parts[:-1])

    return stem


def get_source_key(src_name: str) -> str:
    """
    Example:
    alb_id_00.jpg -> alb_id_00
    """
    return Path(src_name).stem


def find_real_annotation_entry(real_annotation_data: dict, src_name: str):
    """
    Real annotations are in VIA format.
    We find the image metadata entry whose filename matches src index.

    Fake JSON src: alb_id_00.jpg
    Real VIA filename: 00.jpg
    """
    src_stem = Path(src_name).stem
    src_index = src_stem.split("_")[-1]
    expected_filename = f"{src_index}.jpg"

    metadata = real_annotation_data.get("_via_img_metadata", {})

    for _, entry in metadata.items():
        if entry.get("filename") == expected_filename:
            return entry

    return None


def normalize_field_name(name: str) -> str:
    return str(name).strip().lower()


def find_region_for_field(real_entry: dict, field_name: str):
    """
    Finds the VIA region whose region_attributes.field_name matches the tampered field.
    """
    target = normalize_field_name(field_name)

    for region in real_entry.get("regions", []):
        attrs = region.get("region_attributes", {})
        current = normalize_field_name(attrs.get("field_name", ""))

        if current == target:
            return region

    # Fallback: some fields may be stored slightly differently.
    # Example: nationality/nationality_eng
    for region in real_entry.get("regions", []):
        attrs = region.get("region_attributes", {})
        current = normalize_field_name(attrs.get("field_name", ""))

        if target in current or current in target:
            return region

    return None


def region_to_bbox(region: dict):
    """
    Converts VIA shape attributes to x1, y1, x2, y2.

    Supports:
    - polygon with all_points_x/all_points_y
    - rect with x/y/width/height
    """
    shape = region.get("shape_attributes", {})

    if "all_points_x" in shape and "all_points_y" in shape:
        xs = shape["all_points_x"]
        ys = shape["all_points_y"]

        if not xs or not ys:
            return None

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        return int(x1), int(y1), int(x2), int(y2)

    if all(k in shape for k in ["x", "y", "width", "height"]):
        x1 = shape["x"]
        y1 = shape["y"]
        x2 = x1 + shape["width"]
        y2 = y1 + shape["height"]

        return int(x1), int(y1), int(x2), int(y2)

    return None


def clamp_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, width, height):
    box_w = x2 - x1
    box_h = y2 - y1

    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    return (
        center_x / width,
        center_y / height,
        box_w / width,
        box_h / height,
    )


def copy_image_to_dataset(src_image_path: Path, out_image_path: Path):
    """
    Re-saves as RGB JPG to keep output consistent.
    """
    img = Image.open(src_image_path).convert("RGB")
    img.save(out_image_path, quality=95)


def export_clean_reals(metadata_rows):
    real_images_dir = SIDTD_ROOT / "Images" / "reals"
    real_image_paths = sorted(real_images_dir.glob("*.jpg"))

    if MAX_REALS is not None:
        real_image_paths = real_image_paths[:MAX_REALS]

    exported = 0

    for src_image_path in real_image_paths:
        source_key = src_image_path.stem
        split = stable_split(source_key)

        image_id = f"clean_{src_image_path.stem}"
        out_image_path = DATASET_DIR / "images" / split / f"{image_id}.jpg"
        out_label_path = DATASET_DIR / "labels" / split / f"{image_id}.txt"

        copy_image_to_dataset(src_image_path, out_image_path)
        out_label_path.write_text("")

        metadata_rows.append({
            "image_id": image_id,
            "split": split,
            "image_path": str(out_image_path),
            "label_path": str(out_label_path),
            "is_tampered": 0,
            "tamper_type": "clean",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "source_image": str(src_image_path),
        })

        exported += 1

    return exported


def export_fake_samples(metadata_rows):
    fake_ann_dir = SIDTD_ROOT / "Annotations" / "fakes"
    fake_img_dir = SIDTD_ROOT / "Images" / "fakes"

    real_ann_dir = SIDTD_ROOT / "Annotations" / "reals"

    fake_json_paths = sorted(fake_ann_dir.glob("*.json"))

    if MAX_FAKES is not None:
        fake_json_paths = fake_json_paths[:MAX_FAKES]

    real_annotation_cache = {}

    exported = 0
    skipped = 0
    skip_reasons = {}

    for fake_json_path in fake_json_paths:
        try:
            fake_meta = load_json(fake_json_path)
        except Exception as exc:
            skipped += 1
            skip_reasons["fake_json_unreadable"] = skip_reasons.get("fake_json_unreadable", 0) + 1
            print(f"Skipping unreadable fake JSON: {fake_json_path} | {exc}")
            continue

        fake_stem = fake_json_path.stem
        fake_image_path = fake_img_dir / f"{fake_stem}.jpg"

        if not fake_image_path.exists():
            skipped += 1
            skip_reasons["fake_image_missing"] = skip_reasons.get("fake_image_missing", 0) + 1
            continue

        field = fake_meta.get("field")
        src = fake_meta.get("src")
        ctype = fake_meta.get("ctype", "")

        if not field or not src:
            skipped += 1
            skip_reasons["missing_field_or_src"] = skip_reasons.get("missing_field_or_src", 0) + 1
            continue

        source_key = get_source_key(src)
        doc_type = get_doc_type_from_source(src)

        real_annotation_path = real_ann_dir / f"{doc_type}.json"

        if not real_annotation_path.exists():
            skipped += 1
            skip_reasons["real_annotation_missing"] = skip_reasons.get("real_annotation_missing", 0) + 1
            continue

        if real_annotation_path not in real_annotation_cache:
            real_annotation_cache[real_annotation_path] = load_json(real_annotation_path)

        real_annotation_data = real_annotation_cache[real_annotation_path]
        real_entry = find_real_annotation_entry(real_annotation_data, src)

        if real_entry is None:
            skipped += 1
            skip_reasons["real_entry_missing"] = skip_reasons.get("real_entry_missing", 0) + 1
            continue

        region = find_region_for_field(real_entry, field)

        if region is None:
            skipped += 1
            skip_reasons["field_region_missing"] = skip_reasons.get("field_region_missing", 0) + 1
            continue

        bbox = region_to_bbox(region)

        if bbox is None:
            skipped += 1
            skip_reasons["bbox_missing"] = skip_reasons.get("bbox_missing", 0) + 1
            continue

        img = Image.open(fake_image_path).convert("RGB")
        width, height = img.size

        padding = max(8, int(min(width, height) * 0.005))

        x1, y1, x2, y2 = bbox

        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
        cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, width, height)

        split = stable_split(source_key)

        image_id = fake_stem
        out_image_path = DATASET_DIR / "images" / split / f"{image_id}.jpg"
        out_label_path = DATASET_DIR / "labels" / split / f"{image_id}.txt"

        img.save(out_image_path, quality=95)

        out_label_path.write_text(
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )

        metadata_rows.append({
            "image_id": image_id,
            "split": split,
            "image_path": str(out_image_path),
            "label_path": str(out_label_path),
            "is_tampered": 1,
            "tamper_type": ctype,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "source_image": str(SIDTD_ROOT / "Images" / "reals" / src),
        })

        exported += 1

    return exported, skipped, skip_reasons


def write_metadata(metadata_rows):
    fieldnames = [
        "image_id",
        "split",
        "image_path",
        "label_path",
        "is_tampered",
        "tamper_type",
        "x1",
        "y1",
        "x2",
        "y2",
        "source_image",
    ]

    metadata_path = DATASET_DIR / "metadata.csv"

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    return metadata_path


def print_dataset_summary(metadata_rows):
    print("\nDataset summary")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        image_count = len(list((DATASET_DIR / "images" / split).glob("*.jpg")))
        label_count = len(list((DATASET_DIR / "labels" / split).glob("*.txt")))

        split_rows = [r for r in metadata_rows if r["split"] == split]
        clean_count = sum(1 for r in split_rows if r["is_tampered"] == 0)
        fake_count = sum(1 for r in split_rows if r["is_tampered"] == 1)

        print(
            f"{split}: "
            f"{image_count} images, "
            f"{label_count} labels, "
            f"{clean_count} clean, "
            f"{fake_count} tampered"
        )


def main():
    if not SIDTD_ROOT.exists():
        raise SystemExit(f"SIDTD root not found: {SIDTD_ROOT}")

    reset_dataset_dirs()
    write_data_yaml()

    metadata_rows = []

    clean_count = export_clean_reals(metadata_rows)
    fake_count, skipped, skip_reasons = export_fake_samples(metadata_rows)

    metadata_path = write_metadata(metadata_rows)

    print("\nExport complete")
    print("=" * 80)
    print(f"Clean real images exported: {clean_count}")
    print(f"Tampered fake images exported: {fake_count}")
    print(f"Skipped fake samples: {skipped}")
    print(f"Metadata written to: {metadata_path}")

    if skip_reasons:
        print("\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"- {reason}: {count}")

    print_dataset_summary(metadata_rows)


if __name__ == "__main__":
    main()
