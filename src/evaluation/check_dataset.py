from pathlib import Path
import argparse
import yaml
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_label_file(label_path: Path):
    """
    Returns:
        boxes_count, errors
    """
    errors = []

    if not label_path.exists():
        return 0, [f"Missing label file: {label_path}"]

    text = label_path.read_text(encoding="utf-8").strip()

    # Clean image: empty label file is valid
    if text == "":
        return 0, []

    boxes_count = 0

    for line_idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(f"{label_path}, line {line_idx}: expected 5 values, got {len(parts)}")
            continue

        class_id, cx, cy, w, h = parts

        try:
            class_id = int(class_id)
            cx, cy, w, h = map(float, [cx, cy, w, h])
        except ValueError:
            errors.append(f"{label_path}, line {line_idx}: non-numeric YOLO values")
            continue

        if class_id != 0:
            errors.append(f"{label_path}, line {line_idx}: invalid class_id {class_id}, expected 0")

        for name, value in [("center_x", cx), ("center_y", cy), ("width", w), ("height", h)]:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{label_path}, line {line_idx}: {name}={value} is outside [0, 1]")

        if w <= 0 or h <= 0:
            errors.append(f"{label_path}, line {line_idx}: width and height must be > 0")

        boxes_count += 1

    return boxes_count, errors


def check_split(dataset_dir: Path, split: str):
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists():
        return {
            "split": split,
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "errors": [f"Missing images directory: {images_dir}"],
        }

    if not labels_dir.exists():
        return {
            "split": split,
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "errors": [f"Missing labels directory: {labels_dir}"],
        }

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    label_paths = sorted(labels_dir.glob("*.txt"))

    errors = []
    total_boxes = 0
    clean_images = 0
    tampered_images = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes_count, label_errors = check_label_file(label_path)

        errors.extend(label_errors)
        total_boxes += boxes_count

        if boxes_count == 0:
            clean_images += 1
        else:
            tampered_images += 1

    return {
        "split": split,
        "images": len(image_paths),
        "labels": len(label_paths),
        "boxes": total_boxes,
        "clean_images": clean_images,
        "tampered_images": tampered_images,
        "errors": errors,
    }


def check_metadata(dataset_dir: Path):
    metadata_path = dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        return ["Missing metadata.csv"]

    required_cols = {
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
    }

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        return [f"Could not read metadata.csv: {e}"]

    missing = required_cols - set(df.columns)
    if missing:
        return [f"metadata.csv missing columns: {sorted(missing)}"]

    return []


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO document tampering dataset.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset folder")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    print(f"\nChecking dataset: {dataset_dir.resolve()}\n")

    errors = []

    yaml_path = dataset_dir / "data.yaml"
    try:
        data_yaml = load_yaml(yaml_path)
        print("data.yaml found.")

        expected_names = {0: "tampered_region"}
        if data_yaml.get("names") != expected_names:
            errors.append(f"data.yaml names should be {expected_names}, got {data_yaml.get('names')}")

    except Exception as e:
        errors.append(str(e))

    metadata_errors = check_metadata(dataset_dir)
    errors.extend(metadata_errors)

    split_results = []
    for split in ["train", "val", "test"]:
        result = check_split(dataset_dir, split)
        split_results.append(result)
        errors.extend(result["errors"])

    print("\nDataset Summary")
    print("-" * 60)

    total_images = 0
    total_boxes = 0
    total_clean = 0
    total_tampered = 0

    for result in split_results:
        total_images += result["images"]
        total_boxes += result["boxes"]
        total_clean += result.get("clean_images", 0)
        total_tampered += result.get("tampered_images", 0)

        print(
            f"{result['split']:>5}: "
            f"images={result['images']}, "
            f"labels={result['labels']}, "
            f"clean={result.get('clean_images', 0)}, "
            f"tampered={result.get('tampered_images', 0)}, "
            f"boxes={result['boxes']}"
        )

    print("-" * 60)
    print(f"Total images:   {total_images}")
    print(f"Clean images:   {total_clean}")
    print(f"Tampered images:{total_tampered}")
    print(f"Total boxes:    {total_boxes}")

    if errors:
        print("\nERRORS FOUND")
        print("-" * 60)
        for err in errors[:50]:
            print(f"- {err}")

        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")

        raise SystemExit(1)

    print("\nDataset check passed. No errors found.")


if __name__ == "__main__":
    main()