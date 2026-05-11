from pathlib import Path
import json
from pprint import pprint


ROOT = Path("data/raw/templates")


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read JSON {path}: {exc}")
        return None


def show_tree_examples():
    print_section("Top-level folders")
    for path in sorted(ROOT.iterdir()):
        print(path)

    print_section("First 50 files under templates")
    for path in list(ROOT.rglob("*"))[:50]:
        print(path)


def show_annotation_examples():
    real_ann_dir = ROOT / "Annotations" / "reals"
    fake_ann_dir = ROOT / "Annotations" / "fakes"

    print_section("Annotation folders")
    print(f"Real annotation dir exists: {real_ann_dir.exists()} -> {real_ann_dir}")
    print(f"Fake annotation dir exists: {fake_ann_dir.exists()} -> {fake_ann_dir}")

    real_jsons = sorted(real_ann_dir.glob("*.json")) if real_ann_dir.exists() else []
    fake_jsons = sorted(fake_ann_dir.glob("*.json")) if fake_ann_dir.exists() else []

    print(f"Real JSON count: {len(real_jsons)}")
    print(f"Fake JSON count: {len(fake_jsons)}")

    print_section("Example real JSON paths")
    for path in real_jsons[:10]:
        print(path)

    print_section("Example fake JSON paths")
    for path in fake_jsons[:10]:
        print(path)

    if real_jsons:
        path = real_jsons[0]
        data = safe_load_json(path)
        print_section(f"Real JSON sample: {path}")
        if isinstance(data, dict):
            print("Top-level keys:")
            print(list(data.keys()))
            print("\nPartial content:")
            pprint(data, depth=3)

    if fake_jsons:
        path = fake_jsons[0]
        data = safe_load_json(path)
        print_section(f"Fake JSON sample: {path}")
        if isinstance(data, dict):
            print("Top-level keys:")
            print(list(data.keys()))
            print("\nPartial content:")
            pprint(data, depth=4)


def show_image_examples():
    print_section("Image files by folder")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]

    print(f"Total image files: {len(image_files)}")

    print("\nFirst 100 image files:")
    for path in image_files[:100]:
        print(path)


def search_matching_images_for_fake_examples():
    print_section("Matching image search for fake annotation examples")

    fake_ann_dir = ROOT / "Annotations" / "fakes"
    fake_jsons = sorted(fake_ann_dir.glob("*.json"))[:10]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]

    for fake_json in fake_jsons:
        stem = fake_json.stem
        print("\nFake annotation:")
        print(fake_json)
        print(f"Stem: {stem}")

        direct_matches = [img for img in image_files if img.stem == stem]
        contains_matches = [img for img in image_files if stem in img.stem or img.stem in stem]

        print(f"Direct image matches: {len(direct_matches)}")
        for img in direct_matches[:5]:
            print(f"  direct: {img}")

        print(f"Contains image matches: {len(contains_matches)}")
        for img in contains_matches[:10]:
            print(f"  contains: {img}")


def main():
    if not ROOT.exists():
        raise SystemExit(f"Root does not exist: {ROOT}")

    show_tree_examples()
    show_annotation_examples()
    show_image_examples()
    search_matching_images_for_fake_examples()


if __name__ == "__main__":
    main()