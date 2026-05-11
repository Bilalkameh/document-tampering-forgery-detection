from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

RAW_DIR = Path("data/raw/documents")

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
    image_paths.extend(RAW_DIR.rglob(ext))

print(f"Found {len(image_paths)} raw document images")

if not image_paths:
    raise SystemExit("No images found. Add images to data/raw/documents/ first.")

for path in image_paths[:10]:
    img = Image.open(path).convert("RGB")
    print(f"{path} | size={img.size}")

sample = random.sample(image_paths, min(6, len(image_paths)))

plt.figure(figsize=(12, 8))

for i, path in enumerate(sample):
    img = Image.open(path).convert("RGB")
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(path.name)
    plt.axis("off")

plt.tight_layout()
plt.show()