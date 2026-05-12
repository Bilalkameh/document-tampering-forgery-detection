from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
from ultralytics import YOLO


def parse_cache(value: str):
    value = str(value).lower()
    if value in {"false", "none", "no", "0"}:
        return False
    if value in {"true", "yes", "1", "ram"}:
        return "ram"
    if value == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be one of: false, ram, disk")


def resolve_project_path(project: str) -> str:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = Path.cwd() / project_path
    return str(project_path.resolve())


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for document tampering localization.")

    # Core
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="sidtd_yolo_train")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--exist-ok", action="store_true")

    # Optimization
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)

    # Performance
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=parse_cache, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    # Augmentations
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)

    # Saving
    parser.add_argument("--save-period", type=int, default=-1)

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    project_path = resolve_project_path(args.project)

    print("\nYOLO Document Tampering Training")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Device:     {args.device}")
    print(f"Project:    {project_path}")
    print(f"Run name:   {args.name}")
    print("-" * 70)
    print(f"Torch:      {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    config_dir = Path("outputs/training_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved config to: {config_path}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=project_path,
        exist_ok=args.exist_ok,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        workers=args.workers,
        cache=args.cache,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=not args.no_amp,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        save_period=args.save_period,
        save=True,
        plots=True,
        val=True,
    )

    print("\nTraining finished.")


if __name__ == "__main__":
    main()