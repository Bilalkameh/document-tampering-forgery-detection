from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for document tampering localization.")

    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO pretrained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", type=str, default="tamper_yolov8n_baseline", help="Run name")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, 0, 1, etc.")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    print("Starting YOLO training")
    print(f"Data:   {args.data}")
    print(f"Model:  {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image:  {args.imgsz}")
    print(f"Batch:  {args.batch}")
    print(f"Device: {args.device}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        project="runs/detect",
        patience=10,
        save=True,
        plots=True,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()