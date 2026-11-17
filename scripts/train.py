"""Fine-tune a COCO-pretrained YOLOv8 on NEU-DET.

Usage (repo root, venv active):
    python scripts/train.py                       # yolov8s, 640 px, 100 epochs
    python scripts/train.py --epochs 3 --name smoke

Outputs go to runs/<name>/:
    weights/best.pt   epoch with the highest val mAP50-95 (the "fitness" in Ultralytics 8.4)
    weights/last.pt   final epoch
    results.csv       per-epoch train/val losses and val metrics
    args.yaml         every setting used, including the defaults not set here
    *.png / *.jpg     loss curves, PR curves, confusion matrix, example batches

Settings not passed below are Ultralytics 8.4.215 defaults (see args.yaml in the run folder).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on NEU-DET.")
    p.add_argument("--data", type=Path, default=Path("data/neu-det-yolo/data.yaml"))
    p.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained weights to start from")
    p.add_argument("--imgsz", type=int, default=640, help="training/validation image size (pixels)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="run folder name (default: <model>_<imgsz>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"

    model = YOLO(args.model)  # downloads the COCO weights on first use
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        # Absolute path on purpose: a relative 'project' is placed under the runs_dir
        # setting plus the task name (runs/detect/...), which depends on where you run from.
        project=str(Path("runs").resolve()),
        name=name,
    )
    print(f"\nRun folder:   {model.trainer.save_dir}")
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
