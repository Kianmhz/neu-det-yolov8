"""Export the fine-tuned YOLOv8 weights to ONNX (FP32) and a TensorRT engine (FP16).

Usage (repo root, venv active):
    python scripts/export.py                    # both formats
    python scripts/export.py --formats onnx     # ONNX only, e.g. if TensorRT will not install

Writes next to the PyTorch weights:
    runs/<name>/weights/best.onnx     ONNX, FP32, fixed input 1x3x640x640
    runs/<name>/weights/best.engine   TensorRT, FP16, only valid on THIS GPU model and TensorRT version

Ultralytics 8.4.215 specifics (checked in the installed source):
- FP16 is requested with quantize=16; the older half=True is deprecated.
- An engine export first writes an intermediate best.onnx, so the engine is built before the
  plain ONNX export. That way best.onnx always ends up as the standalone FP32 file.
"""
import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export best.pt to ONNX and TensorRT.")
    p.add_argument("--weights", type=Path, default=Path("runs/yolov8s_640/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640, help="must match the size used in training")
    p.add_argument("--formats", nargs="+", choices=("onnx", "engine"), default=["onnx", "engine"])
    p.add_argument("--workspace", type=float, default=None,
                   help="TensorRT builder workspace in GiB (default: let TensorRT decide)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")

    for fmt in [f for f in ("engine", "onnx") if f in args.formats]:  # engine first, see docstring
        kwargs = {
            "format": fmt,
            "imgsz": args.imgsz,
            "batch": 1,         # latency is measured one image at a time
            "dynamic": False,   # fixed input shape: lets ONNX Runtime keep data on the GPU (IO binding)
            "device": 0,        # TensorRT must build on the GPU it will run on
        }
        if fmt == "engine":
            kwargs.update(quantize=16, workspace=args.workspace)  # FP16 engine

        t0 = time.perf_counter()
        out = Path(YOLO(args.weights).export(**kwargs))
        print(f"\n[{fmt}] {out}  {out.stat().st_size / 2**20:.1f} MB, exported in {time.perf_counter() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
