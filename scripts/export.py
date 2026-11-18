"""Export the fine-tuned YOLOv8 weights to ONNX (FP32) and TensorRT engines.

Usage (repo root, venv active):
    python scripts/export.py                                          # best.engine (FP16) + best.onnx
    python scripts/export.py --formats onnx                           # ONNX only
    python scripts/export.py --formats engine --int8 --tag int8       # best_int8.engine
    python scripts/export.py --formats engine --imgsz 672 --tag 672   # best_672.engine (FP16, 672x672)

Writes next to the PyTorch weights:
    runs/<name>/weights/best.onnx         ONNX, FP32, fixed input 1x3x640x640
    runs/<name>/weights/best.engine       TensorRT FP16, only valid on THIS GPU model and TensorRT version
    runs/<name>/weights/best_<tag>.*      tagged variants; the tag keeps best.engine from being overwritten

Ultralytics 8.4.215 specifics (checked in the installed source):
- Precision is set with quantize=16 (FP16) or quantize=8 (INT8); half=True / int8=True are deprecated.
- Exports are named after the source file, so a tagged export copies best.pt to best_<tag>.pt first.
- An engine export first writes an intermediate ONNX file with the same stem, so the engine is built
  before the plain ONNX export. That way best.onnx always ends up as the standalone FP32 file.
- INT8 on TensorRT 10 uses MinMax calibration over the images of the `split` passed in. FP16 is also
  enabled, and TensorRT picks INT8 or FP16 per layer by speed, not accuracy. For accuracy, Ultralytics
  keeps the head's Sigmoid in FP32 and its output and DFL convolutions in FP16 (utils/export/engine.py).
  This script calibrates on the TRAIN split, so val and test images are never seen during calibration.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export best.pt to ONNX and TensorRT.")
    p.add_argument("--weights", type=Path, default=Path("runs/yolov8s_640/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640, help="export input size (pixels, square)")
    p.add_argument("--formats", nargs="+", choices=("onnx", "engine"), default=["onnx", "engine"])
    p.add_argument("--int8", action="store_true", help="engine only: INT8 instead of FP16")
    p.add_argument("--data", type=Path, default=Path("data/neu-det-yolo/data.yaml"),
                   help="dataset yaml, used for INT8 calibration images")
    p.add_argument("--tag", default=None, help="name suffix, e.g. 'int8' -> best_int8.engine")
    p.add_argument("--workspace", type=float, default=None,
                   help="TensorRT builder workspace in GiB (default: let TensorRT decide)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    if (args.int8 or args.imgsz != 640) and not args.tag:
        sys.exit("Use --tag for INT8 or non-640 exports so best.engine / best.onnx are not overwritten.")

    src = args.weights
    if args.tag:  # Ultralytics names exports after the source file
        src = args.weights.with_name(f"{args.weights.stem}_{args.tag}{args.weights.suffix}")
        shutil.copy2(args.weights, src)

    for fmt in [f for f in ("engine", "onnx") if f in args.formats]:  # engine first, see docstring
        kwargs = {
            "format": fmt,
            "imgsz": args.imgsz,
            "batch": 1,         # latency is measured one image at a time
            "dynamic": False,   # fixed input shape: lets ONNX Runtime keep data on the GPU (IO binding)
            "device": 0,        # TensorRT must build on the GPU it will run on
        }
        if fmt == "engine":
            if args.int8:
                kwargs.update(quantize=8, data=str(args.data.resolve()), split="train")
            else:
                kwargs.update(quantize=16)
            kwargs["workspace"] = args.workspace

        t0 = time.perf_counter()
        out = Path(YOLO(src).export(**kwargs))
        print(f"\n[{fmt}] {out}  {out.stat().st_size / 2**20:.1f} MB, exported in {time.perf_counter() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
