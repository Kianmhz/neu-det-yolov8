"""Benchmark accuracy and latency of the fine-tuned model in PyTorch, ONNX Runtime and TensorRT.

Usage (repo root, venv active, nothing else using the GPU):
    python scripts/benchmark.py

Variants (a missing file is skipped and noted, e.g. if TensorRT failed):
    PyTorch FP32        best.pt
    PyTorch FP16        best.pt with quantize=16
    ONNX Runtime FP32   best.onnx on the CUDA execution provider
    TensorRT FP16       best.engine

For each variant:
  1. Accuracy: Ultralytics validation on the TEST split (270 images never used for training or for
     choosing best.pt), batch 1, rect=False so every format sees the same 640x640 input
     -> mAP50, mAP50-95, per-class AP, plus Ultralytics' own per-image
     preprocess / inference / postprocess times.
  2. Latency: one fixed 1x3x640x640 input, warm-up passes, then timed passes with a CUDA sync
     after each -> median and p95 of the model forward pass alone (no pre/post-processing).
     Ultralytics validation warms up with a single pass, so its averages can include slow first runs.

Writes results/benchmark.json (everything) and results/benchmark.md (tables for the README).
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

VARIANTS = [  # (label, file name, quantize passed to val / AutoBackend)
    ("PyTorch FP32", "best.pt", None),
    ("PyTorch FP16", "best.pt", 16),
    ("ONNX Runtime FP32", "best.onnx", None),
    ("TensorRT FP16", "best.engine", None),  # precision is baked into the engine at build time
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PyTorch, ONNX Runtime and TensorRT.")
    p.add_argument("--weights-dir", type=Path, default=Path("runs/yolov8s_640/weights"))
    p.add_argument("--data", type=Path, default=Path("data/neu-det-yolo/data.yaml"))
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50, help="untimed forward passes before timing")
    p.add_argument("--runs", type=int, default=500, help="timed forward passes")
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def evaluate(path: Path, quantize, label: str, args: argparse.Namespace) -> dict:
    """mAP and per-class AP on the test split, one image at a time."""
    m = YOLO(str(path), task="detect").val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=1,
        device=0,
        quantize=quantize,
        # Same input for every format. model.val() defaults to rect=True, and validation pads with
        # pad=0.5, so a .pt model sees a 672x672 canvas (640 px image + 16 px grey border) while
        # fixed-shape ONNX/TensorRT files are forced to rect=False and see exactly 640x640.
        rect=False,
        plots=label == "PyTorch FP32",  # confusion matrix and PR curves on the test set, once
        verbose=False,
        project=str(Path("runs/benchmark").resolve()),
        name=slug(label),
        exist_ok=True,
    )
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[m.names[int(c)]] = {"precision": float(p), "recall": float(r),
                                      "ap50": float(ap50), "ap50_95": float(ap)}
    return {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
        "val_speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
    }


@torch.inference_mode()
def time_forward(path: Path, quantize, args: argparse.Namespace) -> dict:
    """Latency of the forward pass alone on a fixed input."""
    backend = AutoBackend(model=str(path), device=torch.device("cuda:0"), fp16=quantize == 16, verbose=False)
    dtype = torch.half if backend.fp16 else torch.float  # the engine/ONNX file decides its own input type
    im = torch.rand(1, 3, args.imgsz, args.imgsz, device=backend.device, dtype=dtype)

    for _ in range(args.warmup):
        backend(im)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        backend(im)
        torch.cuda.synchronize()  # GPU work is asynchronous: wait for it before stopping the clock
        times.append((time.perf_counter() - t0) * 1e3)

    session = getattr(backend.backend, "session", None)  # ONNX Runtime only
    return {
        "median": statistics.median(times),
        "p95": float(np.percentile(times, 95)),
        "mean": statistics.fmean(times),
        "min": min(times),
        "warmup": args.warmup,
        "runs": args.runs,
        "input_dtype": str(dtype).replace("torch.", ""),
        "onnxruntime_providers": session.get_providers() if session is not None else None,
    }


def versions() -> dict:
    v = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    }
    for name in ("onnxruntime", "tensorrt"):
        try:
            v[name] = __import__(name).__version__
        except ImportError:
            v[name] = None
    return v


def markdown(res: dict) -> str:
    env, rows = res["env"], res["variants"]
    ran = {k: v for k, v in rows.items() if "skipped" not in v}
    base = ran.get("PyTorch FP32", {}).get("latency_ms", {}).get("median")
    out = [
        f"Generated by `scripts/benchmark.py` on {env['date']}. {env['gpu']}, torch {env['torch']} "
        f"(CUDA {env['cuda']}), Ultralytics {env['ultralytics']}, ONNX Runtime {env['onnxruntime']}, "
        f"TensorRT {env['tensorrt']}. Test split ({res['split']}), batch 1, {res['imgsz']} px. "
        f"Forward latency: median and p95 of {res['runs']} timed passes after {res['warmup']} warm-up passes.",
        "",
        "| Variant | File (MB) | mAP50 | mAP50-95 | Forward median (ms) | Forward p95 (ms) | Speed-up vs PyTorch FP32 "
        "| Ultralytics val pre / inf / post (ms/img) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in rows.items():
        if "skipped" in r:
            out.append(f"| {label} | – | – | – | – | – | – | {r['skipped']} |")
            continue
        lat, sp = r["latency_ms"], r["val_speed_ms_per_image"]
        speedup = f"{base / lat['median']:.2f}x" if base else "–"
        out.append(
            f"| {label} | {r['size_mb']:.1f} | {r['map50']:.3f} | {r['map50_95']:.3f} | {lat['median']:.2f} "
            f"| {lat['p95']:.2f} | {speedup} | {sp['preprocess']:.2f} / {sp['inference']:.2f} / {sp['postprocess']:.2f} |"
        )
    classes = next(iter(ran.values()))["per_class"].keys() if ran else []
    out += ["", "Per-class AP50 / AP50-95 on the test split:", "",
            "| Class | " + " | ".join(ran) + " |", "|---|" + "---|" * len(ran)]
    for c in classes:
        cells = [f"{r['per_class'][c]['ap50']:.3f} / {r['per_class'][c]['ap50_95']:.3f}" for r in ran.values()]
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    res = {"env": versions(), "split": args.split, "imgsz": args.imgsz,
           "warmup": args.warmup, "runs": args.runs, "variants": {}}

    for label, fname, quantize in VARIANTS:
        path = args.weights_dir / fname
        if not path.is_file():
            print(f"\n=== {label}: SKIPPED, {path} not found")
            res["variants"][label] = {"skipped": f"{fname} not found"}
            continue
        print(f"\n=== {label}: {path}")
        acc = evaluate(path, quantize, label, args)
        lat = time_forward(path, quantize, args)
        res["variants"][label] = {"file": str(path), "size_mb": path.stat().st_size / 2**20, **acc, "latency_ms": lat}
        print(f"    mAP50 {acc['map50']:.3f}  mAP50-95 {acc['map50_95']:.3f}  "
              f"forward median {lat['median']:.2f} ms  p95 {lat['p95']:.2f} ms"
              + (f"  providers {lat['onnxruntime_providers']}" if lat["onnxruntime_providers"] else ""))

    args.out.mkdir(exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
    md = markdown(res)
    (args.out / "benchmark.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {args.out / 'benchmark.json'} and {args.out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
