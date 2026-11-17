"""Make the README figures in assets/.

Usage (repo root, venv active, after train.py and benchmark.py):
    python scripts/make_figures.py

Writes:
    assets/test_predictions.jpg          ground truth vs prediction for the first 2 test images of each
                                         class (sorted by file name, not hand-picked)
    assets/confusion_matrix_conf025.png  test-set confusion matrix at confidence 0.25
    assets/training_curves.png           losses and val metrics per epoch (from the training run)
    assets/pr_curve_test.png             precision-recall curves on the test set (from benchmark.py)
    assets/f1_curve_test.png             F1 vs confidence on the test set (from benchmark.py)

Why a separate confusion matrix: Ultralytics validation uses conf=0.001 so the PR curve covers every
threshold, and its confusion matrix is built at that same 0.001. Almost every ground-truth box then gets
matched by some low-confidence box, which hides the misses you would see in deployment. The extra
validation run here uses conf=0.25 for the plot only; its mAP is truncated and must not be reported.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SHORT = ["Cr", "In", "Pa", "PS", "RS", "Sc"]  # abbreviations used by the dataset authors
COLORS = [(255, 128, 0), (0, 200, 255), (0, 220, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0)]  # BGR
PANEL = 320  # pixels per image panel (200 px originals are upscaled for readability)
TITLE_H = 28


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make README figures.")
    p.add_argument("--weights", type=Path, default=Path("runs/yolov8s_640/weights/best.engine"),
                   help="model used for predictions and the conf=0.25 confusion matrix")
    p.add_argument("--data", type=Path, default=Path("data/neu-det-yolo/data.yaml"))
    p.add_argument("--train-run", type=Path, default=Path("runs/yolov8s_640"))
    p.add_argument("--bench-run", type=Path, default=Path("runs/benchmark/pytorch_fp32"))
    p.add_argument("--per-class", type=int, default=2, help="test images per class in the prediction grid")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for predictions")
    p.add_argument("--out", type=Path, default=Path("assets"))
    return p.parse_args()


def draw_boxes(img: np.ndarray, boxes, scale: float) -> np.ndarray:
    """boxes: iterable of (x1, y1, x2, y2, cls, conf or None) in original-image pixels."""
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    for x1, y1, x2, y2, c, conf in boxes:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(out, p1, p2, COLORS[c], 2)
        label = SHORT[c] if conf is None else f"{SHORT[c]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), COLORS[c], -1)
        cv2.putText(out, label, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def with_title(panel: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((TITLE_H, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def prediction_grid(model: YOLO, args: argparse.Namespace) -> np.ndarray:
    root = args.data.parent
    test_files = sorted(Path("splits/test.txt").read_text().split())
    rows = []
    for ci, cname in enumerate(CLASSES):
        picks = [f for f in test_files if f.rsplit("_", 1)[0] == cname][: args.per_class]
        cells = []
        for fname in picks:
            img_path = root / "images" / "test" / fname
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gt = []
            for line in (root / "labels" / "test" / f"{Path(fname).stem}.txt").read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                gt.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, int(c), None))
            r = model.predict(str(img_path), imgsz=640, conf=args.conf, device=0, verbose=False)[0]
            pred = [(*xyxy, int(c), float(p)) for xyxy, c, p in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            scale = PANEL / max(h, w)
            stem = Path(fname).stem
            cells.append(with_title(draw_boxes(img, gt, scale), f"{stem}: ground truth ({len(gt)})"))
            cells.append(with_title(draw_boxes(img, pred, scale), f"prediction ({len(pred)})"))
            print(f"{stem:<22} ground truth {len(gt)}  predicted {len(pred)}")
        rows.append(np.hstack(cells))

    legend = np.full((TITLE_H + 6, rows[0].shape[1], 3), 255, np.uint8)
    x = 6
    for s, cname, col in zip(SHORT, CLASSES, COLORS):
        cv2.rectangle(legend, (x, 8), (x + 14, 24), col, -1)
        text = f"{s} = {cname}"
        cv2.putText(legend, text, (x + 20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x += 34 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    note = f"Model: {args.weights.name}, confidence >= {args.conf}. First {args.per_class} test images per class."
    cv2.putText(legend, note, (x + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([legend, *rows])


def main() -> None:
    args = parse_args()
    args.out.mkdir(exist_ok=True)
    model = YOLO(str(args.weights), task="detect")

    grid = prediction_grid(model, args)
    cv2.imwrite(str(args.out / "test_predictions.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Confusion matrix at the deployment threshold (plot only; this run's mAP is not valid).
    cm_run = Path("runs/figures/cm_conf025").resolve()
    model.val(data=str(args.data.resolve()), split="test", imgsz=640, batch=1, rect=False, device=0,
              conf=args.conf, plots=True, verbose=False, project=str(cm_run.parent), name=cm_run.name,
              exist_ok=True)
    print("(The mAP printed above uses conf=0.25 and is truncated; it is NOT a valid result.)")

    copies = {
        cm_run / "confusion_matrix_normalized.png": "confusion_matrix_conf025.png",
        args.train_run / "results.png": "training_curves.png",
        args.bench_run / "BoxPR_curve.png": "pr_curve_test.png",
        args.bench_run / "BoxF1_curve.png": "f1_curve_test.png",
    }
    for src, dst in copies.items():
        shutil.copy2(src, args.out / dst)
    print(f"\nWrote {args.out}/test_predictions.jpg and {', '.join(copies.values())}")


if __name__ == "__main__":
    main()
