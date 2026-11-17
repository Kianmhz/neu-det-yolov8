"""Convert NEU-DET (Pascal VOC XML) to YOLO format with a stratified train/val/test split.

Usage (from the repo root, venv active):
    python scripts/prepare_dataset.py

Input, the official download unzipped into data/raw:
    data/raw/NEU-DET/IMAGES/<class>_<n>.jpg
    data/raw/NEU-DET/ANNOTATIONS/<class>_<n>.xml

Output, what Ultralytics trains on:
    data/neu-det-yolo/images/{train,val,test}/*.jpg
    data/neu-det-yolo/labels/{train,val,test}/*.txt   one line per box: class cx cy w h, all 0-1
    data/neu-det-yolo/data.yaml
    splits/{train,val,test}.txt                        file lists, committed so the split is reproducible
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

# List order defines the class ids 0..5 written into every label file and data.yaml.
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NEU-DET to YOLO format and split it.")
    p.add_argument("--src", type=Path, default=Path("data/raw/NEU-DET"))
    p.add_argument("--dst", type=Path, default=Path("data/neu-det-yolo"))
    p.add_argument("--splits-dir", type=Path, default=Path("splits"))
    p.add_argument("--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="delete --dst first if it exists")
    return p.parse_args()


def category_of(stem: str) -> str:
    """'rolled-in_scale_12' -> 'rolled-in_scale' (the image's class, taken from its file name)."""
    return stem.rsplit("_", 1)[0]


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int, stats: Counter) -> tuple[list[str], list[str]]:
    """Convert one VOC XML file. Returns (YOLO label lines, class name of each kept box)."""
    root = ET.parse(xml_path).getroot()

    # Images are paired with XML files by file stem, because the <filename> field is unreliable
    # (174 files omit '.jpg'). Count any that name a *different* image, in case that matters.
    if Path((root.findtext("filename") or "").strip()).stem != xml_path.stem:
        stats["xml <filename> names another image"] += 1
    if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (img_w, img_h):
        stats["xml <size> differs from real image size"] += 1

    lines, names = [], []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip()
        if name not in CLASSES:
            raise ValueError(f"{xml_path.name}: unknown class {name!r}")

        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Clip to the image. Boxes that become empty (zero width/height) are dropped.
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats["boxes clipped to image"] += 1
        if cx2 <= cx1 or cy2 <= cy1:
            stats["boxes dropped (empty)"] += 1
            continue

        # VOC: pixel corners. YOLO: box centre and size, divided by image size so all values are 0-1.
        # (VOC's 1-based pixel convention shifts boxes by at most 1 px = 0.5% of a 200 px image.)
        cx, cy = (cx1 + cx2) / 2 / img_w, (cy1 + cy2) / 2 / img_h
        w, h = (cx2 - cx1) / img_w, (cy2 - cy1) / img_h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        names.append(name)
    return lines, names


def main() -> None:
    args = parse_args()
    if abs(sum(args.fractions) - 1.0) > 1e-6:
        sys.exit(f"--fractions must sum to 1, got {args.fractions}")
    img_dir, ann_dir = args.src / "IMAGES", args.src / "ANNOTATIONS"
    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        sys.exit(f"No XML files found in {ann_dir}")

    if args.dst.exists():
        if not args.overwrite:
            sys.exit(f"{args.dst} already exists. Rerun with --overwrite to rebuild it.")
        shutil.rmtree(args.dst)

    # 1) Group images by class, taken from the file name. Each class has 300 images.
    by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
    for x in xmls:
        cat = category_of(x.stem)
        if cat not in by_class:
            sys.exit(f"{x.name}: file name does not start with a known class")
        if not (img_dir / f"{x.stem}.jpg").is_file():
            sys.exit(f"{x.name}: no matching image {x.stem}.jpg")
        by_class[cat].append(x.stem)

    # 2) Stratified split: shuffle each class with a fixed seed, then cut it 70/15/15.
    #    Every split gets the same share of every class, and the same seed gives the same split.
    rng = random.Random(args.seed)
    split_of: dict[str, str] = {}
    for cat in CLASSES:
        stems = sorted(by_class[cat])  # sort first so the result never depends on directory order
        rng.shuffle(stems)
        n_train = round(len(stems) * args.fractions[0])
        n_val = round(len(stems) * args.fractions[1])
        for i, stem in enumerate(stems):
            split_of[stem] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    # 3) Copy images and write one label file per image.
    for split in SPLITS:
        (args.dst / "images" / split).mkdir(parents=True)
        (args.dst / "labels" / split).mkdir(parents=True)
    stats: Counter = Counter()
    n_images: Counter = Counter()  # (split, class from file name)
    n_boxes: Counter = Counter()   # (split, class from box label)
    for stem, split in sorted(split_of.items()):
        img_path = img_dir / f"{stem}.jpg"
        with Image.open(img_path) as im:
            w, h = im.size
            stats[f"images in PIL mode {im.mode}, size {w}x{h}"] += 1
        lines, names = voc_to_yolo(ann_dir / f"{stem}.xml", w, h, stats)
        if any(n != category_of(stem) for n in names):
            stats["images with a box of another class"] += 1
        if not lines:
            stats["images with no boxes"] += 1

        shutil.copy2(img_path, args.dst / "images" / split / img_path.name)
        (args.dst / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
        n_images[split, category_of(stem)] += 1
        for n in names:
            n_boxes[split, n] += 1

    # 4) Split lists (committed to the repo) and data.yaml (machine-specific, absolute path).
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        files = sorted(f"{s}.jpg" for s, sp in split_of.items() if sp == split)
        (args.splits_dir / f"{split}.txt").write_text("".join(f"{f}\n" for f in files))
    yaml_lines = [
        f"# Generated by scripts/prepare_dataset.py (seed={args.seed}, fractions={list(args.fractions)})",
        f"path: {args.dst.resolve()}",  # absolute: a relative path that doesn't exist is looked up in DATASETS_DIR
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(CLASSES)),
    ]
    (args.dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    # 5) Report.
    print(f"{'images / boxes':<16}" + "".join(f"{s:>14}" for s in SPLITS) + f"{'total':>14}")
    for c in [*CLASSES, None]:
        row = f"{c or 'TOTAL':<16}"
        for s in [*SPLITS, None]:
            imgs = sum(v for (sp, cl), v in n_images.items() if (s is None or sp == s) and (c is None or cl == c))
            bxs = sum(v for (sp, cl), v in n_boxes.items() if (s is None or sp == s) and (c is None or cl == c))
            row += f"{imgs:>8} /{bxs:>4}"
        print(row)
    print("\nData checks:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\nWrote {args.dst / 'data.yaml'} and {args.splits_dir}/{{train,val,test}}.txt")


if __name__ == "__main__":
    main()
