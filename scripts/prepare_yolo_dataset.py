#!/usr/bin/env python3
"""Prepare DeepPCB datasets for YOLOv8 Experiment B (Single image) and Experiment C (3-channel).

Experiment B:
  - Input: <id>_test.jpg (grayscale saved as 3-channel or 1-channel, standard image)
  - Labels: class cx cy w h (0-indexed)

Experiment C:
  - Input: 3-channel PNG [template, test, absdiff]
  - Labels: identical class cx cy w h

Splits:
  - test: Official test.txt (500 pairs)
  - trainval: 1000 pairs split 90% train (900 pairs) / 10% val (100 pairs), seed=42
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import cv2

# Canonical class names corresponding to DeepPCB class_id 1..6
CLASS_NAMES = [
    "open",             # 1 -> 0
    "short",            # 2 -> 1
    "mousebite",        # 3 -> 2
    "spur",             # 4 -> 3
    "copper",           # 5 -> 4 (Spurious Copper)
    "pin-hole",         # 6 -> 5
]


def parse_manifest(manifest_path: Path, pcb_data_root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Parse DeepPCB manifest into list of (sample_id, template_path, test_path, ann_path)."""
    samples = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        img_rel, ann_rel = parts
        stem_rel = img_rel[:-4] if img_rel.endswith(".jpg") else img_rel
        sample_id = Path(stem_rel).name
        template_path = pcb_data_root / f"{stem_rel}_temp.jpg"
        test_path = pcb_data_root / f"{stem_rel}_test.jpg"
        ann_path = pcb_data_root / ann_rel
        if not (template_path.is_file() and test_path.is_file() and ann_path.is_file()):
            raise FileNotFoundError(f"Missing file for sample {sample_id} in {manifest_path}")
        samples.append((sample_id, template_path, test_path, ann_path))
    return samples


def convert_annotations_to_yolo(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert DeepPCB bounding box (x1, y1, x2, y2, class_id) to YOLO format (cls, cx, cy, w, h)."""
    lines = []
    for raw in ann_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        x1, y1, x2, y2, class_id = [int(p) for p in parts]
        if not 1 <= class_id <= 6:
            raise ValueError(f"Invalid class_id {class_id} in {ann_path}")
        cls_idx = class_id - 1
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = max(0.0, (x2 - x1) / float(img_w))
        h = max(0.0, (y2 - y1) / float(img_h))
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def build_experiment_b(
    splits: dict[str, list[tuple[str, Path, Path, Path]]],
    output_dir: Path,
) -> Path:
    """Build dataset for Experiment B: Single test image."""
    print(f"\n[Exp B] Building single-image dataset in {output_dir}...")
    for split_name, sample_list in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, _template_p, test_p, ann_p in sample_list:
            img = cv2.imread(str(test_p))
            if img is None:
                raise ValueError(f"Failed to read {test_p}")
            h, w = img.shape[:2]

            # Save test image as JPG
            out_img_path = img_dir / f"{sample_id}.jpg"
            cv2.imwrite(str(out_img_path), img)

            # Convert labels
            yolo_labels = convert_annotations_to_yolo(ann_p, w, h)
            out_lbl_path = lbl_dir / f"{sample_id}.txt"
            out_lbl_path.write_text("\n".join(yolo_labels) + ("\n" if yolo_labels else ""), encoding="utf-8")

    yaml_path = output_dir / "deeppcb_b.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_block}\n"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[Exp B] Dataset YAML written to {yaml_path}")
    return yaml_path


def build_experiment_c(
    splits: dict[str, list[tuple[str, Path, Path, Path]]],
    output_dir: Path,
) -> Path:
    """Build dataset for Experiment C: 3-channel composite [template, test, absdiff] as PNG."""
    print(f"\n[Exp C] Building 3-channel composite dataset in {output_dir}...")
    for split_name, sample_list in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, template_p, test_p, ann_p in sample_list:
            t_gray = cv2.imread(str(template_p), cv2.IMREAD_GRAYSCALE)
            x_gray = cv2.imread(str(test_p), cv2.IMREAD_GRAYSCALE)
            if t_gray is None or x_gray is None:
                raise ValueError(f"Failed to read images for sample {sample_id}")
            h, w = t_gray.shape[:2]

            # Compute absolute difference
            diff = cv2.absdiff(t_gray, x_gray)

            # Stack into 3-channel image: [B, G, R] = [template, test, diff]
            composite = cv2.merge([t_gray, x_gray, diff])

            # Save as PNG to avoid compression artifacts on difference channel
            out_img_path = img_dir / f"{sample_id}.png"
            cv2.imwrite(str(out_img_path), composite)

            # Convert labels
            yolo_labels = convert_annotations_to_yolo(ann_p, w, h)
            out_lbl_path = lbl_dir / f"{sample_id}.txt"
            out_lbl_path.write_text("\n".join(yolo_labels) + ("\n" if yolo_labels else ""), encoding="utf-8")

    yaml_path = output_dir / "deeppcb_c.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_block}\n"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[Exp C] Dataset YAML written to {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DeepPCB YOLO datasets for Exp B & C.")
    parser.add_argument(
        "--pcb-data",
        type=Path,
        default=Path("data/DeepPCB/PCBData"),
        help="Path to DeepPCB PCBData directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root output directory for YOLO datasets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Ratio of trainval to use for val")
    args = parser.parse_args()

    pcb_root = args.pcb_data.resolve()
    if not pcb_root.is_dir():
        raise FileNotFoundError(f"DeepPCB PCBData directory not found at {pcb_root}")

    trainval_manifest = pcb_root / "trainval.txt"
    test_manifest = pcb_root / "test.txt"

    trainval_samples = parse_manifest(trainval_manifest, pcb_root)
    test_samples = parse_manifest(test_manifest, pcb_root)

    print(f"Loaded {len(trainval_samples)} trainval samples, {len(test_samples)} test samples.")

    # Deterministic train / val split
    rng = random.Random(args.seed)
    # Shuffle a copy
    shuffled = list(trainval_samples)
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * args.val_ratio)
    val_samples = sorted(shuffled[:val_count], key=lambda x: x[0])
    train_samples = sorted(shuffled[val_count:], key=lambda x: x[0])
    test_samples = sorted(test_samples, key=lambda x: x[0])

    print(f"Split breakdown: Train = {len(train_samples)}, Val = {len(val_samples)}, Test = {len(test_samples)}")

    # Audit zero leakage
    train_ids = {s[0] for s in train_samples}
    val_ids = {s[0] for s in val_samples}
    test_ids = {s[0] for s in test_samples}

    assert not (train_ids & val_ids), f"Leakage train & val: {train_ids & val_ids}"
    assert not (train_ids & test_ids), f"Leakage train & test: {train_ids & test_ids}"
    assert not (val_ids & test_ids), f"Leakage val & test: {val_ids & test_ids}"

    split_hashes = {
        "train": hashlib.sha256("\n".join(sorted(train_ids)).encode("utf-8")).hexdigest()[:16],
        "val": hashlib.sha256("\n".join(sorted(val_ids)).encode("utf-8")).hexdigest()[:16],
        "test": hashlib.sha256("\n".join(sorted(test_ids)).encode("utf-8")).hexdigest()[:16],
    }
    print(f"Zero-leakage audit PASSED. Split hashes: {split_hashes}")

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    # Generate Exp B and Exp C datasets
    exp_b_dir = (args.output_root / "yolo_exp_b").resolve()
    exp_c_dir = (args.output_root / "yolo_exp_c").resolve()

    build_experiment_b(splits, exp_b_dir)
    build_experiment_c(splits, exp_c_dir)
    print("\nDataset generation finished successfully.")


if __name__ == "__main__":
    main()
