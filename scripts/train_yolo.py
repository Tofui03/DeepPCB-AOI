#!/usr/bin/env python3
"""Training runner for DeepPCB YOLOv8n Experiment B & Experiment C on local GPU.

Experiment B:
  - Input: Single test image (data/yolo_exp_b/deeppcb_b.yaml)
  - Standard augmentations

Experiment C:
  - Input: 3-channel composite [template, test, absdiff] (data/yolo_exp_c/deeppcb_c.yaml)
  - Crucial: disable color jitter (hsv_h=0, hsv_s=0, hsv_v=0) to preserve channel semantics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_experiment(
    exp_name: str,
    data_yaml: Path,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str | int = 0,
    seed: int = 42,
    export_onnx: bool = True,
) -> Path:
    print("\n" + "=" * 60)
    print(f"Starting Training: {exp_name}")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Device: {device}, Seed: {seed}")
    print("=" * 60)

    model = YOLO("yolov8n.pt")

    # Extra training parameters
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "seed": seed,
        "project": "runs/train",
        "name": exp_name,
        "exist_ok": True,
        "plots": True,
        "save": True,
        "val": True,
    }

    # For Experiment C, turn off color/HSV augmentations
    if exp_name.endswith("_c"):
        print("[Notice] Disabling HSV augmentations for Experiment C (channel-semantic integrity).")
        train_kwargs.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        })

    model.train(**train_kwargs)

    best_pt = Path(f"runs/train/{exp_name}/weights/best.pt")
    print(f"\nTraining completed for {exp_name}. Best weights: {best_pt}")

    if export_onnx and best_pt.is_file():
        print(f"Exporting {best_pt} to ONNX format...")
        best_model = YOLO(str(best_pt))
        exported_onnx = best_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"Exported ONNX model: {exported_onnx}")

    return best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on DeepPCB (Exp B / C)")
    parser.add_argument("--exp", choices=["b", "c", "both"], default="both", help="Experiment to run")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (e.g. 16 or 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default=0, help="CUDA device index or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-export-onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    export_onnx = not args.no_export_onnx

    if args.exp in ["b", "both"]:
        yaml_b = Path("data/yolo_exp_b/deeppcb_b.yaml").resolve()
        if not yaml_b.is_file():
            raise FileNotFoundError(f"Missing {yaml_b}. Run scripts/prepare_yolo_dataset.py first.")
        train_experiment(
            exp_name="deeppcb_yolov8n_exp_b",
            data_yaml=yaml_b,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            seed=args.seed,
            export_onnx=export_onnx,
        )

    if args.exp in ["c", "both"]:
        yaml_c = Path("data/yolo_exp_c/deeppcb_c.yaml").resolve()
        if not yaml_c.is_file():
            raise FileNotFoundError(f"Missing {yaml_c}. Run scripts/prepare_yolo_dataset.py first.")
        train_experiment(
            exp_name="deeppcb_yolov8n_exp_c",
            data_yaml=yaml_c,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            seed=args.seed,
            export_onnx=export_onnx,
        )


if __name__ == "__main__":
    main()
