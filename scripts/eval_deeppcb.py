#!/usr/bin/env python3
"""Unified evaluation script for DeepPCB defect detectors.

Evaluates:
  - Method A: Traditional Differential ONNX Detector (core.vision.onnx_detector.ONNXPCBDetector)
  - Method B: YOLOv8n Single Image (<id>_test.jpg)
  - Method C: YOLOv8n 3-Channel Composite ([template, test, absdiff])

Supports:
  - Exact IoU matching at IoU=0.33 (Tang et al. DeepPCB benchmark standard) and IoU=0.50
  - Class-aware metrics (P, R, F1 per class and overall)
  - Class-agnostic metrics (pure localization P, R, F1)
  - Latency profiling (mean, p50, p95 ms per frame)
  - Automatic failure analysis and visualization of Top Bad Cases
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

CLASS_NAMES = ["Open", "Short", "Mousebite", "Spur", "Spurious Copper", "Pin-hole"]
DEEPPCB_CLASS_MAP = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Spurious Copper",
    6: "Pin-hole",
}
NAME_TO_CLASS = {name.lower(): name for name in CLASS_NAMES}
# Also map short names
NAME_TO_CLASS["copper"] = "Spurious Copper"


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    defect_class: str
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def compute_iou(b1: Box, b2: Box) -> float:
    ix1 = max(b1.x1, b2.x1)
    iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2)
    iy2 = min(b1.y2, b2.y2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = b1.area + b2.area - inter
    return float(inter) / float(union) if union > 0 else 0.0


def parse_ground_truth(ann_path: Path) -> list[Box]:
    boxes = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            x1, y1, x2, y2, cid = [int(p) for p in parts]
            cls_name = DEEPPCB_CLASS_MAP.get(cid, f"Class{cid}")
            boxes.append(Box(x1, y1, x2, y2, cls_name, 1.0))
    return boxes


def match_detections(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_thresh: float = 0.33,
    class_aware: bool = True,
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    """Greedy IoU matching.

    Returns:
        (tp, fp, fn, per_class_counts)
    """
    sorted_preds = sorted(pred_boxes, key=lambda b: b.confidence, reverse=True)
    matched_gt = set()
    tp = 0
    fp = 0

    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if class_aware and pred.defect_class.lower() != gt.defect_class.lower():
                continue

            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp += 1
            canonical_cls = NAME_TO_CLASS.get(pred.defect_class.lower(), pred.defect_class)
            if canonical_cls in per_class:
                per_class[canonical_cls]["tp"] += 1
        else:
            fp += 1
            canonical_cls = NAME_TO_CLASS.get(pred.defect_class.lower(), pred.defect_class)
            if canonical_cls in per_class:
                per_class[canonical_cls]["fp"] += 1

    fn = len(gt_boxes) - len(matched_gt)
    for i, gt in enumerate(gt_boxes):
        if i not in matched_gt:
            canonical_cls = NAME_TO_CLASS.get(gt.defect_class.lower(), gt.defect_class)
            if canonical_cls in per_class:
                per_class[canonical_cls]["fn"] += 1

    return tp, fp, fn, per_class


def compute_prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def run_evaluation(
    model_fn: Callable[[np.ndarray, np.ndarray], list[Box]],
    dataset_manifest: Path,
    pcb_root: Path,
    iou_thresholds: list[float] = [0.33, 0.50],
    visualize_bad_cases_dir: Path | None = None,
    max_bad_cases: int = 10,
) -> dict[str, Any]:
    lines = dataset_manifest.read_text(encoding="utf-8").splitlines()
    samples = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            img_rel, ann_rel = parts
            stem_rel = img_rel[:-4] if img_rel.endswith(".jpg") else img_rel
            samples.append((
                Path(stem_rel).name,
                pcb_root / f"{stem_rel}_temp.jpg",
                pcb_root / f"{stem_rel}_test.jpg",
                pcb_root / ann_rel,
            ))

    print(f"Evaluating {len(samples)} samples from {dataset_manifest}...")

    results = {}
    for iou_th in iou_thresholds:
        results[f"iou_{iou_th}"] = {
            "class_aware": {"tp": 0, "fp": 0, "fn": 0, "per_class": {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}},
            "class_agnostic": {"tp": 0, "fp": 0, "fn": 0},
        }

    latencies_ms = []
    bad_case_candidates = []

    for idx, (sample_id, temp_p, test_p, ann_p) in enumerate(samples):
        temp_img = cv2.imread(str(temp_p), cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(str(test_p), cv2.IMREAD_GRAYSCALE)
        gt_boxes = parse_ground_truth(ann_p)

        t0 = time.perf_counter()
        pred_boxes = model_fn(temp_img, test_img)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        for iou_th in iou_thresholds:
            tp, fp, fn, pc = match_detections(gt_boxes, pred_boxes, iou_thresh=iou_th, class_aware=True)
            res_ca = results[f"iou_{iou_th}"]["class_aware"]
            res_ca["tp"] += tp
            res_ca["fp"] += fp
            res_ca["fn"] += fn
            for c in CLASS_NAMES:
                res_ca["per_class"][c]["tp"] += pc[c]["tp"]
                res_ca["per_class"][c]["fp"] += pc[c]["fp"]
                res_ca["per_class"][c]["fn"] += pc[c]["fn"]

            tp_ag, fp_ag, fn_ag, _ = match_detections(gt_boxes, pred_boxes, iou_thresh=iou_th, class_aware=False)
            res_ag = results[f"iou_{iou_th}"]["class_agnostic"]
            res_ag["tp"] += tp_ag
            res_ag["fp"] += fp_ag
            res_ag["fn"] += fn_ag

            # Record error count for bad cases ranking (using IoU=0.33)
            if iou_th == 0.33:
                error_count = fp + fn
                if error_count > 0:
                    bad_case_candidates.append({
                        "sample_id": sample_id,
                        "error_count": error_count,
                        "fp": fp,
                        "fn": fn,
                        "temp_p": temp_p,
                        "test_p": test_p,
                        "gt_boxes": gt_boxes,
                        "pred_boxes": pred_boxes,
                    })

    # Summarize metrics
    summary = {
        "num_samples": len(samples),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p50": round(float(np.percentile(latencies_ms, 50)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "fps": round(1000.0 / float(np.mean(latencies_ms)), 1) if np.mean(latencies_ms) > 0 else 0,
        },
    }

    for iou_th in iou_thresholds:
        key = f"iou_{iou_th}"
        ca = results[key]["class_aware"]
        p, r, f1 = compute_prf1(ca["tp"], ca["fp"], ca["fn"])
        ag = results[key]["class_agnostic"]
        ag_p, ag_r, ag_f1 = compute_prf1(ag["tp"], ag["fp"], ag["fn"])

        per_class_metrics = {}
        for c in CLASS_NAMES:
            c_tp = ca["per_class"][c]["tp"]
            c_fp = ca["per_class"][c]["fp"]
            c_fn = ca["per_class"][c]["fn"]
            cp, cr, cf1 = compute_prf1(c_tp, c_fp, c_fn)
            per_class_metrics[c] = {"precision": cp, "recall": cr, "f1": cf1, "gt_count": c_tp + c_fn}

        summary[key] = {
            "class_aware": {"precision": p, "recall": r, "f1": f1, "tp": ca["tp"], "fp": ca["fp"], "fn": ca["fn"]},
            "class_agnostic_localization": {"precision": ag_p, "recall": ag_r, "f1": ag_f1},
            "per_class": per_class_metrics,
        }

    # Render Bad Cases if requested
    if visualize_bad_cases_dir and bad_case_candidates:
        visualize_bad_cases_dir.mkdir(parents=True, exist_ok=True)
        bad_case_candidates.sort(key=lambda x: x["error_count"], reverse=True)
        saved = 0
        for item in bad_case_candidates[:max_bad_cases]:
            sid = item["sample_id"]
            img_bgr = cv2.imread(str(item["test_p"]))
            if img_bgr is None:
                continue

            # Draw Ground Truth in GREEN
            for b in item["gt_boxes"]:
                cv2.rectangle(img_bgr, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"GT:{b.defect_class}", (b.x1, max(12, b.y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Draw Predictions in RED
            for b in item["pred_boxes"]:
                cv2.rectangle(img_bgr, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
                cv2.putText(img_bgr, f"{b.defect_class} {b.confidence:.2f}", (b.x1, min(635, b.y2 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            out_path = visualize_bad_cases_dir / f"bad_{sid}_fp{item['fp']}_fn{item['fn']}.jpg"
            cv2.imwrite(str(out_path), img_bgr)
            saved += 1
        print(f"Saved {saved} bad case visualizations to {visualize_bad_cases_dir}")

    return summary


def onnx_providers_for_device(device: str = "0") -> list[str]:
    """Map the CLI ``--device`` value to an ONNX Runtime provider priority list."""
    if str(device).strip().lower() == "cpu":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def make_onnx_method_a_detector(device: str = "0"):
    """Build the Method A inference fn.

    Returns ``(infer_fn, execution_provider)`` where ``execution_provider`` is the
    provider ONNX Runtime actually selected (not just the requested device).
    """
    from core.vision.onnx_detector import ONNXPCBDetector
    detector = ONNXPCBDetector(score_threshold=28.0, providers=onnx_providers_for_device(device))

    def _infer(temp_img: np.ndarray, test_img: np.ndarray) -> list[Box]:
        detections, _ = detector.detect(temp_img, test_img)
        return [Box(d.x1, d.y1, d.x2, d.y2, d.defect_class, d.confidence) for d in detections]

    return _infer, detector.execution_provider


def make_yolo_detector(weights_path: Path, mode: str = "b", conf: float = 0.25, device: str = "0"):
    """Factory for YOLOv8 model inference.

    mode:
      'b' -> single test image
      'c' -> 3-channel composite [template, test, absdiff]
    device:
      '0', 'cuda', 'cpu', etc.
    """
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    def _infer(temp_img: np.ndarray, test_img: np.ndarray) -> list[Box]:
        if mode == "b":
            # Test image converted to 3-channel BGR
            inp = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        elif mode == "c":
            diff = cv2.absdiff(temp_img, test_img)
            inp = cv2.merge([temp_img, test_img, diff])
        else:
            raise ValueError(f"Unknown mode {mode}")

        results = model.predict(source=inp, imgsz=640, conf=conf, device=device, verbose=False)
        boxes = []
        if results and len(results) > 0:
            r = results[0]
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
                boxes.append(Box(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), cls_name, confidence))
        return boxes

    return _infer


def get_system_metadata(device: str) -> dict[str, Any]:
    """Collect git commit, device details, and environment info for reproducibility."""
    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass

    device_name = "CPU"
    if device not in ["cpu", "CPU"]:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    else:
        # Try to get CPU model name
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        device_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            device_name = platform.processor() or "CPU"

    return {
        "git_commit": commit,
        "target_device": device,
        "device_name": device_name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepPCB Defect Detectors")
    parser.add_argument("--method", choices=["a", "b", "c", "all"], default="a", help="Method to evaluate")
    parser.add_argument("--device", type=str, default="0", help="Inference device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--weights-b", type=Path, default=None, help="Weights path for YOLO Exp B")
    parser.add_argument("--weights-c", type=Path, default=None, help="Weights path for YOLO Exp C")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--pcb-root", type=Path, default=Path("data/DeepPCB/PCBData"))
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--bad-cases-dir", type=Path, default=Path("docs/bad_cases"))
    args = parser.parse_args()

    test_manifest = args.pcb_root / "test.txt"
    all_results = {}
    metadata = get_system_metadata(args.device)

    if args.method in ["a", "all"]:
        print("\n" + "=" * 50)
        print("Evaluating Method A (Differential ONNX Detector Baseline)...")
        print("=" * 50)
        fn_a, ep_a = make_onnx_method_a_detector(args.device)
        print(f"Method A ONNX Runtime execution provider: {ep_a}")
        bad_dir = args.bad_cases_dir / "method_a"
        res_a = run_evaluation(fn_a, test_manifest, args.pcb_root, visualize_bad_cases_dir=bad_dir)
        res_a["metadata"] = {**metadata, "execution_provider": ep_a}
        all_results["Method_A_Differential_ONNX"] = res_a

    if args.method in ["b", "all"] and args.weights_b:
        print("\n" + "=" * 50)
        print(f"Evaluating Method B (YOLOv8n Single Image: {args.weights_b}) on device={args.device}...")
        print("=" * 50)
        fn_b = make_yolo_detector(args.weights_b, mode="b", conf=args.conf, device=args.device)
        bad_dir = args.bad_cases_dir / "method_b"
        res_b = run_evaluation(fn_b, test_manifest, args.pcb_root, visualize_bad_cases_dir=bad_dir)
        res_b["metadata"] = {**metadata, "yolo_device": args.device}
        all_results["Method_B_YOLOv8n_SingleImage"] = res_b

    if args.method in ["c", "all"] and args.weights_c:
        print("\n" + "=" * 50)
        print(f"Evaluating Method C (YOLOv8n 3-Channel Composite: {args.weights_c}) on device={args.device}...")
        print("=" * 50)
        fn_c = make_yolo_detector(args.weights_c, mode="c", conf=args.conf, device=args.device)
        bad_dir = args.bad_cases_dir / "method_c"
        res_c = run_evaluation(fn_c, test_manifest, args.pcb_root, visualize_bad_cases_dir=bad_dir)
        res_c["metadata"] = {**metadata, "yolo_device": args.device}
        all_results["Method_C_YOLOv8n_3Channel"] = res_c

    # Save summary json
    args.output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation summary to {args.output_json}")

    # Print markdown table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE (IoU=0.33 DeepPCB Benchmark Standard)")
    print(f"Device: {metadata['device_name']} ({metadata['target_device']}) | Commit: {metadata['git_commit'][:8]}")
    print("=" * 80)
    print(f"{'Method':<32} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Mean Latency':<12}")
    print("-" * 80)
    for name, res in all_results.items():
        ca = res["iou_0.33"]["class_aware"]
        lat = res["latency_ms"]["mean"]
        print(f"{name:<32} | {ca['precision']:<9.4f} | {ca['recall']:<9.4f} | {ca['f1']:<9.4f} | {lat:<6.2f} ms")


if __name__ == "__main__":
    main()
