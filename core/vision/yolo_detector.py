"""YOLOv8 3-channel ONNX Runtime defect detector for DeepPCB-AOI."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.agent.schemas import (
    BOARD_DIM_PX,
    BoardInspectionReport,
    BoundingBoxDetection,
    DefectClass,
)
from core.vision.diff_preprocess import ensure_grayscale_640
from core.vision.onnx_detector import PROVIDER_FALLBACK_ORDER, apply_nms

DEFAULT_YOLO_ONNX_PATH = (
    Path(__file__).resolve().parent / "models" / "deeppcb_yolov8n_3channel.onnx"
)

CLASS_MAP: dict[int, DefectClass] = {
    0: "Open",
    1: "Short",
    2: "Mousebite",
    3: "Spur",
    4: "Spurious Copper",
    5: "Pin-hole",
}


def apply_class_aware_nms(
    boxes: list[BoundingBoxDetection],
    iou_threshold: float = 0.70,
) -> list[BoundingBoxDetection]:
    """Apply NMS independently per defect class, matching YOLO standard evaluation."""
    if not boxes:
        return []
    kept: list[BoundingBoxDetection] = []
    by_class: dict[str, list[BoundingBoxDetection]] = {}
    for b in boxes:
        by_class.setdefault(b.defect_class, []).append(b)
    for cls_boxes in by_class.values():
        kept.extend(apply_nms(cls_boxes, iou_threshold=iou_threshold))
    return sorted(
        kept,
        key=lambda b: (b.confidence, b.area_px, -b.x1, -b.y1),
        reverse=True,
    )


class YOLOv8PCBDetector:
    """YOLOv8 3-channel ONNX defect detector ([template, test, absdiff])."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_ONNX_PATH,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        class_agnostic: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLOv8 ONNX model not found at {self.model_path}. "
                "Ensure trained weights have been exported."
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_agnostic = class_agnostic

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=PROVIDER_FALLBACK_ORDER,
            )

        active_providers = self.session.get_providers()
        self.execution_provider: str = (
            active_providers[0] if active_providers else "CPUExecutionProvider"
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warm up session once at initialization
        dummy = np.zeros((1, 3, BOARD_DIM_PX, BOARD_DIM_PX), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def detect(
        self,
        template_img: np.ndarray,
        test_img: np.ndarray,
    ) -> tuple[list[BoundingBoxDetection], float]:
        """Run Stage 1 YOLOv8 3-channel ONNX detection and NMS coordinate mapping.

        Returns:
            `(detections, latency_ms)`
        """
        t_start = time.perf_counter()

        template_gray = ensure_grayscale_640(template_img, name="template_img")
        test_gray = ensure_grayscale_640(test_img, name="test_img")

        # Fast-path: bypass ML model if buffers are identical
        if np.array_equal(template_gray, test_gray):
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            return [], latency_ms

        # Compute absdiff
        diff = cv2.absdiff(template_gray, test_gray)

        # Composite into 3 channels [diff, test, template] matching RGB tensor order
        # (cv2.imwrite saves BGR=[template, test, diff], which Ultralytics reads as RGB=[diff, test, template])
        composite = np.stack([diff, test_gray, template_gray], axis=0).astype(np.float32) / 255.0
        tensor_in = composite[np.newaxis, ...]  # Shape [1, 3, 640, 640]

        outputs = self.session.run([self.output_name], {self.input_name: tensor_in})
        # Output shape: [1, 10, 8400]
        preds = outputs[0][0]  # [10, 8400]

        boxes = preds[:4, :]       # [4, 8400] cx, cy, w, h
        scores = preds[4:10, :]     # [6, 8400] class confidences

        # Vectorized filtering
        max_scores = np.max(scores, axis=0)
        class_ids = np.argmax(scores, axis=0)
        valid_mask = max_scores >= self.conf_threshold

        candidates: list[BoundingBoxDetection] = []
        if np.any(valid_mask):
            valid_boxes = boxes[:, valid_mask]
            valid_scores = max_scores[valid_mask]
            valid_classes = class_ids[valid_mask]

            cx = valid_boxes[0]
            cy = valid_boxes[1]
            w = valid_boxes[2]
            h = valid_boxes[3]

            x1 = np.clip(np.round(cx - w / 2.0).astype(int), 0, BOARD_DIM_PX)
            y1 = np.clip(np.round(cy - h / 2.0).astype(int), 0, BOARD_DIM_PX)
            x2 = np.clip(np.round(cx + w / 2.0).astype(int), 0, BOARD_DIM_PX)
            y2 = np.clip(np.round(cy + h / 2.0).astype(int), 0, BOARD_DIM_PX)

            for i in range(len(valid_scores)):
                if x2[i] <= x1[i] or y2[i] <= y1[i]:
                    continue
                cls_id = int(valid_classes[i])
                candidates.append(
                    BoundingBoxDetection(
                        x1=int(x1[i]),
                        y1=int(y1[i]),
                        x2=int(x2[i]),
                        y2=int(y2[i]),
                        defect_class=CLASS_MAP[cls_id],
                        confidence=round(float(valid_scores[i]), 4),
                    )
                )

        if self.class_agnostic:
            filtered = apply_nms(candidates, iou_threshold=self.iou_threshold)
        else:
            filtered = apply_class_aware_nms(candidates, iou_threshold=self.iou_threshold)
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
        return filtered, latency_ms

    def inspect_fast_path(
        self,
        template_img: np.ndarray,
        test_img: np.ndarray,
    ) -> BoardInspectionReport:
        """Run detection and set ``fast_path_pass`` when there are no boxes."""
        detections, latency_ms = self.detect(template_img, test_img)
        is_fast_path = len(detections) == 0
        return BoardInspectionReport(
            stage1_latency_ms=latency_ms,
            detections=detections,
            fast_path_pass=is_fast_path,
            decisions=[],
        )

    def benchmark(self, num_runs: int = 50) -> dict[str, float | str | int]:
        """Benchmark Stage 1 YOLOv8 detector latency (ms/frame) and throughput (FPS)."""
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}.")

        rng = np.random.default_rng(seed=42)
        template_img = rng.integers(200, 255, size=(BOARD_DIM_PX, BOARD_DIM_PX), dtype=np.uint8)
        test_img = template_img.copy()
        test_img[120:152, 200:240] = 10
        test_img[400:428, 450:482] = 15

        for _ in range(3):
            self.detect(template_img, test_img)

        latencies: list[float] = []
        for _ in range(num_runs):
            _, lat_ms = self.detect(template_img, test_img)
            latencies.append(lat_ms)

        arr = np.asarray(latencies, dtype=np.float64)
        mean_ms = float(np.mean(arr))
        p50_ms = float(np.percentile(arr, 50))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        fps = 1000.0 / mean_ms if mean_ms > 0.0 else 0.0

        return {
            "num_runs": num_runs,
            "p50_ms": round(p50_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "mean_ms": round(mean_ms, 4),
            "fps": round(fps, 2),
            "execution_provider": self.execution_provider,
        }

