"""ONNX Runtime difference detector and non-maximum suppression.

Coordinates, confidence, and area are computed here. Identical images, or a score
map below ``score_threshold``, return no boxes.
"""

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

DEFAULT_ONNX_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "deeppcb_diff_detector.onnx"
)
PROVIDER_FALLBACK_ORDER: list[str] = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
_STRIDE: int = 4
_GRID_DIM: int = BOARD_DIM_PX // _STRIDE


# ONNX ModelProto field numbers (IR v8, opset 13). See the ONNX protobuf schema.
def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _field_varint(field_num: int, value: int) -> bytes:
    return _encode_varint((field_num << 3) | 0) + _encode_varint(value)


def _field_bytes(field_num: int, payload: bytes | str) -> bytes:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return _encode_varint((field_num << 3) | 2) + _encode_varint(len(raw)) + raw


def _value_info_proto(name: str, elem_type: int, dims: list[int]) -> bytes:
    shape_proto = b"".join(_field_bytes(1, _field_varint(1, d)) for d in dims)
    tensor_type = _field_varint(1, elem_type) + _field_bytes(2, shape_proto)
    type_proto = _field_bytes(1, tensor_type)
    return _field_bytes(1, name) + _field_bytes(2, type_proto)


def _tensor_proto_float32(name: str, dims: list[int], values: np.ndarray) -> bytes:
    raw_bytes = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    out = b"".join(_field_varint(1, d) for d in dims)
    out += _field_varint(2, 1)  # TensorProto.DataType.FLOAT = 1
    out += _field_bytes(8, name)
    out += _field_bytes(9, raw_bytes)
    return out


def _attr_ints(name: str, values: list[int]) -> bytes:
    out = _field_bytes(1, name) + _field_varint(20, 7)  # AttributeProto.INTS = 7
    for val in values:
        out += _field_varint(8, val)
    return out


def _node_proto(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    name: str,
    attrs: list[bytes] | None = None,
) -> bytes:
    out = b"".join(_field_bytes(1, inp) for inp in inputs)
    out += b"".join(_field_bytes(2, outp) for outp in outputs)
    out += _field_bytes(3, name)
    out += _field_bytes(4, op_type)
    for attr in attrs or []:
        out += _field_bytes(5, attr)
    return out


def build_differential_onnx_model(
    output_path: str | Path = DEFAULT_ONNX_MODEL_PATH,
) -> Path:
    """Programmatically build and save the lightweight Stage 1 differential ONNX detector.

    Graph topology (`ir_version=8`, `opset=13`):
      Inputs:
        - `template`: float32 `[1, 1, 640, 640]`
        - `test`:     float32 `[1, 1, 640, 640]`
      Nodes:
        1. `Sub(template, test) -> diff_signed`
        2. `Abs(diff_signed) -> diff_abs`
        3. `Conv(diff_abs, conv_w, kernel=4x4, stride=4x4) -> diff_score [1, 1, 160, 160]`
        4. `Conv(diff_signed, polarity_w, kernel=4x4, stride=4x4) -> polarity_map [1, 1, 160, 160]`
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    kernel = np.full((1, 1, _STRIDE, _STRIDE), 1.0 / float(_STRIDE * _STRIDE), dtype=np.float32)

    node_sub = _node_proto("Sub", ["template", "test"], ["diff_signed"], "sub_template_test")
    node_abs = _node_proto("Abs", ["diff_signed"], ["diff_abs"], "abs_diff")
    node_conv_score = _node_proto(
        "Conv",
        ["diff_abs", "conv_w"],
        ["diff_score"],
        "conv_stride4_score",
        [_attr_ints("kernel_shape", [_STRIDE, _STRIDE]), _attr_ints("strides", [_STRIDE, _STRIDE])],
    )
    node_conv_polarity = _node_proto(
        "Conv",
        ["diff_signed", "polarity_w"],
        ["polarity_map"],
        "conv_stride4_polarity",
        [_attr_ints("kernel_shape", [_STRIDE, _STRIDE]), _attr_ints("strides", [_STRIDE, _STRIDE])],
    )

    graph_proto = (
        _field_bytes(1, node_sub)
        + _field_bytes(1, node_abs)
        + _field_bytes(1, node_conv_score)
        + _field_bytes(1, node_conv_polarity)
        + _field_bytes(2, "deeppcb_stage1_diff_detector")
        + _field_bytes(5, _tensor_proto_float32("conv_w", [1, 1, _STRIDE, _STRIDE], kernel))
        + _field_bytes(5, _tensor_proto_float32("polarity_w", [1, 1, _STRIDE, _STRIDE], kernel))
        + _field_bytes(11, _value_info_proto("template", 1, [1, 1, BOARD_DIM_PX, BOARD_DIM_PX]))
        + _field_bytes(11, _value_info_proto("test", 1, [1, 1, BOARD_DIM_PX, BOARD_DIM_PX]))
        + _field_bytes(12, _value_info_proto("diff_score", 1, [1, 1, _GRID_DIM, _GRID_DIM]))
        + _field_bytes(12, _value_info_proto("polarity_map", 1, [1, 1, _GRID_DIM, _GRID_DIM]))
    )

    opset_import = _field_bytes(1, "") + _field_varint(2, 13)
    model_proto = (
        _field_varint(1, 8)
        + _field_bytes(2, "DeepPCB-AOI-Stage1")
        + _field_bytes(7, graph_proto)
        + _field_bytes(8, opset_import)
    )

    target_path.write_bytes(model_proto)
    return target_path


def compute_iou(box_a: BoundingBoxDetection, box_b: BoundingBoxDetection) -> float:
    """Compute exact Intersection-over-Union (IoU) between two `BoundingBoxDetection` boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    union_area = box_a.area_px + box_b.area_px - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area) / float(union_area)


def apply_nms(
    boxes: list[BoundingBoxDetection],
    iou_threshold: float = 0.45,
) -> list[BoundingBoxDetection]:
    """Filter overlapping `BoundingBoxDetection` candidates via Non-Maximum Suppression."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}.")
    if not boxes:
        return []

    # Sort deterministically by confidence descending, then area descending
    sorted_boxes = sorted(
        boxes,
        key=lambda b: (b.confidence, b.area_px, -b.x1, -b.y1),
        reverse=True,
    )

    kept: list[BoundingBoxDetection] = []
    for candidate in sorted_boxes:
        if all(compute_iou(candidate, survivor) < iou_threshold for survivor in kept):
            kept.append(candidate)

    return kept


class ONNXPCBDetector:
    """PCB defect detector backed by the checked-in difference graph."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ONNX_MODEL_PATH,
        score_threshold: float = 28.0,
        iou_threshold: float = 0.45,
        min_Box_px: int = 8,
        providers: list[str] | None = None,
    ) -> None:
        """Load (or synthesize) the difference graph and start an ONNX Runtime session.

        ``providers`` overrides the execution-provider priority list. ``None`` keeps
        the default ``PROVIDER_FALLBACK_ORDER`` (CUDA first, then CPU).
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            build_differential_onnx_model(self.model_path)

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.min_box_px = min_Box_px

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=list(providers) if providers else PROVIDER_FALLBACK_ORDER,
            )

        active_providers = self.session.get_providers()
        self.execution_provider: str = (
            active_providers[0] if active_providers else "CPUExecutionProvider"
        )

        # Warm up full Stage 1 pipeline (ONNX session + OpenCV connected components + NMS)
        dummy_tpl = np.full((BOARD_DIM_PX, BOARD_DIM_PX), 255, dtype=np.uint8)
        dummy_tst = dummy_tpl.copy()
        dummy_tst[64:96, 64:96] = 0
        self.detect(dummy_tpl, dummy_tst)

    @staticmethod
    def _classify_defect_morphology(
        mean_polarity: float,
        width_px: int,
        height_px: int,
        fill_ratio: float,
    ) -> DefectClass:
        """Deterministically assign canonical `DefectClass` from differential polarity & geometry.

        In DeepPCB binary/grayscale images (0 = copper trace, 255 = substrate background):
        - `template - test < 0` (test is brighter -> missing copper):
          `Open`, `Mousebite`, or `Pin-hole`.
        - `template - test > 0` (test is darker -> excess copper):
          `Short`, `Spur`, or `Spurious Copper`.
        """
        aspect = max(width_px, height_px) / max(1.0, float(min(width_px, height_px)))
        area = width_px * height_px

        if mean_polarity < 0.0:
            if area <= 256 and aspect < 1.6:
                return "Pin-hole"
            if fill_ratio < 0.55 or aspect >= 1.8:
                return "Open"
            return "Mousebite"
        else:
            if area <= 256 and fill_ratio >= 0.65:
                return "Spurious Copper"
            if aspect >= 1.75 or area >= 550:
                return "Short"
            return "Spur"

    def detect(
        self,
        template_img: np.ndarray,
        test_img: np.ndarray,
    ) -> tuple[list[BoundingBoxDetection], float]:
        """Run Stage 1 ONNX differential detection and NMS coordinate mapping.

        Returns:
            `(detections, latency_ms)` where `detections` is a list of validated
            `BoundingBoxDetection` items and `latency_ms` is Stage 1 execution time.
        """
        t_start = time.perf_counter()

        template_gray = ensure_grayscale_640(template_img, name="template_img")
        test_gray = ensure_grayscale_640(test_img, name="test_img")

        # Fast-path memory comparison before tensor conversion if buffers are identical
        if np.array_equal(template_gray, test_gray):
            t_tensor = template_gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]
            self.session.run(None, {"template": t_tensor, "test": t_tensor})
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            return [], latency_ms

        t_tensor = template_gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        x_tensor = test_gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]

        diff_score_out, polarity_out = self.session.run(
            None, {"template": t_tensor, "test": x_tensor}
        )
        score_map = diff_score_out[0, 0]
        polarity_map = polarity_out[0, 0]

        max_score = float(score_map.max())
        if max_score < self.score_threshold:
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            return [], latency_ms

        mask = (score_map >= self.score_threshold).astype(np.uint8) * 255
        # Dilate by 1 grid cell to merge closely adjacent defect fragments
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        candidates: list[BoundingBoxDetection] = []
        for label_idx in range(1, num_labels):
            gx = int(stats[label_idx, cv2.CC_STAT_LEFT])
            gy = int(stats[label_idx, cv2.CC_STAT_TOP])
            gw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            gh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            g_area = int(stats[label_idx, cv2.CC_STAT_AREA])

            if g_area < 2:
                continue

            x1 = max(0, gx * _STRIDE)
            y1 = max(0, gy * _STRIDE)
            x2 = min(BOARD_DIM_PX, (gx + gw) * _STRIDE)
            y2 = min(BOARD_DIM_PX, (gy + gh) * _STRIDE)

            width_px = max(x2 - x1, self.min_box_px)
            height_px = max(y2 - y1, self.min_box_px)
            x2 = min(BOARD_DIM_PX, x1 + width_px)
            y2 = min(BOARD_DIM_PX, y1 + height_px)
            x1 = max(0, x2 - width_px)
            y1 = max(0, y2 - height_px)

            if x2 <= x1 or y2 <= y1:
                continue

            roi_scores = score_map[gy : gy + gh, gx : gx + gw]
            roi_polarity = polarity_map[gy : gy + gh, gx : gx + gw]
            component_mask = labels[gy : gy + gh, gx : gx + gw] == label_idx

            peak_val = float(roi_scores[component_mask].max()) if np.any(component_mask) else max_score
            mean_pol = float(roi_polarity[component_mask].mean()) if np.any(component_mask) else 0.0
            fill_ratio = float(g_area) / float(max(1, gw * gh))

            confidence = round(min(0.99, max(0.55, peak_val / 255.0)), 4)
            defect_class = self._classify_defect_morphology(
                mean_polarity=mean_pol,
                width_px=x2 - x1,
                height_px=y2 - y1,
                fill_ratio=fill_ratio,
            )

            candidates.append(
                BoundingBoxDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    defect_class=defect_class,
                    confidence=confidence,
                )
            )

        filtered = apply_nms(candidates, iou_threshold=self.iou_threshold)
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
        """Benchmark Stage 1 ONNX detector latency (ms/frame) and throughput (FPS)."""
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}.")

        rng = np.random.default_rng(seed=42)
        template_img = rng.integers(200, 255, size=(BOARD_DIM_PX, BOARD_DIM_PX), dtype=np.uint8)
        test_img = template_img.copy()
        # Inject two synthetic copper defects so full post-processing + NMS is exercised
        test_img[120:152, 200:240] = 10
        test_img[400:428, 450:482] = 15

        # Warm-up passes
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
