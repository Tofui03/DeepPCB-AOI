"""Unit and integration tests for DeepPCB data loading, differential preprocessing, and Stage 1 ONNX vision."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.schemas import BOARD_DIM_PX, BoundingBoxDetection  # noqa: E402
from core.data.deeppcb_loader import (  # noqa: E402
    DEEPPCB_CLASS_MAP,
    DEFAULT_DEEPPCB_ROOT,
    DeepPCBDataset,
    parse_annotation_file,
)
from core.vision.diff_preprocess import (  # noqa: E402
    compute_differential_map,
    crop_defect_roi,
)
from core.vision.onnx_detector import (  # noqa: E402
    DEFAULT_ONNX_MODEL_PATH,
    ONNXPCBDetector,
    apply_nms,
    compute_iou,
)


class TestDeepPCBLoader:
    """Tests for DeepPCB annotation parsing, split loading, and zero-leakage auditing."""

    def test_class_map_matches_six_canonical_classes(self) -> None:
        """DEEPPCB_CLASS_MAP must map IDs 1..6 to the 6 canonical defect classes."""
        assert DEEPPCB_CLASS_MAP == {
            1: "Open",
            2: "Short",
            3: "Mousebite",
            4: "Spur",
            5: "Spurious Copper",
            6: "Pin-hole",
        }

    def test_parse_real_deeppcb_annotation_file(self) -> None:
        """Parsing a real DeepPCB `.txt` file should return ground-truth boxes with confidence=1.0."""
        sample_txt = DEFAULT_DEEPPCB_ROOT / "group20085" / "20085_not" / "20085000.txt"
        boxes = parse_annotation_file(sample_txt)
        assert len(boxes) == 6
        assert all(isinstance(b, BoundingBoxDetection) for b in boxes)
        assert all(b.confidence == 1.0 for b in boxes)
        assert boxes[0].x1 == 409
        assert boxes[0].y1 == 394
        assert boxes[0].x2 == 435
        assert boxes[0].y2 == 422
        assert boxes[0].defect_class == "Mousebite"

    def test_parse_annotation_invalid_class_id_raises(self, tmp_path: Path) -> None:
        """An annotation line with an unknown class ID outside 1..6 must raise ValueError."""
        bad_txt = tmp_path / "bad.txt"
        bad_txt.write_text("10 20 40 50 9\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown DeepPCB class_id=9"):
            parse_annotation_file(bad_txt)

    def test_dataset_splits_and_zero_leakage_audit(self) -> None:
        """DeepPCBDataset should load all 1,500 pairs and verify zero trainval/test leakage."""
        dataset = DeepPCBDataset(DEFAULT_DEEPPCB_ROOT)
        assert len(dataset.trainval_samples) == 1000
        assert len(dataset.test_samples) == 500

        audit = dataset.verify_zero_leakage()
        assert audit["zero_leakage"] is True
        assert audit["overlap_count"] == 0
        assert audit["trainval_count"] == 1000
        assert audit["test_count"] == 500
        assert audit["total_count"] == 1500
        assert isinstance(audit["trainval_sha256"], str) and len(audit["trainval_sha256"]) == 64
        assert isinstance(audit["test_sha256"], str) and len(audit["test_sha256"]) == 64
        assert audit["trainval_sha256"] != audit["test_sha256"]

        # Verify loading an actual image pair and its annotations
        template_img, test_img, gt_boxes = dataset.load_sample(0, split="test")
        assert template_img.shape == (BOARD_DIM_PX, BOARD_DIM_PX)
        assert test_img.shape == (BOARD_DIM_PX, BOARD_DIM_PX)
        assert len(gt_boxes) > 0


class TestDiffPreprocess:
    """Tests for 640x640 differential image preprocessing and clamped ROI cropping."""

    def test_identical_images_produce_zero_differential_map(self) -> None:
        """Identical template and test images must yield an all-zero differential map."""
        img = np.full((BOARD_DIM_PX, BOARD_DIM_PX), 220, dtype=np.uint8)
        diff = compute_differential_map(img, img.copy())
        assert diff.shape == (BOARD_DIM_PX, BOARD_DIM_PX)
        assert diff.dtype == np.uint8
        assert int(diff.max()) == 0

    def test_bgr_and_grayscale_defect_normalization(self) -> None:
        """BGR 640x640 inputs with a localized defect should normalize peak difference to 255."""
        template_bgr = np.full((BOARD_DIM_PX, BOARD_DIM_PX, 3), 240, dtype=np.uint8)
        test_bgr = template_bgr.copy()
        test_bgr[200:230, 300:340] = 20

        diff = compute_differential_map(template_bgr, test_bgr)
        assert diff.shape == (BOARD_DIM_PX, BOARD_DIM_PX)
        assert diff.dtype == np.uint8
        assert int(diff[215, 320]) == 255
        assert int(diff[10, 10]) == 0

    def test_invalid_dimensions_rejected(self) -> None:
        """Images not matching 640x640 spatial dimensions must raise ValueError."""
        bad_img = np.zeros((512, 512), dtype=np.uint8)
        good_img = np.zeros((BOARD_DIM_PX, BOARD_DIM_PX), dtype=np.uint8)
        with pytest.raises(ValueError, match="spatial dimensions must be"):
            compute_differential_map(bad_img, good_img)

    def test_crop_defect_roi_clamped_at_borders(self) -> None:
        """ROI cropping near (0, 0) and (640, 640) borders must clamp safely without out-of-bounds."""
        test_img = np.zeros((BOARD_DIM_PX, BOARD_DIM_PX, 3), dtype=np.uint8)

        # Top-left corner box (x1=4, y1=6, x2=24, y2=26) with padding=32 -> [0:58, 0:56]
        tl_box = BoundingBoxDetection(
            x1=4, y1=6, x2=24, y2=26, defect_class="Spur", confidence=0.92
        )
        tl_crop = crop_defect_roi(test_img, tl_box, padding=32)
        assert tl_crop.shape == (58, 56, 3)

        # Bottom-right corner box (x1=620, y1=615, x2=638, y2=639) with padding=32 -> [583:640, 588:640]
        br_box = BoundingBoxDetection(
            x1=620, y1=615, x2=638, y2=639, defect_class="Open", confidence=0.95
        )
        br_crop = crop_defect_roi(test_img, br_box, padding=32)
        assert br_crop.shape == (57, 52, 3)


class TestONNXDetector:
    """Tests for ONNX Runtime session execution, NMS filtering, Fast-Path bypass, and <15ms SLA."""

    def test_nms_filters_overlapping_boxes(self) -> None:
        """apply_nms should suppress lower-confidence overlapping boxes above iou_threshold."""
        high_conf = BoundingBoxDetection(
            x1=100, y1=100, x2=160, y2=160, defect_class="Short", confidence=0.96
        )
        overlapping_low_conf = BoundingBoxDetection(
            x1=104, y1=104, x2=162, y2=162, defect_class="Short", confidence=0.72
        )
        distinct_box = BoundingBoxDetection(
            x1=400, y1=400, x2=450, y2=450, defect_class="Open", confidence=0.88
        )

        assert compute_iou(high_conf, overlapping_low_conf) > 0.45
        assert compute_iou(high_conf, distinct_box) == 0.0

        kept = apply_nms([overlapping_low_conf, high_conf, distinct_box], iou_threshold=0.45)
        assert len(kept) == 2
        assert kept[0] == high_conf
        assert kept[1] == distinct_box

    def test_onnx_detector_real_deeppcb_and_latency_sla(self) -> None:
        """ONNXPCBDetector should load the .onnx model, detect defects on real DeepPCB images, and run in <15ms."""
        detector = ONNXPCBDetector()
        assert DEFAULT_ONNX_MODEL_PATH.is_file()
        assert detector.execution_provider in ("CUDAExecutionProvider", "CPUExecutionProvider")

        dataset = DeepPCBDataset(DEFAULT_DEEPPCB_ROOT)
        template_img, test_img, _ = dataset.load_sample(0, split="test")

        detections, latency_ms = detector.detect(template_img, test_img)
        assert len(detections) > 0
        assert all(isinstance(d, BoundingBoxDetection) for d in detections)
        assert latency_ms < 15.0, f"Stage 1 latency {latency_ms:.2f}ms exceeded 15ms SLA"

    def test_rule2_zero_token_fast_path_bypass_on_identical_board(self) -> None:
        """Identical template and test images return no boxes and fast_path_pass in under 15 ms."""
        detector = ONNXPCBDetector()
        dataset = DeepPCBDataset(DEFAULT_DEEPPCB_ROOT)
        template_img, _, _ = dataset.load_sample(0, split="test")

        report = detector.inspect_fast_path(template_img, template_img.copy())
        assert report.fast_path_pass is True
        assert report.detections == []
        assert report.decisions == []
        assert report.stage1_latency_ms < 15.0

    def test_benchmark_suite_metrics_and_sla(self) -> None:
        """benchmark(num_runs=50) should return latency percentiles (<15ms) and FPS metrics."""
        detector = ONNXPCBDetector()
        metrics = detector.benchmark(num_runs=50)

        for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps", "execution_provider"):
            assert key in metrics

        assert float(metrics["mean_ms"]) < 15.0
        assert float(metrics["p95_ms"]) < 15.0
        assert float(metrics["fps"]) > 66.0
