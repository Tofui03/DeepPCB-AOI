"""Unit tests for strict Pydantic v2 contracts in `core/agent/schemas.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure project root is importable when invoking `.venv/bin/pytest` directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.schemas import (  # noqa: E402
    BOARD_DIM_PX,
    BOARD_TOTAL_AREA_PX,
    BatchTrendResult,
    BoardInspectionReport,
    BoundingBoxDetection,
    InspectionDecision,
    IPCComplianceResult,
)


class TestBoundingBoxDetection:
    """Tests for Stage 1 ONNX bounding box geometry, bounds, and area calculations."""

    @pytest.mark.parametrize(
        "defect_class",
        [
            "Open",
            "Short",
            "Mousebite",
            "Spur",
            "Spurious Copper",
            "Pin-hole",
        ],
    )
    def test_valid_canonical_defect_classes(self, defect_class: str) -> None:
        """All 6 canonical DeepPCB defect classes should validate cleanly."""
        box = BoundingBoxDetection(
            x1=100,
            y1=120,
            x2=164,
            y2=184,
            defect_class=defect_class,  # type: ignore[arg-type]
            confidence=0.95,
        )
        assert box.defect_class == defect_class
        assert box.area_px == 64 * 64 == 4096
        assert box.relative_area_pct == pytest.approx(1.0)

    def test_full_board_boundary_area_math(self) -> None:
        """A box spanning the full 640x640 image should equal 409,600 px and 100.0% area."""
        box = BoundingBoxDetection(
            x1=0,
            y1=0,
            x2=BOARD_DIM_PX,
            y2=BOARD_DIM_PX,
            defect_class="Short",
            confidence=1.0,
        )
        assert box.area_px == BOARD_TOTAL_AREA_PX == 409_600
        assert box.relative_area_pct == 100.0

        dumped = box.model_dump()
        assert dumped["area_px"] == 409_600
        assert dumped["relative_area_pct"] == 100.0

    @pytest.mark.parametrize(
        ("x1", "y1", "x2", "y2"),
        [
            (-1, 10, 50, 50),    # Negative x1
            (10, -5, 50, 50),    # Negative y1
            (10, 10, 641, 50),   # x2 exceeds 640
            (10, 10, 50, 700),   # y2 exceeds 640
            (100, 10, 100, 50),  # Zero width (x2 == x1)
            (120, 10, 80, 50),   # Inverted width (x2 < x1)
            (10, 200, 50, 200),  # Zero height (y2 == y1)
            (10, 250, 50, 150),  # Inverted height (y2 < y1)
        ],
    )
    def test_invalid_bounding_box_coordinates_rejected(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        """Out-of-bounds or inverted coordinates must raise ValidationError."""
        with pytest.raises(ValidationError):
            BoundingBoxDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                defect_class="Open",
                confidence=0.85,
            )

    @pytest.mark.parametrize("bad_class", ["Scratch", "open", "SolderBridge", ""])
    def test_invalid_defect_class_rejected(self, bad_class: str) -> None:
        """Non-canonical defect labels must be rejected by the Literal contract."""
        with pytest.raises(ValidationError):
            BoundingBoxDetection(
                x1=10,
                y1=10,
                x2=30,
                y2=30,
                defect_class=bad_class,  # type: ignore[arg-type]
                confidence=0.9,
            )

    @pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, 2.5])
    def test_invalid_confidence_rejected(self, bad_confidence: float) -> None:
        """Confidence scores outside [0.0, 1.0] must raise ValidationError."""
        with pytest.raises(ValidationError):
            BoundingBoxDetection(
                x1=10,
                y1=10,
                x2=30,
                y2=30,
                defect_class="Pin-hole",
                confidence=bad_confidence,
            )


class TestStage2ToolContracts:
    """Tests for Stage 2 deterministic tool output contracts."""

    def test_valid_ipc_compliance_result(self) -> None:
        """Valid IPC-A-600 Class 2/3 compliance result should validate cleanly."""
        result = IPCComplianceResult(
            defect_class="Mousebite",
            ipc_class="Class 3",
            violates_ipc=True,
            ipc_clause="IPC-A-600 Section 2.2.2 (Conductor Width Reduction > 20%)",
            max_allowed_px=120,
            measured_px=210,
        )
        assert result.violates_ipc is True
        assert result.ipc_class == "Class 3"

    def test_invalid_ipc_class_rejected(self) -> None:
        """Unsupported IPC classes (e.g., Class 1) must be rejected."""
        with pytest.raises(ValidationError):
            IPCComplianceResult(
                defect_class="Mousebite",
                ipc_class="Class 1",  # type: ignore[arg-type]
                violates_ipc=False,
                ipc_clause="IPC-A-600 Section 2.2.2",
                max_allowed_px=200,
                measured_px=100,
            )

    def test_valid_batch_trend_result(self) -> None:
        """Valid SQLite batch telemetry result should validate cleanly."""
        trend = BatchTrendResult(
            batch_id="BATCH-2026-10A",
            defect_class="Short",
            occurrence_count=3,
            threshold_exceeded=True,
        )
        assert trend.occurrence_count == 3
        assert trend.threshold_exceeded is True

    def test_negative_batch_occurrence_rejected(self) -> None:
        """Negative occurrence counts must raise ValidationError."""
        with pytest.raises(ValidationError):
            BatchTrendResult(
                batch_id="BATCH-2026-10A",
                defect_class="Short",
                occurrence_count=-1,
                threshold_exceeded=False,
            )


class TestInspectionDecisionAndReport:
    """Tests for Stage 3/4 InspectionDecision and end-to-end BoardInspectionReport."""

    def test_valid_inspection_decision_defaults(self) -> None:
        """InspectionDecision should default guardrail_overridden=False and override_reason=None."""
        decision = InspectionDecision(
            disposition="REWORK",
            severity="MEDIUM",
            ipc_standard_clause="IPC-A-600 Section 2.3.1",
            electrical_failure_risk="Parasitic capacitive coupling between adjacent signal traces.",
            root_cause_hypothesis="Incomplete photo-resist stripping prior to acid etching.",
            recommended_corrective_action="Remove spurious copper speck via micro-abrasion and inspect etching bath nozzle pressure.",
            halt_production_line=False,
        )
        assert decision.guardrail_overridden is False
        assert decision.override_reason is None

    def test_invalid_disposition_and_severity_rejected(self) -> None:
        """Non-canonical disposition or severity values must be rejected."""
        with pytest.raises(ValidationError):
            InspectionDecision(
                disposition="IGNORE",  # type: ignore[arg-type]
                severity="CRITICAL",
                ipc_standard_clause="IPC-A-600 Section 2.1",
                electrical_failure_risk="Power rail dead short.",
                root_cause_hypothesis="Under-etching.",
                recommended_corrective_action="Scrap board.",
                halt_production_line=True,
            )

    def test_fast_path_board_inspection_report(self) -> None:
        """A report with no detections and fast_path_pass set should validate."""
        report = BoardInspectionReport(
            stage1_latency_ms=4.8,
            detections=[],
            fast_path_pass=True,
            decisions=[],
        )
        assert report.fast_path_pass is True
        assert report.stage1_latency_ms == 4.8
        assert len(report.detections) == 0
        assert len(report.decisions) == 0

    def test_full_board_inspection_report_with_guardrail_override(self) -> None:
        """Full report combining Stage 1 detection and Stage 4 guardrail-overridden decision."""
        detection = BoundingBoxDetection(
            x1=200,
            y1=300,
            x2=240,
            y2=330,
            defect_class="Open",
            confidence=0.98,
        )
        decision = InspectionDecision(
            disposition="SCRAP",
            severity="CRITICAL",
            ipc_standard_clause="IPC-A-600 Section 2.2.1 (Conductor Continuity)",
            electrical_failure_risk="Complete open circuit interrupting high-speed clock trace.",
            root_cause_hypothesis="Mechanical scratch on photoresist film before copper etching.",
            recommended_corrective_action="Scrap PCB and inspect cleanroom film handling rollers.",
            halt_production_line=True,
            guardrail_overridden=True,
            override_reason="Rule 5 Safety Override: 'Open' defect cannot receive PASS disposition.",
        )
        report = BoardInspectionReport(
            stage1_latency_ms=8.3,
            detections=[detection],
            fast_path_pass=False,
            decisions=[decision],
        )
        assert report.fast_path_pass is False
        assert report.detections[0].area_px == 1200
        assert report.decisions[0].guardrail_overridden is True
        assert report.decisions[0].disposition == "SCRAP"
