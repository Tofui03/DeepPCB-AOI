"""Adversarial evaluation suite for Stage 3 VLM Triage, Stage 4 Safety Guardrails, and the 4-Stage State Graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.guardrails import apply_safety_guardrails  # noqa: E402
from core.agent.schemas import (  # noqa: E402
    BOARD_DIM_PX,
    BatchTrendResult,
    BoundingBoxDetection,
    InspectionDecision,
    IPCComplianceResult,
)
from core.agent.state_graph import (  # noqa: E402
    MAX_VLM_RETRIES,
    PCBAOIAgentGraph,
    build_deterministic_fallback_decision,
)
from core.tools.batch_db import BatchTelemetryStore  # noqa: E402
from core.vision.onnx_detector import ONNXPCBDetector  # noqa: E402


def _make_single_short_board_pair() -> tuple[np.ndarray, np.ndarray]:
    """Create a 640x640 template/test pair with a single synthetic copper Short bridge."""
    template_img = np.full((BOARD_DIM_PX, BOARD_DIM_PX), 240, dtype=np.uint8)
    test_img = template_img.copy()
    # Dark copper bridge (excess copper: template - test > 0, elongated -> Short)
    test_img[200:220, 260:320] = 10
    return template_img, test_img


class TestSafetyGuardrailsUnit:
    """Direct unit tests for Stage 4 Override A, Override B, and Override C in `guardrails.py`."""

    def test_override_a_short_or_open_never_passes(self) -> None:
        """Override A: Any Open or Short returned as PASS/LOW must be forced to SCRAP/CRITICAL."""
        bbox = BoundingBoxDetection(
            x1=260, y1=200, x2=320, y2=220, defect_class="Short", confidence=0.96
        )
        ipc = IPCComplianceResult(
            defect_class="Short",
            ipc_class="Class 2",
            violates_ipc=True,
            ipc_clause="IPC-A-600 Section 2.2.2 (Short / Bridging)",
            max_allowed_px=0,
            measured_px=bbox.area_px,
        )
        batch = BatchTrendResult(
            batch_id="BATCH-01",
            defect_class="Short",
            occurrence_count=1,
            threshold_exceeded=False,
        )
        rogue_decision = InspectionDecision(
            disposition="PASS",
            severity="LOW",
            ipc_standard_clause="IPC-A-600 Section 2.2.2",
            electrical_failure_risk="Minimal risk claimed by hallucinating VLM.",
            root_cause_hypothesis="Minor dust speck.",
            recommended_corrective_action="No action needed.",
            halt_production_line=False,
        )

        safe = apply_safety_guardrails(rogue_decision, bbox, ipc, batch)
        assert safe.disposition == "SCRAP"
        assert safe.severity == "CRITICAL"
        assert safe.guardrail_overridden is True
        assert safe.override_reason is not None
        assert "Override A" in safe.override_reason

    def test_override_b_ipc_violating_mousebite_forced_to_rework_high(self) -> None:
        """Override B: Non-Open/Short defect violating IPC cannot receive PASS or LOW severity."""
        bbox = BoundingBoxDetection(
            x1=100, y1=100, x2=120, y2=120, defect_class="Mousebite", confidence=0.90
        )
        ipc = IPCComplianceResult(
            defect_class="Mousebite",
            ipc_class="Class 2",
            violates_ipc=True,
            ipc_clause="IPC-A-600 Section 2.2.3 (Mousebite / Edge Nick)",
            max_allowed_px=250,
            measured_px=400,
        )
        batch = BatchTrendResult(
            batch_id="BATCH-01",
            defect_class="Mousebite",
            occurrence_count=1,
            threshold_exceeded=False,
        )
        rogue_decision = InspectionDecision(
            disposition="PASS",
            severity="LOW",
            ipc_standard_clause="IPC-A-600 Section 2.2.3",
            electrical_failure_risk="Minor cosmetic edge nick.",
            root_cause_hypothesis="Slight over-etching.",
            recommended_corrective_action="None.",
            halt_production_line=False,
        )

        safe = apply_safety_guardrails(rogue_decision, bbox, ipc, batch)
        assert safe.disposition == "REWORK"
        assert safe.severity == "HIGH"
        assert safe.guardrail_overridden is True
        assert safe.override_reason is not None
        assert "Override B" in safe.override_reason


class TestAdversarialAgentGraphEvals:
    """End-to-end adversarial evaluations for the 4-stage `PCBAOIAgentGraph`."""

    def test_eval1_fast_path_bypass_zero_tool_and_vlm_calls(self) -> None:
        """Eval 1 (Rule 2): Identical boards must exit at Stage 1 with 0 tool/VLM invocations."""
        vlm_call_count = 0

        def spy_vlm(context: dict[str, Any], roi: np.ndarray) -> dict[str, Any]:
            nonlocal vlm_call_count
            vlm_call_count += 1
            raise AssertionError("VLM must never be called on Fast-Path boards!")

        store = BatchTelemetryStore(":memory:")
        graph = PCBAOIAgentGraph(
            detector=ONNXPCBDetector(),
            batch_store=store,
            vlm_callable=spy_vlm,
        )

        clean_board = np.full((BOARD_DIM_PX, BOARD_DIM_PX), 235, dtype=np.uint8)
        report, trace = graph.run_inspection_with_trace(
            clean_board, clean_board.copy(), batch_id="BATCH-FAST"
        )

        assert report.fast_path_pass is True
        assert report.detections == []
        assert report.decisions == []
        assert report.stage1_latency_ms < 15.0
        assert vlm_call_count == 0
        assert store.get_batch_summary("BATCH-FAST") == []
        assert any(step.get("fast_path_pass") is True for step in trace)
        store.close()

    def test_eval2_adversarial_hallucination_overridden_by_stage4(self) -> None:
        """Eval 2 (Rule 5): Rogue VLM returning PASS/LOW on a Short defect is overridden to SCRAP/CRITICAL."""
        template_img, test_img = _make_single_short_board_pair()

        def rogue_vlm(context: dict[str, Any], roi: np.ndarray) -> str:
            return json.dumps(
                {
                    "disposition": "PASS",
                    "severity": "LOW",
                    "ipc_standard_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
                    "electrical_failure_risk": "Hallucinated safe board.",
                    "root_cause_hypothesis": "Lighting reflection artifact.",
                    "recommended_corrective_action": "Ship board as-is.",
                    "halt_production_line": False,
                }
            )

        store = BatchTelemetryStore(":memory:")
        graph = PCBAOIAgentGraph(
            detector=ONNXPCBDetector(),
            batch_store=store,
            vlm_callable=rogue_vlm,
        )

        report = graph.run_inspection(template_img, test_img, batch_id="BATCH-ADV")
        assert report.fast_path_pass is False
        assert len(report.detections) == 1
        assert report.detections[0].defect_class == "Short"

        decision = report.decisions[0]
        assert decision.disposition == "SCRAP"
        assert decision.severity == "CRITICAL"
        assert decision.guardrail_overridden is True
        assert decision.override_reason is not None
        store.close()

    def test_eval3_flaky_vlm_retry_recovery_on_attempt_2(self) -> None:
        """Eval 3 (Rule 4): Flaky VLM returning invalid JSON on attempt 1 recovers on attempt 2 without fallback."""
        template_img, test_img = _make_single_short_board_pair()
        attempts = 0

        def flaky_vlm(context: dict[str, Any], roi: np.ndarray) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return "{broken_json: true, disposition: 'INVALID'}"
            return json.dumps(
                {
                    "disposition": "SCRAP",
                    "severity": "CRITICAL",
                    "ipc_standard_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
                    "electrical_failure_risk": "Direct power-to-ground short across adjacent copper traces.",
                    "root_cause_hypothesis": "Incomplete acid etching between high-density traces.",
                    "recommended_corrective_action": "Scrap PCB and inspect etchant nozzle pressure.",
                    "halt_production_line": False,
                }
            )

        store = BatchTelemetryStore(":memory:")
        graph = PCBAOIAgentGraph(
            detector=ONNXPCBDetector(),
            batch_store=store,
            vlm_callable=flaky_vlm,
        )

        report, trace = graph.run_inspection_with_trace(
            template_img, test_img, batch_id="BATCH-RETRY"
        )
        assert attempts == 2
        stage3_steps = [s for s in trace if s["stage"].startswith("Stage 3")]
        assert len(stage3_steps) == 1
        assert stage3_steps[0]["vlm_attempts"] == 2
        assert stage3_steps[0]["used_fallback"] is False
        assert report.decisions[0].disposition == "SCRAP"
        store.close()

    def test_eval4_exhausted_retries_caps_at_max_retries_and_uses_fallback(self) -> None:
        """Eval 4 (Rule 4): Broken VLM is capped at 1 + MAX_VLM_RETRIES (3 total calls) and falls back cleanly."""
        template_img, test_img = _make_single_short_board_pair()
        call_count = 0

        def broken_vlm(context: dict[str, Any], roi: np.ndarray) -> str:
            nonlocal call_count
            call_count += 1
            return "NOT_VALID_JSON_AT_ALL"

        store = BatchTelemetryStore(":memory:")
        graph = PCBAOIAgentGraph(
            detector=ONNXPCBDetector(),
            batch_store=store,
            vlm_callable=broken_vlm,
        )

        report, trace = graph.run_inspection_with_trace(
            template_img, test_img, batch_id="BATCH-OFFLINE"
        )
        assert MAX_VLM_RETRIES == 2
        assert call_count == 1 + MAX_VLM_RETRIES == 3

        stage3_steps = [s for s in trace if s["stage"].startswith("Stage 3")]
        assert stage3_steps[0]["used_fallback"] is True
        assert stage3_steps[0]["vlm_attempts"] == 3
        assert report.decisions[0].disposition == "SCRAP"
        assert report.decisions[0].severity == "CRITICAL"
        store.close()

    def test_eval5_batch_drift_line_halt_on_third_consecutive_board(self) -> None:
        """Eval 5 (Rule 5): 3 consecutive boards with the same defect class in one batch force halt_production_line=True."""
        template_img, test_img = _make_single_short_board_pair()

        # Use a mock VLM that always stubbornly outputs halt_production_line=False
        def stubborn_vlm(context: dict[str, Any], roi: np.ndarray) -> dict[str, Any]:
            return {
                "disposition": "SCRAP",
                "severity": "CRITICAL",
                "ipc_standard_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
                "electrical_failure_risk": "Dead short across adjacent traces.",
                "root_cause_hypothesis": "Under-etching.",
                "recommended_corrective_action": "Scrap board.",
                "halt_production_line": False,
            }

        store = BatchTelemetryStore(":memory:")
        graph = PCBAOIAgentGraph(
            detector=ONNXPCBDetector(),
            batch_store=store,
            vlm_callable=stubborn_vlm,
        )

        r1 = graph.run_inspection(template_img, test_img, batch_id="BATCH-DRIFT-99")
        r2 = graph.run_inspection(template_img, test_img, batch_id="BATCH-DRIFT-99")
        r3 = graph.run_inspection(template_img, test_img, batch_id="BATCH-DRIFT-99")

        assert r1.decisions[0].halt_production_line is False
        assert r2.decisions[0].halt_production_line is False
        assert r3.decisions[0].halt_production_line is True
        assert r3.decisions[0].guardrail_overridden is True
        assert "Override C" in (r3.decisions[0].override_reason or "")
        store.close()
