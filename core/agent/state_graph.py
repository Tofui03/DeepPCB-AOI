"""4-Stage DeepPCB-AOI State Graph orchestrating Edge Vision, Ground-Truth Tools, VLM Triage, and Safety Guardrails.

Pipeline Architecture:
- Stage 1: Deterministic ONNX Runtime detection (`ONNXPCBDetector.detect`) + Rule 2 Fast-Path Gate.
- Stage 2: Ground-Truth Tool Execution (`crop_defect_roi`, `evaluate_detection_ipc`, `record_and_query_batch`).
- Stage 3: Bounded VLM Semantic Triage (`MAX_VLM_RETRIES = 2`) with deterministic IPC rule engine fallback.
- Stage 4: Deterministic Python Safety Guardrails (`apply_safety_guardrails`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.agent.guardrails import apply_safety_guardrails
from core.agent.schemas import (
    BatchTrendResult,
    BoardInspectionReport,
    BoundingBoxDetection,
    Disposition,
    InspectionDecision,
    IPCClass,
    IPCComplianceResult,
    Severity,
)
from core.tools.batch_db import BatchTelemetryStore
from core.tools.ipc_rules import evaluate_detection_ipc, get_ipc_metadata
from core.vision.diff_preprocess import crop_defect_roi
from core.vision.onnx_detector import ONNXPCBDetector

MAX_VLM_RETRIES: int = 2
MAX_RETRIES: int = MAX_VLM_RETRIES

VLMCallable = Callable[
    [dict[str, Any], np.ndarray],
    dict[str, Any] | str | InspectionDecision,
]


def build_deterministic_fallback_decision(
    bbox: BoundingBoxDetection,
    ipc_result: IPCComplianceResult,
    batch_result: BatchTrendResult,
) -> InspectionDecision:
    """Build a 100% deterministic `InspectionDecision` from verified Stage 2 tool outputs."""
    meta = get_ipc_metadata(bbox.defect_class)

    if bbox.defect_class in ("Open", "Short"):
        disposition: Disposition = "SCRAP"
        severity: Severity = "CRITICAL"
    elif ipc_result.violates_ipc:
        disposition = "REWORK"
        severity = "HIGH"
    else:
        disposition = "PASS"
        severity = "LOW"

    return InspectionDecision(
        disposition=disposition,
        severity=severity,
        ipc_standard_clause=ipc_result.ipc_clause,
        electrical_failure_risk=meta["electrical_risk"],
        root_cause_hypothesis=meta["root_cause"],
        recommended_corrective_action=meta["corrective_action"],
        halt_production_line=batch_result.threshold_exceeded,
        guardrail_overridden=False,
        override_reason=None,
    )


class PCBAOIAgentGraph:
    """Deterministic 4-stage inspection state graph with bounded VLM retries and safety guardrails."""

    def __init__(
        self,
        detector: ONNXPCBDetector | None = None,
        batch_store: BatchTelemetryStore | None = None,
        vlm_callable: VLMCallable | None = None,
    ) -> None:
        self.detector = detector if detector is not None else ONNXPCBDetector()
        self.batch_store = (
            batch_store if batch_store is not None else BatchTelemetryStore(":memory:")
        )
        self.vlm_callable = vlm_callable

    def _invoke_stage3_triage(
        self,
        bbox: BoundingBoxDetection,
        ipc_result: IPCComplianceResult,
        batch_result: BatchTrendResult,
        roi_crop: np.ndarray,
    ) -> tuple[InspectionDecision, dict[str, Any]]:
        """Execute Stage 3 VLM triage with strict `MAX_VLM_RETRIES = 2` cap and deterministic fallback."""
        if self.vlm_callable is None:
            fallback = build_deterministic_fallback_decision(bbox, ipc_result, batch_result)
            return fallback, {
                "stage": "Stage 3: Deterministic Triage (No External VLM Configured)",
                "vlm_attempts": 0,
                "used_fallback": True,
                "last_error": None,
                "raw_decision": fallback.model_dump(),
            }

        max_attempts = 1 + MAX_VLM_RETRIES
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            prompt_context: dict[str, Any] = {
                "bbox": bbox.model_dump(),
                "ipc_compliance": ipc_result.model_dump(),
                "batch_trend": batch_result.model_dump(),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "previous_validation_error": last_error,
            }
            try:
                raw_response = self.vlm_callable(prompt_context, roi_crop)
                if isinstance(raw_response, InspectionDecision):
                    validated = raw_response
                elif isinstance(raw_response, str):
                    validated = InspectionDecision.model_validate_json(raw_response)
                elif isinstance(raw_response, dict):
                    validated = InspectionDecision.model_validate(raw_response)
                else:
                    raise TypeError(
                        f"Unsupported VLM response type: {type(raw_response).__name__}"
                    )

                return validated, {
                    "stage": "Stage 3: VLM Semantic Triage",
                    "vlm_attempts": attempt,
                    "used_fallback": False,
                    "last_error": last_error,
                    "raw_decision": validated.model_dump(),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

        fallback = build_deterministic_fallback_decision(bbox, ipc_result, batch_result)
        return fallback, {
            "stage": "Stage 3: VLM Triage Exhausted -> Deterministic Fallback",
            "vlm_attempts": max_attempts,
            "used_fallback": True,
            "last_error": last_error,
            "raw_decision": fallback.model_dump(),
        }

    def run_inspection_with_trace(
        self,
        template_img: np.ndarray,
        test_img: np.ndarray,
        batch_id: str = "BATCH-DEFAULT",
        ipc_class: IPCClass = "Class 2",
    ) -> tuple[BoardInspectionReport, list[dict[str, Any]]]:
        """Execute the 4-stage inspection pipeline and return `(report, execution_trace)`."""
        trace: list[dict[str, Any]] = []

        # Stage 1: Deterministic ONNX Vision
        detections, stage1_latency_ms = self.detector.detect(template_img, test_img)
        trace.append(
            {
                "stage": "Stage 1: ONNX Edge Vision",
                "latency_ms": stage1_latency_ms,
                "num_detections": len(detections),
                "execution_provider": self.detector.execution_provider,
            }
        )

        # Rule 2: Zero-Token Fast-Path Bypass
        if len(detections) == 0:
            trace.append(
                {
                    "stage": "Stage 1 Fast-Path Gate",
                    "fast_path_pass": True,
                    "disposition": "PASS",
                    "stage2_tool_calls": 0,
                    "stage3_vlm_calls": 0,
                }
            )
            report = BoardInspectionReport(
                stage1_latency_ms=stage1_latency_ms,
                detections=[],
                fast_path_pass=True,
                decisions=[],
            )
            return report, trace

        decisions: list[InspectionDecision] = []
        for idx, bbox in enumerate(detections):
            # Stage 2: Ground-Truth Pre-Triage Tools (Rule 3)
            roi_crop = crop_defect_roi(test_img, bbox, padding=32)
            ipc_result = evaluate_detection_ipc(bbox, ipc_class=ipc_class)
            batch_result = self.batch_store.record_and_query_batch(
                batch_id=batch_id,
                defect_class=bbox.defect_class,
                increment=1,
            )
            trace.append(
                {
                    "stage": "Stage 2: Ground-Truth Tools",
                    "defect_index": idx,
                    "defect_class": bbox.defect_class,
                    "roi_shape": list(roi_crop.shape),
                    "ipc_compliance": ipc_result.model_dump(),
                    "batch_trend": batch_result.model_dump(),
                }
            )

            # Stage 3: Bounded VLM Triage or Deterministic Fallback (Rule 4)
            stage3_decision, stage3_trace = self._invoke_stage3_triage(
                bbox=bbox,
                ipc_result=ipc_result,
                batch_result=batch_result,
                roi_crop=roi_crop,
            )
            stage3_trace["defect_index"] = idx
            trace.append(stage3_trace)

            # Stage 4: Deterministic Python Safety Guardrails (Rule 5)
            final_decision = apply_safety_guardrails(
                decision=stage3_decision,
                bbox=bbox,
                ipc_result=ipc_result,
                batch_result=batch_result,
            )
            trace.append(
                {
                    "stage": "Stage 4: Safety Guardrails",
                    "defect_index": idx,
                    "guardrail_overridden": final_decision.guardrail_overridden,
                    "override_reason": final_decision.override_reason,
                    "final_decision": final_decision.model_dump(),
                }
            )
            decisions.append(final_decision)

        report = BoardInspectionReport(
            stage1_latency_ms=stage1_latency_ms,
            detections=detections,
            fast_path_pass=False,
            decisions=decisions,
        )
        return report, trace

    def run_inspection(
        self,
        template_img: np.ndarray,
        test_img: np.ndarray,
        batch_id: str = "BATCH-DEFAULT",
        ipc_class: IPCClass = "Class 2",
    ) -> BoardInspectionReport:
        """Execute the 4-stage inspection pipeline and return `BoardInspectionReport`."""
        report, _ = self.run_inspection_with_trace(
            template_img=template_img,
            test_img=test_img,
            batch_id=batch_id,
            ipc_class=ipc_class,
        )
        return report
