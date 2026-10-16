"""Stage 4 Deterministic Python Safety Guardrails (Final Arbiter — Rule 5).

Architectural Rules Enforced (AGENTS.md Rule 5):
- Override A (Critical Open/Short): Any `Open` or `Short` defect can NEVER receive
  `disposition = "PASS"` or non-critical severity. Guardrail forces `disposition = "SCRAP"`
  and `severity = "CRITICAL"`.
- Override B (IPC Violation): Any defect with `ipc_result.violates_ipc == True` can NEVER
  receive `disposition = "PASS"`. Guardrail forces `"SCRAP"` (for Open/Short) or `"REWORK"`
  (for other classes) and ensures `severity` is at least `"HIGH"`.
- Override C (Batch Drift Line Halt): If `batch_result.threshold_exceeded` is `True`
  (`occurrence_count >= 3`), guardrail forces `halt_production_line = True`.
"""

from __future__ import annotations

from core.agent.schemas import (
    BatchTrendResult,
    BoundingBoxDetection,
    Disposition,
    InspectionDecision,
    IPCComplianceResult,
    Severity,
)

_SEVERITY_RANK: dict[Severity, int] = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def apply_safety_guardrails(
    decision: InspectionDecision,
    bbox: BoundingBoxDetection,
    ipc_result: IPCComplianceResult,
    batch_result: BatchTrendResult,
) -> InspectionDecision:
    """Enforce deterministic Stage 4 Python safety overrides on Stage 3 triage output."""
    disposition: Disposition = decision.disposition
    severity: Severity = decision.severity
    halt_line: bool = decision.halt_production_line
    overridden: bool = decision.guardrail_overridden

    reasons: list[str] = []
    if decision.override_reason:
        reasons.append(decision.override_reason)

    # Override A: Critical Open / Short zero-tolerance enforcement
    if bbox.defect_class in ("Open", "Short"):
        if disposition != "SCRAP" or severity != "CRITICAL":
            prev_disp, prev_sev = disposition, severity
            disposition = "SCRAP"
            severity = "CRITICAL"
            overridden = True
            reasons.append(
                f"Override A (Zero-Tolerance {bbox.defect_class}): "
                f"Forced disposition='SCRAP' and severity='CRITICAL' "
                f"(was disposition='{prev_disp}', severity='{prev_sev}')."
            )

    # Override B: IPC-A-600 violation cannot pass or be downgraded below HIGH severity
    if ipc_result.violates_ipc:
        if disposition == "PASS":
            target_disp: Disposition = (
                "SCRAP" if bbox.defect_class in ("Open", "Short") else "REWORK"
            )
            disposition = target_disp
            if _SEVERITY_RANK[severity] < _SEVERITY_RANK["HIGH"]:
                severity = "HIGH"
            overridden = True
            reasons.append(
                f"Override B (IPC Violation on {bbox.defect_class}): "
                f"Defect measured {ipc_result.measured_px}px > {ipc_result.max_allowed_px}px "
                f"({ipc_result.ipc_clause}); forced disposition='{target_disp}' and severity='{severity}'."
            )
        elif _SEVERITY_RANK[severity] < _SEVERITY_RANK["HIGH"]:
            prev_sev = severity
            severity = "HIGH"
            overridden = True
            reasons.append(
                f"Override B (IPC Severity Floor on {bbox.defect_class}): "
                f"Upgraded severity from '{prev_sev}' to 'HIGH' due to IPC violation."
            )

    # Override C: Batch drift line halt when occurrence_count >= 3
    if batch_result.threshold_exceeded and not halt_line:
        halt_line = True
        overridden = True
        reasons.append(
            f"Override C (Batch Drift Line Halt): "
            f"{bbox.defect_class} count reached {batch_result.occurrence_count} in batch "
            f"'{batch_result.batch_id}'; forced halt_production_line=True."
        )

    override_reason = " | ".join(reasons) if reasons else None

    return InspectionDecision(
        disposition=disposition,
        severity=severity,
        ipc_standard_clause=decision.ipc_standard_clause,
        electrical_failure_risk=decision.electrical_failure_risk,
        root_cause_hypothesis=decision.root_cause_hypothesis,
        recommended_corrective_action=decision.recommended_corrective_action,
        halt_production_line=halt_line,
        guardrail_overridden=overridden,
        override_reason=override_reason,
    )
