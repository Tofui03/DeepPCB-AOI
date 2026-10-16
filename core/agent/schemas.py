"""Strict Pydantic v2 data contracts for the DeepPCB-AOI 4-stage inspection pipeline.

Architectural Constraints (AGENTS.md):
- Rule 1: All spatial coordinates (x1, y1, x2, y2) and pixel area measurements are
  produced deterministically by Stage 1 (ONNX Vision) within 640x640 image bounds.
- Rule 2: Zero-token fast-path bypass when 0 defects are detected (`fast_path_pass=True`).
- Rule 3: Ground-truth tool outputs (`IPCComplianceResult`, `BatchTrendResult`) precede triage.
- Rule 4: Every agent triage output validates strictly against `InspectionDecision`.
- Rule 5: Deterministic Python safety guardrails have final override authority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

# DeepPCB canonical image dimensions (640x640 pixels)
BOARD_DIM_PX: int = 640
BOARD_TOTAL_AREA_PX: int = BOARD_DIM_PX * BOARD_DIM_PX

# Canonical 6 DeepPCB defect classes
DefectClass = Literal[
    "Open",
    "Short",
    "Mousebite",
    "Spur",
    "Spurious Copper",
    "Pin-hole",
]

IPCClass = Literal["Class 2", "Class 3"]
Disposition = Literal["PASS", "REWORK", "SCRAP"]
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class BoundingBoxDetection(BaseModel):
    """Stage 1 deterministic ONNX bounding box detection within a 640x640 PCB tile."""

    model_config = ConfigDict(extra="forbid")

    x1: int = Field(
        ...,
        ge=0,
        le=BOARD_DIM_PX,
        description="Left bounding box coordinate in pixels (0 to 640).",
    )
    y1: int = Field(
        ...,
        ge=0,
        le=BOARD_DIM_PX,
        description="Top bounding box coordinate in pixels (0 to 640).",
    )
    x2: int = Field(
        ...,
        ge=0,
        le=BOARD_DIM_PX,
        description="Right bounding box coordinate in pixels (0 to 640).",
    )
    y2: int = Field(
        ...,
        ge=0,
        le=BOARD_DIM_PX,
        description="Bottom bounding box coordinate in pixels (0 to 640).",
    )
    defect_class: DefectClass = Field(
        ...,
        description="Canonical DeepPCB defect class label.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ONNX detector confidence score bounded between 0.0 and 1.0.",
    )

    @model_validator(mode="after")
    def validate_box_geometry(self) -> BoundingBoxDetection:
        """Ensure bounding box has strictly positive width and height (x2 > x1 and y2 > y1)."""
        if self.x2 <= self.x1:
            raise ValueError(
                f"Invalid bounding box width: x2 ({self.x2}) must be strictly greater than x1 ({self.x1})."
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"Invalid bounding box height: y2 ({self.y2}) must be strictly greater than y1 ({self.y1})."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area_px(self) -> int:
        """Deterministic bounding box area in square pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def relative_area_pct(self) -> float:
        """Bounding box area as a percentage (0.0 to 100.0) of the 640x640 PCB inspection region."""
        return round((self.area_px / BOARD_TOTAL_AREA_PX) * 100.0, 4)


class IPCComplianceResult(BaseModel):
    """Stage 2 deterministic IPC-A-600 compliance evaluation output."""

    model_config = ConfigDict(extra="forbid")

    defect_class: DefectClass = Field(
        ...,
        description="Canonical DeepPCB defect class evaluated against IPC-A-600.",
    )
    ipc_class: IPCClass = Field(
        ...,
        description="Target IPC-A-600 product classification ('Class 2' or 'Class 3').",
    )
    violates_ipc: bool = Field(
        ...,
        description="True if measured defect geometry exceeds the maximum allowed IPC threshold.",
    )
    ipc_clause: str = Field(
        ...,
        min_length=1,
        description="Specific IPC-A-600 standard clause reference (e.g., 'IPC-A-600 Section 2.2.1').",
    )
    max_allowed_px: int = Field(
        ...,
        ge=0,
        description="Maximum permissible defect area in pixels under the specified IPC class.",
    )
    measured_px: int = Field(
        ...,
        ge=0,
        description="Measured defect area in pixels from Stage 1 deterministic localization.",
    )


class BatchTrendResult(BaseModel):
    """Stage 2 SQLite batch telemetry lookup result for recurring process defects."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Active manufacturing lot/batch identifier.",
    )
    defect_class: DefectClass = Field(
        ...,
        description="Defect class queried in the active production batch.",
    )
    occurrence_count: int = Field(
        ...,
        ge=0,
        description="Cumulative count of this defect class recorded in the active batch.",
    )
    threshold_exceeded: bool = Field(
        ...,
        description="True if occurrence_count >= 3, signaling systemic upstream process drift.",
    )


class InspectionDecision(BaseModel):
    """Stage 3 VLM semantic triage output and Stage 4 deterministic safety guardrail decision."""

    model_config = ConfigDict(extra="forbid")

    disposition: Disposition = Field(
        ...,
        description="Final board disposition: 'PASS', 'REWORK', or 'SCRAP'.",
    )
    severity: Severity = Field(
        ...,
        description="Defect severity rating: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.",
    )
    ipc_standard_clause: str = Field(
        ...,
        min_length=1,
        description="Cited IPC-A-600 compliance clause governing this decision.",
    )
    electrical_failure_risk: str = Field(
        ...,
        min_length=1,
        description="Downstream electrical failure mechanism if the defect escapes.",
    )
    root_cause_hypothesis: str = Field(
        ...,
        min_length=1,
        description="Hypothesized upstream PCB fabrication or SMT process root cause.",
    )
    recommended_corrective_action: str = Field(
        ...,
        min_length=1,
        description="Actionable upstream process adjustment for line operators.",
    )
    halt_production_line: bool = Field(
        ...,
        description="True if recurring defects require an immediate SMT/etching line halt.",
    )
    guardrail_overridden: bool = Field(
        default=False,
        description="True if Stage 4 deterministic Python guardrails overrode Stage 3 VLM output.",
    )
    override_reason: str | None = Field(
        default=None,
        description="Audit explanation when Stage 4 safety guardrails override the VLM decision.",
    )


class BoardInspectionReport(BaseModel):
    """Complete end-to-end PCB inspection report across all 4 pipeline stages."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stage1_latency_ms: float = Field(
        ...,
        ge=0.0,
        validation_alias=AliasChoices("stage1_latency_ms", "latency_ms"),
        description="Stage 1 deterministic ONNX inference latency in milliseconds.",
    )
    detections: list[BoundingBoxDetection] = Field(
        default_factory=list,
        description="List of deterministic bounding box detections from Stage 1.",
    )
    fast_path_pass: bool = Field(
        ...,
        description="True when 0 defects are detected and Stages 2-4 are bypassed (Rule 2).",
    )
    decisions: list[InspectionDecision] = Field(
        default_factory=list,
        description="List of per-defect triage decisions after Stage 4 safety guardrails.",
    )
