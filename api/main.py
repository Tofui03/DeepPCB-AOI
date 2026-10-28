"""HTTP API for template/test inspection, triage, and batch counts."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.agent.schemas import BOARD_DIM_PX, BoardInspectionReport, IPCClass
from core.agent.state_graph import PCBAOIAgentGraph
from core.data.deeppcb_loader import (
    DeepPCBDataset,
    DeepPCBSample,
    parse_annotation_file,
)
from core.tools.batch_db import BatchTelemetryStore
from core.vision.diff_preprocess import compute_differential_map
from core.vision.onnx_detector import ONNXPCBDetector

WEB_INDEX_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"

app = FastAPI(
    title="DeepPCB-AOI",
    description=(
        "Compare a bare-board template with a test image, localize six defect classes, "
        "and apply pixel-area rules plus safety overrides. An optional VLM callable "
        "can phrase the decision; it does not edit boxes."
    ),
    version="1.1.0",
)

dataset = DeepPCBDataset()
detector = ONNXPCBDetector(score_threshold=145.0)
batch_store = BatchTelemetryStore()
agent_graph = PCBAOIAgentGraph(detector=detector, batch_store=batch_store)

_ALL_SAMPLES: list[DeepPCBSample] = dataset.test_samples + dataset.trainval_samples
_SAMPLE_INDEX: dict[str, DeepPCBSample] = {s.sample_id: s for s in _ALL_SAMPLES}
_ALL_GROUPS: list[str] = sorted({s.template_path.parts[-3] for s in _ALL_SAMPLES})
_SAMPLE_META_CACHE: dict[str, dict[str, Any]] = {}


def _get_sample_metadata(sample: DeepPCBSample) -> dict[str, Any]:
    """Lazily compute and cache annotation metadata for a `DeepPCBSample`."""
    cached = _SAMPLE_META_CACHE.get(sample.sample_id)
    if cached is not None:
        return cached

    gt_boxes = parse_annotation_file(sample.annotation_path)
    meta = {
        "sample_id": sample.sample_id,
        "group": sample.template_path.parts[-3],
        "split": sample.split,
        "gt_defect_count": len(gt_boxes),
        "defect_classes": sorted({b.defect_class for b in gt_boxes}),
    }
    _SAMPLE_META_CACHE[sample.sample_id] = meta
    return meta


class InspectRequest(BaseModel):
    """Request payload for Stage 1 `/api/inspect`."""

    sample_id: str | None = Field(
        default=None,
        description="DeepPCB sample ID (defaults to first test sample if omitted).",
    )
    clean_board_mode: bool = Field(
        default=False,
        description="When True, the test image is replaced with the template.",
    )
    colorize_pcb: bool = Field(
        default=False,
        description="When True, renders realistic industrial green solder-mask + copper trace visuals.",
    )


class AgentTriageRequest(BaseModel):
    """Request payload for full 4-stage `/api/agent-triage`."""

    sample_id: str | None = Field(
        default=None,
        description="DeepPCB sample ID (defaults to first test sample if omitted).",
    )
    batch_id: str = Field(
        default="BATCH-SMT-01",
        min_length=1,
        description="Active manufacturing lot/batch identifier.",
    )
    ipc_class: IPCClass = Field(
        default="Class 2",
        description="Target IPC-A-600 inspection standard ('Class 2' or 'Class 3').",
    )
    clean_board_mode: bool = Field(
        default=False,
        description="When True, the test image is replaced with the template.",
    )
    simulate_rogue_vlm: bool = Field(
        default=False,
        description=(
            "When True, a stub response returns PASS / LOW so the safety override can be checked."
        ),
    )
    colorize_pcb: bool = Field(
        default=False,
        description="When True, renders realistic industrial green solder-mask + copper trace visuals.",
    )


class BatchResetRequest(BaseModel):
    """Request payload for `/api/batch-reset`."""

    batch_id: str = Field(
        default="BATCH-SMT-01",
        min_length=1,
        description="Manufacturing batch ID to clear in SQLite telemetry.",
    )


def _resolve_sample(sample_id: str | None) -> DeepPCBSample:
    if not sample_id:
        return dataset.test_samples[0]
    sample = _SAMPLE_INDEX.get(sample_id)
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown DeepPCB sample_id={sample_id!r}.",
        )
    return sample


def colorize_pcb_mask(gray_img: np.ndarray) -> np.ndarray:
    """Render a realistic industrial green FR4 solder-mask and copper/gold trace BGR image."""
    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    norm = gray_img.astype(np.float32) / 255.0
    # Dark pixels (0) -> bright copper/gold trace BGR; Bright pixels (255) -> deep green solder-mask BGR
    copper_bgr = np.array([42.0, 175.0, 225.0], dtype=np.float32)
    solder_mask_bgr = np.array([26.0, 72.0, 20.0], dtype=np.float32)
    blended = (1.0 - norm[:, :, np.newaxis]) * copper_bgr + (
        norm[:, :, np.newaxis] * solder_mask_bgr
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _encode_png_b64(
    img: np.ndarray,
    apply_heatmap: bool = False,
    colorize_pcb: bool = False,
) -> str:
    target = img
    if apply_heatmap and img.ndim == 2:
        target = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    elif colorize_pcb:
        target = colorize_pcb_mask(img)
    success, encoded = cv2.imencode(".png", target)
    if not success:
        raise RuntimeError("Failed to encode numpy image to PNG.")
    b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def _rogue_vlm_callable(context: dict[str, Any], roi_crop: np.ndarray) -> str:
    """Deliberately unsafe mock VLM output used to demonstrate Stage 4 Safety Guardrails."""
    ipc_clause = context.get("ipc_compliance", {}).get(
        "ipc_clause", "IPC-A-600 Section 2.2"
    )
    return json.dumps(
        {
            "disposition": "PASS",
            "severity": "LOW",
            "ipc_standard_clause": ipc_clause,
            "electrical_failure_risk": (
                "[ROGUE VLM HALLUCINATION] Claimed negligible cosmetic surface variation."
            ),
            "root_cause_hypothesis": (
                "[ROGUE VLM HALLUCINATION] Harmless optical reflection on copper trace."
            ),
            "recommended_corrective_action": (
                "[ROGUE VLM HALLUCINATION] Approve board for shipment without rework."
            ),
            "halt_production_line": False,
        }
    )


def _execute_triage_pipeline(
    template_img: np.ndarray,
    test_img: np.ndarray,
    sample_id: str,
    batch_id: str,
    ipc_class: IPCClass,
    clean_board_mode: bool,
    simulate_rogue_vlm: bool,
    colorize_pcb: bool,
    gt_defect_count: int,
) -> dict[str, Any]:
    """Shared execution helper for `/api/agent-triage` and `/api/upload-triage`."""
    if clean_board_mode:
        test_img = template_img.copy()

    active_graph = (
        PCBAOIAgentGraph(
            detector=detector,
            batch_store=batch_store,
            vlm_callable=_rogue_vlm_callable,
        )
        if simulate_rogue_vlm
        else agent_graph
    )

    report, trace = active_graph.run_inspection_with_trace(
        template_img=template_img,
        test_img=test_img,
        batch_id=batch_id,
        ipc_class=ipc_class,
    )
    diff_map = compute_differential_map(template_img, test_img)
    batch_summary = batch_store.get_batch_summary(batch_id)
    line_halted = any(d.halt_production_line for d in report.decisions) or any(
        b.threshold_exceeded for b in batch_summary
    )

    if report.fast_path_pass:
        board_disposition = "PASS"
    elif any(d.disposition == "SCRAP" for d in report.decisions):
        board_disposition = "SCRAP"
    elif any(d.disposition == "REWORK" for d in report.decisions):
        board_disposition = "REWORK"
    else:
        board_disposition = "PASS"

    return {
        "sample_id": sample_id,
        "batch_id": batch_id,
        "ipc_class": ipc_class,
        "clean_board_mode": clean_board_mode,
        "simulate_rogue_vlm": simulate_rogue_vlm,
        "colorize_pcb": colorize_pcb,
        "board_disposition": board_disposition,
        "line_halted": line_halted,
        "gt_defect_count": 0 if clean_board_mode else gt_defect_count,
        "report": report.model_dump(),
        "trace": trace,
        "batch_summary": [b.model_dump() for b in batch_summary],
        "template_b64": _encode_png_b64(template_img, colorize_pcb=colorize_pcb),
        "test_b64": _encode_png_b64(test_img, colorize_pcb=colorize_pcb),
        "diff_b64": _encode_png_b64(diff_map, apply_heatmap=True),
    }


@app.get("/", response_class=HTMLResponse)
def serve_console() -> HTMLResponse:
    """Serve ``web/index.html``."""
    if not WEB_INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail="web/index.html not found.")
    return HTMLResponse(content=WEB_INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/samples")
def list_curated_samples(
    group: str | None = Query(default=None, description="Optional PCB group filter (e.g., 'group20085')."),
    defect_class: str | None = Query(default=None, description="Optional defect class filter (e.g., 'Short')."),
    split: str = Query(default="test", description="Dataset split ('test', 'trainval', or 'all')."),
    limit: int = Query(default=100, ge=1, le=1500, description="Max samples to return."),
) -> dict[str, Any]:
    """Return DeepPCB samples across all 11 PCB groups with optional group and defect_class filtering."""
    if split == "test":
        pool = dataset.test_samples
    elif split == "trainval":
        pool = dataset.trainval_samples
    elif split == "all":
        pool = _ALL_SAMPLES
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported split={split!r}.")

    # Round-robin across groups so the default page is not one board repeated.
    by_group: dict[str, list[DeepPCBSample]] = {}
    for s in pool:
        g_name = s.template_path.parts[-3]
        if group and g_name != group:
            continue
        by_group.setdefault(g_name, []).append(s)

    interleaved: list[DeepPCBSample] = []
    max_len = max((len(v) for v in by_group.values()), default=0)
    sorted_groups = sorted(by_group.keys())
    for idx in range(max_len):
        for g_name in sorted_groups:
            if idx < len(by_group[g_name]):
                interleaved.append(by_group[g_name][idx])

    results: list[dict[str, Any]] = []
    for sample in interleaved:
        meta = _get_sample_metadata(sample)
        if defect_class and defect_class not in meta["defect_classes"]:
            continue
        results.append(meta)
        if len(results) >= limit:
            break

    return {
        "samples": results,
        "groups": _ALL_GROUPS,
        "total_available": len(_ALL_SAMPLES),
        "total_test_samples": len(dataset.test_samples),
    }


@app.post("/api/inspect")
def inspect_stage1(req: InspectRequest) -> dict[str, Any]:
    """Execute Stage 1 deterministic ONNX edge inspection and return report + base64 images."""
    sample = _resolve_sample(req.sample_id)
    template_img, test_img, gt_boxes = dataset.load_sample(sample)
    if req.clean_board_mode:
        test_img = template_img.copy()

    report: BoardInspectionReport = detector.inspect_fast_path(template_img, test_img)
    diff_map = compute_differential_map(template_img, test_img)

    return {
        "sample_id": sample.sample_id,
        "clean_board_mode": req.clean_board_mode,
        "colorize_pcb": req.colorize_pcb,
        "gt_defect_count": 0 if req.clean_board_mode else len(gt_boxes),
        "stage1_latency_ms": report.stage1_latency_ms,
        "fast_path_pass": report.fast_path_pass,
        "detections": [d.model_dump() for d in report.detections],
        "decisions": [d.model_dump() for d in report.decisions],
        "report": report.model_dump(),
        "template_b64": _encode_png_b64(template_img, colorize_pcb=req.colorize_pcb),
        "test_b64": _encode_png_b64(test_img, colorize_pcb=req.colorize_pcb),
        "diff_b64": _encode_png_b64(diff_map, apply_heatmap=True),
    }


@app.post("/api/agent-triage")
@app.post("/api/triage")
def run_agent_triage(req: AgentTriageRequest) -> dict[str, Any]:
    """Execute the full 4-Stage DeepPCB-AOI State Graph with trace and SQLite batch telemetry."""
    sample = _resolve_sample(req.sample_id)
    template_img, test_img, gt_boxes = dataset.load_sample(sample)

    return _execute_triage_pipeline(
        template_img=template_img,
        test_img=test_img,
        sample_id=sample.sample_id,
        batch_id=req.batch_id,
        ipc_class=req.ipc_class,
        clean_board_mode=req.clean_board_mode,
        simulate_rogue_vlm=req.simulate_rogue_vlm,
        colorize_pcb=req.colorize_pcb,
        gt_defect_count=len(gt_boxes),
    )


@app.post("/api/upload-triage")
async def upload_and_triage(
    template_file: UploadFile = File(..., description="Uploaded golden template PCB image (.jpg/.png)."),
    test_file: UploadFile = File(..., description="Uploaded tested PCB image (.jpg/.png)."),
    batch_id: str = Form(default="BATCH-UPLOAD"),
    ipc_class: IPCClass = Form(default="Class 2"),
    simulate_rogue_vlm: bool = Form(default=False),
    colorize_pcb: bool = Form(default=False),
) -> dict[str, Any]:
    """Decode uploaded Template & Tested PCB images, normalize to 640x640, and run 4-stage triage."""
    tpl_bytes = await template_file.read()
    tst_bytes = await test_file.read()

    if not tpl_bytes or not tst_bytes:
        raise HTTPException(status_code=400, detail="Both template_file and test_file must be non-empty.")

    tpl_arr = np.frombuffer(tpl_bytes, dtype=np.uint8)
    tst_arr = np.frombuffer(tst_bytes, dtype=np.uint8)

    template_img = cv2.imdecode(tpl_arr, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imdecode(tst_arr, cv2.IMREAD_GRAYSCALE)

    if template_img is None or test_img is None:
        raise HTTPException(status_code=400, detail="Failed to decode uploaded image files.")

    if template_img.shape != (BOARD_DIM_PX, BOARD_DIM_PX):
        template_img = cv2.resize(
            template_img, (BOARD_DIM_PX, BOARD_DIM_PX), interpolation=cv2.INTER_AREA
        )
    if test_img.shape != (BOARD_DIM_PX, BOARD_DIM_PX):
        test_img = cv2.resize(
            test_img, (BOARD_DIM_PX, BOARD_DIM_PX), interpolation=cv2.INTER_AREA
        )

    custom_id = f"UPLOAD:{Path(test_file.filename or 'board').stem}"
    return _execute_triage_pipeline(
        template_img=template_img,
        test_img=test_img,
        sample_id=custom_id,
        batch_id=batch_id,
        ipc_class=ipc_class,
        clean_board_mode=False,
        simulate_rogue_vlm=simulate_rogue_vlm,
        colorize_pcb=colorize_pcb,
        gt_defect_count=0,
    )


@app.get("/api/benchmark")
def get_benchmark(num_runs: int = Query(default=30, ge=5, le=200)) -> dict[str, Any]:
    """Run live Stage 1 ONNX latency benchmark and return dataset zero-leakage SHA-256 audit."""
    perf = detector.benchmark(num_runs=num_runs)
    split_audit = dataset.verify_zero_leakage()
    return {
        "onnx_benchmark": perf,
        "dataset_audit": split_audit,
        "sla_target_ms": 15.0,
        "sla_met": float(perf["mean_ms"]) < 15.0,
    }


@app.get("/api/batch-telemetry")
def get_batch_telemetry(
    batch_id: str = Query(default="BATCH-SMT-01", min_length=1),
) -> dict[str, Any]:
    """Return active SQLite batch defect counts and line-halt status for `batch_id`."""
    summary = batch_store.get_batch_summary(batch_id)
    return {
        "batch_id": batch_id,
        "line_halted": any(item.threshold_exceeded for item in summary),
        "batch_summary": [item.model_dump() for item in summary],
    }


@app.post("/api/batch-reset")
def reset_batch_telemetry(req: BatchResetRequest) -> dict[str, Any]:
    """Clear SQLite batch telemetry counts for `req.batch_id`."""
    batch_store.reset_batch(req.batch_id)
    return {
        "batch_id": req.batch_id,
        "status": "RESET_COMPLETE",
        "line_halted": False,
        "batch_summary": [],
    }
