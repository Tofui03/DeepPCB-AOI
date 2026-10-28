"""End-to-end integration tests for DeepPCB-AOI FastAPI REST endpoints."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_serve_console_index_html(client: TestClient) -> None:
    """GET / should serve the Industrial AOI Web Console HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "DeepPCB-AOI" in res.text
    assert "PRODUCTION LINE HALTED" in res.text


def test_list_curated_samples_and_filtering(client: TestClient) -> None:
    """GET /api/samples should support default listing, groups list, and group/defect filtering."""
    # Default listing
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert "groups" in data
    if data["total_available"] >= 1000:
        assert len(data["groups"]) >= 10
        assert len(data["samples"]) >= 10
    else:
        assert len(data["groups"]) >= 1
        assert len(data["samples"]) >= 1
        assert data["total_available"] >= 1
    first = data["samples"][0]
    assert "sample_id" in first
    assert "group" in first
    assert "gt_defect_count" in first
    assert isinstance(first["defect_classes"], list)

    # Filter by group
    target_group = data["groups"][0]
    res_group = client.get(f"/api/samples?group={target_group}&limit=50")
    assert res_group.status_code == 200
    group_data = res_group.json()
    assert all(s["group"] == target_group for s in group_data["samples"])

    # Filter by defect_class
    res_defect = client.get("/api/samples?defect_class=Short&limit=30")
    assert res_defect.status_code == 200
    defect_data = res_defect.json()
    assert all("Short" in s["defect_classes"] for s in defect_data["samples"])


def test_inspect_stage1_normal_and_fast_path(client: TestClient) -> None:
    """POST /api/inspect should test Stage 1 detection and clean_board_mode fast-path bypass."""
    # Normal sample
    res_normal = client.post("/api/inspect", json={"clean_board_mode": False})
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert data_normal["stage1_latency_ms"] < 15.0
    assert data_normal["fast_path_pass"] is False
    assert len(data_normal["detections"]) > 0
    assert data_normal["template_b64"].startswith("data:image/png;base64,")
    assert data_normal["test_b64"].startswith("data:image/png;base64,")
    assert data_normal["diff_b64"].startswith("data:image/png;base64,")

    # Fast-Path clean board bypass
    res_clean = client.post("/api/inspect", json={"clean_board_mode": True})
    assert res_clean.status_code == 200
    data_clean = res_clean.json()
    assert data_clean["fast_path_pass"] is True
    assert len(data_clean["detections"]) == 0
    assert data_clean["stage1_latency_ms"] < 15.0


def test_colorize_pcb_rendering(client: TestClient) -> None:
    """colorize_pcb=True should generate valid base64 PNGs in inspect and agent-triage."""
    res_inspect = client.post("/api/inspect", json={"colorize_pcb": True})
    assert res_inspect.status_code == 200
    data_inspect = res_inspect.json()
    assert data_inspect["colorize_pcb"] is True
    assert data_inspect["template_b64"].startswith("data:image/png;base64,")

    res_triage = client.post(
        "/api/agent-triage",
        json={"colorize_pcb": True, "batch_id": "BATCH-COLOR-TEST"},
    )
    assert res_triage.status_code == 200
    data_triage = res_triage.json()
    assert data_triage["colorize_pcb"] is True
    assert data_triage["template_b64"].startswith("data:image/png;base64,")


def test_agent_triage_full_pipeline_and_fast_path(client: TestClient) -> None:
    """POST /api/agent-triage should execute all 4 stages with trace and handle fast-path bypass."""
    # Clean board fast-path bypass
    res_fast = client.post(
        "/api/agent-triage",
        json={"clean_board_mode": True, "batch_id": "BATCH-API-TEST"},
    )
    assert res_fast.status_code == 200
    data_fast = res_fast.json()
    assert data_fast["report"]["fast_path_pass"] is True
    assert data_fast["board_disposition"] == "PASS"
    assert any(step.get("fast_path_pass") is True for step in data_fast["trace"])

    # Normal multi-stage triage
    res_triage = client.post(
        "/api/agent-triage",
        json={
            "clean_board_mode": False,
            "batch_id": "BATCH-API-TEST",
            "ipc_class": "Class 2",
        },
    )
    assert res_triage.status_code == 200
    data_triage = res_triage.json()
    assert data_triage["report"]["fast_path_pass"] is False
    assert len(data_triage["report"]["decisions"]) > 0
    assert len(data_triage["trace"]) >= 4


def test_agent_triage_rogue_vlm_guardrail_override(client: TestClient) -> None:
    """POST /api/agent-triage with simulate_rogue_vlm=True must trigger Stage 4 guardrail override."""
    res = client.post(
        "/api/agent-triage",
        json={
            "simulate_rogue_vlm": True,
            "clean_board_mode": False,
            "batch_id": "BATCH-ROGUE-TEST",
        },
    )
    assert res.status_code == 200
    data = res.json()
    decisions = data["report"]["decisions"]
    assert len(decisions) > 0
    overridden_decisions = [d for d in decisions if d["guardrail_overridden"] is True]
    assert len(overridden_decisions) > 0
    first_override = overridden_decisions[0]
    assert first_override["disposition"] in ("SCRAP", "REWORK")
    assert first_override["override_reason"] is not None


def test_upload_triage_endpoint(client: TestClient) -> None:
    """POST /api/upload-triage should accept multipart image uploads, resize to 640x640, and run triage."""
    # Synthesize template (uniform 240) and test (with a dark copper short bridge)
    tpl_img = np.full((640, 640), 240, dtype=np.uint8)
    tst_img = tpl_img.copy()
    tst_img[200:230, 300:350] = 10  # Excess copper defect

    _, tpl_encoded = cv2.imencode(".png", tpl_img)
    _, tst_encoded = cv2.imencode(".png", tst_img)

    files = {
        "template_file": ("custom_template.png", io.BytesIO(tpl_encoded.tobytes()), "image/png"),
        "test_file": ("custom_tested.png", io.BytesIO(tst_encoded.tobytes()), "image/png"),
    }
    data = {
        "batch_id": "BATCH-CUSTOM-UPLOAD",
        "ipc_class": "Class 2",
        "simulate_rogue_vlm": "false",
        "colorize_pcb": "true",
    }

    res = client.post("/api/upload-triage", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()
    assert resp_data["sample_id"] == "UPLOAD:custom_tested"
    assert resp_data["colorize_pcb"] is True
    assert len(resp_data["report"]["detections"]) > 0
    assert len(resp_data["report"]["decisions"]) > 0
    assert resp_data["board_disposition"] in ("SCRAP", "REWORK")
    assert resp_data["template_b64"].startswith("data:image/png;base64,")


def test_batch_drift_line_halt_and_reset(client: TestClient) -> None:
    """Three triage runs on defective boards in the same batch must trigger line_halted=True, then reset cleanly."""
    batch_id = "BATCH-DRIFT-API"

    # Reset initially to guarantee clean slate
    client.post("/api/batch-reset", json={"batch_id": batch_id})

    # Run 1
    r1 = client.post("/api/agent-triage", json={"batch_id": batch_id})
    assert r1.status_code == 200

    # Run 2
    r2 = client.post("/api/agent-triage", json={"batch_id": batch_id})
    assert r2.status_code == 200

    # Run 3
    r3 = client.post("/api/agent-triage", json={"batch_id": batch_id})
    assert r3.status_code == 200
    data_r3 = r3.json()
    assert data_r3["line_halted"] is True

    # Check telemetry query endpoint
    tel = client.get(f"/api/batch-telemetry?batch_id={batch_id}")
    assert tel.status_code == 200
    assert tel.json()["line_halted"] is True

    # Reset batch
    reset_res = client.post("/api/batch-reset", json={"batch_id": batch_id})
    assert reset_res.status_code == 200
    assert reset_res.json()["line_halted"] is False

    # Check telemetry query again
    tel_after = client.get(f"/api/batch-telemetry?batch_id={batch_id}")
    assert tel_after.json()["line_halted"] is False
    assert tel_after.json()["batch_summary"] == []


def test_benchmark_endpoint(client: TestClient) -> None:
    """GET /api/benchmark should return latency metrics (<15ms) and dataset SHA-256 split audit."""
    res = client.get("/api/benchmark?num_runs=10")
    assert res.status_code == 200
    data = res.json()
    assert "onnx_benchmark" in data
    assert "dataset_audit" in data
    assert data["sla_met"] is True
    assert float(data["onnx_benchmark"]["mean_ms"]) < 15.0
    assert data["dataset_audit"]["zero_leakage"] is True
    if not data["dataset_audit"].get("is_mock"):
        assert data["dataset_audit"]["trainval_count"] == 1000
        assert data["dataset_audit"]["test_count"] == 500
    else:
        assert data["dataset_audit"]["trainval_count"] >= 1
        assert data["dataset_audit"]["test_count"] >= 1
