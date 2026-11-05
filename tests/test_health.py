"""Health payload and startup when the DeepPCB corpus is absent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import (  # noqa: E402
    _select_yolov8,
    app,
    detector,
    detector_backend_name,
)
from core.vision.onnx_detector import ONNXPCBDetector  # noqa: E402
from core.vision.yolo_detector import YOLOv8PCBDetector  # noqa: E402

MOCK_TEMPLATE = (
    PROJECT_ROOT
    / "core"
    / "data"
    / "mock_pcb"
    / "group20085"
    / "20085"
    / "20085000_temp.jpg"
)
MOCK_TEST = (
    PROJECT_ROOT
    / "core"
    / "data"
    / "mock_pcb"
    / "group20085"
    / "20085"
    / "20085000_test.jpg"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_loaded_detector_not_env_var(client: TestClient) -> None:
    """pytest leaves DETECTOR_BACKEND at auto and forces the differential detector."""
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"status", "version", "detector_backend"}
    assert body["status"] == "ok"
    assert body["version"] == app.version == "1.1.0"
    assert isinstance(detector, ONNXPCBDetector)
    assert body["detector_backend"] == detector_backend_name(detector) == "differential"
    assert os.environ.get("DETECTOR_BACKEND", "auto").strip().lower() == "auto"
    assert body["detector_backend"] != "auto"


def test_backend_name_follows_instance_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A differential instance stays 'differential' even if the env var says yolov8."""
    monkeypatch.setenv("DETECTOR_BACKEND", "yolov8")

    class _Yolo(YOLOv8PCBDetector):
        def __init__(self) -> None:
            pass

    class _Diff(ONNXPCBDetector):
        def __init__(self) -> None:
            pass

    assert detector_backend_name(_Yolo()) == "yolov8"
    assert detector_backend_name(_Diff()) == "differential"
    assert os.environ["DETECTOR_BACKEND"] == "yolov8"
    with pytest.raises(TypeError, match="Unknown detector type"):
        detector_backend_name(object())


@pytest.mark.parametrize(
    ("requested", "weights_present", "is_test_env", "expect_yolo"),
    [
        ("auto", True, False, True),
        ("auto", True, True, False),
        ("auto", False, False, False),
        ("yolov8", False, True, True),
        ("yolo", False, False, True),
        (" YOLOV8 ", True, True, True),
        ("differential", True, False, False),
        ("onnx", True, False, False),
        ("", True, False, False),
    ],
)
def test_select_yolov8_rules(
    requested: str,
    weights_present: bool,
    is_test_env: bool,
    expect_yolo: bool,
) -> None:
    assert (
        _select_yolov8(
            requested,
            weights_present=weights_present,
            is_test_env=is_test_env,
        )
        is expect_yolo
    )


def test_server_imports_and_serves_without_deeppcb_dataset() -> None:
    """Import must not construct DeepPCBDataset. Sample routes 503; health and upload work."""
    script = r"""
import os
from pathlib import Path

os.environ.pop("DEEPPCB_ALLOW_MOCK", None)
os.environ["DETECTOR_BACKEND"] = "not-a-backend"

import core.data.deeppcb_loader as loader

loader.REAL_DEEPPCB_ROOT = Path("/tmp/deeppcb-absent/PCBData")
calls = {"n": 0}
_orig = loader.DeepPCBDataset.__init__

def _counting(self, *args, **kwargs):
    calls["n"] += 1
    return _orig(self, *args, **kwargs)

loader.DeepPCBDataset.__init__ = _counting

import api.main as main

assert calls["n"] == 0, calls
from fastapi.testclient import TestClient

client = TestClient(main.app)
health = client.get("/api/health")
assert health.status_code == 200, health.text
body = health.json()
assert body["status"] == "ok"
assert body["version"] == main.app.version
assert body["detector_backend"] == "differential"
assert body["detector_backend"] != os.environ["DETECTOR_BACKEND"]
assert calls["n"] == 0, "health must not open the dataset"

samples = client.get("/api/samples")
assert samples.status_code == 503, samples.text
detail = samples.json()["detail"]
assert "DeepPCB dataset" in detail
assert "/api/upload-triage" in detail
assert calls["n"] == 1

again = client.get("/api/samples")
assert again.status_code == 503
assert calls["n"] == 1

for path, kwargs in (
    ("/api/inspect", {"json": {}}),
    ("/api/agent-triage", {"json": {}}),
    ("/api/triage", {"json": {}}),
    ("/api/benchmark", {}),
):
    res = client.request("POST" if path != "/api/benchmark" else "GET", path, **kwargs)
    assert res.status_code == 503, (path, res.status_code, res.text)
    assert "DeepPCB dataset" in res.json()["detail"]

root = Path("core/data/mock_pcb/group20085/20085")
files = {
    "template_file": ("20085000_temp.jpg", (root / "20085000_temp.jpg").read_bytes(), "image/jpeg"),
    "test_file": ("20085000_test.jpg", (root / "20085000_test.jpg").read_bytes(), "image/jpeg"),
}
upload = client.post("/api/upload-triage", files=files, data={"batch_id": "BATCH-NO-DATASET"})
assert upload.status_code == 200, upload.text
payload = upload.json()
assert payload["sample_id"] == "UPLOAD:20085000_test"
assert payload["board_disposition"] in {"PASS", "REWORK", "SCRAP"}
assert calls["n"] == 1
print("STARTUP_OK")
"""
    env = os.environ.copy()
    env.pop("DEEPPCB_ALLOW_MOCK", None)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "STARTUP_OK" in result.stdout
    assert MOCK_TEMPLATE.is_file() and MOCK_TEST.is_file()
