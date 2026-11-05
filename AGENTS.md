# DeepPCB-AOI

FastAPI service that compares a bare-board template with a test image, localizes six defect classes, and returns a rule-based PASS, REWORK, or SCRAP verdict. The WPF client spec is `docs/PRD.md`. Section 9 lists backend changes BC-1 through BC-8.

## Stack and layout

Python 3.12, FastAPI, uvicorn, Pydantic, ONNX Runtime, opencv-python-headless, NumPy. SQLite batch counts are created at runtime in `data/batch_telemetry.db` (gitignored).

- `api/main.py` — HTTP routes, detector selection, lazy DeepPCB load
- `core/vision/` — `YOLOv8PCBDetector`, differential `ONNXPCBDetector`, preprocess; ONNX files in `core/vision/models/`
- `core/agent/` — schemas, inspection pipeline, guardrails
- `core/tools/` — pixel-area rules, SQLite batch telemetry
- `core/data/` — DeepPCB loader; `mock_pcb/` is the pytest fixture only
- `web/index.html` — console served at `/`
- `scripts/` — dataset prep, training, evaluation; the API does not import these
- `tests/` — pytest
- `requirements.txt` — pinned inference, API, and test dependencies
- `requirements-train.txt` — pinned torch, torchvision, and ultralytics
- `Dockerfile` — `DETECTOR_BACKEND=yolov8`, healthcheck on `GET /api/health`, dataset not copied

`yolo` and `yolov8` load YOLOv8. `auto` loads YOLOv8 when `core/vision/models/deeppcb_yolov8n_3channel.onnx` exists, except when `DEEPPCB_ALLOW_MOCK=1` (pytest), which loads the differential detector. Any other value loads the differential detector. `GET /api/health` returns `detector_backend` `yolov8` or `differential` from the constructed object, not from the environment variable.

`/api/samples`, `/api/inspect`, `/api/agent-triage`, `/api/triage`, and `/api/benchmark` need `data/DeepPCB/PCBData` and return 503 when it is missing. `/api/health` and `/api/upload-triage` do not open the dataset.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
DETECTOR_BACKEND=yolov8 uvicorn api.main:app --host 0.0.0.0 --port 8000
docker build -t deeppcb-aoi .
docker run --rm -p 8000:8000 deeppcb-aoi
```

## Rules

- Response shapes come from the code and `docs/fixtures/upload_triage_sample.json`. Do not invent fields.
- Validate any LLM output before it is used.
- Do not commit the DeepPCB dataset (`data/`) or training weights (`*.pt`, `*.pth`, checkpoints). The ONNX files already in `core/vision/models/` stay tracked.
- Every change ships with proof: pytest output, or curl output for the endpoints.
