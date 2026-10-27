# DeepPCB-AOI

Bare-board optical inspection for the six defect classes in the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset (Tang, He, Huang, and Yang, 2019). A template image is compared with a test image. The difference is turned into boxes, each box is checked against a pixel-area limit, and a short rule set can override the final disposition.

The checked-in ONNX file is a fixed difference graph, and the class label is a geometry heuristic. There is no trained checkpoint and no accuracy table in this repo. Tests check that Stage 1 stays under 15 ms. That budget is not a measured frames-per-second claim.

![Template, test image with detector boxes, and the difference map for one DeepPCB pair.](docs/inspect-loop.gif)

## Defect classes

| Class | What it is on the copper |
| --- | --- |
| Open | Break in a trace |
| Short | Copper bridge between traces |
| Mousebite | Nick in the trace edge |
| Spur | Sharp copper protrusion |
| Spurious Copper | Isolated copper speck |
| Pin-hole | Small void in copper |

DeepPCB labels use ids 1–6 in that order. Images are 640×640.

## Pipeline

1. **Detect.** `core/vision` resizes nothing (inputs must already be 640×640), runs the ONNX graph, and groups high-scoring cells with connected components. Class comes from the sign of `template - test` (missing copper vs extra copper), plus area, aspect ratio, and fill. Non-maximum suppression drops overlaps. No boxes means `disposition = PASS` and stages 2–4 do not run.
2. **Rules and batch history.** `core/tools` compares box area with the table below and increments a SQLite count for `(batch_id, defect_class)`.
3. **Optional model triage.** If a callable is configured, its output must validate as `InspectionDecision`. It is tried at most three times (`MAX_VLM_RETRIES = 2`). Anything else falls back to the rule engine. The default server does not call an external model. The callable does not edit box coordinates.
4. **Overrides** in `core/agent/guardrails.py`. Open and Short cannot be PASS (forced to SCRAP / CRITICAL). An over-limit box cannot be PASS. The third occurrence of the same class in a batch sets `halt_production_line`.

![Pipeline diagram](docs/architecture/system-architecture.svg)

The editable source is `docs/architecture/system-architecture.excalidraw`. `docs/architecture/system-architecture.png` is an older export and still shows a 512×512 caption and a 263 FPS figure. Those numbers are not what the code does. Use the SVG.

## Area limits

`core/tools/ipc_rules.py` uses these pixel areas as a Class 2 / Class 3 stand-in. IPC-A-600 itself is written in conductor width and spacing; the clause strings here are names for this table. Open and Short always fail the check.

| Class | Class 2 max area (px) | Class 3 max area (px) |
| --- | ---: | ---: |
| Open | 0 | 0 |
| Short | 0 | 0 |
| Mousebite | 250 | 120 |
| Spur | 200 | 100 |
| Spurious Copper | 180 | 80 |
| Pin-hole | 160 | 75 |

`check_ipc_rule(..., relative_width_pct=)` can also fail a non-fatal class above 25% (Class 2) or 20% (Class 3). The live pipeline calls `evaluate_detection_ipc`, which does not pass that argument, so decisions use area only.

Batch halt threshold: 3 occurrences of the same class (`BATCH_HALT_THRESHOLD`).

## ONNX file

`core/vision/models/deeppcb_diff_detector.onnx` is 696 bytes, small enough to keep in normal git. The graph is IR version 8, opset 13:

- inputs `template`, `test`: float32 `[1, 1, 640, 640]`
- `Sub`, then `Abs`
- two stride-4, kernel-4 convolutions (box average)
- outputs `diff_score`, `polarity_map`: float32 `[1, 1, 160, 160]`

`ONNXPCBDetector` writes the same graph if the file is missing (`build_differential_onnx_model`). Nothing in this repo trains weights.

The API constructs the detector with `score_threshold=145`. The class default is 28. Cells below the threshold are ignored before connected components.

## Layout

```
core/vision/    difference image, ONNX session, NMS
core/tools/     area table, SQLite batch counts
core/agent/     Pydantic models, pipeline, overrides
core/data/      DeepPCB manifests and annotation parser
api/            FastAPI app
web/index.html  console served at /
tests/          pytest
```

HTTP routes: `GET /`, `GET /api/samples`, `POST /api/inspect`, `POST /api/agent-triage` (alias `POST /api/triage`), `POST /api/upload-triage`, `GET /api/benchmark`, `GET /api/batch-telemetry`, `POST /api/batch-reset`.

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The image set is not in git (see `.gitignore`). Place the upstream tree so `PCBData` sits here:

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
```

The loader expects `data/DeepPCB/PCBData/trainval.txt` (1,000 lines) and `test.txt` (500 lines), plus `*_temp.jpg`, `*_test.jpg`, and the annotation txt files. `api.main` builds `DeepPCBDataset()` on import, so the server and the API tests need that directory.

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`. The page is `web/index.html`.

`GET /api/benchmark` times the detector on one fixed synthetic pair (two dark rectangles) and returns latency percentiles. The `sla_met` flag is `mean_ms < 15`.

SQLite state defaults to `data/batch_telemetry.db` (gitignored). Tests use `:memory:` or a temp file.

## Tests

```bash
pytest
```

Coverage is the schema, the area table, batch counts, guardrail overrides (including a stub model that tries to PASS a short), upload decoding, and Stage 1 latency under 15 ms. There is no held-out accuracy eval.

## Design choices

- Boxes, scores, and areas are computed in `core/vision`. Later stages only read them.
- An empty detection list returns immediately. The optional model is not called.
- The area check and the batch counter run before that model call.
- Model output that fails `InspectionDecision` is retried twice, then replaced by the rule engine.
- Open, Short, and a repeated class are overridden in Python after triage. `guardrail_overridden` records that.

## Limitations

- Class labels are thresholds in `_classify_defect_morphology`, not a learned classifier. A different `score_threshold` changes which blobs survive.
- The clause strings are names for the pixel table above.
- No language model is configured. `simulate_rogue_vlm` sends a fixed PASS payload so the override can be seen.
- The console is one HTML file with rounded controls, a purple accent, emoji on a few buttons, and placeholder line names.
- The console, dataset, and batch database resolve from the repo root: `web/index.html`, `data/DeepPCB/PCBData`, and `data/batch_telemetry.db`.
