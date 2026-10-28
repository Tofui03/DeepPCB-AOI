# Implementation details

Notes behind the [DeepPCB-AOI](../README.md) front page.

## Pipeline

1. **Detect.** `core/vision` expects 640×640 inputs and does not resize. It runs the ONNX graph and groups high-scoring cells with connected components. Class comes from the sign of `template - test` (missing copper vs extra copper), plus area, aspect ratio, and fill. Non-maximum suppression drops overlaps. No boxes means `disposition = PASS` and stages 2–4 do not run.
2. **Rules and batch history.** `core/tools` compares box area with the table below and increments a SQLite count for `(batch_id, defect_class)`.
3. **Optional model triage.** If a callable is configured, its output must validate as `InspectionDecision`. It is tried at most three times (`MAX_VLM_RETRIES = 2`). Anything else falls back to the rule engine. The default server does not call an external model. The callable does not edit box coordinates.
4. **Overrides** in `core/agent/guardrails.py`. Open and Short cannot be PASS (forced to SCRAP / CRITICAL). An over-limit box cannot be PASS. The third occurrence of the same class in a batch sets `halt_production_line`.

The editable source for the pipeline diagram is [system-architecture.excalidraw](architecture/system-architecture.excalidraw). One detected box follows [ipc-decision-flowchart.svg](architecture/ipc-decision-flowchart.svg).

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

The checked-in ONNX file is a fixed difference graph. `core/vision/models/deeppcb_diff_detector.onnx` is 696 bytes, small enough to keep in normal git. The graph is IR version 8, opset 13:

- inputs `template`, `test`: float32 `[1, 1, 640, 640]`
- `Sub`, then `Abs`
- two stride-4, kernel-4 convolutions (box average)
- outputs `diff_score`, `polarity_map`: float32 `[1, 1, 160, 160]`

`ONNXPCBDetector` writes the same graph if the file is missing (`build_differential_onnx_model`). Nothing in this repo trains weights.

The API constructs the detector with `score_threshold=145`. The class default is 28. Cells below the threshold are ignored before connected components.

## Benchmark

`ONNXPCBDetector.benchmark` times Stage 1 `detect` on one synthetic 640×640 pair (two dark rectangles) and returns `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, and `fps`. `GET /api/benchmark` calls that method and returns those latency percentiles. Tests and the `sla_met` flag pass when `mean_ms < 15`. That call times `detect` as a whole. It does not separately time the fast path, non-maximum suppression, SQLite, guardrails, or a VLM. This repo has no CUDA or TensorRT benchmark.

The ~0.79 ms mean in the README results table is this benchmark on an Intel Xeon.

## HTTP routes

`GET /`, `GET /api/samples`, `POST /api/inspect`, `POST /api/agent-triage` (alias `POST /api/triage`), `POST /api/upload-triage`, `GET /api/benchmark`, `GET /api/batch-telemetry`, `POST /api/batch-reset`.

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

## Dataset and fixture

`core/data/mock_pcb` is a two-pair fixture so `pytest` can run without the upstream images. The server does not load it. `DeepPCBDataset()` reads `data/DeepPCB/PCBData` and raises `FileNotFoundError` when that directory is missing. Tests pass `allow_mock=True`. `tests/conftest.py` sets `DEEPPCB_ALLOW_MOCK=1` before the API import, which is the only way the fixture is selected, and only when the real tree is absent.

To run the server, or the 1,500-pair checks:

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
```

The loader expects `data/DeepPCB/PCBData/trainval.txt` (1,000 lines) and `test.txt` (500 lines), plus `*_temp.jpg`, `*_test.jpg`, and the annotation txt files. With that tree present, tests require those counts. Without it, they require the fixture's one trainval id and one test id.

## SQLite

SQLite state defaults to `data/batch_telemetry.db` (gitignored). Tests use `:memory:` or a temp file.

## Design choices

- Boxes, scores, and areas are computed in `core/vision`. Later stages only read them.
- An empty detection list returns immediately. The optional model is not called.
- The area check and the batch counter run before that model call.
- Model output that fails `InspectionDecision` is retried twice, then replaced by the rule engine.
- Open, Short, and a repeated class are overridden in Python after triage. `guardrail_overridden` records that.

## Tests

Coverage is the schema, the area table, batch counts, guardrail overrides (including a stub model that tries to PASS a short), upload decoding, and Stage 1 latency against the gate above. There is no held-out accuracy eval.

## Limitations

- Class labels are thresholds in `_classify_defect_morphology`, not a learned classifier. A different `score_threshold` changes which blobs survive.
- The clause strings are names for the pixel table above.
- No language model is configured. `simulate_rogue_vlm` sends a fixed PASS payload so the override can be seen.
- The console, dataset, and batch database resolve from the repo root: `web/index.html`, `data/DeepPCB/PCBData`, and `data/batch_telemetry.db`.
