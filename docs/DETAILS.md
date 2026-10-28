# Implementation details

Notes behind the [DeepPCB-AOI](../README.md) front page.

## Pipeline

1. **Detect.** `core/vision` expects 640×640 inputs and does not resize. When `DETECTOR_BACKEND=yolov8`, `YOLOv8PCBDetector` stacks `[diff, test, template]` into a normalized 3-channel tensor (`[1, 3, 640, 640]`), runs the learned ONNX model, and applies class-aware NMS. In fallback mode (`DETECTOR_BACKEND=diff`), `ONNXPCBDetector` runs the 696-byte difference graph, groups high-scoring cells with connected components, labels each box from the sign of `template - test` plus geometry, and drops overlaps. No boxes means `disposition = PASS` and stages 2–4 do not run.
2. **Rules and batch history.** `core/tools` compares box area with the table below and increments a SQLite count for `(batch_id, defect_class)`.
3. **Optional model triage.** If a callable is configured, its output must validate as `InspectionDecision`. It is tried at most three times (`MAX_VLM_RETRIES = 2`). Anything else falls back to the rule engine. The default server does not call an external model. The callable does not edit box coordinates.
4. **Overrides** in `core/agent/guardrails.py`. Open and Short cannot be PASS (forced to SCRAP / CRITICAL). An over-limit box cannot be PASS. The third occurrence of the same class in a batch sets `halt_production_line`.

The editable HTML source for the pipeline diagram is [system-architecture.html](architecture/system-architecture.html) (exported to [system-architecture.svg](architecture/system-architecture.svg)). One detected box follows [ipc-decision-flowchart.html](architecture/ipc-decision-flowchart.html) ([ipc-decision-flowchart.svg](architecture/ipc-decision-flowchart.svg)).

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

## ONNX models

Two Stage 1 detector backends live in `core/vision/`:

1. **`YOLOv8PCBDetector` (`core/vision/yolo_detector.py`, activated with `DETECTOR_BACKEND=yolov8`)**:
   - Checked-in weights: `core/vision/models/deeppcb_yolov8n_3channel.onnx` (and single-image ablation baseline `core/vision/models/deeppcb_yolov8n_single.onnx`).
   - Input tensor: 3-channel composite `[template, test, absdiff]` normalized to `float32 [1, 3, 640, 640]`.
   - Post-processing: confidence filtering (`conf_threshold=0.25`) and class-aware NMS (`iou_threshold=0.45`).
2. **`ONNXPCBDetector` (`core/vision/onnx_detector.py`, fallback heuristic baseline)**:
   - Checked-in graph: `core/vision/models/deeppcb_diff_detector.onnx` (696 bytes, IR version 8, opset 13).
   - Inputs `template`, `test`: `float32 [1, 1, 640, 640]`; runs `Sub`, `Abs`, and two stride-4, kernel-4 box-average convolutions to output `diff_score` and `polarity_map` (`[1, 1, 160, 160]`).
   - `ONNXPCBDetector` synthesizes the same graph if the file is missing (`build_differential_onnx_model`). The API constructs it with `score_threshold=145` (class default is `28`).

## Benchmark & Evaluation

- **Held-out dataset evaluation (`scripts/eval_deeppcb.py`)**: Evaluates Methods A, B, and C across all 500 test pairs (`data/DeepPCB/PCBData/test.txt`, 3,140 ground-truth defects) at `IoU=0.33` and `IoU=0.50`, recording per-class Precision/Recall/F1, GPU/CPU latencies, and hardware metadata to `evaluation_results.json`.
- **Stage 1 synthetic micro-benchmark (`GET /api/benchmark`)**: Calls `detector.benchmark()` on one synthetic 640×640 pair and returns `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, and `fps`. Tests and the `sla_met` flag pass when `mean_ms < 15`.

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

`pytest` covers Pydantic schemas, the IPC pixel-area table, SQLite batch counts, guardrail overrides (including a stub model that tries to PASS a short), image upload decoding, `YOLOv8PCBDetector` 3-channel inference and fast-path bypass, and Stage 1 latency against the `<15 ms` gate. Held-out accuracy across the 500 test pairs is evaluated separately via `scripts/eval_deeppcb.py`.

## Limitations

- In the default `ONNXPCBDetector` fallback, class labels come from geometry thresholds in `_classify_defect_morphology`. Set `DETECTOR_BACKEND=yolov8` to use the learned 3-channel detector (`0.9401 F1`).
- The clause strings are names for the pixel table above.
- No language model is configured by default. `simulate_rogue_vlm` sends a fixed PASS payload so the guardrail override can be verified.
- The console, dataset, and batch database resolve from the repo root: `web/index.html`, `data/DeepPCB/PCBData`, and `data/batch_telemetry.db`.
