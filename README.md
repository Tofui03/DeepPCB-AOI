# DeepPCB-AOI

Bare-board optical inspection for the six defect classes in the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset (Tang, He, Huang, and Yang, 2019). A template image is compared with a test image. The difference is turned into boxes, each box is checked against a pixel-area limit, and a short rule set can override the final disposition.

Supports two selectable Stage 1 detector backends:
1. **`ONNXPCBDetector` (Default)**: Lightweight 696-byte hand-crafted differential baseline (`mean < 1ms` on CPU).
2. **`YOLOv8PCBDetector`**: Learned 3-channel composite model (`[template, test, absdiff]`) reaching **0.9401 F1** (0.9828 Recall, 0.9010 Precision) on the official 500-pair held-out test set. Set `DETECTOR_BACKEND=yolov8` to activate.

![Template, test image with detector boxes, and the difference map for one DeepPCB pair.](docs/inspect-loop.gif)

## Detection Benchmarks & Ablation Study

Evaluated on the official held-out DeepPCB test set (500 pairs, 3,140 ground-truth defect instances) under the standard IoU=0.33 (Tang et al. 2019) and IoU=0.50 benchmarks:

| Method | Input Format | Precision (IoU=0.33) | Recall (IoU=0.33) | F1 (IoU=0.33) | Precision (IoU=0.50) | Recall (IoU=0.50) | F1 (IoU=0.50) | Mean Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A: Differential ONNX Baseline** | Dual `template`, `test` | 0.0119 | 0.0914 | 0.0211 | 0.0046 | 0.0350 | 0.0081 | 3.16 ms (GPU) / 0.79 ms (CPU) |
| **B: YOLOv8n Single Image** | Single `<id>_test.jpg` | 0.8703 | 0.9723 | 0.9185 | 0.8677 | 0.9694 | 0.9158 | 10.38 ms (GPU) / 24.72 ms (CPU) |
| **C: YOLOv8n 3-Channel Composite** | 3-Ch `[template, test, absdiff]` | **0.9010** | **0.9828** | **0.9401** | **0.8999** | **0.9815** | **0.9389** | **9.98 ms (GPU)** / 22.15 ms (CPU) |

### Per-Class F1 Breakdown (IoU = 0.33)

| Defect Class | Ground Truth Count | Method A (Heuristic Diff) | Method B (Single Image) | Method C (3-Channel Composite) | Key Improvement Note |
| --- | ---: | ---: | ---: | ---: | --- |
| **Open** | 659 | 0.0170 | 0.9523 | **0.9335** | High recall (>98%) on circuit trace breaks |
| **Short** | 478 | 0.0135 | 0.8580 | **0.9273** | **+6.9% F1 over B**: template disambiguates layout traces from short bridges |
| **Mousebite** | 586 | 0.0826 | 0.9593 | **0.9312** | Reliable edge nick detection |
| **Spur** | 483 | 0.0745 | 0.9308 | **0.9294** | Robust against copper trace protrusions |
| **Spurious Copper** | 464 | 0.0000 | 0.9488 | **0.9625** | Method A is completely blind (0.0 F1); C achieves 99.6% recall |
| **Pin-hole** | 470 | 0.0023 | 0.8507 | **0.9631** | **+11.2% F1 over B**: template prevents normal drill vias from triggering false alarms |

### Why Method C Wins: Failure Mode Analysis
1. **Short defects without template reference**: In Method B (single image), the model achieves only 0.8580 F1 on `Short` defects because normal trace proximity can look identical to a copper bridge without the template reference. Method C leverages Channel 0 to check intended connectivity, boosting Short F1 to 0.9273.
2. **Pin-holes vs Normal Vias/Pads**: In single-image mode, isolated dark circular pads are frequently misclassified as pin-holes (Method B precision on Pin-hole is only 74.0%). In Method C, because the template board explicitly shows normal pads, Pin-hole precision jumps to 92.9% and F1 reaches 0.9631.
3. **Traditional Heuristic Failure**: Method A generates 23,760 False Positives on 500 test images (~47 FP/image) due to sub-pixel registration jitter along high-contrast copper trace edges.

Visualized failure samples and bounding box overlays are archived under `docs/bad_cases/method_a`, `docs/bad_cases/method_b`, and `docs/bad_cases/method_c`.

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

1. **Detect.** `core/vision` expects 640×640 inputs and does not resize. It runs the ONNX graph and groups high-scoring cells with connected components. Class comes from the sign of `template - test` (missing copper vs extra copper), plus area, aspect ratio, and fill. Non-maximum suppression drops overlaps. No boxes means `disposition = PASS` and stages 2–4 do not run.
2. **Rules and batch history.** `core/tools` compares box area with the table below and increments a SQLite count for `(batch_id, defect_class)`.
3. **Optional model triage.** If a callable is configured, its output must validate as `InspectionDecision`. It is tried at most three times (`MAX_VLM_RETRIES = 2`). Anything else falls back to the rule engine. The default server does not call an external model. The callable does not edit box coordinates.
4. **Overrides** in `core/agent/guardrails.py`. Open and Short cannot be PASS (forced to SCRAP / CRITICAL). An over-limit box cannot be PASS. The third occurrence of the same class in a batch sets `halt_production_line`.

![Pipeline diagram](docs/architecture/system-architecture.svg)

The editable source is `docs/architecture/system-architecture.excalidraw`.

`ONNXPCBDetector.benchmark` times Stage 1 `detect` on one synthetic 640×640 pair (two dark rectangles) and returns `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, and `fps`. Tests and `sla_met` pass when `mean_ms < 15`. That call times `detect` as a whole. It does not separately time the fast path, non-maximum suppression, SQLite, guardrails, or a VLM. This repo has no CUDA or TensorRT benchmark.

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

One detected box takes the path below.

![Decision flow for one detected box](docs/architecture/ipc-decision-flowchart.svg)

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

`core/data/mock_pcb` is a two-pair fixture so `pytest` can run without the upstream images. The server does not load it. `DeepPCBDataset()` reads `data/DeepPCB/PCBData` and raises `FileNotFoundError` when that directory is missing. Tests pass `allow_mock=True`. `tests/conftest.py` sets `DEEPPCB_ALLOW_MOCK=1` before the API import, which is the only way the fixture is selected, and only when the real tree is absent.

To run the server, or the 1,500-pair checks:

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
```

The loader expects `data/DeepPCB/PCBData/trainval.txt` (1,000 lines) and `test.txt` (500 lines), plus `*_temp.jpg`, `*_test.jpg`, and the annotation txt files. With that tree present, tests require those counts. Without it, they require the fixture's one trainval id and one test id.

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
- The console, dataset, and batch database resolve from the repo root: `web/index.html`, `data/DeepPCB/PCBData`, and `data/batch_telemetry.db`.
