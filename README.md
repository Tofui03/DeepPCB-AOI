# DeepPCB-AOI

DeepPCB-AOI compares a template image with a test image for bare-board inspection of the six defect classes in the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset (Tang, He, Huang, and Yang, 2019). An ONNX Runtime difference detector proposes boxes, pixel-area rules check each box, a short rule set can override the disposition, and a FastAPI app serves a web console. Class labels come from a geometry heuristic on the difference image rather than a trained model.

![Template, test image with detector boxes, and the difference map for one DeepPCB pair.](docs/inspect-loop.gif)

## Quick start

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`.

```bash
pytest
```

Tests run without the dataset via a small fixture in `core/data/mock_pcb`, used only when `data/DeepPCB/PCBData` is absent.

## Results

| Metric | Value |
| --- | --- |
| Stage 1 mean latency | ~0.79 ms on an Intel Xeon CPU (synthetic 640×640 pair) |
| Stage 1 test gate | 15 ms mean |
| Defect classes | 6 |
| Tests | 85 passing |

## How it works

1. **Detect.** `core/vision` runs the ONNX difference graph on 640×640 inputs, groups high-scoring cells into boxes, labels each box from the sign of `template - test` plus area, aspect ratio, and fill, and drops overlaps; no boxes means `disposition = PASS` and stages 2–4 do not run.
2. **Rules and batch history.** `core/tools` checks each box against a pixel-area limit and increments a SQLite count for `(batch_id, defect_class)`.
3. **Optional model triage.** A configured callable must return a valid `InspectionDecision` or the rule engine is used; the default server calls no external model, and the callable does not edit box coordinates.
4. **Overrides.** `core/agent/guardrails.py` blocks PASS for Open, Short, and an over-limit box, and sets `halt_production_line` on the third occurrence of the same class in a batch.

![Pipeline diagram](docs/architecture/system-architecture.svg)

One detected box takes the path below.

![Decision flow for one detected box](docs/architecture/ipc-decision-flowchart.svg)

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

## Limitations

- Class labels come from a geometry heuristic rather than a trained classifier.
- Pixel-area limits stand in for an IPC-A-600 table. The check does not measure conductor width or spacing.
- This repository has no accuracy evaluation on held-out labels.
- No external model is configured by default.

Implementation details: [docs/DETAILS.md](docs/DETAILS.md).
