# DeepPCB-AOI

An industrial bare-board Automated Optical Inspection (AOI) system combining **3-channel reference fusion (`[template, test, |template - test|]`)** with a **learned YOLOv8 ONNX detector** and **deterministic IPC-A-600 safety guardrails**.

On the official held-out [DeepPCB](https://github.com/tangsanli5201/DeepPCB) test set (500 board pairs, 3,140 ground-truth defects), the 3-channel composite detector achieves **0.9401 F1 (98.28% Recall, 90.10% Precision)** at **9.95 ms mean GPU latency** (`NVIDIA GeForce RTX 5060 Laptop GPU`) and **34.13 ms CPU latency** (`Intel Core Ultra 7 255HX`), cutting single-image false alarms on drill vias and dense copper traces by over 25%.

![Template, test image with detector boxes, and the difference map for one DeepPCB pair.](docs/inspect-loop.gif)

---

## Benchmark & Ablation Study

Evaluated on the official 500-pair held-out DeepPCB test set (`data/DeepPCB/PCBData/test.txt`, 3,140 defect instances) under both the standard **IoU = 0.33** benchmark (Tang et al., 2019) and **IoU = 0.50**. Hardware: **NVIDIA GeForce RTX 5060 Laptop GPU** (`device=0`) and **Intel Core Ultra 7 255HX** (`device=cpu`). All metrics are directly traceable to [evaluation_results.json](evaluation_results.json) via [scripts/eval_deeppcb.py](scripts/eval_deeppcb.py):

| Method | Input Tensor | Precision (IoU=0.33) | Recall (IoU=0.33) | **F1 (IoU=0.33)** | Precision (IoU=0.50) | Recall (IoU=0.50) | F1 (IoU=0.50) | Mean Latency (GPU / CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Morphological Diff Baseline** | Dual `template`, `test` | 0.0119 | 0.0914 | 0.0211 | 0.0046 | 0.0350 | 0.0081 | 3.85 ms / **3.64 ms** |
| **B: YOLOv8n Single Image** | 1-Ch `test` | 0.8703 | 0.9723 | 0.9185 | 0.8677 | 0.9694 | 0.9158 | 15.59 ms / 37.01 ms |
| **C: YOLOv8n 3-Channel (Ours)** | 3-Ch `[template, test, absdiff]` | **0.9010** | **0.9828** | **0.9401** | **0.8999** | **0.9815** | **0.9389** | **9.95 ms** / 34.13 ms |

### Per-Class F1 Breakdown & Failure Mode Analysis (IoU = 0.33)

| Defect Class | GT Count | Method A (Heuristic Diff) | Method B (Single Image) | **Method C (3-Ch Composite)** | Causal Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pin-hole** | 470 | 0.0023 | 0.8507 | **0.9631 (+11.2%)** | Without the template board, Method B misclassifies normal circular drill vias as pin-holes (74.0% precision). Channel 0 (`template`) exposes intended vias, raising precision to **92.9%** at **100% recall**. |
| **Short** | 478 | 0.0135 | 0.8580 | **0.9273 (+6.9%)** | Dense parallel traces resemble copper bridges in single-image views. Template reference disambiguates intended connectivity, lifting Short precision from 80.7% to **89.7%**. |
| **Spurious Copper** | 464 | 0.0000 | 0.9488 | **0.9625 (+1.4%)** | Channel 2 (`absdiff`) isolates stray copper islands on bare substrate, achieving **99.6% recall**. |
| **Open** | 659 | 0.0170 | **0.9523** | 0.9335 | High recall (**98.0%**) across broken conductor traces. |
| **Mousebite** | 586 | 0.0826 | **0.9593** | 0.9312 | Reliable trace-edge nick localization (**98.1% recall**). |
| **Spur** | 483 | 0.0745 | **0.9308** | 0.9294 | Detects sharp copper protrusions (**98.1% recall**). |

- **Why Heuristic Diff (Method A) Fails on Real Boards**: Sub-pixel registration jitter along high-contrast trace edges triggers **23,760 false positives** across 500 test boards (~47.5 FP/board).
- Visualized failure overlays for all three methods are archived in [`docs/bad_cases/`](docs/bad_cases).

---

## Quick Start

### 1. Install & Run Web Inspection Console

Requires Python 3.10+. Both trained ONNX models (`deeppcb_yolov8n_3channel.onnx` and `deeppcb_yolov8n_single.onnx`) are checked into `core/vision/models/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
DETECTOR_BACKEND=yolov8 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` to inspect boards interactively.

### 2. Reproduce Benchmark Numbers

The production server runs inference purely on `onnxruntime` (included in `requirements.txt`). To run the full 3-method evaluation harness (`scripts/eval_deeppcb.py`), install `ultralytics` and specify `--device 0` (GPU) or `--device cpu`:

```bash
pip install ultralytics
python3 scripts/eval_deeppcb.py --method all --device 0 \
  --weights-b core/vision/models/deeppcb_yolov8n_single.onnx \
  --weights-c core/vision/models/deeppcb_yolov8n_3channel.onnx \
  --pcb-root data/DeepPCB/PCBData \
  --output-json evaluation_results_gpu.json
```

### 3. Run Unit & Integration Tests (87 Tests)

```bash
pytest
```

Tests run without the upstream dataset via the built-in two-pair fixture in `core/data/mock_pcb`.

---

## System Architecture & Design Decisions

![DeepPCB-AOI System Architecture](docs/architecture/system-architecture.svg)

1. **Why 3-Channel Input Fusion (`[template, test, |template - test|]`)**: Subtracting images alone destroys structural context (is a bright diff pixel on a pad, a trace edge, or bare FR4?). Conversely, feeding only the test image forces the detector to guess which holes and bridges were part of the CAD layout. Stacking `[T, T', |Δ|]` into a standard 3-channel tensor lets a lightweight YOLOv8n backbone jointly attend to layout geometry and differential anomalies without custom multi-branch encoders.
2. **Zero-Token Fast-Path Bypass**: In Stage 1 (`core/vision`), if no bounding boxes survive confidence and class-aware NMS filtering (or if `template` and `test` buffers match), the pipeline immediately returns `disposition = PASS` without invoking rule checks, SQLite writes, or external triage.
3. **Deterministic IPC-A-600 Guardrails Over Model Decisions**: Neural detectors and optional VLM triage (`core/agent/state_graph.py`) cannot override hard manufacturing safety rules (`core/agent/guardrails.py`):
   - **Zero-Tolerance Classes**: `Open` and `Short` defects are unconditionally forced to `SCRAP / CRITICAL` (0 px maximum allowed area).
   - **Pixel-Area Thresholding**: `Mousebite`, `Spur`, `Spurious Copper`, and `Pin-hole` are checked against Class 2 / Class 3 pixel-area caps (`core/tools/ipc_rules.py`); violations force `REWORK / HIGH`.
   - **3-Strike Batch Line Halt**: Each defect increments a persistent SQLite counter (`core/tools/batch_db.py`) keyed by `(batch_id, defect_class)`. The 3rd occurrence of the same defect class in a batch sets `halt_production_line = True`.

![Decision flow for one detected box](docs/architecture/ipc-decision-flowchart.svg)

---

## Limitations

1. **Pre-Aligned Board Assumption**: The 3-channel tensor assumes `template` and `test` images are already spatially registered (as in DeepPCB). On an uncalibrated conveyor camera with rotational or thermal scale drift, an ECC or ORB homography alignment step must run before channel fusion.
2. **Binarized Grayscale Domain Gap**: DeepPCB images are thresholded binary/grayscale bare-board scans (`640×640`). Production solder-mask (green/red/black FR4) and HASL/ENIG copper finishes introduce specular glare and color variance that require illumination normalization or RGB-pair retraining.
3. **Pixel-Area Proxy vs. True IPC-A-600 Geometry**: `core/tools/ipc_rules.py` approximates IPC-A-600 Class 2/3 limits using bounding-box pixel area rather than calibrated physical units ($\mu\text{m}/\text{pixel}$) and skeletonized minimum conductor width / spacing measurements.

---

## Documentation & Artifacts

- **Implementation & API Reference**: [docs/DETAILS.md](docs/DETAILS.md)
- **Interactive Architecture Spec**: [docs/architecture/system-architecture.html](docs/architecture/system-architecture.html)
- **Interactive IPC Decision Flowchart**: [docs/architecture/ipc-decision-flowchart.html](docs/architecture/ipc-decision-flowchart.html)
- **Bad-Case Visual Galleries**: [docs/bad_cases/](docs/bad_cases)
