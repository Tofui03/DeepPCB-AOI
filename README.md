# DeepPCB-AOI

High-speed PCB defect detection and automated IPC-A-600 inspection pipeline. Combines ONNX edge vision preprocessing with rule-based geometric verification to detect, classify, and triage manufacturing defects under 15ms.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)]()

---

## Overview

Traditional Automated Optical Inspection (AOI) setups suffer from high false-alarm rates and slow inference. **DeepPCB-AOI** uses a dual-engine architecture:

1. **Edge Preprocessing (`< 4ms`)**: Aligned template subtraction and morphological operations identify defect candidates using ONNX Runtime. Defect-free boards bypass further inspection immediately.
2. **IPC-A-600 Geometric Rule Engine**: Candidate regions of interest (ROIs) are measured against IPC Class 2 and Class 3 tolerances (pinholes, conductor reduction, clearance violations, and shorts).
3. **Structured Triage & Guardrails**: Standardized output schemas with hard overrides that enforce line-halt triggers on recurring defects or zero-tolerance electrical faults (`Open` / `Short`).

```
                    +-----------------------------+
                    |  Input (Golden + Test Pair) |
                    +--------------+--------------+
                                   |
                                   v
             [Stage 1: ONNX Differential Detector (< 4ms)]
                                   |
                  +----------------+----------------+
                  | No defects                      | Defect candidates
                  v                                 v
          +---------------+         [Stage 2: IPC-A-600 Rule Engine]
          | Immediate PASS|                         |
          +---------------+                         v
                                    [Stage 3: Decision & Guardrails]
                                                    |
                                                    v
                                    +-------------------------------+
                                    | Inspection Report / Halt Line |
                                    +-------------------------------+
```

---

## Performance Benchmark

Evaluated on the **DeepPCB** dataset ($640 \times 640$ image pairs):

| Metric | Measured | Target SLA |
| :--- | :---: | :---: |
| **Edge Latency (p50)** | **3.65 ms** | $< 15.0\text{ ms}$ |
| **Edge Latency (p95)** | **4.76 ms** | $< 15.0\text{ ms}$ |
| **Throughput** | **263 FPS** | $> 60\text{ FPS}$ |
| **Defect Classes** | `Open`, `Short`, `Mousebite`, `Spur`, `Copper`, `Pinhole` | 6 Canonical Types |
| **Test Suite** | **81 passing** | Unit, Adversarial, Integration |

---

## Project Structure

```
├── core/
│   ├── vision/          # Template subtraction, morphology, ONNX inference
│   ├── tools/           # IPC-A-600 Class 2/3 rules, SQLite batch tracking
│   └── agent/           # Inspection schemas, deterministic guardrails
├── api/                 # FastAPI inspection service endpoints
├── web/                 # Interactive inspection console (dual-pane viewer)
├── tests/               # 81 automated pytest suites
└── data/                # Dataset loaders and split verification
```

---

## Quickstart

### 1. Prerequisites & Installation

```bash
git clone https://github.com/Tofui03/DeepPCB-AOI.git
cd DeepPCB-AOI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest -v
```

### 3. Launch Local Inspection Server & UI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to view the inspection console, toggle between IPC Class 2 and Class 3 thresholds, and inspect test pairs.

---

## License

MIT License.
