# DeepPCB-AOI — Engineering & Agent Guidelines

FastAPI & ONNX Runtime bare-board PCB defect inspection and 4-stage guardrail pipeline.

## Architecture & Invariants

1. **4-Stage Pipeline & Fast-Path**:
   - Stage 1 (`core/vision`): `ONNXPCBDetector` (differential baseline) and `YOLOv8PCBDetector` (`[template, test, absdiff]` 3-channel composite). Boards with zero detections return `PASS` immediately via fast-path.
   - Stage 2 (`core/tools`): Deterministic IPC-A-600 pixel-area evaluation (`ipc_rules.py`) and SQLite batch drift counter (`batch_db.py`).
   - Stage 3 (`core/agent/state_graph.py`): Optional VLM/LLM triage validated against `InspectionDecision` with deterministic rule fallback.
   - Stage 4 (`core/agent/guardrails.py`): Hard safety overrides—`Open` and `Short` are forced to `SCRAP`, over-limit boxes cannot `PASS`, and 3 occurrences of the same defect class in a batch trigger `halt_production_line = True`.
2. **Testing Gate**:
   - Run `pytest` to verify all unit, schema, guardrail, and API tests pass before committing.

## Graphify Navigation Rules (Token Optimization)

- **When to consult Graphify**: Before making cross-module changes, modifying API contracts (`api/main.py` / `schemas.py`), or tracing pipeline dependencies, query the knowledge graph (`graphify-out/GRAPH_REPORT.md` or `graphify query "<question>"` via a lightweight `flash` subagent).
- **When to skip Graphify**: When the target file and line number are already known in context or from a traceback, read and edit the file directly.
