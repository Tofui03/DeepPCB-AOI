# AGENTS.md — Context Engineering & System Constraints for DeepPCB-AOI

> **Methodology:** Google's *"The New SDLC With Vibe Coding: From Ad-Hoc Prompting to Agentic Engineering"*
> **Core Principle:** An industrial AI agent is **10% Model + 90% Deterministic Harness**.

---

## 1. System Intent & Domain Scope
**DeepPCB-AOI** is an industrial Automated Optical Inspection (AOI) and IPC-A-600 compliance agent designed for Surface Mount Technology (SMT) and PCB fabrication lines.

### The 6 Canonical DeepPCB Defect Classes:
1. `Open` — Broken conductive trace causing circuit discontinuity.
2. `Short` — Unintended copper bridge between traces causing electrical short.
3. `Mousebite` — Trace edge erosion / nick reducing cross-sectional width.
4. `Spur` — Sharp copper protrusion reducing dielectric clearance.
5. `Spurious Copper` — Isolated residual copper speck risking capacitive coupling or shorting.
6. `Pin-hole` — Small void inside a copper trace or pad increasing localized resistance.

---

## 2. Non-Negotiable Architectural Rules (The 90% Harness)

### Rule 1: Deterministic Vision First (No VLM Coordinate Guessing)
- All spatial localization (`x1, y1, x2, y2`), confidence scores, and pixel area measurements **MUST** be produced by the deterministic CV / ONNX Runtime layer (`core/vision/`).
- The Vision-Language Model (VLM) is strictly forbidden from inventing or modifying bounding box coordinates.

### Rule 2: Zero-Token Fast-Path Bypass (Economics of AI)
- If Stage 1 (ONNX Vision) detects `0` defects on a board, the pipeline **MUST** immediately return `disposition = "PASS"` in `<15ms` with **zero LLM/VLM calls**.

### Rule 3: Ground-Truth Tool Execution Before Reasoning
- Before the VLM/Agent makes a triage decision, the state graph **MUST** deterministically invoke:
  1. `check_ipc_rule(defect_class, area_px, relative_width_pct)` against verified IPC-A-600 Class 2/3 thresholds.
  2. `query_batch_history(batch_id, defect_class)` against the local SQLite batch telemetry store.

### Rule 4: Strict Pydantic Schema & Bounded Retry Cap
- Every agent triage output **MUST** validate against `InspectionDecision` (`core/agent/schemas.py`).
- Validation retries are strictly capped at `MAX_RETRIES = 2`. If the VLM is offline or fails validation after 2 retries, the system **MUST** fall back to the deterministic rule engine without crashing.

### Rule 5: Deterministic Python Safety Guardrails (Final Arbiter)
- After Stage 3 (VLM Triage), Stage 4 (`core/agent/guardrails.py`) enforces hard safety overrides:
  - Any `Open` or `Short` defect can **NEVER** receive `disposition = "PASS"`. If the VLM outputs `PASS`, the guardrail forces `SCRAP` (for `Short` / `Open`) and sets `guardrail_overridden = True`.
  - If `batch_occurrence_count >= 3` for the same defect class in the active batch, the guardrail forces `halt_production_line = True`.

### Rule 6: Frontend Swiss-Industrial Precision Design Mandate
- When generating or modifying `web/index.html` or any frontend UI, the agent **MUST** activate and follow `.agents/skills/industrial-ui-design/SKILL.md` and `.impeccable.md`.
- **BANNED:** Rounded SaaS cards (`border-radius: 8px/12px`), blurry drop shadows, purple/neon gradients, and emoji clutter.
- **ENFORCED:** Swiss-Industrial Machine Vision Console aesthetic (`border-radius: 0px`, 1px exposed hairline structural grid, tabular monospaced telemetry, reticle HUD viewports, and industrial hazard striping for emergency line halts).

---

## 3. Repository Boundaries & Testing Mandate
- `core/vision/` — Aligned differential preprocessing & ONNX Runtime detector.
- `core/tools/` — IPC-A-600 rule lookup & SQLite batch telemetry tracker.
- `core/agent/` — Pydantic schemas, 4-stage state graph, and safety guardrails.
- `api/` — FastAPI REST endpoints (`/api/inspect`, `/api/agent-triage`, `/api/benchmark`, `/api/batch-telemetry`).
- `web/` — Dual-pane industrial AOI web console (Vibe-Coded UI layer).
- `tests/` — Automated `pytest` verification and adversarial evaluation harness.
