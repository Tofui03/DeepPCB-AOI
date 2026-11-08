# PCB Inspection Viewer & DeepPCB-AOI — Implementation Progress & Scope Guard

> **Owner:** Gwenfred Tham  
> **Spec Version:** 1.0 (5 Nov 2026)  
> **Backend Repo:** `github.com/Tofui03/DeepPCB-AOI` (FastAPI)  
> **Client Repo:** `github.com/Tofui03/pcb-inspection-viewer` (C# .NET 8 WPF, MVVM)

---

## 0. Strict Scope Guardrails (Never Violate)

### Core Principles
1. **The rule engine decides; the LLM only explains and queries.**
   - Client never recomputes or overrides `board_disposition` (`PASS` / `REWORK` / `SCRAP`).
   - All LLM output must carry an `"AI-generated"` label.
   - The app must be 100% functional when `LLM Provider = None` or offline (using deterministic template fallback).
2. **Grounding:**
   - Only use fields actually returned by `DeepPCB-AOI` (`api/main.py`, `core/agent/schemas.py`, `core/tools/batch_db.py`, `core/tools/ipc_rules.py`).
   - Any new backend fields must be tracked in Phase 1 (`§9 Backend Changes Required`) and remain nullable/backward-compatible on the client.
3. **No Secrets in Repo:**
   - API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) read from environment variables or `.NET User Secrets` only — never in `appsettings.json` or git history.

### Explicitly Out of Scope (§13 — Do Not Build)
- [x] ❌ Camera / PLC / MES hardware integration, real-time line triggering, or barcode reading.
- [x] ❌ Training or changing the detection model, editing bounding boxes, or overriding verdicts in the UI.
- [x] ❌ Multi-user server, authentication, or cloud database.
- [x] ❌ Certified IPC-A-600 compliance claims or copying copyrighted IPC-A-600 standard text verbatim.
- [x] ❌ Cross-platform UI frameworks (Avalonia/MAUI) or MSI installers (self-contained `.zip` publish only).
- [x] ❌ Mobile or new web clients.

---

## Phase 1: Backend Additions in `DeepPCB-AOI` (Days 1–2)

*Goal: Make `DeepPCB-AOI` container-ready, able to boot without the full DeepPCB dataset mounted, and expose `/api/health` and decision-to-detection links.*

- [x] **Step 1.1 — Server starts without full DeepPCB dataset (`BC-5`, Must)**
  - [x] Update [`core/data/deeppcb_loader.py`](core/data/deeppcb_loader.py) / [`api/main.py`](api/main.py) so missing `data/DeepPCB/PCBData` does not raise `FileNotFoundError` at import time (gracefully fall back to `core/data/mock_pcb` or empty dataset list).
  - [x] Ensure `POST /api/upload-triage` works completely without the dataset volume.
- [x] **Step 1.2 — Health Endpoint (`BC-1`, Must)**
  - [x] Add `GET /api/health` to [`api/main.py`](api/main.py) returning `{status, version, detector_backend, dataset_loaded}`.
- [x] **Step 1.3 — Decision-to-Detection Link & Payload Options (`BC-2` & `BC-3`, Should)**
  - [x] `BC-2`: Add optional `detection_index: int | None = None` (and/or `defect_class`) to [`InspectionDecision`](core/agent/schemas.py) and populate it inside [`PCBAOIAgentGraph.run_inspection_with_trace`](core/agent/state_graph.py).
  - [x] `BC-3`: Add `return_images: bool = Form(default=True)` to `POST /api/upload-triage` so batch runs can optionally skip base64 PNG encoding.
  - [x] Bump FastAPI version in [`api/main.py`](api/main.py) to `1.2.0`.
- [x] **Step 1.4 — Reproducible `requirements.txt` & `Dockerfile` (`BC-8`, Must)**
  - [x] Pin real, installable headless PyPI versions in [`requirements.txt`](requirements.txt) (`opencv-python-headless`, `onnxruntime`, `fastapi`, `uvicorn`, `numpy`, `pydantic`, `python-multipart`).
  - [x] Create `.dockerignore` (exclude `.venv`, `data/DeepPCB`, `.git`, caches; keep `core/vision/models/deeppcb_yolov8n_3channel.onnx` and `core/data/mock_pcb`).
  - [x] Create [`Dockerfile`](Dockerfile) (`python:3.12-slim`, `ENV DETECTOR_BACKEND=yolov8`, `EXPOSE 8000`, `HEALTHCHECK` hitting `/api/health`).
- [x] **Step 1.5 — Backend Verification & Fixture Capture**
  - [x] Add unit tests in `tests/` for `/api/health`, dataset-less startup, `detection_index`, and `return_images=False`.
  - [x] Run `pytest` and verify 100% pass.
  - [ ] Build Docker image & verify `curl http://localhost:8000/api/health` returns `200 OK`. (Skipped locally: docker daemon not installed on host, containerized CI will execute this)
  - [x] Capture real `POST /api/upload-triage` JSON response from `core/data/mock_pcb/group20085/20085000_test.jpg` for client fixture testing.

---

## Phase 2: Milestone D1–3 — Solution Skeleton, DI Host, Settings & Docker Compose

*Acceptance Criteria: `docker compose up` serves the API; `curl /api/health` returns 200; WPF app launches, Settings saved; status pill green/red works; CI build passes on `windows-latest`.*

- [ ] **Step 2.1 — Repository & Solution Setup**
  - [ ] Initialize `pcb-inspection-viewer` repo with `.gitignore` (covering `appsettings.*.local.json`, `*.db`, `cache/`, `logs/`, `bin/`, `obj/`).
  - [ ] Create `Directory.Build.props` (`Nullable=enable`, `TreatWarningsAsErrors=true`, `LangVersion=latest`).
  - [ ] Create `PcbInspectionViewer.sln`:
    - [ ] `src/PcbInspectionViewer.Core` (`net8.0`, no WPF references)
    - [ ] `src/PcbInspectionViewer.App` (`net8.0-windows`, WPF)
    - [ ] `tests/PcbInspectionViewer.Core.Tests` (`net8.0`, xUnit + FluentAssertions + NSubstitute)
    - [ ] `tests/PcbInspectionViewer.App.Tests` (`net8.0-windows`, xUnit)
    - [ ] `tests/PcbInspectionViewer.IntegrationTests` (`net8.0`, xUnit)
- [ ] **Step 2.2 — Docker Compose (`backend/`)**
  - [ ] Add `backend/docker-compose.yml` mapping `8000:8000`, `DETECTOR_BACKEND=yolov8`, optional volume `./data/DeepPCB:/app/data/DeepPCB:ro`, `restart: unless-stopped`.
- [ ] **Step 2.3 — DI, Logging & Settings (`F7`)**
  - [ ] Configure `Microsoft.Extensions.Hosting` in `App.xaml.cs`.
  - [ ] Configure `Serilog` rolling daily file sink in `%LOCALAPPDATA%\PcbInspectionViewer\logs` (never log API keys or full prompts at `Information` level).
  - [ ] Implement Options classes (`ApiOptions`, `LlmOptions`, `BatchOptions`) + Settings UI screen (`F7`).
  - [ ] Implement 30s background health check polling + bottom status bar pill (`Backend online` green / `Backend offline` red + detector/API version).
- [ ] **Step 2.4 — Initial CI Workflow**
  - [ ] Add `.github/workflows/build-test.yml` (`windows-latest`, `dotnet build -warnaserror`, `dotnet test`) and `.github/workflows/gitleaks.yml`.

---

## Phase 3: Milestone D4–7 — Single-Image Inspection (`F1`) & Overlay Canvas

*Acceptance Criteria: Inspecting mock board `20085000` shows boxes aligned to defects at any window size; verdict equals server `board_disposition`; Cancel works mid-request; API-down shows error not crash; ≥ 15 unit tests.*

- [ ] **Step 3.1 — Typed API Client & DTOs**
  - [ ] Create exact DTOs in `PcbInspectionViewer.Core/Api/Dtos/` matching `/api/upload-triage` response (`BoardInspectionReport`, `BoundingBoxDetection`, `InspectionDecision`, `BatchTrendResult`).
  - [ ] Implement `IInspectionApiClient` / `InspectionApiClient` with `HttpClient` + Polly resilience (retry up to 2 times with jitter on connect/502/503/504; no retry on 400/422).
- [ ] **Step 3.2 — Inspect View & ViewModel (`F1`)**
  - [ ] Inputs panel: Golden template picker (remembered per session), test image picker (file dialog + drag-and-drop for `.jpg/.jpeg/.png/.bmp`), `batch_id`, `IPC Class 2/3`, `Inspect` & `Cancel` buttons.
  - [ ] Keyboard shortcuts: `Ctrl+O` open, `F5` inspect, `Esc` cancel.
  - [ ] Right panel: Verdict badge (`PASS` green, `REWORK` amber, `SCRAP` red + text/icon), `line_halted` banner, Stage-1 latency + round-trip ms, and `InspectionDecision` list (severity, IPC clause, risk, root cause, corrective action, guardrail override reason).
- [ ] **Step 3.3 — `DefectOverlayCanvas` Control**
  - [ ] Scale 640×640 bounding box coordinates accurately to displayed image geometry at any window size / DPI.
  - [ ] Support zoom/pan and layer toggles (`Test` / `Template` / `Diff Heatmap` / `Boxes`).
  - [ ] Colour-code boxes per defect class with `<Class> <confidence:0.00>` labels.
- [ ] **Step 3.4 — Unit Tests (D4–7)**
  - [ ] `InspectionApiClient` tests with fake `HttpMessageHandler` + recorded `tests/fixtures/upload_triage_sample.json`.
  - [ ] `InspectViewModel` tests (command disable during run, cancellation token trigger, exact verdict mapping, API error handling).
  - [ ] Overlay geometry 640→display scaling and zoom math unit tests.

---

## Phase 4: Milestone D8–10 — Batch Runner (`F2`), EF Core SQLite (`F6`) & History (`F3`)

*Acceptance Criteria: 100-image batch completes with UI responsive; failed items recorded; restart app and history persists; filter by class+verdict correct; drill-down reopens stored result without calling API.*

- [ ] **Step 4.1 — EF Core SQLite Data Model (`§5.5` & `F6`)**
  - [ ] Database path: `%LOCALAPPDATA%\PcbInspectionViewer\inspections.db` (`BusyTimeout = 5s`).
  - [ ] Entities: `Batch`, `Inspection`, `Defect`, `Decision`, `LlmSummary`, `QueryLog` (P2), `KbChunk` (P3).
  - [ ] Indexes on `Inspection(InspectedAt)`, `Inspection(BatchId)`, `Inspection(BoardDisposition)`, `Defect(DefectClass)`.
  - [ ] Create read-only SQL views (`v_inspections`, `v_defects`, `v_batches`) excluding raw LLM payloads and local file paths.
- [ ] **Step 4.2 — Local Image Cache (`§6.4`)**
  - [ ] Implement `IImageCache`: decode returned `test_b64` and `diff_b64` PNG data URIs and save to `%LOCALAPPDATA%\PcbInspectionViewer\cache\<sha256>.png`.
  - [ ] Add `"Open data folder"` menu item.
- [ ] **Step 4.3 — Bounded-Concurrency Batch Runner (`F2`)**
  - [ ] Implement `IBatchRunner` using `SemaphoreSlim` / `Parallel.ForEachAsync` (default concurrency 2, configurable 1–8).
  - [ ] Folder scanner: skip template file and `*_temp.*` files; support optional DeepPCB auto-pairing mode (`<id>_test.jpg` ↔ `<id>_temp.jpg`).
  - [ ] Report progress (`n / N`, ETA, per-item status) via `IProgress<BatchProgress>` at least every 250 ms.
  - [ ] Graceful per-item error capture (undecodable file, timeout, 4xx/5xx) without aborting the batch; `Cancel` finishes in-flight requests and marks remaining as `Cancelled`.
- [ ] **Step 4.4 — History Table & Drill-Down (`F3`)**
  - [ ] Virtualized `DataGrid` (smooth scrolling at ≥10k rows): time (local), file name, batch, verdict, #defects, classes, latency.
  - [ ] Filters: verdict, defect class (multi-select), batch, date range, filename search.
  - [ ] Double-click row → opens `F1` result view from SQLite + local image cache without calling the backend API.

---

## Phase 5: Milestone D11–12 — Dashboard (`F4`), CSV/PDF Export (`F5`) & P1 LLM Summary (`§4.2`)

*Acceptance Criteria: KPIs match SQL hand-check; CSV opens in Excel with Chinese intact; PDF has KPIs, chart, table, labelled summary; with no key → template summary; with key → valid JSON summary whose numbers match facts; validator tests green.*

- [ ] **Step 5.1 — Daily Dashboard (`F4`)**
  - [ ] Implement `IDashboardService` aggregating by chosen local day: boards inspected, yield (`PASS / completed`, excluding errors), `REWORK` & `SCRAP` counts, top 3 defect classes (bar chart), hourly throughput (line chart), and halted batches.
  - [ ] Build Dashboard UI view using `LiveCharts2` (or `OxyPlot`).
- [ ] **Step 5.2 — CSV & PDF Export (`F5`)**
  - [ ] CSV (`CsvHelper`): export per-inspection or per-defect with current filter applied, encoded as **UTF-8 with BOM** (Excel-compatible for English + 中文).
  - [ ] PDF (`QuestPDF` Community licence): batch/day report with header, KPIs, chart image, table, and optional LLM summary explicitly marked `"AI-generated explanation; verdicts are rule-based"`.
- [ ] **Step 5.3 — P1 LLM Inspection Summary (`§4.2`, Must)**
  - [ ] Implement `ILlmProvider` (`GeminiProvider`, `OpenAiProvider`, `NullProvider`) with `Temperature = 0.2`, `Timeout = 30s`.
  - [ ] Build aggregated facts input (counts by verdict, yield, counts by defect class, 3×3 board grid centroid distribution, top corrective actions, halted flags, IPC class, batch id, time range — **never raw images**).
  - [ ] Implement `SummaryValidator`:
    - [ ] Strip markdown code fences and strictly parse JSON against `InspectionSummary v1` schema.
    - [ ] **Numeric consistency check:** every number in `headline` and `yield_statement` must match an input fact (within `±0.1` for percentages).
    - [ ] **Verdict contradiction check:** reject claims like `"all boards passed"` when `SCRAP > 0` or `REWORK > 0`.
  - [ ] Implement retry coordinator: up to 3 technical retries (timeout, 5xx, 429) with exponential backoff + up to 2 semantic re-prompts appending the validation error.
  - [ ] Implement `TemplateSummaryProvider` deterministic fallback (EN & 中文 templates) when provider is `None`, offline, or retries are exhausted.
  - [ ] Persist to `LlmSummary` table and show source badge in UI.

---

## Phase 6: Milestone D13–14 — Hardening, Integration CI, Docs & P2 NL→SQL (`§4.3`)

*Acceptance Criteria: Core coverage ≥ 70%; all workflows green; README quick start works on a clean Windows machine; (P2) SqlGuard suite ≥ 30 cases passes, SQL shown before run, write attempts impossible.*

- [ ] **Step 6.1 — P2 Natural-Language Query over History (`§4.3`, Should)**
  - [ ] Implement `ISqlGuard` (enforced in C# code, not prompt):
    1. Separate SQLite connection with `Mode=ReadOnly` + `PRAGMA query_only = ON;`.
    2. SQL tokenizer validation: single statement only; must start with `SELECT` or `WITH ... SELECT`; reject `;` followed by anything, `ATTACH`, `PRAGMA`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `load_extension`, and SQL comments (`--`, `/*`).
    3. Allowlisted views only: `v_inspections`, `v_defects`, `v_batches`.
    4. Named parameters only (max 10, string length ≤ 100).
    5. Wrap query as `SELECT * FROM (<query>) LIMIT 500` with 2-second execution timeout (`SqliteCommand.Cancel`).
    6. Audit every attempt to `QueryLog`.
  - [ ] Build Ask UI screen (`4. Ask`): question box, read-only generated SQL preview with Copy button, explicit `Run` button (auto-run off by default), results grid, and optional LLM explanation.
  - [ ] Add table-driven `SqlGuardTests` with **≥ 30 test cases**.
- [ ] **Step 6.2 — Cross-Platform Integration Tests & Linux CI**
  - [ ] Build `PcbInspectionViewer.IntegrationTests` testing `PcbInspectionViewer.Core` against the live Dockerized `DeepPCB-AOI` backend using `core/data/mock_pcb/group20085`.
  - [ ] Add `.github/workflows/integration.yml` (`ubuntu-latest`, builds backend Docker image, starts container, waits for `/api/health`, runs integration tests).
- [ ] **Step 6.3 — Documentation, Benchmarks & Release Polish**
  - [ ] Measure real latency & batch throughput on local hardware and record in `docs/benchmarks.md`.
  - [ ] Capture UI screenshots (`docs/screenshots/`) and record <2-minute demo GIF (`docs/demo.gif`).
  - [ ] Write `docs/architecture.md` and short ADRs in `docs/adr/`.
  - [ ] Complete `README.md` following `§11.1` outline.

---

## Optional / Post-Sprint Backlog (Only After D1–14 Are 100% Green)
- [ ] **P3 RAG over Defect-Handling Knowledge Base (`§4.4`, Could):** `kb/*.md` authored in own words, chunked by heading, embedded or BM25-indexed in `KbChunk`, inline citation verification.
- [ ] **Backend `BC-4` / `BC-6` / `BC-7` (Could):** Telemetry idempotency key, total pipeline latency ms field, structured error code schema.
- [ ] **Release workflow (`release.yml`):** `dotnet publish` self-contained `win-x64` zip on git tag.
