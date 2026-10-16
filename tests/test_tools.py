"""Unit tests for Stage 2 Ground-Truth Agent Tools (`ipc_rules.py` and `batch_db.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.schemas import BoundingBoxDetection, DefectClass  # noqa: E402
from core.tools.batch_db import BatchTelemetryStore  # noqa: E402
from core.tools.ipc_rules import (  # noqa: E402
    IPC_A_600_SPEC_TABLE,
    check_ipc_rule,
    evaluate_detection_ipc,
    get_ipc_metadata,
)


class TestIPCRulesEngine:
    """Tests for IPC-A-600 Class 2 and Class 3 deterministic rule lookup."""

    @pytest.mark.parametrize(
        "defect_class",
        [
            "Open",
            "Short",
            "Mousebite",
            "Spur",
            "Spurious Copper",
            "Pin-hole",
        ],
    )
    def test_spec_table_covers_all_six_classes_and_metadata(
        self, defect_class: DefectClass
    ) -> None:
        """Every canonical defect class must have Class 2/3 specs and fallback engineering metadata."""
        assert defect_class in IPC_A_600_SPEC_TABLE
        meta = get_ipc_metadata(defect_class)
        assert meta["ipc_clause"].startswith("IPC-A-600 Section 2.2.")
        assert len(meta["electrical_risk"]) > 10
        assert len(meta["root_cause"]) > 10
        assert len(meta["corrective_action"]) > 10

    @pytest.mark.parametrize("defect_class", ["Open", "Short"])
    @pytest.mark.parametrize("ipc_class", ["Class 2", "Class 3"])
    @pytest.mark.parametrize("measured_px", [1, 16, 400])
    def test_open_and_short_always_violate_ipc(
        self,
        defect_class: DefectClass,
        ipc_class: str,
        measured_px: int,
    ) -> None:
        """Open and Short have max_allowed_px=0 and must ALWAYS violate IPC-A-600."""
        res = check_ipc_rule(
            defect_class=defect_class,
            measured_px=measured_px,
            ipc_class=ipc_class,  # type: ignore[arg-type]
        )
        assert res.max_allowed_px == 0
        assert res.violates_ipc is True
        assert res.measured_px == measured_px

    @pytest.mark.parametrize(
        "defect_class",
        ["Mousebite", "Spur", "Spurious Copper", "Pin-hole"],
    )
    def test_class3_strictly_tighter_than_class2_and_exact_boundaries(
        self, defect_class: DefectClass
    ) -> None:
        """Class 3 max_allowed_px must be strictly smaller than Class 2, with exact boundary behavior."""
        c2_limit = int(IPC_A_600_SPEC_TABLE[defect_class]["Class 2"]["max_allowed_px"])
        c3_limit = int(IPC_A_600_SPEC_TABLE[defect_class]["Class 3"]["max_allowed_px"])

        assert 0 < c3_limit < c2_limit

        # Boundary test on Class 2: measured_px == max_allowed_px -> False; +1 -> True
        at_c2 = check_ipc_rule(defect_class, measured_px=c2_limit, ipc_class="Class 2")
        above_c2 = check_ipc_rule(
            defect_class, measured_px=c2_limit + 1, ipc_class="Class 2"
        )
        assert at_c2.violates_ipc is False
        assert above_c2.violates_ipc is True

        # Boundary test on Class 3: measured_px == max_allowed_px -> False; +1 -> True
        at_c3 = check_ipc_rule(defect_class, measured_px=c3_limit, ipc_class="Class 3")
        above_c3 = check_ipc_rule(
            defect_class, measured_px=c3_limit + 1, ipc_class="Class 3"
        )
        assert at_c3.violates_ipc is False
        assert above_c3.violates_ipc is True

        # A defect of size (c3_limit + 1) passes Class 2 but violates Class 3
        mid_size = c3_limit + 1
        assert check_ipc_rule(defect_class, mid_size, "Class 2").violates_ipc is False
        assert check_ipc_rule(defect_class, mid_size, "Class 3").violates_ipc is True

    def test_evaluate_detection_ipc_helper(self) -> None:
        """evaluate_detection_ipc should pass bbox.area_px into check_ipc_rule."""
        # 12 x 12 = 144 px Spur -> <= 200 (Class 2 pass), > 100 (Class 3 violation)
        bbox = BoundingBoxDetection(
            x1=100, y1=100, x2=112, y2=112, defect_class="Spur", confidence=0.91
        )
        assert bbox.area_px == 144

        res_c2 = evaluate_detection_ipc(bbox, ipc_class="Class 2")
        res_c3 = evaluate_detection_ipc(bbox, ipc_class="Class 3")
        assert res_c2.violates_ipc is False
        assert res_c3.violates_ipc is True


class TestBatchTelemetryStore:
    """Tests for SQLite BatchTelemetryStore persistence, isolation, and threshold_exceeded logic."""

    def test_occurrence_transition_2_to_3_and_summary(self) -> None:
        """occurrence_count=1,2 must have threshold_exceeded=False; count>=3 must be True."""
        store = BatchTelemetryStore(":memory:")

        r1 = store.record_and_query_batch("LOT-2026-A", "Short")
        assert r1.occurrence_count == 1
        assert r1.threshold_exceeded is False

        r2 = store.record_and_query_batch("LOT-2026-A", "Short")
        assert r2.occurrence_count == 2
        assert r2.threshold_exceeded is False

        r3 = store.record_and_query_batch("LOT-2026-A", "Short")
        assert r3.occurrence_count == 3
        assert r3.threshold_exceeded is True

        store.record_and_query_batch("LOT-2026-A", "Pin-hole")
        summary = store.get_batch_summary("LOT-2026-A")
        assert len(summary) == 2
        by_class = {item.defect_class: item for item in summary}
        assert by_class["Short"].occurrence_count == 3
        assert by_class["Short"].threshold_exceeded is True
        assert by_class["Pin-hole"].occurrence_count == 1
        assert by_class["Pin-hole"].threshold_exceeded is False
        store.close()

    def test_multi_batch_isolation_and_reset(self) -> None:
        """Counts in one batch must not bleed into another batch, and reset_batch must clear only target batch."""
        store = BatchTelemetryStore(":memory:")

        for _ in range(3):
            store.record_and_query_batch("BATCH-ALPHA", "Open")
        store.record_and_query_batch("BATCH-BETA", "Open")

        alpha_status = store.query_batch_history("BATCH-ALPHA", "Open")
        beta_status = store.query_batch_history("BATCH-BETA", "Open")
        assert alpha_status.occurrence_count == 3
        assert alpha_status.threshold_exceeded is True
        assert beta_status.occurrence_count == 1
        assert beta_status.threshold_exceeded is False

        store.reset_batch("BATCH-ALPHA")
        assert store.get_batch_summary("BATCH-ALPHA") == []
        assert len(store.get_batch_summary("BATCH-BETA")) == 1
        store.close()

    def test_file_persistence_and_parameterized_sql_safety(self, tmp_path: Path) -> None:
        """File-backed SQLite DB should persist across store instances and safely parameterize special characters."""
        db_file = tmp_path / "batch_telemetry.db"
        tricky_batch_id = "LOT-99'; DROP TABLE batch_defect_counts; --"

        store1 = BatchTelemetryStore(db_file)
        store1.record_and_query_batch(tricky_batch_id, "Mousebite", increment=2)
        store1.close()

        store2 = BatchTelemetryStore(db_file)
        res = store2.record_and_query_batch(tricky_batch_id, "Mousebite", increment=1)
        assert res.batch_id == tricky_batch_id
        assert res.occurrence_count == 3
        assert res.threshold_exceeded is True
        store2.close()
