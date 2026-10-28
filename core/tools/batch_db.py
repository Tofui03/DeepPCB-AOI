"""SQLite counts of defect classes per manufacturing batch.

``threshold_exceeded`` is true once ``occurrence_count`` reaches ``BATCH_HALT_THRESHOLD`` (3).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import get_args

from core.agent.schemas import BatchTrendResult, DefectClass

DEFAULT_BATCH_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "batch_telemetry.db"
BATCH_HALT_THRESHOLD: int = 3
_VALID_DEFECT_CLASSES: set[str] = set(get_args(DefectClass))


class BatchTelemetryStore:
    """Thread-safe SQLite store tracking defect occurrences per manufacturing batch."""

    def __init__(self, db_path: str | Path = DEFAULT_BATCH_DB_PATH) -> None:
        self.db_target: str = (
            ":memory:" if str(db_path) == ":memory:" else str(Path(db_path))
        )
        if self.db_target != ":memory:":
            Path(self.db_target).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_target, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the `batch_defect_counts` table if it does not already exist."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_defect_counts (
                    batch_id TEXT NOT NULL,
                    defect_class TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (batch_id, defect_class)
                )
                """
            )

    @staticmethod
    def _validate_inputs(
        batch_id: str,
        defect_class: DefectClass | None = None,
        increment: int = 1,
    ) -> str:
        cleaned_batch_id = batch_id.strip()
        if not cleaned_batch_id:
            raise ValueError("batch_id must be a non-empty string.")
        if defect_class is not None and defect_class not in _VALID_DEFECT_CLASSES:
            raise ValueError(
                f"Unsupported defect_class={defect_class!r}. "
                f"Expected one of {sorted(_VALID_DEFECT_CLASSES)}."
            )
        if increment < 0:
            raise ValueError(f"increment must be >= 0, got {increment}.")
        return cleaned_batch_id

    def record_and_query_batch(
        self,
        batch_id: str,
        defect_class: DefectClass,
        increment: int = 1,
    ) -> BatchTrendResult:
        """Atomically increment `(batch_id, defect_class)` count and return `BatchTrendResult`."""
        cleaned_batch_id = self._validate_inputs(batch_id, defect_class, increment)

        with self._conn:
            if increment > 0:
                self._conn.execute(
                    """
                    INSERT INTO batch_defect_counts (batch_id, defect_class, occurrence_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(batch_id, defect_class)
                    DO UPDATE SET
                        occurrence_count = occurrence_count + excluded.occurrence_count,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (cleaned_batch_id, defect_class, increment),
                )

            cursor = self._conn.execute(
                """
                SELECT occurrence_count
                FROM batch_defect_counts
                WHERE batch_id = ? AND defect_class = ?
                """,
                (cleaned_batch_id, defect_class),
            )
            row = cursor.fetchone()

        count = int(row[0]) if row is not None else 0
        return BatchTrendResult(
            batch_id=cleaned_batch_id,
            defect_class=defect_class,
            occurrence_count=count,
            threshold_exceeded=count >= BATCH_HALT_THRESHOLD,
        )

    def query_batch_history(
        self,
        batch_id: str,
        defect_class: DefectClass,
    ) -> BatchTrendResult:
        """Read-only lookup of current occurrence count for `(batch_id, defect_class)`."""
        return self.record_and_query_batch(
            batch_id=batch_id,
            defect_class=defect_class,
            increment=0,
        )

    def get_batch_summary(self, batch_id: str) -> list[BatchTrendResult]:
        """Return all recorded `BatchTrendResult` entries for `batch_id` ordered by defect_class."""
        cleaned_batch_id = self._validate_inputs(batch_id)

        cursor = self._conn.execute(
            """
            SELECT defect_class, occurrence_count
            FROM batch_defect_counts
            WHERE batch_id = ?
            ORDER BY defect_class ASC
            """,
            (cleaned_batch_id,),
        )
        rows = cursor.fetchall()

        return [
            BatchTrendResult(
                batch_id=cleaned_batch_id,
                defect_class=row[0],
                occurrence_count=int(row[1]),
                threshold_exceeded=int(row[1]) >= BATCH_HALT_THRESHOLD,
            )
            for row in rows
        ]

    def reset_batch(self, batch_id: str) -> None:
        """Delete all telemetry records for `batch_id`."""
        cleaned_batch_id = self._validate_inputs(batch_id)
        with self._conn:
            self._conn.execute(
                "DELETE FROM batch_defect_counts WHERE batch_id = ?",
                (cleaned_batch_id,),
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
