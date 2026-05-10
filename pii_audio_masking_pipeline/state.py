from __future__ import annotations

from pathlib import Path
from typing import Optional
import sqlite3
import time


class SQLiteState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_status (
                input_path TEXT PRIMARY KEY,
                output_path TEXT,
                status TEXT NOT NULL,
                error TEXT,
                duration_sec REAL,
                num_words INTEGER,
                num_entities INTEGER,
                num_spans INTEGER,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, input_path: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT input_path, output_path, status, error, duration_sec, num_words, num_entities, num_spans, updated_at FROM file_status WHERE input_path=?",
            (input_path,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = ["input_path", "output_path", "status", "error", "duration_sec", "num_words", "num_entities", "num_spans", "updated_at"]
        return dict(zip(keys, row))

    def upsert(
        self,
        input_path: str,
        output_path: str | None,
        status: str,
        error: str | None = None,
        duration_sec: float | None = None,
        num_words: int | None = None,
        num_entities: int | None = None,
        num_spans: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO file_status (
                input_path, output_path, status, error, duration_sec, num_words, num_entities, num_spans, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(input_path) DO UPDATE SET
                output_path=excluded.output_path,
                status=excluded.status,
                error=excluded.error,
                duration_sec=excluded.duration_sec,
                num_words=excluded.num_words,
                num_entities=excluded.num_entities,
                num_spans=excluded.num_spans,
                updated_at=excluded.updated_at
            """,
            (
                input_path,
                output_path,
                status,
                error,
                duration_sec,
                num_words,
                num_entities,
                num_spans,
                time.time(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
