from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Optional, Sequence

from .asr import ASRResult


def _chmod_private_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            if candidate.exists():
                os.chmod(candidate, 0o600)
        except OSError:
            pass


class SQLiteState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._chmod_private()

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
        self._chmod_private()

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
        self._chmod_private()
        self.conn.close()

    def _chmod_private(self) -> None:
        _chmod_private_sqlite_files(self.path)


class ASRResultCache:
    """SQLite-backed cache for model-major ASR passes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self._closed = False
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._chmod_private()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asr_results (
                input_path TEXT NOT NULL,
                engine TEXT NOT NULL,
                channel INTEGER NOT NULL,
                file_id TEXT,
                transcript TEXT NOT NULL,
                words_json TEXT NOT NULL,
                language TEXT,
                language_probability REAL,
                duration REAL,
                timestamp_retry_used INTEGER NOT NULL,
                timestamp_suspicious INTEGER NOT NULL,
                error TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (input_path, engine, channel)
            )
            """
        )
        self.conn.commit()

    def upsert_result(self, input_path: str | Path, engine: str, channel: int, result: ASRResult) -> None:
        self.conn.execute(
            """
            INSERT INTO asr_results (
                input_path, engine, channel, file_id, transcript, words_json, language,
                language_probability, duration, timestamp_retry_used, timestamp_suspicious,
                error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(input_path, engine, channel) DO UPDATE SET
                file_id=excluded.file_id,
                transcript=excluded.transcript,
                words_json=excluded.words_json,
                language=excluded.language,
                language_probability=excluded.language_probability,
                duration=excluded.duration,
                timestamp_retry_used=excluded.timestamp_retry_used,
                timestamp_suspicious=excluded.timestamp_suspicious,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (
                str(input_path),
                str(engine),
                int(channel),
                result.file_id,
                result.transcript or "",
                json.dumps(result.words or [], ensure_ascii=False),
                result.language,
                result.language_probability,
                result.duration,
                int(bool(result.timestamp_retry_used)),
                int(bool(result.timestamp_suspicious)),
                result.error,
                time.time(),
            ),
        )
        self.conn.commit()
        self._chmod_private()

    def upsert_results(self, rows: Sequence[tuple[str | Path, str, int, ASRResult]]) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO asr_results (
                input_path, engine, channel, file_id, transcript, words_json, language,
                language_probability, duration, timestamp_retry_used, timestamp_suspicious,
                error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(input_path, engine, channel) DO UPDATE SET
                file_id=excluded.file_id,
                transcript=excluded.transcript,
                words_json=excluded.words_json,
                language=excluded.language,
                language_probability=excluded.language_probability,
                duration=excluded.duration,
                timestamp_retry_used=excluded.timestamp_retry_used,
                timestamp_suspicious=excluded.timestamp_suspicious,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            [
                (
                    str(input_path),
                    str(engine),
                    int(channel),
                    result.file_id,
                    result.transcript or "",
                    json.dumps(result.words or [], ensure_ascii=False),
                    result.language,
                    result.language_probability,
                    result.duration,
                    int(bool(result.timestamp_retry_used)),
                    int(bool(result.timestamp_suspicious)),
                    result.error,
                    time.time(),
                )
                for input_path, engine, channel, result in rows
            ],
        )
        self.conn.commit()
        self._chmod_private()

    def has_result(self, input_path: str | Path, engine: str, channel: int, include_errors: bool = False) -> bool:
        error_clause = "" if include_errors else " AND error IS NULL"
        cur = self.conn.execute(
            f"SELECT 1 FROM asr_results WHERE input_path=? AND engine=? AND channel=?{error_clause} LIMIT 1",
            (str(input_path), str(engine), int(channel)),
        )
        return cur.fetchone() is not None

    def has_results_for_file(self, input_path: str | Path, engines: list[str], channels: list[int], include_errors: bool = False) -> bool:
        if not engines or not channels:
            return False
        engine_placeholders = ",".join("?" for _ in engines)
        channel_placeholders = ",".join("?" for _ in channels)
        error_clause = "" if include_errors else " AND error IS NULL"
        cur = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM asr_results
            WHERE input_path=?
              AND engine IN ({engine_placeholders})
              AND channel IN ({channel_placeholders})
              {error_clause}
            """,
            (str(input_path), *[str(engine) for engine in engines], *[int(channel) for channel in channels]),
        )
        return int(cur.fetchone()[0]) == len(engines) * len(channels)

    def get_results_for_file(self, input_path: str | Path) -> list[ASRResult]:
        cur = self.conn.execute(
            """
            SELECT engine, channel, file_id, transcript, words_json, language,
                   language_probability, duration, timestamp_retry_used,
                   timestamp_suspicious, error
            FROM asr_results
            WHERE input_path=?
            ORDER BY channel ASC, engine ASC
            """,
            (str(input_path),),
        )
        rows = []
        for row in cur.fetchall():
            (
                engine,
                channel,
                file_id,
                transcript,
                words_json,
                language,
                language_probability,
                duration,
                timestamp_retry_used,
                timestamp_suspicious,
                error,
            ) = row
            rows.append(ASRResult(
                channel=int(channel),
                transcript=str(transcript or ""),
                words=json.loads(words_json or "[]"),
                engine=str(engine),
                language=language,
                language_probability=language_probability,
                duration=duration,
                timestamp_retry_used=bool(timestamp_retry_used),
                timestamp_suspicious=bool(timestamp_suspicious),
                error=error,
                file_id=file_id,
            ))
        return rows

    def close(self) -> None:
        if self._closed:
            return
        self._chmod_private()
        self.conn.close()
        self._closed = True

    def delete_cache_files(self) -> None:
        self.close()
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                if path.exists():
                    path.unlink()
            except FileNotFoundError:
                pass

    def _chmod_private(self) -> None:
        _chmod_private_sqlite_files(self.path)
