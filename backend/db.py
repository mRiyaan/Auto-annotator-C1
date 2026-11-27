"""SQLite persistence layer for runs, telemetry, and video re-encoding metadata."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from schemas import RunRecord

_DB_LOCK = threading.Lock()


def _connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = config.DB_PATH) -> None:
    with _DB_LOCK:
        with _connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at_iso TEXT NOT NULL,
                    status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    internal_meta_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.commit()


def upsert_run(
    record: RunRecord,
    internal_meta: Optional[Dict[str, Any]] = None,
    db_path: Path = config.DB_PATH,
) -> None:
    payload_json = record.model_dump_json()
    with _DB_LOCK:
        with _connect(db_path) as conn:
            if internal_meta is None:
                existing = conn.execute(
                    "SELECT internal_meta_json FROM runs WHERE id = ?", (record.id,)
                ).fetchone()
                meta_str = existing["internal_meta_json"] if existing else "{}"
            else:
                meta_str = json.dumps(internal_meta)

            conn.execute(
                """
                INSERT INTO runs (id, created_at_iso, status, record_json, internal_meta_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at_iso = excluded.created_at_iso,
                    status = excluded.status,
                    record_json = excluded.record_json,
                    internal_meta_json = excluded.internal_meta_json
                """,
                (
                    record.id,
                    record.createdAtIso,
                    record.status,
                    payload_json,
                    meta_str,
                ),
            )
            conn.commit()


def get_run(run_id: str, db_path: Path = config.DB_PATH) -> Optional[RunRecord]:
    with _DB_LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT record_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None
            return RunRecord.model_validate_json(row["record_json"])


def get_internal_meta(
    run_id: str, db_path: Path = config.DB_PATH
) -> Dict[str, Any]:
    with _DB_LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT internal_meta_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row or not row["internal_meta_json"]:
                return {}
            return json.loads(row["internal_meta_json"])


def list_runs(
    limit: int = 200, db_path: Path = config.DB_PATH
) -> List[RunRecord]:
    with _DB_LOCK:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT record_json FROM runs ORDER BY created_at_iso DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [RunRecord.model_validate_json(r["record_json"]) for r in rows]


def delete_run(run_id: str, db_path: Path = config.DB_PATH) -> bool:
    with _DB_LOCK:
        with _connect(db_path) as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return cur.rowcount > 0
