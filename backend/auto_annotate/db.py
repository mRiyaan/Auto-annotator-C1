"""Separate SQLite persistence for Roboflow Auto-Annotation (never touches studio.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_annotate import config as aa_config
from auto_annotate.schemas import (
    AutoBatchSummary,
    AutoDetection,
    AutoItemRecord,
    AutoJobRecord,
    RejectReason,
    ReviewStatus,
)

_AUTO_DB_LOCK = threading.Lock()


def _connect(db_path: Path = aa_config.AUTO_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_auto_db(db_path: Path = aa_config.AUTO_DB_PATH) -> None:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_jobs (
                    id TEXT PRIMARY KEY,
                    created_at_iso TEXT NOT NULL,
                    status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    internal_meta_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_batches (
                    job_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    batch_index INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    start_index INTEGER NOT NULL,
                    end_index INTEGER NOT NULL,
                    checked INTEGER NOT NULL DEFAULT 0,
                    checked_at_iso TEXT,
                    PRIMARY KEY (job_id, batch_id),
                    FOREIGN KEY (job_id) REFERENCES auto_jobs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_items (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    batch_index INTEGER NOT NULL,
                    seq_index INTEGER NOT NULL,
                    roboflow_image_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    split TEXT NOT NULL DEFAULT 'train',
                    source_batch_id TEXT,
                    existing_annotations_count INTEGER NOT NULL DEFAULT 0,
                    inference_status TEXT NOT NULL DEFAULT 'pending',
                    detections_json TEXT NOT NULL DEFAULT '[]',
                    latency_sec REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    reject_reason TEXT,
                    push_status TEXT NOT NULL DEFAULT 'not_pushed',
                    push_error TEXT,
                    pushed_at_iso TEXT,
                    FOREIGN KEY (job_id) REFERENCES auto_jobs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_auto_items_job_seq ON auto_items(job_id, seq_index);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_auto_items_job_batch ON auto_items(job_id, batch_id, seq_index);"
            )
            conn.commit()


def _row_to_item(row: sqlite3.Row) -> AutoItemRecord:
    raw_dets = json.loads(row["detections_json"] or "[]")
    dets = [AutoDetection.model_validate(d) for d in raw_dets]
    return AutoItemRecord(
        id=row["id"],
        jobId=row["job_id"],
        batchId=row["batch_id"],
        batchIndex=int(row["batch_index"]),
        seqIndex=int(row["seq_index"]),
        roboflowImageId=row["roboflow_image_id"],
        filename=row["filename"],
        imageUrl=row["image_url"],
        width=int(row["width"]),
        height=int(row["height"]),
        split=row["split"],
        sourceBatchId=row["source_batch_id"],
        existingAnnotationsCount=int(row["existing_annotations_count"] or 0),
        inferenceStatus=row["inference_status"],
        detections=dets,
        latencySec=float(row["latency_sec"] or 0.0),
        errorMessage=row["error_message"],
        reviewStatus=row["review_status"],
        rejectReason=row["reject_reason"],
        pushStatus=row["push_status"],
        pushError=row["push_error"],
        pushedAtIso=row["pushed_at_iso"],
    )


def _hydrate_job(conn: sqlite3.Connection, row: sqlite3.Row) -> AutoJobRecord:
    job = AutoJobRecord.model_validate_json(row["record_json"])
    batch_rows = conn.execute(
        "SELECT * FROM auto_batches WHERE job_id = ? ORDER BY batch_index ASC",
        (job.id,),
    ).fetchall()
    item_rows = conn.execute(
        "SELECT * FROM auto_items WHERE job_id = ? ORDER BY seq_index ASC",
        (job.id,),
    ).fetchall()

    batch_stats: Dict[str, Dict[str, int]] = {}
    reason_counts: Dict[str, int] = {
        "missed_object": 0,
        "false_detection": 0,
        "bad_box": 0,
        "double_box": 0,
    }
    total = len(item_rows)
    annotated = 0
    accepted = 0
    rejected = 0
    pending = 0
    pushed = 0

    for it in item_rows:
        bid = it["batch_id"]
        st = batch_stats.setdefault(
            bid,
            {"annotated": 0, "accepted": 0, "rejected": 0, "pending": 0, "pushed": 0},
        )
        if it["inference_status"] == "completed":
            annotated += 1
            st["annotated"] += 1
        rs = it["review_status"]
        if rs == "accepted":
            accepted += 1
            st["accepted"] += 1
        elif rs == "rejected":
            rejected += 1
            st["rejected"] += 1
            rr = it["reject_reason"]
            if rr in reason_counts:
                reason_counts[rr] += 1
        else:
            pending += 1
            st["pending"] += 1
        if it["push_status"] == "pushed":
            pushed += 1
            st["pushed"] += 1

    batches: List[AutoBatchSummary] = []
    checked_batches = 0
    for br in batch_rows:
        bid = br["batch_id"]
        st = batch_stats.get(
            bid,
            {"annotated": 0, "accepted": 0, "rejected": 0, "pending": 0, "pushed": 0},
        )
        is_checked = bool(br["checked"])
        if is_checked:
            checked_batches += 1
        b_total = int(br["end_index"]) - int(br["start_index"]) + 1
        batches.append(
            AutoBatchSummary(
                batchId=bid,
                batchIndex=int(br["batch_index"]),
                label=br["label"],
                startIndex=int(br["start_index"]),
                endIndex=int(br["end_index"]),
                total=max(b_total, st["accepted"] + st["rejected"] + st["pending"]),
                annotated=st["annotated"],
                accepted=st["accepted"],
                rejected=st["rejected"],
                pending=st["pending"],
                pushed=st["pushed"],
                checked=is_checked,
                checkedAtIso=br["checked_at_iso"],
            )
        )

    job.totalImages = total
    job.annotatedCount = annotated
    job.acceptedCount = accepted
    job.rejectedCount = rejected
    job.pendingCount = pending
    job.pushedCount = pushed
    job.checkedBatchesCount = checked_batches
    job.batches = batches
    job.rejectReasonCounts = reason_counts
    return job


def upsert_job(
    job: AutoJobRecord,
    internal_meta: Optional[Dict[str, Any]] = None,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> None:
    payload_json = job.model_dump_json()
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            if internal_meta is None:
                existing = conn.execute(
                    "SELECT internal_meta_json FROM auto_jobs WHERE id = ?", (job.id,)
                ).fetchone()
                meta_str = existing["internal_meta_json"] if existing else "{}"
            else:
                meta_str = json.dumps(internal_meta)

            conn.execute(
                """
                INSERT INTO auto_jobs (id, created_at_iso, status, record_json, internal_meta_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at_iso = excluded.created_at_iso,
                    status = excluded.status,
                    record_json = excluded.record_json,
                    internal_meta_json = excluded.internal_meta_json
                """,
                (job.id, job.createdAtIso, job.status, payload_json, meta_str),
            )
            conn.commit()


def get_job(
    job_id: str, db_path: Path = aa_config.AUTO_DB_PATH
) -> Optional[AutoJobRecord]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM auto_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            return _hydrate_job(conn, row)


def get_job_internal_meta(
    job_id: str, db_path: Path = aa_config.AUTO_DB_PATH
) -> Dict[str, Any]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT internal_meta_json FROM auto_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row or not row["internal_meta_json"]:
                return {}
            return json.loads(row["internal_meta_json"])


def list_jobs(
    limit: int = 100, db_path: Path = aa_config.AUTO_DB_PATH
) -> List[AutoJobRecord]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM auto_jobs ORDER BY created_at_iso DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [_hydrate_job(conn, r) for r in rows]


def delete_job(job_id: str, db_path: Path = aa_config.AUTO_DB_PATH) -> bool:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            cur = conn.execute("DELETE FROM auto_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0


def insert_batches(
    job_id: str,
    batches: List[Dict[str, Any]],
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> None:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO auto_batches (
                    job_id, batch_id, batch_index, label, start_index, end_index, checked, checked_at_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        b["batchId"],
                        int(b["batchIndex"]),
                        b["label"],
                        int(b["startIndex"]),
                        int(b["endIndex"]),
                        1 if b.get("checked") else 0,
                        b.get("checkedAtIso"),
                    )
                    for b in batches
                ],
            )
            conn.commit()


def insert_items(
    items: List[Dict[str, Any]], db_path: Path = aa_config.AUTO_DB_PATH
) -> None:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO auto_items (
                    id, job_id, batch_id, batch_index, seq_index,
                    roboflow_image_id, filename, local_path, image_url,
                    width, height, split, source_batch_id, existing_annotations_count,
                    inference_status, detections_json, latency_sec, error_message,
                    review_status, reject_reason, push_status, push_error, pushed_at_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        it["id"],
                        it["jobId"],
                        it["batchId"],
                        int(it["batchIndex"]),
                        int(it["seqIndex"]),
                        it["roboflowImageId"],
                        it["filename"],
                        it["localPath"],
                        it["imageUrl"],
                        int(it["width"]),
                        int(it["height"]),
                        it.get("split", "train"),
                        it.get("sourceBatchId"),
                        int(it.get("existingAnnotationsCount", 0)),
                        it.get("inferenceStatus", "pending"),
                        json.dumps(it.get("detections", [])),
                        float(it.get("latencySec", 0.0)),
                        it.get("errorMessage"),
                        it.get("reviewStatus", "pending"),
                        it.get("rejectReason"),
                        it.get("pushStatus", "not_pushed"),
                        it.get("pushError"),
                        it.get("pushedAtIso"),
                    )
                    for it in items
                ],
            )
            conn.commit()


def list_items(
    job_id: str,
    *,
    batch_id: Optional[str] = None,
    review_status: Optional[str] = None,
    reject_reason: Optional[str] = None,
    detection_filter: Optional[str] = None,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> List[AutoItemRecord]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            clauses = ["job_id = ?"]
            params: List[Any] = [job_id]
            if batch_id and batch_id != "all":
                clauses.append("batch_id = ?")
                params.append(batch_id)
            if review_status and review_status != "all":
                clauses.append("review_status = ?")
                params.append(review_status)
            if reject_reason and reject_reason != "all":
                clauses.append("reject_reason = ?")
                params.append(reject_reason)

            sql = f"SELECT * FROM auto_items WHERE {' AND '.join(clauses)} ORDER BY seq_index ASC"
            rows = conn.execute(sql, tuple(params)).fetchall()
            items = [_row_to_item(r) for r in rows]

    if detection_filter == "with_detections":
        items = [it for it in items if len(it.detections) > 0]
    elif detection_filter == "no_detections":
        items = [it for it in items if len(it.detections) == 0]
    return items


def get_item_with_path(
    job_id: str, item_id: str, db_path: Path = aa_config.AUTO_DB_PATH
) -> Optional[Dict[str, Any]]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM auto_items WHERE job_id = ? AND id = ?",
                (job_id, item_id),
            ).fetchone()
            if not row:
                return None
            return {"item": _row_to_item(row), "localPath": row["local_path"]}


def get_all_items_with_paths(
    job_id: str, db_path: Path = aa_config.AUTO_DB_PATH
) -> List[Dict[str, Any]]:
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM auto_items WHERE job_id = ? ORDER BY seq_index ASC",
                (job_id,),
            ).fetchall()
            return [
                {"item": _row_to_item(r), "localPath": r["local_path"]}
                for r in rows
            ]


def update_item_inference(
    item_id: str,
    *,
    inference_status: str,
    detections: List[AutoDetection],
    latency_sec: float = 0.0,
    error_message: Optional[str] = None,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> None:
    det_json = json.dumps([d.model_dump() for d in detections])
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            conn.execute(
                """
                UPDATE auto_items
                SET inference_status = ?, detections_json = ?, latency_sec = ?, error_message = ?
                WHERE id = ?
                """,
                (inference_status, det_json, latency_sec, error_message, item_id),
            )
            conn.commit()


def update_item_review(
    job_id: str,
    item_id: str,
    *,
    review_status: ReviewStatus,
    reject_reason: Optional[RejectReason] = None,
    detections: Optional[List[AutoDetection]] = None,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> Optional[Dict[str, Any]]:
    clean_reason = reject_reason if review_status == "rejected" else None
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            if detections is not None:
                det_json = json.dumps([d.model_dump() for d in detections])
                conn.execute(
                    """
                    UPDATE auto_items
                    SET review_status = ?, reject_reason = ?, detections_json = ?,
                        push_status = 'not_pushed', push_error = NULL
                    WHERE job_id = ? AND id = ?
                    """,
                    (review_status, clean_reason, det_json, job_id, item_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE auto_items
                    SET review_status = ?, reject_reason = ?
                    WHERE job_id = ? AND id = ?
                    """,
                    (review_status, clean_reason, job_id, item_id),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM auto_items WHERE job_id = ? AND id = ?",
                (job_id, item_id),
            ).fetchone()
            if not row:
                return None
            return {"item": _row_to_item(row), "localPath": row["local_path"]}


def bulk_approve_batch_pending(
    job_id: str,
    batch_id: str,
    *,
    accept_pending: bool = True,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Approve a batch without overriding rejected items:
    - If `accept_pending=True`, updates ONLY items with `review_status == 'pending'` -> 'accepted'.
    - Explicitly rejected items in this batch REMAIN 'rejected'.
    - Marks the batch `checked = 1`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            if accept_pending:
                conn.execute(
                    """
                    UPDATE auto_items
                    SET review_status = 'accepted', reject_reason = NULL
                    WHERE job_id = ? AND batch_id = ? AND review_status = 'pending'
                    """,
                    (job_id, batch_id),
                )
            conn.execute(
                """
                UPDATE auto_batches
                SET checked = 1, checked_at_iso = ?
                WHERE job_id = ? AND batch_id = ?
                """,
                (now_iso, job_id, batch_id),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM auto_items WHERE job_id = ? AND batch_id = ? ORDER BY seq_index ASC",
                (job_id, batch_id),
            ).fetchall()
            return [
                {"item": _row_to_item(r), "localPath": r["local_path"]}
                for r in rows
            ]


def mark_items_push_status(
    item_ids: List[str],
    *,
    push_status: str,
    push_error: Optional[str] = None,
    db_path: Path = aa_config.AUTO_DB_PATH,
) -> None:
    if not item_ids:
        return
    now_iso = datetime.now(timezone.utc).isoformat() if push_status == "pushed" else None
    with _AUTO_DB_LOCK:
        with _connect(db_path) as conn:
            conn.executemany(
                """
                UPDATE auto_items
                SET push_status = ?, push_error = ?, pushed_at_iso = ?
                WHERE id = ?
                """,
                [(push_status, push_error, now_iso, iid) for iid in item_ids],
            )
            conn.commit()
