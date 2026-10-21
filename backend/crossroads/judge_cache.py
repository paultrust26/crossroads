"""SQLite cache for LLM judge results.

Stores judge outputs so re-runs and page refreshes don't re-invoke the model.
DB lives in CROSSROADS_CACHE_DIR (default ~/.cache/crossroads).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .settings import CACHE_DIR

_DB_PATH = CACHE_DIR / "judge_cache.db"
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS judge_results (
                session_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                judge_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def get_cached_judge(session_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT judge_json FROM judge_results WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def save_judge_result(session_id: str, source: str, judge_dict: dict) -> None:
    if judge_dict.get("error"):
        return
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO judge_results (session_id, source, judge_json, created_at) VALUES (?, ?, ?, ?)",
        (session_id, source, json.dumps(judge_dict), time.time()),
    )
    conn.commit()


def get_all_cached(source: Optional[str] = None) -> dict[str, dict]:
    conn = _get_conn()
    if source:
        rows = conn.execute(
            "SELECT session_id, judge_json FROM judge_results WHERE source = ? ORDER BY created_at DESC",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_id, judge_json FROM judge_results ORDER BY created_at DESC"
        ).fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def clear_cache() -> int:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM judge_results")
    conn.commit()
    return cur.rowcount
