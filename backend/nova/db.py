from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from nova import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS datasets(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, name TEXT NOT NULL,
  path TEXT NOT NULL, rows INTEGER NOT NULL, license TEXT, report TEXT,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, engine TEXT NOT NULL,
  kind TEXT NOT NULL, config TEXT NOT NULL, status TEXT NOT NULL,
  output_dir TEXT, error TEXT, created_at REAL NOT NULL,
  started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS metrics(
  id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL, ts REAL NOT NULL,
  step INTEGER, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_metrics_job ON metrics(job_id, id);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL, ts REAL NOT NULL, line TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_logs_job ON logs(job_id, id);
"""


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        cols = [r[1] for r in c.execute("PRAGMA table_info(datasets)")]
        if "kind" not in cols:  # migración: 'text' (JSONL para LLM) o 'images' (carpetas por clase)
            c.execute("ALTER TABLE datasets ADD COLUMN kind TEXT NOT NULL DEFAULT 'text'")


def now() -> float:
    return time.time()


def rows(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def one(sql: str, args: tuple = ()) -> dict[str, Any] | None:
    r = rows(sql, args)
    return r[0] if r else None


def execute(sql: str, args: tuple = ()) -> int:
    with conn() as c:
        return c.execute(sql, args).lastrowid or 0


def dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False)
