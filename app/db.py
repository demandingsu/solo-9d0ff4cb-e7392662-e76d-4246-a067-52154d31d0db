"""Database access for the reconciliation app (SQLite via the stdlib driver).

The app deliberately uses plain SQL instead of an ORM.  That keeps the Docker
image small, makes the schema easy to inspect, and lets every write happen in an
explicit transaction so that a failed publish leaves the previous published
snapshot untouched.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    -- Monotonically increasing input version.  Any upload that can change the
    -- result of a calculation bumps this value, which invalidates previews.
    input_version   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS revisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    input_version   INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    kind            TEXT    NOT NULL CHECK (kind IN ('mapping','line_forecast','published')),
    filename        TEXT,
    content_hash    TEXT    NOT NULL,
    payload_json    TEXT    NOT NULL,
    UNIQUE (dataset_id, input_version, kind, content_hash)
);

CREATE TABLE IF NOT EXISTS previews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    input_version   INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    request_json    TEXT    NOT NULL,
    result_json     TEXT    NOT NULL,
    superseded      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    input_version   INTEGER NOT NULL,
    parent_id       INTEGER REFERENCES snapshots(id),
    result_json     TEXT    NOT NULL,
    UNIQUE (dataset_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_revisions_dataset ON revisions(dataset_id, input_version);
CREATE INDEX IF NOT EXISTS idx_previews_dataset ON previews(dataset_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_dataset ON snapshots(dataset_id, version_number);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at      TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    message         TEXT    NOT NULL,
    detail_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_dataset ON events(dataset_id, id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Dataset:
    id: int
    name: str
    created_at: str
    input_version: int


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, roll back on error."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                if conn.in_transaction:
                    conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def create_dataset(self, name: str) -> Dataset:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO datasets (name, created_at, input_version) VALUES (?,?,?)",
                (name, utc_now(), 1),
            )
            dataset_id = int(cur.lastrowid)
        return Dataset(dataset_id, name, utc_now(), 1)

    def get_dataset(self, dataset_id: int, conn: sqlite3.Connection | None = None) -> Dataset | None:
        if conn is not None:
            row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        else:
            with self.connect() as own_conn:
                row = own_conn.execute(
                    "SELECT * FROM datasets WHERE id=?", (dataset_id,)
                ).fetchone()
        if row is None:
            return None
        return Dataset(row["id"], row["name"], row["created_at"], row["input_version"])

    def list_datasets(self) -> list[Dataset]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        return [Dataset(r["id"], r["name"], r["created_at"], r["input_version"]) for r in rows]

    def bump_input_version(self, conn: sqlite3.Connection, dataset_id: int) -> int:
        conn.execute(
            "UPDATE datasets SET input_version = input_version + 1 WHERE id=?",
            (dataset_id,),
        )
        conn.execute(
            "UPDATE previews SET superseded=1 WHERE dataset_id=? AND superseded=0",
            (dataset_id,),
        )
        row = conn.execute(
            "SELECT input_version FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        return int(row["input_version"])

    # ------------------------------------------------------------------
    # Revisions (latest input of each kind within an input version)
    # ------------------------------------------------------------------
    def add_revision(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        input_version: int,
        kind: str,
        filename: str | None,
        content_hash: str,
        payload: Any,
    ) -> int:
        cur = conn.execute(
            """INSERT OR IGNORE INTO revisions
               (dataset_id, input_version, created_at, kind, filename, content_hash, payload_json)
               VALUES (?,?,?,?,?,?,?)""",
            (
                dataset_id,
                input_version,
                utc_now(),
                kind,
                filename,
                content_hash,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = conn.execute(
            """SELECT id FROM revisions
               WHERE dataset_id=? AND input_version=? AND kind=? AND content_hash=?""",
            (dataset_id, input_version, kind, content_hash),
        ).fetchone()
        return int(row["id"])

    def latest_revisions(
        self, dataset_id: int, conn: sqlite3.Connection | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return the newest revision of each kind for a dataset."""
        query = "SELECT * FROM revisions WHERE dataset_id=? ORDER BY id DESC"
        if conn is not None:
            rows = conn.execute(query, (dataset_id,)).fetchall()
        else:
            with self.connect() as own_conn:
                rows = own_conn.execute(query, (dataset_id,)).fetchall()
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["kind"] in latest:
                continue
            latest[row["kind"]] = {
                "id": row["id"],
                "input_version": row["input_version"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "content_hash": row["content_hash"],
                "payload": json.loads(row["payload_json"]),
            }
        return latest

    def log_event(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        kind: str,
        message: str,
        detail: Any | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO events (dataset_id, created_at, kind, message, detail_json) VALUES (?,?,?,?,?)",
            (
                dataset_id,
                utc_now(),
                kind,
                message,
                None if detail is None else json.dumps(detail, ensure_ascii=False, default=str),
            ),
        )

    def list_events(self, dataset_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
                (dataset_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "kind": r["kind"],
                "message": r["message"],
                "detail": json.loads(r["detail_json"]) if r["detail_json"] else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Previews
    # ------------------------------------------------------------------
    def create_preview(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        input_version: int,
        request: Any,
        result: Any,
    ) -> int:
        cur = conn.execute(
            """INSERT INTO previews (dataset_id, input_version, created_at, request_json, result_json)
               VALUES (?,?,?,?,?)""",
            (
                dataset_id,
                input_version,
                utc_now(),
                json.dumps(request, ensure_ascii=False, sort_keys=True),
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            ),
        )
        return int(cur.lastrowid)

    def get_preview(self, preview_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM previews WHERE id=?", (preview_id,)).fetchone()
            if row is None:
                return None
            # Is the input still the newest version for this dataset?
            ds = conn.execute(
                "SELECT input_version FROM datasets WHERE id=?", (row["dataset_id"],)
            ).fetchone()
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "input_version": row["input_version"],
            "created_at": row["created_at"],
            "superseded": bool(row["superseded"]),
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]),
            "current_input_version": ds["input_version"] if ds else None,
        }

    def latest_preview(self, dataset_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM previews WHERE dataset_id=? ORDER BY id DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
        return self.get_preview(row["id"]) if row else None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def publish(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        input_version: int,
        result: Any,
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) AS v FROM snapshots WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        version_number = int(row["v"]) + 1
        parent = conn.execute(
            "SELECT id FROM snapshots WHERE dataset_id=? ORDER BY version_number DESC LIMIT 1",
            (dataset_id,),
        ).fetchone()
        cur = conn.execute(
            """INSERT INTO snapshots
               (dataset_id, version_number, created_at, input_version, parent_id, result_json)
               VALUES (?,?,?,?,?,?)""",
            (
                dataset_id,
                version_number,
                utc_now(),
                input_version,
                parent["id"] if parent else None,
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            ),
        )
        return {"id": int(cur.lastrowid), "version_number": version_number}

    def list_snapshots(self, dataset_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT id, version_number, created_at, input_version, parent_id
                   FROM snapshots WHERE dataset_id=? ORDER BY version_number DESC""",
                (dataset_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshot(self, dataset_id: int, version_number: int | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            if version_number is None:
                row = conn.execute(
                    """SELECT * FROM snapshots WHERE dataset_id=?
                       ORDER BY version_number DESC LIMIT 1""",
                    (dataset_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM snapshots WHERE dataset_id=? AND version_number=?",
                    (dataset_id, version_number),
                ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "version_number": row["version_number"],
            "created_at": row["created_at"],
            "input_version": row["input_version"],
            "parent_id": row["parent_id"],
            "result": json.loads(row["result_json"]),
        }
