from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .output import validate_payload


class ProgressStore:
    """Crash-safe per-sample inference progress backed by SQLite."""

    def __init__(self, path: Path, identity: str, phase: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.phase = phase
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (sample_id TEXT PRIMARY KEY, item_json TEXT NOT NULL)"
        )
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        existing_phase = self.connection.execute(
            "SELECT value FROM metadata WHERE key='phase'"
        ).fetchone()
        if (
            existing is not None and existing[0] != identity
        ) or (
            existing_phase is not None and existing_phase[0] != str(phase)
        ):
            self.connection.close()
            raise RuntimeError(f"推理进度身份不匹配: {path}")
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('identity', ?)", (identity,)
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('phase', ?)", (str(phase),)
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ProgressStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def load(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for sample_id, item_json in self.connection.execute(
            "SELECT sample_id, item_json FROM results ORDER BY sample_id"
        ):
            item = json.loads(item_json)
            validate_payload({sample_id: item}, self.phase)
            result[sample_id] = item
        return result

    def put_many(self, items: dict[str, dict[str, Any]]) -> None:
        if not items:
            return
        validate_payload(items, self.phase)
        rows = [
            (sample_id, json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            for sample_id, item in items.items()
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO results(sample_id, item_json) VALUES(?, ?)", rows
            )

