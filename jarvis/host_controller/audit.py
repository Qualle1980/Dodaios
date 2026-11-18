"""Append-only JSON-lines audit of every intent and host action."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Local JSONL log. Durable enough for an audit trail; GitHub is source of truth."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        intent_id: str | None,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "intent_id": intent_id,
            "action": action,
            "detail": detail or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
