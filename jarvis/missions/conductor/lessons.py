"""Lessons: every failure is written down, and a rule keeps it from happening again.

Two layers:

* **Incidents** (no model, always): every failed or revised step is appended
  to ``incidents.jsonl`` with lane, executor, cause and the user's request.
  Nothing is lost when a provider is down.
* **Rules** (one call on the conductor's subscription seat, after a failure):
  the model sees the incident and the current rules and may AGREE, EDIT,
  REMOVE or ADD at most four times. Python validates every operation; a rule
  that gets agreed or edited gains weight, a removal costs weight, and a rule
  at weight 0 is dropped. The strongest rules for a lane go into the next
  plan (:data:`.ledger.PLAN_PROMPT`), so the same mistake is named before the
  executor starts.

The rule operations and their weighting are adapted from ExpeL
(LeapLabTHU/ExpeL, ``prompts/templates/human.py``, ``insight_extraction.py``),
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: operations apply to
coding-order failures instead of benchmark trials, at most one model call per
failure, rules are scoped to a lane, and every text passes the memory guard.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

from jarvis.memory.learning.guard import refusal

logger = logging.getLogger(__name__)

MAX_RULES: Final[int] = 30
MAX_OPS: Final[int] = 4
#: Weight of a new rule; AGREE/EDIT add one, REMOVE takes one, 0 drops it.
NEW_RULE_WEIGHT: Final[int] = 2
#: The memory guard refuses imperatives for user facts; a lesson IS one.
_ALLOWED_REFUSAL: Final[str] = "phrased as an order, not a fact"

Op = Literal["AGREE", "EDIT", "REMOVE", "ADD"]

LESSON_PROMPT: Final[str] = """An order failed. Turn it into rules that prevent it next time.

Lane: {lane}
Executor: {executor}
Request: {request}
What went wrong: {cause}

Existing rules:
{rules}

You may perform these operations on the rules: AGREE (an existing rule is strongly
relevant here), REMOVE (an existing rule is contradictory or duplicates another),
EDIT (an existing rule is not general enough; rewrite it), ADD (a new rule, very
different from the existing ones). Rules must be GENERALLY APPLICABLE to future
orders: do not mention this request. Each rule is one concise sentence.
At most {max_ops} operations; each existing rule gets at most one.{full_hint}
Use exactly this format, one per line:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>
"""

_FULL_HINT: Final[str] = (
    " The list is full: prefer REMOVE, and ADD only a rule that is very different."
)

_OP_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(AGREE|REMOVE|EDIT|ADD)\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE
)


@dataclass
class Rule:
    """One learned rule."""

    text: str
    lane: str
    weight: int = NEW_RULE_WEIGHT
    updated_ms: int = 0


@dataclass(frozen=True)
class Incident:
    """One documented failure."""

    lane: str
    executor: str
    cause: str
    request: str
    mission_id: str = ""


def parse_ops(raw: str, n_existing: int) -> list[tuple[Op, int, str]]:
    """Valid operations from a model reply, in order, at most :data:`MAX_OPS`.

    Dropped: unknown numbers, a second operation on the same rule, text the
    memory guard refuses.
    """
    ops: list[tuple[Op, int, str]] = []
    touched: set[int] = set()
    for line in (raw or "").splitlines():
        match = _OP_RE.match(line)
        if match is None:
            continue
        op = match.group(1).upper()
        number = int(match.group(2))
        text = match.group(3).strip()
        if op != "ADD":
            if not 1 <= number <= n_existing or number in touched:
                continue
            touched.add(number)
        if op in ("EDIT", "ADD"):
            reason = refusal(text)
            if reason is not None and reason != _ALLOWED_REFUSAL:
                logger.info("lessons: refused %s text (%s)", op, reason)
                continue
        ops.append((op, number, text))  # type: ignore[arg-type]
        if len(ops) >= MAX_OPS:
            break
    return ops


class LessonStore:
    """Incidents and rules on disk, one directory, safe for concurrent writers."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._rules_path = root / "rules.json"
        self._incidents_path = root / "incidents.jsonl"
        self._lock = threading.Lock()

    # -- incidents -------------------------------------------------------

    def record_incident(self, incident: Incident) -> None:
        """Append one failure. Never raises: losing a record must not fail an order."""
        line = json.dumps({**asdict(incident), "ts_ms": int(time.time() * 1000)},
                          ensure_ascii=False)
        try:
            with self._lock:
                self._root.mkdir(parents=True, exist_ok=True)
                with self._incidents_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            logger.warning("lessons: could not record incident", exc_info=True)

    # -- rules -----------------------------------------------------------

    def rules(self) -> list[Rule]:
        try:
            data = json.loads(self._rules_path.read_text(encoding="utf-8"))
        except FileNotFoundError:  # no lesson learned yet: an empty list is the truth
            return []
        except (OSError, ValueError):
            logger.warning("lessons: unreadable rules file, starting empty", exc_info=True)
            return []
        return [Rule(**item) for item in data if isinstance(item, dict)]

    def _write(self, rules: list[Rule]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix=".rules-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump([asdict(r) for r in rules], fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self._rules_path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise

    def lesson_prompt(self, incident: Incident) -> str:
        """The prompt for one rule update after ``incident``."""
        rules = self.rules()
        listing = "\n".join(f"{i}. {r.text}" for i, r in enumerate(rules, 1)) or "(none yet)"
        return LESSON_PROMPT.format(
            lane=incident.lane, executor=incident.executor, request=incident.request[:500],
            cause=incident.cause[:500], rules=listing, max_ops=MAX_OPS,
            full_hint=_FULL_HINT if len(rules) >= MAX_RULES else "",
        )

    def apply(self, raw: str, lane: str) -> int:
        """Apply a model reply to the rules; returns the number of operations applied."""
        with self._lock:
            rules = self.rules()
            ops = parse_ops(raw, len(rules))
            now = int(time.time() * 1000)
            for op, number, text in ops:
                if op == "ADD":
                    if len(rules) < MAX_RULES and all(r.text != text for r in rules):
                        rules.append(Rule(text=text, lane=lane, updated_ms=now))
                    continue
                rule = rules[number - 1]
                if op == "AGREE":
                    rule.weight += 1
                elif op == "EDIT":
                    rule.text, rule.weight = text, rule.weight + 1
                else:
                    rule.weight -= 1
                rule.updated_ms = now
            kept = [r for r in rules if r.weight > 0]
            if ops:
                self._write(kept)
            return len(ops)

    def render(self, lane: str, limit: int = 5) -> str:
        """The strongest rules for ``lane`` (and the lane-free ones), as bullets."""
        relevant = [r for r in self.rules() if r.lane in (lane, "*")]
        relevant.sort(key=lambda r: (-r.weight, -r.updated_ms))
        return "\n".join(f"- {r.text}" for r in relevant[:limit]) or "(none)"


__all__ = [
    "LESSON_PROMPT", "MAX_OPS", "MAX_RULES", "Incident", "LessonStore", "Rule", "parse_ops",
]
