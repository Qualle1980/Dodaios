"""Task and progress records: the conductor knows at every step where an order stands.

The task record holds the facts and the plan. After every executor result the
conductor refreshes the progress record (strict JSON): is the request
satisfied, is work looping, is it moving, who acts next, with what
instruction. No progress or a loop counts as a stall; a step that moves
forward pays one stall back. At ``max_stalls`` the plan is rewritten once
from "what went wrong", at ``max_replans`` the order stops with an honest
report instead of retrying the same instruction.

The prompts and the stall rule are adapted from Magentic-One in
microsoft/autogen (``autogen_agentchat/teams/_group_chat/_magentic_one``):

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel, ValidationError

DEFAULT_MAX_STALLS: Final[int] = 2
DEFAULT_MAX_REPLANS: Final[int] = 2

FACTS_PROMPT: Final[str] = """Below is a request. Before work starts, answer this pre-survey.

Request:
{task}

1. GIVEN OR VERIFIED FACTS: specific facts or figures given in the request itself (possibly none).
2. FACTS TO LOOK UP: facts that must be looked up, and WHERE exactly (a file, a site, a tool).
3. FACTS TO DERIVE: facts to work out by reasoning or computation.
4. EDUCATED GUESSES: facts recalled from memory, hunches, well-reasoned guesses.

Use exactly these four headings. Do not list next steps or plans yet.
"""

PLAN_PROMPT: Final[str] = """These executors are available, cheapest first:
{team}

Lessons learned from earlier failures (follow them):
{lessons}

Devise a short bullet-point plan for the request. Name one executor per
bullet. Prefer the cheapest executor that can do the bullet; there is no
requirement to involve every executor.
"""

PROGRESS_PROMPT: Final[str] = """Recall the request:
{task}

Executors:
{team}

Answer these questions, each with a short reason:
- Is the request fully satisfied? (False until it is SUCCESSFULLY and FULLY addressed.)
- Are we in a loop, repeating the same requests or getting the same responses?
- Are we making forward progress? (True when just starting or when recent results add value;
  False when stuck or blocked, e.g. a required file cannot be read.)
- Which executor acts next? (one of: {names})
- What instruction do you give it? (Speak to it directly and include what it needs.)

Output ONLY one JSON object of this schema, nothing else:
{{
  "is_request_satisfied": {{"reason": string, "answer": boolean}},
  "is_in_loop": {{"reason": string, "answer": boolean}},
  "is_progress_being_made": {{"reason": string, "answer": boolean}},
  "next_executor": {{"reason": string, "answer": string}},
  "instruction": {{"reason": string, "answer": string}}
}}
"""

FACTS_UPDATE_PROMPT: Final[str] = """We are working on this request:
{task}

We are not making the progress we should, but we may have learned something.
Rewrite the fact sheet below with what we learned: move guesses to verified
facts where justified, and add or update at least one educated guess with its
reasoning.

Old fact sheet:
{facts}
"""

PLAN_UPDATE_PROMPT: Final[str] = """Briefly explain what went wrong on the last attempt (the root
cause), then write a new short bullet-point plan that overcomes it and does NOT
repeat the same mistake. Use only these executors:
{team}
"""

FINAL_ANSWER_PROMPT: Final[str] = """We worked on this request:
{task}

The work above is done. Based on what was found and produced, give the final
answer to the user, phrased as if speaking to them. Say plainly what was not
achieved, if anything.
"""


class _BoolAnswer(BaseModel):
    reason: str
    answer: bool


class _TextAnswer(BaseModel):
    reason: str
    answer: str


class ProgressEntry(BaseModel):
    """One refresh of the progress record."""

    is_request_satisfied: _BoolAnswer
    is_in_loop: _BoolAnswer
    is_progress_being_made: _BoolAnswer
    next_executor: _TextAnswer
    instruction: _TextAnswer


_JSON_OBJECT_RE: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)


def parse_progress(raw: str, executors: list[str]) -> ProgressEntry | None:
    """Parse one model reply into a progress entry, or ``None`` when unusable.

    With a single executor the next one is set deterministically; an
    unsatisfied request naming an unknown executor is unusable (the caller
    asks again, bounded).
    """
    match = _JSON_OBJECT_RE.search(raw or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:  # unusable reply: the caller asks again, bounded
        return None
    if isinstance(data, dict) and len(executors) == 1:
        data["next_executor"] = {
            "reason": "Only one executor is available.", "answer": executors[0],
        }
    try:
        entry = ProgressEntry.model_validate(data)
    except ValidationError:  # wrong shape: the caller asks again, bounded
        return None
    if not entry.is_request_satisfied.answer and entry.next_executor.answer not in executors:
        return None
    return entry


Decision = Literal["done", "continue", "replan", "stop"]


@dataclass
class StallCounter:
    """Stall bookkeeping for one order."""

    max_stalls: int = DEFAULT_MAX_STALLS
    max_replans: int = DEFAULT_MAX_REPLANS
    stalls: int = 0
    replans: int = 0
    history: list[str] = field(default_factory=list)

    def decide(self, entry: ProgressEntry) -> Decision:
        """What the conductor does after this refresh."""
        if entry.is_request_satisfied.answer:
            return "done"
        if not entry.is_progress_being_made.answer or entry.is_in_loop.answer:
            self.stalls += 1
            self.history.append(
                entry.is_in_loop.reason if entry.is_in_loop.answer
                else entry.is_progress_being_made.reason
            )
        else:
            self.stalls = max(0, self.stalls - 1)
        if self.stalls < self.max_stalls:
            return "continue"
        if self.replans >= self.max_replans:
            return "stop"
        self.replans += 1
        self.stalls = 0
        return "replan"


__all__ = [
    "DEFAULT_MAX_REPLANS", "DEFAULT_MAX_STALLS", "FACTS_PROMPT", "FACTS_UPDATE_PROMPT",
    "FINAL_ANSWER_PROMPT", "PLAN_PROMPT", "PLAN_UPDATE_PROMPT", "PROGRESS_PROMPT",
    "ProgressEntry", "StallCounter", "parse_progress",
]
