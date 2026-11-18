"""Project-mode stream hold must not blank the consumer for ordinary prose.

Live-m10b 2026-11-18: after e594040529 every project-mode turn set
``text_consumer=None`` (manager.py stream hold). That stopped host JSON from
leaking, but it also forced the aggregate-without-consumer path on ordinary
local questions (branch / line count) that answer in prose and need tools.
Chat feeds ``generate(..., text_consumer=delta_publisher.feed)`` directly
(launcher.py), so the hold must be JSON-shaped only: prose stays live.
"""

from __future__ import annotations

from jarvis.brain.manager import _project_mode_stream_consumer
from jarvis.host_controller.chat_bridge import parse_host_intent


def test_prose_chunks_reach_the_consumer_in_project_mode() -> None:
    seen: list[str] = []
    wrapped = _project_mode_stream_consumer(seen.append)
    assert wrapped is not None

    wrapped("Checked out branch is ")
    wrapped("ahmad/devtest-combined.")

    assert "".join(seen) == "Checked out branch is ahmad/devtest-combined."


def test_host_intent_json_is_held_back_from_the_consumer() -> None:
    seen: list[str] = []
    wrapped = _project_mode_stream_consumer(seen.append)
    assert wrapped is not None

    payload = (
        '{"kind":"REPORT_PROGRESS","intent_id":"dodaios-open-pull-requests",'
        '"project":"PROBE-chief-2026-11-18"}'
    )
    wrapped(payload[:12])
    wrapped(payload[12:])

    assert seen == []
    assert parse_host_intent(payload) is not None


def test_none_consumer_stays_none() -> None:
    assert _project_mode_stream_consumer(None) is None
