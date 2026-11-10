"""``set_background_may_use_api_key`` persists ``[brain] background_may_use_api_key``.

The switch decides whether work nobody asked for may bill one of the owner's API
keys, so it has to be settable without hand-editing ``jarvis.toml`` (AGENTS.md
section 2, AP-7). A boolean written as a TOML string would read as truthy for
``"false"``, so the round trip is pinned here rather than assumed.
"""
from __future__ import annotations

import tomllib

from jarvis.core import config_writer


def test_turning_it_on_round_trips_as_a_boolean(tmp_path):
    cfg = tmp_path / "jarvis.toml"
    cfg.write_text('[brain]\nprimary = "gemini"\n', encoding="utf-8")

    config_writer.set_background_may_use_api_key(True, path=cfg)

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert data["brain"]["background_may_use_api_key"] is True


def test_turning_it_off_writes_false_not_a_missing_key(tmp_path):
    # Absent and false both read as false today, but an explicit false is the
    # owner having decided, which a later reader must be able to tell apart.
    cfg = tmp_path / "jarvis.toml"
    cfg.write_text("[brain]\nbackground_may_use_api_key = true\n", encoding="utf-8")

    config_writer.set_background_may_use_api_key(False, path=cfg)

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert data["brain"]["background_may_use_api_key"] is False


def test_it_creates_the_table_and_preserves_siblings(tmp_path):
    cfg = tmp_path / "jarvis.toml"
    cfg.write_text('[ui]\norb_style = "jarvis_bar"\n', encoding="utf-8")

    config_writer.set_background_may_use_api_key(True, path=cfg)

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert data["brain"]["background_may_use_api_key"] is True
    assert data["ui"]["orb_style"] == "jarvis_bar"


def test_comments_survive_the_write(tmp_path):
    cfg = tmp_path / "jarvis.toml"
    cfg.write_text(
        "# the owner's own notes\n[brain]\nprimary = \"gemini\"  # chosen 2026-11-10\n",
        encoding="utf-8",
    )

    config_writer.set_background_may_use_api_key(True, path=cfg)

    text = cfg.read_text(encoding="utf-8")
    assert "# the owner's own notes" in text
    assert "# chosen 2026-11-10" in text


def test_a_missing_config_file_is_created_and_the_write_still_lands(tmp_path, caplog):
    # Pinned against what the code DOES, not what the sibling writers' docstrings
    # claim. They say FileNotFoundError; `_ensure_writable_config_path` in fact
    # creates an empty config and warns, which is right for a first run on a
    # fresh machine. The warning is the part that must not disappear.
    target = tmp_path / "absent.toml"

    with caplog.at_level("WARNING"):
        config_writer.set_background_may_use_api_key(True, path=target)

    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["brain"]["background_may_use_api_key"] is True
    assert any("NEW empty configuration file" in record.message for record in caplog.records)


def test_the_policy_reads_back_what_the_writer_wrote(tmp_path, monkeypatch):
    # The writer and the reader must agree on the key name; a typo in either
    # would leave a switch that looks set and changes nothing.
    cfg = tmp_path / "jarvis.toml"
    cfg.write_text("[brain]\n", encoding="utf-8")
    config_writer.set_background_may_use_api_key(True, path=cfg)

    from jarvis.core.config import BrainConfig

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert BrainConfig(**data["brain"]).background_may_use_api_key is True
