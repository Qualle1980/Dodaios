"""Live coding-CLI model lists with a self-refreshing app-data snapshot.

Grok and Claude still read their local catalog files. Codex prefers the
account-confirmed ``model/list`` lister (``codex app-server``); the on-disk
``models_cache.json`` is only an **unconfirmed** fallback. Instant and coding
hand-offs may choose only confirmed Codex ids.

Refresh triggers: first read, age older than :data:`REFRESH_TTL_S` (10 min,
chosen in ``_research-model-lists-2026-11-17.md``), and immediately after a
model id is refused. A failed refresh keeps the last good list and reports
it; an empty parse is an error, never "no models".
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

#: App-side max age before re-reading the CLI caches (research: 600 s).
REFRESH_TTL_S: Final[float] = 600.0
#: An unconfirmed list (cache file, failed account probe) is re-read at most
#: this often. Without it every description access ran the CLI again — a
#: synchronous subprocess on the event loop for each tool-declaration build.
UNCONFIRMED_RETRY_TTL_S: Final[float] = 60.0
#: Codex account listing waits this long (research: warm ~6 s, 8 s default timed out).
CODEX_LIST_TIMEOUT_S: Final[float] = 15.0

CLI_CLAUDE: Final[str] = "claude"
CLI_CODEX: Final[str] = "codex"
CLI_GROK: Final[str] = "grok"
KNOWN_CLIS: Final[tuple[str, ...]] = (CLI_CLAUDE, CLI_CODEX, CLI_GROK)

#: ``CliModelList.source`` values.
SOURCE_LIVE: Final[str] = "live"
SOURCE_STORED: Final[str] = "stored"
SOURCE_KEPT: Final[str] = "kept"
SOURCE_CACHE_UNCONFIRMED: Final[str] = "cache_unconfirmed"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model a coding CLI will accept."""

    id: str
    display_name: str
    efforts: tuple[str, ...]
    default_effort: str
    read_at: float


@dataclass(frozen=True, slots=True)
class CliModelList:
    """Normalised list for one CLI, plus honesty about how it was obtained."""

    cli: str
    models: tuple[ModelEntry, ...]
    read_at: float
    source: str  # live | stored | kept | cache_unconfirmed
    ok: bool
    error: str = ""
    kept_last_good: bool = False
    fetched_at: str = ""
    #: False for Codex cache-only fallback, or when the account listing failed.
    confirmed: bool = True

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.read_at)

    def ids(self) -> set[str]:
        return {m.id for m in self.models}

    def get(self, model_id: str) -> ModelEntry | None:
        wanted = (model_id or "").strip()
        for row in self.models:
            if row.id == wanted:
                return row
        return None


#: Returns account-confirmed Codex rows ``{id,label,efforts,default_effort}``, or
#: ``None`` when discovery fails. Tests inject fakes; production uses app-server.
CodexLister = Callable[[], Sequence[Mapping[str, Any]] | None]


@dataclass(frozen=True, slots=True)
class ModelListSources:
    """Paths the store reads. Tests inject fakes; production fills defaults."""

    grok_cache: Path | None = None
    codex_cache: Path | None = None
    claude_catalog_file: Path | None = None
    claude_catalog_dir: Path | None = None
    #: Account-confirmed Codex listing. When set, the cache file is unconfirmed-only.
    codex_lister: CodexLister | None = None


def _production_codex_lister() -> Sequence[Mapping[str, Any]] | None:
    """Ask the installed Codex CLI via ``model/list`` (no coding turn).

    Uses the 15 s listing budget from research (the 8 s discovery default timed
    out on this PC and poisoned confirmation for every id). Imported from
    ``agent_chat.codex_listing`` so this package never touches ``runner_cli``.
    """
    from jarvis.agent_chat.codex_listing import list_confirmed_codex_models

    rows = list_confirmed_codex_models(timeout_s=CODEX_LIST_TIMEOUT_S)
    if rows is None:
        return None
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("id") or "").strip()
        if not mid:
            continue
        efforts = row.get("efforts") or []
        if not isinstance(efforts, list):
            efforts = []
        effort_vals = [str(e).strip().lower() for e in efforts if str(e).strip()]
        out.append(
            {
                "id": mid,
                "label": str(row.get("label") or mid),
                "efforts": effort_vals,
                "default_effort": effort_vals[0] if effort_vals else "",
            }
        )
    return out


def default_sources() -> ModelListSources:
    """The installed CLIs' catalog locations on this machine."""
    home = Path.home()
    return ModelListSources(
        grok_cache=home / ".grok" / "models_cache.json",
        codex_cache=home / ".codex" / "models_cache.json",
        claude_catalog_dir=home / ".claude" / "cache" / "model-catalog",
        codex_lister=_production_codex_lister,
    )


def default_store_path() -> Path:
    """Snapshot under the app data directory (never the repo)."""
    from jarvis.core.config import DATA_DIR

    return Path(DATA_DIR) / "cli_model_lists.json"


def _efforts_from_grok(info: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    seen: list[str] = []
    default = ""
    for level in info.get("reasoning_efforts") or []:
        if isinstance(level, dict):
            value = str(level.get("value") or level.get("id") or "").strip().lower()
            if level.get("default") is True and value:
                default = value
        else:
            value = str(level).strip().lower()
        if value and value not in seen:
            seen.append(value)
    if not default and seen:
        default = "high" if "high" in seen else seen[0]
    return tuple(seen), default


def parse_grok_cache(data: Any, *, read_at: float) -> tuple[ModelEntry, ...]:
    """Normalise Grok Build's ``models_cache.json`` body."""
    if not isinstance(data, dict):
        raise ValueError("Grok models cache is not an object")
    models = data.get("models")
    if isinstance(models, dict):
        entries: Iterable[Any] = models.values()
    elif isinstance(models, list):
        entries = models
    else:
        raise ValueError("Grok models cache has no models field")
    rows: list[ModelEntry] = []
    for entry in entries:
        info = entry.get("info") if isinstance(entry, dict) else None
        if not isinstance(info, dict):
            info = entry if isinstance(entry, dict) else None
        if not isinstance(info, dict) or info.get("hidden") is True:
            continue
        slug = str(info.get("id") or info.get("model") or "").strip()
        if not slug:
            continue
        efforts, default = _efforts_from_grok(info)
        rows.append(
            ModelEntry(
                id=slug,
                display_name=str(info.get("name") or slug),
                efforts=efforts,
                default_effort=default,
                read_at=read_at,
            )
        )
    return tuple(rows)


def _entries_from_codex_rows(
    rows: Sequence[Mapping[str, Any]], *, read_at: float
) -> tuple[ModelEntry, ...]:
    """Normalise account ``model/list`` / fake-lister rows into :class:`ModelEntry`."""
    out: list[ModelEntry] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        mid = str(row.get("id") or row.get("model") or "").strip()
        if not mid:
            continue
        efforts_raw = row.get("efforts") or []
        efforts: list[str] = []
        if isinstance(efforts_raw, list):
            for level in efforts_raw:
                value = str(level).strip().lower()
                if value and value not in efforts:
                    efforts.append(value)
        default = str(row.get("default_effort") or row.get("defaultReasoningLevel") or "").strip().lower()
        if default and default not in efforts:
            efforts.insert(0, default)
        if not default and efforts:
            default = efforts[0]
        out.append(
            ModelEntry(
                id=mid,
                display_name=str(row.get("label") or row.get("display_name") or mid),
                efforts=tuple(efforts),
                default_effort=default,
                read_at=read_at,
            )
        )
    return tuple(out)


def parse_codex_cache(data: Any, *, read_at: float) -> tuple[ModelEntry, ...]:
    """Normalise Codex ``models_cache.json`` (slug + supported_reasoning_levels)."""
    if not isinstance(data, dict):
        raise ValueError("Codex models cache is not an object")
    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError("Codex models cache has no models list")
    rows: list[ModelEntry] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("visibility") == "hide":
            continue
        slug = str(model.get("slug") or model.get("id") or model.get("model") or "").strip()
        if not slug:
            continue
        levels = model.get("supported_reasoning_levels") or model.get("supportedReasoningEfforts") or []
        efforts: list[str] = []
        if isinstance(levels, list):
            for level in levels:
                if isinstance(level, dict):
                    value = str(
                        level.get("effort")
                        or level.get("reasoningEffort")
                        or level.get("value")
                        or ""
                    ).strip().lower()
                else:
                    value = str(level).strip().lower()
                if value and value not in efforts:
                    efforts.append(value)
        default = str(model.get("default_reasoning_level") or model.get("defaultReasoningLevel") or "").strip().lower()
        if default and default not in efforts:
            efforts.insert(0, default)
        if not default and efforts:
            default = efforts[0]
        rows.append(
            ModelEntry(
                id=slug,
                display_name=str(model.get("display_name") or model.get("displayName") or slug),
                efforts=tuple(efforts),
                default_effort=default,
                read_at=read_at,
            )
        )
    return tuple(rows)


def parse_claude_catalog(data: Any, *, read_at: float) -> tuple[ModelEntry, ...]:
    """Normalise Claude Code's CCD ``model-catalog`` JSON."""
    if not isinstance(data, dict):
        raise ValueError("Claude model catalog is not an object")
    catalog = data.get("catalog")
    config = catalog.get("config") if isinstance(catalog, dict) else None
    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(models, list):
        # Some snapshots nest models at the top level.
        models = data.get("models") if isinstance(data.get("models"), list) else None
    if not isinstance(models, list):
        raise ValueError("Claude model catalog has no models list")
    rows: list[ModelEntry] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = str(model.get("id") or "").strip()
        if not slug:
            continue
        runtime = model.get("runtime") if isinstance(model.get("runtime"), dict) else {}
        levels = runtime.get("effort_levels") or []
        efforts: list[str] = []
        if isinstance(levels, list):
            for level in levels:
                value = str(level).strip().lower()
                if value and value not in efforts:
                    efforts.append(value)
        # Prefer runtime.default_effort; fall back to thinking badge "Recommended".
        default = str(runtime.get("default_effort") or "").strip().lower()
        thinking = model.get("thinking") if isinstance(model.get("thinking"), dict) else {}
        for opt in thinking.get("effort_options") or []:
            if not isinstance(opt, dict):
                continue
            value = str(opt.get("id") or "").strip().lower()
            if value and value not in efforts:
                efforts.append(value)
            badge = opt.get("badge") if isinstance(opt.get("badge"), dict) else None
            if badge and str(badge.get("message") or "").lower() == "recommended" and value:
                if not default:
                    default = value
        if not default and efforts:
            default = "medium" if "medium" in efforts else efforts[0]
        rows.append(
            ModelEntry(
                id=slug,
                display_name=str(model.get("name") or slug),
                efforts=tuple(efforts),
                default_effort=default,
                read_at=read_at,
            )
        )
    return tuple(rows)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _newest_claude_ccd(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(
        (
            p
            for p in directory.glob("*.json")
            if p.name.endswith("-ccd.json") or p.name.endswith("-cc.json")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    all_json = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_json[0] if all_json else None


@dataclass
class ModelListStore:
    """Reads CLI catalogs, snapshots them under app data, refreshes on age/refusal."""

    store_path: Path
    sources: ModelListSources = field(default_factory=default_sources)
    now: Callable[[], float] = time.time
    refresh_ttl_s: float = REFRESH_TTL_S
    _memory: dict[str, CliModelList] = field(default_factory=dict, init=False, repr=False)

    def ensure(self, cli: str, *, force: bool = False) -> CliModelList:
        """Return the current list for ``cli``, refreshing when stale or forced."""
        key = _normalize_cli(cli)
        current = self._memory.get(key) or self._read_stored(key)
        age = (self.now() - current.read_at) if current is not None else self.refresh_ttl_s + 1.0
        if (
            current is not None
            and current.ok
            and current.models
            and current.confirmed
            and not force
            and age < self.refresh_ttl_s
        ):
            return current
        if (
            current is not None
            and not current.confirmed
            and not force
            and age < UNCONFIRMED_RETRY_TTL_S
        ):
            # Asked again soon: answer from the last read instead of running the CLI.
            return current
        return self._refresh(key, previous=current)

    def after_refusal(self, cli: str, model_id: str) -> CliModelList:
        """Force a refresh immediately after a CLI refused ``model_id``."""
        log.info(
            "model lists: refreshing %s after refusal of %r",
            cli,
            (model_id or "").strip(),
        )
        return self.ensure(cli, force=True)

    def contains(self, cli: str, model_id: str) -> bool:
        result = self.ensure(cli)
        return bool(model_id) and result.confirmed and model_id in result.ids()

    def pick_codex_reviewer(self) -> ModelEntry | None:
        """A confirmed Codex model for review: default-ish first list entry.

        Prefers the first visible model and its highest effort the list allows.
        Never invents a fixed name such as ``gpt-5.2``. Returns ``None`` when
        the account listing is unconfirmed or empty.
        """
        result = self.ensure(CLI_CODEX)
        if not result.confirmed or not result.models:
            return None
        # Prefer a non-review-only slug when several exist; else first entry.
        preferred = next(
            (m for m in result.models if m.id != "codex-auto-review"),
            result.models[0],
        )
        return preferred

    def highest_effort(self, entry: ModelEntry) -> str:
        """Highest effort the entry lists (review), else its default."""
        order = ("low", "medium", "high", "xhigh", "max", "ultra")
        ranked = [level for level in order if level in entry.efforts]
        if ranked:
            return ranked[-1]
        return entry.default_effort

    # -- internals -----------------------------------------------------------

    def _refresh(self, cli: str, *, previous: CliModelList | None) -> CliModelList:
        read_at = float(self.now())
        try:
            models, fetched_at, source, confirmed, note = self._read_live(cli, read_at=read_at)
        except Exception as exc:  # noqa: BLE001 — keep last good; report honestly
            error = f"refresh failed: {type(exc).__name__}: {exc}"
            log.warning("model lists: %s (%s)", error, cli)
            if previous is not None and previous.models:
                kept = CliModelList(
                    cli=cli,
                    models=previous.models,
                    read_at=previous.read_at,
                    source=SOURCE_KEPT,
                    ok=False,
                    error=error,
                    kept_last_good=True,
                    fetched_at=previous.fetched_at,
                    confirmed=previous.confirmed,
                )
                self._memory[cli] = kept
                return kept
            empty = CliModelList(
                cli=cli,
                models=(),
                read_at=read_at,
                source=SOURCE_LIVE,
                ok=False,
                error=error,
                confirmed=False,
            )
            self._memory[cli] = empty
            return empty

        if not models:
            error = note or "empty model list from CLI catalog"
            if previous is not None and previous.models and previous.confirmed:
                kept = CliModelList(
                    cli=cli,
                    models=previous.models,
                    read_at=previous.read_at,
                    source=SOURCE_KEPT,
                    ok=False,
                    error=error,
                    kept_last_good=True,
                    fetched_at=previous.fetched_at,
                    confirmed=previous.confirmed,
                )
                self._memory[cli] = kept
                return kept
            empty = CliModelList(
                cli=cli,
                models=(),
                read_at=read_at,
                source=source,
                ok=False,
                error=error,
                confirmed=False,
            )
            self._memory[cli] = empty
            self._write_stored(cli, empty)
            return empty

        fresh = CliModelList(
            cli=cli,
            models=models,
            read_at=read_at,
            source=source,
            ok=True,
            error=note,
            fetched_at=fetched_at,
            confirmed=confirmed,
        )
        self._memory[cli] = fresh
        self._write_stored(cli, fresh)
        return fresh

    def _read_live(
        self, cli: str, *, read_at: float
    ) -> tuple[tuple[ModelEntry, ...], str, str, bool, str]:
        """Return ``(models, fetched_at, source, confirmed, note)``."""
        if cli == CLI_GROK:
            path = self.sources.grok_cache
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Grok models cache missing: {path}")
            data = _load_json(path)
            return (
                parse_grok_cache(data, read_at=read_at),
                str(data.get("fetched_at") or ""),
                SOURCE_LIVE,
                True,
                "",
            )
        if cli == CLI_CODEX:
            return self._read_codex_live(read_at=read_at)
        if cli == CLI_CLAUDE:
            path = self.sources.claude_catalog_file
            if path is None:
                directory = self.sources.claude_catalog_dir
                path = _newest_claude_ccd(directory) if directory is not None else None
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Claude model catalog missing: {path}")
            data = _load_json(path)
            fetched = data.get("fetchedAt")
            fetched_at = str(fetched) if fetched is not None else ""
            return (
                parse_claude_catalog(data, read_at=read_at),
                fetched_at,
                SOURCE_LIVE,
                True,
                "",
            )
        raise ValueError(f"unknown coding CLI {cli!r}")

    def _read_codex_live(
        self, *, read_at: float
    ) -> tuple[tuple[ModelEntry, ...], str, str, bool, str]:
        """Account ``model/list`` first; cache file only as unconfirmed fallback."""
        lister = self.sources.codex_lister
        list_error = ""
        if lister is not None:
            try:
                rows = lister()
            except Exception as exc:  # noqa: BLE001 — fall through to unconfirmed cache
                list_error = f"account listing failed: {type(exc).__name__}: {exc}"
                log.warning("model lists: %s", list_error)
                rows = None
            if rows is not None:
                models = _entries_from_codex_rows(rows, read_at=read_at)
                if models:
                    return models, "", SOURCE_LIVE, True, ""
                list_error = list_error or "account listing returned no models"
            elif not list_error:
                list_error = "account listing unavailable"

        # Unconfirmed fallback — Instant must not pick from this alone.
        path = self.sources.codex_cache
        if path is not None and path.is_file():
            data = _load_json(path)
            models = parse_codex_cache(data, read_at=read_at)
            note = list_error or "Codex account listing unavailable; cache is unconfirmed"
            return (
                models,
                str(data.get("fetched_at") or ""),
                SOURCE_CACHE_UNCONFIRMED,
                False,
                note,
            )
        if list_error:
            raise RuntimeError(list_error)
        raise FileNotFoundError(f"Codex models cache missing: {path}")

    def _read_stored(self, cli: str) -> CliModelList | None:
        path = self.store_path
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.debug("model lists: store unreadable at %s: %s", path, exc)
            return None
        block = raw.get(cli) if isinstance(raw, dict) else None
        if not isinstance(block, dict):
            return None
        models_raw = block.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            return None
        read_at = float(block.get("read_at") or 0.0)
        models = tuple(
            ModelEntry(
                id=str(row.get("id") or ""),
                display_name=str(row.get("display_name") or row.get("id") or ""),
                efforts=tuple(str(e) for e in (row.get("efforts") or ()) if str(e).strip()),
                default_effort=str(row.get("default_effort") or ""),
                read_at=read_at,
            )
            for row in models_raw
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        )
        if not models:
            return None
        confirmed = bool(block.get("confirmed", True))
        source = str(block.get("source") or SOURCE_STORED)
        if source == SOURCE_CACHE_UNCONFIRMED:
            confirmed = False
        return CliModelList(
            cli=cli,
            models=models,
            read_at=read_at,
            source=SOURCE_STORED if source == SOURCE_LIVE else source,
            ok=bool(block.get("ok", True)) and confirmed,
            error=str(block.get("error") or ""),
            fetched_at=str(block.get("fetched_at") or ""),
            confirmed=confirmed,
        )

    def _write_stored(self, cli: str, result: CliModelList) -> None:
        path = self.store_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except (OSError, ValueError) as exc:
                    # Corrupt or unreadable snapshot: start fresh for this write.
                    log.warning(
                        "model lists: ignored unreadable store %s while saving %s: %s",
                        path,
                        cli,
                        exc,
                    )
                    existing = {}
            existing[cli] = {
                "read_at": result.read_at,
                "fetched_at": result.fetched_at,
                "ok": result.ok,
                "error": result.error,
                "confirmed": result.confirmed,
                "source": result.source,
                "models": [
                    {
                        "id": m.id,
                        "display_name": m.display_name,
                        "efforts": list(m.efforts),
                        "default_effort": m.default_effort,
                    }
                    for m in result.models
                ],
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            log.warning("model lists: could not write store %s: %s", path, exc)


_NORMALIZE: Final[dict[str, str]] = {
    "claude": CLI_CLAUDE,
    "claude-cli": CLI_CLAUDE,
    "claude-code": CLI_CLAUDE,
    "claude-api": CLI_CLAUDE,
    "codex": CLI_CODEX,
    "codex-cli": CLI_CODEX,
    "openai-codex": CLI_CODEX,
    "grok": CLI_GROK,
    "grok-cli": CLI_GROK,
    "grok-build": CLI_GROK,
}


def _normalize_cli(cli: str) -> str:
    key = (cli or "").strip().lower()
    if key not in _NORMALIZE:
        raise ValueError(f"unknown coding CLI {cli!r}")
    return _NORMALIZE[key]


_SHARED: ModelListStore | None = None


def shared_store() -> ModelListStore:
    """Process-wide store (app start + later reads)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = ModelListStore(store_path=default_store_path(), sources=default_sources())
    return _SHARED


def reset_shared_store_for_tests() -> None:
    """Drop the process singleton (unit tests only)."""
    global _SHARED
    _SHARED = None


def install_shared_store(store: ModelListStore | None) -> ModelListStore | None:
    """Replace the process singleton (unit tests with fake caches)."""
    global _SHARED
    previous = _SHARED
    _SHARED = store
    return previous


def format_for_instant(cli: str, result: CliModelList) -> list[dict[str, Any]]:
    """Compact rows Instant can put in a tool description or dispatch context.

    Unconfirmed Codex snapshots yield an empty list so Instant cannot pick them.
    """
    if cli == CLI_CODEX and not result.confirmed:
        return []
    return [
        {
            "id": m.id,
            "label": m.display_name,
            "efforts": list(m.efforts),
            "default_effort": m.default_effort,
        }
        for m in result.models
    ]


def live_models_blurb(*, store: ModelListStore | None = None, limit_per_cli: int = 6) -> str:
    """Short live-id block for Instant tool descriptions / coding-dispatch context."""
    lists = store or shared_store()
    parts: list[str] = ["Live model ids only (never invent):"]
    for cli in KNOWN_CLIS:
        try:
            result = lists.ensure(cli)
        except Exception as exc:  # noqa: BLE001 — skip a CLI that cannot load
            parts.append(f"{cli}:unavailable({type(exc).__name__})")
            continue
        if cli == CLI_CODEX and not result.confirmed:
            note = result.error or "unconfirmed"
            parts.append(f"{cli}:unconfirmed({note})")
            continue
        if not result.models:
            note = result.error or "empty"
            parts.append(f"{cli}:none({note})")
            continue
        age = max(0, int(lists.now() - result.read_at))
        ids = ",".join(m.id for m in result.models[:limit_per_cli])
        more = len(result.models) - limit_per_cli
        suffix = f"+{more}" if more > 0 else ""
        parts.append(f"{cli}@{age}s:{ids}{suffix}")
    return " ".join(parts)


def validate_model_id(cli: str, model_id: str, *, store: ModelListStore | None = None) -> dict[str, Any] | str:
    """Return the model id when it is on the confirmed list, else an honest refusal.

    Codex ids from an unconfirmed cache are refused. A failed account listing
    names the failure so Instant opens no pane.
    """
    wanted = (model_id or "").strip()
    if not wanted:
        return ""
    lists = store or shared_store()
    result = lists.ensure(cli)
    if result.confirmed and wanted in result.ids():
        return wanted
    # Refusal / unconfirmed: force one refresh in case the CLI just gained the id.
    result = lists.after_refusal(cli, wanted)
    if result.confirmed and wanted in result.ids():
        return wanted
    age = max(0, int(lists.now() - result.read_at)) if result.read_at else 0
    if not result.confirmed:
        note = result.error or "Codex account listing did not confirm any models"
        return {
            "status": "not_accepted",
            "reason": (
                f"review not run: Codex model list is unconfirmed ({note}). "
                f"Model {wanted!r} was not opened. Refresh Codex sign-in or retry listing."
            ),
        }
    offered = ", ".join(sorted(result.ids())[:12]) or "(none)"
    return {
        "status": "not_accepted",
        "reason": (
            f"Model {wanted!r} is not in the confirmed {cli} list "
            f"(list age {age}s, source={result.source}"
            + (f", note={result.error}" if result.error else "")
            + f"). Use one of: {offered}."
        ),
    }


__all__ = [
    "CLI_CLAUDE",
    "CLI_CODEX",
    "CLI_GROK",
    "CODEX_LIST_TIMEOUT_S",
    "CliModelList",
    "CodexLister",
    "KNOWN_CLIS",
    "ModelEntry",
    "ModelListSources",
    "ModelListStore",
    "REFRESH_TTL_S",
    "SOURCE_CACHE_UNCONFIRMED",
    "SOURCE_KEPT",
    "SOURCE_LIVE",
    "SOURCE_STORED",
    "default_sources",
    "default_store_path",
    "format_for_instant",
    "install_shared_store",
    "live_models_blurb",
    "parse_claude_catalog",
    "parse_codex_cache",
    "parse_grok_cache",
    "reset_shared_store_for_tests",
    "shared_store",
    "validate_model_id",
]
