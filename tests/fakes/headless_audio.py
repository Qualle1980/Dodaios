"""Pin a unit test to the headless audio condition: no real sound device.

``SpeechPipeline.__init__`` builds a real ``AudioPlayer`` and binds it as the
process-wide shared player (``jarvis/speech/pipeline.py``, ``bind_shared_audio_player``).
Nothing closes it, which cost nothing as long as ``sounddevice`` was absent and
every audio module bound ``sd = None``.

Ubuntu 26.04 changed that. Its PortAudio 19.7 reaches a running sound server,
so the import succeeds, the player grabs a real output device, and the process
calls ``abort()`` (exit -6) at interpreter shutdown while sounddevice's own
atexit handler tears PortAudio down under objects that still hold it. The tests
themselves all pass first -- only the exit dies, which is why the shard printed
five dots and then failed.

A unit test about idle timeouts has no business opening a speaker, on any OS.
This fixture restores exactly the condition these tests have always run under
and makes it explicit instead of accidental. It patches the module attribute,
never the installed package, so it cannot leak past the test.
"""
from __future__ import annotations

import pytest

#: Every audio module that binds `sounddevice` to a module-level `sd`.
#: `jarvis.audio.device_init` is deliberately absent: it resolves the module
#: lazily inside a function, so there is no attribute to patch. A module that
#: loses its `sd` fails loudly below rather than in a later crash.
_MODULES = (
    "jarvis.audio.player",
    "jarvis.audio.capture",
    "jarvis.audio.devices",
)


def silence_real_audio_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind ``sd = None`` in every audio module for the duration of a test."""
    import importlib

    for name in _MODULES:
        module = importlib.import_module(name)
        if not hasattr(module, "sd"):
            raise AttributeError(
                f"{name} has no module-level `sd`; headless_audio needs updating"
            )
        monkeypatch.setattr(module, "sd", None)
