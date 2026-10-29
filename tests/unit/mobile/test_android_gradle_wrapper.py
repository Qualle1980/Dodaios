"""The committed Android Gradle wrapper must be usable on every OS, and genuine.

A wrapper is four files a contributor runs without installing Gradle first, and
two of them are easy to commit in a shape that only works on the machine they
came from: a `gradlew` without the executable bit cannot be run on macOS or
Linux, and one checked out with CRLF fails there with "bad interpreter". The
jar is a binary nobody reads, so its identity has to be asserted rather than
assumed — it executes on every contributor's machine and in CI.

No network: the expected checksum is Gradle's published value for this
version, pinned here.
"""
from __future__ import annotations

import hashlib
import re
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_ANDROID = _REPO / "mobile" / "android"
_WRAPPER = _ANDROID / "gradle" / "wrapper"

#: Gradle's own published SHA-256 for the 8.9 wrapper jar, from
#: https://downloads.gradle.org/distributions/gradle-8.9-wrapper.jar.sha256
#: (verified 2026-10-28). Bump both this and the distributionUrl together.
_EXPECTED_WRAPPER_JAR_SHA256 = (
    "498495120a03b9a6ab5d155f5de3c8f0d986a449153702fb80fc80e134484f17"
)
_EXPECTED_GRADLE_VERSION = "8.9"


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=_REPO, check=True, capture_output=True, text=True,
    )
    return (out.stdout or "").strip()


def test_all_four_wrapper_files_are_committed() -> None:
    """Three of the four are useless without the fourth."""
    for rel in (
        "gradlew",
        "gradlew.bat",
        "gradle/wrapper/gradle-wrapper.jar",
        "gradle/wrapper/gradle-wrapper.properties",
    ):
        assert (_ANDROID / rel).is_file(), rel
        assert _git("ls-files", "--error-unmatch", f"mobile/android/{rel}")


def test_the_wrapper_jar_is_the_official_gradle_release() -> None:
    """A committed binary that runs on every machine states its identity."""
    digest = hashlib.sha256(
        (_WRAPPER / "gradle-wrapper.jar").read_bytes()
    ).hexdigest()
    assert digest == _EXPECTED_WRAPPER_JAR_SHA256, (
        "the committed gradle-wrapper.jar is not Gradle's published "
        f"{_EXPECTED_GRADLE_VERSION} wrapper jar (got {digest})"
    )


def test_the_properties_pin_the_same_version_as_the_jar() -> None:
    text = (_WRAPPER / "gradle-wrapper.properties").read_text(encoding="utf-8")
    match = re.search(r"distributionUrl=.*gradle-([0-9.]+)-bin\.zip", text)
    assert match, f"no distributionUrl in:\n{text}"
    assert match.group(1) == _EXPECTED_GRADLE_VERSION
    # Gradle checks the URL shape itself before downloading anything.
    assert "validateDistributionUrl=true" in text


def test_gradlew_is_executable_in_git() -> None:
    """Mode 100755, or `./gradlew` cannot be run on macOS or Linux.

    The working-tree bit does not survive a clone; git's index mode is what
    every other machine gets.
    """
    entry = _git("ls-files", "-s", "mobile/android/gradlew")
    assert entry.startswith("100755"), f"expected mode 100755, got: {entry}"


def test_gradlew_bat_is_not_marked_executable() -> None:
    """A Windows batch file has no business carrying the POSIX exec bit."""
    entry = _git("ls-files", "-s", "mobile/android/gradlew.bat")
    assert entry.startswith("100644"), f"expected mode 100644, got: {entry}"


def test_gradlew_is_pinned_to_lf_endings() -> None:
    """`gradlew` has no `.sh` extension, so the repo's `*.sh` rule misses it.

    Left to ``core.autocrlf`` it becomes CRLF in a Windows checkout and then
    fails on macOS/Linux and in Docker with "bad interpreter".
    """
    attrs = _git("check-attr", "eol", "--", "mobile/android/gradlew")
    assert attrs.endswith("eol: lf"), attrs
    # And the committed blob really has no CR, whatever the local checkout did.
    blob = subprocess.run(
        ["git", "show", "HEAD:mobile/android/gradlew"],
        cwd=_REPO, check=False, capture_output=True,
    ).stdout
    if blob:  # empty before the first commit of this file
        assert b"\r\n" not in blob


def test_the_wrapper_jar_is_marked_binary() -> None:
    """An eol conversion would corrupt it beyond use."""
    attrs = _git(
        "check-attr", "binary", "--",
        "mobile/android/gradle/wrapper/gradle-wrapper.jar",
    )
    assert attrs.endswith("binary: set"), attrs


@pytest.mark.skipif(
    not (_ANDROID / "gradlew").is_file(), reason="wrapper not present"
)
def test_the_local_working_copy_is_runnable_on_posix() -> None:
    """On a POSIX checkout the file on disk must also be executable.

    Skipped on Windows, where the bit does not exist and git's index mode
    (asserted above) is the only thing that travels.
    """
    import os

    if os.name == "nt":
        pytest.skip("POSIX permission bits do not exist on Windows")
    mode = (_ANDROID / "gradlew").stat().st_mode
    assert mode & stat.S_IXUSR, "gradlew is not executable in this checkout"
