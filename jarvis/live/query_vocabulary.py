"""Spoken vocabulary for the on-demand tool index: the words, in both languages.

Everything committed to this repo is English (AGENTS.md), tool names and
descriptions included. The user speaks German to it. ``discovery.discover``
ranks keyword overlap, so a spoken query and the index it searches can share no
vocabulary at all: measured on 2026-11-16 over a 25-tool catalog, three of
eight everyday German questions returned ZERO tools where their English twins
returned between three and twenty-one.

This module is the speech-input vocabulary AGENTS.md exempts from the English
rule, kept apart from the ranking so the German words live in one declared
place rather than scattered through logic.

It generalises ``tools.SUBJECT_ALIASES``, which bridges the same gap per tool,
by hand, after a live incident: that table covers 4 of the 147 tools a live
call holds. This one sits on the QUERY, so it reaches every tool in the
catalog, including plugin and MCP tools nobody annotated.

It is a lexical bridge and nothing more. It claims no capability: a bridged
word can only raise tools that already exist.
"""

from __future__ import annotations

from typing import Final

#: Filler words that carry no capability, in both languages.
STOP_WORDS: Final[frozenset[str]] = frozenset(
    (
        "a an and are for in is it of on or the to with jarvis"
        " der die das dem den ein eine einen einem einer ich du mir mich mein"
        " meine meinem meinen was wie wo wann warum und oder auf an von vom zu"
        " zum zur ist sind war bitte kannst kann mal gerade eigentlich doch"
        " noch nicht bei beim im ins aus um dass dir dich sie er es wir ihr"
        " hast habe haben"
    ).split()
)

#: Everyday capability words mapped onto the English stems our tool names and
#: descriptions are written in. Keys are what ``discovery._words`` produces
#: from a spoken query: casefolded, punctuation stripped.
QUERY_BRIDGE: Final[dict[str, tuple[str, ...]]] = {
    "kalender": ("calendar", "event", "appointment", "schedule"),
    "termin": ("calendar", "event", "appointment", "meeting"),
    "nachricht": ("message", "email", "mail", "send", "chat"),
    "mail": ("email", "mail", "message"),
    "schick": ("send", "message"),
    "schicken": ("send", "message"),
    "schreib": ("write", "send", "message"),
    "schreiben": ("write", "send", "message"),
    "anruf": ("call", "phone", "dial"),
    "anrufen": ("call", "phone", "dial"),
    "kontakt": ("contact", "person", "address"),
    "bildschirm": ("screen", "display", "monitor"),
    "bild": ("picture", "image", "screenshot", "photo"),
    "foto": ("photo", "picture", "image", "screenshot"),
    "aufnahme": ("capture", "record", "screenshot"),
    "aufgabe": ("task", "job", "work"),
    "auftrag": ("task", "job", "mission", "order"),
    "laufen": ("run", "running", "active"),
    "laeuft": ("run", "running", "active"),
    "läuft": ("run", "running", "active"),
    "stand": ("status", "state", "progress"),
    "musik": ("music", "song", "audio"),
    "lied": ("music", "song", "track"),
    "spiel": ("play", "playback", "start"),
    "spielen": ("play", "playback"),
    "abspielen": ("play", "playback"),
    "lautstaerke": ("volume", "loudness", "audio"),
    "lautstärke": ("volume", "loudness", "audio"),
    "erinner": ("remember", "memory", "recall"),
    "erinnere": ("remember", "memory", "recall"),
    "erinnerung": ("memory", "remember", "recall", "note"),
    "gedaechtnis": ("memory", "remember", "know", "recall"),
    "gedächtnis": ("memory", "remember", "know", "recall"),
    "merk": ("remember", "memory", "save"),
    "merken": ("remember", "memory", "save"),
    "weisst": ("know", "memory", "about"),
    "weißt": ("know", "memory", "about"),
    "notiz": ("note", "wiki", "vault"),
    "notizen": ("note", "wiki", "vault"),
    "projekt": ("project", "work", "repo"),
    "datei": ("file", "document", "path"),
    "ordner": ("folder", "directory", "path"),
    "oeffne": ("open", "launch", "start"),
    "öffne": ("open", "launch", "start"),
    "oeffnen": ("open", "launch", "start"),
    "öffnen": ("open", "launch", "start"),
    "starte": ("start", "launch", "open", "run"),
    "starten": ("start", "launch", "open", "run"),
    "schliess": ("close", "quit", "stop"),
    "schließ": ("close", "quit", "stop"),
    "fenster": ("window", "app", "screen"),
    "programm": ("app", "application", "program"),
    "anwendung": ("app", "application", "program"),
    "suche": ("search", "find", "look"),
    "suchen": ("search", "find", "look"),
    "finde": ("find", "search", "look"),
    "finden": ("find", "search", "look"),
    "zeig": ("show", "display", "view"),
    "zeige": ("show", "display", "view"),
    "zeigen": ("show", "display", "view"),
    "lies": ("read", "open", "view"),
    "lesen": ("read", "open", "view"),
    "erstell": ("create", "make", "new"),
    "erstelle": ("create", "make", "new"),
    "erstellen": ("create", "make", "new"),
    "loesch": ("delete", "remove"),
    "lösch": ("delete", "remove"),
    "loeschen": ("delete", "remove"),
    "löschen": ("delete", "remove"),
    "aendere": ("change", "edit", "update"),
    "ändere": ("change", "edit", "update"),
    "uebersetze": ("translate", "translation"),
    "übersetze": ("translate", "translation"),
    "wetter": ("weather", "forecast", "temperature"),
    "rechner": ("computer", "machine", "desktop"),
    "einstellung": ("setting", "config", "preference"),
    "einstellungen": ("setting", "config", "preference"),
    "hilfe": ("help", "guide", "documentation"),
    "delegier": ("delegate", "agent", "worker", "spawn"),
    "delegiere": ("delegate", "agent", "worker", "spawn"),
}


def bridge(terms: set[str]) -> set[str]:
    """The query's own words, plus the English stems they stand for."""
    bridged = set(terms)
    for term in terms:
        bridged.update(QUERY_BRIDGE.get(term, ()))
    return bridged
