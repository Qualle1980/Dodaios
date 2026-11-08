# ADR-0044 — One personal memory vault

**Status:** Accepted · **Date:** 2026-11-08

## Context

What Dodaios knows about its user is spread over several stores, and none of
them can answer three ordinary questions: *what do you know about me?*, *how
did that fact change?*, and *give me my memory as a file I can take with me.*

The wiki is page-shaped. A page is a document, so a changed fact becomes a
second paragraph next to the first and both stay true-looking. The assistant
then has to guess which is current, and the user has to argue with it about
which is right. Appending more pages cannot fix that; the storage unit is
wrong.

This decision adopts the upstream `Dodaios/Dodaios` ADR-0044 design and ports
its implementation (`dodaios/memory/vault/`, MIT, same project upstream). The
numbering is kept so code comments, tests and the port plan cite one number in
both trees.

## Decision

**An atomic, bi-temporal memory store in one SQLite file.**

1. **Atomic memories, not pages.** One statement per memory, typed
   (`profile`, `preference`, `relationship`, `project`, `procedural`,
   `episode`, `reference`, `ephemeral`), with its origin and source. A page is
   a view; it is never the unit of storage.
2. **Supersede, never overwrite.** A changed fact closes the old row with a
   date and writes a new one that points back at it. The store can therefore
   answer both *now* and *before*, and a contradiction is a timeline rather
   than an argument.
3. **Forgetting erases.** `forget` purges the text from the row, the full-text
   index and the audit log, across every version in the chain, and leaves a
   tombstone. An inferred write can never bring a forgotten fact back; only
   the user can.
4. **Secrets are refused at the door.** Every write path — `remember`,
   `supersede`, `insert_imported` — runs the regex-only guard in
   `jarvis/memory/wiki/secret_guard.py` (AP-2). A refused write reaches
   neither the row, the index, nor a voice readback.
5. **A small profile block is always in the prompt; the rest is searched per
   turn.** `profile_selection` orders pinned memories first, then the stable
   kinds, and cuts whole lines to a character budget. Episodes and anything
   marked `sensitive` stay out of the block and reach the model only through
   search.

   That block is **not a new one.** `jarvis/brain/identity_card.py` already
   held the hard character budget, the byte-stable caching that keeps a
   provider prompt cache intact, and the "never read this out" contract. So
   the vault became a third *source* of that card, placed first and sharing
   the same budget: the prompt does not grow by a character, and a vault
   memory outranks a line distilled from a note because only the memory can be
   superseded, pinned and forgotten. As the vault fills, the notes fade out of
   the card on their own — no flag day.

   **Every surface carries it.** The typed brain, a live call and a realtime
   session each build their own instruction stack, and only the first had the
   card: a call could not use a standing fact about the person it was speaking
   to. Both voice paths now append it at a **compact** budget — half the typed
   one, floored so the person's own name cannot drop out — because a voice
   session re-pays its instructions on every turn, the provider holding the
   session context.

   **The per-message half is the typed path only**, and that is a decision, not
   an omission. It rides the same suffix as the wiki snippets, so
   cache-optimized mode moves both onto the user message and the cached system
   prompt stays byte-stable. A live session has no per-turn channel for it at
   all, and a realtime session re-sends its whole instruction block every turn,
   where a block that changes per message is exactly what defeats a
   prefix-caching server. Both already have the standing card and the
   `memory-search` tool, so a call can reach the same memories — on demand,
   paid for once, instead of on every turn. Revisit it with a measurement, not
   a guess.

   Two rules make the per-message block worth its tokens: it never repeats what
   the standing card already holds, so a memory is not paid for twice in one
   prompt, and a hit must score within a share of the best hit — keyword search
   always returns something, and "how is the weather" must not drag in whatever
   shares a word with it.
6. **Portable.** A bundle is a manifest, one JSONL per kind, entities, the
   audit log, tombstones and `CHECKSUMS`, as a folder or a zip. Import applies
   tombstones first and skips anything already known, so importing the same
   bundle twice changes nothing. A readable Markdown copy is exported beside
   it.
7. **Being surfaced is not evidence of having helped.** Two counters, not one:
   `use_count` records that a memory was put in front of the model,
   `helpful_count` that something confirmed it helped. Only the second decides
   what the always-on block holds. Ordering the block by surfacing makes the
   block feed itself, because being in it is what gets a memory surfaced.
8. **Confidence can fall.** `mark_unhelpful` lowers it and is audited; below
   `PROFILE_MIN_CONFIDENCE` a memory leaves the always-on block while staying
   searchable and intact in its history. A signal that changes nothing would be
   theatre. It stops above zero, because removing a memory is what forgetting
   is for, and a pin outranks the floor: that is the user's own word.
9. **An inference enters as a candidate.** The store caps the confidence of any
   write whose origin is not `user` at `INFERRED_MAX_CONFIDENCE`, in the one
   place every write passes, so no caller — including a model that answers
   `confidence: 1.0`, and including an imported bundle row — can present a
   guess as something the user stated.

Principles 7 to 9 are not upstream's. They are failures the AK Matrix memory
import plan recorded against its own store: *"hit_count currently means
surfaced/recalled, not proven useful"*, *"mere retrieval must not count as
success"*, *"Confidence must be able to go down when retrieved memory was
irrelevant or harmful"* and *"Default imported inferences to
candidate/unverified"*. The ported store arrived with the first two, so they
are corrected before anything writes to it.

**The vault does not replace the wiki.** `[wiki_integration].external_sources`
is this fork's own extension: it registers read-only folders Dodaios searches
but never writes, and the vault has no concept of them. Upstream's last step
removes the wiki; that step does not apply here and is not planned. The two run
side by side — the vault for what Dodaios learns about its user, the wiki for
the read-only sources.

Deliberately not part of this decision: embeddings. Search is BM25 over FTS5
with entity and recency boosts, and it works without a vector store. An
optional embedding table can be added later without changing anything above.

## Consequences

- A contradiction has one answer, with a date, instead of two paragraphs.
- A place in the always-on block is earned, not accumulated: a memory that
  nothing confirmed stays where it was, however often it was shown.
- Every write is auditable, and anything written can be taken away whole.
- The vault's file (`data/memory/vault.db`) is separate from the wiki's
  (`data/jarvis.db`) on purpose: someone must be able to export or delete their
  personal memory without touching the wiki's pages.
- Two memories exist during the rollout. The prompt must not read from both, or
  every turn pays twice; the wiki injector is replaced in the step that adds
  the profile block, not after it.
- A permanent profile block grows every prompt, and a live voice session pays it
  again per turn because the server holds the context. `profile_max_chars` is
  the lever, and the effect is measured at `tokens_in`, never estimated.
- The background writer that proposes memories is exactly the kind of
  long-running job that once burned EUR 5 in three hours. It belongs on
  `jarvis/brain/background_policy.py` with a call cap.

## Alternatives

- **Harden the wiki further.** Rejected: the page is the wrong storage unit, so
  better search and warmer caches still cannot say which of two paragraphs is
  current. (The hardening already landed is not wasted — the wiki keeps the
  external sources regardless.)
- **Build our own store.** Rejected: upstream had already decided this and
  shipped it in six pieces. Porting keeps one schema, one bundle format and one
  ADR number across both trees.
- **Embeddings from the start.** Rejected for now: it adds a model dependency
  and an index to keep warm for a gain the keyword search has not yet been
  shown to need.
