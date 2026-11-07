"""What a Live session actually costs, counted once instead of every turn.

A Live session keeps its context on the server. The client does not re-send
the system prompt and tool declarations, but ``prompt_token_count`` reports
everything the model attended to, so it GROWS turn by turn. Summing that
field bills the same context again on every turn.

Measured 2026-11-07 against Google's own quota counter: our ledger made
138 339 input tokens out of 12 model turns where Google recorded 12 370 for
the whole day. A later session read 10 353 then 10 454 -- 101 genuinely new
tokens reported as 20 807.
"""
from __future__ import annotations

import inspect

from jarvis.live.native import split_new_input_tokens


class TestSplitNewInputTokens:
    def test_first_turn_counts_the_whole_context(self) -> None:
        """The opening turn really does pay for the prompt and the catalog."""
        new_total, text_in, audio_in = split_new_input_tokens(
            {"input_total": 10_353, "input_audio": 75}, previous_total=0
        )
        assert new_total == 10_353
        assert audio_in == 75
        assert text_in == 10_278

    def test_later_turn_counts_only_what_was_added(self) -> None:
        new_total, text_in, audio_in = split_new_input_tokens(
            {"input_total": 10_454, "input_audio": 75}, previous_total=10_353
        )
        assert new_total == 101
        # The speech is new; the rest of the growth is new text.
        assert audio_in == 75
        assert text_in == 26

    def test_audio_never_exceeds_the_growth_it_belongs_to(self) -> None:
        """A provider that reports cumulative audio must not invent tokens."""
        new_total, text_in, audio_in = split_new_input_tokens(
            {"input_total": 10_400, "input_audio": 9_000}, previous_total=10_353
        )
        assert new_total == 47
        assert audio_in == 47
        assert text_in == 0

    def test_a_shrunken_context_was_compacted_and_is_read_again(self) -> None:
        """Compaction rebuilds the context, so the model re-reads all of it."""
        new_total, _text_in, _audio_in = split_new_input_tokens(
            {"input_total": 4_000, "input_audio": 50}, previous_total=12_000
        )
        assert new_total == 4_000

    def test_an_unchanged_total_adds_nothing(self) -> None:
        new_total, text_in, audio_in = split_new_input_tokens(
            {"input_total": 10_353, "input_audio": 0}, previous_total=10_353
        )
        assert (new_total, text_in, audio_in) == (0, 0, 0)

    def test_a_session_sums_to_its_last_reading(self) -> None:
        """The whole point: the session's bill is its final context, not the sum.

        The 2026-11-07 session read 10 353 then 10 454. Counted by growth that
        is 10 454 in total -- exactly the context the model ended up holding --
        rather than the 20 807 a naive sum produces.
        """
        totals = [10_353, 10_454]
        previous = 0
        billed = 0
        for total in totals:
            new_total, _t, _a = split_new_input_tokens(
                {"input_total": total, "input_audio": 0}, previous_total=previous
            )
            billed += new_total
            previous = total
        assert billed == 10_454


class TestTheClosingRecordCarriesTheTotals:
    """The spend has to reach the event, not just the accumulator.

    The first attempt at this fix landed in ``jarvis/realtime/session.py`` --
    a second, unused Live implementation. Its tests were green and the running
    app was unchanged, because the app runs ``jarvis/live/``. A missing keyword
    on an event whose fields carry defaults fails silently: it publishes zeros.
    """

    def test_voice_turn_completed_is_given_the_session_spend(self) -> None:
        from jarvis.live import session as live_session

        source = inspect.getsource(live_session.LiveVoiceSession)
        start = source.index("VoiceTurnCompleted(")
        call = source[start : start + 1500]
        for field in ("tokens_in=", "tokens_out=", "cost_usd="):
            assert field in call, f"the closing record still defaults {field}0"

    def test_the_accumulators_exist_before_the_first_usage_event(self) -> None:
        from jarvis.live.session import LiveVoiceSession

        source = inspect.getsource(LiveVoiceSession.__init__)
        assert "_live_input_high_water" in source
        assert "_live_spend" in source

    def test_the_running_path_is_the_one_that_counts(self) -> None:
        """jarvis/live is where the desktop app's voice actually runs."""
        from jarvis.live.native import split_new_input_tokens

        assert callable(split_new_input_tokens)
