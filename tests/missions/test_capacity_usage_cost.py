"""One usage block, two provider shapes, one bill.

``usage_cost_usd`` reads whatever usage dictionary a provider hands it, and the
providers disagree about one thing: whether the prompt-token count already
contains the cached tokens. Anthropic's ``input_tokens`` does not; an
OpenAI-shaped ``prompt_tokens`` does. Charging the inclusive total at the full
input rate and the cached part again at the cache rate bills the cache twice,
and because the capacity reservation fails closed, an overstated cost parks a
task that would have fit.
"""
from __future__ import annotations

import pytest

from jarvis.brain.cost import cache_read_fraction, resolve_rates
from jarvis.missions.capacity import usage_cost_usd

MODEL = "claude-opus-4-7-20251022"


@pytest.fixture(autouse=True)
def _model_is_priced() -> None:
    if resolve_rates(MODEL) is None:
        pytest.skip(f"{MODEL} is not in the pricing table")


def test_an_exclusive_input_count_is_billed_as_given() -> None:
    """Anthropic: ``input_tokens`` excludes both cache reads and cache writes."""
    in_rate, out_rate = resolve_rates(MODEL)  # type: ignore[misc]
    cached_rate = in_rate * cache_read_fraction(MODEL)

    cost = usage_cost_usd(
        MODEL,
        {"input_tokens": 1_000, "output_tokens": 500, "cache_read_input_tokens": 90_000},
    )

    expected = (1_000 * in_rate + 500 * out_rate + 90_000 * cached_rate) / 1_000_000
    assert cost == pytest.approx(expected)


def test_an_inclusive_input_count_does_not_pay_for_its_cache_twice() -> None:
    """OpenAI: ``prompt_tokens`` is the whole prompt, cached part included."""
    in_rate, out_rate = resolve_rates(MODEL)  # type: ignore[misc]
    cached_rate = in_rate * cache_read_fraction(MODEL)

    cost = usage_cost_usd(
        MODEL,
        {"prompt_tokens": 91_000, "completion_tokens": 500, "cached_input_tokens": 90_000},
    )

    # 1,000 tokens were actually new; the other 90,000 were served from cache.
    expected = (1_000 * in_rate + 500 * out_rate + 90_000 * cached_rate) / 1_000_000
    assert cost == pytest.approx(expected)


def test_the_two_shapes_of_one_call_cost_the_same() -> None:
    """The same call reported two ways must not have two prices."""
    anthropic = usage_cost_usd(
        MODEL,
        {"input_tokens": 1_000, "output_tokens": 500, "cache_read_input_tokens": 90_000},
    )
    openai = usage_cost_usd(
        MODEL,
        {"prompt_tokens": 91_000, "completion_tokens": 500, "cached_input_tokens": 90_000},
    )
    assert anthropic == pytest.approx(openai)


def test_an_inclusive_count_smaller_than_its_cache_never_goes_negative() -> None:
    """An incoherent report must not pay the caller back."""
    cost = usage_cost_usd(
        MODEL, {"prompt_tokens": 10, "completion_tokens": 0, "cached_input_tokens": 90_000}
    )
    assert cost > 0.0


def test_a_cache_write_is_charged_once_on_top_of_an_exclusive_count() -> None:
    """Anthropic bills a cache write separately, at a premium over input."""
    plain = usage_cost_usd(MODEL, {"input_tokens": 1_000, "output_tokens": 0})
    with_write = usage_cost_usd(
        MODEL,
        {"input_tokens": 1_000, "output_tokens": 0, "cache_creation_input_tokens": 1_000},
    )
    assert with_write > plain


def test_a_non_numeric_count_is_read_as_nothing_usable() -> None:
    assert usage_cost_usd(MODEL, {"input_tokens": "lots", "output_tokens": 0}) == 0.0
