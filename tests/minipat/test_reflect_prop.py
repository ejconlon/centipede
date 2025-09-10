"""Property-based tests for minipat.reflect using Hypothesis."""

from __future__ import annotations

from fractions import Fraction
from typing import List

from hypothesis import given
from hypothesis import strategies as st

from minipat.arc import CycleArc
from minipat.pat import Pat
from minipat.reflect import minimize_pattern
from minipat.stream import Stream
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def pat_strategy(draw: st.DrawFn, max_depth: int = 2) -> Pat[str]:
    """Generate arbitrary patterns for testing with controlled recursion depth."""

    # Base case - always available
    def pure_pattern() -> Pat[str]:
        value = draw(st.sampled_from(["a", "b", "c", "d"]))
        return Pat.pure(value)

    # If we're at max depth, only generate pure patterns
    if max_depth <= 0:
        return pure_pattern()

    def sequence_pattern() -> Pat[str]:
        # Generate smaller patterns with reduced depth
        elements = draw(
            st.lists(pat_strategy(max_depth=max_depth - 1), min_size=1, max_size=3)
        )
        return Pat.seq(elements)

    def single_seq_pattern() -> Pat[str]:
        inner = draw(pat_strategy(max_depth=max_depth - 1))
        return Pat.seq([inner])

    def repetitive_seq_pattern() -> Pat[str]:
        base_element = draw(pat_strategy(max_depth=max_depth - 1))
        repeat_count = draw(st.integers(min_value=2, max_value=3))
        return Pat.seq([base_element] * repeat_count)

    # Weight pure patterns more heavily to avoid deep nesting
    pattern_type = draw(
        st.sampled_from(
            [
                "pure",
                "pure",
                "pure",  # More weight on pure patterns
                "sequence",
                "single_seq",
                "repetitive_seq",
            ]
        )
    )

    match pattern_type:
        case "pure":
            return pure_pattern()
        case "sequence":
            return sequence_pattern()
        case "single_seq":
            return single_seq_pattern()
        case "repetitive_seq":
            return repetitive_seq_pattern()
        case _:
            return pure_pattern()


def get_cycle_events(pat: Pat[str], cycle: int = 0) -> List[tuple[Fraction, str]]:
    """Get all events from a pattern in the specified cycle.

    Returns list of (start_time_within_cycle, value) tuples, sorted by time.
    """
    stream = Stream.pat(pat)
    arc = CycleArc.cycle(cycle)
    events = stream.unstream(arc)

    # Convert to list of (time_within_cycle, value) tuples
    result = []
    for span, ev in events:
        # Get the start time within the cycle (fractional part)
        cycle_start = Fraction(span.active.start) % 1
        result.append((cycle_start, ev.val))

    # Sort by time within cycle
    result.sort(key=lambda x: x[0])
    return result


@given(pat_strategy())
def test_minimization_preserves_semantics(original_pat: Pat[str]) -> None:
    """Test that pattern minimization preserves semantic equivalence.

    Two patterns are semantically equivalent if they produce the same
    sequence of timed events when evaluated over a cycle.
    """
    minimized_pat = minimize_pattern(original_pat)

    # Test equivalence over multiple cycles to catch cycle-dependent behavior
    for cycle in range(3):  # Test cycles 0, 1, 2
        original_events = get_cycle_events(original_pat, cycle)
        minimized_events = get_cycle_events(minimized_pat, cycle)

        assert original_events == minimized_events, (
            f"Events differ in cycle {cycle}:\n"
            f"Original:  {original_events}\n"
            f"Minimized: {minimized_events}\n"
            f"Original pattern: {original_pat}\n"
            f"Minimized pattern: {minimized_pat}"
        )


@given(pat_strategy())
def test_minimization_is_idempotent(original_pat: Pat[str]) -> None:
    """Test that minimizing an already minimized pattern doesn't change it."""
    first_minimized = minimize_pattern(original_pat)
    second_minimized = minimize_pattern(first_minimized)

    # Should be identical after first minimization
    assert first_minimized == second_minimized, (
        f"Minimization is not idempotent:\n"
        f"Original: {original_pat}\n"
        f"First:    {first_minimized}\n"
        f"Second:   {second_minimized}"
    )


@given(pat_strategy())
def test_specific_minimizations(pat: Pat[str]) -> None:
    """Test specific known minimizations work correctly."""
    # Test single-element sequence minimization
    single_seq = Pat.seq([pat])
    minimized_single = minimize_pattern(single_seq)

    # Should produce same events
    original_events = get_cycle_events(single_seq)
    minimized_events = get_cycle_events(minimized_single)
    assert original_events == minimized_events

    # Test repetition minimization
    triple_seq = Pat.seq([pat, pat, pat])
    minimized_triple = minimize_pattern(triple_seq)

    # Should produce same events
    original_events = get_cycle_events(triple_seq)
    minimized_events = get_cycle_events(minimized_triple)
    assert original_events == minimized_events


# Regression tests for specific patterns that could be problematic
@given(st.lists(st.sampled_from(["x", "y"]), min_size=2, max_size=8))
def test_alternating_patterns(values: List[str]) -> None:
    """Test patterns with alternating values."""
    pat = Pat.seq([Pat.pure(v) for v in values])
    minimized = minimize_pattern(pat)

    original_events = get_cycle_events(pat)
    minimized_events = get_cycle_events(minimized)
    assert original_events == minimized_events


@given(
    st.sampled_from(["kick", "snare", "hihat"]), st.integers(min_value=2, max_value=6)
)
def test_pure_repetition(value: str, repeat_count: int) -> None:
    """Test pure value repeated multiple times."""
    pat = Pat.seq([Pat.pure(value)] * repeat_count)
    minimized = minimize_pattern(pat)

    original_events = get_cycle_events(pat)
    minimized_events = get_cycle_events(minimized)
    assert original_events == minimized_events
