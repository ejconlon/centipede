"""Compatibility tests for pat string parsing and event generation."""

from __future__ import annotations

from fractions import Fraction

from minipat.ev import Ev
from minipat.parser import parse_sym_pattern
from minipat.stream import Stream
from minipat.time import CycleArc, CycleSpan, CycleTime


def _test_pattern_events(
    pattern_str: str, expected_events: list[tuple[Fraction, Fraction, str]]
) -> None:
    """Helper function to test pattern consistency across multiple query strategies."""
    pattern = parse_sym_pattern(pattern_str)
    stream = Stream.pat(pattern)

    def _events_to_tuples(
        events: list[tuple[CycleSpan, Ev[str]]],
    ) -> list[tuple[Fraction, Fraction, str]]:
        """Convert events to comparable tuples using whole spans for logical equivalence."""
        return [(ev.span.whole.start, ev.span.whole.end, ev.val) for _, ev in events]

    def _filter_valid_events(
        events: list[tuple[CycleSpan, Ev[str]]],
    ) -> list[tuple[CycleSpan, Ev[str]]]:
        """Filter events to include only those where whole span start is contained in active span.

        This ensures each logical event is counted exactly once when combining results from
        overlapping query arcs, particularly important for slow patterns that create
        long-duration events spanning multiple query boundaries.
        """
        return [
            (span, ev)
            for span, ev in events
            if span.active.start <= span.whole.start < span.active.end
        ]

    # Test strategy 1: (0, 2) in one call - use as golden truth
    arc_full = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    events_full = stream.unstream(arc_full)
    events_full_sorted = sorted(events_full, key=lambda x: x[0].active.start)
    full_tuples = _events_to_tuples(events_full_sorted)

    # Verify against expected events
    assert full_tuples == expected_events, (
        f"Pattern '{pattern_str}' full arc (0,2): Expected {expected_events}, got {full_tuples}"
    )

    # Test strategy 2: (0, 1) + (1, 2) in two calls - should match full query
    arc1 = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    arc2 = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    events1 = stream.unstream(arc1)
    events2 = stream.unstream(arc2)
    events_split = sorted(
        list(events1) + list(events2), key=lambda x: x[0].active.start
    )
    split_tuples = _events_to_tuples(events_split)

    assert split_tuples == full_tuples, (
        f"Pattern '{pattern_str}' split arcs (0,1)+(1,2): Expected {full_tuples}, got {split_tuples}"
    )

    # Test strategy 3: (0, 0.5) + (0.5, 1) + (1, 1.5) + (1.5, 2) in quarters
    quarter_arcs = [
        CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1, 2))),  # (0, 0.5)
        CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(1))),  # (0.5, 1)
        CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(3, 2))),  # (1, 1.5)
        CycleArc(CycleTime(Fraction(3, 2)), CycleTime(Fraction(2))),  # (1.5, 2)
    ]
    quarter_events: list[tuple[CycleSpan, Ev[str]]] = []
    for arc in quarter_arcs:
        events = stream.unstream(arc)
        quarter_events.extend(events)

    # Filter quarter events to avoid double-counting overlapping events
    quarter_events_filtered = _filter_valid_events(quarter_events)
    quarter_events_sorted = sorted(
        quarter_events_filtered, key=lambda x: x[0].active.start
    )
    quarter_tuples = _events_to_tuples(quarter_events_sorted)

    assert quarter_tuples == full_tuples, (
        f"Pattern '{pattern_str}' quarter arcs: Expected {full_tuples}, got {quarter_tuples}"
    )

    # Test strategy 3: Partial arcs - just verify they run without error and produce sensible results
    # We don't check exact values since partial arc queries can have different event boundaries
    test_arcs = [
        CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(3, 2))),  # (0.5, 1.5)
        CycleArc(CycleTime(Fraction(2, 3)), CycleTime(Fraction(4, 3))),  # (2/3, 4/3)
    ]

    for arc in test_arcs:
        events = stream.unstream(arc)
        # Just verify events are within the arc bounds and values are reasonable
        for _, ev in events:
            # Events should intersect with the arc (start or end should be within bounds)
            intersects = (
                ev.span.active.start < arc.end and ev.span.active.end > arc.start
            )
            assert intersects, (
                f"Pattern '{pattern_str}' arc {arc}: Event ({ev.span.active.start} to {ev.span.active.end}) doesn't intersect with arc"
            )
            assert ev.val in [val for _, _, val in expected_events], (
                f"Pattern '{pattern_str}' arc {arc}: Unexpected event value '{ev.val}'"
            )


def test_simple_replication() -> None:
    """Test basic replication without alternation."""
    # bd!3 should produce 3 bd events per cycle, 6 total over 2 cycles
    _test_pattern_events(
        "bd!3",
        [
            # Cycle 0
            (Fraction(0), Fraction(1, 3), "bd"),
            (Fraction(1, 3), Fraction(2, 3), "bd"),
            (Fraction(2, 3), Fraction(1), "bd"),
            # Cycle 1
            (Fraction(1), Fraction(4, 3), "bd"),
            (Fraction(4, 3), Fraction(5, 3), "bd"),
            (Fraction(5, 3), Fraction(2), "bd"),
        ],
    )


def test_alternation_across_cycles() -> None:
    """Test alternation behavior changes across cycles."""
    # <hh oh> should alternate: hh in cycle 0, oh in cycle 1
    _test_pattern_events(
        "<hh oh>",
        [
            (Fraction(0), Fraction(1), "hh"),  # Cycle 0: hh
            (Fraction(1), Fraction(2), "oh"),  # Cycle 1: oh (alternation)
        ],
    )


def test_replication_with_alternation() -> None:
    """Test replication with alternation across cycles."""
    # [bd <hh oh>]!2 should produce 2 repetitions per cycle with proper alternation
    _test_pattern_events(
        "[bd <hh oh>]!2",
        [
            # Cycle 0 (alternation uses "hh")
            (Fraction(0), Fraction(1, 4), "bd"),
            (Fraction(1, 4), Fraction(1, 2), "hh"),
            (Fraction(1, 2), Fraction(3, 4), "bd"),
            (Fraction(3, 4), Fraction(1), "hh"),
            # Cycle 1 (alternation switches to "oh")
            (Fraction(1), Fraction(5, 4), "bd"),
            (Fraction(5, 4), Fraction(3, 2), "oh"),
            (Fraction(3, 2), Fraction(7, 4), "bd"),
            (Fraction(7, 4), Fraction(2), "oh"),
        ],
    )


def test_fast_alternation() -> None:
    """Test fast operator with alternation."""
    # [bd <hh oh>]*2 speeds up pattern by 2x, fitting 2 repetitions per cycle
    _test_pattern_events(
        "[bd <hh oh>]*2",
        [
            # Cycle 0 (alternation uses "hh")
            (Fraction(0), Fraction(1, 4), "bd"),
            (Fraction(1, 4), Fraction(1, 2), "hh"),
            (Fraction(1, 2), Fraction(3, 4), "bd"),
            (Fraction(3, 4), Fraction(1), "hh"),
            # Cycle 1 (alternation switches to "oh")
            (Fraction(1), Fraction(5, 4), "bd"),
            (Fraction(5, 4), Fraction(3, 2), "oh"),
            (Fraction(3, 2), Fraction(7, 4), "bd"),
            (Fraction(7, 4), Fraction(2), "oh"),
        ],
    )


def test_slow_alternation() -> None:
    """Test slow operator with alternation."""
    # [bd <hh oh>]/2 slows down the pattern by 2x, taking 2 cycles to complete
    _test_pattern_events(
        "[bd <hh oh>]/2",
        [
            (Fraction(0), Fraction(1), "bd"),  # bd (0, 1)
            (
                Fraction(1),
                Fraction(2),
                "hh",
            ),  # hh (1, 2) - alternation produces hh in first cycle
        ],
    )


def test_single_element() -> None:
    """Test single element pattern."""
    # "c" should produce one event per cycle, repeated over 2 cycles
    _test_pattern_events(
        "c",
        [
            (Fraction(0), Fraction(1), "c"),  # Cycle 0: c
            (Fraction(1), Fraction(2), "c"),  # Cycle 1: c (repeated)
        ],
    )


def test_four_element_sequence() -> None:
    """Test four element sequence pattern."""
    # "c d e f" should produce 4 events per cycle, each lasting 1/4 cycle
    _test_pattern_events(
        "c d e f",
        [
            # Cycle 0
            (Fraction(0), Fraction(1, 4), "c"),
            (Fraction(1, 4), Fraction(1, 2), "d"),
            (Fraction(1, 2), Fraction(3, 4), "e"),
            (Fraction(3, 4), Fraction(1), "f"),
            # Cycle 1
            (Fraction(1), Fraction(5, 4), "c"),
            (Fraction(5, 4), Fraction(3, 2), "d"),
            (Fraction(3, 2), Fraction(7, 4), "e"),
            (Fraction(7, 4), Fraction(2), "f"),
        ],
    )
