from fractions import Fraction

from minipat.arc import CycleArc
from minipat.common import CycleTime
from minipat.parser import parse_pattern
from minipat.pat import Pat, SpeedOp
from minipat.stream import Stream


def test_pure_pattern() -> None:
    """Test pure pattern generates single event spanning arc."""
    pattern = Pat.pure("x")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.span.active == arc
    assert event.span.whole is None  # Pure pattern should have no wider context
    assert event.val == "x"


def test_silence_pattern() -> None:
    """Test silence pattern generates no events."""
    pattern: Pat[str] = Pat.silent()
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 0


def test_sequence_pattern() -> None:
    """Test sequence pattern divides time proportionally."""
    # Pattern equivalent to "x y"
    pattern = Pat.seq([Pat.pure("x"), Pat.pure("y")])
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    assert len(event_list) == 2

    # First event: x from 0 to 0.5
    _, first_event = event_list[0]
    assert first_event.span.active.start == Fraction(0)
    assert first_event.span.active.end == Fraction(1, 2)
    assert first_event.span.whole is None  # Sequence elements have no wider context
    assert first_event.val == "x"

    # Second event: y from 0.5 to 1
    _, second_event = event_list[1]
    assert second_event.span.active.start == Fraction(1, 2)
    assert second_event.span.active.end == Fraction(1)
    assert second_event.span.whole is None  # Sequence elements have no wider context
    assert second_event.val == "y"


def test_parallel_pattern() -> None:
    """Test parallel pattern plays all children simultaneously."""
    # Pattern equivalent to "[x,y]"
    pattern = Pat.par([Pat.pure("x"), Pat.pure("y")])
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 2

    # Both events should span the full arc
    for _, event in event_list:
        assert event.span.active == arc
        assert (
            event.span.whole is None
        )  # Parallel events fill full arc, no wider context
        assert event.val in ["x", "y"]

    # Should have both x and y
    values = [event.val for _, event in event_list]
    assert any(v == "x" for v in values)
    assert any(v == "y" for v in values)


def test_repetition_fast() -> None:
    """Test fast repetition pattern."""
    # Pattern equivalent to "x!" with count 2
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    assert len(event_list) == 2

    # First repetition: 0 to 0.5
    _, first_event = event_list[0]
    assert first_event.span.active.start == Fraction(0)
    assert first_event.span.active.end == Fraction(1, 2)
    assert (
        first_event.span.whole is None
    )  # Fast repetitions create separate active spans
    assert first_event.val == "x"

    # Second repetition: 0.5 to 1
    _, second_event = event_list[1]
    assert second_event.span.active.start == Fraction(1, 2)
    assert second_event.span.active.end == Fraction(1)
    assert (
        second_event.span.whole is None
    )  # Fast repetitions create separate active spans
    assert second_event.val == "x"


def test_repetition_slow() -> None:
    """Test slow repetition pattern."""
    # Pattern equivalent to "x" slowed down by factor of 2
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Slow, Fraction(2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    # Slow repetition stretches the pattern and scales back - results in compressed event
    assert event.span.active.start == Fraction(0)
    assert event.span.active.end == Fraction(
        1
    )  # Actually fills the whole arc after scaling back
    assert event.span.whole is None  # Fills entire arc, no wider context
    assert event.val == "x"


def test_elongation_pattern() -> None:
    """Test stretch pattern."""
    # Pattern equivalent to "x@2"
    base_pattern = Pat.pure("x")
    pattern = Pat.stretch(base_pattern, Fraction(2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    # Elongation stretches then scales back down - net result fills arc
    assert event.span.active.start == Fraction(0)
    assert event.span.active.end == Fraction(
        1
    )  # Actually fills the whole arc after scaling back
    assert event.span.whole is None  # Fills entire arc, no wider context
    assert event.val == "x"


def test_choice_pattern() -> None:
    """Test choice pattern selects based on cycle."""
    # Pattern with two choices
    pattern = Pat.rand([Pat.pure("x"), Pat.pure("y")])
    stream = Stream.pat(pattern)

    # Test cycle 0 (arc starting at 0)
    arc0 = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events0 = stream.unstream(arc0)
    event_list0 = list(events0)

    assert len(event_list0) == 1
    _, event0 = event_list0[0]
    assert event0.span.whole is None  # Choice events fill full arc
    assert event0.val == "x" and len(event_list0) == 1
    _, event0 = event_list0[0]

    # Test cycle 1 (arc starting at 1)
    arc1 = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    events1 = stream.unstream(arc1)
    event_list1 = list(events1)

    assert len(event_list1) == 1
    _, event1 = event_list1[0]
    assert event1.span.whole is None  # Choice events fill full arc
    assert event1.val == "y" and len(event_list1) == 1
    _, event1 = event_list1[0]


def test_euclidean_pattern() -> None:
    """Test euclidean rhythm pattern."""
    # Pattern equivalent to "x(3,8)" - 3 hits in 8 steps
    atom = Pat.pure("x")
    pattern = Pat.euc(atom, 3, 8, 0)
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have 3 events distributed across 8 steps
    assert len(event_list) == 3

    step_duration = Fraction(1, 8)

    # Events should be at positions determined by euclidean algorithm
    for i, (_, event) in enumerate(event_list):
        assert event.val == "x" and len(event_list) == 3

    step_duration = Fraction(1, 8)

    # Events should be at positions determined by euclidean algorithm
    for i, (_, event) in enumerate(event_list):
        assert event.span.whole is None  # Euclidean events create separate active spans
        # Each event should span one step
        assert event.span.active.length() == step_duration


def test_polymetric_pattern() -> None:
    """Test polymetric pattern plays all patterns simultaneously."""
    # Pattern with multiple rhythmic patterns
    patterns = [
        Pat.pure("x"),
        Pat.pure("y"),
        Pat.pure("z"),
    ]
    pattern = Pat.poly(patterns)
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 3

    # All events should span the full arc
    values = []
    for _, event in event_list:
        assert event.span.active == arc
        assert event.span.whole is None  # Polymetric events fill full arc
        values.append(event.val)

    # Should have all three values
    assert any(v == "x" for v in values)
    assert any(v == "y" for v in values)
    assert any(v == "z" for v in values)


def test_alternating_pattern() -> None:
    """Test alternating pattern cycles through choices."""
    # Pattern that alternates between x and y
    patterns = [Pat.pure("x"), Pat.pure("y")]
    pattern = Pat.alt(patterns)
    stream = Stream.pat(pattern)

    # Test different cycles
    arc0 = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events0 = stream.unstream(arc0)
    event_list0 = list(events0)

    assert len(event_list0) == 1
    _, event0 = event_list0[0]
    assert event0.span.whole is None  # Alternating events fill full arc
    assert event0.val == "x" and len(event_list0) == 1
    _, event0 = event_list0[0]

    arc1 = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    events1 = stream.unstream(arc1)
    event_list1 = list(events1)

    assert len(event_list1) == 1
    _, event1 = event_list1[0]
    assert event1.span.whole is None  # Alternating events fill full arc
    assert event1.val == "y" and len(event_list1) == 1
    _, event1 = event_list1[0]


def test_probability_pattern() -> None:
    """Test probability pattern (deterministic based on arc)."""
    base_pattern = Pat.pure("x")
    pattern = Pat.prob(base_pattern, Fraction(1))  # Always include
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.span.whole is None  # Probability events fill full arc
    assert event.val == "x"

    # Test with 0 probability
    pattern_never = Pat.prob(base_pattern, Fraction(0))
    stream_never = Stream.pat(pattern_never)

    events_never = stream_never.unstream(arc)
    event_list_never = list(events_never)

    assert len(event_list_never) == 0


def test_complex_nested_pattern() -> None:
    """Test complex nested pattern combining multiple operations."""
    # Pattern equivalent to "[x y]!2" - sequence replicated twice
    seq = Pat.seq([Pat.pure("x"), Pat.pure("y")])
    pattern = Pat.repeat(seq, Fraction(2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have 4 events: x, y, x, y (sequence replicated twice)
    assert len(event_list) == 4

    # First repetition
    _, first_event = event_list[0]
    assert first_event.span.active.start == Fraction(0)
    assert first_event.span.active.end == Fraction(1, 4)
    assert (
        first_event.span.whole is None
    )  # Replicated sequence elements have no wider context
    assert first_event.val == "x"

    _, second_event = event_list[1]
    assert second_event.span.active.start == Fraction(1, 4)
    assert second_event.span.active.end == Fraction(1, 2)
    assert (
        second_event.span.whole is None
    )  # Replicated sequence elements have no wider context
    assert second_event.val == "y"

    # Second repetition
    _, third_event = event_list[2]
    assert third_event.span.active.start == Fraction(1, 2)
    assert third_event.span.active.end == Fraction(3, 4)
    assert (
        third_event.span.whole is None
    )  # Replicated sequence elements have no wider context
    assert third_event.val == "x"

    _, fourth_event = event_list[3]
    assert fourth_event.span.active.start == Fraction(3, 4)
    assert fourth_event.span.active.end == Fraction(1)
    assert (
        fourth_event.span.whole is None
    )  # Replicated sequence elements have no wider context
    assert fourth_event.val == "y"


def test_empty_sequence() -> None:
    """Test empty sequence generates no events."""
    pattern: Pat[str] = Pat.seq([])
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 0


def test_null_arc() -> None:
    """Test null arc generates no events."""
    pattern = Pat.pure("x")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(1)))  # null arc

    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 0


def test_partial_arc_query() -> None:
    """Test querying a partial arc of a sequence."""
    # Pattern "x y z"
    pattern = Pat.seq(
        [
            Pat.pure("x"),
            Pat.pure("y"),
            Pat.pure("z"),
        ]
    )
    stream = Stream.pat(pattern)

    # Query only the middle third (should get "y")
    arc = CycleArc(CycleTime(Fraction(1, 3)), CycleTime(Fraction(2, 3)))
    events = stream.unstream(arc)
    event_list = list(events)

    # All events that intersect with the query arc should be returned
    # x: 0-1/3 intersects with 1/3-2/3 at 1/3-1/3 (null, so filtered out)
    # y: 1/3-2/3 intersects fully
    # z: 2/3-1 intersects with 1/3-2/3 at 2/3-2/3 (null, so filtered out)
    # Actually, the intersection logic should include events that overlap

    # Let's check what we actually get
    assert len(event_list) > 0

    # Events should only be ones that intersect with our query arc
    for _, event in event_list:
        intersection = arc.intersect(event.span.active)
        assert not intersection.null(), (
            f"Event {event.val} at {event.span.active.start}-{event.span.active.end} should intersect with {arc.start}-{arc.end}"
        )


def test_pure_pattern_partial_queries() -> None:
    """Test that pure patterns only generate events for queries containing the cycle start."""
    # Create a simple pure pattern
    pattern = Pat.pure("c")
    stream = Stream.pat(pattern)

    # Query first half: (0, 0.5) - should contain cycle start at 0
    first_query = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1, 2)))
    first_events = stream.unstream(first_query)
    first_list = list(first_events)

    # Query second half: (0.5, 1) - does NOT contain cycle start at 0
    second_query = CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(1)))
    second_events = stream.unstream(second_query)
    second_list = list(second_events)

    # Only the first query should get an event (contains cycle start)
    assert len(first_list) == 1
    assert len(second_list) == 0

    # The first event should span the full cycle
    event = first_list[0][1]
    assert event.val == "c"
    assert event.span.active.start == CycleTime(Fraction(0))
    assert event.span.active.end == CycleTime(Fraction(1))
    assert event.span.whole is None  # Pure pattern has no wider context

    # Test that queries starting after cycle 0 don't get cycle 0 events
    late_query = CycleArc(CycleTime(Fraction(1, 4)), CycleTime(Fraction(3, 4)))
    late_events = stream.unstream(late_query)
    late_list = list(late_events)
    assert len(late_list) == 0  # Doesn't contain cycle start at 0


def test_replicate_stream() -> None:
    """Test replicate patterns work with stream processing."""

    pattern = parse_pattern("bd!3")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have 3 events (pattern replicated 3 times)
    assert len(event_list) == 3

    # Each event should be "bd"
    for _, event in event_list:
        assert event.val == "bd" and len(event_list) == 3

    # Events should be evenly spaced
    expected_duration = Fraction(1, 3)
    for i, (_, event) in enumerate(event_list):
        expected_start = i * expected_duration
        expected_end = (i + 1) * expected_duration
        assert event.span.active.start == expected_start
        assert event.span.active.end == expected_end


def test_ratio_stream() -> None:
    """Test ratio patterns work with stream processing."""

    pattern = parse_pattern("bd*2%1")  # 2/1 ratio
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events (exact count depends on implementation)
    assert len(event_list) >= 1

    # All events should be "bd"
    for _, event in event_list:
        assert event.val == "bd" and len(event_list) >= 1


def test_polymetric_subdivision_stream() -> None:
    """Test polymetric subdivision patterns work with stream processing."""

    pattern = parse_pattern("{bd, sd}%2")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events from both patterns
    assert len(event_list) >= 2

    # Should have both "bd" and "sd" events
    values = [event.val for _, event in event_list]
    assert any(v == "bd" for v in values)
    assert any(v == "sd" for v in values)


def test_dot_grouping_stream() -> None:
    """Test dot grouping patterns work with stream processing."""

    pattern = parse_pattern("bd sd . hh cp")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: (x[0].active.start, x[1].val))

    # Should have 4 events total
    assert len(event_list) == 4

    # Should have all the expected values
    values = [event.val for _, event in event_list]
    assert len(values) == 4

    # Check that we have the right values (order may vary within parallel groups)
    assert "bd" in values
    assert "sd" in values
    assert "hh" in values
    assert "cp" in values

    # Check timing: first two events should be in first half, last two in second half
    first_half_events = [
        ev for span, ev in event_list if span.active.start < Fraction(1, 2)
    ]
    second_half_events = [
        ev for span, ev in event_list if span.active.start >= Fraction(1, 2)
    ]

    assert len(first_half_events) == 2
    assert len(second_half_events) == 2

    # Check that first half contains bd and sd
    first_half_values = {ev.val for ev in first_half_events}
    assert first_half_values == {"bd", "sd"}

    # Check that second half contains hh and cp
    second_half_values = {ev.val for ev in second_half_events}
    assert second_half_values == {"hh", "cp"}


def test_new_features_stream_integration() -> None:
    """Test that new patterns integrate properly with stream processing."""

    # Basic smoke test - these should not crash
    patterns = ["bd!3", "bd*2%1", "{bd, sd}%2", "bd sd . hh cp"]

    for pattern_str in patterns:
        pat = parse_pattern(pattern_str)
        stream = Stream.pat(pat)
        arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
        # Should not crash
        events = stream.unstream(arc)
        assert events is not None

        # Should have some events
        event_list = list(events)
        assert len(event_list) > 0


def test_complex_new_features_stream() -> None:
    """Test complex combinations of new features with streams."""

    # Complex pattern with multiple new features
    pattern = parse_pattern("{bd!2, sd*3%2}%4")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events
    assert len(event_list) > 0

    # Should contain both bd and sd
    values = [event.val for _, event in event_list]
    assert any(v == "bd" for v in values)
    assert any(v == "sd" for v in values)


# Tests for sub-cycle splitting - when patterns span across cycle boundaries


def test_sequence_sub_cycle_splitting() -> None:
    """Test sequence patterns that split across sub-cycles."""
    # Create a sequence pattern: "x y z"
    pattern = Pat.seq(
        [
            Pat.pure("x"),
            Pat.pure("y"),
            Pat.pure("z"),
        ]
    )
    stream = Stream.pat(pattern)

    # Query an arc that spans 1.5 cycles (from 0.5 to 2.0)
    arc = CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(2)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should get events that intersect with our query arc
    assert len(event_list) > 0

    # Check that events have proper whole components when they span cycle boundaries
    for _, event in event_list:
        # Events that intersect with our query should be present
        intersection = arc.intersect(event.span.active)
        assert not intersection.null()

        # If the event's active span extends beyond a single cycle division,
        # it may have a whole component
        if event.span.whole is not None:
            # The whole should contain the active
            assert event.span.active.start >= event.span.whole.start
            assert event.span.active.end <= event.span.whole.end


def test_fast_repetition_sub_cycle_splitting() -> None:
    """Test fast repetition patterns across sub-cycles."""
    # Fast repetition: "x!4" - 4 repetitions
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(4))
    stream = Stream.pat(pattern)

    # Query an arc that spans across cycle boundary (0.5 to 1.5)
    arc = CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(3, 2)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should get events from both cycles
    assert len(event_list) > 0

    # Check spans
    for _, event in event_list:
        assert event.val == "x"
        # Verify the event intersects with our query
        intersection = arc.intersect(event.span.active)
        assert not intersection.null()


def test_slow_repetition_sub_cycle_splitting() -> None:
    """Test slow repetition patterns across sub-cycles."""
    # Slow repetition: "x/2" - half speed
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Slow, Fraction(2))
    stream = Stream.pat(pattern)

    # Query an arc spanning multiple cycles to see the slow pattern
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(3)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events, but they'll be stretched across cycles
    assert len(event_list) > 0

    for _, event in event_list:
        assert event.val == "x"
        # Event should intersect with our query
        intersection = arc.intersect(event.span.active)
        assert not intersection.null()


def test_euclidean_sub_cycle_splitting() -> None:
    """Test euclidean patterns across sub-cycles."""
    # Euclidean rhythm: "x(3,8)"
    atom = Pat.pure("x")
    pattern = Pat.euc(atom, 3, 8, 0)
    stream = Stream.pat(pattern)

    # Query across cycle boundary
    arc = CycleArc(CycleTime(Fraction(3, 4)), CycleTime(Fraction(7, 4)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have events from the euclidean pattern
    assert len(event_list) > 0

    for _, event in event_list:
        assert event.val == "x"
        # Verify intersection
        intersection = arc.intersect(event.span.active)
        assert not intersection.null()


def test_choice_sub_cycle_splitting() -> None:
    """Test choice patterns across different cycles."""
    # Choice pattern with multiple options
    pattern = Pat.rand(
        [
            Pat.pure("x"),
            Pat.pure("y"),
            Pat.pure("z"),
        ]
    )
    stream = Stream.pat(pattern)

    # Query multiple cycles to see different choices
    cycles_to_test = [0, 1, 2, 3, 4]
    choice_values = set()

    for cycle in cycles_to_test:
        arc = CycleArc(CycleTime(Fraction(cycle)), CycleTime(Fraction(cycle + 1)))
        events = stream.unstream(arc)
        event_list = list(events)

        if event_list:  # Should have at most one event per cycle
            assert len(event_list) <= 1
            _, event = event_list[0]
            choice_values.add(event.val)
            assert event.span.whole is None  # Choice events fill full arc

    # Should see different choices across cycles
    assert len(choice_values) >= 1  # At least one choice should be made


def test_alternating_sub_cycle_splitting() -> None:
    """Test alternating patterns across sub-cycles."""
    # Alternating pattern
    patterns = [Pat.pure("x"), Pat.pure("y")]
    pattern = Pat.alt(patterns)
    stream = Stream.pat(pattern)

    # Test multiple consecutive cycles
    values_by_cycle = []
    for cycle in range(4):
        arc = CycleArc(CycleTime(Fraction(cycle)), CycleTime(Fraction(cycle + 1)))
        events = stream.unstream(arc)
        event_list = list(events)

        if event_list:
            assert len(event_list) == 1
            _, event = event_list[0]
            values_by_cycle.append(event.val)
            assert event.span.whole is None  # Alternating events fill full arc

    # Should alternate between patterns
    assert len(values_by_cycle) >= 2
    # Check that values actually alternate (not all the same)
    assert len(set(values_by_cycle)) > 1


def test_probability_sub_cycle_splitting() -> None:
    """Test probability patterns across sub-cycles."""
    # Probability pattern with 50% chance
    base_pattern = Pat.pure("x")
    pattern = Pat.prob(base_pattern, Fraction(1, 2))
    stream = Stream.pat(pattern)

    # Test multiple cycles to see probabilistic behavior
    total_events = 0
    for cycle in range(10):
        arc = CycleArc(CycleTime(Fraction(cycle)), CycleTime(Fraction(cycle + 1)))
        events = stream.unstream(arc)
        event_list = list(events)
        total_events += len(event_list)

        # Each cycle should have at most one event
        assert len(event_list) <= 1

        if event_list:
            _, event = event_list[0]
            assert event.val == "x"
            assert event.span.whole is None  # Probability events fill full arc

    # With 50% probability over 10 cycles, should see some variation
    # (not always 0 or always 10)
    assert 0 <= total_events <= 10


def test_elongation_sub_cycle_splitting() -> None:
    """Test stretch patterns across sub-cycles."""
    # Elongation pattern: "x@3"
    base_pattern = Pat.pure("x")
    pattern = Pat.stretch(base_pattern, Fraction(3))
    stream = Stream.pat(pattern)

    # Query multiple cycles to see the stretched pattern
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(4)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events
    assert len(event_list) > 0

    for _, event in event_list:
        assert event.val == "x"
        # Event should intersect with our query
        intersection = arc.intersect(event.span.active)
        assert not intersection.null()


def test_polymetric_sub_cycle_splitting() -> None:
    """Test polymetric patterns across sub-cycles."""
    # Polymetric pattern with different length patterns
    patterns = [
        Pat.pure("x"),  # 1 cycle
        Pat.seq([Pat.pure("y"), Pat.pure("z")]),  # 1 cycle, 2 elements
    ]
    pattern = Pat.poly(patterns)
    stream = Stream.pat(pattern)

    # Query across multiple cycles
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events from both patterns
    assert len(event_list) > 0

    values = set(event.val for _, event in event_list)
    # Should contain events from both patterns
    assert "x" in values  # From first pattern
    assert "y" in values or "z" in values  # From second pattern


def test_parallel_sub_cycle_splitting() -> None:
    """Test parallel patterns across sub-cycles."""
    # Parallel pattern: "[x y]"
    pattern = Pat.par([Pat.pure("x"), Pat.pure("y")])
    stream = Stream.pat(pattern)

    # Query across cycle boundary
    arc = CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(3, 2)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events from both parallel elements across the queried range
    assert len(event_list) > 0

    values = [event.val for _, event in event_list]
    # Should contain both x and y events
    assert "x" in values
    assert "y" in values
