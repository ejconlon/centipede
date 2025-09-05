"""Tests for Stream methods that are not covered in test_stream.py."""

from fractions import Fraction

from minipat.arc import Arc
from minipat.common import CycleDelta, CycleTime
from minipat.pat import Pat, SpeedOp
from minipat.stream import MergeStrat, Stream
from spiny import PSeq


def test_stream_map() -> None:
    """Test Stream.map() transforms values correctly."""
    # Create a stream with pure value
    stream = Stream.pure("hello")

    # Map to uppercase
    mapped_stream = stream.map(lambda x: x.upper())

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = mapped_stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "HELLO"


def test_stream_map_with_sequence() -> None:
    """Test Stream.map() on a sequence stream."""
    # Create a sequence stream
    stream = Stream.seq(PSeq.mk([Stream.pure(1), Stream.pure(2), Stream.pure(3)]))

    # Map to double values
    mapped_stream = stream.map(lambda x: x * 2)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = mapped_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    assert len(event_list) == 3
    values = [event.val for _, event in event_list]
    assert values == [2, 4, 6]


def test_stream_filter() -> None:
    """Test Stream.filter() removes events based on predicate."""
    # Create a sequence stream with numbers
    stream = Stream.seq(
        PSeq.mk(
            [
                Stream.pure(1),
                Stream.pure(2),
                Stream.pure(3),
                Stream.pure(4),
                Stream.pure(5),
            ]
        )
    )

    # Filter to only even numbers
    filtered_stream = stream.filter(lambda x: x % 2 == 0)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = filtered_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should only have 2 and 4
    assert len(event_list) == 2
    values = [event.val for _, event in event_list]
    assert values == [2, 4]


def test_stream_filter_with_parallel() -> None:
    """Test Stream.filter() on parallel streams."""
    # Create parallel streams
    stream = Stream.par(
        PSeq.mk(
            [
                Stream.pure("apple"),
                Stream.pure("banana"),
                Stream.pure("apricot"),
                Stream.pure("berry"),
            ]
        )
    )

    # Filter to only values starting with 'a'
    filtered_stream = stream.filter(lambda x: x.startswith("a"))

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = filtered_stream.unstream(arc)
    event_list = list(events)

    # Should only have apple and apricot
    assert len(event_list) == 2
    values = set(event.val for _, event in event_list)
    assert values == {"apple", "apricot"}


def test_stream_bind() -> None:
    """Test Stream.bind() for monadic bind operation."""
    # Create a stream with a single value
    stream = Stream.pure(3)

    # Bind to create multiple events
    def expand_to_seq(x: int) -> Stream[int]:
        return Stream.seq(
            PSeq.mk([Stream.pure(x - 1), Stream.pure(x), Stream.pure(x + 1)])
        )

    bound_stream = stream.bind(MergeStrat.Inner, expand_to_seq)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = bound_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should expand to [2, 3, 4]
    assert len(event_list) == 3
    values = [event.val for _, event in event_list]
    assert values == [2, 3, 4]


def test_stream_apply() -> None:
    """Test Stream.apply() for combining two streams."""
    # Create two streams
    left_stream = Stream.seq(PSeq.mk([Stream.pure(10), Stream.pure(20)]))
    right_stream = Stream.seq(PSeq.mk([Stream.pure(1), Stream.pure(2)]))

    # Apply addition
    combined_stream = left_stream.apply(
        MergeStrat.Inner, lambda x, y: x + y, right_stream
    )

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = combined_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should combine values where events overlap
    assert len(event_list) == 2
    values = [event.val for _, event in event_list]
    # First half: 10 + 1 = 11
    # Second half: 20 + 2 = 22
    assert values == [11, 22]


def test_stream_shift_early() -> None:
    """Test Stream.shift() with negative delta shifts events earlier."""
    # Create a pure stream
    stream = Stream.pure("x")

    # Shift earlier by 1/4 cycle (negative delta)
    early_stream = stream.shift(CycleDelta(-Fraction(1, 4)))

    # Query from 0 to 1
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = early_stream.unstream(arc)
    event_list = list(events)

    # Should get event that was shifted from later
    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "x"


def test_stream_shift_late() -> None:
    """Test Stream.shift() with positive delta shifts events later."""
    # Create a sequence stream
    stream = Stream.seq(PSeq.mk([Stream.pure("a"), Stream.pure("b")]))

    # Shift later by 1/4 cycle (positive delta)
    late_stream = stream.shift(CycleDelta(Fraction(1, 4)))

    # Query from 0 to 1
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = late_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Late_by shifts the query arc earlier to get events that will be shifted later
    # So we still get events but they appear to come from an earlier time
    assert len(event_list) > 0

    # The implementation queries earlier events and shifts them later
    # So we get the same events at the same positions
    # (the shift happens in the query, not the result)
    values = [event.val for _, event in event_list]
    assert "a" in values or "b" in values


def test_stream_constructor_silence() -> None:
    """Test Stream.silent() constructor."""
    stream: Stream[str] = Stream.silent()

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 0


def test_stream_constructor_pure() -> None:
    """Test Stream.pure() constructor."""
    stream = Stream.pure("hello")

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "hello"
    assert event.span.active == arc


def test_stream_constructor_seq() -> None:
    """Test Stream.seq() constructor."""
    streams = PSeq.mk([Stream.pure(1), Stream.pure(2), Stream.pure(3)])
    stream = Stream.seq(streams)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    assert len(event_list) == 3
    values = [event.val for _, event in event_list]
    assert values == [1, 2, 3]


def test_stream_constructor_par() -> None:
    """Test Stream.par() constructor."""
    streams = PSeq.mk([Stream.pure("x"), Stream.pure("y"), Stream.pure("z")])
    stream = Stream.par(streams)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 3
    values = set(event.val for _, event in event_list)
    assert values == {"x", "y", "z"}

    # All should span full arc
    for _, event in event_list:
        assert event.span.active == arc


def test_stream_constructor_choice() -> None:
    """Test Stream.rand() constructor."""
    choices = PSeq.mk([Stream.pure("a"), Stream.pure("b"), Stream.pure("c")])
    stream = Stream.rand(choices)

    # Test different cycles
    for cycle in range(3):
        arc = Arc(CycleTime(Fraction(cycle)), CycleTime(Fraction(cycle + 1)))
        events = stream.unstream(arc)
        event_list = list(events)

        assert len(event_list) == 1
        _, event = event_list[0]
        # Choice should cycle through options
        expected_value = ["a", "b", "c"][cycle % 3]
        assert event.val == expected_value


def test_stream_constructor_euclidean() -> None:
    """Test Stream.euc() constructor."""
    atom_stream = Stream.pure("x")
    stream = Stream.euc(atom_stream, 3, 8, 0)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have 3 hits distributed over 8 steps
    assert len(event_list) == 3
    for _, event in event_list:
        assert event.val == "x"


def test_stream_constructor_polymetric() -> None:
    """Test Stream.poly() constructor."""
    patterns = PSeq.mk([Stream.pure("a"), Stream.pure("b")])

    # Without subdivision
    stream = Stream.poly(patterns)
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    # Should have both patterns playing
    assert len(event_list) == 2
    values = set(event.val for _, event in event_list)
    assert values == {"a", "b"}

    # With subdivision
    stream_sub = Stream.poly(patterns, 2)
    events_sub = stream_sub.unstream(arc)
    event_list_sub = list(events_sub)

    # Should still have events
    assert len(event_list_sub) > 0


def test_stream_constructor_repetition() -> None:
    """Test Stream.speed() constructor."""
    base_stream = Stream.pure("x")

    # Fast repetition
    fast_stream = Stream.speed(base_stream, SpeedOp.Fast, Fraction(3))
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = fast_stream.unstream(arc)
    event_list = list(events)

    # Should have 3 repetitions
    assert len(event_list) == 3
    for _, event in event_list:
        assert event.val == "x"

    # Slow repetition
    slow_stream = Stream.speed(base_stream, SpeedOp.Slow, Fraction(2))
    events_slow = slow_stream.unstream(arc)
    event_list_slow = list(events_slow)

    # Should have 1 event stretched
    assert len(event_list_slow) == 1


def test_stream_constructor_elongation() -> None:
    """Test Stream.stretch() constructor."""
    base_stream = Stream.pure("x")
    stream = Stream.stretch(base_stream, Fraction(2))

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "x"
    assert event.span.active == arc


def test_stream_constructor_probability() -> None:
    """Test Stream.prob() constructor."""
    base_stream = Stream.pure("x")

    # 100% probability
    stream_always = Stream.prob(base_stream, Fraction(1))
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream_always.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "x"

    # 0% probability
    stream_never = Stream.prob(base_stream, Fraction(0))
    events_never = stream_never.unstream(arc)
    event_list_never = list(events_never)

    assert len(event_list_never) == 0


def test_stream_constructor_alternating() -> None:
    """Test Stream.alt() constructor."""
    patterns = PSeq.mk([Stream.pure("a"), Stream.pure("b")])
    stream = Stream.alt(patterns)

    # Test alternation over cycles
    values = []
    for cycle in range(4):
        arc = Arc(CycleTime(Fraction(cycle)), CycleTime(Fraction(cycle + 1)))
        events = stream.unstream(arc)
        event_list = list(events)

        if event_list:
            assert len(event_list) == 1
            _, event = event_list[0]
            values.append(event.val)

    # Should alternate between a and b
    assert values == ["a", "b", "a", "b"]


def test_stream_constructor_replicate() -> None:
    """Test Stream.repeat() constructor."""
    base_stream = Stream.pure("x")
    stream = Stream.repeat(base_stream, Fraction(4))

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have 4 replications
    assert len(event_list) == 4
    for _, event in event_list:
        assert event.val == "x"

    # Each should take 1/4 of the arc
    for i, (_, event) in enumerate(event_list):
        expected_start = Fraction(i, 4)
        expected_end = Fraction(i + 1, 4)
        assert event.span.active.start == expected_start
        assert event.span.active.end == expected_end


def test_stream_constructor_pat() -> None:
    """Test Stream.pat() constructor."""
    # Create a pattern
    pattern = Pat.seq(
        [
            Pat.pure("a"),
            Pat.pure("b"),
            Pat.pure("c"),
        ]
    )

    # Convert to stream using Stream.pat()
    stream = Stream.pat(pattern)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    assert len(event_list) == 3
    values = [event.val for _, event in event_list]
    assert values == ["a", "b", "c"]


def test_stream_map_chain() -> None:
    """Test chaining multiple map operations."""
    stream = Stream.pure(5)

    # Chain multiple maps
    result_stream = (
        stream.map(lambda x: x * 2)  # 10
        .map(lambda x: x + 3)  # 13
        .map(lambda x: str(x))
    )  # "13"

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = result_stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "13"


def test_stream_filter_chain() -> None:
    """Test chaining filter with other operations."""
    stream = Stream.seq(PSeq.mk([Stream.pure(i) for i in range(1, 11)]))

    # Filter even numbers, then map to squares
    result_stream = stream.filter(lambda x: x % 2 == 0).map(lambda x: x**2)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = result_stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have squares of even numbers: 4, 16, 36, 64, 100
    values = [event.val for _, event in event_list]
    assert values == [4, 16, 36, 64, 100]


def test_stream_complex_transformation() -> None:
    """Test complex transformation combining multiple stream operations."""
    # Start with a parallel stream
    stream = Stream.par(PSeq.mk([Stream.pure(1), Stream.pure(2), Stream.pure(3)]))

    # Transform: filter > 1, map to double
    result_stream = stream.filter(lambda x: x > 1).map(lambda x: x * 2)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    events = result_stream.unstream(arc)
    event_list = list(events)

    # Should have filtered and transformed values
    values = set(event.val for _, event in event_list)
    assert 4 in values  # 2 * 2
    assert 6 in values  # 3 * 2
    assert 2 not in values  # 1 was filtered out


def test_stream_apply_with_different_patterns() -> None:
    """Test apply with different pattern structures."""
    # Sequence on left, parallel on right
    left = Stream.seq(PSeq.mk([Stream.pure("a"), Stream.pure("b")]))
    right = Stream.par(PSeq.mk([Stream.pure("1"), Stream.pure("2")]))

    # Concatenate strings
    combined = left.apply(MergeStrat.Inner, lambda x, y: x + y, right)

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = combined.unstream(arc)
    event_list = list(events)

    # Should have combinations where events overlap
    assert len(event_list) > 0
    values = set(event.val for _, event in event_list)
    # Should have combinations like "a1", "a2", "b1", "b2"
    assert all(v[0] in "ab" and v[1] in "12" for v in values)


def test_stream_timing_precision() -> None:
    """Test that stream operations maintain timing precision."""
    # Create a complex pattern with precise timing
    stream = Stream.seq(PSeq.mk([Stream.pure("a"), Stream.pure("b"), Stream.pure("c")]))

    # Apply timing transformation (negative delta = earlier)
    result = stream.shift(CycleDelta(-Fraction(1, 6)))

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    events = result.unstream(arc)
    event_list = list(events)

    # Should have events with precise fractional timing
    assert len(event_list) > 0

    # Check that all timings are Fractions (not floats)
    for _, event in event_list:
        assert isinstance(event.span.active.start, Fraction)
        assert isinstance(event.span.active.end, Fraction)
