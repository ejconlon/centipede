"""Tests for composition functionality."""

from __future__ import annotations

from fractions import Fraction

from minipat.pat import Pat
from minipat.stream import ComposeStream, Stream, compose_once
from minipat.time import CycleArc, CycleTime


def test_compose_empty_sections() -> None:
    """Test compose with empty sections list."""
    from minipat.ev import EvHeap

    result: EvHeap[str] = compose_once([])
    event_list = list(result)
    assert len(event_list) == 0


def test_compose_single_section() -> None:
    """Test compose with a single section."""
    stream = Stream.pat(Pat.pure("note"))
    section = (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream)
    sections = [section]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 1
    _, event = event_list[0]
    assert event.val == "note"
    assert event.span.active.start == CycleTime(Fraction(0))
    assert event.span.active.end == CycleTime(Fraction(1))
    assert event.span.whole == event.span.active  # No clipping occurred


def test_compose_multiple_sections() -> None:
    """Test compose with multiple sequential sections."""
    stream1 = Stream.pat(Pat.pure("a"))
    stream2 = Stream.pat(Pat.pure("b"))
    stream3 = Stream.pat(Pat.pure("c"))

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream1),
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2))), stream2),
        (CycleArc(CycleTime(Fraction(2)), CycleTime(Fraction(3))), stream3),
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 3

    # Sort by start time
    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    assert sorted_events[0][1].val == "a"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1))

    assert sorted_events[1][1].val == "b"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(2))

    assert sorted_events[2][1].val == "c"
    assert sorted_events[2][1].span.active.start == CycleTime(Fraction(2))
    assert sorted_events[2][1].span.active.end == CycleTime(Fraction(3))


def test_compose_overlapping_sections() -> None:
    """Test compose with overlapping sections."""
    stream1 = Stream.pat(Pat.pure("x"))
    stream2 = Stream.pat(Pat.pure("y"))

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2))), stream1),  # 0-2
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(3))), stream2),  # 1-3
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 2

    # Sort by start time
    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    # First section: x maintains its natural 1-cycle duration, placed at section start
    assert sorted_events[0][1].val == "x"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1))

    # Second section: y maintains its natural 1-cycle duration, placed at section start
    assert sorted_events[1][1].val == "y"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(2))


def test_compose_with_sequence_patterns() -> None:
    """Test compose with more complex patterns."""
    stream1 = Stream.pat(Pat.seq([Pat.pure("a"), Pat.pure("b")]))
    stream2 = Stream.pat(Pat.pure("x"))

    sections = [
        (
            CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2))),
            stream1,
        ),  # seq splits over 2 cycles
        (
            CycleArc(CycleTime(Fraction(2)), CycleTime(Fraction(3))),
            stream2,
        ),  # single note
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 3

    # Sort by start time
    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    # Sequence pattern: "a" in first half, "b" in second half
    assert sorted_events[0][1].val == "a"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1, 2))

    assert sorted_events[1][1].val == "b"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1, 2))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(1))

    # Pure pattern: "x"
    assert sorted_events[2][1].val == "x"
    assert sorted_events[2][1].span.active.start == CycleTime(Fraction(2))
    assert sorted_events[2][1].span.active.end == CycleTime(Fraction(3))


def test_compose_stream_class_directly() -> None:
    """Test ComposeStream class directly."""
    stream1 = Stream.pat(Pat.pure("test1"))
    stream2 = Stream.pat(Pat.pure("test2"))

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream1),
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2))), stream2),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Test the containing arc property
    assert compose_stream.containing_arc.start == CycleTime(Fraction(0))
    assert compose_stream.containing_arc.end == CycleTime(Fraction(2))

    # Test querying the full stream
    events = compose_stream.unstream(containing_arc)
    event_list = list(events)

    assert len(event_list) == 2

    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)
    assert sorted_events[0][1].val == "test1"
    assert sorted_events[1][1].val == "test2"


def test_compose_stream_partial_queries() -> None:
    """Test ComposeStream with partial queries to verify whole arc behavior."""
    stream = Stream.pat(Pat.pure("note"))
    sections = [
        (
            CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2))),
            stream,
        ),  # Note spans 0-2
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Query only part of the note (should get whole arc)
    partial_query = CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(3, 2)))
    events = compose_stream.unstream(partial_query)
    event_list = list(events)

    assert len(event_list) == 1
    _, event = event_list[0]

    # Active arc should be clipped to the intersection of query and event
    assert event.span.active.start == CycleTime(Fraction(1, 2))
    assert event.span.active.end == CycleTime(
        Fraction(1)
    )  # Event ends at 1, query goes to 1.5

    # Whole arc should show the full extent of the original event
    assert event.span.whole is not None
    assert event.span.whole.start == CycleTime(Fraction(0))
    assert event.span.whole.end == CycleTime(
        Fraction(1)
    )  # Pure pattern naturally spans 0-1


def test_compose_stream_out_of_bounds_query() -> None:
    """Test ComposeStream with infinite looping behavior."""
    stream = Stream.pat(Pat.pure("note"))
    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Query before the composition - should get the pattern shifted back
    before_query = CycleArc(CycleTime(Fraction(-1)), CycleTime(Fraction(0)))
    events = compose_stream.unstream(before_query)
    event_list = list(events)
    assert len(event_list) == 1
    assert event_list[0][1].span.active.start == CycleTime(Fraction(-1))
    assert event_list[0][1].span.active.end == CycleTime(Fraction(0))

    # Query after the composition - should get the pattern shifted forward
    after_query = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    events = compose_stream.unstream(after_query)
    event_list = list(events)
    assert len(event_list) == 1
    assert event_list[0][1].span.active.start == CycleTime(Fraction(1))
    assert event_list[0][1].span.active.end == CycleTime(Fraction(2))

    # Query overlapping two repetitions
    overlap_begin = CycleArc(CycleTime(Fraction(-1, 2)), CycleTime(Fraction(1, 2)))
    events = compose_stream.unstream(overlap_begin)
    event_list = list(events)
    assert (
        len(event_list) == 2
    )  # Two events: tail of previous repetition and head of current
    # First event is from the previous repetition (-1 to 0)
    assert event_list[0][1].span.active.start == CycleTime(Fraction(-1, 2))
    assert event_list[0][1].span.active.end == CycleTime(Fraction(0))
    # Second event is from the current repetition (0 to 1)
    assert event_list[1][1].span.active.start == CycleTime(Fraction(0))
    assert event_list[1][1].span.active.end == CycleTime(Fraction(1, 2))


def test_compose_stream_null_arc_query() -> None:
    """Test ComposeStream with null arc query."""
    stream = Stream.pat(Pat.pure("note"))
    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Query with null arc
    null_query = CycleArc(
        CycleTime(Fraction(1)), CycleTime(Fraction(1))
    )  # Start == end
    events = compose_stream.unstream(null_query)
    assert len(list(events)) == 0


def test_compose_sections_sorting() -> None:
    """Test that sections are sorted by start time."""
    stream1 = Stream.pat(Pat.pure("first"))
    stream2 = Stream.pat(Pat.pure("second"))
    stream3 = Stream.pat(Pat.pure("third"))

    # Provide sections in non-chronological order
    sections = [
        (CycleArc(CycleTime(Fraction(2)), CycleTime(Fraction(3))), stream3),  # third
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream1),  # first
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2))), stream2),  # second
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 3

    # Events should be in chronological order regardless of input order
    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    assert sorted_events[0][1].val == "first"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))

    assert sorted_events[1][1].val == "second"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1))

    assert sorted_events[2][1].val == "third"
    assert sorted_events[2][1].span.active.start == CycleTime(Fraction(2))


def test_compose_fractional_timing() -> None:
    """Test compose with fractional timing."""
    stream1 = Stream.pat(Pat.pure("quarter"))
    stream2 = Stream.pat(Pat.pure("half"))

    sections = [
        (
            CycleArc(CycleTime(Fraction(1, 4)), CycleTime(Fraction(1, 2))),
            stream1,
        ),  # 0.25-0.5
        (
            CycleArc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(1))),
            stream2,
        ),  # 0.5-1.0
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 2

    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    assert sorted_events[0][1].val == "quarter"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(1, 4))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1, 2))

    assert sorted_events[1][1].val == "half"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1, 2))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(1))


def test_compose_with_gaps() -> None:
    """Test compose with gaps between sections."""
    stream1 = Stream.pat(Pat.pure("start"))
    stream2 = Stream.pat(Pat.pure("end"))

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream1),  # 0-1
        (
            CycleArc(CycleTime(Fraction(3)), CycleTime(Fraction(4))),
            stream2,
        ),  # 3-4 (gap from 1-3)
    ]

    result = compose_once(sections)
    event_list = list(result)

    assert len(event_list) == 2

    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)

    assert sorted_events[0][1].val == "start"
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1))

    assert sorted_events[1][1].val == "end"
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(3))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(4))


def test_compose_with_silence() -> None:
    """Test compose with silent patterns."""
    stream1 = Stream.pat(Pat.pure("sound"))
    stream2: Stream[str] = Stream.pat(Pat.silent())

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), stream1),
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2))), stream2),
    ]

    result = compose_once(sections)
    event_list = list(result)

    # Should only get events from the non-silent section
    assert len(event_list) == 1
    assert event_list[0][1].val == "sound"


def test_compose_stream_invalid_section_arcs() -> None:
    """Test ComposeStream with invalid section arcs."""
    stream = Stream.pat(Pat.pure("note"))

    # Test with negative start time
    sections = [
        (CycleArc(CycleTime(Fraction(-1)), CycleTime(Fraction(1))), stream),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(-1)), CycleTime(Fraction(1)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Should still work, just with unusual timing
    events = compose_stream.unstream(containing_arc)
    event_list = list(events)
    assert len(event_list) == 1
    assert event_list[0][1].span.active.start == CycleTime(Fraction(-1))


def test_compose_stream_zero_length_sections() -> None:
    """Test ComposeStream with zero-length sections."""
    stream = Stream.pat(Pat.pure("instant"))

    # Section with zero duration
    sections = [
        (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(1))), stream),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    compose_stream = ComposeStream(containing_arc, sections)

    # Zero-length sections should produce no events
    events = compose_stream.unstream(containing_arc)
    event_list = list(events)
    assert len(event_list) == 0


def test_compose_stream_very_small_sections() -> None:
    """Test ComposeStream with very small section durations."""
    stream = Stream.pat(Pat.pure("tiny"))

    # Very small section duration
    tiny_duration = Fraction(1, 1000)
    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(tiny_duration)), stream),
    ]

    containing_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(tiny_duration))
    compose_stream = ComposeStream(containing_arc, sections)

    events = compose_stream.unstream(containing_arc)
    event_list = list(events)

    # Should get an event in the tiny section
    assert len(event_list) == 1
    assert event_list[0][1].val == "tiny"
    assert event_list[0][1].span.active.start == CycleTime(Fraction(0))
    assert event_list[0][1].span.active.end == CycleTime(tiny_duration)


def test_compose_with_complex_nested_patterns() -> None:
    """Test compose with complex nested patterns."""
    # Create a nested sequence pattern
    nested_seq = Pat.seq([Pat.pure("x"), Pat.pure("y"), Pat.pure("z")])
    stream = Stream.pat(nested_seq)

    sections = [
        (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(3))), stream),
    ]

    result = compose_once(sections)
    event_list = list(result)

    # Should get events from the sequence pattern
    assert len(event_list) == 3

    # Verify we have the expected values in order
    sorted_events = sorted(event_list, key=lambda x: x[1].span.active.start)
    assert sorted_events[0][1].val == "x"
    assert sorted_events[1][1].val == "y"
    assert sorted_events[2][1].val == "z"

    # Check timing - should be evenly spaced in the first cycle (0-1)
    assert sorted_events[0][1].span.active.start == CycleTime(Fraction(0))
    assert sorted_events[0][1].span.active.end == CycleTime(Fraction(1, 3))
    assert sorted_events[1][1].span.active.start == CycleTime(Fraction(1, 3))
    assert sorted_events[1][1].span.active.end == CycleTime(Fraction(2, 3))
    assert sorted_events[2][1].span.active.start == CycleTime(Fraction(2, 3))
    assert sorted_events[2][1].span.active.end == CycleTime(Fraction(1))
