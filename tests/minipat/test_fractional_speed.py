"""Tests for fractional speed factors in patterns."""

from fractions import Fraction

from minipat.parser import parse_pattern
from minipat.pat import Pat, SpeedOp
from minipat.printer import print_pattern
from minipat.stream import Stream
from minipat.time import CycleArc, CycleTime


def test_pat_speed_fractional_fast() -> None:
    """Test Pat.speed with fractional fast factor like 3/2."""
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(3, 2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = sorted(events, key=lambda x: x[0].active.start)

    # Should have 2 events (1 full + 0.5 partial) for x*1.5
    assert len(event_list) == 2

    # First event should be full duration (0 to 2/3)
    _, first_event = event_list[0]
    assert first_event.val == "x"
    assert first_event.span.active.start == Fraction(0)
    assert first_event.span.active.end == Fraction(2, 3)

    # Second event should be partial duration (2/3 to 1)
    _, second_event = event_list[1]
    assert second_event.val == "x"
    assert second_event.span.active.start == Fraction(2, 3)
    assert second_event.span.active.end == Fraction(1)


def test_pat_speed_fractional_slow() -> None:
    """Test Pat.speed with fractional slow factor like 1/2."""
    base_pattern = Pat.pure("x")
    pattern = Pat.speed(base_pattern, SpeedOp.Slow, Fraction(1, 2))
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    # Pattern should be half speed (slower)
    assert len(event_list) >= 1
    for _, event in event_list:
        assert event.val == "x"


def test_parser_fractional_speed() -> None:
    """Test parsing fractional speed factors."""
    from minipat.pat import PatSpeed

    # Test parsing "x*3%2" - fast by 3/2
    pattern = parse_pattern("x*3%2")
    assert isinstance(pattern.unwrap, PatSpeed)
    speed_pat = pattern.unwrap
    assert isinstance(speed_pat.pat.unwrap, type(Pat.pure("x").unwrap))
    assert speed_pat.op == SpeedOp.Fast
    assert speed_pat.factor == Fraction(3, 2)

    # Test parsing "x/1%2" - slow by 1/2
    pattern = parse_pattern("x/1%2")
    assert isinstance(pattern.unwrap, PatSpeed)
    speed_pat = pattern.unwrap
    assert isinstance(speed_pat.pat.unwrap, type(Pat.pure("x").unwrap))
    assert speed_pat.op == SpeedOp.Slow
    assert speed_pat.factor == Fraction(1, 2)


def test_parser_integer_speed() -> None:
    """Test parsing integer speed factors."""
    # Test parsing "x*2" - fast by 2
    pattern = parse_pattern("x*2")
    from minipat.pat import PatSpeed

    assert isinstance(pattern.unwrap, PatSpeed)
    speed_pat = pattern.unwrap
    assert speed_pat.pat.unwrap.value == "x"
    assert speed_pat.op == SpeedOp.Fast
    assert speed_pat.factor == Fraction(2)

    # Test parsing "x/2" - slow by 2
    pattern = parse_pattern("x/2")
    assert isinstance(pattern.unwrap, PatSpeed)
    speed_pat = pattern.unwrap
    assert speed_pat.pat.unwrap.value == "x"
    assert speed_pat.op == SpeedOp.Slow
    assert speed_pat.factor == Fraction(2)


def test_printer_fractional_speed() -> None:
    """Test printing fractional speed factors."""
    # Test printing fractional fast speed
    pattern = Pat.speed(Pat.pure("x"), SpeedOp.Fast, Fraction(3, 2))
    result = print_pattern(pattern)
    assert "x*3%2" in result

    # Test printing fractional slow speed
    pattern = Pat.speed(Pat.pure("x"), SpeedOp.Slow, Fraction(1, 2))
    result = print_pattern(pattern)
    assert "x/1%2" in result

    # Test printing integer speed (should not use % notation)
    pattern = Pat.speed(Pat.pure("x"), SpeedOp.Fast, Fraction(2))
    result = print_pattern(pattern)
    assert "x*2" in result


def test_complex_fractional_speed_pattern() -> None:
    """Test complex pattern with fractional speeds."""
    # Parse a pattern with sequence and fractional speed
    pattern = parse_pattern("[x y]*3%2")
    stream = Stream.pat(pattern)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    events = stream.unstream(arc)
    event_list = list(events)

    # Should have events from both x and y, sped up by 3/2
    assert len(event_list) >= 2
    values = [event.val for _, event in event_list]
    assert "x" in values
    assert "y" in values


def test_fractional_repetition_semantics() -> None:
    """Test that fractional repetitions work correctly."""
    base_pattern = Pat.pure("x")

    # Test x*1 = 1 event
    pattern1 = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(1))
    stream1 = Stream.pat(pattern1)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events1 = list(stream1.unstream(arc))
    assert len(events1) == 1

    # Test x*2 = 2 events
    pattern2 = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(2))
    stream2 = Stream.pat(pattern2)
    events2 = list(stream2.unstream(arc))
    assert len(events2) == 2

    # Test x*2.5 = 3 events (2 full + 0.5 partial)
    pattern25 = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(5, 2))
    stream25 = Stream.pat(pattern25)
    events25 = list(stream25.unstream(arc))
    assert len(events25) == 3  # 2 full + 1 partial


def test_fractional_repeat_semantics() -> None:
    """Test that fractional repeat works correctly."""
    base_pattern = Pat.pure("x")

    # Test x!1 = 1 repetition
    pattern1 = Pat.repeat(base_pattern, Fraction(1))
    stream1 = Stream.pat(pattern1)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events1 = list(stream1.unstream(arc))
    assert len(events1) == 1

    # Test x!2 = 2 repetitions
    pattern2 = Pat.repeat(base_pattern, Fraction(2))
    stream2 = Stream.pat(pattern2)
    events2 = list(stream2.unstream(arc))
    assert len(events2) == 2

    # Test x!1.5 = 1.5 repetitions (1 full + 0.5 partial)
    pattern15 = Pat.repeat(base_pattern, Fraction(3, 2))
    stream15 = Stream.pat(pattern15)
    events15 = list(stream15.unstream(arc))
    assert len(events15) == 2  # 1 full + 1 partial

    # Verify timing for x!1.5
    sorted_events = sorted(events15, key=lambda x: x[0].active.start)
    _, first_event = sorted_events[0]
    _, second_event = sorted_events[1]

    # First repetition should be full duration (0 to 2/3)
    assert first_event.span.active.start == Fraction(0)
    assert first_event.span.active.end == Fraction(2, 3)

    # Second repetition should be partial duration (2/3 to 1)
    assert second_event.span.active.start == Fraction(2, 3)
    assert second_event.span.active.end == Fraction(1)


def test_parser_fractional_repeat() -> None:
    """Test parsing fractional repeat factors."""
    from minipat.pat import PatRepeat

    # Test parsing "x!3%2" - repeat by 3/2
    pattern = parse_pattern("x!3%2")
    assert isinstance(pattern.unwrap, PatRepeat)
    repeat_pat = pattern.unwrap
    assert repeat_pat.pat.unwrap.value == "x"
    assert repeat_pat.count == Fraction(3, 2)

    # Test parsing "x!2" - repeat by 2
    pattern = parse_pattern("x!2")
    assert isinstance(pattern.unwrap, PatRepeat)
    repeat_pat = pattern.unwrap
    assert repeat_pat.pat.unwrap.value == "x"
    assert repeat_pat.count == Fraction(2)


def test_extreme_fractional_values() -> None:
    """Test extreme fractional values for edge cases."""
    base_pattern = Pat.pure("x")

    # Very small fraction
    pattern_small = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(1, 100))
    stream_small = Stream.pat(pattern_small)
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events_small = stream_small.unstream(arc)
    event_list_small = list(events_small)
    assert len(event_list_small) >= 0  # Should not crash

    # Very large fraction
    pattern_large = Pat.speed(base_pattern, SpeedOp.Fast, Fraction(100, 1))
    stream_large = Stream.pat(pattern_large)
    events_large = stream_large.unstream(arc)
    event_list_large = list(events_large)
    assert len(event_list_large) >= 0  # Should not crash
