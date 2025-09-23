"""Tests for DSL functionality including Flow methods."""

from fractions import Fraction

from minipat.dsl import note
from minipat.messages import NoteKey, VelocityKey
from minipat.time import CycleArc, CycleTime


def test_flow_transpose_basic() -> None:
    """Test basic transpose functionality with constant offset."""
    # Create a simple melody
    melody = note("c4 d4 e4")  # C4=48, D4=50, E4=52

    # Transpose up by 12 semitones (one octave)
    transposed = melody.transpose("12")

    # Query events from the transposed flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = transposed.stream.unstream(arc)
    event_list = list(events)

    # Should have 3 events
    assert len(event_list) == 3

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # Should be transposed up by 12: [60, 62, 64]
    assert notes == [60, 62, 64]


def test_flow_transpose_pattern() -> None:
    """Test transpose with varying offset pattern."""
    # Create a simple melody
    melody = note("c4 d4 e4")  # C4=48, D4=50, E4=52

    # Transpose with varying offsets: 0, 5, 7 (perfect fourth, perfect fifth)
    transposed = melody.transpose("0 5 7")

    # Query events from the transposed flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = transposed.stream.unstream(arc)
    event_list = list(events)

    # Should have 3 events
    assert len(event_list) == 3

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # Should be: C4+0=48, D4+5=55, E4+7=59
    assert notes == [48, 55, 59]


def test_flow_transpose_out_of_range() -> None:
    """Test that transpose silences notes that go out of valid MIDI range."""
    # Create notes near the high boundary
    high_note = note("g10")  # MIDI note 127

    # Transpose up (should be silenced)
    transposed_high = high_note.transpose("5")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    # Test high out-of-range silencing
    events_high = transposed_high.stream.unstream(arc)
    event_list_high = list(events_high)

    # Should have an event, but without a note (silenced)
    assert len(event_list_high) == 1
    _, event_high = event_list_high[0]
    note_val_high = event_high.val.lookup(NoteKey())
    assert note_val_high is None  # Note removed (silenced)


def test_flow_transpose_preserves_other_attributes() -> None:
    """Test that transpose preserves non-note MIDI attributes."""
    from minipat.dsl import vel

    # Create a melody with velocity
    melody = note("c4 d4") >> vel("80 100")

    # Transpose the melody
    transposed = melody.transpose("5")

    # Query events from the transposed flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = transposed.stream.unstream(arc)
    event_list = list(events)

    # Should have 2 events
    assert len(event_list) == 2

    # Check that both note and velocity are present and correct

    # First event: C4+5=53, velocity=80
    _, event1 = event_list[0]
    note1 = event1.val.lookup(NoteKey())
    vel1 = event1.val.lookup(VelocityKey())
    assert note1 is not None and int(note1) == 53
    assert vel1 is not None and int(vel1) == 80

    # Second event: D4+5=55, velocity=100
    _, event2 = event_list[1]
    note2 = event2.val.lookup(NoteKey())
    vel2 = event2.val.lookup(VelocityKey())
    assert note2 is not None and int(note2) == 55
    assert vel2 is not None and int(vel2) == 100
