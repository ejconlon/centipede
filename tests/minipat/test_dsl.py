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
    """Test that transpose removes events that only contain notes when out of range."""
    # Create notes near the high boundary
    high_note = note("g10")  # MIDI note 127

    # Transpose up (should be removed entirely since only has NoteKey)
    transposed_high = high_note.transpose("5")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    # Test high out-of-range removal
    events_high = transposed_high.stream.unstream(arc)
    event_list_high = list(events_high)

    # Should have no events since the event only contained NoteKey
    assert len(event_list_high) == 0


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


def test_flow_transpose_removes_note_only_events() -> None:
    """Test that transpose removes events that only contain NoteKey when out of range."""
    # Create a note-only event (no velocity or other attributes)
    bare_note = note("g10")  # MIDI note 127 (highest valid note)

    # Transpose up to go out of range (127 + 1 = 128, which is invalid)
    transposed = bare_note.transpose("1")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = transposed.stream.unstream(arc)
    event_list = list(events)

    # Should have no events since the only attribute was NoteKey
    assert len(event_list) == 0


def test_flow_transpose_preserves_event_with_other_attrs() -> None:
    """Test that transpose preserves events with other attributes when note is out of range."""
    from minipat.dsl import vel

    # Create a note with velocity
    note_with_vel = note("g10") >> vel("80")  # MIDI note 127 with velocity

    # Transpose up to go out of range (127 + 1 = 128, which is invalid)
    transposed = note_with_vel.transpose("1")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = transposed.stream.unstream(arc)
    event_list = list(events)

    # Should have one event (preserved because it has velocity)
    assert len(event_list) == 1
    _, event = event_list[0]

    # Note should be removed
    note_val = event.val.lookup(NoteKey())
    assert note_val is None

    # But velocity should be preserved
    vel_val = event.val.lookup(VelocityKey())
    assert vel_val is not None
    assert int(vel_val) == 80


def test_note_with_sharps() -> None:
    """Test parsing notes with sharps."""
    # Create a melody with sharp notes
    melody = note("c#4 d#4 f#4 g#4 a#4")  # C#4=49, D#4=51, F#4=54, G#4=56, A#4=58

    # Query events from the flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = melody.stream.unstream(arc)
    event_list = list(events)

    # Should have 5 events
    assert len(event_list) == 5

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # Should be the correct MIDI note numbers for sharp notes
    assert notes == [49, 51, 54, 56, 58]


def test_note_with_flats() -> None:
    """Test parsing notes with flats."""
    # Create a melody with flat notes
    melody = note("db4 eb4 gb4 ab4 bb4")  # Db4=49, Eb4=51, Gb4=54, Ab4=56, Bb4=58

    # Query events from the flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = melody.stream.unstream(arc)
    event_list = list(events)

    # Should have 5 events
    assert len(event_list) == 5

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # Should be the correct MIDI note numbers for flat notes (same as sharps)
    assert notes == [49, 51, 54, 56, 58]


def test_note_mixed_sharps_and_naturals() -> None:
    """Test parsing notes with mixed sharps and naturals."""
    # Create a melody mixing natural and sharp notes
    melody = note("c4 c#4 d4 d#4 e4")  # C4=48, C#4=49, D4=50, D#4=51, E4=52

    # Query events from the flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = melody.stream.unstream(arc)
    event_list = list(events)

    # Should have 5 events
    assert len(event_list) == 5

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # Should be the correct chromatic progression
    assert notes == [48, 49, 50, 51, 52]


def test_note_case_insensitive() -> None:
    """Test that note names are case-insensitive."""
    # Test uppercase, lowercase, and mixed case notes
    melody = note("C4 c4 C#4 c#4 Db4 db4")  # All should be parsed correctly

    # Query events from the flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = melody.stream.unstream(arc)
    event_list = list(events)

    # Should have 6 events
    assert len(event_list) == 6

    # Extract note values
    notes = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        assert note_val is not None
        notes.append(int(note_val))

    # C4=48, c4=48, C#4=49, c#4=49, Db4=49, db4=49
    assert notes == [48, 48, 49, 49, 49, 49]
