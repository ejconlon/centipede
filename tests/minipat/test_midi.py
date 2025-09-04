"""Tests for MIDI functionality."""

from fractions import Fraction

from minipat.common import CycleTime, PosixTime
from minipat.live import Instant, Orbit
from minipat.midi import (
    MidiAttrs,
    MidiDom,
    MidiProcessor,
    Note,
    NoteKey,
    Vel,
    VelKey,
    combine,
    note,
    vel,
)
from spiny.dmap import DMap


def test_note_parsing():
    """Test parsing note names."""
    # Test basic note parsing
    note_stream = note("c4 d4 e4")

    # Should create a stream that produces MIDI attributes
    from minipat.arc import Arc
    from minipat.common import CycleTime

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = note_stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 3

    # Check that we get the right MIDI notes (C4=60, D4=62, E4=64)
    values = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            values.append(int(note_val))

    assert 60 in values  # C4
    assert 62 in values  # D4
    assert 64 in values  # E4


def test_velocity_parsing():
    """Test parsing velocity values."""
    vel_stream = vel("64 80 100")

    from minipat.arc import Arc
    from minipat.common import CycleTime

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = vel_stream.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 3

    # Check velocity values
    values = []
    for _, event in event_list:
        vel_val = event.val.lookup(VelKey())
        if vel_val is not None:
            values.append(int(vel_val))

    assert 64 in values
    assert 80 in values
    assert 100 in values


def test_combine_streams():
    """Test combining note and velocity streams."""
    note_stream = note("c4 d4")
    vel_stream = vel("64 80")

    combined = combine(note_stream, vel_stream)

    from minipat.arc import Arc
    from minipat.common import CycleTime

    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = combined.unstream(arc)
    event_list = list(events)

    # Should have events with both note and velocity
    assert len(event_list) > 0

    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        vel_val = event.val.lookup(VelKey())

        # At least one should have both attributes
        if note_val is not None and vel_val is not None:
            assert 0 <= int(note_val) <= 127
            assert 0 <= int(vel_val) <= 127
            break
    else:
        assert False, "No event found with both note and velocity"


def test_midi_processor():
    """Test MidiProcessor converts MidiAttrs to MIDI messages."""
    processor = MidiProcessor(default_velocity=64)

    # Create test MIDI attributes
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom).put(NoteKey(), Note(60)).put(VelKey(), Vel(80))
    )

    # Create test event heap
    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(
        active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1, 4))), whole=None
    )
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    # Create test instant
    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=Fraction(2),  # 2 cycles per second
        posix_start=PosixTime(0.0),
    )

    # Process events
    timed_messages = processor.process(instant, Orbit(0), event_heap)
    message_list = list(timed_messages)

    # Should have note on and note off messages
    assert len(message_list) == 2

    note_on_msg = message_list[0]
    note_off_msg = message_list[1]

    # Check note on message
    assert getattr(note_on_msg.message, "type") == "note_on"
    assert getattr(note_on_msg.message, "note") == 60
    assert getattr(note_on_msg.message, "velocity") == 80
    assert getattr(note_on_msg.message, "channel") == 0  # Orbit 0 -> Channel 0

    # Check note off message
    assert getattr(note_off_msg.message, "type") == "note_off"
    assert getattr(note_off_msg.message, "note") == 60
    assert getattr(note_off_msg.message, "velocity") == 0
    assert getattr(note_off_msg.message, "channel") == 0

    # Check timing
    assert note_on_msg.time == PosixTime(0.0)  # Start of arc
    assert note_off_msg.time == PosixTime(
        0.125
    )  # End of arc (1/4 cycle at 2 cps = 0.125 seconds)


def test_midi_processor_defaults():
    """Test MidiProcessor uses defaults for missing attributes."""
    processor = MidiProcessor(default_velocity=100)

    # Create MIDI attributes with only note (no velocity)
    midi_attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), Note(72))

    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), whole=None)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    timed_messages = processor.process(
        instant, Orbit(1), event_heap
    )  # Use Orbit(1) for channel 1
    message_list = list(timed_messages)

    note_on_msg = message_list[0]

    # Should use default velocity and orbit as channel
    assert getattr(note_on_msg.message, "velocity") == 100
    assert getattr(note_on_msg.message, "channel") == 1  # Orbit 1 -> Channel 1
    assert getattr(note_on_msg.message, "note") == 72


def test_midi_processor_empty_events():
    """Test MidiProcessor handles empty event heap."""
    processor = MidiProcessor()

    from minipat.ev import ev_heap_empty

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    timed_messages = processor.process(instant, Orbit(0), ev_heap_empty())
    message_list = list(timed_messages)

    assert len(message_list) == 0


def test_midi_processor_clamps_values():
    """Test MidiProcessor clamps MIDI values to valid range."""
    processor = MidiProcessor()

    # Create MIDI attributes with out-of-range values
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom).put(NoteKey(), Note(200)).put(VelKey(), Vel(-10))
    )

    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), whole=None)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    timed_messages = processor.process(instant, Orbit(0), event_heap)
    message_list = list(timed_messages)

    note_on_msg = message_list[0]

    # Should clamp to valid MIDI range
    assert getattr(note_on_msg.message, "note") == 127  # Clamped from 200
    assert getattr(note_on_msg.message, "velocity") == 0  # Clamped from -10


def test_midi_processor_orbit_as_channel():
    """Test MidiProcessor uses orbit as MIDI channel."""
    processor = MidiProcessor()

    # Create test MIDI attributes
    midi_attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), Note(60))

    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), whole=None)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    # Test different orbits map to different channels
    for orbit_num in [0, 1, 5, 15, 20]:  # Last one should clamp to 15
        timed_messages = processor.process(instant, Orbit(orbit_num), event_heap)
        message_list = list(timed_messages)

        expected_channel = min(15, orbit_num)  # Should clamp to 0-15 range
        assert getattr(message_list[0].message, "channel") == expected_channel
