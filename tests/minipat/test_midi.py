"""Tests for MIDI functionality."""

import time
from fractions import Fraction
from threading import Event
from typing import List

import mido
from mido.frozen import FrozenMessage

from minipat.common import CycleTime, PosixTime
from minipat.live import Instant, Orbit
from minipat.midi import (
    ChannelField,
    ControlField,
    MidiAttrs,
    MidiDom,
    MidiProcessor,
    MsgTypeField,
    Note,
    NoteField,
    NoteKey,
    ValueField,
    VelKey,
    Velocity,
    VelocityField,
    combine,
    echo_system,
    note,
    vel,
)
from spiny.dmap import DMap


def test_note_parsing() -> None:
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


def test_velocity_parsing() -> None:
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


def test_combine_streams() -> None:
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


def test_midi_processor() -> None:
    """Test MidiProcessor converts MidiAttrs to MIDI messages."""
    processor = MidiProcessor(default_velocity=VelocityField.mk(64))

    # Create test MIDI attributes
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(VelKey(), VelocityField.mk(80))
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
    assert (
        MsgTypeField.exists(note_on_msg.message)
        and MsgTypeField.get(note_on_msg.message) == "note_on"
    )
    assert NoteField.unmk(NoteField.get(note_on_msg.message)) == 60
    assert VelocityField.unmk(VelocityField.get(note_on_msg.message)) == 80
    assert (
        ChannelField.unmk(ChannelField.get(note_on_msg.message)) == 0
    )  # Orbit 0 -> Channel 0

    # Check note off message
    assert (
        MsgTypeField.exists(note_off_msg.message)
        and MsgTypeField.get(note_off_msg.message) == "note_off"
    )
    assert NoteField.unmk(NoteField.get(note_off_msg.message)) == 60
    assert VelocityField.unmk(VelocityField.get(note_off_msg.message)) == 0
    assert ChannelField.unmk(ChannelField.get(note_off_msg.message)) == 0

    # Check timing
    assert note_on_msg.time == PosixTime(0.0)  # Start of arc
    assert note_off_msg.time == PosixTime(
        0.125
    )  # End of arc (1/4 cycle at 2 cps = 0.125 seconds)


def test_midi_processor_defaults() -> None:
    """Test MidiProcessor uses defaults for missing attributes."""
    processor = MidiProcessor(default_velocity=VelocityField.mk(100))

    # Create MIDI attributes with only note (no velocity)
    midi_attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(72))

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
    assert VelocityField.unmk(VelocityField.get(note_on_msg.message)) == 100
    assert (
        ChannelField.unmk(ChannelField.get(note_on_msg.message)) == 1
    )  # Orbit 1 -> Channel 1
    assert NoteField.unmk(NoteField.get(note_on_msg.message)) == 72


def test_midi_processor_empty_events() -> None:
    """Test MidiProcessor handles empty event heap."""
    processor = MidiProcessor()

    from minipat.ev import ev_heap_empty

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    timed_messages = processor.process(instant, Orbit(0), ev_heap_empty())
    message_list = list(timed_messages)

    assert len(message_list) == 0


def test_midi_processor_clamps_values() -> None:
    """Test MidiProcessor clamps MIDI values to valid range."""
    processor = MidiProcessor()

    # Create MIDI attributes with out-of-range values (bypass validation for testing)
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom).put(NoteKey(), Note(200)).put(VelKey(), Velocity(-10))
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
    assert NoteField.unmk(NoteField.get(note_on_msg.message)) == 127  # Clamped from 200
    assert (
        VelocityField.unmk(VelocityField.get(note_on_msg.message)) == 0
    )  # Clamped from -10


def test_midi_processor_orbit_as_channel() -> None:
    """Test MidiProcessor uses orbit as MIDI channel."""
    processor = MidiProcessor()

    # Create test MIDI attributes
    midi_attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

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
        assert (
            ChannelField.unmk(ChannelField.get(message_list[0].message))
            == expected_channel
        )


def test_echo_system_integration() -> None:
    """Integration test for echo_system that sends MIDI and verifies echo."""

    received_messages: List[FrozenMessage] = []
    message_received = Event()

    def message_callback(msg):
        """Callback to capture received messages."""
        received_messages.append(msg)
        message_received.set()

    system = None
    input_port = None
    output_port = None
    try:
        # Create the echo system first
        system = echo_system(in_port_name="virt_in", out_port_name="virt_out")

        # Give system time to create virtual ports
        time.sleep(0.1)

        # Open connections to the virtual ports
        input_port = mido.open_output("virt_in")  # pyright: ignore
        output_port = mido.open_input("virt_out", callback=message_callback)  # pyright: ignore

        # Give ports time to connect
        time.sleep(0.1)

        # Create test MIDI messages
        test_messages = [
            FrozenMessage("note_on", channel=0, note=60, velocity=64),
            FrozenMessage("note_off", channel=0, note=60, velocity=0),
            FrozenMessage("note_on", channel=1, note=72, velocity=100),
            FrozenMessage("control_change", channel=0, control=1, value=50),
        ]

        # Send each test message and verify it's echoed
        for i, test_msg in enumerate(test_messages):
            # Clear previous state
            received_messages.clear()
            message_received.clear()

            # Send the message
            input_port.send(test_msg)

            # Wait for the echoed message (with timeout)
            message_arrived = message_received.wait(timeout=1.0)
            assert message_arrived, f"Message {i} was not echoed within timeout"

            # Verify the message was echoed correctly
            assert len(received_messages) >= 1, (
                f"No messages received for test message {i}"
            )
            received_msg = received_messages[0]

            # Compare message content (excluding timing-related fields)
            assert received_msg.type == test_msg.type, (
                f"Message type mismatch for message {i}"
            )

            if ChannelField.exists(test_msg):
                assert ChannelField.get(received_msg) == ChannelField.get(test_msg), (
                    f"Channel mismatch for message {i}"
                )
            if NoteField.exists(test_msg):
                assert NoteField.get(received_msg) == NoteField.get(test_msg), (
                    f"Note mismatch for message {i}"
                )
            if VelocityField.exists(test_msg):
                assert VelocityField.get(received_msg) == VelocityField.get(test_msg), (
                    f"Velocity mismatch for message {i}"
                )
            if ControlField.exists(test_msg):
                assert ControlField.get(received_msg) == ControlField.get(test_msg), (
                    f"Control mismatch for message {i}"
                )
            if ValueField.exists(test_msg):
                assert ValueField.get(received_msg) == ValueField.get(test_msg), (
                    f"Value mismatch for message {i}"
                )

    finally:
        # Clean up: close ports and stop system
        if system is not None:
            # Stop the actor system
            system.stop()

            # Wait for system to shut down cleanly
            exceptions = system.wait(timeout=2.0)

            # Verify no fatal exceptions occurred
            fatal_exceptions = [exc for exc in exceptions if exc.fatal]
            assert len(fatal_exceptions) == 0, (
                f"Fatal exceptions occurred: {fatal_exceptions}"
            )

        # Close ports
        if input_port is not None:
            input_port.close()
        if output_port is not None:
            output_port.close()
