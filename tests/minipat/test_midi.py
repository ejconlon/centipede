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
    DEFAULT_VELOCITY,
    ChannelField,
    ControlField,
    ControlNum,
    ControlNumKey,
    ControlVal,
    ControlValKey,
    MidiAttrs,
    MidiDom,
    MidiProcessor,
    MsgTypeField,
    Note,
    NoteField,
    NoteKey,
    ProgramField,
    ProgramKey,
    ValueField,
    Velocity,
    VelocityField,
    VelocityKey,
    combine,
    echo_system,
    note,
    parse_message,
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
        vel_val = event.val.lookup(VelocityKey())
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
        vel_val = event.val.lookup(VelocityKey())

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
        .put(VelocityKey(), VelocityField.mk(80))
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


def test_midi_processor_validates_values() -> None:
    """Test MidiProcessor validates MIDI values and raises ValueError for invalid events."""
    processor = MidiProcessor()

    # Create MIDI attributes with out-of-range values (bypass validation for testing)
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom).put(NoteKey(), Note(200)).put(VelocityKey(), Velocity(-10))
    )

    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), whole=None)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    # Should raise ValueError for invalid MIDI values
    try:
        processor.process(instant, Orbit(0), event_heap)
        assert False, "Should have raised ValueError for out-of-range values"
    except ValueError as e:
        assert "in range 0" in str(e)  # mido's error message format


def test_midi_processor_orbit_as_channel() -> None:
    """Test MidiProcessor uses orbit as MIDI channel and validates range."""
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

    # Test valid orbits map to correct channels
    for orbit_num in [0, 1, 5, 15]:
        timed_messages = processor.process(instant, Orbit(orbit_num), event_heap)
        message_list = list(timed_messages)

        assert len(message_list) == 2  # note_on and note_off
        assert ChannelField.unmk(ChannelField.get(message_list[0].message)) == orbit_num

    # Test invalid orbit (out of range) - should raise ValueError
    try:
        processor.process(instant, Orbit(20), event_heap)
        assert False, "Should have raised ValueError for out-of-range orbit"
    except ValueError as e:
        assert "out of valid MIDI channel range" in str(e)


def test_echo_system_integration() -> None:
    """Integration test for echo_system that sends MIDI and verifies echo."""

    received_messages: List[FrozenMessage] = []
    message_received = Event()

    def message_callback(msg: FrozenMessage) -> None:
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
            assert MsgTypeField.get(received_msg) == MsgTypeField.get(test_msg), (
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


def test_parse_message_note_on() -> None:
    """Test parse_message creates note_on message from note attributes."""
    # Test with note only (should use default velocity)
    attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

    msg = parse_message(Orbit(0), attrs)

    assert MsgTypeField.get(msg) == "note_on"
    assert ChannelField.unmk(ChannelField.get(msg)) == 0
    assert NoteField.unmk(NoteField.get(msg)) == 60
    assert VelocityField.unmk(VelocityField.get(msg)) == int(
        DEFAULT_VELOCITY
    )  # Default velocity

    # Test with note and velocity
    attrs_with_vel: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(72))
        .put(VelocityKey(), VelocityField.mk(100))
    )

    msg2 = parse_message(Orbit(1), attrs_with_vel)

    assert MsgTypeField.get(msg2) == "note_on"
    assert ChannelField.unmk(ChannelField.get(msg2)) == 1
    assert NoteField.unmk(NoteField.get(msg2)) == 72
    assert VelocityField.unmk(VelocityField.get(msg2)) == 100


def test_parse_message_program_change() -> None:
    """Test parse_message creates program_change message from program attributes."""
    attrs: MidiAttrs = DMap.empty(MidiDom).put(ProgramKey(), ProgramField.mk(42))

    msg = parse_message(Orbit(2), attrs)

    assert MsgTypeField.get(msg) == "program_change"
    assert ChannelField.unmk(ChannelField.get(msg)) == 2
    assert ProgramField.unmk(ProgramField.get(msg)) == 42


def test_parse_message_control_change() -> None:
    """Test parse_message creates control_change message from control attributes."""
    attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(ControlNumKey(), ControlNum(7))  # Volume control
        .put(ControlValKey(), ControlVal(80))
    )

    msg = parse_message(Orbit(3), attrs)

    assert MsgTypeField.get(msg) == "control_change"
    assert ChannelField.unmk(ChannelField.get(msg)) == 3
    assert ControlField.unmk(ControlField.get(msg)) == 7
    assert ValueField.unmk(ValueField.get(msg)) == 80


def test_parse_message_channel_validation() -> None:
    """Test parse_message validates orbit is in valid MIDI channel range (0-15)."""
    attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

    # Test valid orbit values
    valid_test_cases = [(Orbit(0), 0), (Orbit(15), 15), (Orbit(8), 8)]

    for orbit, expected_channel in valid_test_cases:
        msg = parse_message(orbit, attrs)
        assert ChannelField.unmk(ChannelField.get(msg)) == expected_channel

    # Test invalid orbit values should raise ValueError
    invalid_orbits = [Orbit(16), Orbit(100), Orbit(-1)]

    for invalid_orbit in invalid_orbits:
        try:
            parse_message(invalid_orbit, attrs)
            assert False, f"Should have raised ValueError for orbit {invalid_orbit}"
        except ValueError as e:
            assert "out of valid MIDI channel range" in str(e)


def test_parse_message_conflicting_attributes() -> None:
    """Test parse_message rejects conflicting attribute combinations."""
    # Note + Program should raise ValueError
    attrs_note_and_program: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ProgramKey(), ProgramField.mk(42))
    )

    try:
        parse_message(Orbit(0), attrs_note_and_program)
        assert False, "Should have raised ValueError for note + program"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "note/velocity, program" in str(e)

    # Note + Control should raise ValueError
    attrs_note_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    try:
        parse_message(Orbit(0), attrs_note_and_control)
        assert False, "Should have raised ValueError for note + control"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "note/velocity, control" in str(e)

    # Program + Control should raise ValueError
    attrs_program_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(ProgramKey(), ProgramField.mk(42))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    try:
        parse_message(Orbit(0), attrs_program_and_control)
        assert False, "Should have raised ValueError for program + control"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "program, control" in str(e)

    # All three types should raise ValueError
    attrs_all_three: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ProgramKey(), ProgramField.mk(42))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    try:
        parse_message(Orbit(0), attrs_all_three)
        assert False, "Should have raised ValueError for all three types"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "note/velocity, program, control" in str(e)


def test_parse_message_velocity_only_conflicting() -> None:
    """Test parse_message rejects velocity-only attributes when combined with other types."""
    # Velocity + Program should raise ValueError (velocity implies note message type)
    attrs_velocity_and_program: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ProgramKey(), ProgramField.mk(42))
    )

    try:
        parse_message(Orbit(0), attrs_velocity_and_program)
        assert False, "Should have raised ValueError for velocity + program"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "note/velocity, program" in str(e)

    # Velocity + Control should raise ValueError
    attrs_velocity_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    try:
        parse_message(Orbit(0), attrs_velocity_and_control)
        assert False, "Should have raised ValueError for velocity + control"
    except ValueError as e:
        assert "Conflicting MIDI attributes found" in str(e)
        assert "note/velocity, control" in str(e)


def test_parse_message_control_incomplete() -> None:
    """Test parse_message handles incomplete control change attributes."""
    # Only control number, no value - should raise ValueError
    attrs_control_num_only: MidiAttrs = DMap.empty(MidiDom).put(
        ControlNumKey(), ControlNum(7)
    )

    try:
        parse_message(Orbit(0), attrs_control_num_only)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Incomplete control change attributes" in str(e)
        assert "missing control_val" in str(e)

    # Only control value, no number - should raise ValueError
    attrs_control_val_only: MidiAttrs = DMap.empty(MidiDom).put(
        ControlValKey(), ControlVal(80)
    )

    try:
        parse_message(Orbit(0), attrs_control_val_only)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Incomplete control change attributes" in str(e)
        assert "missing control_num" in str(e)


def test_parse_message_empty_attributes() -> None:
    """Test parse_message raises ValueError with empty attributes."""
    empty_attrs: MidiAttrs = DMap.empty(MidiDom)

    try:
        parse_message(Orbit(0), empty_attrs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Insufficient MIDI attributes" in str(e)
        assert (
            "Expected one of: (note), (program), or (control_num + control_val)"
            in str(e)
        )


def test_parse_message_velocity_only() -> None:
    """Test parse_message raises ValueError with only velocity (no note)."""
    velocity_only_attrs: MidiAttrs = DMap.empty(MidiDom).put(
        VelocityKey(), VelocityField.mk(100)
    )

    try:
        parse_message(Orbit(0), velocity_only_attrs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Velocity attribute found without note" in str(e)
        assert "Velocity can only be used with note attributes" in str(e)


def test_midi_processor_with_parse_message() -> None:
    """Test MidiProcessor can handle different message types via parse_message."""
    processor = MidiProcessor()

    # Test program change message
    program_attrs: MidiAttrs = DMap.empty(MidiDom).put(
        ProgramKey(), ProgramField.mk(42)
    )

    from minipat.arc import Arc, Span
    from minipat.ev import Ev, ev_heap_singleton

    span = Span(active=Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), whole=None)
    event = Ev(span, program_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)), cps=Fraction(1), posix_start=PosixTime(0.0)
    )

    timed_messages = processor.process(instant, Orbit(1), event_heap)
    message_list = list(timed_messages)

    # Should have one program change message
    assert len(message_list) == 1
    msg = message_list[0].message

    assert MsgTypeField.get(msg) == "program_change"
    assert ChannelField.unmk(ChannelField.get(msg)) == 1
    assert ProgramField.unmk(ProgramField.get(msg)) == 42

    # Test control change message
    control_attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(ControlNumKey(), ControlNum(7))  # Volume
        .put(ControlValKey(), ControlVal(100))
    )

    control_event = Ev(span, control_attrs)
    control_heap = ev_heap_singleton(control_event)

    control_messages = processor.process(instant, Orbit(2), control_heap)
    control_list = list(control_messages)

    # Should have one control change message
    assert len(control_list) == 1
    control_msg = control_list[0].message

    assert MsgTypeField.get(control_msg) == "control_change"
    assert ChannelField.unmk(ChannelField.get(control_msg)) == 2
    assert ControlField.unmk(ControlField.get(control_msg)) == 7
    assert ValueField.unmk(ValueField.get(control_msg)) == 100
