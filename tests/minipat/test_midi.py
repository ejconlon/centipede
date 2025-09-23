"""Tests for MIDI functionality."""

import time
from fractions import Fraction
from threading import Event
from typing import List

import mido
from mido.frozen import FrozenMessage

from minipat.combinators import (
    combine,
    combine_all,
    note_stream,
    program_stream,
    velocity_stream,
)
from minipat.ev import Ev, ev_heap_empty, ev_heap_singleton
from minipat.live import Instant, Orbit
from minipat.messages import (
    DEFAULT_VELOCITY,
    ChannelField,
    ControlField,
    ControlNum,
    ControlNumKey,
    ControlVal,
    ControlValKey,
    MidiAttrs,
    MidiDom,
    MsgHeap,
    MsgTypeField,
    Note,
    NoteField,
    NoteKey,
    Program,
    ProgramField,
    ProgramKey,
    TimedMessage,
    ValueField,
    Velocity,
    VelocityField,
    VelocityKey,
    midi_message_sort_key,
)
from minipat.midi import (
    MidiProcessor,
    echo_system,
    parse_messages,
)
from minipat.time import CycleArc, CycleSpan, CycleTime, PosixTime, mk_cps
from spiny.dmap import DMap


def test_note_parsing() -> None:
    """Test parsing note names."""
    # Test basic note parsing
    notes = note_stream("c4 d4 e4")

    # Should create a stream that produces MIDI attributes

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = notes.unstream(arc)
    event_list = list(events)

    assert len(event_list) == 3

    # Check that we get the right MIDI notes (C4=48, D4=50, E4=52)
    values = []
    for _, event in event_list:
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            values.append(int(note_val))

    assert 48 in values  # C4
    assert 50 in values  # D4
    assert 52 in values  # E4


def test_velocity_parsing() -> None:
    """Test parsing velocity values."""
    vels = velocity_stream("64 80 100")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = vels.unstream(arc)
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


def test_combine_two_streams() -> None:
    """Test combining two streams with combine function."""
    notes = note_stream("c4 d4")
    vels = velocity_stream("64 80")

    combined = combine(notes, vels)

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
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


def test_combine_all_streams() -> None:
    """Test combining multiple streams with combine_all function."""
    # Test with no streams
    empty = combine_all([])
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = list(empty.unstream(arc))
    assert len(events) == 0  # Should be silent

    # Test with one stream
    single = combine_all([note_stream("c4")])
    events = list(single.unstream(arc))
    assert len(events) > 0

    # Test with multiple streams
    notes = note_stream("c4 d4")
    vels = velocity_stream("64 80")
    # Test with two streams
    combined = combine_all([notes, vels])
    events = list(combined.unstream(arc))
    assert len(events) > 0


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

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1, 4)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    # Create test instant
    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(2)),  # 2 cycles per second
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

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
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

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
    )

    timed_messages = processor.process(instant, Orbit(0), ev_heap_empty())
    message_list = list(timed_messages)

    assert len(message_list) == 0


def test_midi_processor_validates_values() -> None:
    """Test MidiProcessor logs and skips invalid MIDI values."""
    processor = MidiProcessor()

    # Create MIDI attributes with out-of-range values (bypass validation for testing)
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom).put(NoteKey(), Note(200)).put(VelocityKey(), Velocity(-10))
    )

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
    )

    # Should log and skip invalid MIDI values, returning empty list
    timed_messages = processor.process(instant, Orbit(0), event_heap)
    message_list = list(timed_messages)

    # Should have no messages since the invalid event was skipped
    assert len(message_list) == 0


def test_midi_processor_orbit_as_channel() -> None:
    """Test MidiProcessor uses orbit as MIDI channel and validates range."""
    processor = MidiProcessor()

    # Create test MIDI attributes
    midi_attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
    )

    # Test valid orbits map to correct channels
    for orbit_num in [0, 1, 5, 15]:
        timed_messages = processor.process(instant, Orbit(orbit_num), event_heap)
        message_list = list(timed_messages)

        assert len(message_list) == 2  # note_on and note_off
        assert ChannelField.unmk(ChannelField.get(message_list[0].message)) == orbit_num

    # Test invalid orbit (out of range) - should log and skip
    timed_messages = processor.process(instant, Orbit(20), event_heap)
    message_list = list(timed_messages)

    # Should have no messages since the invalid orbit event was skipped
    assert len(message_list) == 0


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


def test_parse_messages_note_on() -> None:
    """Test parse_messages creates note_on message from note attributes."""
    # Test with note only (should use default velocity)
    attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

    msgs = parse_messages(Orbit(0), attrs)
    assert len(msgs) == 1
    msg = msgs[0]

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

    msgs2 = parse_messages(Orbit(1), attrs_with_vel)
    assert len(msgs2) == 1
    msg2 = msgs2[0]

    assert MsgTypeField.get(msg2) == "note_on"
    assert ChannelField.unmk(ChannelField.get(msg2)) == 1
    assert NoteField.unmk(NoteField.get(msg2)) == 72
    assert VelocityField.unmk(VelocityField.get(msg2)) == 100


def test_parse_messages_program_change() -> None:
    """Test parse_messages creates program_change message from program attributes."""
    attrs: MidiAttrs = DMap.empty(MidiDom).put(ProgramKey(), ProgramField.mk(42))

    msgs = parse_messages(Orbit(2), attrs)
    assert len(msgs) == 1
    msg = msgs[0]

    assert MsgTypeField.get(msg) == "program_change"
    assert ChannelField.unmk(ChannelField.get(msg)) == 2
    assert ProgramField.unmk(ProgramField.get(msg)) == 42


def test_parse_messages_control_change() -> None:
    """Test parse_messages creates control_change message from control attributes."""
    attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(ControlNumKey(), ControlNum(7))  # Volume control
        .put(ControlValKey(), ControlVal(80))
    )

    msgs = parse_messages(Orbit(3), attrs)
    assert len(msgs) == 1
    msg = msgs[0]

    assert MsgTypeField.get(msg) == "control_change"
    assert ChannelField.unmk(ChannelField.get(msg)) == 3
    assert ControlField.unmk(ControlField.get(msg)) == 7
    assert ValueField.unmk(ValueField.get(msg)) == 80


def test_parse_messages_channel_validation() -> None:
    """Test parse_messages validates orbit is in valid MIDI channel range (0-15)."""
    attrs: MidiAttrs = DMap.empty(MidiDom).put(NoteKey(), NoteField.mk(60))

    # Test valid orbit values
    valid_test_cases = [(Orbit(0), 0), (Orbit(15), 15), (Orbit(8), 8)]

    for orbit, expected_channel in valid_test_cases:
        msgs = parse_messages(orbit, attrs)
        assert len(msgs) == 1
        assert ChannelField.unmk(ChannelField.get(msgs[0])) == expected_channel

    # Test invalid orbit values should raise ValueError
    invalid_orbits = [Orbit(16), Orbit(100), Orbit(-1)]

    for invalid_orbit in invalid_orbits:
        try:
            parse_messages(invalid_orbit, attrs)
            assert False, f"Should have raised ValueError for orbit {invalid_orbit}"
        except ValueError as e:
            assert "out of valid MIDI channel range" in str(e)


def test_parse_messages_conflicting_attributes() -> None:
    """Test parse_messages now accepts mixed attribute combinations."""
    # Note + Program should now work - creates both messages
    attrs_note_and_program: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ProgramKey(), ProgramField.mk(42))
    )

    msgs = parse_messages(Orbit(0), attrs_note_and_program)
    assert len(msgs) == 2
    msg_types = {MsgTypeField.get(msg) for msg in msgs}
    assert "note_on" in msg_types
    assert "program_change" in msg_types

    # Note + Control should now work - creates both messages
    attrs_note_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    msgs = parse_messages(Orbit(0), attrs_note_and_control)
    assert len(msgs) == 2
    msg_types = {MsgTypeField.get(msg) for msg in msgs}
    assert "note_on" in msg_types
    assert "control_change" in msg_types

    # Program + Control should now work - creates both messages
    attrs_program_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(ProgramKey(), ProgramField.mk(42))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    msgs = parse_messages(Orbit(0), attrs_program_and_control)
    assert len(msgs) == 2
    msg_types = {MsgTypeField.get(msg) for msg in msgs}
    assert "program_change" in msg_types
    assert "control_change" in msg_types

    # All three types should now work - creates all three messages
    attrs_all_three: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(ProgramKey(), ProgramField.mk(42))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    msgs = parse_messages(Orbit(0), attrs_all_three)
    assert len(msgs) == 3
    msg_types = {MsgTypeField.get(msg) for msg in msgs}
    assert "note_on" in msg_types
    assert "program_change" in msg_types
    assert "control_change" in msg_types


def test_parse_messages_velocity_with_other_types() -> None:
    """Test parse_messages handles velocity combined with other message types."""
    # Velocity + Program should work fine now (velocity without note is allowed)
    attrs_velocity_and_program: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ProgramKey(), ProgramField.mk(42))
    )

    # Should not raise an error - velocity without note is now allowed
    messages = parse_messages(Orbit(0), attrs_velocity_and_program)

    # Should create a program change message, velocity is ignored since there's no note
    assert len(messages) == 1
    msg = messages[0]
    assert MsgTypeField.get(msg) == "program_change"
    assert ProgramField.unmk(ProgramField.get(msg)) == 42

    # Velocity + Control should also work fine
    attrs_velocity_and_control: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    # Should not raise an error
    messages = parse_messages(Orbit(0), attrs_velocity_and_control)

    # Should create a control change message, velocity is ignored since there's no note
    assert len(messages) == 1
    msg = messages[0]
    assert MsgTypeField.get(msg) == "control_change"
    assert ControlField.unmk(ControlField.get(msg)) == 7
    assert ValueField.unmk(ValueField.get(msg)) == 80


def test_parse_messages_control_incomplete() -> None:
    """Test parse_messages handles incomplete control change attributes gracefully."""
    # Only control number, no value - should not create a control message
    attrs_control_num_only: MidiAttrs = DMap.empty(MidiDom).put(
        ControlNumKey(), ControlNum(7)
    )

    # Should not raise an error - incomplete control attributes are now allowed
    messages = parse_messages(Orbit(0), attrs_control_num_only)

    # Should return empty list since control message requires both number and value
    assert len(messages) == 0

    # Only control value, no number - should not create a control message
    attrs_control_val_only: MidiAttrs = DMap.empty(MidiDom).put(
        ControlValKey(), ControlVal(80)
    )

    # Should not raise an error
    messages = parse_messages(Orbit(0), attrs_control_val_only)

    # Should return empty list since control message requires both number and value
    assert len(messages) == 0


def test_parse_messages_empty_attributes() -> None:
    """Test parse_messages returns empty list with empty attributes."""
    empty_attrs: MidiAttrs = DMap.empty(MidiDom)

    # Empty attributes should return empty list
    msgs = parse_messages(Orbit(0), empty_attrs)
    assert len(msgs) == 0


def test_parse_messages_velocity_only() -> None:
    """Test parse_messages handles velocity-only attributes gracefully."""
    velocity_only_attrs: MidiAttrs = DMap.empty(MidiDom).put(
        VelocityKey(), VelocityField.mk(100)
    )

    # Should not raise an error - velocity without note is now allowed
    messages = parse_messages(Orbit(0), velocity_only_attrs)

    # Should return empty list since there's no note to create a message for
    assert len(messages) == 0


def test_midi_processor_with_parse_messages() -> None:
    """Test MidiProcessor can handle different message types via parse_messages."""
    processor = MidiProcessor()

    # Test program change message
    program_attrs: MidiAttrs = DMap.empty(MidiDom).put(
        ProgramKey(), ProgramField.mk(42)
    )

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, program_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
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


def test_parse_messages_mixed_types() -> None:
    """Test parse_messages can extract multiple message types from same attributes."""
    # Test with note and program
    attrs_note_and_program: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ProgramKey(), ProgramField.mk(42))
    )

    msgs = parse_messages(Orbit(0), attrs_note_and_program)
    assert len(msgs) == 2

    # Check message types
    msg_types = {MsgTypeField.get(msg) for msg in msgs}
    assert "note_on" in msg_types
    assert "program_change" in msg_types

    # Verify message contents
    for msg in msgs:
        if MsgTypeField.get(msg) == "note_on":
            assert NoteField.unmk(NoteField.get(msg)) == 60
            assert VelocityField.unmk(VelocityField.get(msg)) == 100
        elif MsgTypeField.get(msg) == "program_change":
            assert ProgramField.unmk(ProgramField.get(msg)) == 42

    # Test with all three types: note, program, and control
    attrs_all_types: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(72))
        .put(ProgramKey(), ProgramField.mk(1))
        .put(ControlNumKey(), ControlNum(7))
        .put(ControlValKey(), ControlVal(80))
    )

    msgs_all = parse_messages(Orbit(1), attrs_all_types)
    assert len(msgs_all) == 3

    # Check all message types are present
    msg_types_all = {MsgTypeField.get(msg) for msg in msgs_all}
    assert "note_on" in msg_types_all
    assert "program_change" in msg_types_all
    assert "control_change" in msg_types_all


def test_midi_message_sort_key() -> None:
    """Test MIDI message sorting order."""
    # Create test messages
    note_on_msg = FrozenMessage("note_on", channel=0, note=60, velocity=64)
    note_off_msg = FrozenMessage("note_off", channel=0, note=60, velocity=0)
    program_msg = FrozenMessage("program_change", channel=0, program=42)
    control_msg = FrozenMessage("control_change", channel=0, control=7, value=100)

    # Test sort keys
    assert midi_message_sort_key(note_off_msg) < midi_message_sort_key(program_msg)
    assert midi_message_sort_key(program_msg) < midi_message_sort_key(control_msg)
    assert midi_message_sort_key(control_msg) < midi_message_sort_key(note_on_msg)

    # Test sorting a list of messages
    messages = [note_on_msg, control_msg, note_off_msg, program_msg]
    sorted_messages = sorted(messages, key=midi_message_sort_key)

    assert sorted_messages[0] == note_off_msg
    assert sorted_messages[1] == program_msg
    assert sorted_messages[2] == control_msg
    assert sorted_messages[3] == note_on_msg

    # Test that the sort key function is resilient (though FrozenMessage ensures valid types)
    assert midi_message_sort_key(note_on_msg) == 3  # Should still work normally


def test_midi_processor_with_mixed_messages() -> None:
    """Test MidiProcessor handles mixed message types."""
    processor = MidiProcessor()

    # Create MIDI attributes with multiple message types
    mixed_attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(VelocityKey(), VelocityField.mk(100))
        .put(ProgramKey(), ProgramField.mk(42))
        .put(ControlNumKey(), ProgramField.mk(43))
        .put(ControlValKey(), ProgramField.mk(44))
    )

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1, 4)))
    span = CycleSpan.mk(whole=arc, active=arc)
    event = Ev(span, mixed_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(1)),
        posix_start=PosixTime(0.0),
    )

    timed_messages = processor.process(instant, Orbit(0), event_heap)
    message_list = list(timed_messages)

    # Should have 4 messages: program_change, control_change, note_on, note_off
    assert len(message_list) == 4

    on_msg = next(
        tm for tm in message_list if MsgTypeField.get(tm.message) == "note_on"
    )
    assert NoteField.get(on_msg.message) == Note(60)
    assert VelocityField.get(on_msg.message) == Velocity(100)
    assert on_msg.time == PosixTime(0.0)

    prog_msg = next(
        tm for tm in message_list if MsgTypeField.get(tm.message) == "program_change"
    )
    assert ProgramField.get(prog_msg.message) == Program(42)
    assert prog_msg.time == PosixTime(0.0)

    con_msg = next(
        tm for tm in message_list if MsgTypeField.get(tm.message) == "control_change"
    )
    assert ControlField.get(con_msg.message) == ControlNum(43)
    assert ValueField.get(con_msg.message) == ControlVal(44)
    assert con_msg.time == PosixTime(0.0)

    off_msg = next(
        tm for tm in message_list if MsgTypeField.get(tm.message) == "note_off"
    )
    assert NoteField.get(off_msg.message) == Note(60)
    assert off_msg.time == PosixTime(0.25)


def test_mixed_streams_with_combine() -> None:
    """Test combining note and program streams."""
    notes = note_stream("c4 d4")
    programs = program_stream("0 1")

    combined = combine(notes, programs)

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = combined.unstream(arc)
    event_list = list(events)

    # Should have events with both note and program attributes
    assert len(event_list) > 0

    for _, event in event_list:
        # Check if attributes contain both note and program
        note_val = event.val.lookup(NoteKey())
        program_val = event.val.lookup(ProgramKey())

        # At least one event should have both
        if note_val is not None and program_val is not None:
            # This should now work with parse_messages
            msgs = parse_messages(Orbit(0), event.val)
            assert len(msgs) == 2  # Should have both note_on and program_change

            msg_types = {MsgTypeField.get(msg) for msg in msgs}
            assert "note_on" in msg_types
            assert "program_change" in msg_types
            break
    else:
        assert False, "No event found with both note and program"


def test_timed_message_comparison() -> None:
    """Test TimedMessage comparison methods."""
    from mido.frozen import FrozenMessage

    # Create test messages with different types and times
    note_on_msg = FrozenMessage("note_on", channel=0, note=60, velocity=64)
    note_off_msg = FrozenMessage("note_off", channel=0, note=60, velocity=0)
    program_msg = FrozenMessage("program_change", channel=0, program=42)
    control_msg = FrozenMessage("control_change", channel=0, control=43, value=44)

    # Same time, different message types
    time1 = PosixTime(1.0)
    tm_note_on = TimedMessage(time1, note_on_msg)
    tm_note_off = TimedMessage(time1, note_off_msg)
    tm_program = TimedMessage(time1, program_msg)
    tm_control = TimedMessage(time1, control_msg)

    # Different time
    time2 = PosixTime(2.0)
    tm_note_on_later = TimedMessage(time2, note_on_msg)

    # Test __lt__ (less than)
    assert tm_note_off < tm_program  # note_off sorts before program_change
    assert tm_program < tm_control  # program_change sorts before control
    assert tm_control < tm_note_on  # control_change sorts before note_on
    assert tm_note_on < tm_note_on_later  # earlier time sorts first
    assert not (tm_program < tm_note_off)  # program doesn't sort before note_off

    # Test __le__ (less than or equal)
    assert tm_note_off <= tm_program
    assert tm_program <= tm_control
    assert tm_control <= tm_note_on
    assert tm_note_on <= tm_note_on_later
    assert tm_note_off <= tm_note_off  # equal to itself
    assert not (tm_program <= tm_note_off)

    # Test __gt__ (greater than)
    assert tm_program > tm_note_off
    assert tm_control > tm_program
    assert tm_note_on > tm_control
    assert tm_note_on_later > tm_note_on
    assert not (tm_note_off > tm_program)

    # Test __ge__ (greater than or equal)
    assert tm_program >= tm_note_off
    assert tm_control >= tm_program
    assert tm_note_on >= tm_control
    assert tm_note_on_later >= tm_note_on
    assert tm_note_off >= tm_note_off  # equal to itself
    assert not (tm_note_off >= tm_program)

    # Test __eq__ (equality)
    tm_note_on_copy = TimedMessage(time1, note_on_msg)
    assert tm_note_on == tm_note_on_copy
    assert not (tm_note_on == tm_program)
    assert not (tm_note_on == tm_note_on_later)

    # Test __ne__ (not equal)
    assert tm_note_on != tm_program
    assert tm_note_on != tm_note_on_later
    assert not (tm_note_on != tm_note_on_copy)

    # Test sorting
    messages = [tm_note_on_later, tm_control, tm_note_on, tm_program, tm_note_off]
    sorted_messages = sorted(messages)

    # Should be ordered by time first, then by message type priority
    expected_order = [tm_note_off, tm_program, tm_control, tm_note_on, tm_note_on_later]
    assert sorted_messages == expected_order

    # Test heap ordering with the new MsgHeap implementation
    heap = MsgHeap.empty()
    tms = [tm_note_on_later, tm_note_on, tm_note_off, tm_program, tm_control]
    for tm in tms:
        heap.push(tm)

    # Pop messages and verify they come out in the correct order
    popped_messages = []
    while True:
        ptm = heap.pop()
        if ptm is None:
            break
        popped_messages.append(ptm)

    # Should be ordered by time first, then by message type priority
    assert popped_messages == expected_order


def test_midi_processor_whole_arc_timing() -> None:
    """Test MidiProcessor uses whole arc end time for note_off when present."""
    processor = MidiProcessor()

    # Create MIDI attributes for a note
    midi_attrs: MidiAttrs = (
        DMap.empty(MidiDom)
        .put(NoteKey(), NoteField.mk(60))
        .put(VelocityKey(), VelocityField.mk(80))
    )

    # Test case 1: Partial event at the END of a note (should generate note_off)
    # This simulates the final partial event of a long note
    active_arc = CycleArc(
        CycleTime(Fraction(3, 4)), CycleTime(Fraction(1))
    )  # 0.75 to 1.0
    whole_arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))  # 0.0 to 1.0
    span = CycleSpan(active_arc, whole_arc)
    event = Ev(span, midi_attrs)
    event_heap = ev_heap_singleton(event)

    instant = Instant(
        cycle_time=CycleTime(Fraction(0)),
        cps=mk_cps(Fraction(2)),
        posix_start=PosixTime(0.0),
    )

    timed_messages = processor.process(instant, Orbit(0), event_heap)
    message_list = list(timed_messages)

    # Should have 1 message: note_off (no note_on since this isn't the start)
    assert len(message_list) == 1

    note_off_msg = message_list[0]

    # Verify note_off timing (should use whole arc end, not active arc end)
    assert note_off_msg.time == PosixTime(0.5)  # 1.0 cycle at 2 cps = 0.5 seconds
    # Not 0.375 which would be 3/4 cycle at 2 cps (active arc end)

    # Verify it's actually a note_off message
    assert MsgTypeField.get(note_off_msg.message) == "note_off"


def test_timed_message_comparison_edge_cases() -> None:
    """Test TimedMessage comparison edge cases."""
    from mido.frozen import FrozenMessage

    # Messages with same time and same type but different content
    note_on_msg1 = FrozenMessage("note_on", channel=0, note=60, velocity=64)
    note_on_msg2 = FrozenMessage(
        "note_on", channel=1, note=72, velocity=80
    )  # Different params

    time1 = PosixTime(1.0)
    tm1 = TimedMessage(time1, note_on_msg1)
    tm2 = TimedMessage(time1, note_on_msg2)

    # Same time and message type have same sort priority
    # For sorting purposes, they should be equivalent even though content differs
    assert tm1 != tm2  # Different message content (dataclass equality)
    assert not (tm1 < tm2)  # Neither sorts before the other
    assert not (tm2 < tm1)
    # Current implementation: >= is not (self < other), so it's True
    # <= is (self == other) or (self < other), so it's False
    # This asymmetry shows the issue with the current implementation
    assert tm1 >= tm2  # This is True because not (tm1 < tm2)
    assert not (tm1 <= tm2)  # This is False because tm1 != tm2 and not (tm1 < tm2)

    # Messages with identical content should be equal
    tm1_copy = TimedMessage(time1, note_on_msg1)
    assert tm1 == tm1_copy
    assert tm1 <= tm1_copy
    assert tm1 >= tm1_copy

    # Test with very close times
    time_close = PosixTime(1.0000001)
    tm_close = TimedMessage(time_close, note_on_msg1)

    assert tm1 < tm_close  # Earlier time should sort first
    assert tm_close > tm1
