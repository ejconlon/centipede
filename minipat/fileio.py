"""File I/O utilities for minipat patterns and events."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mido

from minipat.ev import EvHeap
from minipat.live import Orbit
from minipat.messages import (
    DEFAULT_VELOCITY,
    ChannelField,
    ControlField,
    MidiAttrs,
    MsgHeap,
    MsgTypeField,
    NoteField,
    ProgramField,
    TimedMessage,
    ValueField,
    Velocity,
    VelocityField,
    mido_bundle_iterator,
    msg_note_off,
)
from minipat.midi import parse_messages
from minipat.time import Bpc, Cps, CycleTime, PosixTime


def render_midi_file(
    start: CycleTime,
    events: EvHeap[MidiAttrs],
    filepath: str,
    cps: Cps = Cps(Fraction(1, 2)),
    bpc: Bpc = Bpc(4),
    default_velocity: Optional[Velocity] = None,
) -> None:
    """Render an EvHeap to a MIDI file.

    Args:
        start: Start cycle time - events will be timed relative to this
        events: The event heap containing MIDI attribute events
        filepath: Path where to save the MIDI file
        cps: Cycles per second (tempo)
        bpc: Beats per cycle (used to calculate ticks_per_beat)
        default_velocity: Default velocity when not specified in events

    Examples:
        # Create some events and render to MIDI file
        events = compose_once([
            (CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1))), note_stream("c4")),
            (CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2))), note_stream("d4")),
        ])
        render_midi_file(CycleTime(Fraction(0)), events, "output.mid", cps=Cps(2.0), bpc=Bpc(4))
    """
    default_vel = default_velocity if default_velocity is not None else DEFAULT_VELOCITY

    # Calculate ticks_per_beat (ticks per quarter note)
    # Standard MIDI files typically use 480 ticks per quarter note
    ticks_per_beat = 480

    # Create a new MIDI file with one track (type=0 for single track file)
    midi_file = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Set tempo (microseconds per quarter note)
    # With cps cycles per second, and bpc beats per cycle:
    # beats per second = cps * bpc
    # seconds per beat = 1 / (cps * bpc)
    # microseconds per beat = (1 / (cps * bpc)) * 1_000_000
    tempo = int((1.0 / (float(cps) * int(bpc))) * 1_000_000)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Build a heap of TimedMessages for proper sorting
    msg_heap = MsgHeap.empty()

    for span, ev in events:
        # Parse the MIDI attributes into messages
        # Use orbit 0 as default channel if no channel is specified
        msgs = parse_messages(Orbit(0), ev.val, default_vel)

        # Calculate timing for this event relative to start
        event_time_cycles = float(span.active.start - start)
        event_duration_cycles = float(span.active.end - span.active.start)

        # Convert cycle time to POSIX timestamp (treating start as timestamp 0)
        event_time = PosixTime(event_time_cycles / float(cps))
        event_end_time = PosixTime(
            (event_time_cycles + event_duration_cycles) / float(cps)
        )

        # Process each MIDI message
        for midi_msg in msgs:
            msg_type = MsgTypeField.get(midi_msg)

            if msg_type == "note_on":
                # Add note_on at start time
                msg_heap.push(TimedMessage(event_time, midi_msg))

                # Create and add note_off at end time
                channel = ChannelField.get(midi_msg)
                note = NoteField.get(midi_msg)
                note_off_msg = msg_note_off(channel=channel, note=note)
                msg_heap.push(TimedMessage(event_end_time, note_off_msg))
            else:
                # For program_change and control_change, just add at start time
                msg_heap.push(TimedMessage(event_time, midi_msg))

    # Extract sorted messages from heap and track whether notes expect note-off
    # Key: (channel, note) -> boolean indicating if note_off is expected
    expects_note_off: dict[tuple[int, int], bool] = {}
    last_time = 0.0

    # Process all messages in time order
    while True:
        # Pop the next message
        timed_msg = msg_heap.pop()
        if timed_msg is None:
            break
        abs_time = timed_msg.time
        time = int((abs_time - last_time) * float(cps) * int(bpc) * ticks_per_beat)
        for frozen_msg in mido_bundle_iterator(timed_msg.bundle):
            msg_type = MsgTypeField.get(frozen_msg)

            # Handle note tracking
            if msg_type == "note_on":
                channel = ChannelField.get(frozen_msg)
                note = NoteField.get(frozen_msg)
                velocity = VelocityField.get(frozen_msg)
                key = (channel, note)

                if velocity == 0:
                    # note_on with velocity=0 is equivalent to note_off
                    if expects_note_off.get(key, False):
                        expects_note_off[key] = False
                        note_off_msg = mido.Message(
                            "note_off",
                            channel=channel,
                            note=note,
                            velocity=0,
                            time=time,
                        )
                        track.append(note_off_msg)
                        last_time = abs_time
                    # else: skip this note_off as we don't expect one
                else:
                    # Regular note_on with velocity > 0
                    expects_note_off[key] = True
                    note_on_msg = mido.Message(
                        "note_on",
                        channel=channel,
                        note=note,
                        velocity=velocity,
                        time=time,
                    )
                    track.append(note_on_msg)
                    last_time = abs_time

            elif msg_type == "note_off":
                channel = ChannelField.get(frozen_msg)
                note = NoteField.get(frozen_msg)
                key = (channel, note)

                # Only send note_off if we expect one
                if expects_note_off.get(key, False):
                    expects_note_off[key] = False
                    note_off_msg = mido.Message(
                        "note_off",
                        channel=channel,
                        note=note,
                        velocity=0,
                        time=time,
                    )
                    track.append(note_off_msg)
                    last_time = abs_time
                # else: skip this note_off as we don't expect one

            elif msg_type == "program_change":
                # Program change
                pc_msg = mido.Message(
                    "program_change",
                    channel=ChannelField.get(frozen_msg),
                    program=ProgramField.get(frozen_msg),
                    time=time,
                )
                track.append(pc_msg)
                last_time = abs_time

            elif msg_type == "control_change":
                # Control change
                cc_msg = mido.Message(
                    "control_change",
                    channel=ChannelField.get(frozen_msg),
                    control=ControlField.get(frozen_msg),
                    value=ValueField.get(frozen_msg),
                    time=time,
                )
                track.append(cc_msg)
                last_time = abs_time

    # Save the MIDI file
    midi_file.save(filepath)
