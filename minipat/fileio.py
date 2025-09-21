"""File I/O utilities for minipat patterns and events."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mido

from minipat.common import Bpc, Cps, PosixTime
from minipat.ev import EvHeap
from minipat.live import Orbit
from minipat.messages import (
    DEFAULT_VELOCITY,
    ChannelField,
    MidiAttrs,
    MsgHeap,
    MsgTypeField,
    NoteField,
    TimedMessage,
    Velocity,
    msg_note_off,
)
from minipat.midi import parse_messages


def render_midi_file(
    events: EvHeap[MidiAttrs],
    filepath: str,
    cps: Cps = Cps(Fraction(1, 2)),
    bpc: Bpc = Bpc(4),
    default_velocity: Optional[Velocity] = None,
) -> None:
    """Render an EvHeap to a MIDI file.

    Args:
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
        render_midi_file(events, "output.mid", cps=Cps(2.0), bpc=Bpc(4))
    """
    default_vel = default_velocity if default_velocity is not None else DEFAULT_VELOCITY

    # Calculate ticks_per_beat based on standard MIDI resolution
    # Standard MIDI files typically use 480 ticks per quarter note
    # We'll use 480 * bpc to get appropriate resolution for the given beats per cycle
    ticks_per_beat = 480 * int(bpc)

    # Create a new MIDI file with one track
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
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

        # Calculate timing for this event
        event_time_cycles = float(span.active.start)
        event_duration_cycles = float(span.active.end - span.active.start)

        # Convert cycle time to POSIX timestamp (treating cycle 0 as timestamp 0)
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

    # Extract sorted messages from heap and track active notes for smart note-off handling
    # Key: (channel, note) -> count of active notes
    active_notes: dict[tuple[int, int], int] = {}
    last_time = 0.0

    # Process all messages in time order
    while True:
        # Pop the next message
        timed_msg = msg_heap.pop()
        if timed_msg is None:
            break
        abs_time = timed_msg.time
        frozen_msg = timed_msg.message

        msg_type = MsgTypeField.get(frozen_msg)

        # Handle note tracking for smart note-off
        if msg_type == "note_on":
            channel = ChannelField.get(frozen_msg)
            note = NoteField.get(frozen_msg)
            key = (channel, note)

            # Increment active note count
            active_notes[key] = active_notes.get(key, 0) + 1

            # Always send note_on
            note_on_msg = mido.Message(
                "note_on",
                channel=channel,
                note=note,
                velocity=frozen_msg.velocity,
                time=int((abs_time - last_time) * float(cps) * ticks_per_beat),
            )
            track.append(note_on_msg)
            last_time = abs_time

        elif msg_type == "note_off":
            channel = ChannelField.get(frozen_msg)
            note = NoteField.get(frozen_msg)
            key = (channel, note)

            # Only send note_off if this is the last active instance
            count = active_notes.get(key, 0)
            if count > 0:
                active_notes[key] = count - 1
                if active_notes[key] == 0:
                    # This is the last note_off needed
                    del active_notes[key]
                    note_off_msg = mido.Message(
                        "note_off",
                        channel=channel,
                        note=note,
                        velocity=0,
                        time=int((abs_time - last_time) * float(cps) * ticks_per_beat),
                    )
                    track.append(note_off_msg)
                    last_time = abs_time
                # else: skip this note_off as there are still active notes

        elif msg_type == "program_change":
            # Program change
            pc_msg = mido.Message(
                "program_change",
                channel=ChannelField.get(frozen_msg),
                program=frozen_msg.program,
                time=int((abs_time - last_time) * float(cps) * ticks_per_beat),
            )
            track.append(pc_msg)
            last_time = abs_time

        elif msg_type == "control_change":
            # Control change
            cc_msg = mido.Message(
                "control_change",
                channel=ChannelField.get(frozen_msg),
                control=frozen_msg.control,
                value=frozen_msg.value,
                time=int((abs_time - last_time) * float(cps) * ticks_per_beat),
            )
            track.append(cc_msg)
            last_time = abs_time

    # Save the MIDI file
    midi_file.save(filepath)
