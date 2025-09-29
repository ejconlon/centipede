"""File I/O utilities for minipat patterns and events."""

from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import mido

from minipat.dsl import FlowLike, convert_to_midi_stream
from minipat.ev import EvHeap, ev_heap_empty
from minipat.live import Orbit
from minipat.messages import (
    DEFAULT_VELOCITY,
    ChannelField,
    ControlField,
    MidiAttrs,
    MsgHeap,
    MsgTypeField,
    NoteField,
    NoteKey,
    ProgramField,
    TabInstKey,
    TabStringKey,
    TimedMessage,
    ValueField,
    VelocityField,
    mido_bundle_iterator,
    msg_note_off,
)
from minipat.midi import parse_messages
from minipat.tab import TabInst
from minipat.time import (
    Bpc,
    Cps,
    CycleArc,
    CycleDeltaLike,
    CycleTime,
    PosixTime,
    mk_cycle_delta,
)
from minipat.types import Velocity

# LilyPond note names for each semitone in an octave
_LILYPOND_NOTE_NAMES: list[str] = [
    "c",
    "cis",
    "d",
    "dis",
    "e",
    "f",
    "fis",
    "g",
    "gis",
    "a",
    "ais",
    "b",
]

# LilyPond tuning definitions for different instruments
_LILYPOND_TUNINGS: dict[TabInst, str] = {
    TabInst.StandardGuitar: "guitar-tuning",
    TabInst.DropDGuitar: "guitar-drop-d-tuning",
    TabInst.StandardBass: "bass-tuning",
    TabInst.Ukulele: "ukulele-tuning",
    TabInst.Mandolin: "mandolin-tuning",
    TabInst.Banjo: "banjo-open-g-tuning",
}


def render_midi(
    flows: Iterable[tuple[CycleDeltaLike, FlowLike]],
    cps: Optional[Cps] = None,
    bpc: Optional[Bpc] = None,
    default_velocity: Optional[Velocity] = None,
) -> mido.MidiFile:
    events: EvHeap[MidiAttrs] = ev_heap_empty()
    start_time = CycleTime(Fraction(0))
    current_time = start_time

    for duration_like, flow_like in flows:
        # Convert duration to CycleDelta
        duration = mk_cycle_delta(duration_like)
        stream = convert_to_midi_stream(flow_like)

        # Create arc for this section
        end_time = CycleTime(current_time + duration)
        arc = CycleArc(current_time, end_time)

        # Unstream events in this arc
        section_events = stream.unstream(arc)

        # Add all events to the heap
        for span, ev in section_events:
            events = events.insert(span, ev)

        # Move to next time position
        current_time = end_time

    return render_midi_events(
        start=start_time,
        events=events,
        cps=cps,
        bpc=bpc,
        default_velocity=default_velocity,
    )


def render_midi_events(
    start: CycleTime,
    events: EvHeap[MidiAttrs],
    cps: Optional[Cps] = None,
    bpc: Optional[Bpc] = None,
    default_velocity: Optional[Velocity] = None,
) -> mido.MidiFile:
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
        render_midi(CycleTime(Fraction(0)), events, "output.mid", cps=Cps(2.0), bpc=Bpc(4))
    """
    cps = cps if cps is not None else Cps(Fraction(1, 2))
    bpc = bpc if bpc is not None else Bpc(4)
    default_velocity = (
        default_velocity if default_velocity is not None else DEFAULT_VELOCITY
    )

    # Calculate ticks_per_beat (ticks per quarter note)
    # Standard MIDI files typically use 480 ticks per quarter note
    ticks_per_beat = 480

    # Create a new MIDI file with one track (type=0 for single track file)
    mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

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
        msgs = parse_messages(Orbit(0), ev.val, default_velocity)

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

    return mid


def play_midi(mid: mido.MidiFile, name: Optional[str] = None) -> None:
    """Play a MIDI file using FluidSynth.

    Args:
        mid: The MIDI file to play
        name: Optional name for the temporary file (defaults to "minipat.mid")
    """
    # Save MIDI file to temp directory
    temp_dir = Path(tempfile.gettempdir())
    filename = name or "minipat.mid"
    if not filename.endswith(".mid"):
        filename += ".mid"
    filepath = temp_dir / filename
    mid.save(filepath)

    # Get default soundfont path
    soundfont_path = Path.home() / ".local" / "share" / "minipat" / "sf" / "default.sf2"

    # Check if fluidsynth is available
    try:
        subprocess.run(
            ["which", "fluidsynth"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "FluidSynth is not installed. Please install it to play MIDI files."
        )

    # Check if soundfont exists
    if not soundfont_path.exists():
        raise FileNotFoundError(f"Soundfont not found at {soundfont_path}")

    # Play the MIDI file using fluidsynth
    # -a alsa: use ALSA audio driver (Linux) or auto-detect on other platforms
    # -i: don't read commands from stdin (non-interactive)
    # -q: quiet mode (suppress most output)
    cmd = [
        "fluidsynth",
        "-qi",
        str(soundfont_path),
        str(filepath),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to play MIDI file: {e.stderr}")


def render_lilypond(
    start: CycleTime,
    events: EvHeap[MidiAttrs],
    name: str,
    directory: Optional[Path] = None,
    cps: Optional[Cps] = None,
    bpc: Optional[Bpc] = None,
) -> Path:
    """Render an EvHeap to LilyPond file and compile to PDF with tablature information.

    Args:
        start: Start cycle time - events will be timed relative to this
        events: The event heap containing MIDI attribute events with tab info
        name: Filename without extension (e.g., "my_song")
        directory: Output directory (defaults to temporary directory if None)
        cps: Cycles per second (tempo) - affects note duration calculations
        bpc: Beats per cycle (time signature numerator)

    Returns:
        Path to the generated PDF file

    Note:
        One cycle equals one bar, bpc is the numerator of the time signature.
        Events with TabInstKey, TabStringKey, and TabFretKey will be rendered
        with explicit string/fret notation in tablature.
        Both .ly and .pdf files are created in the specified directory.
    """
    cps = cps if cps is not None else Cps(Fraction(1, 2))
    bpc = bpc if bpc is not None else Bpc(4)

    # Determine output directory and create paths
    if directory is None:
        import tempfile

        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = directory
        output_dir.mkdir(parents=True, exist_ok=True)

    ly_path = output_dir / f"{name}.ly"
    pdf_path = output_dir / f"{name}.pdf"

    # Collect events with timing and tab information
    timed_events: list[tuple[float, float, MidiAttrs]] = []

    for span, ev in events:
        # Calculate timing relative to start
        event_start_cycles = float(span.active.start - start)
        event_duration_cycles = float(span.active.end - span.active.start)

        timed_events.append((event_start_cycles, event_duration_cycles, ev.val))

    # Sort events by start time
    timed_events.sort(key=lambda x: x[0])

    # Group events by their timing to handle chords
    chord_groups: dict[float, list[tuple[float, MidiAttrs]]] = {}
    for start_time, duration, attrs in timed_events:
        if start_time not in chord_groups:
            chord_groups[start_time] = []
        chord_groups[start_time].append((duration, attrs))

    # Generate LilyPond content
    version_line = '\\version "2.24.4"'

    # Determine instrument type (use first tab instrument found, default to guitar)
    instrument = TabInst.StandardGuitar
    tab_inst_key = TabInstKey()
    for _, duration_attrs_list in chord_groups.items():
        for _, attrs in duration_attrs_list:
            try:
                instrument = attrs.get(tab_inst_key)
                break
            except KeyError:
                continue
        else:
            continue
        break

    # Note: instrument configuration is available via TAB_CONFIGS[instrument] if needed

    # Build the music content
    music_parts = []
    for event_time in sorted(chord_groups.keys()):
        duration_attrs_list = chord_groups[event_time]

        if len(duration_attrs_list) == 1:
            # Single note
            duration, attrs = duration_attrs_list[0]
            note_str = _format_lilypond_note(attrs, duration, float(bpc))
            if note_str:
                music_parts.append(note_str)
        else:
            # Chord
            chord_notes = []
            chord_duration = None
            for duration, attrs in duration_attrs_list:
                note_str = _format_lilypond_note(
                    attrs, duration, float(bpc), in_chord=True
                )
                if note_str:
                    chord_notes.append(note_str)
                    if chord_duration is None:
                        chord_duration = duration

            if chord_notes:
                if len(chord_notes) == 1:
                    music_parts.append(chord_notes[0])
                else:
                    duration_suffix = _get_lilypond_duration(
                        chord_duration or 1.0, float(bpc)
                    )
                    music_parts.append(f"<{' '.join(chord_notes)}>{duration_suffix}")

    # Create complete LilyPond file content
    content = f"""{version_line}

myMusic = {{
  \\set TabStaff.stringTunings = #{_get_lilypond_tuning(instrument)}
  {" ".join(music_parts) if music_parts else "r4"}
}}

\\score {{
  \\new StaffGroup <<
    \\new Staff {{
      \\clef "treble_8"
      \\myMusic
    }}
    \\new TabStaff {{
      \\myMusic
    }}
  >>
  \\layout {{ }}
}}"""

    # Write LilyPond file
    with open(ly_path, "w") as f:
        f.write(content)

    # Compile to PDF
    _compile_lilypond_to_pdf(ly_path, pdf_path)

    return pdf_path


def _compile_lilypond_to_pdf(ly_path: Path, pdf_path: Path) -> None:
    """Compile a LilyPond file to PDF.

    Args:
        ly_path: Path to the .ly file to compile
        pdf_path: Expected path to the output PDF file

    Raises:
        RuntimeError: If LilyPond compilation fails
    """
    try:
        # Try with full path first, fall back to just 'lilypond'
        lilypond_cmd = "/usr/local/bin/lilypond"
        for cmd in [lilypond_cmd, "lilypond"]:
            try:
                subprocess.run(
                    [cmd, f"--output={ly_path.stem}", str(ly_path.name)],
                    cwd=ly_path.parent,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                break
            except subprocess.CalledProcessError as e:
                if cmd == "lilypond":  # Last attempt failed
                    raise RuntimeError(f"LilyPond compilation failed: {e.stderr}")
            except FileNotFoundError:
                if cmd == "lilypond":  # Last attempt failed
                    raise RuntimeError("LilyPond not found. Please install LilyPond.")

        # Verify PDF was created
        if not pdf_path.exists():
            raise RuntimeError(f"PDF was not generated at {pdf_path}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LilyPond compilation failed: {e.stderr}")


def _format_lilypond_note(
    attrs: MidiAttrs, duration: float, bpc: float, in_chord: bool = False
) -> Optional[str]:
    """Format a single note with tab information for LilyPond."""
    # Extract note information
    note_key = NoteKey()
    try:
        midi_note = attrs.get(note_key)
    except KeyError:
        return None

    note_name = _midi_to_lilypond_note(midi_note)

    # Get tab information if available
    string_key = TabStringKey()
    try:
        string_num = attrs.get(string_key)
    except KeyError:
        string_num = None

    # Format the note with string indication
    if string_num is not None and not in_chord:
        note_with_string = f"{note_name}\\{string_num}"
    else:
        note_with_string = note_name

    # Add duration for non-chord notes
    if not in_chord:
        duration_suffix = _get_lilypond_duration(duration, bpc)
        return f"{note_with_string}{duration_suffix}"
    else:
        return note_with_string


def _midi_to_lilypond_note(midi_note: int) -> str:
    """Convert MIDI note number to LilyPond note name."""
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    base_name = _LILYPOND_NOTE_NAMES[note_index]

    # LilyPond octave notation: 4 is middle octave, apostrophes for higher, commas for lower
    if octave >= 4:
        octave_suffix = "'" * (octave - 3)
    elif octave == 3:
        octave_suffix = ""
    else:
        octave_suffix = "," * (3 - octave)

    return f"{base_name}{octave_suffix}"


def _get_lilypond_duration(duration_cycles: float, bpc: float) -> str:
    """Convert cycle duration to LilyPond duration notation."""
    # Convert cycle duration to beat duration
    beat_duration = duration_cycles * bpc

    # Map to standard durations (quarter note = 1 beat)
    if beat_duration >= 4:
        return "1"  # whole note
    elif beat_duration >= 2:
        return "2"  # half note
    elif beat_duration >= 1:
        return "4"  # quarter note
    elif beat_duration >= 0.5:
        return "8"  # eighth note
    elif beat_duration >= 0.25:
        return "16"  # sixteenth note
    else:
        return "32"  # thirty-second note


def _get_lilypond_tuning(instrument: TabInst) -> str:
    """Get LilyPond tuning definition for the instrument."""
    return _LILYPOND_TUNINGS.get(instrument, "guitar-tuning")
