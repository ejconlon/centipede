"""File I/O utilities for minipat patterns and events."""

from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import mido

from minipat.chords import chord_data_to_notes
from minipat.dsl import FlowLike, convert_to_midi_stream
from minipat.ev import EvHeap, ev_heap_empty
from minipat.live import Orbit
from minipat.messages import (
    DEFAULT_VELOCITY,
    ChannelField,
    ChordDataKey,
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
from minipat.stream import Stream
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
from minipat.types import ChordData, TabData, Velocity

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

# Mapping of chord names to LilyPond chord symbols
_CHORD_MAP: dict[str, str] = {
    "maj": "",  # Major chords have no suffix in LilyPond
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
    "dom9": "9",  # Dominant 9th
    "dom11": "11",  # Dominant 11th
    "dom13": "13",  # Dominant 13th
    "maj9": "maj9",
    "min9": "m9",
    "9": "9",
    "maj11": "maj11",
    "min11": "m11",
    "11": "11",
    "maj13": "maj13",
    "min13": "m13",
    "13": "13",
    "6": "6",
    "min6": "m6",
    "69": "6.9",
    "min69": "m6.9",
    "dim7": "dim7",
    "mmaj7": "mMaj7",
    "add9": "add9",
    "add11": "add11",
    "add13": "add13",
    "7f5": "7.5-",
    "7s5": "7.5+",
    "7s11": "7.11+",  # Seven sharp 11 (lydian dominant)
    "7f9": "7.9-",
    "7s9": "7.9+",  # Seven sharp 9
    "min7f5": "m7.5-",
    "min7s5": "m7.5+",
    "min7f9": "m7.9-",
    "min7s9": "m7.9+",
    "mins5": "m.5+",  # Minor sharp 5
    "7sus2": "7sus2",
    "7sus4": "7sus4",
    "9sus4": "9sus4",
    "5": "5",
    "1": "",  # Root only
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


def _key_to_lilypond_signature(key: Optional[str]) -> str:
    """Convert key name to LilyPond key signature.

    Args:
        key: Key name like "bb", "f#", "dm", etc. None returns empty string

    Returns:
        LilyPond key signature string (e.g., "\\key bes \\major")
    """
    if not key:
        return ""

    key = key.lower().strip()

    # Map key names to LilyPond key signatures
    key_signatures = {
        # Major keys
        "c": "\\key c \\major",
        "g": "\\key g \\major",
        "d": "\\key d \\major",
        "a": "\\key a \\major",
        "e": "\\key e \\major",
        "b": "\\key b \\major",
        "f#": "\\key fis \\major",
        "c#": "\\key cis \\major",
        "f": "\\key f \\major",
        "bb": "\\key bes \\major",
        "eb": "\\key ees \\major",
        "ab": "\\key aes \\major",
        "db": "\\key des \\major",
        "gb": "\\key ges \\major",
        "cb": "\\key ces \\major",
        # Minor keys
        "am": "\\key a \\minor",
        "em": "\\key e \\minor",
        "bm": "\\key b \\minor",
        "f#m": "\\key fis \\minor",
        "c#m": "\\key cis \\minor",
        "g#m": "\\key gis \\minor",
        "d#m": "\\key dis \\minor",
        "a#m": "\\key ais \\minor",
        "dm": "\\key d \\minor",
        "gm": "\\key g \\minor",
        "cm": "\\key c \\minor",
        "fm": "\\key f \\minor",
        "bbm": "\\key bes \\minor",
        "ebm": "\\key ees \\minor",
        "abm": "\\key aes \\minor",
    }

    return key_signatures.get(key, "")


def _key_uses_flats(key: Optional[str]) -> bool:
    """Determine if a key signature typically uses flats or sharps.

    Args:
        key: Key name like "bb", "f#", "c", etc. None defaults to C major (no preference)

    Returns:
        True if the key typically uses flat notation, False for sharp notation
    """
    if not key:
        return False  # C major, no preference

    key = key.lower().strip()

    # Keys that typically use flats
    flat_keys = {
        "f",  # F major (1 flat)
        "bb",  # Bb major (2 flats)
        "eb",  # Eb major (3 flats)
        "ab",  # Ab major (4 flats)
        "db",  # Db major (5 flats)
        "gb",  # Gb major (6 flats)
        "cb",  # Cb major (7 flats)
        "dm",  # D minor (1 flat, relative to F major)
        "gm",  # G minor (2 flats, relative to Bb major)
        "cm",  # C minor (3 flats, relative to Eb major)
        "fm",  # F minor (4 flats)
        "bbm",  # Bb minor (5 flats)
        "ebm",  # Eb minor (6 flats)
        "abm",  # Ab minor (7 flats)
    }

    return key in flat_keys


def render_lilypond(
    arc: CycleArc,
    tab_stream: Stream[TabData],
    chord_stream: Optional[Stream[ChordData]] = None,
    name: str = "lilypond_output",
    directory: Optional[Path] = None,
    cps: Optional[Cps] = None,
    bpc: Optional[Bpc] = None,
    pdf: bool = False,
    svg: bool = False,
    key: Optional[str] = None,
) -> dict[str, Path]:
    """Render streams to LilyPond file and compile to PDF and/or SVG with tablature information.

    Args:
        arc: The cycle arc defining the time range to render
        tab_stream: Stream[TabData] containing tab data (e.g., from tab_data_stream)
        chord_stream: Optional Stream[ChordData] for rendering chord names (e.g., from chord_data_stream)
        name: Filename without extension (e.g., "my_song")
        directory: Output directory (defaults to temporary directory if None)
        cps: Cycles per second (tempo) - affects note duration calculations
        bpc: Beats per cycle (time signature numerator)
        pdf: Whether to generate PDF output (default: False)
        svg: Whether to generate SVG output (default: False)
        key: Key signature (e.g., "bb", "f#", "c") - determines flat vs sharp notation for chords

    Returns:
        Dictionary with keys for generated files:
        - "ly": Path to the LilyPond source file (always generated)
        - "pdf": Path to the PDF file (if pdf=True)
        - "svg": Path to the SVG file (if svg=True)

    Note:
        One cycle equals one bar, bpc is the numerator of the time signature.
        TabData objects are converted to MIDI attributes with tab information
        and rendered with explicit string/fret notation in tablature.
        The .ly source file is always created in the specified directory.
        Use tab_data_stream() and chord_data_stream() to create appropriate streams.
    """
    cps = cps if cps is not None else Cps(Fraction(1, 2))
    bpc = bpc if bpc is not None else Bpc(4)

    # Determine chord notation preference based on key signature
    prefer_flats = _key_uses_flats(key)
    key_signature = _key_to_lilypond_signature(key)

    # Determine output directory and create paths
    if directory is None:
        import tempfile

        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = directory
        output_dir.mkdir(parents=True, exist_ok=True)

    ly_path = output_dir / f"{name}.ly"
    pdf_path = output_dir / f"{name}.pdf"
    svg_path = output_dir / f"{name}.svg"

    # Collect events with timing and tab information
    timed_events: list[tuple[float, float, MidiAttrs]] = []
    chord_names: dict[float, ChordData] = {}  # Store chord data by timing

    # Get the start time from the arc for relative timing calculations
    start = arc.start

    # Process tab_stream into events by converting TabData to MidiAttrs
    all_events: EvHeap[MidiAttrs] = ev_heap_empty()
    for span, tab_ev in tab_stream.unstream(arc):
        # Convert TabData to MidiAttrs using TabBinder
        from minipat.combinators import TabBinder

        tab_binder = TabBinder()
        midi_attrs_pat = tab_binder.apply(tab_ev.val)

        # Process the pattern and add events to our heap
        # The TabBinder creates patterns that span a unit cycle (0-1), so we need to
        # unstream over a unit cycle and then shift the results to the correct timing
        unit_cycle = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
        for inner_span, inner_ev in Stream.pat(midi_attrs_pat).unstream(unit_cycle):
            # Shift the span to match the timing of the original TabData event
            span_duration = span.active.end - span.active.start
            adjusted_start = span.active.start + (
                inner_span.active.start * span_duration
            )
            adjusted_end = span.active.start + (inner_span.active.end * span_duration)

            from minipat.time import CycleSpan

            adjusted_span = CycleSpan(
                inner_span.whole,
                CycleArc(CycleTime(adjusted_start), CycleTime(adjusted_end)),
            )
            all_events = all_events.insert(adjusted_span, inner_ev)

    # Process chord_stream to collect chord names by timing
    if chord_stream is not None:
        for span, chord_ev in chord_stream.unstream(arc):
            event_start_cycles = float(span.active.start - start)
            # ChordData is directly in the event now
            chord_names[event_start_cycles] = chord_ev.val

    events_to_process = all_events

    for span, ev in events_to_process:
        # Calculate timing relative to start
        event_start_cycles = float(span.active.start - start)
        event_duration_cycles = float(span.active.end - span.active.start)

        # Check if this event contains chord data
        chord_data_key = ChordDataKey()
        try:
            chord_data = ev.val.get(chord_data_key)
            # Store chord data for chord name annotation
            chord_names[event_start_cycles] = chord_data
            # Expand chord data into individual note events
            chord_notes = chord_data_to_notes(chord_data)
            for note in chord_notes:
                # Create new attributes with the note
                note_attrs = ev.val.put(NoteKey(), note)
                # Remove chord data to avoid confusion
                note_attrs = note_attrs.remove(chord_data_key)
                timed_events.append(
                    (event_start_cycles, event_duration_cycles, note_attrs)
                )
        except KeyError:
            # Regular event without chord data
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

    # Build the music content and chord symbols separately
    music_parts = []
    chord_symbols = []

    # Process chord symbols independently of tab/note events
    if chord_names:
        prev_chord_time = 0.0
        sorted_chord_times = sorted(chord_names.keys())

        for i, event_time in enumerate(sorted_chord_times):
            # Add spacer rest if needed
            if event_time > prev_chord_time:
                rest_duration = event_time - prev_chord_time
                rest_suffix = _get_lilypond_duration(rest_duration, float(bpc))
                chord_symbols.append(f"s{rest_suffix}")

            # Add chord symbol
            chord_data = chord_names[event_time]
            # Get root note and chord type for LilyPond
            # Use notation preference based on key signature
            root_note_str = _midi_to_lilypond_note(
                chord_data.root_note, prefer_flats=prefer_flats
            )
            chord_type = _CHORD_MAP.get(chord_data.chord_name, chord_data.chord_name)

            # Get duration to next chord or end of piece
            if i + 1 < len(sorted_chord_times):
                chord_duration = sorted_chord_times[i + 1] - event_time
            else:
                # Last chord - sustain for remaining duration
                chord_duration = float(arc.end - arc.start) - event_time

            duration_suffix = _get_lilypond_duration(chord_duration, float(bpc))
            # Format: root+duration:type (e.g., a4:m or c4 for major)
            if chord_type:
                chord_symbols.append(f"{root_note_str}{duration_suffix}:{chord_type}")
            else:
                # Major chord has no suffix
                chord_symbols.append(f"{root_note_str}{duration_suffix}")
            prev_chord_time = event_time + chord_duration

    # Process tab/note events
    for event_time in sorted(chord_groups.keys()):
        duration_attrs_list = chord_groups[event_time]

        # Handle notes/tabs
        if len(duration_attrs_list) == 1:
            # Single note
            duration, attrs = duration_attrs_list[0]
            note_str = _format_lilypond_note(attrs, duration, float(bpc))
            if note_str:
                music_parts.append(note_str)
        else:
            # Chord
            chord_note_strings: list[str] = []
            chord_note_duration: Optional[float] = None
            for duration, attrs in duration_attrs_list:
                note_str = _format_lilypond_note(
                    attrs, duration, float(bpc), in_chord=True
                )
                if note_str:
                    chord_note_strings.append(note_str)
                    if chord_note_duration is None:
                        chord_note_duration = duration

            if chord_note_strings:
                if len(chord_note_strings) == 1:
                    music_parts.append(chord_note_strings[0])
                else:
                    duration_suffix = _get_lilypond_duration(
                        chord_note_duration or 1.0, float(bpc)
                    )
                    music_parts.append(
                        f"<{' '.join(chord_note_strings)}>{duration_suffix}"
                    )

    # Create complete LilyPond file content
    # Only include ChordNames if we have chord symbols
    chord_names_section = ""
    if chord_symbols:
        chord_names_section = f"""myChords = \\chordmode {{
  {" ".join(chord_symbols)}
}}

"""

    # Generate rests for the full duration if no music parts
    if not music_parts:
        total_duration = float(arc.end - arc.start)
        rest_notation = _get_lilypond_duration(total_duration, float(bpc))
        # Use R for multi-bar rests, r for single notes
        if total_duration * float(bpc) > 4:
            music_content = f"R{rest_notation}"
        else:
            music_content = f"r{rest_notation}"
    else:
        music_content = " ".join(music_parts)

    content = f"""{version_line}

{chord_names_section}myMusic = {{
  \\set TabStaff.stringTunings = #{_get_lilypond_tuning(instrument)}
  {key_signature}
  \\time 4/4
  {music_content}
}}

\\score {{
  <<
    {"\\new ChordNames { \\myChords }" if chord_symbols else ""}
    \\new StaffGroup <<
      \\new Staff {{
        \\clef "treble_8"
        \\override StringNumber.stencil = ##f
        \\myMusic
      }}
      \\new TabStaff {{
        \\myMusic
      }}
    >>
  >>
  \\layout {{ }}
}}"""

    # Write LilyPond file
    with open(ly_path, "w") as f:
        f.write(content)

    # Compile to PDF and/or SVG
    _compile_lilypond_to_outputs(ly_path, pdf_path, svg_path, pdf, svg)

    # Build result dictionary with generated files
    result = {"ly": ly_path}
    if pdf:
        result["pdf"] = pdf_path
    if svg:
        result["svg"] = svg_path

    return result


def _compile_lilypond_to_outputs(
    ly_path: Path,
    pdf_path: Path,
    svg_path: Path,
    generate_pdf: bool = True,
    generate_svg: bool = True,
) -> None:
    """Compile a LilyPond file to PDF and/or SVG.

    Args:
        ly_path: Path to the .ly file to compile
        pdf_path: Expected path to the output PDF file
        svg_path: Expected path to the output SVG file
        generate_pdf: Whether to generate PDF output
        generate_svg: Whether to generate SVG output

    Raises:
        RuntimeError: If LilyPond compilation fails
    """
    try:
        # Try with full path first, fall back to just 'lilypond'
        lilypond_cmd = "/usr/local/bin/lilypond"
        for cmd in [lilypond_cmd, "lilypond"]:
            try:
                # Compile to PDF if requested
                if generate_pdf:
                    subprocess.run(
                        [cmd, "--pdf", f"--output={ly_path.stem}", str(ly_path.name)],
                        cwd=ly_path.parent,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                # Compile to SVG if requested
                if generate_svg:
                    subprocess.run(
                        [cmd, "--svg", f"--output={ly_path.stem}", str(ly_path.name)],
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

        # Verify requested outputs were created
        if generate_pdf and not pdf_path.exists():
            raise RuntimeError(f"PDF was not generated at {pdf_path}")
        if generate_svg and not svg_path.exists():
            raise RuntimeError(f"SVG was not generated at {svg_path}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LilyPond compilation failed: {e.stderr}")


def _chord_data_to_lilypond_name(
    chord_data: ChordData, prefer_flats: bool = True
) -> str:
    """Convert ChordData to a LilyPond chord name."""
    # Get the root note name
    # Use notation preference (defaults to flats for standard jazz notation)
    root_note_str = _midi_to_lilypond_note(
        chord_data.root_note, prefer_flats=prefer_flats
    )

    # Convert chord name to LilyPond format
    chord_name = chord_data.chord_name

    lily_chord_name = _CHORD_MAP.get(chord_name, chord_name)

    # Handle inversions and drop voicings in chord name comment
    modifiers = []
    for mod_type, mod_value in chord_data.modifiers:
        if mod_type == "inv":
            modifiers.append(f"inv{mod_value}")
        elif mod_type == "drop":
            modifiers.append(f"drop{mod_value}")

    modifier_str = f" ({', '.join(modifiers)})" if modifiers else ""

    return f"{root_note_str}:{lily_chord_name}{modifier_str}"


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

    # Add duration for non-chord notes
    if not in_chord:
        duration_suffix = _get_lilypond_duration(duration, bpc)
        # Format: note + duration + string indication (hidden from display)
        if string_num is not None:
            return f"{note_name}{duration_suffix}\\{string_num}"
        else:
            return f"{note_name}{duration_suffix}"
    else:
        # For chord notes, just return note name (no duration, string handled elsewhere)
        return note_name


def _midi_to_lilypond_note(midi_note: int, prefer_flats: bool = False) -> str:
    """Convert MIDI note number to LilyPond note name.

    Args:
        midi_note: MIDI note number (0-127)
        prefer_flats: If True, use flat notation for accidentals (bes instead of ais)
                     This is needed for chord symbols
    """
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12

    if prefer_flats and note_index in [1, 3, 6, 8, 10]:
        # Use flat notation for black keys
        flat_names = {
            1: "des",  # Db
            3: "ees",  # Eb
            6: "ges",  # Gb
            8: "aes",  # Ab
            10: "bes",  # Bb
        }
        base_name = flat_names[note_index]
    else:
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

    # For multi-bar durations (more than 4 beats)
    if beat_duration > 4:
        # Calculate full bars and remaining beats
        full_bars = int(beat_duration // 4)
        remaining_beats = beat_duration % 4

        if remaining_beats == 0:
            # Exact number of bars
            return f"1*{full_bars}"
        else:
            # Bars plus remaining beats - need to handle this more carefully
            # For simplicity, just use the multi-bar notation
            return f"1*{duration_cycles:.0f}"
    # Map to standard durations (quarter note = 1 beat)
    elif beat_duration == 4:
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
