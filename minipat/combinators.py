from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Sequence, override

from minipat.messages import (
    Channel,
    ChannelField,
    ChannelKey,
    ControlField,
    ControlNum,
    ControlNumKey,
    ControlVal,
    ControlValKey,
    MidiAttrs,
    Note,
    NoteField,
    NoteKey,
    Program,
    ProgramField,
    ProgramKey,
    ValueField,
    Velocity,
    VelocityField,
    VelocityKey,
)
from minipat.parser import parse_pattern
from minipat.stream import MergeStrat, Stream
from spiny.dmap import DMap

# =============================================================================
# Pattern Parsing and Rendering
# =============================================================================


class ElemParser[V](metaclass=ABCMeta):
    """Abstract base class for parsing and rendering pattern values.

    An ElemParser handles the bidirectional conversion between string representations
    in patterns and strongly-typed values. This enables the pattern system to work
    with domain-specific  types like MIDI notes and velocities while maintaining
    readable string syntax in patterns.

    The pattern system uses str values which ElemParser instances split apart.

    Type Parameters:
        V: The value type this parser handles (e.g., Note, Vel)

    Example:
        A NoteElemParser might parse "c4" -> Note(60) and render Note(60) -> "c4"
    """

    @classmethod
    @abstractmethod
    def parse(cls, s: str) -> V:
        """Parse a string pattern value into a strongly-typed value.

        Args:
            s: A string containing the pattern value

        Returns:
            The parsed value of type V

        Raises:
            ValueError: If the string cannot be parsed into a valid value
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def render(cls, value: V) -> str:
        """Render a strongly-typed value back to a string pattern representation.

        Args:
            value: The strongly-typed value to render

        Returns:
            A string representation suitable for patterns
        """
        raise NotImplementedError


class NoteNumElemParser(ElemParser[Note]):
    """Selector for parsing numeric MIDI note representations.

    Handles direct numeric MIDI note values in the range 0-127.
    This is useful when you want to specify exact MIDI note numbers
    rather than using note names.

    Examples:
        "60" -> Note(60)  # Middle C
        "72" -> Note(72)  # C5
        "0"  -> Note(0)   # C-1 (lowest MIDI note)
        "127" -> Note(127) # G9 (highest MIDI note)

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-127 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> Note:
        note_num = int(s)
        return NoteField.mk(note_num)

    @override
    @classmethod
    def render(cls, value: Note) -> str:
        return str(value)


class NoteNameElemParser(ElemParser[Note]):
    """Selector for parsing musical note names with octave numbers.

    Converts standard musical notation to MIDI note numbers. Supports
    note names (C, D, E, F, G, A, B) with optional sharps (#) or flats (b)
    and octave numbers.

    Note naming follows standard MIDI convention where:
    - C4 = 60 (Middle C)
    - Octaves range from -1 to 9
    - Each octave starts at C

    Examples:
        "c4"  -> Note(60)  # Middle C
        "c#4" -> Note(61)  # C sharp 4
        "db4" -> Note(61)  # D flat 4 (enharmonic equivalent)
        "a0"  -> Note(21)  # A0 (lowest A on piano)
        "c8"  -> Note(108) # C8 (high C)

    Parsing is case-insensitive: "C4", "c4", and "C4" are all equivalent.

    Raises:
        ValueError: If the note name format is invalid, octave is missing,
                   or the resulting MIDI note is outside the valid range (0-127)
    """

    @override
    @classmethod
    def parse(cls, s: str) -> Note:
        # Parse note names like "c4" (middle C = 60), "d#4", "eb3", etc.
        note_str = s.lower()

        # Note name to semitone mapping (C is 0)
        note_map = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

        # Parse note name and octave
        if len(note_str) < 2:
            raise ValueError(f"Invalid note name: {s}")

        # Get base note
        base_note = note_str[0]
        if base_note not in note_map:
            raise ValueError(f"Invalid note name: {s}")

        semitone = note_map[base_note]
        pos = 1

        # Check for sharp or flat
        if pos < len(note_str):
            if note_str[pos] == "#":
                semitone += 1
                pos += 1
            elif note_str[pos] == "b":
                semitone -= 1
                pos += 1

        # Get octave
        try:
            octave = int(note_str[pos:])
        except ValueError:
            raise ValueError(f"Invalid octave in note: {s}")

        # Calculate MIDI note number (C4 = 60, so octave 4 starts at 60-12=48 for C)
        # MIDI note = (octave + 1) * 12 + semitone
        midi_note = (octave + 1) * 12 + semitone

        return NoteField.mk(midi_note)

    @override
    @classmethod
    def render(cls, value: Note) -> str:
        # Convert MIDI note back to note name
        note_num = int(value)
        octave = (note_num // 12) - 1
        semitone = note_num % 12

        # Semitone to note name mapping (using sharps)
        note_names = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
        note_name = note_names[semitone] + str(octave)

        return note_name


class VelElemParser(ElemParser[Velocity]):
    """Selector for parsing MIDI velocity values.

    Handles MIDI velocity values which control the volume/intensity of notes.
    Velocities range from 0 (silent) to 127 (maximum volume).

    In musical terms:
    - 0: Note off (silent)
    - 1-31: Very soft (ppp)
    - 32-63: Soft (pp-p)
    - 64-95: Medium (mp-mf)
    - 96-127: Loud (f-fff)

    Examples:
        "0"   -> Vel(0)   # Silent
        "64"  -> Vel(64)  # Medium velocity (common default)
        "127" -> Vel(127) # Maximum velocity
        "100" -> Vel(100) # Loud

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-127 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> Velocity:
        vel_num = int(s)
        return VelocityField.mk(vel_num)

    @override
    @classmethod
    def render(cls, value: Velocity) -> str:
        return str(value)


class ChannelElemParser(ElemParser[Channel]):
    """Selector for parsing MIDI channel values.

    Handles MIDI channel values which specify which MIDI channel to use.
    Channels range from 0 to 15 (16 channels total).

    Examples:
        "0"   -> Channel(0)   # Channel 1 (0-based)
        "9"   -> Channel(9)   # Channel 10 (commonly drums)
        "15"  -> Channel(15)  # Channel 16

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-15 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> Channel:
        channel_num = int(s)
        return ChannelField.mk(channel_num)

    @override
    @classmethod
    def render(cls, value: Channel) -> str:
        return str(value)


class ProgramElemParser(ElemParser[Program]):
    """Selector for parsing MIDI program values.

    Handles MIDI program change values which select instrument patches.
    Programs range from 0 to 127.

    Examples:
        "0"   -> Program(0)   # Acoustic Grand Piano
        "40"  -> Program(40)  # Violin
        "127" -> Program(127) # Gunshot

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-127 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> Program:
        program_num = int(s)
        return ProgramField.mk(program_num)

    @override
    @classmethod
    def render(cls, value: Program) -> str:
        return str(value)


class ControlNumElemParser(ElemParser[ControlNum]):
    """Selector for parsing MIDI control number values.

    Handles MIDI control change numbers which specify which parameter to control.
    Control numbers range from 0 to 127.

    Common control numbers:
        "1"   -> ControlNum(1)   # Modulation Wheel
        "7"   -> ControlNum(7)   # Volume
        "10"  -> ControlNum(10)  # Pan
        "64"  -> ControlNum(64)  # Sustain Pedal

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-127 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> ControlNum:
        control_num = int(s)
        return ControlField.mk(control_num)

    @override
    @classmethod
    def render(cls, value: ControlNum) -> str:
        return str(value)


class ControlValElemParser(ElemParser[ControlVal]):
    """Selector for parsing MIDI control value values.

    Handles MIDI control change values which specify the parameter value.
    Control values range from 0 to 127.

    Examples:
        "0"   -> ControlVal(0)   # Minimum value
        "64"  -> ControlVal(64)  # Center/default value
        "127" -> ControlVal(127) # Maximum value

    Raises:
        ValueError: If the string is not a valid integer or is outside 0-127 range
    """

    @override
    @classmethod
    def parse(cls, s: str) -> ControlVal:
        control_val = int(s)
        return ValueField.mk(control_val)

    @override
    @classmethod
    def render(cls, value: ControlVal) -> str:
        return str(value)


# =============================================================================
# Pattern Stream Functions
# =============================================================================


def _convert_midinote(s: str) -> MidiAttrs:
    """Convert numeric note to MIDI attributes."""
    note = NoteNumElemParser.parse(s)
    return DMap.singleton(NoteKey(), note)


def _convert_note(s: str) -> MidiAttrs:
    """Convert note name to MIDI attributes."""
    note = NoteNameElemParser.parse(s)
    return DMap.singleton(NoteKey(), note)


def _convert_vel(s: str) -> MidiAttrs:
    """Convert velocity to MIDI attributes."""
    velocity = VelElemParser.parse(s)
    return DMap.singleton(VelocityKey(), velocity)


def _convert_channel(s: str) -> MidiAttrs:
    """Convert channel to MIDI attributes."""
    channel = ChannelElemParser.parse(s)
    return DMap.singleton(ChannelKey(), channel)


def _convert_program(s: str) -> MidiAttrs:
    """Convert program to MIDI attributes."""
    program = ProgramElemParser.parse(s)
    return DMap.singleton(ProgramKey(), program)


def _convert_control(s: str) -> MidiAttrs:
    """Convert control number to MIDI attributes."""
    control_num = ControlNumElemParser.parse(s)
    return DMap.singleton(ControlNumKey(), control_num)


def _convert_value(s: str) -> MidiAttrs:
    """Convert control value to MIDI attributes."""
    control_val = ControlValElemParser.parse(s)
    return DMap.singleton(ControlValKey(), control_val)


def midinote_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from numeric MIDI notes.

    Parses a pattern string containing numeric MIDI note values (0-127)
    and creates a stream of MIDI attributes.

    Examples:
        midinote("60 62 64")     # C4, D4, E4 (C major triad)
        midinote("36 ~ 42")      # Kick, rest, snare pattern
        midinote("[60,64,67]")   # C major chord (simultaneous)
    """
    return Stream.pat(parse_pattern(s).map(_convert_midinote))


def note_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from note names.

    Parses a pattern string containing musical note names with octaves
    and creates a stream of MIDI attributes.

    Examples:
        note("c4 d4 e4")         # C major scale fragment
        note("c4 ~ g4")          # C4, rest, G4
        note("[c4,e4,g4]")       # C major chord (simultaneous)
        note("c#4 db5 f4")       # Mixed sharps and flats
    """
    return Stream.pat(parse_pattern(s).map(_convert_note))


def vel_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from velocity values.

    Parses a pattern string containing MIDI velocity values (0-127)
    and creates a stream of MIDI attributes for controlling note dynamics.

    Examples:
        vel_stream("64 80 100")         # Medium, loud, very loud
        vel_stream("127 0 64")          # Loud, silent, medium
        vel_stream("100*8")             # Repeat loud velocity 8 times
    """
    return Stream.pat(parse_pattern(s).map(_convert_vel))


def channel_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from channel values.

    Parses a pattern string containing MIDI channel values (0-15)
    and creates a stream of MIDI attributes for specifying channels.
    If not specified, the orbit number will be used as the channel.

    Examples:
        channel_stream("0 1 9")          # Channels 1, 2, 10 (drums)
        channel_stream("15 ~ 0")         # Channel 16, rest, Channel 1
        channel_stream("9*4")            # Repeat Channel 10 (drums) 4 times
    """
    return Stream.pat(parse_pattern(s).map(_convert_channel))


def program_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from program values.

    Parses a pattern string containing MIDI program values (0-127)
    and creates a stream of MIDI attributes for program change messages.

    Examples:
        program_stream("0 1 40")         # Piano, Bright Piano, Violin
        program_stream("128 ~ 0")        # Invalid program, rest, Piano (will error on 128)
        program_stream("1*4")            # Repeat Bright Piano 4 times
    """
    return Stream.pat(parse_pattern(s).map(_convert_program))


def control_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from control number values.

    Parses a pattern string containing MIDI control numbers (0-127)
    and creates a stream of MIDI attributes for control change messages.
    Note: Control change messages also require a control value.

    Examples:
        control_stream("1 7 10")         # Modulation, Volume, Pan
        control_stream("64 ~ 1")         # Sustain, rest, Modulation
        control_stream("7*8")            # Repeat Volume control 8 times
    """
    return Stream.pat(parse_pattern(s).map(_convert_control))


def value_stream(s: str) -> Stream[MidiAttrs]:
    """Create stream from control value values.

    Parses a pattern string containing MIDI control values (0-127)
    and creates a stream of MIDI attributes for control change messages.
    Note: Control change messages also require a control number.

    Examples:
        value_stream("0 64 127")         # Min, center, max values
        value_stream("127 ~ 0")          # Max, rest, min
        value_stream("64*8")             # Repeat center value 8 times
    """
    return Stream.pat(parse_pattern(s).map(_convert_value))


def _merge_attrs(x: MidiAttrs, y: MidiAttrs) -> MidiAttrs:
    """Merge two MIDI attribute maps."""
    return x.merge(y)


def combine(s1: Stream[MidiAttrs], s2: Stream[MidiAttrs]) -> Stream[MidiAttrs]:
    """Combine two MIDI streams into a single stream.

    Takes two streams of MIDI attributes and merges them together
    using inner join semantics. This allows you to layer different
    MIDI aspects (notes, velocities, etc.) into complete MIDI events.

    Args:
        s1: First MIDI attribute stream
        s2: Second MIDI attribute stream

    Returns:
        A single stream containing merged MIDI attributes

    Examples:
        # Combine notes with velocities
        notes = note_stream("c4 d4 e4")
        velocities = vel_stream("64 80 100")
        combined = combine(notes, velocities)
    """
    return s1.apply(MergeStrat.Inner, _merge_attrs, s2)


def combine_all(ss: Sequence[Stream[MidiAttrs]]) -> Stream[MidiAttrs]:
    """Combine multiple MIDI streams into a single stream.

    Takes multiple streams of MIDI attributes and merges them together
    using inner join semantics. This allows you to layer different
    MIDI aspects (notes, velocities, etc.) into complete MIDI events.

    Args:
        *ss: Variable number of MIDI attribute streams to combine

    Returns:
        A single stream containing merged MIDI attributes, or silence
        if no streams are provided

    Examples:
        # Combine notes with velocities
        notes = note_stream("c4 d4 e4")
        velocities = vel_stream("64 80 100")
        combined = combine_all(notes, velocities)

        # Layer multiple attributes
        notes = note_stream("c4 ~ g4")
        velocities = vel_stream("100 ~ 80")
        channels = ...  # hypothetical channel stream
        result = combine_all(notes, velocities, channels)
    """
    if len(ss) == 0:
        return Stream.silent()
    elif len(ss) == 1:
        return ss[0]
    else:
        acc = ss[0]
        for el in ss[1:]:
            acc = combine(acc, el)
        return acc
