from __future__ import annotations

from fractions import Fraction
from typing import Sequence, override

from minipat.kit import DrumSoundElemParser, Kit
from minipat.messages import (
    Channel,
    ChannelField,
    ControlField,
    ControlNum,
    ControlVal,
    MessageField,
    MidiAttrs,
    Note,
    NoteField,
    Program,
    ProgramField,
    ValueField,
    Velocity,
    VelocityField,
)
from minipat.parser import parse_pattern
from minipat.pat import Pat, PatBinder
from minipat.stream import MergeStrat, Stream
from spiny.arrow import Iso
from spiny.common import Singleton
from spiny.dmap import DMap

# =============================================================================
# Type Aliases
# =============================================================================


type IntStreamLike = str | Pat[int] | Stream[int]
"""Types accepted by functions for integer patterns.

Accepts:
- str: Pattern string containing integers (e.g., "0 5 7")
- Pat[int]: Pattern of integers
- Stream[int]: Pre-constructed integer stream
"""

type StringStreamLike = str | Pat[str] | Stream[str]
"""Types accepted by functions for string patterns.

Accepts:
- str: Pattern string (e.g., "c4 d4 e4")
- Pat[str]: Pattern of strings
- Stream[str]: Pre-constructed string stream
"""


# =============================================================================
# Pattern Parsing and Rendering
# =============================================================================

# Default octave for notes specified without octave number
DEFAULT_OCTAVE = 4

# Note name to semitone mapping (C is 0)
NOTE_NAME_TO_SEMITONE: dict[str, int] = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
}

# Semitone to note name mapping (using sharps)
SEMITONE_TO_NOTE_NAME: list[str] = [
    "c",
    "c#",
    "d",
    "d#",
    "e",
    "f",
    "f#",
    "g",
    "g#",
    "a",
    "a#",
    "b",
]


class IntBinder(PatBinder[str, int]):
    """Binder that converts string integers to integer values."""

    def apply(self, value: str) -> Pat[int]:
        try:
            int_val = int(value)
            return Pat.pure(int_val)
        except ValueError:
            raise ValueError(f"Invalid integer value: '{value}'. Expected integer.")


class IntElemParser(Iso[str, int], Singleton):
    """Parser for integer values from string representations.

    Parses string representations of integers.

    Examples:
        "42"  -> 42
        "-5"  -> -5
        "0"   -> 0

    Raises:
        ValueError: If the string is not a valid integer
    """

    @override
    def forward(self, value: str) -> int:
        return int(value)

    @override
    def backward(self, value: int) -> str:
        return str(value)


class FractionElemParser(Iso[str, Fraction], Singleton):
    """Parser for Fraction values from string representations.

    Parses string representations of numbers and converts them to Fractions:
    - Integers are converted to fractions with denominator 1
    - Floats are converted to exact fraction representations
    - Fraction strings (e.g., "3/4") are parsed directly

    Examples:
        "42"    -> Fraction(42, 1)
        "3.14"  -> Fraction(157, 50)
        "3/4"   -> Fraction(3, 4)
        "-2.5"  -> Fraction(-5, 2)
        "0.333" -> Fraction(333, 1000)

    Raises:
        ValueError: If the string is not a valid numeric representation
    """

    @override
    def forward(self, value: str) -> Fraction:
        # Fraction constructor handles all these cases:
        # - "3/4" -> Fraction(3, 4)
        # - "42" -> Fraction(42, 1)
        # - "3.14" -> Fraction(157, 50)
        return Fraction(value)

    @override
    def backward(self, value: Fraction) -> str:
        # Render fraction in simplest form
        if value.denominator == 1:
            return str(value.numerator)
        else:
            return f"{value.numerator}/{value.denominator}"


class NoteNumElemParser(Iso[str, Note], Singleton):
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
    def forward(self, value: str) -> Note:
        return NoteField.mk(int(value))

    @override
    def backward(self, value: Note) -> str:
        return str(value)


class NoteNameElemParser(Iso[str, Note], Singleton):
    """Selector for parsing musical note names with optional octave numbers.

    Converts standard musical notation to MIDI note numbers. Supports
    note names (C, D, E, F, G, A, B) with optional sharps (#) or flats (b)
    and optional octave numbers. If no octave is specified, uses DEFAULT_OCTAVE.

    Note naming convention:
    - C0 = 0 (lowest MIDI note C)
    - C4 = 48
    - C5 = 60 (Middle C)
    - Each octave starts at C

    Examples:
        "c5"  -> Note(60)  # Middle C
        "c#5" -> Note(61)  # C sharp 5
        "db5" -> Note(61)  # D flat 5 (enharmonic equivalent)
        "c0"  -> Note(0)   # C0 (MIDI note 0)
        "c"   -> Note(48)  # C with default octave 4
        "f#"  -> Note(54)  # F# with default octave 4

    Parsing is case-insensitive: "C5", "c5", and "C5" are all equivalent.

    Raises:
        ValueError: If the note name format is invalid
                   or the resulting MIDI note is outside the valid range (0-127)
    """

    @override
    def forward(self, s: str) -> Note:
        # Parse note names like "c4" (middle C = 60), "d#4", "eb3", etc.
        # Also supports notes without octave like "c", "f#", "gb" using DEFAULT_OCTAVE
        note_str = s.lower()

        # Parse note name and octave
        if len(note_str) < 1:
            raise ValueError(f"Invalid note name: {s}")

        # Get base note
        base_note = note_str[0]
        if base_note not in NOTE_NAME_TO_SEMITONE:
            raise ValueError(f"Invalid note name: {s}")

        semitone = NOTE_NAME_TO_SEMITONE[base_note]
        pos = 1

        # Check for sharp or flat
        if pos < len(note_str):
            if note_str[pos] == "#":
                semitone += 1
                pos += 1
            elif note_str[pos] == "b":
                semitone -= 1
                pos += 1

        # Get octave (use DEFAULT_OCTAVE if not specified)
        if pos < len(note_str):
            octave_str = note_str[pos:]
            # Check if remaining string is a valid number (possibly negative)
            if octave_str.lstrip("-").isdigit():
                octave = int(octave_str)
            else:
                raise ValueError(f"Invalid octave in note: {s}")
        else:
            octave = DEFAULT_OCTAVE

        # Calculate MIDI note number (C0 = 0, C4 = 48, C5 = 60)
        # MIDI note = octave * 12 + semitone
        midi_note = octave * 12 + semitone

        return NoteField.mk(midi_note)

    @override
    def backward(self, value: Note) -> str:
        # Convert MIDI note back to note name
        note_num = int(value)
        octave = note_num // 12
        semitone = note_num % 12

        note_name = SEMITONE_TO_NOTE_NAME[semitone] + str(octave)

        return note_name


class VelocityElemParser(Iso[str, Velocity], Singleton):
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
    def forward(self, value: str) -> Velocity:
        return VelocityField.mk(int(value))

    @override
    def backward(self, value: Velocity) -> str:
        return str(value)


class ChannelElemParser(Iso[str, Channel], Singleton):
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
    def forward(self, value: str) -> Channel:
        return ChannelField.mk(int(value))

    @override
    def backward(self, value: Channel) -> str:
        return str(value)


class ProgramElemParser(Iso[str, Program], Singleton):
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
    def forward(self, value: str) -> Program:
        return ProgramField.mk(int(value))

    @override
    def backward(self, value: Program) -> str:
        return str(value)


class ControlNumElemParser(Iso[str, ControlNum], Singleton):
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
    def forward(self, value: str) -> ControlNum:
        return ControlField.mk(int(value))

    @override
    def backward(self, value: ControlNum) -> str:
        return str(value)


class ControlValElemParser(Iso[str, ControlVal], Singleton):
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
    def forward(self, value: str) -> ControlVal:
        return ValueField.mk(int(value))

    @override
    def backward(self, value: ControlVal) -> str:
        return str(value)


class ElemBinder(PatBinder[str, MidiAttrs]):
    def __init__[V, X](self, iso: Iso[str, V], field: MessageField[V, X]) -> None:
        self._iso = iso
        self._field = field

    @override
    def apply(self, value: str) -> Pat[MidiAttrs]:
        parsed = self._iso.forward(value)
        key = self._field.key()
        assert key is not None
        attrs = DMap.singleton(key, parsed)
        return Pat.pure(attrs)


# =============================================================================
# Stream Conversion Helpers
# =============================================================================


def convert_to_int_stream(input_val: IntStreamLike) -> Stream[int]:
    """Convert various input types to a Stream[int]."""
    if isinstance(input_val, str):
        string_pat = parse_pattern(input_val)
        return Stream.pat_bind(string_pat, IntBinder())
    elif isinstance(input_val, Pat):
        return Stream.pat(input_val)
    elif isinstance(input_val, Stream):
        return input_val
    else:
        raise ValueError(f"Unsupported type for IntStreamLike: {type(input_val)}")


def convert_to_string_stream(input_val: StringStreamLike) -> Stream[str]:
    """Convert various input types to a Stream[str]."""
    if isinstance(input_val, str):
        return Stream.pat(parse_pattern(input_val))
    elif isinstance(input_val, Pat):
        return Stream.pat(input_val)
    elif isinstance(input_val, Stream):
        return input_val
    else:
        raise ValueError(f"Unsupported type for StringStreamLike: {type(input_val)}")


# =============================================================================
# Pattern Stream Functions
# =============================================================================


_NOTE_NAME_BINDER = ElemBinder(NoteNameElemParser(), NoteField())
_NOTE_NUM_BINDER = ElemBinder(NoteNumElemParser(), NoteField())
_VELOCITY_BINDER = ElemBinder(VelocityElemParser(), VelocityField())
_CHANNEL_BINDER = ElemBinder(ChannelElemParser(), ChannelField())
_PROGRAM_BINDER = ElemBinder(ProgramElemParser(), ProgramField())
_CONTROL_NUM_BINDER = ElemBinder(ControlNumElemParser(), ControlField())
_CONTROL_VAL_BINDER = ElemBinder(ControlValElemParser(), ValueField())


def midinote_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from numeric MIDI notes.

    Parses a pattern containing numeric MIDI note values (0-127)
    and creates a stream of MIDI attributes.

    Args:
        input_val: Pattern string or stream containing
                  numeric MIDI note values (0-127)

    Examples:
        midinote_stream("60 62 64")     # C4, D4, E4 (C major triad)
        midinote_stream("36 ~ 42")      # Kick, rest, snare pattern
        midinote_stream("[60,64,67]")   # C major chord (simultaneous)
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _NOTE_NUM_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with note number binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(note_num: int) -> Stream[MidiAttrs]:
            return Stream.pat(_NOTE_NUM_BINDER.apply(str(note_num)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for midinote_stream: {type(input_val)}")


def note_stream(input_val: StringStreamLike) -> Stream[MidiAttrs]:
    """Create stream from note names.

    Parses a pattern containing musical note names with octaves
    and creates a stream of MIDI attributes.

    Args:
        input_val: Pattern string or stream containing
                  musical note names with octaves

    Examples:
        note_stream("c4 d4 e4")         # C major scale fragment
        note_stream("c4 ~ g4")          # C4, rest, G4
        note_stream("[c4,e4,g4]")       # C major chord (simultaneous)
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _NOTE_NAME_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to string stream and bind with note name binder
        string_stream = convert_to_string_stream(input_val)

        # Use the same approach as note_stream: bind pattern with binder
        def convert_to_midi_attrs(note_name: str) -> Stream[MidiAttrs]:
            return Stream.pat(_NOTE_NAME_BINDER.apply(note_name))

        return string_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for note_stream: {type(input_val)}")


def sound_stream(kit: Kit, input_val: StringStreamLike) -> Stream[MidiAttrs]:
    """Create stream from drum kit sound identifiers using a specific kit.

    Parses a pattern containing drum sound identifiers (like "bd", "sd", "hh")
    and creates a stream of MIDI attributes using the provided drum kit mapping.

    Args:
        kit: The DrumKit instance to use for sound mapping
        input_val: Pattern string or stream containing drum sound identifiers

    Returns:
        A Stream containing MIDI attributes for drum sounds

    Examples:
        sound_stream(kit, "bd sd bd sd")       # Bass drum, snare, bass drum, snare
        sound_stream(kit, "bd ~ sd ~")         # Bass drum, rest, snare, rest
        sound_stream(kit, "[bd,sd,hh]")        # Bass drum + snare + hi-hat (simultaneous)
        sound_stream(kit, "hh*8")              # Hi-hat repeated 8 times
        sound_stream(kit, "bd sd:2 hh:3")      # Different speeds for each element
    """
    if isinstance(input_val, str):
        drum_binder = ElemBinder(DrumSoundElemParser(kit), NoteField())
        return Stream.pat_bind(parse_pattern(input_val), drum_binder)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to string stream and bind with drum binder
        string_stream = convert_to_string_stream(input_val)
        drum_binder = ElemBinder(DrumSoundElemParser(kit), NoteField())

        # Convert each string to MIDI attributes
        def convert_to_midi_attrs(sound_name: str) -> Stream[MidiAttrs]:
            return Stream.pat(drum_binder.apply(sound_name))

        return string_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for sound_stream: {type(input_val)}")


def velocity_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from velocity values.

    Parses a pattern containing MIDI velocity values (0-127)
    and creates a stream of MIDI attributes for controlling note dynamics.

    Args:
        input_val: Pattern string or stream containing
                  MIDI velocity values (0-127)

    Examples:
        velocity_stream("64 80 100")         # Medium, loud, very loud
        velocity_stream("127 0 64")          # Loud, silent, medium
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _VELOCITY_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with velocity binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(velocity_val: int) -> Stream[MidiAttrs]:
            return Stream.pat(_VELOCITY_BINDER.apply(str(velocity_val)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for velocity_stream: {type(input_val)}")


def channel_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from channel values.

    Parses a pattern containing MIDI channel values (0-15)
    and creates a stream of MIDI attributes for specifying channels.
    If not specified, the orbit number will be used as the channel.

    Args:
        input_val: Pattern string or stream containing
                  MIDI channel values (0-15)

    Examples:
        channel_stream("0 1 9")          # Channels 1, 2, 10 (drums)
        channel_stream("15 ~ 0")         # Channel 16, rest, Channel 1
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _CHANNEL_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with channel binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(channel_val: int) -> Stream[MidiAttrs]:
            return Stream.pat(_CHANNEL_BINDER.apply(str(channel_val)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for channel_stream: {type(input_val)}")


def program_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from program values.

    Parses a pattern containing MIDI program values (0-127)
    and creates a stream of MIDI attributes for program change messages.

    Args:
        input_val: Pattern string or stream containing
                  MIDI program values (0-127)

    Examples:
        program_stream("0 1 40")         # Piano, Bright Piano, Violin
        program_stream("128 ~ 0")        # Invalid program, rest, Piano (will error on 128)
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _PROGRAM_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with program binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(program_val: int) -> Stream[MidiAttrs]:
            return Stream.pat(_PROGRAM_BINDER.apply(str(program_val)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for program_stream: {type(input_val)}")


def control_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from control number values.

    Parses a pattern containing MIDI control numbers (0-127)
    and creates a stream of MIDI attributes for control change messages.
    Note: Control change messages also require a control value.

    Args:
        input_val: Pattern string or stream containing
                  MIDI control numbers (0-127)

    Examples:
        control_stream("1 7 10")         # Modulation, Volume, Pan
        control_stream("64 ~ 1")         # Sustain, rest, Modulation
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _CONTROL_NUM_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with control binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(control_val: int) -> Stream[MidiAttrs]:
            return Stream.pat(_CONTROL_NUM_BINDER.apply(str(control_val)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for control_stream: {type(input_val)}")


def value_stream(input_val: IntStreamLike) -> Stream[MidiAttrs]:
    """Create stream from control value values.

    Parses a pattern containing MIDI control values (0-127)
    and creates a stream of MIDI attributes for control change messages.
    Note: Control change messages also require a control number.

    Args:
        input_val: Pattern string or stream containing
                  MIDI control values (0-127)

    Examples:
        value_stream("0 64 127")         # Min, center, max values
        value_stream("127 ~ 0")          # Max, rest, min
    """
    if isinstance(input_val, str):
        return Stream.pat_bind(parse_pattern(input_val), _CONTROL_VAL_BINDER)
    elif isinstance(input_val, (Pat, Stream)):
        # For pattern/stream inputs, convert to int stream and bind with value binder
        int_stream = convert_to_int_stream(input_val)

        # Convert each int to a string for the binder
        def convert_to_midi_attrs(value_val: int) -> Stream[MidiAttrs]:
            return Stream.pat(_CONTROL_VAL_BINDER.apply(str(value_val)))

        return int_stream.bind(MergeStrat.Outer, convert_to_midi_attrs)
    else:
        raise ValueError(f"Unsupported type for value_stream: {type(input_val)}")


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
