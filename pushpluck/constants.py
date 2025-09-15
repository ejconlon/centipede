"""Constants and enumerations for the Ableton Push interface.

This module defines MIDI control constants, button mappings, and hardware
specifications for the Ableton Push controller used in the PushPluck application.
"""

from enum import Enum, auto, unique
from typing import Dict, Tuple, Type, TypeVar

E = TypeVar("E", bound=Enum)
"""Type variable for enum types."""


def make_enum_value_lookup(enum_type: Type[E]) -> Dict[int, E]:
    """Create a reverse lookup dictionary from enum values to enum instances.

    Args:
        enum_type: The enum class to create a lookup for.

    Returns:
        A dictionary mapping enum values to enum instances.
    """
    lookup: Dict[int, E] = {}
    for enum_val in enum_type.__members__.values():
        lookup[enum_val.value] = enum_val
    return lookup


DEFAULT_PUSH_DELAY = 0.0008
"""Default delay between MIDI output messages to prevent flooding the Push (seconds)."""


@unique
class ButtonIllum(Enum):
    """Button illumination states for the Ableton Push."""

    Half = 1
    """Half brightness illumination."""
    # HalfBlinkSlow = 2
    # HalfBlinkFast = 3
    Full = 4
    """Full brightness illumination."""
    # FullBlinkSlow = 5
    # FullBlinkFast = 6
    Off = 0
    """Button illumination off."""
    # TODO Use full?
    # On = 127


@unique
class ButtonColor(Enum):
    """Available colors for Push buttons."""

    Orange = 7
    """Orange button color."""
    Red = 1
    """Red button color."""
    Green = 19
    """Green button color."""
    Yellow = 13
    """Yellow button color."""


@unique
class KnobGroup(Enum):
    """Knob groups on the Push controller."""

    Left = auto()
    """Left group of knobs."""
    Center = auto()
    """Center group of knobs."""
    Right = auto()
    """Right group of knobs."""


@unique
class KnobCC(Enum):
    """MIDI Control Change numbers for Push knobs."""

    L0 = 14
    """Left knob 0."""
    L1 = 15
    """Left knob 1."""
    C0 = 71
    """Center knob 0."""
    C1 = 72
    """Center knob 1."""
    C2 = 73
    """Center knob 2."""
    C3 = 74
    """Center knob 3."""
    C4 = 75
    """Center knob 4."""
    C5 = 76
    """Center knob 5."""
    C6 = 77
    """Center knob 6."""
    C7 = 78
    """Center knob 7."""
    R0 = 79
    """Right knob 0."""


KNOB_CC_VALUE_LOOKUP: Dict[int, KnobCC] = make_enum_value_lookup(KnobCC)
"""Reverse lookup from CC value to KnobCC enum."""


def knob_group_and_offset(knob: KnobCC) -> Tuple[KnobGroup, int]:
    """Determine the group and offset for a given knob.

    Args:
        knob: The knob to analyze.

    Returns:
        A tuple of (group, offset within group).
    """
    if knob.value >= KnobCC.R0.value:
        return KnobGroup.Right, knob.value - KnobCC.R0.value
    elif knob.value >= KnobCC.C0.value:
        return KnobGroup.Center, knob.value - KnobCC.C0.value
    else:
        return KnobGroup.Left, knob.value - KnobCC.L0.value


@unique
class ButtonCC(Enum):
    """MIDI Control Change numbers for Push buttons."""

    TapTempo = 3
    """Tap tempo button for setting BPM."""
    Metronome = 9
    """Metronome on/off button."""
    Undo = 119
    """Undo last action button."""
    Delete = 118
    """Delete selected content button."""
    Double = 117
    """Double/loop selected content button."""
    Quantize = 116
    """Quantize timing button."""
    FixedLength = 90
    """Fixed length recording mode button."""
    Automation = 89
    """Automation mode button."""
    Duplicate = 88
    """Duplicate selected content button."""
    New = 87
    """Create new clip/track button."""
    Rec = 86
    """Record button."""
    Play = 85
    """Play/pause button."""
    Master = 28
    """Master track selection button."""
    Stop = 29
    """Stop playback button."""
    Left = 44
    """Navigate left arrow button."""
    Right = 45
    """Navigate right arrow button."""
    Up = 46
    """Navigate up arrow button."""
    Down = 47
    """Navigate down arrow button."""
    Volume = 114
    """Volume control mode button."""
    PanAndSend = 115
    """Pan and send control mode button."""
    Track = 112
    """Track selection mode button."""
    Clip = 113
    """Clip mode button."""
    Device = 110
    """Device control mode button."""
    Browse = 111
    """Browse sounds/devices button."""
    StepIn = 62
    """Step into nested view button."""
    StepOut = 63
    """Step out of nested view button."""
    Mute = 60
    """Mute mode button."""
    Solo = 61
    """Solo mode button."""
    Scales = 58
    """Scale selection mode button."""
    User = 59
    """User mode button."""
    Repeat = 56
    """Note repeat mode button."""
    Accent = 57
    """Accent/velocity mode button."""
    OctaveDown = 54
    """Lower octave range button."""
    OctaveUp = 55
    """Raise octave range button."""
    AddEffect = 52
    """Add audio effect button."""
    AddTrack = 53
    """Add new track button."""
    Note = 50
    """Note/drum pad mode button."""
    Session = 51
    """Session view mode button."""
    Select = 48
    """Select button for confirming choices."""
    Shift = 49
    """Shift button for accessing secondary functions."""


BUTTON_CC_VALUE_LOOKUP: Dict[int, ButtonCC] = make_enum_value_lookup(ButtonCC)
"""Reverse lookup from CC value to ButtonCC enum."""


@unique
class TimeDivCC(Enum):
    """MIDI Control Change numbers for time division buttons."""

    TimeQuarter = 36
    """Quarter note time division (1/4)."""
    TimeQuarterTriplet = 37
    """Quarter note triplet time division (1/4T)."""
    TimeEighth = 38
    """Eighth note time division (1/8)."""
    TimeEighthTriplet = 39
    """Eighth note triplet time division (1/8T)."""
    TimeSixteenth = 40
    """Sixteenth note time division (1/16)."""
    TimeSixteenthTriplet = 41
    """Sixteenth note triplet time division (1/16T)."""
    TimeThirtysecond = 42
    """Thirty-second note time division (1/32)."""
    TimeThirtysecondTriplet = 43
    """Thirty-second note triplet time division (1/32T)."""


TIME_DIV_CC_VALUE_LOOKUP: Dict[int, TimeDivCC] = make_enum_value_lookup(TimeDivCC)
"""Reverse lookup from CC value to TimeDivCC enum."""

MIDI_BASE_CHANNEL = 1
"""Base MIDI channel number."""
MIDI_MIN_CHANNEL = 1
"""Minimum valid MIDI channel."""
MIDI_MAX_CHANNEL = 16
"""Maximum valid MIDI channel."""

DEFAULT_PUSH_PORT_NAME = "Ableton Push User Port"
"""Default MIDI port name for the Push controller."""
DEFAULT_PROCESSED_PORT_NAME = "pushpluck"
"""Default name for the processed MIDI output port."""
LOW_NOTE = 36
"""Lowest MIDI note number for the pad grid."""
NUM_PAD_ROWS = 8
"""Number of rows in the Push pad grid."""
NUM_PAD_COLS = 8
"""Number of columns in the Push pad grid."""
NUM_PADS = NUM_PAD_ROWS * NUM_PAD_COLS
"""Total number of pads in the grid."""
HIGH_NOTE = LOW_NOTE + NUM_PADS
"""Highest MIDI note number for the pad grid (exclusive)."""
PUSH_SYSEX_PREFIX = (71, 127, 21)
"""System Exclusive message prefix for Push communication."""
LOW_CHAN_CONTROL = 20
"""Lowest CC number for channel controls."""
HIGH_CHAN_CONTROL = LOW_CHAN_CONTROL + NUM_PAD_COLS
"""Highest CC number for channel controls (exclusive)."""
LOW_GRID_CONTROL = 102
"""Lowest CC number for grid controls."""
HIGH_GRID_CONTROL = LOW_GRID_CONTROL + NUM_PAD_COLS
"""Highest CC number for grid controls (exclusive)."""

DISPLAY_MAX_ROWS = 4
"""Maximum number of rows on the Push display."""
DISPLAY_MAX_BLOCKS = 4
"""Maximum number of blocks per display row."""
DISPLAY_BLOCK_LEN = 17
"""Character length of each display block."""
DISPLAY_HALF_BLOCK_LEN = DISPLAY_BLOCK_LEN // 2
"""Half the length of a display block."""
DISPLAY_MAX_LINE_LEN = DISPLAY_MAX_BLOCKS * DISPLAY_BLOCK_LEN
"""Maximum character length of a display line."""
DISPLAY_BUFFER_LEN = DISPLAY_MAX_ROWS * DISPLAY_MAX_LINE_LEN
"""Total character capacity of the display buffer."""

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
"""Standard guitar tuning in MIDI note numbers (E-A-D-G-B-E)."""

HARPEJJI_TUNING = [48, 50, 52, 54, 56, 58, 60, 62]
"""Harpejji tuning in MIDI note numbers starting from C3, whole steps (C-D-E-F#-G#-A#-C-D)."""
