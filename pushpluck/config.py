"""Configuration module for pushpluck.

This module defines the core configuration classes, enums, and utilities for managing
the visual and behavioral aspects of the pushpluck musical instrument interface.
It handles color schemes, layout configurations, play modes, and pad color mapping
for different types of interface elements.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Dict, List, Optional

from pushpluck import constants
from pushpluck.base import MatchException
from pushpluck.color import COLORS, Color
from pushpluck.scale import SCALE_LOOKUP, NoteName, Scale


@unique
class NoteType(Enum):
    """Represents the type of a musical note in relation to the current scale.

    Used to categorize notes for visual representation and color mapping.
    """

    Root = auto()  # Root note of the current scale
    Member = auto()  # Note that is a member of the current scale
    Other = auto()  # Note that is not in the current scale


@dataclass(frozen=True)
class ColorScheme:
    """Defines the color palette used throughout the interface.

    This class contains all the colors used for different types of pads
    and interface elements, providing a consistent visual theme.
    """

    root_note: Color  # Color for root notes of the scale
    member_note: Color  # Color for scale member notes
    other_note: Color  # Color for notes outside the scale
    primary_note: Color  # Color for the currently selected/active note
    disabled_note: Color  # Color for disabled or inactive notes
    linked_note: Color  # Color for notes linked to the primary note
    misc_pressed: Color  # Color for misc pads when pressed
    control: Color  # Default color for control pads
    control_pressed: Color  # Color for control pads when pressed


@unique
class VisState(Enum):
    """Represents the visual state of a pad or interface element.

    This enum tracks whether a pad is off, active in different modes,
    or in special states like disabled or linked.
    """

    Off = auto()  # Pad is inactive/off
    OnPrimary = auto()  # Pad is the primary active element
    OnDisabled = auto()  # Pad is active but disabled
    OnLinked = auto()  # Pad is active and linked to primary

    @property
    def primary(self) -> bool:
        """Check if this state represents the primary active state.

        Returns:
            True if this is the primary active state, False otherwise.
        """
        return self == VisState.OnPrimary

    @property
    def active(self) -> bool:
        """Check if this state represents any active state.

        Returns:
            True if the pad is active in any way, False if off.
        """
        return self != VisState.Off

    @property
    def enabled(self) -> bool:
        """Check if this state represents an enabled state.

        Returns:
            True if the pad is enabled, False if disabled.
        """
        return self != VisState.OnDisabled


class PadColorMapper(metaclass=ABCMeta):
    """Abstract base class for mapping pad visual states to colors.

    This class defines the interface for determining what color a pad should
    display based on its visual state and the current color scheme. Different
    types of pads (note, misc, control) have different color mapping logic.
    """

    @abstractmethod
    def get_color(self, scheme: ColorScheme, vis: VisState) -> Optional[Color]:
        """Get the appropriate color for a pad given its state.

        Args:
            scheme: The current color scheme to use.
            vis: The visual state of the pad.

        Returns:
            The color to display for this pad, or None if no color should be shown.
        """
        raise NotImplementedError()

    @staticmethod
    def note(note_type: NoteType) -> NotePadColorMapper:
        """Create a color mapper for note pads.

        Args:
            note_type: The type of note (root, member, or other).

        Returns:
            A NotePadColorMapper instance for the specified note type.
        """
        return NotePadColorMapper(note_type)

    @staticmethod
    def misc(pressable: bool) -> MiscPadColorMapper:
        """Create a color mapper for miscellaneous pads.

        Args:
            pressable: Whether this misc pad can be pressed/activated.

        Returns:
            A MiscPadColorMapper instance with the specified pressable behavior.
        """
        return MiscPadColorMapper(pressable)

    @staticmethod
    def control() -> ControlPadColorMapper:
        """Create a color mapper for control pads.

        Returns:
            A ControlPadColorMapper instance.
        """
        return ControlPadColorMapper()


@dataclass(frozen=True)
class NotePadColorMapper(PadColorMapper):
    """Color mapper for note pads.

    Maps note pad visual states to colors based on the note type and
    current visual state. Handles special states like primary, disabled,
    and linked, falling back to note-type-specific colors otherwise.
    """

    note_type: NoteType  # The type of note this mapper handles

    def get_color(self, scheme: ColorScheme, vis: VisState) -> Optional[Color]:
        """Get the color for a note pad based on its state.

        Priority order:
        1. Special states (primary, disabled, linked) override note type
        2. Fall back to note-type-specific colors for normal state

        Args:
            scheme: The color scheme to use.
            vis: The visual state of the note pad.

        Returns:
            The appropriate color for this note pad.

        Raises:
            MatchException: If an unknown note type is encountered.
        """
        if vis == VisState.OnPrimary:
            return scheme.primary_note
        elif vis == VisState.OnDisabled:
            return scheme.disabled_note
        elif vis == VisState.OnLinked:
            return scheme.linked_note
        else:
            if self.note_type == NoteType.Root:
                return scheme.root_note
            elif self.note_type == NoteType.Member:
                return scheme.member_note
            elif self.note_type == NoteType.Other:
                return scheme.other_note
            else:
                raise MatchException(self.note_type)


@dataclass(frozen=True)
class MiscPadColorMapper(PadColorMapper):
    """Color mapper for miscellaneous interface pads.

    These are pads that aren't note pads or control pads, such as
    navigation elements or decorative elements. They may or may not
    be pressable/interactive.
    """

    pressable: bool  # Whether this misc pad can be pressed/activated

    def get_color(self, scheme: ColorScheme, vis: VisState) -> Optional[Color]:
        """Get the color for a misc pad based on its state.

        Misc pads only show color when they are both active and pressable.
        Non-pressable misc pads never show color.

        Args:
            scheme: The color scheme to use.
            vis: The visual state of the misc pad.

        Returns:
            The misc pressed color if active and pressable, None otherwise.
        """
        return scheme.misc_pressed if vis.active and self.pressable else None


@dataclass(frozen=True)
class ControlPadColorMapper(PadColorMapper):
    """Color mapper for control pads.

    Control pads are always visible and change color based on whether
    they are currently active/pressed. They handle functions like
    mode switching, configuration, etc.
    """

    def get_color(self, scheme: ColorScheme, vis: VisState) -> Optional[Color]:
        """Get the color for a control pad based on its state.

        Control pads always show color - either the pressed color when
        active or the default control color when inactive.

        Args:
            scheme: The color scheme to use.
            vis: The visual state of the control pad.

        Returns:
            The control pressed color if active, otherwise the default control color.
        """
        return scheme.control_pressed if vis.active else scheme.control


# Dihedral group D_4 multiplication table
# This table defines the composition of layout transformations
_LAYOUT_MULT_TABLE: Dict["Layout", Dict["Layout", "Layout"]] = {}

# Precomputed arrow offset table: (Layout, Arrow) -> (sem_off_delta, str_off_delta)
_ARROW_OFFSET_TABLE: Dict[tuple["Layout", "Arrow"], tuple[int, int]] = {}


@unique
class Arrow(Enum):
    """Arrow directions for navigation."""

    Up = auto()
    Down = auto()
    Left = auto()
    Right = auto()

    @property
    def direction(self) -> tuple[int, int]:
        """Get the arrow direction as (row_delta, col_delta) with viewport semantics.

        Returns:
            Tuple representing the arrow direction.
        """
        if self == Arrow.Up:
            return (1, 0)  # Viewport: increase row
        elif self == Arrow.Down:
            return (-1, 0)  # Viewport: decrease row
        elif self == Arrow.Left:
            return (0, 1)  # Viewport: increase column
        elif self == Arrow.Right:
            return (0, -1)  # Viewport: decrease column
        else:
            raise MatchException(self)


@unique
class Layout(Enum):
    """Defines layout transformations as elements of the dihedral group D_4.

    The dihedral group D_4 represents the 8 symmetries of a square:
    4 rotations and 4 reflections. These transformations are applied
    to the coordinate mapping between pad positions and fretboard coordinates.
    """

    Identity = auto()  # No transformation (0° rotation)
    Rot90 = auto()  # 90° clockwise rotation
    Rot180 = auto()  # 180° rotation
    Rot270 = auto()  # 270° clockwise rotation (90° counter-clockwise)
    FlipH = auto()  # Horizontal flip (reflect across horizontal axis)
    FlipV = auto()  # Vertical flip (reflect across vertical axis)
    FlipD = auto()  # Diagonal flip (reflect across main diagonal)
    FlipAD = auto()  # Anti-diagonal flip (reflect across anti-diagonal)

    @property
    def display_name(self) -> str:
        """Get user-friendly display name for the layout.

        Returns:
            Human-readable name for menu display.
        """
        names = {
            Layout.Identity: "Normal",
            Layout.Rot90: "Rot90",
            Layout.Rot180: "Rot180",
            Layout.Rot270: "Rot270",
            Layout.FlipH: "FlipH",
            Layout.FlipV: "FlipV",
            Layout.FlipD: "FlipD",
            Layout.FlipAD: "FlipAD",
        }
        return names[self]

    def __mul__(self, other: "Layout") -> "Layout":
        """Multiply two layout transformations (compose them).

        The multiplication represents composition of transformations:
        (a * b) means "apply b first, then apply a".

        Args:
            other: The transformation to compose with this one.

        Returns:
            The composed transformation.
        """
        return _LAYOUT_MULT_TABLE[self][other]

    def apply_to_coords(
        self, row: int, col: int, max_row: int = 7, max_col: int = 7
    ) -> tuple[int, int]:
        """Apply this layout transformation to coordinates.

        Args:
            row: The row coordinate (0-based).
            col: The column coordinate (0-based).
            max_row: Maximum row index (default 7 for 8x8 grid).
            max_col: Maximum column index (default 7 for 8x8 grid).

        Returns:
            Tuple of (new_row, new_col) after transformation.
        """
        if self == Layout.Identity:
            return (row, col)
        elif self == Layout.Rot90:
            return (col, max_row - row)
        elif self == Layout.Rot180:
            return (max_row - row, max_col - col)
        elif self == Layout.Rot270:
            return (max_col - col, row)
        elif self == Layout.FlipH:
            return (max_row - row, col)
        elif self == Layout.FlipV:
            return (row, max_col - col)
        elif self == Layout.FlipD:
            return (col, row)
        elif self == Layout.FlipAD:
            return (max_col - col, max_row - row)
        else:
            raise MatchException(self)

    def inverse(self) -> "Layout":
        """Get the inverse transformation.

        Returns:
            The layout transformation that undoes this one.
        """
        # For the dihedral group, compute the inverse
        if self == Layout.Identity:
            return Layout.Identity
        elif self == Layout.Rot90:
            return Layout.Rot270
        elif self == Layout.Rot180:
            return Layout.Rot180
        elif self == Layout.Rot270:
            return Layout.Rot90
        elif self == Layout.FlipH:
            return Layout.FlipH  # Reflections are self-inverse
        elif self == Layout.FlipV:
            return Layout.FlipV
        elif self == Layout.FlipD:
            return Layout.FlipD
        elif self == Layout.FlipAD:
            return Layout.FlipAD
        else:
            raise MatchException(self)

    def arrow_to_offset_deltas(self, arrow: "Arrow") -> tuple[int, int]:
        """Map an arrow to semitone and string offset deltas using precomputed table.

        Args:
            arrow: The arrow direction.

        Returns:
            Tuple of (sem_off_delta, str_off_delta) for the given arrow.
        """
        return _ARROW_OFFSET_TABLE[(self, arrow)]


def _init_layout_mult_table() -> None:
    """Initialize the dihedral group D_4 multiplication table.

    This implements the group operation for the 8 symmetries of a square.
    The multiplication represents composition: (a * b) means apply b first, then a.
    """
    # Import Layout enum values for convenience
    Id, R1, R2, R3 = Layout.Identity, Layout.Rot90, Layout.Rot180, Layout.Rot270
    H, V, D, AD = Layout.FlipH, Layout.FlipV, Layout.FlipD, Layout.FlipAD

    # Initialize the multiplication table
    # Each row represents the left operand, each column the right operand
    _LAYOUT_MULT_TABLE.update(
        {
            Id: {Id: Id, R1: R1, R2: R2, R3: R3, H: H, V: V, D: D, AD: AD},
            R1: {Id: R1, R1: R2, R2: R3, R3: Id, H: D, V: AD, D: V, AD: H},
            R2: {Id: R2, R1: R3, R2: Id, R3: R1, H: V, V: H, D: AD, AD: D},
            R3: {Id: R3, R1: Id, R2: R1, R3: R2, H: AD, V: D, D: H, AD: V},
            H: {Id: H, R1: AD, R2: V, R3: D, H: Id, V: R2, D: R3, AD: R1},
            V: {Id: V, R1: D, R2: H, R3: AD, H: R2, V: Id, D: R1, AD: R3},
            D: {Id: D, R1: H, R2: AD, R3: V, H: R1, V: R3, D: Id, AD: R2},
            AD: {Id: AD, R1: V, R2: D, R3: H, H: R3, V: R1, D: R2, AD: Id},
        }
    )


def _init_arrow_offset_table() -> None:
    """Initialize the precomputed arrow offset table.

    Creates a lookup table mapping (Layout, Arrow) to (sem_off_delta, str_off_delta).
    This provides fast, explicit arrow behavior for all layout combinations.
    """
    for layout in Layout:
        for arrow in Arrow:
            # Get arrow direction with viewport semantics
            row_delta, col_delta = arrow.direction

            # Apply layout transformation to the arrow direction
            logical_row, logical_col = layout.apply_to_coords(
                row_delta, col_delta, max_row=0, max_col=0
            )

            # Convert logical coordinates to offset deltas
            sem_off_delta = 0
            str_off_delta = 0

            if logical_col > 0:
                sem_off_delta = 1
            elif logical_col < 0:
                sem_off_delta = -1

            if logical_row > 0:
                str_off_delta = -1  # Positive row = higher string number = lower pitch
            elif logical_row < 0:
                str_off_delta = 1  # Negative row = lower string number = higher pitch

            # Special case for FlipD layout to match expected Harpejji behavior
            if layout == Layout.FlipD:
                # Flip the deltas for FlipD layout
                sem_off_delta = -sem_off_delta
                str_off_delta = -str_off_delta

            _ARROW_OFFSET_TABLE[(layout, arrow)] = (sem_off_delta, str_off_delta)


# Initialize the tables when the module is loaded
_init_layout_mult_table()
_init_arrow_offset_table()


@unique
class PlayMode(Enum):
    """Defines how notes are played and sustained.

    Controls the behavior of note triggering and sustain when pads are
    pressed and released.
    """

    Tap = auto()  # Notes are triggered on press, no sustain
    # Pick = auto()  # Reserved for future picking simulation mode
    Poly = auto()  # Multiple notes can be sustained simultaneously
    Mono = auto()  # Only one note can be sustained at a time


@unique
class ChannelMode(Enum):
    """Defines MIDI channel usage for note output.

    Determines whether all notes are sent on a single MIDI channel
    or distributed across multiple channels.
    """

    Single = auto()  # All notes sent on one MIDI channel
    Multi = auto()  # Notes distributed across multiple MIDI channels


@unique
class Instrument(Enum):
    """Defines different instrument configurations available."""

    Guitar = auto()  # Guitar with standard horizontal layout
    Harpejji = auto()  # Harpejji with vertical chromatic layout
    Chromatic = auto()  # Chromatic layout with half-step between strings
    Fourths = auto()  # Fourths tuning, mimics Push's default pad layout


# TODO This needs to be hierarchical
# Each instrument has a default orientation and tuning
# As well as multiple possible tunings
# @dataclass(frozen=True)
# class Profile:
#     instrument_name: str
#     tuning_name: str
#     tuning: List[int]
#     orientation: Orientation


@dataclass(frozen=True)
class Config:
    """Main configuration class containing all instrument and interface settings.

    This class holds the complete configuration state for the pushpluck instrument,
    including tuning, visual layout, play behavior, scale information, and
    display offsets.
    """

    instrument: Instrument  # The instrument type (Guitar, Harpejji, etc.)
    instrument_name: str  # Name of the instrument (e.g., "Guitar", "Bass")
    tuning_name: str  # Name of the tuning (e.g., "Standard", "Drop D")
    tuning: List[int]  # MIDI note numbers for each string's open note
    pre_layout: Layout  # Instrument-specific layout transformation applied first
    layout: Layout  # User-selectable layout transformation (defaults to Identity)
    play_mode: PlayMode  # How notes are triggered and sustained
    chan_mode: ChannelMode  # MIDI channel usage strategy
    midi_channel: int  # Base MIDI channel (1-16)
    repeat_steps: int  # Semitone range of tuning pattern for infinite strings
    view_offset: int  # String offset for centering/positioning (0 = no-op)
    scale: Scale  # Current musical scale for note classification
    root: NoteName  # Root note of the current scale
    min_velocity: int  # Minimum MIDI velocity for note output
    max_velocity: int  # Maximum MIDI velocity for note output
    output_port: str  # Name of the MIDI output port for processed notes
    str_offset: int  # String display offset for scrolling
    fret_offset: int  # Fret display offset for scrolling

    @property
    def effective_layout(self) -> Layout:
        """Get the effective layout by composing pre_layout and layout.

        Returns:
            The result of pre_layout * layout (pre_layout applied first, then layout).
        """
        return self.pre_layout * self.layout


def init_config(
    min_velocity: int, max_velocity: int = 127, output_port: str = "pushpluck"
) -> Config:
    """Initialize a default configuration with Guitar settings.

    Creates a Config instance with sensible defaults for a Guitar
    setup: standard tuning, rotated layout, tap mode, single channel,
    C major scale, and appropriate display offsets.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.
        max_velocity: The maximum MIDI velocity to use for note output.
        output_port: The name of the MIDI output port for processed notes.

    Returns:
        A Config instance with default settings and the specified parameters.
    """
    return Config(
        instrument=Instrument.Guitar,
        instrument_name="Guitar",
        tuning_name="Standard",
        tuning=constants.STANDARD_TUNING,
        pre_layout=Layout.Rot90,  # Guitar: rotated 90 degrees clockwise
        layout=Layout.Identity,  # User-selectable layout starts as Identity
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=24,  # Guitar: high E - low E interval (2 octaves)
        view_offset=1,  # Guitar: center the 5-string pattern (access to low B and high A)
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        output_port=output_port,
        str_offset=0,
        fret_offset=0,
    )


def init_guitar_config(
    min_velocity: int, max_velocity: int = 127, output_port: str = "pushpluck"
) -> Config:
    """Initialize a configuration with Guitar settings.

    Creates a Config instance with defaults for a Guitar setup:
    standard tuning, horizontal layout, tap mode, single channel,
    C major scale, and no display offsets.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.

    Returns:
        A Config instance with Guitar settings and the specified min_velocity.
    """
    return Config(
        instrument=Instrument.Guitar,
        instrument_name="Guitar",
        tuning_name="Standard",
        tuning=constants.STANDARD_TUNING,
        pre_layout=Layout.Rot90,  # Guitar: rotated 90 degrees clockwise
        layout=Layout.Identity,  # User-selectable layout starts as Identity
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=24,  # Guitar: high E - low E interval (2 octaves)
        view_offset=1,  # Guitar: center the 5-string pattern (access to low B and high A)
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        output_port=output_port,
        str_offset=0,
        fret_offset=0,
    )


def init_harpejji_config(
    min_velocity: int, max_velocity: int = 127, output_port: str = "pushpluck"
) -> Config:
    """Initialize a configuration with Harpejji settings.

    Creates a Config instance with defaults for a Harpejji setup:
    whole-step tuning starting from C3, vertical layout, tap mode,
    single channel, C major scale, and no display offsets.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.

    Returns:
        A Config instance with Harpejji settings and the specified min_velocity.
    """
    return Config(
        instrument=Instrument.Harpejji,
        instrument_name="Harpejji",
        tuning_name="Chromatic",
        tuning=constants.HARPEJJI_TUNING,
        pre_layout=Layout.FlipD,  # Harpejji: diagonal flip layout by default
        layout=Layout.Identity,  # User-selectable layout starts as Identity
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=2,  # Harpejji: whole steps between strings
        view_offset=0,  # Harpejji: start from string 0
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        output_port=output_port,
        str_offset=0,
        fret_offset=0,
    )


def init_chromatic_config(
    min_velocity: int, max_velocity: int = 127, output_port: str = "pushpluck"
) -> Config:
    """Initialize a configuration with Chromatic settings.

    Creates a Config instance with defaults for a Chromatic setup:
    half-step tuning (chromatic scale), horizontal layout, tap mode,
    single channel, C major scale, and appropriate display offsets.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.
        max_velocity: The maximum MIDI velocity to use for note output.
        output_port: The name of the MIDI output port for processed notes.

    Returns:
        A Config instance with Chromatic settings and the specified parameters.
    """
    return Config(
        instrument=Instrument.Chromatic,
        instrument_name="Chromatic",
        tuning_name="Chromatic",
        tuning=[60],  # Start from C4 (middle C)
        pre_layout=Layout.Identity,  # Chromatic: horizontal layout
        layout=Layout.Identity,  # User-selectable layout starts as Identity
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=3,
        repeat_steps=1,  # Chromatic: half-step between strings
        view_offset=0,  # Start from string 0
        scale=SCALE_LOOKUP["Chromatic"],
        root=NoteName.C,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        output_port=output_port,
        str_offset=0,
        fret_offset=0,
    )


def init_fourths_config(
    min_velocity: int, max_velocity: int = 127, output_port: str = "pushpluck"
) -> Config:
    """Initialize a configuration with Fourths tuning settings.

    Creates a Config instance with defaults for a Fourths setup that
    mimics the Push's default pad layout: perfect fourth intervals,
    horizontal layout, tap mode, single channel, C major scale.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.
        max_velocity: The maximum MIDI velocity to use for note output.
        output_port: The name of the MIDI output port for processed notes.

    Returns:
        A Config instance with Fourths settings and the specified parameters.
    """
    return Config(
        instrument=Instrument.Fourths,
        instrument_name="Fourths",
        tuning_name="Fourths",
        tuning=[60],  # Start from C4 (middle C)
        pre_layout=Layout.Identity,  # Fourths: horizontal layout
        layout=Layout.Identity,  # User-selectable layout starts as Identity
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=4,
        repeat_steps=5,  # Fourths: perfect fourth (5 semitones) between strings
        view_offset=0,  # Start from string 0
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        output_port=output_port,
        str_offset=0,
        fret_offset=0,
    )


def get_config_for_instrument(
    instrument: Instrument,
    min_velocity: int,
    max_velocity: int = 127,
    output_port: str = "pushpluck",
) -> Config:
    """Get the appropriate configuration for a given instrument.

    Args:
        instrument: The instrument type to configure.
        min_velocity: The minimum MIDI velocity to use for note output.

    Returns:
        A Config instance with appropriate settings for the instrument.
    """
    if instrument == Instrument.Guitar:
        return init_guitar_config(min_velocity, max_velocity, output_port)
    elif instrument == Instrument.Harpejji:
        return init_harpejji_config(min_velocity, max_velocity, output_port)
    elif instrument == Instrument.Chromatic:
        return init_chromatic_config(min_velocity, max_velocity, output_port)
    elif instrument == Instrument.Fourths:
        return init_fourths_config(min_velocity, max_velocity, output_port)
    else:
        raise ValueError(f"Unknown instrument: {instrument}")


def default_scheme() -> ColorScheme:
    """Create a default color scheme with predefined colors.

    Returns a ColorScheme with a standard color palette that provides
    good visual contrast and intuitive color associations (e.g., blue
    for root notes, green for primary/active, red for disabled).

    Returns:
        A ColorScheme instance with the default color assignments.
    """
    return ColorScheme(
        root_note=COLORS["Blue"],
        member_note=COLORS["White"],
        other_note=COLORS["Black"],
        primary_note=COLORS["Green"],
        disabled_note=COLORS["Orange"],
        linked_note=COLORS["Red"],
        misc_pressed=COLORS["Sky"],
        control=COLORS["Yellow"],
        control_pressed=COLORS["Pink"],
    )
