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
from typing import List, Optional

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


@unique
class Layout(Enum):
    """Defines the layout orientation for the instrument interface.

    Determines how the strings and frets are oriented on the display,
    affecting the visual arrangement of the note grid.
    """

    Horiz = auto()  # Horizontal layout (strings horizontal, frets vertical)
    Vert = auto()  # Vertical layout (strings vertical, frets horizontal)


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
    layout: Layout  # Visual layout orientation (horizontal/vertical)
    play_mode: PlayMode  # How notes are triggered and sustained
    chan_mode: ChannelMode  # MIDI channel usage strategy
    midi_channel: int  # Base MIDI channel (1-16)
    repeat_steps: int  # Semitone range of tuning pattern for infinite strings
    view_offset: int  # String offset for centering/positioning (0 = no-op)
    scale: Scale  # Current musical scale for note classification
    root: NoteName  # Root note of the current scale
    min_velocity: int  # Minimum MIDI velocity for note output
    str_offset: int  # String display offset for scrolling
    fret_offset: int  # Fret display offset for scrolling


def init_config(min_velocity: int) -> Config:
    """Initialize a default configuration with Harpejji settings.

    Creates a Config instance with sensible defaults for a Harpejji
    setup: whole-step tuning, vertical layout, tap mode, single channel,
    C major scale, and no display offsets.

    Args:
        min_velocity: The minimum MIDI velocity to use for note output.

    Returns:
        A Config instance with default settings and the specified min_velocity.
    """
    return Config(
        instrument=Instrument.Harpejji,
        instrument_name="Harpejji",
        tuning_name="Chromatic",
        tuning=constants.HARPEJJI_TUNING,
        layout=Layout.Vert,
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=2,  # Harpejji: whole steps between strings
        view_offset=0,  # Harpejji: start from string 0
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        str_offset=0,
        fret_offset=0,
    )


def init_guitar_config(min_velocity: int) -> Config:
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
        layout=Layout.Horiz,
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=24,  # Guitar: high E - low E interval (2 octaves)
        view_offset=1,  # Guitar: center the 5-string pattern (access to low B and high A)
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        str_offset=0,
        fret_offset=0,
    )


def init_harpejji_config(min_velocity: int) -> Config:
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
        layout=Layout.Vert,
        play_mode=PlayMode.Tap,
        chan_mode=ChannelMode.Single,
        midi_channel=2,
        repeat_steps=2,  # Harpejji: whole steps between strings
        view_offset=0,  # Harpejji: start from string 0
        scale=SCALE_LOOKUP["Major"],
        root=NoteName.C,
        min_velocity=min_velocity,
        str_offset=0,
        fret_offset=0,
    )


def get_config_for_instrument(instrument: Instrument, min_velocity: int) -> Config:
    """Get the appropriate configuration for a given instrument.

    Args:
        instrument: The instrument type to configure.
        min_velocity: The minimum MIDI velocity to use for note output.

    Returns:
        A Config instance with appropriate settings for the instrument.
    """
    if instrument == Instrument.Guitar:
        return init_guitar_config(min_velocity)
    elif instrument == Instrument.Harpejji:
        return init_harpejji_config(min_velocity)
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
        disabled_note=COLORS["Red"],
        linked_note=COLORS["Lime"],
        misc_pressed=COLORS["Sky"],
        control=COLORS["Yellow"],
        control_pressed=COLORS["Pink"],
    )
