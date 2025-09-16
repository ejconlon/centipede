"""Viewport management for mapping between pad positions and fretboard coordinates.

This module handles the coordinate transformations between the Push controller's
8x8 pad grid and the virtual fretboard's string/fret coordinate system.
It supports different layout orientations and scrolling offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pushpluck import constants
from pushpluck.base import Unit
from pushpluck.component import MappedComponent, MappedComponentConfig
from pushpluck.config import Config, Layout
from pushpluck.fretboard import StringBounds, StringPos
from pushpluck.pos import Pos


@dataclass(frozen=True)
class ViewportConfig(MappedComponentConfig[Config]):
    """Configuration for the viewport coordinate mapping.

    Contains the parameters needed to map between pad positions
    and fretboard coordinates, including layout orientation and offsets.
    """

    num_strings: int
    """Number of strings in the current instrument tuning."""
    tuning: List[int]
    """MIDI note numbers for each string's open note."""
    repeat_steps: int
    """Number of semitones before the pattern repeats (for infinite strings)."""
    effective_layout: Layout
    """The effective layout transformation (pre_layout * layout) for coordinate mapping."""
    view_offset: int
    """String offset for centering/positioning the default view."""
    str_offset: int
    """Offset for scrolling through strings (negative values scroll down)."""
    fret_offset: int
    """Offset for scrolling through frets (negative values scroll left)."""

    @classmethod
    def extract(cls, root_config: Config) -> ViewportConfig:
        """Extract viewport configuration from the main config.

        Args:
            root_config: The main application configuration.

        Returns:
            A ViewportConfig with the relevant parameters extracted.
        """
        return cls(
            num_strings=len(root_config.tuning),
            tuning=root_config.tuning,
            repeat_steps=root_config.repeat_steps,
            effective_layout=root_config.effective_layout,
            view_offset=root_config.view_offset,
            str_offset=root_config.str_offset,
            fret_offset=root_config.fret_offset,
        )


class Viewport(MappedComponent[Config, ViewportConfig, Unit]):
    """Manages coordinate mapping between pads and fretboard positions.

    The Viewport handles the complex coordinate transformations between
    the Push controller's pad grid and the virtual fretboard, supporting
    different layout orientations and scrolling through the fretboard.
    """

    @classmethod
    def construct(cls, root_config: Config) -> Viewport:
        """Construct a Viewport from the main configuration.

        Args:
            root_config: The main application configuration.

        Returns:
            A new Viewport instance configured from the input.
        """
        return cls(cls.extract_config(root_config))

    @classmethod
    def extract_config(cls, root_config: Config) -> ViewportConfig:
        """Extract viewport configuration from the main config.

        Args:
            root_config: The main application configuration.

        Returns:
            A ViewportConfig with the relevant parameters.
        """
        return ViewportConfig.extract(root_config)

    def handle_mapped_config(self, config: ViewportConfig) -> Unit:
        """Handle a viewport configuration change.

        Args:
            config: The new viewport configuration.

        Returns:
            Unit instance indicating successful configuration update.
        """
        self._config = config
        return Unit()

    def _view_str_offset(self) -> int:
        # Apply view_offset to shift the string mapping
        # Negative view_offset shifts strings left (showing higher string indices)
        return -self._config.view_offset

    def _total_str_offset(self) -> int:
        return self._view_str_offset() + self._config.str_offset

    def _get_note_for_string_index(self, str_index: int) -> int:
        """Calculate the base MIDI note for a given string index.

        Uses the same logic as InfiniteTuner for infinite string mapping.

        Args:
            str_index: The string index to calculate the note for.

        Returns:
            The base MIDI note number for that string.
        """
        if not self._config.tuning:
            raise ValueError("Tuning cannot be empty")

        # Calculate which tuning note and octave offset
        tuning_index = str_index % len(self._config.tuning)
        octave_offset = (
            str_index // len(self._config.tuning)
        ) * self._config.repeat_steps

        return self._config.tuning[tuning_index] + octave_offset

    def _calculate_midi_note(self, str_pos: StringPos) -> int:
        """Calculate the MIDI note number for a string position.

        Args:
            str_pos: The string position to calculate the note for.

        Returns:
            The calculated MIDI note number (may be outside valid range).
        """
        base_note = self._get_note_for_string_index(str_pos.str_index)
        return base_note + str_pos.fret

    def _is_valid_midi_note(self, str_pos: StringPos) -> bool:
        """Check if a string position would produce a valid MIDI note.

        Args:
            str_pos: The string position to validate.

        Returns:
            True if the position would produce a valid MIDI note (0-127), False otherwise.
        """
        try:
            note = self._calculate_midi_note(str_pos)
            return 0 <= note <= 127
        except (ValueError, IndexError):
            return False

    def str_pos_from_pad_pos(self, pos: Pos) -> Optional[StringPos]:
        """Convert a pad position to a fretboard string position.

        Takes into account the current layout, offsets, and instrument tuning
        to map from the pad grid coordinates to fretboard coordinates.

        Args:
            pos: The pad position to convert.

        Returns:
            The corresponding StringPos, or None if the pad position
            maps to an invalid or out-of-range string/fret combination.
        """
        # Apply layout transformation to get the "logical" string/fret coordinates
        # The effective layout maps from physical pad position to logical fretboard position
        logical_row, logical_col = self._config.effective_layout.apply_to_coords(
            pos.row, pos.col
        )

        # In the logical coordinate system: row = string, col = fret
        str_index = logical_row + self._total_str_offset()
        fret = logical_col + self._config.fret_offset

        # Create the string position
        str_pos = StringPos(str_index=str_index, fret=fret)

        # Validate that this position would produce a valid MIDI note
        if self._is_valid_midi_note(str_pos):
            return str_pos
        else:
            return None

    def str_pos_from_input_note(self, note: int) -> Optional[StringPos]:
        """Convert a MIDI note number to a fretboard string position.

        Args:
            note: The MIDI note number from pad input.

        Returns:
            The corresponding StringPos, or None if the note doesn't
            correspond to a valid pad or fretboard position.
        """
        pos = Pos.from_input_note(note)
        return self.str_pos_from_pad_pos(pos) if pos is not None else None

    def pad_pos_from_str_pos(self, str_pos: StringPos) -> Optional[Pos]:
        """Convert a fretboard string position to a pad position.

        Takes into account the current layout and offsets to map from
        fretboard coordinates back to pad grid coordinates.

        Args:
            str_pos: The string position to convert.

        Returns:
            The corresponding Pos on the pad grid, or None if the
            string position is outside the current viewport.
        """
        # Convert to logical coordinates (before layout transformation)
        logical_row = str_pos.str_index - self._total_str_offset()
        logical_col = str_pos.fret - self._config.fret_offset

        # Apply inverse layout transformation to get physical pad position
        inverse_layout = self._config.effective_layout.inverse()
        row, col = inverse_layout.apply_to_coords(logical_row, logical_col)

        if row < 0 or row >= constants.NUM_PAD_ROWS:
            return None
        elif col < 0 or col >= constants.NUM_PAD_COLS:
            return None
        else:
            return Pos(row=row, col=col)

    def str_bounds(self) -> Optional[StringBounds]:
        """Calculate the fretboard bounds visible in the current viewport.

        Determines which portion of the infinite fretboard is currently visible
        on the pad grid, taking into account layout, offsets, and coordinate transformations.
        With infinite strings, the viewport can access any string index by scrolling.

        Returns:
            StringBounds defining the visible fretboard region.
        """
        # For infinite strings, we need to determine which string indices
        # correspond to the physical pad positions after layout transformation
        min_str_index = None
        max_str_index = None
        min_fret = None
        max_fret = None

        # Sample all pad positions to find the actual bounds after transformation
        for row in range(constants.NUM_PAD_ROWS):
            for col in range(constants.NUM_PAD_COLS):
                pos = Pos(row=row, col=col)
                str_pos = self.str_pos_from_pad_pos(pos)
                if str_pos is not None:
                    if min_str_index is None or str_pos.str_index < min_str_index:
                        min_str_index = str_pos.str_index
                    if max_str_index is None or str_pos.str_index > max_str_index:
                        max_str_index = str_pos.str_index
                    if min_fret is None or str_pos.fret < min_fret:
                        min_fret = str_pos.fret
                    if max_fret is None or str_pos.fret > max_fret:
                        max_fret = str_pos.fret

        if (
            min_str_index is None
            or max_str_index is None
            or min_fret is None
            or max_fret is None
        ):
            return None

        low = StringPos(str_index=min_str_index, fret=min_fret)
        high = StringPos(str_index=max_str_index, fret=max_fret)
        return StringBounds(low, high)
