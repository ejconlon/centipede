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

        # For infinite strings, any string index is valid
        return StringPos(str_index=str_index, fret=fret)

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

        Determines which portion of the fretboard is currently visible
        on the pad grid, taking into account layout, offsets, and the
        number of strings in the tuning.

        Returns:
            StringBounds defining the visible fretboard region, or None
            if no valid strings are visible in the current viewport.
        """
        view_offset = self._view_str_offset()

        # Determine the maximum string and fret dimensions based on effective layout
        # We need to consider how the layout maps logical coordinates to physical ones
        # For now, assume both dimensions are bounded by the grid size
        max_str_dim = constants.NUM_PAD_ROWS
        num_frets_bounded = constants.NUM_PAD_COLS

        # For infinite strings, any string index is valid
        valid_indices: List[int] = []
        for i in range(max_str_dim):
            o = view_offset + i
            valid_indices.append(o)

        if len(valid_indices) == 0:
            return None
        else:
            low_str_index = valid_indices[0]
            high_str_index = valid_indices[-1]
            low = StringPos(str_index=low_str_index, fret=self._config.fret_offset)
            high = StringPos(
                str_index=high_str_index, fret=low.fret + num_frets_bounded - 1
            )
            return StringBounds(low, high)
