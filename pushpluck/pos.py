"""Position classes for the Push controller interface elements.

This module defines position classes for different types of elements
on the Push controller: pad positions, channel selector positions,
and grid selector positions. Each provides conversion between
coordinates and MIDI note/control numbers.
"""

from dataclasses import dataclass
from typing import Generator, Optional

from pushpluck import constants


@dataclass(frozen=True)
class Pos:
    """Represents a position on the Push controller's 8x8 pad grid.

    The coordinate system uses (0,0) as the bottom-left corner (lowest note)
    and (7,7) as the top-right corner (highest note). This matches the
    musical layout where higher pitches are towards the top-right.
    """

    row: int  # Row position (0-7, bottom to top)
    """Row position on the pad grid (0-7, where 0 is bottom)."""
    col: int  # Column position (0-7, left to right)
    """Column position on the pad grid (0-7, where 0 is left)."""

    def __iter__(self) -> Generator[int, None, None]:
        """Iterate over row and column coordinates.

        Yields:
            Row coordinate followed by column coordinate.
        """
        yield self.row
        yield self.col

    def to_index(self) -> int:
        """Convert position to a linear index.

        Returns:
            Linear index (0-63) for this position in row-major order.
        """
        return constants.NUM_PAD_COLS * self.row + self.col

    def to_note(self) -> int:
        """Convert position to its corresponding MIDI note number.

        Returns:
            MIDI note number for this pad position.
        """
        return constants.LOW_NOTE + self.to_index()

    @staticmethod
    def from_input_note(note: int) -> "Optional[Pos]":
        """Create a position from a MIDI note number.

        Args:
            note: The MIDI note number to convert.

        Returns:
            Pos instance for this note, or None if the note is outside
            the valid range for the pad grid.
        """
        if note < constants.LOW_NOTE or note >= constants.HIGH_NOTE:
            return None
        else:
            index = note - constants.LOW_NOTE
            row = index // constants.NUM_PAD_COLS
            col = index % constants.NUM_PAD_COLS
            return Pos(row=row, col=col)

    @staticmethod
    def iter_all() -> "Generator[Pos, None, None]":
        """Iterate over all valid pad positions from lowest to highest.

        Yields:
            Pos instances for all positions in row-major order
            (from bottom-left to top-right).
        """
        for row in range(constants.NUM_PAD_ROWS):
            for col in range(constants.NUM_PAD_COLS):
                yield Pos(row, col)


@dataclass(frozen=True)
class ChanSelPos:
    """Represents a position on the Push controller's channel selector strip.

    The channel selector is a row of controls below the main pad grid
    used for channel-specific functions and navigation.
    """

    col: int  # Column position (0-7)
    """Column position on the channel selector strip (0-7)."""

    def to_control(self) -> int:
        """Convert position to its corresponding MIDI control number.

        Returns:
            MIDI control change number for this channel selector position.
        """
        return constants.LOW_CHAN_CONTROL + self.col

    @staticmethod
    def from_input_control(control: int) -> "Optional[ChanSelPos]":
        """Create a position from a MIDI control number.

        Args:
            control: The MIDI control change number.

        Returns:
            ChanSelPos instance for this control, or None if the control
            is outside the valid range for the channel selector.
        """
        col = control - constants.LOW_CHAN_CONTROL
        if col < 0 or col > constants.NUM_PAD_COLS:
            return None
        else:
            return ChanSelPos(col)

    @staticmethod
    def iter_all() -> "Generator[ChanSelPos, None, None]":
        """Iterate over all valid channel selector positions.

        Yields:
            ChanSelPos instances for all positions from left to right.
        """
        for col in range(constants.NUM_PAD_COLS):
            yield ChanSelPos(col)


@dataclass(frozen=True)
class GridSelPos:
    """Represents a position on the Push controller's grid selector strip.

    The grid selector is another row of controls used for grid-specific
    functions and navigation, typically above the main pad grid.
    """

    col: int  # Column position (0-7)
    """Column position on the grid selector strip (0-7)."""

    def to_control(self) -> int:
        """Convert position to its corresponding MIDI control number.

        Returns:
            MIDI control change number for this grid selector position.
        """
        return constants.LOW_GRID_CONTROL + self.col

    @staticmethod
    def from_input_control(control: int) -> "Optional[GridSelPos]":
        """Create a position from a MIDI control number.

        Args:
            control: The MIDI control change number.

        Returns:
            GridSelPos instance for this control, or None if the control
            is outside the valid range for the grid selector.
        """
        col = control - constants.LOW_GRID_CONTROL
        if col < 0 or col > constants.NUM_PAD_COLS:
            return None
        else:
            return GridSelPos(col)

    @staticmethod
    def iter_all() -> "Generator[GridSelPos, None, None]":
        """Iterate over all valid grid selector positions.

        Yields:
            GridSelPos instances for all positions from left to right.
        """
        for col in range(constants.NUM_PAD_COLS):
            yield GridSelPos(col)
