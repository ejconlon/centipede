"""Shadow state management for efficient Push controller updates.

This module implements a shadow/diff system for the Push controller display,
tracking the current state and only sending updates when changes occur.
This prevents unnecessary MIDI traffic and improves performance.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, Optional

from pushpluck import constants
from pushpluck.base import Resettable
from pushpluck.color import Color
from pushpluck.pos import ChanSelPos, GridSelPos, Pos
from pushpluck.push import ButtonCC, ButtonColor, ButtonIllum, PushInterface, TimeDivCC


class LcdRow:
    """Represents a single row of the Push controller's LCD display.

    Manages the character buffer for one line of text and provides
    methods for updating portions of the line efficiently.
    """

    def __init__(self) -> None:
        """Initialize the LCD row with spaces."""
        self._buffer = [ord(" ") for i in range(constants.DISPLAY_MAX_LINE_LEN)]

    def get_text(self, start: int, length: int) -> str:
        """Get a substring from this LCD row.

        Args:
            start: Starting character position.
            length: Number of characters to retrieve.

        Returns:
            The text substring from the specified range.
        """
        end = start + length
        assert start >= 0 and length >= 0 and end <= constants.DISPLAY_MAX_LINE_LEN
        return "".join(chr(i) for i in self._buffer[start:end])

    def get_all_text(self) -> str:
        """Get the entire text content of this LCD row.

        Returns:
            The complete text content as a string.
        """
        return self.get_text(0, constants.DISPLAY_MAX_LINE_LEN)

    def set_text(self, start: int, text: str) -> bool:
        """Set text at the specified position in this LCD row.

        Args:
            start: Starting position for the text.
            text: Text string to set.

        Returns:
            True if any characters were changed, False otherwise.
        """
        length = len(text)
        end = start + length
        assert start >= 0 and end <= constants.DISPLAY_MAX_LINE_LEN
        changed = False
        for i in range(length):
            old_char = self._buffer[start + i]
            new_char = ord(text[i])
            if old_char != new_char:
                changed = True
            self._buffer[start + i] = new_char
        return changed

    def set_all_text(self, text: str) -> bool:
        """Set the entire text content of this LCD row.

        Args:
            text: Text string to set (will be padded/truncated to fit).

        Returns:
            True if any characters were changed, False otherwise.
        """
        return self.set_text(0, text)


@dataclass(frozen=True, eq=False)
class PushState:
    """Represents the complete state of the Push controller interface.

    Tracks the current state of LCD display, pad colors, and button
    illumination to enable efficient diff-based updates.
    """

    lcd: Dict[int, LcdRow]  # LCD display state by row
    """Dictionary mapping row numbers to LcdRow instances."""
    pads: Dict[Pos, Optional[Color]]  # Pad colors by position
    """Dictionary mapping pad positions to their current colors (None = off)."""
    buttons: Dict[ButtonCC, Optional[ButtonIllum]]  # Button illumination states
    """Dictionary mapping buttons to their illumination states (None = off)."""

    @classmethod
    def reset(cls) -> "PushState":
        """Create a reset state with all elements cleared.

        Returns:
            A PushState with empty LCD, all pads off, and all buttons off.
        """
        return cls(
            lcd={row: LcdRow() for row in range(constants.DISPLAY_MAX_ROWS)},
            pads={pos: None for pos in Pos.iter_all()},
            buttons={button: None for button in ButtonCC},
        )

    @classmethod
    def diff(cls) -> "PushState":
        """Create an empty diff state for tracking changes.

        Returns:
            A PushState with empty dictionaries for collecting changes.
        """
        return cls(lcd={}, pads={}, buttons={})


class PushShadow(Resettable):
    """Shadow state manager for efficient Push controller updates.

    This class maintains a shadow copy of the Push controller's state
    and only sends updates when changes occur, reducing MIDI traffic
    and improving performance.
    """

    def __init__(self, push: PushInterface) -> None:
        """Initialize the shadow with a Push interface.

        Args:
            push: The Push interface to manage.
        """
        self._push = push
        self._state = PushState.reset()

    def reset(self) -> None:
        """Reset the Push controller and shadow state."""
        self._push.reset()
        self._state = PushState.reset()

    @contextmanager
    def context(self) -> Generator["PushInterface", None, None]:
        """Create a managed context for batched Push updates.

        Returns:
            A context manager that yields a PushInterface for making
            changes, then automatically emits only the changes that
            were made when the context exits.
        """
        diff_state = PushState.diff()
        managed = PushShadowManaged(diff_state)
        yield managed
        self._emit(diff_state)

    def _emit(self, diff_state: PushState) -> None:
        self._emit_lcd(diff_state)
        self._emit_pads(diff_state)
        self._emit_buttons(diff_state)

    def _emit_lcd(self, diff_state: PushState) -> None:
        for row, new_row in diff_state.lcd.items():
            old_row = self._state.lcd[row]
            new_text = new_row.get_all_text()
            if old_row.set_all_text(new_text):
                self._push.lcd_display_raw(row, 0, new_text)

    def _emit_pads(self, diff_state: PushState) -> None:
        for pos, new_color in diff_state.pads.items():
            old_color = self._state.pads.get(pos)
            if old_color != new_color:
                if new_color is None:
                    self._push.pad_led_off(pos)
                    if pos in self._state.pads:
                        del self._state.pads[pos]
                else:
                    self._push.pad_set_color(pos, new_color)
                    self._state.pads[pos] = new_color

    def _emit_buttons(self, diff_state: PushState) -> None:
        for button, new_illum in diff_state.buttons.items():
            old_illum = self._state.buttons[button]
            if old_illum != new_illum:
                if new_illum is None:
                    self._push.button_off(button)
                    del self._state.buttons[button]
                else:
                    self._push.button_set_illum(button, new_illum)


class PushShadowManaged(PushInterface):
    """Managed Push interface that collects changes for later emission.

    This class implements the PushInterface but instead of immediately
    sending updates to the hardware, it collects changes in a diff state
    that can be efficiently processed later.
    """

    def __init__(self, state: PushState):
        """Initialize with a diff state for collecting changes.

        Args:
            state: The diff state to collect changes into.
        """
        self._state = state

    def pad_led_off(self, pos: Pos) -> None:
        self._state.pads[pos] = None

    def pad_set_color(self, pos: Pos, color: Color) -> None:
        self._state.pads[pos] = color

    def lcd_display_raw(self, row: int, line_col: int, text: str) -> None:
        if row not in self._state.lcd:
            self._state.lcd[row] = LcdRow()
        self._state.lcd[row].set_text(line_col, text)

    def button_set_illum(self, button: ButtonCC, illum: ButtonIllum) -> None:
        self._state.buttons[button] = illum

    def button_off(self, button: ButtonCC) -> None:
        self._state.buttons[button] = None

    def time_div_off(self, time_div: TimeDivCC) -> None:
        raise NotImplementedError()

    def time_div_reset(self) -> None:
        raise NotImplementedError()

    def chan_sel_set_color(
        self, cs_pos: ChanSelPos, illum: ButtonIllum, color: ButtonColor
    ) -> None:
        raise NotImplementedError()

    def chan_sel_off(self, cs_pos: ChanSelPos) -> None:
        raise NotImplementedError()

    def grid_sel_set_color(self, gs_pos: GridSelPos, color: Color) -> None:
        raise NotImplementedError()

    def grid_sel_off(self, gs_pos: GridSelPos) -> None:
        raise NotImplementedError()
