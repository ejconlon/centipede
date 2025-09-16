"""Push controller interface and MIDI message handling.

This module provides the core interface for communicating with the
Ableton Push controller, including MIDI message creation, event parsing,
port management, and the abstract PushInterface for display updates.
"""

from __future__ import annotations

import logging
import time
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, List, Optional, Type, TypeVar

from mido.frozen import FrozenMessage

from pushpluck import constants
from pushpluck.base import Closeable, Resettable
from pushpluck.color import COLORS, Color
from pushpluck.constants import (
    ButtonCC,
    ButtonColor,
    ButtonIllum,
    KnobCC,
    KnobGroup,
    TimeDivCC,
)
from pushpluck.midi import MidiInput, MidiOutput, is_note_msg
from pushpluck.pos import ChanSelPos, GridSelPos, Pos

__all__ = [
    "ButtonCC",
    "ButtonColor",
    "ButtonIllum",
    "TimeDivCC",
    "frame_sysex",
    "ButtonEvent",
    "KnobEvent",
    "PushEvent",
    "PushInterface",
    "PushOutput",
]


def frame_sysex(raw_data: List[int]) -> FrozenMessage:
    """Frame raw data as a Push SysEx message.

    Args:
        raw_data: The payload data to include in the SysEx message.

    Returns:
        A FrozenMessage containing the properly framed SysEx data.
    """
    data: List[int] = []
    data.extend(constants.PUSH_SYSEX_PREFIX)
    data.extend(raw_data)
    return FrozenMessage("sysex", data=data)


def make_color_msg(pos: Pos, color: Color) -> FrozenMessage:
    """Create a SysEx message to set a pad's RGB color.

    Args:
        pos: The pad position to color.
        color: The RGB color to set.

    Returns:
        A FrozenMessage containing the color setting SysEx command.
    """
    index = pos.to_index()
    msb = [(x & 240) >> 4 for x in color]
    lsb = [x & 15 for x in color]
    raw_data = [4, 0, 8, index, 0, msb[0], lsb[0], msb[1], lsb[1], msb[2], lsb[2]]
    return frame_sysex(raw_data)


def make_led_msg(pos: Pos, value: int) -> FrozenMessage:
    """Create a MIDI message to set a pad's LED brightness.

    Args:
        pos: The pad position to control.
        value: The brightness value (0-127, 0 = off).

    Returns:
        A FrozenMessage note_on with the specified velocity.
    """
    note = pos.to_note()
    return FrozenMessage("note_on", note=note, velocity=value)


def make_lcd_msg(row: int, offset: int, text: str) -> FrozenMessage:
    """Create a SysEx message to display text on the Push LCD.

    Args:
        row: The display row (0-3).
        offset: The column offset for the text.
        text: The text string to display.

    Returns:
        A FrozenMessage containing the LCD text SysEx command.
    """
    raw_data = [27 - row, 0, len(text) + 1, offset]
    for c in text:
        raw_data.append(ord(c))
    return frame_sysex(raw_data)


E = TypeVar("E", bound="PushEvent")  # Type variable for Push event types


class PushEvent(metaclass=ABCMeta):
    """Abstract base class for Push controller events.

    All Push events implement a match method to parse MIDI messages
    into typed event objects.
    """

    @classmethod
    @abstractmethod
    def match(cls: Type[E], msg: FrozenMessage) -> Optional[E]:
        """Parse a MIDI message into this event type.

        Args:
            msg: The MIDI message to parse.

        Returns:
            An instance of this event type if the message matches,
            None otherwise.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class KnobEvent(PushEvent):
    """Represents a knob turn event from the Push controller."""

    knob: KnobCC  # Which specific knob was turned
    """The specific knob that was turned."""
    group: KnobGroup  # Which group the knob belongs to (left, center, right)
    """The group this knob belongs to (Left, Center, Right)."""
    offset: int  # Offset within the group
    """The offset of this knob within its group."""
    clockwise: bool  # Direction of turn
    """True if turned clockwise, False if counterclockwise."""

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[KnobEvent]:
        if msg.type == "control_change":
            knob = constants.KNOB_CC_VALUE_LOOKUP.get(msg.control)
            if knob is not None:
                group, offset = constants.knob_group_and_offset(knob)
                return cls(knob, group, offset, msg.value < 127)
        return None


@dataclass(frozen=True)
class ButtonEvent(PushEvent):
    """Represents a button press/release event from the Push controller."""

    button: ButtonCC  # Which button was pressed/released
    """The specific button that was pressed or released."""
    pressed: bool  # True for press, False for release
    """True if the button was pressed, False if released."""

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[ButtonEvent]:
        if msg.type == "control_change":
            button = constants.BUTTON_CC_VALUE_LOOKUP.get(msg.control)
            if button is not None:
                return cls(button, msg.value > 0)
        return None


@dataclass(frozen=True)
class PadEvent(PushEvent):
    """Represents a pad press/release event from the Push controller."""

    pos: Pos  # Position of the pad that was hit
    """The position of the pad that was pressed or released."""
    velocity: int  # MIDI velocity (0 = release, >0 = press)
    """The MIDI velocity (0 for release, >0 for press with dynamics)."""

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[PadEvent]:
        if is_note_msg(msg):
            pos = Pos.from_input_note(msg.note)
            if pos is not None:
                return PadEvent(pos, msg.velocity)
        return None


@dataclass(frozen=True)
class TimeDivEvent(PushEvent):
    time_div: constants.TimeDivCC
    pressed: bool

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[TimeDivEvent]:
        if msg.type == "control_change":
            time_div = constants.TIME_DIV_CC_VALUE_LOOKUP.get(msg.control)
            if time_div is not None:
                return cls(time_div, msg.value > 0)
        return None


@dataclass(frozen=True)
class GridSelEvent(PushEvent):
    gs_pos: GridSelPos
    pressed: bool

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[GridSelEvent]:
        if msg.type == "control_change":
            gs_pos = GridSelPos.from_input_control(msg.control)
            if gs_pos is not None:
                return cls(gs_pos, msg.value > 0)
        return None


@dataclass(frozen=True)
class ChanSelEvent(PushEvent):
    cs_pos: ChanSelPos
    pressed: bool

    @classmethod
    def match(cls, msg: FrozenMessage) -> Optional[ChanSelEvent]:
        if msg.type == "control_change":
            cs_pos = ChanSelPos.from_input_control(msg.control)
            if cs_pos is not None:
                return cls(cs_pos, msg.value > 0)
        return None


def match_event(msg: FrozenMessage) -> Optional[PushEvent]:
    """Parse a MIDI message into a Push event.

    Attempts to parse the message as different event types in order,
    returning the first successful match.

    Args:
        msg: The MIDI message to parse.

    Returns:
        A PushEvent instance if the message matches any known event type,
        None otherwise.
    """
    knob_event = KnobEvent.match(msg)
    if knob_event is not None:
        return knob_event
    button_event = ButtonEvent.match(msg)
    if button_event is not None:
        return button_event
    pad_event = PadEvent.match(msg)
    if pad_event is not None:
        return pad_event
    td_event = TimeDivEvent.match(msg)
    if td_event is not None:
        return td_event
    gs_event = GridSelEvent.match(msg)
    if gs_event is not None:
        return gs_event
    cs_event = ChanSelEvent.match(msg)
    if cs_event is not None:
        return cs_event
    # TODO polytouch and pitchwheel events
    return None


@dataclass(frozen=True)
class PushPorts(Closeable):
    """Container for all MIDI port connections used by PushPluck.

    Manages the input port from the Push, output port to the Push,
    and the virtual output port for processed musical notes.
    """

    midi_in: MidiInput  # Input from the Push controller
    """MIDI input port receiving messages from the Push controller."""
    midi_out: MidiOutput  # Output to the Push controller
    """MIDI output port for sending messages to the Push controller."""
    midi_processed: MidiOutput  # Virtual output for processed notes
    """Virtual MIDI output port for the processed musical notes."""

    @classmethod
    def open(
        cls, push_port_name: str, processed_port_name: str, delay: Optional[float]
    ) -> PushPorts:
        """Open all required MIDI ports for PushPluck.

        Args:
            push_port_name: Name of the Push controller's MIDI port.
            processed_port_name: Name for the virtual processed output port.
            delay: Optional delay for rate limiting Push messages.

        Returns:
            A PushPorts instance with all ports opened and connected.
        """
        midi_in = MidiInput.open(push_port_name)
        midi_out = MidiOutput.open(push_port_name, delay=delay)
        midi_processed = MidiOutput.open(processed_port_name, virtual=True)
        return cls(midi_in=midi_in, midi_out=midi_out, midi_processed=midi_processed)

    def close(self) -> None:
        self.midi_in.close()
        self.midi_out.close()
        self.midi_processed.close()


@contextmanager
def push_ports_context(
    push_port_name: str, processed_port_name: str, delay: Optional[float]
) -> Generator[PushPorts, None, None]:
    """Context manager for Push MIDI port lifecycle management.

    Opens all required MIDI ports and ensures they are properly closed
    when done, even if an exception occurs.

    Args:
        push_port_name: Name of the Push controller's MIDI port.
        processed_port_name: Name for the virtual processed output port.
        delay: Optional delay for rate limiting Push messages.

    Yields:
        A PushPorts instance with all ports opened and ready for use.
    """
    logging.info("opening ports")
    ports = PushPorts.open(
        push_port_name=push_port_name,
        processed_port_name=processed_port_name,
        delay=delay,
    )
    logging.info("opened ports")
    try:
        yield ports
    finally:
        logging.info("closing ports")
        ports.close()
        logging.info("closed ports")


class PushInterface(Resettable, metaclass=ABCMeta):
    """Abstract interface for controlling the Push controller display.

    This class defines the complete interface for updating the Push
    controller's visual elements: pads, LCD display, and buttons.
    Implementations handle the actual MIDI communication.
    """

    @abstractmethod
    def pad_led_off(self, pos: Pos) -> None:
        raise NotImplementedError()

    @abstractmethod
    def pad_set_color(self, pos: Pos, color: Color) -> None:
        raise NotImplementedError()

    def pad_reset(self) -> None:
        for pos in Pos.iter_all():
            self.pad_led_off(pos)

    @abstractmethod
    def lcd_display_raw(self, row: int, line_col: int, text: str) -> None:
        raise NotImplementedError()

    def lcd_display_line(self, row: int, text: str) -> None:
        text = text.ljust(constants.DISPLAY_MAX_LINE_LEN, " ")
        self.lcd_display_raw(row, 0, text)

    def lcd_display_block(self, row: int, block_col: int, text: str) -> None:
        assert row >= 0 and row < constants.DISPLAY_MAX_ROWS
        assert block_col >= 0 and block_col < constants.DISPLAY_MAX_BLOCKS
        assert len(text) <= constants.DISPLAY_BLOCK_LEN
        text = text.ljust(constants.DISPLAY_BLOCK_LEN, " ")
        line_col = constants.DISPLAY_BLOCK_LEN * block_col
        self.lcd_display_raw(row, line_col, text)

    def lcd_display_half_block(self, row: int, half_col: int, text: str) -> None:
        block_col = half_col // 2
        half = half_col % 2
        assert row >= 0 and row < constants.DISPLAY_MAX_ROWS
        assert block_col >= 0 and block_col < constants.DISPLAY_MAX_BLOCKS
        # Truncate text if it's too long
        if len(text) > constants.DISPLAY_HALF_BLOCK_LEN:
            text = text[: constants.DISPLAY_HALF_BLOCK_LEN]
        offset: int
        just_text: str
        if half == 0:
            offset = 0
            just_text = text.ljust(constants.DISPLAY_HALF_BLOCK_LEN + 1, " ")
        else:
            offset = constants.DISPLAY_HALF_BLOCK_LEN + 1
            just_text = text.ljust(constants.DISPLAY_HALF_BLOCK_LEN, " ")
        line_col = constants.DISPLAY_BLOCK_LEN * block_col + offset
        self.lcd_display_raw(row, line_col, just_text)

    def lcd_reset(self) -> None:
        for row in range(constants.DISPLAY_MAX_ROWS):
            self.lcd_display_line(row, "")

    @abstractmethod
    def button_set_illum(self, button: ButtonCC, illum: ButtonIllum) -> None:
        raise NotImplementedError()

    @abstractmethod
    def button_off(self, button: ButtonCC) -> None:
        raise NotImplementedError()

    def button_reset(self) -> None:
        for button in ButtonCC:
            self.button_off(button)

    @abstractmethod
    def time_div_off(self, time_div: TimeDivCC) -> None:
        raise NotImplementedError()

    def time_div_reset(self) -> None:
        # TODO
        # for time_div in TimeDivCC:
        #     self.time_div_off(time_div)
        pass

    @abstractmethod
    def chan_sel_set_color(
        self, cs_pos: ChanSelPos, illum: ButtonIllum, color: ButtonColor
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def chan_sel_off(self, cs_pos: ChanSelPos) -> None:
        raise NotImplementedError()

    def chan_sel_reset(self) -> None:
        # TODO
        # for cs_pos in ChanSelPos.iter_all():
        #     self.chan_sel_off(cs_pos)
        pass

    @abstractmethod
    def grid_sel_set_color(self, gs_pos: GridSelPos, color: Color) -> None:
        raise NotImplementedError()

    @abstractmethod
    def grid_sel_off(self, gs_pos: GridSelPos) -> None:
        raise NotImplementedError()

    def grid_sel_reset(self) -> None:
        # TODO
        # for gs_pos in GridSelPos.iter_all():
        #     self.grid_sel_off(gs_pos)
        pass

    def reset(self) -> None:
        self.lcd_reset()
        self.button_reset()
        self.pad_reset()
        self.grid_sel_reset()
        self.chan_sel_reset()
        self.time_div_reset()


class PushOutput(PushInterface):
    """Concrete implementation of PushInterface using MIDI output.

    This class implements all the Push interface methods by sending
    appropriate MIDI messages to the Push controller.
    """

    def __init__(self, midi_out: MidiOutput) -> None:
        """Initialize with a MIDI output port.

        Args:
            midi_out: The MIDI output port connected to the Push.
        """
        self._midi_out = midi_out

    def _pad_led_on(self, pos: Pos, value: int = 100) -> None:
        msg = make_led_msg(pos, value)
        self._midi_out.send_msg(msg)

    def pad_led_off(self, pos: Pos) -> None:
        self._pad_led_on(pos, 0)

    def pad_set_color(self, pos: Pos, color: Color) -> None:
        msg = make_color_msg(pos, color)
        self._midi_out.send_msg(msg)

    def lcd_display_raw(self, row: int, line_col: int, text: str) -> None:
        assert row >= 0 and row < constants.DISPLAY_MAX_ROWS
        assert line_col >= 0
        assert len(text) + line_col <= constants.DISPLAY_MAX_LINE_LEN
        msg = make_lcd_msg(row, line_col, text)
        self._midi_out.send_msg(msg)

    def button_set_illum(self, button: ButtonCC, illum: ButtonIllum) -> None:
        msg = FrozenMessage(
            type="control_change", control=button.value, value=illum.value
        )
        self._midi_out.send_msg(msg)

    def button_off(self, button: ButtonCC) -> None:
        msg = FrozenMessage(type="control_change", control=button.value, value=0)
        self._midi_out.send_msg(msg)

    def time_div_off(self, time_div: TimeDivCC) -> None:
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


def rainbow(push: PushOutput) -> None:
    """Display a rainbow color sequence on all pads for testing.

    Cycles through rainbow colors (ROYGBIV) on all pads with a
    1-second delay between each color. Useful for testing the
    Push connection and color display.

    Args:
        push: The Push output interface to use.
    """
    names = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"]
    for name in names:
        color = COLORS[name]
        for pos in Pos.iter_all():
            push.pad_set_color(pos, color)
        time.sleep(1)
