"""MIDI input/output handling for the PushPluck application.

This module provides classes for managing MIDI input and output connections,
including message queuing, rate limiting, and utility functions for
identifying different types of MIDI messages.
"""

from __future__ import annotations

import logging
import time
from abc import ABCMeta, abstractmethod
from queue import SimpleQueue
from typing import Optional, cast

import mido
from mido import Message
from mido.frozen import FrozenMessage, freeze_message
from mido.ports import BaseInput, BaseOutput

from pushpluck.base import Closeable, Resettable


def is_note_msg(msg: FrozenMessage) -> bool:
    """Check if a message is any type of note message.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message is either note_on or note_off.
    """
    return cast(bool, msg.type == "note_on" or msg.type == "note_off")


def is_note_on_msg(msg: FrozenMessage) -> bool:
    """Check if a message is a true note-on message.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message is note_on with velocity > 0.
    """
    return cast(bool, msg.type == "note_on" and msg.velocity > 0)


def is_note_off_msg(msg: FrozenMessage) -> bool:
    """Check if a message is a note-off message.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message is note_off or note_on with velocity 0.
    """
    return cast(
        bool, (msg.type == "note_on" and msg.velocity == 0) or msg.type == "note_off"
    )


class MidiSource(metaclass=ABCMeta):
    """Abstract base class for MIDI input sources."""

    @abstractmethod
    def recv_msg(self) -> FrozenMessage:
        """Receive the next MIDI message.

        Returns:
            The next available MIDI message.
        """
        raise NotImplementedError()


class MidiSink(metaclass=ABCMeta):
    """Abstract base class for MIDI output sinks."""

    @abstractmethod
    def send_msg(self, msg: FrozenMessage) -> None:
        """Send a MIDI message.

        Args:
            msg: The MIDI message to send.
        """
        raise NotImplementedError()


class MidiInput(MidiSource, Closeable):
    """MIDI input connection with message queuing.

    This class manages a MIDI input port and provides a queue-based
    interface for receiving messages. Messages are queued as they
    arrive and can be retrieved synchronously.
    """

    @classmethod
    def open(cls, in_port_name: str) -> MidiInput:
        """Open a MIDI input port.

        Args:
            in_port_name: The name of the MIDI port to open.

        Returns:
            A new MidiInput instance connected to the specified port.
        """
        queue: SimpleQueue[Message] = SimpleQueue()
        in_port = mido.open_input(in_port_name, callback=queue.put_nowait)
        return cls(in_port_name=in_port_name, in_port=in_port, queue=queue)

    def __init__(
        self,
        in_port_name: str,
        in_port: BaseInput,
        queue: "SimpleQueue[Message]",
    ) -> None:
        """Initialize the MIDI input.

        Args:
            in_port_name: The name of the input port.
            in_port: The mido input port object.
            queue: The message queue for incoming messages.
        """
        self._in_port_name = in_port_name
        self._in_port = in_port
        self._queue = queue

    def close(self) -> None:
        """Close the MIDI input port."""
        self._in_port.close()

    def recv_msg(self) -> FrozenMessage:
        """Receive the next message from the queue.

        This method blocks until a message is available.

        Returns:
            The next MIDI message as a FrozenMessage.
        """
        mut_msg = self._queue.get()
        msg = freeze_message(mut_msg)
        logging.debug("Received message from %s: %s", self._in_port_name, msg)
        return msg


class MidiOutput(MidiSink, Resettable, Closeable):
    """MIDI output connection with optional rate limiting.

    This class manages a MIDI output port and provides rate limiting
    to prevent flooding MIDI devices with too many messages.
    """

    @classmethod
    def open(
        cls, out_port_name: str, virtual: bool = False, delay: Optional[float] = None
    ) -> MidiOutput:
        """Open a MIDI output port.

        Args:
            out_port_name: The name of the MIDI port to open.
            virtual: Whether to create a virtual MIDI port.
            delay: Optional minimum delay between messages in seconds.

        Returns:
            A new MidiOutput instance connected to the specified port.
        """
        out_port = mido.open_output(out_port_name, virtual=virtual)
        return cls(out_port_name=out_port_name, out_port=out_port, delay=delay)

    def __init__(
        self, out_port_name: str, out_port: BaseOutput, delay: Optional[float]
    ) -> None:
        """Initialize the MIDI output.

        Args:
            out_port_name: The name of the output port.
            out_port: The mido output port object.
            delay: Optional minimum delay between messages in seconds.
        """
        self._out_port_name = out_port_name
        self._out_port = out_port
        self._delay = delay
        self._last_sent = 0.0

    def reset(self) -> None:
        """Reset the MIDI output port."""
        self._out_port.reset()

    def close(self) -> None:
        """Close the MIDI output port."""
        self._out_port.close()

    def send_msg(self, msg: FrozenMessage) -> None:
        """Send a MIDI message with optional rate limiting.

        Args:
            msg: The MIDI message to send.
        """
        if self._delay is not None:
            now = time.monotonic()
            lim = self._last_sent + self._delay
            diff = lim - now
            if diff > 0:
                time.sleep(diff)
                self._last_sent = lim
            else:
                self._last_sent = now
        logging.debug("Sending message to %s: %s", self._out_port_name, msg)
        self._out_port.send(msg)
