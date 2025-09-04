"""MIDI functionality for the minipat pattern system.

This module provides both high-level pattern-based MIDI functionality and low-level
MIDI message handling utilities.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from logging import Logger
from threading import Event
from time import sleep
from typing import Any, NewType, Optional, Tuple, cast, override

import mido
from mido.frozen import FrozenMessage, freeze_message

from centipede.actor import (
    Actor,
    ActorEnv,
    Callback,
    Mutex,
    Sender,
    System,
    Task,
    new_system,
)
from minipat.common import PosixTime, current_posix_time
from minipat.ev import EvHeap
from minipat.live import (
    BackendEvents,
    BackendMessage,
    BackendPlay,
    Instant,
    Orbit,
    Processor,
)
from minipat.parser import parse_pattern
from minipat.pat import Selected
from minipat.stream import MergeStrat, Stream, pat_stream
from spiny.common import Box
from spiny.dmap import DKey, DMap
from spiny.heapmap import PHeapMap
from spiny.seq import PSeq

# =============================================================================
# MIDI Value Types
# =============================================================================

Note = NewType("Note", int)
"""MIDI note number (0-127)"""

Vel = NewType("Vel", int)
"""MIDI velocity (0-127)"""

Channel = NewType("Channel", int)
"""MIDI channel (0-15)"""

Program = NewType("Program", int)
"""MIDI program number (0-127)"""


def _assert_midi_range(value: int, max_value: int, name: str) -> None:
    """Assert that a value is in valid MIDI range."""
    if not (0 <= value <= max_value):
        raise ValueError(f"{name} {value} out of range (0-{max_value})")


def make_note(value: int) -> Note:
    """Create a Note, validating range."""
    _assert_midi_range(value, 127, "Note")
    return Note(value)


def make_vel(value: int) -> Vel:
    """Create a Vel, validating range."""
    _assert_midi_range(value, 127, "Velocity")
    return Vel(value)


def make_channel(value: int) -> Channel:
    """Create a Channel, validating range."""
    _assert_midi_range(value, 15, "Channel")
    return Channel(value)


def make_program(value: int) -> Program:
    """Create a Program, validating range."""
    _assert_midi_range(value, 127, "Program")
    return Program(value)


# =============================================================================
# MIDI Message Construction
# =============================================================================


def msg_note_on(channel: Channel, note: Note, velocity: Vel) -> FrozenMessage:
    """Create a note-on MIDI message."""
    return FrozenMessage(
        "note_on", channel=int(channel), note=int(note), velocity=int(velocity)
    )


def msg_note_off(channel: Channel, note: Note) -> FrozenMessage:
    """Create a note-off MIDI message."""
    return FrozenMessage("note_off", channel=int(channel), note=int(note), velocity=0)


def msg_pc(channel: Channel, program: Program) -> FrozenMessage:
    """Create a program change MIDI message."""
    return FrozenMessage("program_change", channel=int(channel), program=int(program))


# =============================================================================
# MIDI Message Field Access
# =============================================================================


def get_msg_type(msg: FrozenMessage) -> Optional[str]:
    """Get the message type from a MIDI message.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The message type (e.g., 'note_on', 'note_off', 'program_change'),
        or None if the message has no type field.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> get_msg_type(msg)
        'note_on'
    """
    return getattr(msg, "type", None)


def has_msg_type(msg: FrozenMessage, msg_type: str) -> bool:
    """Check if a MIDI message has a specific type.

    Args:
        msg: The MIDI message to check.
        msg_type: The expected message type (e.g., 'note_on', 'note_off').

    Returns:
        True if the message has the specified type, False otherwise.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> has_msg_type(msg, 'note_on')
        True
        >>> has_msg_type(msg, 'note_off')
        False
    """
    return get_msg_type(msg) == msg_type


def get_channel(msg: FrozenMessage) -> int:
    """Get the channel from a MIDI message, defaulting to 0.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI channel (0-15), or 0 if the message has no channel field.

    Example:
        >>> msg = msg_note_on(make_channel(5), make_note(60), make_vel(64))
        >>> get_channel(msg)
        5
    """
    return getattr(msg, "channel", 0)


def opt_channel(msg: FrozenMessage) -> Optional[int]:
    """Get the channel from a MIDI message, returning None if not present.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI channel (0-15), or None if the message has no channel field.

    Example:
        >>> msg = msg_note_on(make_channel(5), make_note(60), make_vel(64))
        >>> opt_channel(msg)
        5
        >>> # For messages without channel
        >>> opt_channel(some_msg_without_channel)
        None
    """
    return getattr(msg, "channel", None)


def has_channel(msg: FrozenMessage) -> bool:
    """Check if a MIDI message has a channel field.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message has a channel field, False otherwise.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> has_channel(msg)
        True
    """
    return hasattr(msg, "channel")


def get_note(msg: FrozenMessage) -> int:
    """Get the note from a MIDI message, defaulting to 60 (Middle C).

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI note number (0-127), or 60 if the message has no note field.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(72), make_vel(64))
        >>> get_note(msg)
        72
    """
    return getattr(msg, "note", 60)


def opt_note(msg: FrozenMessage) -> Optional[int]:
    """Get the note from a MIDI message, returning None if not present.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI note number (0-127), or None if the message has no note field.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(72), make_vel(64))
        >>> opt_note(msg)
        72
        >>> # For messages without note
        >>> opt_note(some_pc_msg)
        None
    """
    return getattr(msg, "note", None)


def has_note(msg: FrozenMessage) -> bool:
    """Check if a MIDI message has a note field.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message has a note field, False otherwise.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> has_note(msg)
        True
        >>> pc_msg = msg_pc(make_channel(0), make_program(1))
        >>> has_note(pc_msg)
        False
    """
    return hasattr(msg, "note")


def get_velocity(msg: FrozenMessage) -> int:
    """Get the velocity from a MIDI message, defaulting to 64.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI velocity (0-127), or 64 if the message has no velocity field.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(100))
        >>> get_velocity(msg)
        100
    """
    return getattr(msg, "velocity", 64)


def opt_velocity(msg: FrozenMessage) -> Optional[int]:
    """Get the velocity from a MIDI message, returning None if not present.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI velocity (0-127), or None if the message has no velocity field.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(100))
        >>> opt_velocity(msg)
        100
        >>> pc_msg = msg_pc(make_channel(0), make_program(1))
        >>> opt_velocity(pc_msg)
        None
    """
    return getattr(msg, "velocity", None)


def has_velocity(msg: FrozenMessage) -> bool:
    """Check if a MIDI message has a velocity field.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message has a velocity field, False otherwise.

    Example:
        >>> msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> has_velocity(msg)
        True
        >>> pc_msg = msg_pc(make_channel(0), make_program(1))
        >>> has_velocity(pc_msg)
        False
    """
    return hasattr(msg, "velocity")


def get_program(msg: FrozenMessage) -> int:
    """Get the program from a MIDI message, defaulting to 0.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI program number (0-127), or 0 if the message has no program field.

    Example:
        >>> msg = msg_pc(make_channel(0), make_program(42))
        >>> get_program(msg)
        42
    """
    return getattr(msg, "program", 0)


def opt_program(msg: FrozenMessage) -> Optional[int]:
    """Get the program from a MIDI message, returning None if not present.

    Args:
        msg: The MIDI message to inspect.

    Returns:
        The MIDI program number (0-127), or None if the message has no program field.

    Example:
        >>> msg = msg_pc(make_channel(0), make_program(42))
        >>> opt_program(msg)
        42
        >>> note_msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> opt_program(note_msg)
        None
    """
    return getattr(msg, "program", None)


def has_program(msg: FrozenMessage) -> bool:
    """Check if a MIDI message has a program field.

    Args:
        msg: The MIDI message to check.

    Returns:
        True if the message has a program field, False otherwise.

    Example:
        >>> pc_msg = msg_pc(make_channel(0), make_program(1))
        >>> has_program(pc_msg)
        True
        >>> note_msg = msg_note_on(make_channel(0), make_note(60), make_vel(64))
        >>> has_program(note_msg)
        False
    """
    return hasattr(msg, "program")


# =============================================================================
# Timed Messages
# =============================================================================


@dataclass(frozen=True)
class TimedMessage:
    """A timed message with POSIX timestamp."""

    time: PosixTime
    """Timestamp when the message should be sent (POSIX time)."""

    message: FrozenMessage
    """The frozen MIDI message."""


# =============================================================================
# MIDI Message Heap (Low-Level Utilities)
# =============================================================================

type MsgHeap = PHeapMap[PosixTime, TimedMessage]
"""A priority queue of timed MIDI messages ordered by time."""


def mh_empty() -> MsgHeap:
    """Create an empty message heap."""
    return PHeapMap.empty()


def mh_push_note(
    mh: MsgHeap,
    start: PosixTime,
    end: PosixTime,
    channel: Channel,
    note: Note,
    velocity: Vel,
) -> MsgHeap:
    """Add note-on and note-off messages to the heap."""
    if start > end:
        raise ValueError("Note start time must be <= end time")
    m1 = msg_note_on(channel=channel, note=note, velocity=velocity)
    m2 = msg_note_off(channel=channel, note=note)
    tm1 = TimedMessage(start, m1)
    tm2 = TimedMessage(end, m2)
    return mh.insert(start, tm1).insert(end, tm2)


def mh_push_pc(
    mh: MsgHeap, time: PosixTime, channel: Channel, program: Program
) -> MsgHeap:
    """Add a program change message to the heap."""
    m = msg_pc(channel=channel, program=program)
    tm = TimedMessage(time, m)
    return mh.insert(time, tm)


def mh_pop(mh: MsgHeap) -> Tuple[Optional[TimedMessage], MsgHeap]:
    """Pop the earliest message from the heap."""
    x = mh.find_min()
    if x is None:
        return (None, mh)
    else:
        k, v, mh2 = x
        return (v, mh2)


def mh_seek_pop(mh: MsgHeap, time: PosixTime) -> Tuple[Optional[TimedMessage], MsgHeap]:
    """Pop all messages before the given time, returning the first at or after time."""
    while True:
        x = mh_pop(mh)
        if x[0] is None or x[0].time >= time:
            return x
        else:
            mh = x[1]


# =============================================================================
# Thread-Safe Message Heap
# =============================================================================


class ParMsgHeap:
    """Thread-safe wrapper around MsgHeap."""

    def __init__(self):
        self._mutex = Mutex(Box(mh_empty()))

    def push_note(
        self,
        start: PosixTime,
        end: PosixTime,
        channel: Channel,
        note: Note,
        velocity: Vel,
    ) -> None:
        """Add a note to the heap (thread-safe)."""
        with self._mutex as box:
            box.value = mh_push_note(
                mh=box.value,
                start=start,
                end=end,
                channel=channel,
                note=note,
                velocity=velocity,
            )

    def push_pc(self, time: PosixTime, channel: Channel, program: Program) -> None:
        """Add a program change to the heap (thread-safe)."""
        with self._mutex as box:
            box.value = mh_push_pc(
                mh=box.value, time=time, channel=channel, program=program
            )

    def pop(self) -> Optional[TimedMessage]:
        """Pop the earliest message (thread-safe)."""
        with self._mutex as box:
            msg, mh2 = mh_pop(mh=box.value)
            box.value = mh2
            return msg

    def seek_pop(self, time: PosixTime) -> Optional[TimedMessage]:
        """Pop messages until reaching the given time (thread-safe)."""
        with self._mutex as box:
            msg, mh2 = mh_seek_pop(mh=box.value, time=time)
            box.value = mh2
            return msg


class MidiSenderTask(Task):
    """Background task that sends scheduled MIDI messages at the correct time.

    Runs in a separate thread and continuously pops messages from a shared
    ParMsgHeap, sending them to a Mutex-protected MIDI output at the
    appropriate timestamps.
    """

    def __init__(self, heap: ParMsgHeap, output_mutex: Mutex[mido.ports.BaseOutput]):
        """Initialize the MIDI sender task.

        Args:
            heap: Shared message heap for scheduled messages
            output_mutex: Mutex-protected MIDI output port (can be None if disabled)
        """
        self._heap = heap
        self._output_mutex = output_mutex

    @override
    def run(self, logger: Logger, halt: Event) -> None:
        """Execute the task using the actor system's threading model.

        Args:
            logger: Logger for the task to use.
            halt: Event that will be set when the task should stop.
        """
        logger.debug("MIDI sender task started")
        while not halt.is_set():
            try:
                current_time = current_posix_time()

                # Get the next message that's ready to send
                timed_msg = self._heap.seek_pop(current_time)

                if timed_msg is not None:
                    # Send the message if we have an output
                    send_error: Optional[Exception] = None

                    with self._output_mutex as output_port:
                        try:
                            output_port.send(timed_msg.message)
                        except Exception as e:
                            send_error = e

                    # Log outside of mutex
                    if send_error is None:
                        logger.debug(
                            "Sent scheduled MIDI message: %s", timed_msg.message
                        )
                    else:
                        logger.error(
                            "Error sending scheduled MIDI message: %s", send_error
                        )
                else:
                    # No messages ready, sleep briefly
                    # TODO these sleeps should be halt.wait() with interval
                    # based on current cps and gens per cycle - figure out how to
                    # get that info into here
                    sleep(0.001)  # 1ms sleep to avoid busy waiting

            except Exception as e:
                logger.error("Unexpected error in MIDI sender task: %s", e)
                sleep(0.01)  # Longer sleep on error

        logger.debug("MIDI sender task stopped")


# =============================================================================
# Pattern System Integration
# =============================================================================


class MidiDom:
    """Domain marker for MIDI attributes."""

    pass


type MidiAttrs = DMap[MidiDom]
"""MIDI attribute map."""


class MidiKey[V](DKey[MidiDom, V]):
    """Base class for MIDI attribute keys."""

    pass


class NoteKey(MidiKey[Note]):
    """Key for note values in MIDI attributes."""

    pass


class VelKey(MidiKey[Vel]):
    """Key for velocity values in MIDI attributes."""

    pass


# =============================================================================
# Pattern Parsing and Rendering
# =============================================================================


class Selector[V](metaclass=ABCMeta):
    """Abstract base class for parsing and rendering pattern values.

    A Selector handles the bidirectional conversion between string representations
    in patterns and strongly-typed values. This enables the pattern system to work
    with domain-specific types like MIDI notes and velocities while maintaining
    readable string syntax in patterns.

    The pattern system uses Selected[str] values which contain both the string
    value and optional selection context (for advanced pattern features).

    Type Parameters:
        V: The value type this selector handles (e.g., Note, Vel)

    Example:
        A NoteSelector might parse "c4" -> Note(60) and render Note(60) -> "c4"
    """

    @classmethod
    @abstractmethod
    def parse(cls, sel: Selected[str]) -> V:
        """Parse a string pattern value into a strongly-typed value.

        Args:
            sel: A Selected string containing the pattern value and optional context

        Returns:
            The parsed value of type V

        Raises:
            ValueError: If the string cannot be parsed into a valid value
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def render(cls, value: V) -> Selected[str]:
        """Render a strongly-typed value back to a string pattern representation.

        Args:
            value: The strongly-typed value to render

        Returns:
            A Selected string representation suitable for patterns
        """
        raise NotImplementedError


class NoteNumSelector(Selector[Note]):
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
    @classmethod
    def parse(cls, sel: Selected[str]) -> Note:
        try:
            note_num = int(sel.value)
            return make_note(note_num)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid note number: {sel.value}") from e

    @override
    @classmethod
    def render(cls, value: Note) -> Selected[str]:
        return Selected(str(value), None)


class NoteNameSelector(Selector[Note]):
    """Selector for parsing musical note names with octave numbers.

    Converts standard musical notation to MIDI note numbers. Supports
    note names (C, D, E, F, G, A, B) with optional sharps (#) or flats (b)
    and octave numbers.

    Note naming follows standard MIDI convention where:
    - C4 = 60 (Middle C)
    - Octaves range from -1 to 9
    - Each octave starts at C

    Examples:
        "c4"  -> Note(60)  # Middle C
        "c#4" -> Note(61)  # C sharp 4
        "db4" -> Note(61)  # D flat 4 (enharmonic equivalent)
        "a0"  -> Note(21)  # A0 (lowest A on piano)
        "c8"  -> Note(108) # C8 (high C)

    Parsing is case-insensitive: "C4", "c4", and "C4" are all equivalent.

    Raises:
        ValueError: If the note name format is invalid, octave is missing,
                   or the resulting MIDI note is outside the valid range (0-127)
    """

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Note:
        # Parse note names like "c4" (middle C = 60), "d#4", "eb3", etc.
        note_str = sel.value.lower()

        # Note name to semitone mapping (C is 0)
        note_map = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

        # Parse note name and octave
        if len(note_str) < 2:
            raise ValueError(f"Invalid note name: {sel.value}")

        # Get base note
        base_note = note_str[0]
        if base_note not in note_map:
            raise ValueError(f"Invalid note name: {sel.value}")

        semitone = note_map[base_note]
        pos = 1

        # Check for sharp or flat
        if pos < len(note_str):
            if note_str[pos] == "#":
                semitone += 1
                pos += 1
            elif note_str[pos] == "b":
                semitone -= 1
                pos += 1

        # Get octave
        try:
            octave = int(note_str[pos:])
        except ValueError:
            raise ValueError(f"Invalid octave in note: {sel.value}")

        # Calculate MIDI note number (C4 = 60, so octave 4 starts at 60-12=48 for C)
        # MIDI note = (octave + 1) * 12 + semitone
        midi_note = (octave + 1) * 12 + semitone

        return make_note(midi_note)

    @override
    @classmethod
    def render(cls, value: Note) -> Selected[str]:
        # Convert MIDI note back to note name
        note_num = int(value)
        octave = (note_num // 12) - 1
        semitone = note_num % 12

        # Semitone to note name mapping (using sharps)
        note_names = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
        note_name = note_names[semitone] + str(octave)

        return Selected(note_name, None)


class VelSelector(Selector[Vel]):
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
    @classmethod
    def parse(cls, sel: Selected[str]) -> Vel:
        try:
            vel_num = int(sel.value)
            return make_vel(vel_num)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid velocity: {sel.value}") from e

    @override
    @classmethod
    def render(cls, value: Vel) -> Selected[str]:
        return Selected(str(value), None)


# =============================================================================
# Pattern Stream Functions
# =============================================================================


def _convert_midinote(sel: Selected[str]) -> MidiAttrs:
    """Convert numeric note to MIDI attributes."""
    note = NoteNumSelector.parse(sel)
    return DMap.singleton(NoteKey(), note)


def _convert_note(sel: Selected[str]) -> MidiAttrs:
    """Convert note name to MIDI attributes."""
    note = NoteNameSelector.parse(sel)
    return DMap.singleton(NoteKey(), note)


def _convert_vel(sel: Selected[str]) -> MidiAttrs:
    """Convert velocity to MIDI attributes."""
    velocity = VelSelector.parse(sel)
    return DMap.singleton(VelKey(), velocity)


def midinote(s: str) -> Stream[MidiAttrs]:
    """Create stream from numeric MIDI notes.

    Parses a pattern string containing numeric MIDI note values (0-127)
    and creates a stream of MIDI attributes.

    Examples:
        midinote("60 62 64")     # C4, D4, E4 (C major triad)
        midinote("36 ~ 42")      # Kick, rest, snare pattern
        midinote("[60,64,67]")   # C major chord (simultaneous)
    """
    return pat_stream(parse_pattern(s).map(_convert_midinote))


def note(s: str) -> Stream[MidiAttrs]:
    """Create stream from note names.

    Parses a pattern string containing musical note names with octaves
    and creates a stream of MIDI attributes.

    Examples:
        note("c4 d4 e4")         # C major scale fragment
        note("c4 ~ g4")          # C4, rest, G4
        note("[c4,e4,g4]")       # C major chord (simultaneous)
        note("c#4 db5 f4")       # Mixed sharps and flats
    """
    return pat_stream(parse_pattern(s).map(_convert_note))


def vel(s: str) -> Stream[MidiAttrs]:
    """Create stream from velocity values.

    Parses a pattern string containing MIDI velocity values (0-127)
    and creates a stream of MIDI attributes for controlling note dynamics.

    Examples:
        vel("64 80 100")         # Medium, loud, very loud
        vel("127 0 64")          # Loud, silent, medium
        vel("100*8")             # Repeat loud velocity 8 times
    """
    return pat_stream(parse_pattern(s).map(_convert_vel))


def _merge_attrs(x: MidiAttrs, y: MidiAttrs) -> MidiAttrs:
    """Merge two MIDI attribute maps."""
    return x.merge(y)


def combine(*ss: Stream[MidiAttrs]) -> Stream[MidiAttrs]:
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
        notes = note("c4 d4 e4")
        velocities = vel("64 80 100")
        combined = combine(notes, velocities)

        # Layer multiple attributes
        notes = note("c4 ~ g4")
        velocities = vel("100 ~ 80")
        channels = ...  # hypothetical channel stream
        result = combine(notes, velocities, channels)
    """
    if len(ss):
        acc = ss[0]
        for el in ss[1:]:
            acc = acc.apply(MergeStrat.Inner, _merge_attrs, el)
        return acc
    else:
        return Stream.silence()


# =============================================================================
# MIDI Processor (Pattern System to MIDI Messages)
# =============================================================================


class MidiProcessor(Processor[MidiAttrs, TimedMessage]):
    """Processor that converts MidiAttrs to MIDI messages."""

    def __init__(self, default_velocity: Optional[Vel] = None):
        """Initialize the MIDI processor.

        Args:
            default_velocity: Default velocity to use when not specified
        """
        self.default_velocity = (
            default_velocity if default_velocity is not None else make_vel(64)
        )

    @override
    def process(
        self, instant: Instant, orbit: Orbit, events: EvHeap[MidiAttrs]
    ) -> PSeq[TimedMessage]:
        """Process MIDI events into timed MIDI messages."""
        timed_messages = []

        # Use orbit as MIDI channel (clamp to 0-15 range)
        channel = make_channel(max(0, min(15, int(orbit))))

        for span, ev in events:
            # Extract MIDI attributes
            note_num = ev.val.lookup(NoteKey())
            velocity = ev.val.lookup(VelKey())

            # Use defaults if attributes are missing
            note_raw = note_num if note_num is not None else make_note(60)  # Middle C
            vel_raw = velocity if velocity is not None else self.default_velocity

            # Clamp values to valid MIDI range
            note = make_note(max(0, min(127, int(note_raw))))
            vel = make_vel(max(0, min(127, int(vel_raw))))

            # Only send note_on if active start is whole start (or whole is empty)
            send_note_on = span.whole is None or span.active.start == span.whole.start
            if send_note_on:
                # Calculate timestamp for the start of the event
                timestamp = PosixTime(
                    instant.posix_start
                    + (float(span.active.start) / float(instant.cps))
                )

                # Create note on message using helper
                note_on_msg = msg_note_on(channel=channel, note=note, velocity=vel)
                timed_messages.append(TimedMessage(timestamp, note_on_msg))

            # Only send note_off if active end is whole end (or whole is empty)
            send_note_off = span.whole is None or span.active.end == span.whole.end
            if send_note_off:
                # Create note off message (at end of span)
                note_off_timestamp = PosixTime(
                    instant.posix_start + (float(span.active.end) / float(instant.cps))
                )
                note_off_msg = msg_note_off(channel=channel, note=note)
                timed_messages.append(TimedMessage(note_off_timestamp, note_off_msg))

        return PSeq.mk(timed_messages)


# =============================================================================
# MIDI Actor (Message Output)
# =============================================================================


class MidiActor(Actor[BackendMessage[TimedMessage]]):
    """Actor that queues MIDI messages for scheduled sending.

    Instead of sending MIDI messages immediately, this actor queues them
    in a shared ParMsgHeap where a MidiSenderTask will send them at the
    appropriate time.
    """

    def __init__(self, heap: ParMsgHeap, output_mutex: Mutex[mido.ports.BaseOutput]):
        """Initialize the MIDI actor.

        Args:
            heap: Shared message heap for scheduling messages
            output_mutex: Mutex-protected MIDI output port
        """
        self._heap = heap
        self._output_mutex = output_mutex
        self._playing = False

    @override
    def on_stop(self, logger: Logger) -> None:
        """Reset the MIDI output when stopping."""
        reset_error: Optional[Exception] = None

        with self._output_mutex as output_port:
            try:
                output_port.reset()
            except Exception as e:
                reset_error = e

        # Log outside of mutex
        if reset_error is None:
            logger.debug("Reset MIDI output port")
        else:
            logger.error("Error resetting MIDI output: %s", reset_error)

    @override
    def on_message(self, env: ActorEnv, value: BackendMessage[TimedMessage]) -> None:
        match value:
            case BackendPlay(playing):
                self._playing = playing
                if playing:
                    env.logger.info("MIDI: Playing")
                else:
                    env.logger.info("MIDI: Pausing")
                    # Reset output when stopping
                    reset_error: Optional[Exception] = None

                    with self._output_mutex as output_port:
                        try:
                            output_port.reset()
                        except Exception as e:
                            reset_error = e

                    # Log outside of mutex
                    if reset_error is not None:
                        env.logger.error("Error resetting MIDI output: %s", reset_error)
            case BackendEvents(messages):
                if self._playing:
                    self._queue_messages(env, messages)
                else:
                    env.logger.debug("MIDI: Ignoring events while stopped")
            case _:
                env.logger.warning("Unknown MIDI message type: %s", type(value))

    def _queue_messages(self, env: ActorEnv, messages: PSeq[TimedMessage]) -> None:
        """Queue messages in the shared heap for scheduled sending."""
        for timed_message in messages:
            try:
                # Extract note information to use heap's push_note method
                msg = timed_message.message
                if has_msg_type(msg, "note_on"):
                    # Find corresponding note_off message
                    note_off_time = None

                    # Look ahead for matching note_off
                    for other_msg in messages:
                        other = other_msg.message
                        if (
                            has_msg_type(other, "note_off")
                            and opt_channel(other) == opt_channel(msg)
                            and opt_note(other) == opt_note(msg)
                            and other_msg.time > timed_message.time
                        ):
                            note_off_time = other_msg.time
                            break

                    if note_off_time is not None:
                        # Use heap's push_note method for proper note on/off pairing
                        try:
                            channel = make_channel(get_channel(msg))
                            note = make_note(get_note(msg))
                            velocity = make_vel(get_velocity(msg))

                            self._heap.push_note(
                                start=timed_message.time,
                                end=note_off_time,
                                channel=channel,
                                note=note,
                                velocity=velocity,
                            )
                            env.logger.debug(
                                "Queued MIDI note: %s at %s", note, timed_message.time
                            )
                        except Exception as e:
                            env.logger.error("Error queuing MIDI note: %s", e)

                # Skip note_off messages as they're handled by push_note above
                elif has_msg_type(msg, "note_off"):
                    continue

                # Handle other message types (program change, etc.)
                elif has_msg_type(msg, "program_change"):
                    try:
                        channel = make_channel(get_channel(msg))
                        program = make_program(get_program(msg))
                        self._heap.push_pc(timed_message.time, channel, program)
                        env.logger.debug(
                            "Queued MIDI program change: %s at %s",
                            program,
                            timed_message.time,
                        )
                    except Exception as e:
                        env.logger.error("Error queuing MIDI program change: %s", e)

            except Exception as e:
                env.logger.error("Error processing MIDI message: %s", e)


# =============================================================================
# MIDI Supervisor Actor
# =============================================================================


class MidiSupervisor(Actor[BackendMessage[TimedMessage]]):
    """Supervisory actor that manages MidiActor and MidiSenderTask lifetimes together.

    This actor coordinates the startup and shutdown of both the MIDI message processing
    actor and the background sender task, ensuring they work together as a cohesive unit.
    When the supervisor starts, it starts the sender task. When it stops, it properly
    shuts down both components in the correct order.
    """

    def __init__(self, output: mido.ports.BaseOutput):
        """Initialize the MIDI supervisor.

        Args:
            output: MIDI output port, or None to disable actual output
        """
        # Create shared components
        self._heap = ParMsgHeap()
        self._output_mutex = Mutex(output)

        # Create the MIDI actor (handles message processing)
        self._midi_actor = MidiActor(self._heap, self._output_mutex)

        # Create the sender task (handles scheduled message sending)
        self._sender_task = MidiSenderTask(self._heap, self._output_mutex)

    @override
    def on_start(self, env: ActorEnv) -> None:
        """Start the MIDI sender task when the supervisor starts."""
        # Task is now managed by the actor system if spawned properly
        env.logger.debug("MIDI supervisor started")

    @override
    def on_stop(self, logger: Logger) -> None:
        """Clean up when the supervisor stops."""
        try:
            # Let the MIDI actor clean up (reset output port)
            self._midi_actor.on_stop(logger)
            logger.debug("MIDI supervisor completed cleanup")
        except Exception as e:
            logger.error("Error during MIDI supervisor shutdown: %s", e)

    @override
    def on_message(self, env: ActorEnv, value: BackendMessage[TimedMessage]) -> None:
        """Forward all messages to the managed MIDI actor."""
        self._midi_actor.on_message(env, value)

    def get_midi_actor(self) -> MidiActor:
        """Get the managed MIDI actor for direct access if needed.

        Returns:
            The MidiActor instance managed by this supervisor.
        """
        return self._midi_actor

    def get_sender_task(self) -> MidiSenderTask:
        """Get the managed sender task for direct access if needed.

        Returns:
            The MidiSenderTask instance managed by this supervisor.
        """
        return self._sender_task


# =============================================================================
# Low-Level Actor Utilities
# =============================================================================


class SendActor(Actor[FrozenMessage]):
    """Actor that sends raw MIDI messages to an output port."""

    def __init__(self, port: mido.ports.BaseOutput):
        self._port = port

    @override
    def on_message(self, env: ActorEnv, value: FrozenMessage) -> None:
        self._port.send(value)

    @override
    def on_stop(self, logger: Logger) -> None:
        self._port.close()


def _recv_cb(sender: Sender[FrozenMessage], msg: Any) -> None:
    """Callback for receiving MIDI messages."""
    fmsg = cast(FrozenMessage, freeze_message(msg))
    sender.send(fmsg)


class RecvCallback(Callback[FrozenMessage]):
    """Callback for receiving MIDI messages from an input port."""

    def __init__(self, port: mido.ports.BaseInput):
        self._port = port

    @override
    def register(self, sender: Sender[FrozenMessage]) -> None:
        self._port.callback = partial(_recv_cb, sender)  # pyright: ignore

    @override
    def unregister(self) -> None:
        self._port.callback = None  # pyright: ignore


def echo_system() -> System:
    """Create a system that echoes MIDI input to output."""
    system = new_system("echo")
    out_port = mido.open_output(name="virt_out", virtual=True)  # pyright: ignore
    in_port = mido.open_input(name="virt_in", virtual=True)  # pyright: ignore
    send_actor = SendActor(out_port)
    recv_actor = RecvCallback(in_port).produce("send", send_actor)
    system.spawn_actor("recv", recv_actor)
    return system


def create_midi_system(
    output: Optional[mido.ports.BaseOutput] = None,
) -> Tuple[MidiActor, MidiSenderTask]:
    """Create a complete MIDI system with scheduling support.

    Creates a MidiActor that queues messages and a MidiSenderTask that sends them
    at the appropriate time. Both components share a ParMsgHeap for message scheduling
    and a Mutex-protected output port.

    Args:
        output: MIDI output port, or None to disable output

    Returns:
        A tuple of (MidiActor, MidiSenderTask). The MidiSenderTask should be
        started with start() and stopped with stop() as needed.

    Example:
        # Create system with virtual output
        output_port = mido.open_output(virtual=True)
        midi_actor, sender_task = create_midi_system(output_port)

        # Start the sender task
        sender_task.start()

        # Use midi_actor in your actor system...

        # Clean up
        sender_task.stop()
        output_port.close()
    """
    # Create shared components
    heap = ParMsgHeap()
    output_mutex = Mutex(output)

    # Create actor and sender task
    midi_actor = MidiActor(heap, output_mutex)
    sender_task = MidiSenderTask(heap, output_mutex)

    return midi_actor, sender_task
