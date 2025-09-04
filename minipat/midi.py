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
from typing import Any, Iterable, NewType, Optional, Tuple, cast, override

import mido
from mido.frozen import FrozenMessage, freeze_message

from bad_actor import (
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

Velocity = NewType("Velocity", int)
"""MIDI velocity (0-127)"""

Channel = NewType("Channel", int)
"""MIDI channel (0-15)"""

Program = NewType("Program", int)
"""MIDI program number (0-127)"""

ControlNum = NewType("ControlNum", int)
"""MIDI control number (0-127)"""

ControlVal = NewType("ControlVal", int)
"""MIDI control value (0-127)"""


def _assert_midi_range(value: int, max_value: int, name: str) -> None:
    """Assert that a value is in valid MIDI range."""
    if not (0 <= value <= max_value):
        raise ValueError(f"{name} {value} out of range (0-{max_value})")


# =============================================================================
# MIDI Message Construction
# =============================================================================


def msg_note_on(channel: Channel, note: Note, velocity: Velocity) -> FrozenMessage:
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


def msg_cc(channel: Channel, control: ControlNum, value: ControlVal) -> FrozenMessage:
    """Create a program change MIDI message."""
    return FrozenMessage(
        "control_change", channel=int(channel), control=int(control), value=int(value)
    )


# =============================================================================
# MIDI Message Field Access
# =============================================================================


class MessageField[T, U](metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        """Return whether the message has field"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get(cls, msg: FrozenMessage) -> T:
        """Get field value or raise AttributeError"""
        raise NotImplementedError

    @classmethod
    def opt(cls, msg: FrozenMessage) -> Optional[T]:
        """Get field value or return None"""
        return cls.get(msg) if cls.exists(msg) else None

    @classmethod
    @abstractmethod
    def mk(cls, raw_val: U) -> T:
        """Construct value or raise AttributeError if invalid"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unmk(cls, val: T) -> U:
        """Deconstruct value"""
        raise NotImplementedError


class MsgTypeField(MessageField[str, str]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "type")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> str:
        return cast(str, getattr(msg, "type"))

    @override
    @classmethod
    def mk(cls, raw_val: str) -> str:
        # No validation needed for message type strings
        return raw_val

    @override
    @classmethod
    def unmk(cls, val: str) -> str:
        return val


class ChannelField(MessageField[Channel, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "channel")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> Channel:
        return Channel(getattr(msg, "channel"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> Channel:
        _assert_midi_range(raw_val, 15, "channel")
        return Channel(raw_val)

    @override
    @classmethod
    def unmk(cls, val: Channel) -> int:
        return int(val)


class NoteField(MessageField[Note, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "note")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> Note:
        return Note(getattr(msg, "note"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> Note:
        _assert_midi_range(raw_val, 127, "note")
        return Note(raw_val)

    @override
    @classmethod
    def unmk(cls, val: Note) -> int:
        return int(val)


class VelocityField(MessageField[Velocity, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "velocity")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> Velocity:
        return Velocity(getattr(msg, "velocity"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> Velocity:
        _assert_midi_range(raw_val, 127, "velocity")
        return Velocity(raw_val)

    @override
    @classmethod
    def unmk(cls, val: Velocity) -> int:
        return int(val)


class ProgramField(MessageField[Program, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "program")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> Program:
        return Program(getattr(msg, "program"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> Program:
        _assert_midi_range(raw_val, 127, "program")
        return Program(raw_val)

    @override
    @classmethod
    def unmk(cls, val: Program) -> int:
        return int(val)


class ControlField(MessageField[ControlNum, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "control")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> ControlNum:
        return ControlNum(getattr(msg, "control"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> ControlNum:
        _assert_midi_range(raw_val, 127, "control")
        return ControlNum(raw_val)

    @override
    @classmethod
    def unmk(cls, val: ControlNum) -> int:
        return int(val)


class ValueField(MessageField[ControlVal, int]):
    @override
    @classmethod
    def exists(cls, msg: FrozenMessage) -> bool:
        return hasattr(msg, "value")

    @override
    @classmethod
    def get(cls, msg: FrozenMessage) -> ControlVal:
        return ControlVal(getattr(msg, "value"))

    @override
    @classmethod
    def mk(cls, raw_val: int) -> ControlVal:
        _assert_midi_range(raw_val, 127, "value")
        return ControlVal(raw_val)

    @override
    @classmethod
    def unmk(cls, val: ControlVal) -> int:
        return int(val)


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


type MsgHeap = PHeapMap[PosixTime, TimedMessage]
"""A priority queue of timed MIDI messages ordered by time."""


def mh_empty() -> MsgHeap:
    """Create an empty message heap."""
    return PHeapMap.empty()


def mh_push(mh: MsgHeap, tm: TimedMessage) -> MsgHeap:
    return mh.insert(tm.time, tm)


def mh_pop(mh: MsgHeap) -> Tuple[Optional[TimedMessage], MsgHeap]:
    """Pop the earliest message from the heap."""
    x = mh.find_min()
    if x is None:
        return (None, mh)
    else:
        _, v, mh2 = x
        return (v, mh2)


def mh_seek_pop(mh: MsgHeap, time: PosixTime) -> Tuple[Optional[TimedMessage], MsgHeap]:
    """Pop all messages before the given time, returning the first at or after time."""
    while True:
        x = mh_pop(mh)
        if x[0] is None or x[0].time >= time:
            return x
        else:
            mh = x[1]


type ParMsgHeap = Mutex[Box[MsgHeap]]
"""A thread-safe mutable MsgHeap"""


def pmh_empty() -> ParMsgHeap:
    return Mutex(Box(mh_empty()))


def pmh_push(pmh: ParMsgHeap, tm: TimedMessage) -> None:
    with pmh as box:
        box.value = mh_push(box.value, tm)


def pmh_push_all(pmh: ParMsgHeap, tms: Iterable[TimedMessage]) -> None:
    with pmh as box:
        for tm in tms:
            box.value = mh_push(box.value, tm)


def pmh_pop(pmh: ParMsgHeap) -> Optional[TimedMessage]:
    with pmh as box:
        res = mh_pop(box.value)
        if res is None:
            return None
        else:
            tm, mh = res
            box.value = mh
            return tm


def pmh_seek_pop(pmh: ParMsgHeap, time: PosixTime) -> Optional[TimedMessage]:
    with pmh as box:
        res = mh_seek_pop(box.value, time)
        if res is None:
            return None
        else:
            tm, mh = res
            box.value = mh
            return tm


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
                timed_msg = pmh_seek_pop(self._heap, current_time)

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


class VelKey(MidiKey[Velocity]):
    """Key for velocity values in MIDI attributes."""

    pass


# =============================================================================
# Pattern Parsing and Rendering
# =============================================================================


class ElemParser[V](metaclass=ABCMeta):
    """Abstract base class for parsing and rendering pattern values.

    An ElemParser handles the bidirectional conversion between string representations
    in patterns and strongly-typed values. This enables the pattern system to work
    with domain-specific types like MIDI notes and velocities while maintaining
    readable string syntax in patterns.

    The pattern system uses str values which ElemParser instances split apart.

    Type Parameters:
        V: The value type this parser handles (e.g., Note, Vel)

    Example:
        A NoteElemParser might parse "c4" -> Note(60) and render Note(60) -> "c4"
    """

    @classmethod
    @abstractmethod
    def parse(cls, s: str) -> V:
        """Parse a string pattern value into a strongly-typed value.

        Args:
            s: A string containing the pattern value

        Returns:
            The parsed value of type V

        Raises:
            ValueError: If the string cannot be parsed into a valid value
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def render(cls, value: V) -> str:
        """Render a strongly-typed value back to a string pattern representation.

        Args:
            value: The strongly-typed value to render

        Returns:
            A string representation suitable for patterns
        """
        raise NotImplementedError


class NoteNumElemParser(ElemParser[Note]):
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
    def parse(cls, s: str) -> Note:
        try:
            note_num = int(s)
            return NoteField.mk(note_num)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid note number: {s}") from e

    @override
    @classmethod
    def render(cls, value: Note) -> str:
        return str(value)


class NoteNameElemParser(ElemParser[Note]):
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
    def parse(cls, s: str) -> Note:
        # Parse note names like "c4" (middle C = 60), "d#4", "eb3", etc.
        note_str = s.lower()

        # Note name to semitone mapping (C is 0)
        note_map = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

        # Parse note name and octave
        if len(note_str) < 2:
            raise ValueError(f"Invalid note name: {s}")

        # Get base note
        base_note = note_str[0]
        if base_note not in note_map:
            raise ValueError(f"Invalid note name: {s}")

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
            raise ValueError(f"Invalid octave in note: {s}")

        # Calculate MIDI note number (C4 = 60, so octave 4 starts at 60-12=48 for C)
        # MIDI note = (octave + 1) * 12 + semitone
        midi_note = (octave + 1) * 12 + semitone

        return NoteField.mk(midi_note)

    @override
    @classmethod
    def render(cls, value: Note) -> str:
        # Convert MIDI note back to note name
        note_num = int(value)
        octave = (note_num // 12) - 1
        semitone = note_num % 12

        # Semitone to note name mapping (using sharps)
        note_names = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
        note_name = note_names[semitone] + str(octave)

        return note_name


class VelElemParser(ElemParser[Velocity]):
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
    def parse(cls, s: str) -> Velocity:
        try:
            vel_num = int(s)
            return VelocityField.mk(vel_num)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid velocity: {s}") from e

    @override
    @classmethod
    def render(cls, value: Velocity) -> str:
        return str(value)


# =============================================================================
# Pattern Stream Functions
# =============================================================================


def _convert_midinote(s: str) -> MidiAttrs:
    """Convert numeric note to MIDI attributes."""
    note = NoteNumElemParser.parse(s)
    return DMap.singleton(NoteKey(), note)


def _convert_note(s: str) -> MidiAttrs:
    """Convert note name to MIDI attributes."""
    note = NoteNameElemParser.parse(s)
    return DMap.singleton(NoteKey(), note)


def _convert_vel(s: str) -> MidiAttrs:
    """Convert velocity to MIDI attributes."""
    velocity = VelElemParser.parse(s)
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

    def __init__(self, default_velocity: Optional[Velocity] = None):
        """Initialize the MIDI processor.

        Args:
            default_velocity: Default velocity to use when not specified
        """
        self.default_velocity = (
            default_velocity if default_velocity is not None else VelocityField.mk(64)
        )

    @override
    def process(
        self, instant: Instant, orbit: Orbit, events: EvHeap[MidiAttrs]
    ) -> PSeq[TimedMessage]:
        """Process MIDI events into timed MIDI messages."""
        timed_messages = []

        # Use orbit as MIDI channel (clamp to 0-15 range)
        channel = ChannelField.mk(max(0, min(15, int(orbit))))

        for span, ev in events:
            # Extract MIDI attributes
            note_num = ev.val.lookup(NoteKey())
            velocity = ev.val.lookup(VelKey())

            # Use defaults if attributes are missing
            note_raw = (
                note_num if note_num is not None else NoteField.mk(60)
            )  # Middle C
            vel_raw = velocity if velocity is not None else self.default_velocity

            # Clamp values to valid MIDI range
            note = NoteField.mk(max(0, min(127, int(note_raw))))
            vel = VelocityField.mk(max(0, min(127, int(vel_raw))))

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
                    env.logger.debug("MIDI: Pushing %d messages", len(messages))
                    pmh_push_all(self._heap, messages)
                else:
                    env.logger.debug("MIDI: Ignoring events while stopped")
            case _:
                env.logger.warning("Unknown MIDI message type: %s", type(value))


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
    def deregister(self) -> None:
        self._port.callback = None  # pyright: ignore


def echo_system(in_port_name: str, out_port_name: str) -> System:
    """Create a system that echoes MIDI input to output."""
    system = new_system("echo")
    in_port = mido.open_input(name=in_port_name, virtual=True)  # pyright: ignore
    out_port = mido.open_output(name=out_port_name, virtual=True)  # pyright: ignore
    recv_callback = RecvCallback(in_port)
    send_actor = SendActor(out_port)
    system.spawn_callback("recv", send_actor, recv_callback)
    return system
