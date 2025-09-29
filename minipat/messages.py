from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional, cast, override

from mido import Message
from mido.frozen import FrozenMessage

from minipat.tab import TabInst
from minipat.time import PosixTime
from minipat.types import (
    Channel,
    ChordData,
    ControlNum,
    ControlVal,
    Note,
    Program,
    Velocity,
)
from spiny.dmap import DKey, DMap
from spiny.heap import PHeap
from spiny.seq import PSeq

# =============================================================================
# Constants
# =============================================================================

DEFAULT_VELOCITY = Velocity(64)
"""Default MIDI velocity when not specified"""

# =============================================================================
# Bundle Types
# =============================================================================


type MidoMessage = Message | FrozenMessage
type MidoBundle = FrozenMessage | PSeq[FrozenMessage]
type MidiBundle = MidiMessage | PSeq[MidiMessage]


# =============================================================================
# Helper Functions
# =============================================================================


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
    def key(cls) -> Optional[MidiKey[T]]:
        """If this field has a corresponding attrs key, return it"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def exists(cls, msg: MidoMessage) -> bool:
        """Return whether the message has field"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get(cls, msg: MidoMessage) -> T:
        """Get field value or raise AttributeError"""
        raise NotImplementedError

    @classmethod
    def opt(cls, msg: MidoMessage) -> Optional[T]:
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
    def key(cls) -> Optional[MidiKey[str]]:
        # Message type is not stored in attributes
        return None

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "type")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> str:
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
    def key(cls) -> Optional[MidiKey[Channel]]:
        return ChannelKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "channel")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> Channel:
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
    def key(cls) -> Optional[MidiKey[Note]]:
        return NoteKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "note")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> Note:
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
    def key(cls) -> Optional[MidiKey[Velocity]]:
        return VelocityKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "velocity")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> Velocity:
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
    def key(cls) -> Optional[MidiKey[Program]]:
        return ProgramKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "program")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> Program:
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
    def key(cls) -> Optional[MidiKey[ControlNum]]:
        return ControlNumKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "control")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> ControlNum:
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
    def key(cls) -> Optional[MidiKey[ControlVal]]:
        return ControlValKey()

    @override
    @classmethod
    def exists(cls, msg: MidoMessage) -> bool:
        return hasattr(msg, "value")

    @override
    @classmethod
    def get(cls, msg: MidoMessage) -> ControlVal:
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
# Attributes
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
    """Key for note in MIDI attributes."""

    pass


class VelocityKey(MidiKey[Velocity]):
    """Key for velocity in MIDI attributes."""

    pass


class ChannelKey(MidiKey[Channel]):
    """Key for channel in MIDI attributes."""

    pass


class ProgramKey(MidiKey[Program]):
    """Key for program in MIDI attributes."""

    pass


class ControlNumKey(MidiKey[ControlNum]):
    """Key for control number in MIDI attributes."""

    pass


class ControlValKey(MidiKey[ControlVal]):
    """Key for control value in MIDI attributes."""

    pass


class BundleKey(MidiKey[MidiBundle]):
    """Key for control value in MIDI attributes."""

    pass


class TabInstKey(MidiKey[TabInst]):
    """Key for tab instrument type in MIDI attributes."""

    pass


class TabStringKey(MidiKey[int]):
    """Key for tab string number (1-based) in MIDI attributes."""

    pass


class TabFretKey(MidiKey[int]):
    """Key for tab fret number in MIDI attributes."""

    pass


class ChordDataKey(MidiKey[ChordData]):
    """Key for chord data in MIDI attributes."""

    pass


# =============================================================================
# Typed messages
# =============================================================================


# sealed
class MidiMessage(metaclass=ABCMeta):
    @staticmethod
    def parse_midi(midi_msg: MidoMessage) -> Optional[MidiMessage]:
        """Parse a MIDI message into a typed message.

        Returns None if the message type is not supported.
        Raises ValueError if the message is malformed.
        """
        if not MsgTypeField.exists(midi_msg):
            return None

        msg_type = MsgTypeField.get(midi_msg)

        if msg_type == "note_on":
            if not ChannelField.exists(midi_msg) or not NoteField.exists(midi_msg):
                raise ValueError("note_on message missing required fields")
            channel = ChannelField.get(midi_msg)
            note = NoteField.get(midi_msg)
            velocity = VelocityField.opt(midi_msg)
            return NoteOnMessage(channel=channel, note=note, velocity=velocity)

        elif msg_type == "note_off":
            if not ChannelField.exists(midi_msg) or not NoteField.exists(midi_msg):
                raise ValueError("note_off message missing required fields")
            channel = ChannelField.get(midi_msg)
            note = NoteField.get(midi_msg)
            return NoteOffMessage(channel=channel, note=note)

        elif msg_type == "program_change":
            if not ChannelField.exists(midi_msg) or not ProgramField.exists(midi_msg):
                raise ValueError("program_change message missing required fields")
            channel = ChannelField.get(midi_msg)
            program = ProgramField.get(midi_msg)
            return ProgramMessage(channel=channel, program=program)

        elif msg_type == "control_change":
            if (
                not ChannelField.exists(midi_msg)
                or not ControlField.exists(midi_msg)
                or not ValueField.exists(midi_msg)
            ):
                raise ValueError("control_change message missing required fields")
            channel = ChannelField.get(midi_msg)
            control = ControlField.get(midi_msg)
            value = ValueField.get(midi_msg)
            return ControlMessage(channel=channel, control=control, value=value)

        else:
            # Unsupported message type
            return None

    @staticmethod
    def parse_attrs(attrs: MidiAttrs) -> list[MidiMessage]:
        """Extract MIDI messages from attributes.

        Parses MidiAttrs into multiple typed messages.
        Messages are returned in sorted order based on sort_key.

        Raises ValueError if incomplete message attributes are present.
        """
        messages: list[MidiMessage] = []

        # Extract attributes
        note = attrs.lookup(NoteKey())
        velocity = attrs.lookup(VelocityKey())
        channel = attrs.lookup(ChannelKey())
        program = attrs.lookup(ProgramKey())
        control_num = attrs.lookup(ControlNumKey())
        control_val = attrs.lookup(ControlValKey())
        bundle = attrs.lookup(BundleKey())

        # Add bundled messages first
        if bundle is not None:
            for bundled_msg in midi_bundle_iterator(bundle):
                messages.append(bundled_msg)

        # Channel is required for non-bundle messages
        has_non_bundle_messages = (
            note is not None
            or program is not None
            or (control_num is not None and control_val is not None)
        )

        if channel is None and has_non_bundle_messages:
            raise ValueError("Channel attribute is required for MIDI messages")

        # Check for program change message
        if program is not None and channel is not None:
            messages.append(ProgramMessage(channel=channel, program=program))

        # Check for control change message
        if control_num is not None and control_val is not None and channel is not None:
            messages.append(
                ControlMessage(channel=channel, control=control_num, value=control_val)
            )

        # Check for note message (creates both note_on and note_off)
        if note is not None and channel is not None:
            # Note on message
            messages.append(
                NoteOnMessage(channel=channel, note=note, velocity=velocity)
            )

        return messages

    @abstractmethod
    def render_midi(self, default_velocity: Velocity) -> FrozenMessage:
        raise NotImplementedError()

    @abstractmethod
    def render_attrs(self, default_velocity: Velocity) -> MidiAttrs:
        raise NotImplementedError()


@dataclass(frozen=True)
class NoteOnMessage(MidiMessage):
    channel: Channel
    note: Note
    velocity: Optional[Velocity]

    @override
    def render_midi(self, default_velocity: Velocity) -> FrozenMessage:
        velocity = self.velocity if self.velocity is not None else default_velocity
        return msg_note_on(self.channel, self.note, velocity)

    @override
    def render_attrs(self, default_velocity: Velocity) -> MidiAttrs:
        velocity = self.velocity if self.velocity is not None else default_velocity
        attrs = DMap.empty(MidiDom)
        attrs = attrs.put(ChannelKey(), self.channel)
        attrs = attrs.put(NoteKey(), self.note)
        attrs = attrs.put(VelocityKey(), velocity)
        return attrs


@dataclass(frozen=True)
class NoteOffMessage(MidiMessage):
    channel: Channel
    note: Note

    @override
    def render_midi(self, default_velocity: Velocity) -> FrozenMessage:
        return msg_note_off(self.channel, self.note)

    @override
    def render_attrs(self, default_velocity: Velocity) -> MidiAttrs:
        return DMap.empty(MidiDom)


@dataclass(frozen=True)
class ProgramMessage(MidiMessage):
    channel: Channel
    program: Program

    @override
    def render_midi(self, default_velocity: Velocity) -> FrozenMessage:
        return msg_pc(self.channel, self.program)

    @override
    def render_attrs(self, default_velocity: Velocity) -> MidiAttrs:
        attrs = DMap.empty(MidiDom)
        attrs = attrs.put(ChannelKey(), self.channel)
        attrs = attrs.put(ProgramKey(), self.program)
        return attrs


@dataclass(frozen=True)
class ControlMessage(MidiMessage):
    channel: Channel
    control: ControlNum
    value: ControlVal

    @override
    def render_midi(self, default_velocity: Velocity) -> FrozenMessage:
        return msg_cc(self.channel, self.control, self.value)

    @override
    def render_attrs(self, default_velocity: Velocity) -> MidiAttrs:
        attrs = DMap.empty(MidiDom)
        attrs = attrs.put(ChannelKey(), self.channel)
        attrs = attrs.put(ControlNumKey(), self.control)
        attrs = attrs.put(ControlValKey(), self.value)
        return attrs


def midi_bundle_concat(left: MidiBundle, right: MidiBundle) -> MidiBundle:
    if isinstance(left, MidiMessage):
        if isinstance(right, MidiMessage):
            return PSeq.mk([left, right])
        else:
            return right.cons(left)
    else:
        if isinstance(right, MidiMessage):
            return left.snoc(right)
        else:
            return left.concat(right)


def midi_bundle_iterator(bundle: MidiBundle) -> Generator[MidiMessage]:
    if isinstance(bundle, MidiMessage):
        yield bundle
    else:
        yield from bundle


# =============================================================================
# Timed Messages
# =============================================================================


def mido_bundle_concat(left: MidoBundle, right: MidoBundle) -> MidoBundle:
    if isinstance(left, FrozenMessage):
        if isinstance(right, FrozenMessage):
            return PSeq.mk([left, right])
        else:
            return right.cons(left)
    else:
        if isinstance(right, FrozenMessage):
            return left.snoc(right)
        else:
            return left.concat(right)


def mido_bundle_iterator(bundle: MidoBundle) -> Generator[FrozenMessage]:
    if isinstance(bundle, FrozenMessage):
        yield bundle
    else:
        yield from bundle


def mido_bundle_sort_key(bundle: MidoBundle) -> int:
    """Get sort key for a MIDI message.

    Returns a numeric priority for sorting MIDI messages.
    Lower values come first in the sort order.

    Sort order: note_off < {program_change, control_change, bundle} < note_on

    This ensures that:
    - Note offs happen before other events (to clear previous notes)
    - Program and control changes happen between notes (to set up the next note)
    - Note ons happen last (to use the new program/control settings)

    Args:
        msg: The MIDI message to get a sort key for

    Returns:
        An integer sort key (lower values sort first)
    """
    if isinstance(bundle, FrozenMessage):
        if MsgTypeField.exists(bundle):
            msg_type = MsgTypeField.get(bundle)
            if msg_type == "note_off":
                return -1
            elif msg_type == "note_on":
                return 1
    return 0


@dataclass(frozen=True, eq=True, order=False)
class TimedMessage:
    """A timed message with POSIX timestamp."""

    time: PosixTime
    """Timestamp when the message should be sent (POSIX time)."""

    bundle: MidoBundle
    """The frozen MIDI message."""

    def __lt__(self, other: TimedMessage) -> bool:
        """Compare timed messages by time first, then by message type priority."""
        if self.time != other.time:
            return self.time < other.time
        return mido_bundle_sort_key(self.bundle) < mido_bundle_sort_key(other.bundle)

    def __le__(self, other: TimedMessage) -> bool:
        if self.time != other.time:
            return self.time < other.time
        return mido_bundle_sort_key(self.bundle) <= mido_bundle_sort_key(other.bundle)

    def __gt__(self, other: TimedMessage) -> bool:
        return not (self <= other)

    def __ge__(self, other: TimedMessage) -> bool:
        return not (self < other)


@dataclass
class MsgHeap:
    """A priority queue of timed MIDI messages ordered by time and message type."""

    unwrap: PHeap[TimedMessage]

    @staticmethod
    def empty() -> MsgHeap:
        """Create an empty message heap."""
        return MsgHeap(PHeap.empty())

    def push(self, tm: TimedMessage) -> None:
        self.unwrap = self.unwrap.insert(tm)

    def pop(self) -> Optional[TimedMessage]:
        """Pop the earliest message from the heap."""
        x = self.unwrap.find_min()
        if x is None:
            return None
        else:
            v, unwrap2 = x
            self.unwrap = unwrap2
            return v

    def pop_before(self, time: PosixTime) -> Optional[TimedMessage]:
        """Pop head message <= given time"""
        x = self.unwrap.find_min()
        if x is None or x[0].time > time:
            return None
        else:
            v, unwrap2 = x
            self.unwrap = unwrap2
            return v

    def pop_all_before(self, time: PosixTime) -> list[TimedMessage]:
        """Pop all messages <= given time"""
        msgs = []
        while True:
            msg = self.pop_before(time)
            if msg is None:
                break
            else:
                msgs.append(msg)
        return msgs

    def clear(self) -> None:
        self.unwrap = PHeap.empty()
