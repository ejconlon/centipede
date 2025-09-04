from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from logging import Logger
from typing import Any, NewType, Optional, Tuple, cast, override

import mido
from mido.frozen import FrozenMessage, freeze_message

from centipede.actor import Actor, ActorEnv, Callback, Mutex, Sender, System, new_system
from minipat.common import PosixTime
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


class MidiDom:
    pass


type MidiAttrs = DMap[MidiDom]


class MidiKey[V](DKey[MidiDom, V]):
    pass


Note = NewType("Note", int)


class NoteKey(MidiKey[Note]):
    pass


Vel = NewType("Vel", int)


class VelKey(DKey[MidiDom, Vel]):
    pass


class Selector[V](metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def parse(cls, sel: Selected[str]) -> V:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def render(cls, value: V) -> Selected[str]:
        raise NotImplementedError


class NoteNumSelector(Selector[Note]):
    # Parse/render string and for each str assert integer representation (0-127) with empty selected
    # Attrs are singletons with NoteKey key

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Note:
        try:
            note_num = int(sel.value)
            if 0 <= note_num <= 127:
                return Note(note_num)
            else:
                raise ValueError(f"Note number {note_num} out of range (0-127)")
        except ValueError as e:
            raise ValueError(f"Invalid note number: {sel.value}") from e

    @override
    @classmethod
    def render(cls, value: Note) -> Selected[str]:
        return Selected(str(value), None)


class NoteNameSelector(Selector[Note]):
    # Parse/render string and for each str assert string note representation (like c4) with empty selected
    # Attrs are singletons with NoteKey key

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

        if 0 <= midi_note <= 127:
            return Note(midi_note)
        else:
            raise ValueError(
                f"Note {sel.value} results in MIDI note {midi_note} out of range (0-127)"
            )

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
    # Parse/render string and for each str assert integer representation (0-127) with empty selected

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Vel:
        try:
            vel_num = int(sel.value)
            if 0 <= vel_num <= 127:
                return Vel(vel_num)
            else:
                raise ValueError(f"Velocity {vel_num} out of range (0-127)")
        except ValueError as e:
            raise ValueError(f"Invalid velocity: {sel.value}") from e

    @override
    @classmethod
    def render(cls, value: Vel) -> Selected[str]:
        return Selected(str(value), None)


def _convert_midinote(sel: Selected[str]) -> MidiAttrs:
    note = NoteNumSelector.parse(sel)
    return DMap.singleton(NoteKey(), note)


def _convert_note(sel: Selected[str]) -> MidiAttrs:
    note = NoteNameSelector.parse(sel)
    return DMap.singleton(NoteKey(), note)


def _convert_vel(sel: Selected[str]) -> MidiAttrs:
    velocity = VelSelector.parse(sel)
    return DMap.singleton(VelKey(), velocity)


def midinote(s: str) -> Stream[MidiAttrs]:
    return pat_stream(parse_pattern(s).map(_convert_midinote))


def note(s: str) -> Stream[MidiAttrs]:
    return pat_stream(parse_pattern(s).map(_convert_note))


def vel(s: str) -> Stream[MidiAttrs]:
    return pat_stream(parse_pattern(s).map(_convert_vel))


def _merge_attrs(x: MidiAttrs, y: MidiAttrs) -> MidiAttrs:
    return x.merge(y)


def combine(*ss: Stream[MidiAttrs]) -> Stream[MidiAttrs]:
    if len(ss):
        acc = ss[0]
        for el in ss[1:]:
            acc = acc.apply(MergeStrat.Inner, _merge_attrs, el)
        return acc
    else:
        return Stream.silence()


@dataclass(frozen=True)
class TimedMessage:
    """A timed message with POSIX timestamp."""

    time: PosixTime
    """Timestamp when the message should be sent (POSIX time)."""

    message: FrozenMessage
    """The frozen MIDI message."""


class MidiProcessor(Processor[MidiAttrs, TimedMessage]):
    """Processor that converts MidiAttrs to MIDI messages."""

    def __init__(self, default_velocity: int = 64):
        """Initialize the MIDI processor.

        Args:
            default_velocity: Default velocity to use when not specified
        """
        self.default_velocity = default_velocity

    @override
    def process(
        self, instant: Instant, orbit: Orbit, events: EvHeap[MidiAttrs]
    ) -> PSeq[TimedMessage]:
        """Process MIDI events into timed MIDI messages."""
        timed_messages = []

        # Use orbit as MIDI channel (clamp to 0-15 range)
        channel = max(0, min(15, int(orbit)))

        for span, ev in events:
            # Extract MIDI attributes
            note_num = ev.val.lookup(NoteKey())
            velocity = ev.val.lookup(VelKey())

            # Use defaults if attributes are missing
            note = int(note_num) if note_num is not None else 60  # Middle C
            vel = int(velocity) if velocity is not None else self.default_velocity

            # Ensure values are in valid MIDI range
            note = max(0, min(127, note))
            vel = max(0, min(127, vel))

            # Only send note_on if active start is whole start (or whole is empty)
            send_note_on = span.whole is None or span.active.start == span.whole.start
            if send_note_on:
                # Calculate timestamp for the start of the event
                timestamp = PosixTime(
                    instant.posix_start
                    + (float(span.active.start) / float(instant.cps))
                )

                # Create note on message using FrozenMessage
                note_on_msg = FrozenMessage(
                    "note_on", channel=channel, note=note, velocity=vel
                )
                timed_messages.append(TimedMessage(timestamp, note_on_msg))

            # Only send note_off if active end is whole end (or whole is empty)
            send_note_off = span.whole is None or span.active.end == span.whole.end
            if send_note_off:
                # Create note off message (at end of span)
                note_off_timestamp = PosixTime(
                    instant.posix_start + (float(span.active.end) / float(instant.cps))
                )
                note_off_msg = FrozenMessage(
                    "note_off", channel=channel, note=note, velocity=0
                )
                timed_messages.append(TimedMessage(note_off_timestamp, note_off_msg))

        return PSeq.mk(timed_messages)


class MidiActor(Actor[BackendMessage[TimedMessage]]):
    """Actor that sends MIDI messages to outputs."""

    def __init__(self, output: mido.ports.BaseOutput):
        """Initialize the MIDI actor.

        Args:
            output: MIDI output port, or None to disable output
        """
        self._output = output
        self._playing = False

    @override
    def on_stop(self, logger: Logger) -> None:
        self._output.reset()

    @override
    def on_message(self, env: ActorEnv, value: BackendMessage[TimedMessage]) -> None:
        match value:
            case BackendPlay(playing):
                self._playing = playing
                if playing:
                    env.logger.info("MIDI: Playing")
                else:
                    env.logger.info("MIDI: Pausing")
                    self._output.reset()
            case BackendEvents(messages):
                if self._playing:
                    self._send_messages(env, messages)
                else:
                    env.logger.debug("MIDI: Ignoring events while stopped")
            case _:
                env.logger.warning(f"Unknown MIDI message type: {type(value)}")

    def _send_messages(self, env: ActorEnv, messages: PSeq[TimedMessage]) -> None:
        for timed_message in messages:
            try:
                # For now, send immediately - in a real implementation you'd
                # want to schedule based on timestamp
                self._output.send(timed_message.message)
                env.logger.debug(f"Sent MIDI message: {timed_message.message}")
            except Exception as e:
                env.logger.error(f"Error sending MIDI message: {e}")


# === Moved from centipede/midi.py ===


def _assert_pos_lt(x, n):
    assert x >= 0 and x < n


def msg_note_on(channel: int, note: int, velocity: int) -> FrozenMessage:
    _assert_pos_lt(channel, 16)
    _assert_pos_lt(note, 128)
    _assert_pos_lt(velocity, 128)
    return FrozenMessage("note_on", channel=channel, note=note, velocity=velocity)


def msg_note_off(channel: int, note: int) -> FrozenMessage:
    _assert_pos_lt(channel, 16)
    _assert_pos_lt(note, 128)
    return FrozenMessage("note_off", channel=channel, note=note)


def msg_pc(channel: int, program: int) -> FrozenMessage:
    _assert_pos_lt(channel, 16)
    _assert_pos_lt(program, 128)
    return FrozenMessage("program_change", channel=channel, program=program)


type MsgHeap = PHeapMap[float, FrozenMessage]


def mh_empty() -> MsgHeap:
    return PHeapMap.empty()


def mh_push_note(
    mh: MsgHeap, start: float, end: float, channel: int, note: int, velocity: int
) -> MsgHeap:
    assert start <= end
    m1 = msg_note_on(channel=channel, note=note, velocity=velocity)
    m2 = msg_note_off(channel=channel, note=note)
    return mh.insert(start, m1).insert(end, m2)


def mh_push_pc(mh: MsgHeap, time: float, channel: int, program: int):
    m = msg_pc(channel=channel, program=program)
    return mh.insert(time, m)


def mh_pop(mh: MsgHeap) -> Tuple[Optional[Tuple[float, FrozenMessage]], MsgHeap]:
    x = mh.find_min()
    if x is None:
        return (None, mh)
    else:
        k, v, mh2 = x
        return ((k, v), mh2)


def mh_seek_pop(
    mh: MsgHeap, time: float
) -> Tuple[Optional[Tuple[float, FrozenMessage]], MsgHeap]:
    while True:
        x = mh_pop(mh)
        if x[0] is None or x[0][0] >= time:
            return x
        else:
            mh = x[1]


class ParMsgHeap:
    def __init__(self):
        self._mutex = Mutex(Box(mh_empty()))

    def push_note(
        self, start: float, end: float, channel: int, note: int, velocity: int
    ) -> None:
        with self._mutex as box:
            box.value = mh_push_note(
                mh=box.value,
                start=start,
                end=end,
                channel=channel,
                note=note,
                velocity=velocity,
            )

    def push_pc(self, time: float, channel: int, program: int) -> None:
        with self._mutex as box:
            box.value = mh_push_pc(
                mh=box.value, time=time, channel=channel, program=program
            )

    def pop(self) -> Optional[Tuple[float, FrozenMessage]]:
        with self._mutex as box:
            kv, mh2 = mh_pop(mh=box.value)
            box.value = mh2
            return kv

    def seek_pop(self, time: float) -> Optional[Tuple[float, FrozenMessage]]:
        with self._mutex as box:
            kv, mh2 = mh_seek_pop(mh=box.value, time=time)
            box.value = mh2
            return kv


class SendActor(Actor[FrozenMessage]):
    def __init__(self, port: mido.ports.BaseOutput):
        self._port = port

    @override
    def on_message(self, env: ActorEnv, value: FrozenMessage) -> None:
        self._port.send(value)

    @override
    def on_stop(self, logger: Logger) -> None:
        self._port.close()


def _recv_cb(sender: Sender[FrozenMessage], msg: Any) -> None:
    fmsg = cast(FrozenMessage, freeze_message(msg))
    sender.send(fmsg)


class RecvCallback(Callback[FrozenMessage]):
    def __init__(self, port: mido.ports.BaseInput):
        self._port = port

    def register(self, sender: Sender[FrozenMessage]) -> None:
        self._port.callback = partial(_recv_cb, sender)  # pyright: ignore

    def unregister(self) -> None:
        self._port.callback = None  # pyright: ignore


def echo_system() -> System:
    system = new_system("echo")
    out_port = mido.open_output(name="virt_out", virtual=True)  # pyright: ignore
    in_port = mido.open_input(name="virt_in", virtual=True)  # pyright: ignore
    send_actor = SendActor(out_port)
    recv_actor = RecvCallback(in_port).produce("send", send_actor)
    system.spawn_actor("recv", recv_actor)
    return system
