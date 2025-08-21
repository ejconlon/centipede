from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from threading import Event
from typing import Optional, Tuple, override

import mido
from mido.frozen import FrozenMessage
from spiny.heapmap import PHeapMap

from centipede.common import LockBox, Task


def _assert_pos_lt(x, n):
    assert x >= 0 and x < n


@dataclass(eq=False)
class SendPort:
    _handle: mido.ports.BaseOutput

    @staticmethod
    def connect(name: str) -> SendPort:
        return SendPort(_handle=mido.open_output(name))  # pyright: ignore

    def panic(self):
        self._handle.panic()

    def reset(self):
        self._handle.reset()

    def note_on(self, channel: int, note: int, velocity: int):
        _assert_pos_lt(channel, 16)
        _assert_pos_lt(note, 128)
        _assert_pos_lt(velocity, 128)
        m = FrozenMessage("note_on", channel=channel, note=note, velocity=velocity)
        self._handle.send(m)

    def note_off(self, channel: int, note: int):
        _assert_pos_lt(channel, 16)
        _assert_pos_lt(note, 128)
        m = FrozenMessage("note_off", channel=channel, note=note)
        self._handle.send(m)

    def pc(self, channel: int, program: int):
        _assert_pos_lt(channel, 16)
        _assert_pos_lt(program, 128)
        m = FrozenMessage("program_change", channel=channel, program=program)
        self._handle.send(m)

    def close(self):
        self._handle.close()


type MsgHeap = PHeapMap[float, FrozenMessage]


def mh_empty() -> MsgHeap:
    return PHeapMap.empty()


def mh_push_note(
    mh: MsgHeap, start: float, end: float, channel: int, note: int, velocity: int
) -> MsgHeap:
    assert start <= end
    _assert_pos_lt(channel, 16)
    _assert_pos_lt(note, 128)
    _assert_pos_lt(velocity, 128)
    m1 = FrozenMessage("note_on", channel=channel, note=note, velocity=velocity)
    m2 = FrozenMessage("note_off", channel=channel, note=note)
    return mh.insert(start, m1).insert(end, m2)


def mh_push_pc(mh: MsgHeap, time: float, channel: int, program: int):
    m = FrozenMessage("program_change", channel=channel, program=program)
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


@dataclass(eq=False)
class MsgHeapBox:
    _lb: LockBox[MsgHeap]

    @staticmethod
    def empty() -> MsgHeapBox:
        return MsgHeapBox(_lb=LockBox.new(mh_empty()))

    def push_note(
        self, start: float, end: float, channel: int, note: int, velocity: int
    ):
        with self._lb as box:
            box.value = mh_push_note(
                mh=box.value,
                start=start,
                end=end,
                channel=channel,
                note=note,
                velocity=velocity,
            )

    def push_pc(self, time: float, channel: int, program: int):
        with self._lb as box:
            box.value = mh_push_pc(
                mh=box.value, time=time, channel=channel, program=program
            )

    def pop(self) -> Optional[Tuple[float, FrozenMessage]]:
        with self._lb as box:
            kv, mh2 = mh_pop(mh=box.value)
            box.value = mh2
            return kv

    def seek_pop(self, time: float) -> Optional[Tuple[float, FrozenMessage]]:
        with self._lb as box:
            kv, mh2 = mh_seek_pop(mh=box.value, time=time)
            box.value = mh2
            return kv


@dataclass(eq=False)
class SendTask(Task):
    _port: SendPort
    _mhb: MsgHeapBox

    @override
    def run(self, logger: Logger, start_exit: Event):
        raise NotImplementedError

    @override
    def cleanup(self, logger: Logger):
        self._port.close()
