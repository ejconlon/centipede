from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mido
from mido.frozen import FrozenMessage

from centipede.common import LockBox
from centipede.spiny.heapmap import PHeapMap


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


def msg_pc(self, channel: int, program: int) -> FrozenMessage:
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
        return MsgHeapBox(_lb=LockBox(mh_empty()))

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


def connect_fn(name: str) -> mido.ports.BaseOutput:
    return mido.open_output(name=name)  # pyright: ignore


# @dataclass(frozen=True, eq=False)
# class ConnectTask(Task):
#     name: str
#     fut: MutFuture[mido.ports.BaseOutput]
#
#     @override
#     def run(self, env: TaskEnv):
#         applied_connect_fn = partial(connect_fn, self.name)
#         while not env.start_exit.is_set():
#             self.fut.attempt(applied_connect_fn)
#             self.fut.wait_empty()
#
#     @override
#     def cleanup(self, env: TaskEnv, exc: Optional[Exception]):
#         self.fut.close()
#         tagged = self.fut.raw_unwrap()
#         output = tagged.as_resolve()
#         if output is not None:
#             output.close()
#
#
# @dataclass(frozen=True, eq=False)
# class ConsumeTask(Task):
#     fut: MutFuture[mido.ports.BaseOutput]
#     queue: Queue[FrozenMessage]
#
#     @override
#     def run(self, env: TaskEnv):
#         while not env.start_exit.is_set():
#             msg = queue.get_nowait()
