from __future__ import annotations

from functools import partial
from logging import Logger
from typing import Any, Optional, Tuple, cast, override

import mido
from mido.frozen import FrozenMessage, freeze_message

from centipede.actor import (
    Actor,
    ActorEnv,
    Callback,
    Mutex,
    Sender,
    System,
    new_system,
)
from spiny.common import Box
from spiny.heapmap import PHeapMap


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
