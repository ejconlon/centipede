from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from queue import Queue
from threading import Lock
from typing import Any, NewType, Optional, TypeVar

from centipede.spiny.common import Box

T = TypeVar("T")


def is_fatal_exception(exc: Exception) -> bool:
    return isinstance(exc, SystemExit) or isinstance(exc, KeyboardInterrupt)


class Mutex[T]:
    def __init__(self, value: T):
        self._lock = Lock()
        self._value = value

    def __enter__(self) -> T:
        self._lock.acquire()
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock.release()


class Action(Enum):
    Message = 1
    Handle = 2
    Stop = 3
    ReportSelf = 4
    ReportParent = 5


@dataclass(frozen=True)
class ActionException(Exception):
    action: Action
    exc: Exception
    saved_exc: Optional[ActionException]


ActorId = NewType("ActorId", int)


class Packet[T](metaclass=ABCMeta):
    pass


@dataclass(frozen=True)
class StopPacket(Packet[Any]):
    pass


@dataclass(frozen=True)
class MessagePacket[T](Packet[T]):
    value: T


@dataclass(frozen=True)
class ReportPacket[T](Packet[T]):
    child_id: ActorId
    child_exc: Optional[ActionException]


@dataclass(frozen=True, eq=False)
class Sender[T]:
    @abstractmethod
    def dest(self) -> ActorId:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send(self, msg: T) -> None:
        raise NotImplementedError


class Control(metaclass=ABCMeta):
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def spawn(self, actor: Actor[T]) -> Sender[T]:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class ActorEnv:
    logger: Logger
    control: Control


class Actor[T](metaclass=ABCMeta):
    @abstractmethod
    def on_message(self, env: ActorEnv, value: T) -> None:
        return

    def on_handle(self, env: ActorEnv, exc: Exception) -> None:
        raise exc

    def on_stop(self, env: ActorEnv) -> None:
        return

    def on_report(
        self, env: ActorEnv, child_id: ActorId, exc: Optional[Exception]
    ) -> None:
        return


class Topo:
    pass


@dataclass(frozen=True, eq=False)
class ActorRunner[T]:
    _topo: Mutex[Topo]
    _env: ActorEnv
    _parent_id: ActorId
    _own_id: ActorId
    _own_q: Queue[Packet[T]]
    _actor: Actor[T]

    def run(self) -> None:
        while True:
            self._step()

    def _handle(self, saved_exc: Box[Optional[ActionException]]) -> None:
        if saved_exc.value is None:
            return

        try:
            self._actor.on_handle(env=self._env, exc=saved_exc.value.exc)
        except Exception as exc:
            saved_exc.value = ActionException(
                action=Action.Handle, exc=exc, saved_exc=saved_exc.value
            )

        if saved_exc.value is not None:
            self._stop(saved_exc=saved_exc)

    def _stop(self, saved_exc: Box[Optional[ActionException]]) -> None:
        self._own_q.shutdown(immediate=True)

        try:
            self._actor.on_stop(env=self._env)
        except Exception as exc:
            saved_exc.value = ActionException(
                action=Action.Stop, exc=exc, saved_exc=saved_exc.value
            )

        try:
            self._report_parent(saved_exc=saved_exc.value)
        except Exception as exc:
            saved_exc.value = ActionException(
                action=Action.ReportParent, exc=exc, saved_exc=saved_exc.value
            )

    def _message(self, saved_exc: Box[Optional[ActionException]], value: T) -> None:
        try:
            self._actor.on_message(env=self._env, value=value)
        except Exception as exc:
            saved_exc.value = ActionException(
                action=Action.Message, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _report_self(
        self,
        saved_exc: Box[Optional[ActionException]],
        child_id: ActorId,
        child_exc: Optional[ActionException],
    ) -> None:
        try:
            self._actor.on_report(env=self._env, child_id=child_id, exc=child_exc)
        except Exception as exc:
            saved_exc.value = ActionException(
                action=Action.ReportSelf, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _report_parent(self, saved_exc: Optional[ActionException]) -> None:
        raise NotImplementedError

    def _step(self) -> None:
        saved_exc: Box[Optional[ActionException]] = Box(None)
        try:
            pack = self._own_q.get()
            match pack:
                case StopPacket():
                    self._stop(saved_exc=saved_exc)
                case MessagePacket(value):
                    self._message(saved_exc=saved_exc, value=value)
                case ReportPacket(child_id, child_exc):
                    self._report_self(
                        saved_exc=saved_exc, child_id=child_id, child_exc=child_exc
                    )
        finally:
            self._own_q.task_done()
            if saved_exc.value is not None and is_fatal_exception(saved_exc.value):
                raise saved_exc.value
