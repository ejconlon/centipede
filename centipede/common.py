from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from logging import Logger
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Optional, Tuple

from spiny.common import Box


class PartialMatchException(Exception):
    def __init__(self, val: Any):
        super().__init__(f"Unmatched type: {type(val)}")


def ignore_arg[A, B](fn: Callable[[A], B]) -> Callable[[None, A], B]:
    def wrapper(_: None, arg: A) -> B:
        return fn(arg)

    return wrapper


@dataclass(frozen=True, eq=False)
class LockBox[T]:
    _lock: Lock
    _box: Box[T]

    @staticmethod
    def new(value: T) -> LockBox[T]:
        return LockBox(_lock=Lock(), _box=Box(value))

    def __enter__(self) -> Box[T]:
        self._lock.acquire()
        return self._box

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pyright: ignore
        self._lock.release()


class Task(metaclass=ABCMeta):
    @abstractmethod
    def run(self, logger: Logger, start_exit: Event) -> None:
        raise NotImplementedError

    def cleanup(self, logger: Logger) -> None:
        return


@dataclass(eq=False)
class ExecutorState:
    id_src: int
    threads: Dict[int, Tuple[Event, Thread]]

    @staticmethod
    def empty() -> ExecutorState:
        return ExecutorState(0, {})


@dataclass(frozen=True, eq=False)
class Executor:
    _logger: Logger
    _lb: LockBox[ExecutorState]
    _start_exit: Event
    _stop_exit: Event

    @staticmethod
    def new(logger: Logger) -> Executor:
        return Executor(
            _logger=logger,
            _lb=LockBox.new(ExecutorState.empty()),
            _start_exit=Event(),
            _stop_exit=Event(),
        )

    def stop_all(self) -> Event:
        self._start_exit.set()
        return self._stop_exit

    def _fork_run(self, uid: int, uname: str, task: Task) -> None:
        logger = self._logger.getChild(uname)
        try:
            task.run(logger, self._start_exit)
        finally:
            task.cleanup(logger)
            with self._lb as box:
                task_exit, _ = box.value.threads[uid]
                del box.value.threads[uid]
            task_exit.set()

    def fork(self, name: str, task: Task) -> Event:
        task_exit = Event()
        if self._start_exit.is_set():
            task_exit.set()
        else:
            with self._lb as box:
                uid = box.value.id_src
                uname = f"{name}_{uid}"
                thread = Thread(
                    name=uname, target=self._fork_run, args=(uid, uname, task)
                )
                box.value.id_src += 1
                box.value.threads[uid] = (task_exit, thread)
            thread.start()
        return task_exit

    def join_all(self):
        while True:
            next_thread: Optional[Thread]
            with self._lb as box:
                if box.value.threads:
                    next_uid = next(reversed(box.value.threads.keys()))
                    next_thread = box.value.threads[next_uid][1]
                else:
                    next_thread = None
            if next_thread is None:
                return
            else:
                next_thread.join()
