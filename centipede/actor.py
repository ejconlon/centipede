from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from threading import Condition, Lock, Thread
from typing import Any, Dict, List, NewType, Optional, Set, TypeVar, cast, override

from centipede.spiny.common import Box

T = TypeVar("T")


class Mutex[T]:
    def __init__(self, value: T):
        self._lock = Lock()
        self._value = value

    def __enter__(self) -> T:
        self._lock.acquire()
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock.release()


class Queue[T]:
    def __init__(self, items: Optional[Iterable[T]] = None):
        self._cv = Condition()
        self._items: deque[T] = deque()
        if items:
            self._items.extend(items)
        self._draining = False

    def put(self, item: T) -> None:
        with self._cv:
            if not self._draining:
                self._items.append(item)

    def drain(self, item: T, immediate: bool) -> None:
        with self._cv:
            if immediate:
                self._draining = True
                self._items.clear()
                self._items.appendleft(item)
            elif not self._draining:
                self._draining = True
                self._items.append(item)

    def seal(self) -> None:
        with self._cv:
            self._draining = True
            self._items.clear()

    def _resume_get(self) -> bool:
        return self._draining or bool(self._items)

    def get(self) -> Optional[T]:
        with self._cv:
            if not self._items:
                self._cv.wait_for(self._resume_get)
            if self._items:
                return self._items.popleft()
            else:
                return None

    def _resume_wait(self) -> bool:
        return self._draining and not self._items

    def wait(self) -> None:
        with self._cv:
            if not self._resume_wait():
                self._cv.wait_for(self._resume_wait)


class Action(Enum):
    Start = 0
    Message = 1
    Handle = 2
    Stop = 3
    Cleanup = 4
    Report = 5
    Supervise = 6


@dataclass(frozen=True)
class ActionException(Exception):
    actor_name: str
    actor_id: ActorId
    fatal: bool
    action: Action
    exc: Exception
    saved_exc: Optional[ActionException]


def is_fatal_exception(exc: Exception) -> bool:
    if isinstance(exc, SystemExit):
        return True
    elif isinstance(exc, KeyboardInterrupt):
        return True
    elif isinstance(exc, ActionException):
        return exc.fatal
    else:
        return False


ActorId = NewType("ActorId", int)


class Packet[T](metaclass=ABCMeta):
    pass


@dataclass(frozen=True)
class StartPacket(Packet[Any]):
    pass


@dataclass(frozen=True)
class MessagePacket[T](Packet[T]):
    value: T


@dataclass(frozen=True)
class ReportPacket[T](Packet[T]):
    child_id: ActorId
    child_exc: Optional[ActionException]


@dataclass(frozen=True)
class StopPacket(Packet[Any]):
    pass


class Sender[T](metaclass=ABCMeta):
    @abstractmethod
    def dest(self) -> ActorId:
        raise NotImplementedError

    @abstractmethod
    def send(self, msg: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self, immediate: bool) -> None:
        raise NotImplementedError


class Control(metaclass=ABCMeta):
    @abstractmethod
    def spawn(self, name: str, actor: Actor[T]) -> Sender[T]:
        raise NotImplementedError

    @abstractmethod
    def stop(self, immediate: bool) -> None:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class ActorEnv:
    logger: Logger
    control: Control


# Lifecycle:
#   Start
#   Loop: Message/Report, Handle uncaught exceptions
#   Stop: prepare for child cleanup
#   Cleanup: children cleaned up
#   Report to parent
class Actor[T](metaclass=ABCMeta):
    def on_start(self, env: ActorEnv) -> None:
        return

    def on_message(self, env: ActorEnv, value: T) -> None:
        return

    def on_report(
        self, env: ActorEnv, child_id: ActorId, exc: Optional[ActionException]
    ) -> None:
        return

    def on_handle(self, env: ActorEnv, exc: Exception) -> None:
        raise exc

    def on_stop(self, logger: Logger) -> None:
        return

    def on_cleanup(self, logger: Logger) -> None:
        return


@dataclass(frozen=True, eq=False)
class ActorState[T]:
    name: str
    actor_id: ActorId
    parent_id: Optional[ActorId]
    queue: Queue[Packet[T]]
    logger: Logger
    actor: Actor[T]


class ActorLoop[T]:
    def __init__(self, control: Control, state: ActorState[T]):
        self._state = state
        self._env = ActorEnv(logger=self._state.logger, control=control)
        self._stopped = False

    def run(self) -> Box[Optional[ActionException]]:
        saved_exc: Box[Optional[ActionException]] = Box(None)
        while saved_exc.value is None:
            self._step(saved_exc=saved_exc)
        if not self._stopped:
            self._stop(saved_exc=saved_exc)
        return saved_exc

    def cleanup(self, saved_exc: Box[Optional[ActionException]]) -> None:
        try:
            self._state.actor.on_cleanup(logger=self._env.logger)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Cleanup, exc=exc, saved_exc=saved_exc.value
            )

    def _step(self, saved_exc: Box[Optional[ActionException]]) -> None:
        pack = self._state.queue.get()
        if pack is None:
            return
        else:
            match pack:
                case StartPacket():
                    self._start(saved_exc=saved_exc)
                case MessagePacket(value):
                    self._message(saved_exc=saved_exc, value=value)
                case ReportPacket(child_id, child_exc):
                    self._report(
                        saved_exc=saved_exc, child_id=child_id, child_exc=child_exc
                    )
                case StopPacket():
                    self._stop(saved_exc=saved_exc)

    def _start(self, saved_exc: Box[Optional[ActionException]]) -> None:
        try:
            self._state.actor.on_start(env=self._env)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Start, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _message(self, saved_exc: Box[Optional[ActionException]], value: T) -> None:
        try:
            self._state.actor.on_message(env=self._env, value=value)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Message, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _report(
        self,
        saved_exc: Box[Optional[ActionException]],
        child_id: ActorId,
        child_exc: Optional[ActionException],
    ) -> None:
        try:
            self._state.actor.on_report(env=self._env, child_id=child_id, exc=child_exc)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Report, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _handle(self, saved_exc: Box[Optional[ActionException]]) -> None:
        if saved_exc.value is not None and not is_fatal_exception(saved_exc.value):
            try:
                self._state.actor.on_handle(env=self._env, exc=saved_exc.value.exc)
            except Exception as exc:
                saved_exc.value = self._except(
                    action=Action.Handle, exc=exc, saved_exc=saved_exc.value
                )

    def _stop(self, saved_exc: Box[Optional[ActionException]]) -> None:
        try:
            self._state.actor.on_stop(logger=self._env.logger)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Stop, exc=exc, saved_exc=saved_exc.value
            )

        self._stopped = True

    def _except(
        self, action: Action, exc: Exception, saved_exc: Optional[ActionException]
    ) -> ActionException:
        fatal = is_fatal_exception(exc) or (saved_exc is not None and saved_exc.fatal)
        return ActionException(
            actor_name=self._state.name,
            actor_id=self._state.actor_id,
            fatal=fatal,
            action=action,
            exc=exc,
            saved_exc=saved_exc,
        )


@dataclass(frozen=True)
class ActorContext:
    state: ActorState[None]
    thread: Thread
    child_ids: Set[ActorId]


@dataclass(frozen=False)
class GlobalMutState:
    id_src: ActorId
    contexts: Dict[ActorId, ActorContext]
    draining: bool
    saved_excs: List[ActionException]
    logger: Logger

    @staticmethod
    def empty(logger: Optional[Logger] = None) -> GlobalMutState:
        if logger is None:
            logging.basicConfig(level=logging.CRITICAL)
            logger = logging.getLogger()
        return GlobalMutState(
            id_src=ActorId(0),
            contexts={},
            draining=False,
            saved_excs=[],
            logger=logger,
        )


type GlobalState = Mutex[GlobalMutState]


class ActorLifecycle[T]:
    def __init__(self, global_state: GlobalState, state: ActorState[T]):
        self._global_state = global_state
        self._state = state

    def run(self) -> None:
        control = RegularControl(
            global_state=self._global_state,
            state=cast(ActorState[None], self._state),
        )
        loop = ActorLoop(control=control, state=self._state)
        saved_exc = loop.run()
        actor_id = self._state.actor_id
        fatal = saved_exc.value is not None and saved_exc.value.fatal
        child_queues: List[Queue[Packet[None]]] = []
        with self._global_state as gs:
            # Immediately stop spawning if fatal
            if fatal:
                gs.draining = True
            # Seal the queue to wake any waiters
            ctx = gs.contexts[actor_id]
            ctx.state.queue.seal()
            # Send child stops
            for child_id in ctx.child_ids:
                child_ctx = gs.contexts.get(child_id)
                if child_ctx is not None:
                    child_ctx.state.queue.drain(StopPacket(), immediate=fatal)
                    child_queues.append(child_ctx.state.queue)
        # Await child queues
        for queue in child_queues:
            queue.wait()
        # Cleanup
        loop.cleanup(saved_exc=saved_exc)
        with self._global_state as gs:
            parent_ctx: Optional[ActorContext] = None
            if self._state.parent_id is not None:
                parent_ctx = gs.contexts.get(self._state.parent_id)
            if parent_ctx is not None:
                parent_ctx.child_ids.remove(actor_id)
                if fatal:
                    # Propagate fatal upwards
                    parent_ctx.state.queue.drain(StopPacket(), immediate=True)
                else:
                    # Simply report
                    pack: Packet[None] = ReportPacket(
                        child_id=actor_id, child_exc=saved_exc.value
                    )
                    parent_ctx.state.queue.put(pack)
            if fatal:
                # Save exc
                assert saved_exc.value is not None
                gs.saved_excs.append(saved_exc.value)
            # Cleanup context
            del gs.contexts[actor_id]


class RegularSender[T](Sender[T]):
    def __init__(self, child_state: ActorState[T]):
        self._child_state = child_state

    @override
    def dest(self) -> ActorId:
        return self._child_state.actor_id

    @override
    def stop(self, immediate: bool) -> None:
        self._child_state.queue.drain(StopPacket(), immediate=immediate)

    @override
    def send(self, msg: T) -> None:
        self._child_state.queue.put(MessagePacket(msg))


class NullSender[T](Sender[T]):
    def __init__(self, child_id: ActorId):
        self._child_id = child_id

    @override
    def dest(self) -> ActorId:
        return self._child_id

    @override
    def stop(self, immediate: bool) -> None:
        pass

    @override
    def send(self, msg: T) -> None:
        pass


class RegularControl(Control):
    def __init__(self, global_state: GlobalState, state: ActorState[None]):
        self._global_state = global_state
        self._state = state

    @override
    def stop(self, immediate: bool) -> None:
        self._state.queue.drain(StopPacket(), immediate=immediate)

    @override
    def spawn(self, name: str, actor: Actor[T]) -> Sender[T]:
        with self._global_state as gs:
            child_id = gs.id_src
            gs.id_src = ActorId(gs.id_src + 1)
            if gs.draining:
                return NullSender(child_id=child_id)
            else:
                parent_id = self._state.actor_id
                child_queue: Queue[Packet[T]] = Queue([StartPacket()])
                uname = f"{name}_{child_id}"
                logger = self._state.logger.getChild(uname)
                child_state: ActorState[T] = ActorState(
                    name=name,
                    actor_id=child_id,
                    parent_id=parent_id,
                    queue=child_queue,
                    logger=logger,
                    actor=actor,
                )
                lifecycle = ActorLifecycle(
                    global_state=self._global_state, state=child_state
                )
                thread = Thread(name=uname, target=lifecycle.run)
                context = ActorContext(
                    state=cast(ActorState[None], child_state),
                    thread=thread,
                    child_ids=set(),
                )
                gs.contexts[child_id] = context
                gs.contexts[parent_id].child_ids.add(child_id)
                thread.start()
                return RegularSender(child_state)


class RootActor(Actor[None]):
    def __init__(self, global_state: GlobalState):
        self._global_state = global_state

    def on_stop(self, logger: Logger) -> None:
        with self._global_state as gs:
            gs.draining = True


def control(logger: Optional[Logger] = None) -> Control:
    global_state = Mutex(GlobalMutState.empty(logger=logger))
    root_name = "root"
    root_actor = RootActor(global_state)
    with global_state as gs:
        root_id = gs.id_src
        gs.id_src = ActorId(gs.id_src + 1)
        root_queue: Queue[Packet[None]] = Queue([StartPacket()])
        root_uname = f"{root_name}_{root_id}"
        root_logger = gs.logger.getChild(root_uname)
        root_state: ActorState[None] = ActorState(
            name=root_name,
            actor_id=root_id,
            parent_id=None,
            queue=root_queue,
            logger=root_logger,
            actor=root_actor,
        )
        lifecycle = ActorLifecycle(global_state=global_state, state=root_state)
        thread = Thread(name=root_uname, target=lifecycle.run)
        context = ActorContext(
            state=root_state,
            thread=thread,
            child_ids=set(),
        )
        gs.contexts[root_id] = context
        thread.start()
        return RegularControl(global_state=global_state, state=root_state)
