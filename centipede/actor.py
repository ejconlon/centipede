from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from threading import Condition, Event, Lock, Thread
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


UniqId = NewType("UniqId", int)


def fmt_uniq_name(name: str, uniq_id: UniqId) -> str:
    return f"{name}_{uniq_id}"


@dataclass(frozen=True)
class ActionException(Exception):
    name: str
    uniq_id: UniqId
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


class Task(metaclass=ABCMeta):
    @abstractmethod
    def run(self, logger: Logger, halt: Event) -> None:
        raise NotImplementedError


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
    child_id: UniqId
    child_exc: Optional[ActionException]


@dataclass(frozen=True)
class StopPacket(Packet[Any]):
    pass


class Sender[T](metaclass=ABCMeta):
    @abstractmethod
    def dest(self) -> UniqId:
        raise NotImplementedError

    @abstractmethod
    def send(self, msg: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self, immediate: bool) -> None:
        raise NotImplementedError


class Control(metaclass=ABCMeta):
    @abstractmethod
    def spawn_actor(self, name: str, actor: Actor[T]) -> Sender[T]:
        raise NotImplementedError

    @abstractmethod
    def spawn_task(self, name: str, task: Task) -> Sender[None]:
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
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        return

    def on_handle(self, env: ActorEnv, exc: Exception) -> None:
        raise exc

    def on_stop(self, logger: Logger) -> None:
        return

    def on_cleanup(self, logger: Logger) -> None:
        return


@dataclass(frozen=True)
class Node:
    name: str
    uniq_id: UniqId
    parent_id: Optional[UniqId]


class ActorLoop[T]:
    def __init__(
        self,
        control: Control,
        node: Node,
        logger: Logger,
        actor: Actor[T],
        queue: Queue[Packet[T]],
    ):
        self._node = node
        self._env = ActorEnv(logger=logger, control=control)
        self._stopped = False
        self._logger = logger
        self._actor = actor
        self._queue = queue

    def run(self) -> Box[Optional[ActionException]]:
        saved_exc: Box[Optional[ActionException]] = Box(None)
        while saved_exc.value is None:
            self._step(saved_exc=saved_exc)
        if not self._stopped:
            self._stop(saved_exc=saved_exc)
        return saved_exc

    def cleanup(self, saved_exc: Box[Optional[ActionException]]) -> None:
        try:
            self._actor.on_cleanup(logger=self._env.logger)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Cleanup, exc=exc, saved_exc=saved_exc.value
            )

    def _step(self, saved_exc: Box[Optional[ActionException]]) -> None:
        pack = self._queue.get()
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
            self._actor.on_start(env=self._env)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Start, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _message(self, saved_exc: Box[Optional[ActionException]], value: T) -> None:
        try:
            self._actor.on_message(env=self._env, value=value)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Message, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _report(
        self,
        saved_exc: Box[Optional[ActionException]],
        child_id: UniqId,
        child_exc: Optional[ActionException],
    ) -> None:
        try:
            self._actor.on_report(env=self._env, child_id=child_id, exc=child_exc)
        except Exception as exc:
            saved_exc.value = self._except(
                action=Action.Report, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _handle(self, saved_exc: Box[Optional[ActionException]]) -> None:
        if saved_exc.value is not None and not is_fatal_exception(saved_exc.value):
            try:
                self._actor.on_handle(env=self._env, exc=saved_exc.value.exc)
            except Exception as exc:
                saved_exc.value = self._except(
                    action=Action.Handle, exc=exc, saved_exc=saved_exc.value
                )

    def _stop(self, saved_exc: Box[Optional[ActionException]]) -> None:
        try:
            self._actor.on_stop(logger=self._env.logger)
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
            name=self._node.name,
            uniq_id=self._node.uniq_id,
            fatal=fatal,
            action=action,
            exc=exc,
            saved_exc=saved_exc,
        )


@dataclass(frozen=True, eq=False)
class ActorContext[T]:
    node: Node
    thread: Thread
    queue: Queue[Packet[T]]
    child_ids: Set[UniqId]


@dataclass(frozen=True, eq=False)
class TaskContext:
    node: Node
    thread: Thread
    halt: Event


@dataclass(frozen=False)
class GlobalMutState:
    id_src: UniqId
    actors: Dict[UniqId, ActorContext]
    tasks: Dict[UniqId, TaskContext]
    draining: bool
    saved_excs: List[ActionException]
    logger: Logger

    @staticmethod
    def empty(logger: Optional[Logger] = None) -> GlobalMutState:
        if logger is None:
            logging.basicConfig(level=logging.CRITICAL)
            logger = logging.getLogger()
        return GlobalMutState(
            id_src=UniqId(0),
            actors={},
            tasks={},
            draining=False,
            saved_excs=[],
            logger=logger,
        )


type GlobalState = Mutex[GlobalMutState]


class TaskLifecycle:
    def __init__(
        self,
        global_state: GlobalState,
        uniq_id: UniqId,
        logger: Logger,
        halt: Event,
        task: Task,
    ):
        self._global_state = global_state
        self._uniq_id = uniq_id
        self._logger = logger
        self._halt = halt
        self._task = task

    def run(self):
        try:
            self._task.run(logger=self._logger, halt=self._halt)
        finally:
            with self._global_state as gs:
                # Remove from tasks
                pass


class ActorLifecycle[T]:
    def __init__(
        self,
        global_state: GlobalState,
        node: Node,
        logger: Logger,
        actor: Actor[T],
        queue: Queue[Packet[T]],
    ):
        self._global_state = global_state
        self._node = node
        self._logger = logger
        self._actor = actor
        self._queue = queue

    def run(self) -> None:
        control = ControlImpl(
            global_state=self._global_state,
            node=self._node,
            logger=self._logger,
            queue=cast(Queue[Packet[None]], self._queue),
        )
        loop = ActorLoop(
            control=control,
            node=self._node,
            logger=self._logger,
            actor=self._actor,
            queue=self._queue,
        )
        saved_exc = loop.run()
        uniq_id = self._node.uniq_id
        fatal = saved_exc.value is not None and saved_exc.value.fatal
        child_threads: List[Thread] = []
        with self._global_state as gs:
            # Immediately stop spawning if fatal
            if fatal:
                gs.draining = True
            # Seal the queue to wake any waiters
            ctx = gs.actors[uniq_id]
            self._queue.seal()
            # Send child stops or kill child threads
            for child_id in ctx.child_ids:
                if child_id in gs.actors:
                    child_actx = gs.actors[child_id]
                    child_actx.queue.drain(StopPacket(), immediate=fatal)
                    child_threads.append(child_actx.thread)
                elif child_id in gs.tasks:
                    child_tctx = gs.tasks[child_id]
                    child_tctx.halt.set()
                    child_threads.append(child_tctx.thread)
        # Await child threads
        for thread in child_threads:
            thread.join()
        # Cleanup
        loop.cleanup(saved_exc=saved_exc)
        with self._global_state as gs:
            parent_ctx: Optional[ActorContext] = None
            if self._node.parent_id is not None:
                parent_ctx = gs.actors.get(self._node.parent_id)
            if parent_ctx is not None:
                parent_ctx.child_ids.remove(uniq_id)
                if fatal:
                    # Propagate fatal upwards
                    parent_ctx.queue.drain(StopPacket(), immediate=True)
                else:
                    # Simply report
                    pack: Packet[None] = ReportPacket(
                        child_id=uniq_id, child_exc=saved_exc.value
                    )
                    parent_ctx.queue.put(pack)
            if fatal:
                # Save exc
                assert saved_exc.value is not None
                gs.saved_excs.append(saved_exc.value)
            # Cleanup context
            del gs.actors[uniq_id]


class QueueSender[T](Sender[T]):
    def __init__(self, child_node: Node, child_queue: Queue[Packet[T]]):
        self._child_node = child_node
        self._child_queue = child_queue

    @override
    def dest(self) -> UniqId:
        return self._child_node.uniq_id

    @override
    def stop(self, immediate: bool) -> None:
        self._child_queue.drain(StopPacket(), immediate=immediate)

    @override
    def send(self, msg: T) -> None:
        self._child_queue.put(MessagePacket(msg))


class TaskSender(Sender[None]):
    def __init__(self, child_node: Node, child_halt: Event):
        self._child_node = child_node
        self._child_halt = child_halt

    @override
    def dest(self) -> UniqId:
        return self._child_node.uniq_id

    @override
    def stop(self, immediate: bool) -> None:
        self._child_halt.set()

    @override
    def send(self, msg: None) -> None:
        pass


class NullSender[T](Sender[T]):
    def __init__(self, child_id: UniqId):
        self._child_id = child_id

    @override
    def dest(self) -> UniqId:
        return self._child_id

    @override
    def stop(self, immediate: bool) -> None:
        pass

    @override
    def send(self, msg: T) -> None:
        pass


class ControlImpl(Control):
    def __init__(
        self,
        global_state: GlobalState,
        node: Node,
        logger: Logger,
        queue: Queue[Packet[None]],
    ):
        self._global_state = global_state
        self._node = node
        self._logger = logger
        self._queue = queue

    @override
    def stop(self, immediate: bool) -> None:
        self._queue.drain(StopPacket(), immediate=immediate)

    @override
    def spawn_actor(self, name: str, actor: Actor[T]) -> Sender[T]:
        with self._global_state as gs:
            child_id = gs.id_src
            gs.id_src = UniqId(gs.id_src + 1)
            if gs.draining:
                return NullSender(child_id=child_id)
            else:
                parent_id = self._node.uniq_id
                child_queue: Queue[Packet[T]] = Queue([StartPacket()])
                uniq_name = fmt_uniq_name(name, child_id)
                logger = self._logger.getChild(uniq_name)
                child_node = Node(
                    name=name,
                    uniq_id=child_id,
                    parent_id=parent_id,
                )
                lifecycle = ActorLifecycle(
                    global_state=self._global_state,
                    node=child_node,
                    logger=logger,
                    actor=actor,
                    queue=child_queue,
                )
                thread = Thread(name=uniq_name, target=lifecycle.run)
                context = ActorContext(
                    node=child_node,
                    thread=thread,
                    queue=child_queue,
                    child_ids=set(),
                )
                gs.actors[child_id] = context
                gs.actors[parent_id].child_ids.add(child_id)
                thread.start()
                return QueueSender(child_node=child_node, child_queue=child_queue)

    @override
    def spawn_task(self, name: str, task: Task) -> Sender[None]:
        with self._global_state as gs:
            child_id = gs.id_src
            gs.id_src = UniqId(gs.id_src + 1)
            if gs.draining:
                return NullSender(child_id=child_id)
            else:
                raise Exception("TODO")
                # parent_id = self._state.actor_id
                # uname = qual_name(name, child_id)
                # logger = self._state.logger.getChild(uname)
                # thread = Thread(name=uname, target=task.run, args=(logger,))
                # context = TaskContext(
                #     thread=thread,
                # )
                # gs.contexts[child_id] = context
                # gs.contexts[parent_id].child_ids.add(child_id)
                # thread.start()
                # return TaskSender(child_state)


class RootActor(Actor[None]):
    def __init__(self, global_state: GlobalState):
        self._global_state = global_state

    @override
    def on_stop(self, logger: Logger) -> None:
        with self._global_state as gs:
            gs.draining = True


def control(logger: Optional[Logger] = None) -> Control:
    global_state = Mutex(GlobalMutState.empty(logger=logger))
    root_name = "root"
    root_actor = RootActor(global_state)
    with global_state as gs:
        root_id = gs.id_src
        gs.id_src = UniqId(gs.id_src + 1)
        root_queue: Queue[Packet[None]] = Queue([StartPacket()])
        root_uname = fmt_uniq_name(root_name, root_id)
        root_logger = gs.logger.getChild(root_uname)
        root_node = Node(name=root_name, uniq_id=root_id, parent_id=None)
        lifecycle = ActorLifecycle(
            global_state=global_state,
            node=root_node,
            logger=root_logger,
            actor=root_actor,
            queue=root_queue,
        )
        thread = Thread(name=root_uname, target=lifecycle.run)
        context = ActorContext(
            node=root_node,
            thread=thread,
            queue=root_queue,
            child_ids=set(),
        )
        gs.actors[root_id] = context
        thread.start()
        return ControlImpl(
            global_state=global_state,
            node=root_node,
            logger=root_logger,
            queue=root_queue,
        )
