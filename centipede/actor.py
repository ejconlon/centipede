from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from threading import Condition, Event, Lock, Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Never,
    NewType,
    Optional,
    Set,
    TypeVar,
    override,
)

from centipede.spiny.common import Box

T = TypeVar("T")


UniqId = NewType("UniqId", int)


def fmt_uniq_name(name: str, uniq_id: UniqId) -> str:
    """Format a unique name by combining name and unique ID.

    Args:
        name: The base name.
        uniq_id: The unique identifier.

    Returns:
        Formatted string combining name and ID.
    """
    return f"{name}#{uniq_id}"


class Mutex[T]:
    """A thread-safe mutex wrapper that provides exclusive access to a value.

    Uses context manager protocol to automatically handle lock acquisition and release.
    """

    def __init__(self, value: T):
        """Initialize mutex with the given value.

        Args:
            value: The value to protect with mutual exclusion.
        """
        self._lock = Lock()
        self._value = value

    def __enter__(self) -> T:
        """Acquire lock and return the protected value.

        Returns:
            The protected value.
        """
        self._lock.acquire()
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the lock.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self._lock.release()


class Queue[T]:
    """A thread-safe queue with draining and sealing capabilities.

    Supports blocking get operations and can be put into a draining state
    where no new items can be added and existing items can be cleared.
    """

    def __init__(self, items: Optional[Iterable[T]] = None):
        """Initialize queue with optional initial items.

        Args:
            items: Optional iterable of initial items to add to the queue.
        """
        self._cv = Condition()
        self._items: deque[T] = deque()
        if items:
            self._items.extend(items)
        self._draining = False

    def put(self, item: T) -> None:
        """Add item to the queue if not draining.

        Args:
            item: The item to add to the queue.
        """
        with self._cv:
            if not self._draining:
                self._items.append(item)
                self._cv.notify_all()

    def drain(self, item: T, immediate: bool = False) -> None:
        """Put queue into draining state and optionally add a final item.

        Args:
            item: Final item to add to the queue.
            immediate: If True, clear existing items and put this item first.
                      If False, append this item to existing items.
        """
        with self._cv:
            if immediate:
                self._draining = True
                self._items.clear()
                self._items.appendleft(item)
            elif not self._draining:
                self._draining = True
                self._items.append(item)
            self._cv.notify_all()

    def seal(self) -> None:
        """Seal the queue by setting draining state and clearing all items."""
        with self._cv:
            self._draining = True
            self._items.clear()
            self._cv.notify_all()

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get the next item from the queue, blocking if empty.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            The next item from the queue, or None if queue is sealed or timeout occurred.
        """
        with self._cv:
            while not self._items and not self._draining:
                if not self._cv.wait(timeout=timeout):
                    return None  # Timeout occurred
            if self._items:
                return self._items.popleft()
            else:
                return None

    def _resume_wait(self) -> bool:
        """Check if wait operation should resume.

        Returns:
            True if queue is draining and empty.
        """
        return self._draining and not self._items

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the queue is drained and empty.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            True if the queue was drained and empty within the timeout, False otherwise.
        """
        with self._cv:
            if not self._resume_wait():
                return self._cv.wait_for(self._resume_wait, timeout=timeout)
            return True


class Action(Enum):
    """Enumeration of actor lifecycle actions.

    Defines the different phases and operations that can occur
    during an actor's lifecycle.
    """

    Start = 0
    Message = 1
    Handle = 2
    Stop = 3
    Cleanup = 4
    Report = 5
    Supervise = 6
    Run = 7


@dataclass(frozen=True)
class ActionException(Exception):
    """Exception that occurs during actor lifecycle actions.

    Wraps exceptions that occur during actor operations with context
    about which actor and action caused the exception.
    """

    name: str
    uniq_id: UniqId
    fatal: bool
    action: Action
    exc: Exception
    saved_exc: Optional[ActionException]


def is_fatal_exception(exc: BaseException) -> bool:
    """Check if an exception should be treated as fatal.

    Fatal exceptions cause the actor system to shutdown.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception is fatal.
    """
    if isinstance(exc, SystemExit):
        return True
    elif isinstance(exc, KeyboardInterrupt):
        return True
    elif isinstance(exc, ActionException):
        return exc.fatal
    else:
        return False


class Task(metaclass=ABCMeta):
    """Abstract base class for tasks that can be run by actors.

    Tasks are executed in their own threads and can be halted via an Event.
    """

    @abstractmethod
    def run(self, logger: Logger, halt: Event) -> None:
        """Execute the task.

        Args:
            logger: Logger for the task to use.
            halt: Event that will be set when the task should stop.
        """
        raise NotImplementedError


class Packet[T](metaclass=ABCMeta):
    """Abstract base class for messages sent between actors.

    Packets represent different types of communication in the actor system.
    """

    @staticmethod
    def start() -> Packet[T]:
        """Get a start packet instance.

        Returns:
            A start packet.
        """
        return _START_PACKET

    @staticmethod
    def stop() -> Packet[T]:
        """Get a stop packet instance.

        Returns:
            A stop packet.
        """
        return _STOP_PACKET


@dataclass(frozen=True)
class StartPacket(Packet[Any]):
    """Packet sent to start an actor."""

    pass


_START_PACKET: Packet[Any] = StartPacket()


@dataclass(frozen=True)
class MessagePacket[T](Packet[T]):
    """Packet containing a message value to be processed by an actor."""

    value: T


@dataclass(frozen=True)
class ReportPacket[T](Packet[T]):
    """Packet sent to report child actor completion status."""

    child_id: UniqId
    child_exc: Optional[ActionException]


@dataclass(frozen=True)
class StopPacket(Packet[Any]):
    """Packet sent to stop an actor."""

    pass


_STOP_PACKET: Packet[Any] = StopPacket()


class Sender[T](metaclass=ABCMeta):
    """Abstract interface for sending messages to actors or tasks.

    Provides methods to send messages and stop the recipient.
    """

    @abstractmethod
    def dest(self) -> UniqId:
        """Get the unique ID of the destination.

        Returns:
            The unique identifier of the message recipient.
        """
        raise NotImplementedError

    @abstractmethod
    def send(self, msg: T) -> None:
        """Send a message to the recipient.

        Args:
            msg: The message to send.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, immediate: bool = False) -> None:
        """Stop the recipient.

        Args:
            immediate: If True, stop immediately; if False, allow graceful shutdown.
        """
        raise NotImplementedError

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the recipient to terminate.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            True if the recipient terminated within the timeout, False otherwise.
        """
        raise NotImplementedError


class Control(metaclass=ABCMeta):
    """Abstract interface for controlling actors and tasks.

    Provides methods to spawn new actors and tasks, and stop the system.
    """

    @abstractmethod
    def spawn_actor(self, name: str, actor: Actor[T]) -> Sender[T]:
        """Spawn a new actor.

        Args:
            name: The name for the new actor.
            actor: The actor instance to spawn.

        Returns:
            A sender for communicating with the spawned actor.
        """
        raise NotImplementedError

    @abstractmethod
    def spawn_task(self, name: str, task: Task) -> Sender[Never]:
        """Spawn a new task.

        Args:
            name: The name for the new task.
            task: The task instance to spawn.

        Returns:
            A sender for controlling the spawned task.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, immediate: bool = False) -> None:
        """Stop the actor system.

        Args:
            immediate: If True, stop immediately; if False, allow graceful shutdown.
        """
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class ActorEnv:
    """Environment provided to actors during their lifecycle.

    Contains the logger and control interface for the actor.
    """

    logger: Logger
    control: Control


def on_report_stop_fatal(env: ActorEnv, exc: Optional[ActionException]):
    """Report handler that stops the system only on fatal exceptions.

    Args:
        env: The actor environment.
        exc: Optional exception that occurred in the child.
    """
    if exc is not None:
        env.control.stop(immediate=exc.fatal)


def on_report_stop_always(env: ActorEnv, exc: Optional[ActionException]):
    """Report handler that always stops the system when a child reports.

    Args:
        env: The actor environment.
        exc: Optional exception that occurred in the child.
    """
    fatal = exc is not None and exc.fatal
    env.control.stop(immediate=fatal)


# Lifecycle:
#   Start
#   Loop: Message/Report, Handle uncaught exceptions
#   Stop: prepare for child cleanup
#   Cleanup: children cleaned up
#   Report to parent
class Actor[T](metaclass=ABCMeta):
    """Abstract base class for actors in the actor system.

    Actors are concurrent entities that process messages and can spawn children.
    The lifecycle is: Start -> Message/Report loop -> Stop -> Cleanup.
    """

    def on_start(self, env: ActorEnv) -> None:
        """Called when the actor starts.

        Args:
            env: The actor environment with logger and control.
        """
        return

    def on_message(self, env: ActorEnv, value: T) -> None:
        """Called when the actor receives a message.

        Args:
            env: The actor environment.
            value: The message value.
        """
        return

    def on_report(
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        """Called when a child actor reports completion.

        Args:
            env: The actor environment.
            child_id: The unique ID of the child.
            exc: Optional exception from the child.
        """
        on_report_stop_fatal(env, exc)

    def on_handle(self, env: ActorEnv, exc: Exception) -> None:
        """Called to handle exceptions during actor lifecycle.

        Args:
            env: The actor environment.
            exc: The exception to handle.
        """
        raise exc

    def on_stop(self, logger: Logger) -> None:
        """Called when the actor is stopping.

        Args:
            logger: The logger to use.
        """
        return

    def on_cleanup(self, logger: Logger) -> None:
        """Called to clean up resources after children have stopped.

        Args:
            logger: The logger to use.
        """
        return


@dataclass(frozen=True)
class Node:
    """Represents a node in the actor hierarchy.

    Contains identifying information for actors and tasks.
    """

    name: str
    uniq_id: UniqId
    parent_id: Optional[UniqId]


class ActorLoop[T]:
    """Main event loop for processing actor lifecycle events.

    Handles Start, Message, Report, and Stop packets in sequence.
    """

    def __init__(
        self,
        control: Control,
        node: Node,
        logger: Logger,
        actor: Actor[T],
        queue: Queue[Packet[T]],
    ):
        self._node = node
        self._logger = logger
        self._env = ActorEnv(logger=logger, control=control)
        self._stopped = False
        self._actor = actor
        self._queue = queue

    def run(self) -> Box[Optional[ActionException]]:
        """Run the actor loop until completion.

        Returns:
            Box containing any exception that occurred.
        """
        saved_exc: Box[Optional[ActionException]] = Box(None)
        while saved_exc.value is None and not self._stopped:
            self._step(saved_exc=saved_exc)
        if not self._stopped:
            self._stop(saved_exc=saved_exc)
        return saved_exc

    def cleanup(self, saved_exc: Box[Optional[ActionException]]) -> None:
        """Clean up the actor after stopping.

        Args:
            saved_exc: Box to store any cleanup exceptions.
        """
        self._logger.debug(
            "Actor %s#%d cleaning up", self._node.name, self._node.uniq_id
        )
        try:
            self._actor.on_cleanup(logger=self._logger)
            self._logger.debug(
                "Actor %s#%d cleaned up successfully",
                self._node.name,
                self._node.uniq_id,
            )
        except Exception as exc:
            self._logger.debug(
                "Actor %s#%d failed to clean up: %s",
                self._node.name,
                self._node.uniq_id,
                exc,
            )
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
        self._logger.debug("Actor %s#%d starting", self._node.name, self._node.uniq_id)
        try:
            self._actor.on_start(env=self._env)
            self._logger.debug(
                "Actor %s#%d started successfully", self._node.name, self._node.uniq_id
            )
        except Exception as exc:
            self._logger.debug(
                "Actor %s#%d failed to start: %s",
                self._node.name,
                self._node.uniq_id,
                exc,
            )
            saved_exc.value = self._except(
                action=Action.Start, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _message(self, saved_exc: Box[Optional[ActionException]], value: T) -> None:
        self._logger.debug(
            "Actor %s#%d processing message: %s",
            self._node.name,
            self._node.uniq_id,
            value,
        )
        try:
            self._actor.on_message(env=self._env, value=value)
            self._logger.debug(
                "Actor %s#%d processed message successfully",
                self._node.name,
                self._node.uniq_id,
            )
        except Exception as exc:
            self._logger.debug(
                "Actor %s#%d failed to process message: %s",
                self._node.name,
                self._node.uniq_id,
                exc,
            )
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
        self._logger.debug(
            "Actor %s#%d handling report from child %d: %s",
            self._node.name,
            self._node.uniq_id,
            child_id,
            child_exc,
        )
        try:
            self._actor.on_report(env=self._env, child_id=child_id, exc=child_exc)
            self._logger.debug(
                "Actor %s#%d handled child report successfully",
                self._node.name,
                self._node.uniq_id,
            )
        except Exception as exc:
            self._logger.debug(
                "Actor %s#%d failed to handle child report: %s",
                self._node.name,
                self._node.uniq_id,
                exc,
            )
            saved_exc.value = self._except(
                action=Action.Report, exc=exc, saved_exc=saved_exc.value
            )

        self._handle(saved_exc=saved_exc)

    def _handle(self, saved_exc: Box[Optional[ActionException]]) -> None:
        if saved_exc.value is not None and not is_fatal_exception(saved_exc.value):
            self._logger.debug(
                "Actor %s#%d handling exception: %s",
                self._node.name,
                self._node.uniq_id,
                saved_exc.value.exc,
            )
            try:
                self._actor.on_handle(env=self._env, exc=saved_exc.value.exc)
                self._logger.debug(
                    "Actor %s#%d handled exception successfully",
                    self._node.name,
                    self._node.uniq_id,
                )
            except Exception as exc:
                self._logger.debug(
                    "Actor %s#%d failed to handle exception: %s",
                    self._node.name,
                    self._node.uniq_id,
                    exc,
                )
                saved_exc.value = self._except(
                    action=Action.Handle, exc=exc, saved_exc=saved_exc.value
                )

    def _stop(self, saved_exc: Box[Optional[ActionException]]) -> None:
        self._logger.debug("Actor %s#%d stopping", self._node.name, self._node.uniq_id)
        try:
            self._actor.on_stop(logger=self._logger)
            self._logger.debug(
                "Actor %s#%d stopped successfully", self._node.name, self._node.uniq_id
            )
        except Exception as exc:
            self._logger.debug(
                "Actor %s#%d failed to stop: %s",
                self._node.name,
                self._node.uniq_id,
                exc,
            )
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
class Context(metaclass=ABCMeta):
    """Abstract base class for runtime contexts.

    Contexts represent different types of execution units in the actor system.
    """

    node: Node
    thread: Thread


@dataclass(frozen=True, eq=False)
class ActorContext[T](Context):
    """Runtime context for an active actor.

    Contains the actor's thread, message queue, and child tracking.
    """

    node: Node
    thread: Thread
    queue: Queue[Packet[T]]
    child_ids: Set[UniqId]


@dataclass(frozen=True, eq=False)
class TaskContext(Context):
    """Runtime context for an active task.

    Contains the task's thread and halt event for stopping.
    """

    node: Node
    thread: Thread
    halt: Event


@dataclass(frozen=False)
class GlobalMutState:
    """Global mutable state for the actor system.

    Tracks all active actors, tasks, and system state.
    """

    id_src: UniqId
    contexts: Dict[UniqId, Context]
    draining: bool
    saved_excs: List[ActionException]
    logger: Logger

    @staticmethod
    def empty(logger: Logger) -> GlobalMutState:
        """Create an empty global state.

        Args:
            logger: Logger to use.

        Returns:
            New empty global state.
        """
        return GlobalMutState(
            id_src=UniqId(0),
            contexts={},
            draining=False,
            saved_excs=[],
            logger=logger,
        )


type GlobalState = Mutex[GlobalMutState]


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
            queue=self._queue,
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
            ctx = gs.contexts[uniq_id]
            assert isinstance(ctx, ActorContext)
            self._queue.seal()
            # Send child stops or kill child threads
            for child_id in ctx.child_ids:
                if child_id in gs.contexts:
                    child_ctx = gs.contexts[child_id]
                    if isinstance(child_ctx, ActorContext):
                        child_ctx.queue.drain(Packet.stop(), immediate=True)
                        child_threads.append(child_ctx.thread)
                    elif isinstance(child_ctx, TaskContext):
                        child_ctx.halt.set()
                        child_threads.append(child_ctx.thread)
        # Await child threads
        for thread in child_threads:
            thread.join()
        # Cleanup
        loop.cleanup(saved_exc=saved_exc)
        with self._global_state as gs:
            parent_ctx: Optional[ActorContext] = None
            if self._node.parent_id is not None:
                parent_context = gs.contexts.get(self._node.parent_id)
                if isinstance(parent_context, ActorContext):
                    parent_ctx = parent_context
            if parent_ctx is not None:
                parent_ctx.child_ids.remove(uniq_id)
                if fatal:
                    # Propagate fatal upwards
                    parent_ctx.queue.drain(Packet.stop(), immediate=True)
                else:
                    # Simply report
                    pack: Packet[Never] = ReportPacket(
                        child_id=uniq_id, child_exc=saved_exc.value
                    )
                    parent_ctx.queue.put(pack)
            if fatal:
                # Save exc
                assert saved_exc.value is not None
                gs.saved_excs.append(saved_exc.value)
            # Cleanup context
            del gs.contexts[uniq_id]


class TaskLifecycle:
    def __init__(
        self,
        global_state: GlobalState,
        node: Node,
        logger: Logger,
        task: Task,
        halt: Event,
    ):
        self._global_state = global_state
        self._node = node
        self._logger = logger
        self._task = task
        self._halt = halt

    def run(self) -> None:
        saved_exc: Box[Optional[ActionException]] = Box(None)
        try:
            self._task.run(logger=self._logger, halt=self._halt)
        except Exception as exc:
            saved_exc.value = ActionException(
                name=self._node.name,
                uniq_id=self._node.uniq_id,
                fatal=is_fatal_exception(exc),
                action=Action.Run,
                exc=exc,
                saved_exc=None,
            )
        uniq_id = self._node.uniq_id
        fatal = saved_exc.value is not None and saved_exc.value.fatal
        with self._global_state as gs:
            # Immediately stop spawning if fatal
            if fatal:
                gs.draining = True
            # Set halt to wake any waiters
            self._halt.set()
        with self._global_state as gs:
            parent_ctx: Optional[ActorContext] = None
            if self._node.parent_id is not None:
                parent_context = gs.contexts.get(self._node.parent_id)
                if isinstance(parent_context, ActorContext):
                    parent_ctx = parent_context
            if parent_ctx is not None:
                parent_ctx.child_ids.remove(uniq_id)
                if fatal:
                    # Propagate fatal upwards
                    parent_ctx.queue.drain(Packet.stop(), immediate=True)
                else:
                    # Simply report
                    pack: Packet[Never] = ReportPacket(
                        child_id=uniq_id, child_exc=saved_exc.value
                    )
                    parent_ctx.queue.put(pack)
            if fatal:
                # Save exc
                assert saved_exc.value is not None
                gs.saved_excs.append(saved_exc.value)
            # Cleanup context
            del gs.contexts[uniq_id]


class QueueSender[T](Sender[T]):
    """Sender implementation that sends messages via actor queues."""

    def __init__(
        self, child_node: Node, child_queue: Queue[Packet[T]], child_thread: Thread
    ):
        self._child_node = child_node
        self._child_queue = child_queue
        self._child_thread = child_thread

    @override
    def dest(self) -> UniqId:
        return self._child_node.uniq_id

    @override
    def stop(self, immediate: bool = False) -> None:
        self._child_queue.drain(Packet.stop(), immediate=immediate)

    @override
    def send(self, msg: T) -> None:
        self._child_queue.put(MessagePacket(msg))

    @override
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the actor to terminate.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            True if the actor terminated within the timeout, False otherwise.
        """
        self._child_thread.join(timeout=timeout)
        return not self._child_thread.is_alive()


class TaskSender(Sender[Never]):
    """Sender implementation for controlling tasks via halt events."""

    def __init__(self, child_node: Node, child_halt: Event, child_thread: Thread):
        self._child_node = child_node
        self._child_halt = child_halt
        self._child_thread = child_thread

    @override
    def dest(self) -> UniqId:
        return self._child_node.uniq_id

    @override
    def stop(self, immediate: bool = False) -> None:
        self._child_halt.set()

    @override
    def send(self, msg: None) -> None:
        pass

    @override
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the task to terminate.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            True if the task terminated within the timeout, False otherwise.
        """
        # For tasks, we wait on the thread since they don't have a message queue lifecycle
        self._child_thread.join(timeout=timeout)
        return not self._child_thread.is_alive()


class NullSender[T](Sender[T]):
    """No-op sender used when system is draining and spawning is disabled."""

    def __init__(self, child_id: UniqId):
        self._child_id = child_id

    @override
    def dest(self) -> UniqId:
        return self._child_id

    @override
    def stop(self, immediate: bool = False) -> None:
        pass

    @override
    def send(self, msg: T) -> None:
        pass

    @override
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for termination (no-op for null sender).

        Args:
            timeout: Maximum time to wait in seconds (ignored).

        Returns:
            Always returns True as null senders have no backing thread.
        """
        return True


class ControlImpl(Control):
    """Implementation of the Control interface.

    Manages actor and task spawning within the actor system.
    """

    def __init__(
        self,
        global_state: GlobalState,
        node: Node,
        logger: Logger,
        queue: Queue[Packet[T]],
    ):
        self._global_state = global_state
        self._node = node
        self._logger = logger
        self._queue = queue

    @override
    def stop(self, immediate: bool = False) -> None:
        self._queue.drain(Packet.stop(), immediate=immediate)

    @override
    def spawn_actor(self, name: str, actor: Actor[T]) -> Sender[T]:
        with self._global_state as gs:
            child_id = gs.id_src
            gs.id_src = UniqId(gs.id_src + 1)
            if gs.draining:
                return NullSender(child_id=child_id)
            else:
                parent_id = self._node.uniq_id
                child_queue: Queue[Packet[T]] = Queue([Packet.start()])
                child_uname = fmt_uniq_name(name, child_id)
                child_logger = gs.logger.getChild(fmt_uniq_name(name, child_id))
                child_node = Node(
                    name=name,
                    uniq_id=child_id,
                    parent_id=parent_id,
                )
                lifecycle = ActorLifecycle(
                    global_state=self._global_state,
                    node=child_node,
                    logger=child_logger,
                    actor=actor,
                    queue=child_queue,
                )
                thread = Thread(name=child_uname, target=lifecycle.run)
                context = ActorContext(
                    node=child_node,
                    thread=thread,
                    queue=child_queue,
                    child_ids=set(),
                )
                gs.contexts[child_id] = context
                parent_context = gs.contexts[parent_id]
                assert isinstance(parent_context, ActorContext)
                parent_context.child_ids.add(child_id)
                thread.start()
                return QueueSender(
                    child_node=child_node, child_queue=child_queue, child_thread=thread
                )

    @override
    def spawn_task(self, name: str, task: Task) -> Sender[Never]:
        with self._global_state as gs:
            child_id = gs.id_src
            gs.id_src = UniqId(gs.id_src + 1)
            if gs.draining:
                return NullSender(child_id=child_id)
            else:
                parent_id = self._node.uniq_id
                child_uname = fmt_uniq_name(name, child_id)
                child_logger = gs.logger.getChild(fmt_uniq_name(name, child_id))
                child_node = Node(
                    name=name,
                    uniq_id=child_id,
                    parent_id=parent_id,
                )
                child_halt = Event()
                lifecycle = TaskLifecycle(
                    global_state=self._global_state,
                    node=child_node,
                    logger=child_logger,
                    task=task,
                    halt=child_halt,
                )
                thread = Thread(name=child_uname, target=lifecycle.run)
                context = TaskContext(
                    node=child_node,
                    thread=thread,
                    halt=child_halt,
                )
                gs.contexts[child_id] = context
                parent_context = gs.contexts[parent_id]
                assert isinstance(parent_context, ActorContext)
                parent_context.child_ids.add(child_id)
                thread.start()
                return TaskSender(
                    child_node=child_node, child_halt=child_halt, child_thread=thread
                )


class System(Control):
    """Enhanced Control interface for managing an actor system.

    Provides additional functionality beyond the basic Control interface,
    including waiting for shutdown and monitoring system status.
    """

    def __init__(
        self, global_state: GlobalState, root_thread: Thread, control: Control
    ):
        """Initialize the system.

        Args:
            global_state: The global system state.
            root_thread: The root actor thread.
            control: The underlying control implementation.
        """
        self._global_state = global_state
        self._root_thread = root_thread
        self._control = control

    @override
    def spawn_actor(self, name: str, actor: Actor[T]) -> Sender[T]:
        """Spawn a new actor.

        Args:
            name: The name for the new actor.
            actor: The actor instance to spawn.

        Returns:
            A sender for communicating with the spawned actor.
        """
        return self._control.spawn_actor(name, actor)

    @override
    def spawn_task(self, name: str, task: Task) -> Sender[Never]:
        """Spawn a new task.

        Args:
            name: The name for the new task.
            task: The task instance to spawn.

        Returns:
            A sender for controlling the spawned task.
        """
        return self._control.spawn_task(name, task)

    @override
    def stop(self, immediate: bool = False) -> None:
        """Stop the actor system.

        Args:
            immediate: If True, stop immediately; if False, allow graceful shutdown.
        """
        self._control.stop(immediate)

    def wait(self, timeout: Optional[float] = None) -> List[ActionException]:
        """Wait for the actor system to shutdown.

        Blocks until all actors and tasks have completed and returns
        any fatal exceptions that occurred during execution.

        Args:
            timeout: Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            List of fatal exceptions that occurred during system execution.

        Raises:
            TimeoutError: If the timeout expires before system shutdown completes.
        """
        # Wait for the root thread to complete
        with self._global_state as gs:
            gs.logger.debug("Waiting for system shutdown (timeout=%s)", timeout)

        self._root_thread.join(timeout=timeout)

        # Check if thread is still alive (timeout occurred)
        if self._root_thread.is_alive():
            with self._global_state as gs:
                gs.logger.debug("System shutdown timed out after %s seconds", timeout)
            raise TimeoutError(f"System did not shut down within {timeout} seconds")

        # Return any saved exceptions
        with self._global_state as gs:
            saved_excs = gs.saved_excs.copy()
            if saved_excs:
                gs.logger.debug(
                    "System shutdown with %d saved exceptions", len(saved_excs)
                )
            else:
                gs.logger.debug("System shutdown cleanly with no exceptions")
            return saved_excs

    def thread_count(self) -> int:
        """Get the number of currently running threads.

        Returns:
            The number of active actor and task threads.
        """
        with self._global_state as gs:
            return len(gs.contexts)


class RootActor(Actor[Never]):
    """Root actor that manages the actor system lifecycle.

    Sets the system to draining state when it stops.
    """

    def __init__(self, global_state: GlobalState):
        """Initialize with global state reference.

        Args:
            global_state: The global system state.
        """
        self._global_state = global_state

    @override
    def on_stop(self, logger: Logger) -> None:
        """Set the system to draining when root actor stops.

        Args:
            logger: The logger to use.
        """
        with self._global_state as gs:
            gs.draining = True


def new_system(name: str = "system") -> System:
    """Create and start a new actor system.

    Returns:
        A System interface for the actor system.
    """
    logger = logging.getLogger(name)
    global_state = Mutex(GlobalMutState.empty(logger=logger))
    root_name = "root"
    root_actor = RootActor(global_state)
    with global_state as gs:
        gs.logger.debug("Creating actor system")
        root_id = gs.id_src
        gs.id_src = UniqId(gs.id_src + 1)
        root_queue: Queue[Packet[Never]] = Queue([Packet.start()])
        root_uname = fmt_uniq_name(root_name, root_id)
        root_logger = gs.logger.getChild(fmt_uniq_name(root_name, root_id))
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
        gs.contexts[root_id] = context
        gs.logger.debug("Starting root actor thread: %s", root_uname)
        thread.start()
        control = ControlImpl(
            global_state=global_state,
            node=root_node,
            logger=root_logger,
            queue=root_queue,
        )
        gs.logger.debug("Actor system created successfully")
        return System(
            global_state=global_state,
            root_thread=thread,
            control=control,
        )


class PairActor[T](Actor[Never]):
    """Actor that spawns a producer-consumer pair.

    Creates a consumer actor and a producer task that sends messages to it.
    """

    def __init__(
        self,
        consumer_name: str,
        consumer_actor: Actor[T],
        producer_name: str,
        mk_producer_task: Callable[[Sender[T]], Task],
    ):
        self._consumer_name = consumer_name
        self._consumer_actor = consumer_actor
        self._producer_name = producer_name
        self._mk_producer_task = mk_producer_task

    @override
    def on_start(self, env: ActorEnv) -> None:
        sender = env.control.spawn_actor(
            name=self._consumer_name, actor=self._consumer_actor
        )
        producer_task = self._mk_producer_task(sender)
        env.control.spawn_task(name=self._producer_name, task=producer_task)

    @override
    def on_report(
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        on_report_stop_always(env, exc)


class Callback[T](metaclass=ABCMeta):
    """Abstract callback interface for external event sources.

    Allows external systems to register with actors for message delivery.
    """

    @abstractmethod
    def register(self, sender: Sender[T]) -> None:
        """Register a sender to receive callbacks.

        Args:
            sender: The sender to register for callbacks.
        """
        raise NotImplementedError

    @abstractmethod
    def unregister(self) -> None:
        """Unregister from receiving callbacks."""
        raise NotImplementedError

    def produce(self, consumer_name: str, consumer_actor: Actor[T]) -> Actor[Never]:
        """Create an actor that connects this callback to a consumer.

        Args:
            consumer_name: Name for the consumer actor.
            consumer_actor: The actor that will consume callback messages.

        Returns:
            An actor that manages the callback connection.
        """
        return CallbackActor(
            consumer_name=consumer_name, consumer_actor=consumer_actor, cb=self
        )


class CallbackActor[T](Actor[Never]):
    """Actor that manages the connection between a callback and consumer.

    Registers with a callback to receive external events and forwards
    them to a consumer actor.
    """

    def __init__(self, consumer_name: str, consumer_actor: Actor[T], cb: Callback[T]):
        self._consumer_name = consumer_name
        self._consumer_actor = consumer_actor
        self._cb = cb

    @override
    def on_start(self, env: ActorEnv) -> None:
        sender = env.control.spawn_actor(
            name=self._consumer_name, actor=self._consumer_actor
        )
        self._cb.register(sender)

    @override
    def on_report(
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        on_report_stop_always(env, exc)

    @override
    def on_stop(self, logger: Logger):
        self._cb.unregister()
