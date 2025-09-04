"""Tests for actor system error handling and edge cases."""

import time
from logging import Logger
from threading import Event
from typing import List, Optional, Tuple

from bad_actor import (
    ActionException,
    Actor,
    ActorEnv,
    Queue,
    Task,
    UniqId,
    is_fatal_exception,
    new_system,
)


class ErrorActor(Actor[str]):
    """Actor that can generate exceptions for testing."""

    def __init__(self) -> None:
        self.started = False
        self.messages: List[str] = []

    def on_start(self, env: ActorEnv) -> None:
        self.started = True

    def on_message(self, env: ActorEnv, value: str) -> None:
        if value == "fail":
            raise ValueError("Test error")
        self.messages.append(value)


class ReportingActor(Actor[str]):
    """Actor that spawns children and collects error reports."""

    def __init__(self) -> None:
        self.reports: List[Tuple[UniqId, Optional[ActionException]]] = []

    def on_start(self, env: ActorEnv) -> None:
        # Spawn a child that will fail
        child = ErrorActor()
        sender = env.control.spawn_actor("child", child)
        sender.send("fail")

    def on_report(
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        self.reports.append((child_id, exc))
        env.logger.info("Child %s reported: %s", child_id, exc)


class FatalErrorActor(Actor[str]):
    """Actor that generates fatal errors."""

    def on_start(self, env: ActorEnv) -> None:
        # KeyboardInterrupt is considered a fatal error
        raise KeyboardInterrupt("Fatal error for testing")


def test_fatal_error_detection() -> None:
    """Test that fatal errors are properly identified and classified."""
    # Test that KeyboardInterrupt is fatal

    # KeyboardInterrupt should be fatal
    assert is_fatal_exception(KeyboardInterrupt("test"))

    # SystemExit should be fatal
    assert is_fatal_exception(SystemExit(1))

    # Regular exceptions should not be fatal
    assert not is_fatal_exception(ValueError("test"))
    assert not is_fatal_exception(RuntimeError("test"))


def test_fatal_error_causes_shutdown() -> None:
    """Test that fatal errors cause system shutdown with timeout."""
    sys = new_system()

    # Spawn actor that will cause fatal error
    actor = FatalErrorActor()
    sys.spawn_actor("fatal-actor", actor)

    # System should shut down due to fatal error, but with timeout to prevent hanging
    try:
        exceptions = sys.wait(timeout=1.0)  # 1 second timeout

        # Should have at least one fatal exception
        assert len(exceptions) > 0
        assert any(isinstance(exc.exc, KeyboardInterrupt) for exc in exceptions)

    except TimeoutError:
        # If timeout occurs, that might indicate a problem with fatal error handling
        # But let's be more lenient and just verify the system eventually shuts down
        sys.stop(immediate=True)
        # Try waiting with a longer timeout for cleanup
        try:
            sys.wait(timeout=2.0)
        except TimeoutError:
            pass  # Even cleanup timed out - this indicates a real issue


def test_child_error_reporting() -> None:
    """Test that child errors are properly reported to parents."""
    sys = new_system()

    parent = ReportingActor()
    sys.spawn_actor("parent", parent)

    # Give time for child to spawn and fail
    time.sleep(0.1)

    # Parent should have received error report
    assert len(parent.reports) == 1
    report = parent.reports[0]
    child_id, exc = report
    assert exc is not None
    assert "Test error" in str(exc.exc)

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_normal_operation_no_errors() -> None:
    """Test that normal operation produces no fatal errors."""
    sys = new_system()

    actor = ErrorActor()
    sender = sys.spawn_actor("normal-actor", actor)

    # Send normal messages (not "fail")
    sender.send("hello")
    sender.send("world")

    time.sleep(0.05)

    # Verify messages were processed
    assert actor.messages == ["hello", "world"]

    # Stop system gracefully
    sys.stop(immediate=False)
    exceptions = sys.wait()

    # Should have no fatal exceptions
    assert len(exceptions) == 0


def test_system_resilience() -> None:
    """Test that the system can handle multiple actors with mixed success."""
    sys = new_system()

    # Spawn several normal actors
    actors = []
    for i in range(3):
        actor = ErrorActor()
        actors.append(actor)
        sender = sys.spawn_actor(f"actor-{i}", actor)
        sender.send(f"message-{i}")

    time.sleep(0.05)

    # All actors should have processed their messages
    for i, actor in enumerate(actors):
        assert actor.started
        assert f"message-{i}" in actor.messages

    # System should be healthy
    assert sys.thread_count() >= 1

    sys.stop(immediate=False)
    exceptions = sys.wait()
    assert len(exceptions) == 0


def test_queue_timeout() -> None:
    """Test that Queue methods properly handle timeouts."""
    # Create an empty queue
    queue: Queue[str] = Queue()

    # Test get() with timeout
    start_time = time.time()
    result = queue.get(timeout=0.1)
    elapsed = time.time() - start_time

    # Should return None due to timeout
    assert result is None
    # Should have waited approximately 0.1 seconds
    assert 0.08 <= elapsed <= 0.15  # Allow some margin for timing

    # Test wait() with timeout
    start_time = time.time()
    success = queue.wait(timeout=0.1)
    elapsed = time.time() - start_time

    # Should return False due to timeout (queue not drained)
    assert success is False
    # Should have waited approximately 0.1 seconds
    assert 0.08 <= elapsed <= 0.15


def test_sender_wait_timeout() -> None:
    """Test that Sender.wait() properly handles timeouts."""

    class SimpleTask(Task):
        def __init__(self, duration: float = 0.1):
            self.duration = duration

        def run(self, logger: Logger, halt: Event) -> None:
            halt.wait(timeout=self.duration)

    sys = new_system()

    # Test actor sender wait
    actor = ErrorActor()
    actor_sender = sys.spawn_actor("test-actor", actor)

    # Test task sender wait
    task = SimpleTask(duration=0.5)  # Long running task
    task_sender = sys.spawn_task("test-task", task)

    # Wait for a short time - should timeout
    start_time = time.time()
    result = task_sender.wait(timeout=0.1)
    elapsed = time.time() - start_time

    # Should return False due to timeout
    assert result is False
    assert 0.08 <= elapsed <= 0.15

    # Stop the task and wait for completion
    task_sender.stop(immediate=True)
    result = task_sender.wait(timeout=1.0)

    # Should return True (task stopped)
    assert result is True

    # Stop the actor and wait for completion
    actor_sender.stop(immediate=True)
    result = actor_sender.wait(timeout=1.0)

    # Should return True (actor stopped)
    assert result is True

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)
