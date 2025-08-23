"""Tests for actor system error handling and edge cases."""

import time
from typing import Optional

from centipede.actor import (
    ActionException,
    Actor,
    ActorEnv,
    UniqId,
    system,
)


class ErrorActor(Actor[str]):
    """Actor that can generate exceptions for testing."""

    def __init__(self):
        self.started = False
        self.messages = []

    def on_start(self, env: ActorEnv) -> None:
        self.started = True

    def on_message(self, env: ActorEnv, value: str) -> None:
        if value == "fail":
            raise ValueError("Test error")
        self.messages.append(value)


class ReportingActor(Actor[str]):
    """Actor that spawns children and collects error reports."""

    def __init__(self):
        self.reports = []

    def on_start(self, env: ActorEnv) -> None:
        # Spawn a child that will fail
        child = ErrorActor()
        sender = env.control.spawn_actor("child", child)
        sender.send("fail")

    def on_report(
        self, env: ActorEnv, child_id: UniqId, exc: Optional[ActionException]
    ) -> None:
        self.reports.append((child_id, exc))
        env.logger.info(f"Child {child_id} reported: {exc}")


def test_child_error_reporting():
    """Test that child errors are properly reported to parents."""
    sys = system()

    parent = ReportingActor()
    sys.spawn_actor("parent", parent)

    # Give time for child to spawn and fail
    time.sleep(0.1)

    # Parent should have received error report
    assert len(parent.reports) == 1
    child_id, exc = parent.reports[0]
    assert exc is not None
    assert "Test error" in str(exc.exc)

    sys.stop(immediate=False)
    sys.wait()


def test_normal_operation_no_errors():
    """Test that normal operation produces no fatal errors."""
    sys = system()

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


def test_system_resilience():
    """Test that the system can handle multiple actors with mixed success."""
    sys = system()

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
