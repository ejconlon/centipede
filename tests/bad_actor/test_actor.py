"""Tests for the actor system basic functionality."""

import threading
import time
from typing import List

from bad_actor import (
    Actor,
    ActorEnv,
    Sender,
    Task,
    new_system,
)


class SimpleActor(Actor[str]):
    """Simple test actor that logs received messages."""

    def __init__(self):
        self.messages: List[str] = []
        self.started = False
        self.stopped = False

    def on_start(self, env: ActorEnv) -> None:
        self.started = True
        env.logger.info("SimpleActor started")

    def on_message(self, env: ActorEnv, value: str) -> None:
        self.messages.append(value)
        env.logger.info("Received message: %s", value)

    def on_stop(self, logger) -> None:
        self.stopped = True
        logger.info("SimpleActor stopped")


class CountingActor(Actor[int]):
    """Actor that counts received numbers."""

    def __init__(self):
        self.total = 0
        self.count = 0

    def on_message(self, env: ActorEnv, value: int) -> None:
        self.total += value
        self.count += 1


class SimpleTask(Task):
    """Simple test task that runs for a short duration."""

    def __init__(self, duration: float = 0.1):
        self.duration = duration
        self.completed = False
        self.was_halted = False

    def run(self, logger, halt) -> None:
        logger.info("SimpleTask starting")

        # Wait for either duration or halt event
        if halt.wait(timeout=self.duration):
            self.was_halted = True
            logger.info("SimpleTask halted")
        else:
            self.completed = True
            logger.info("SimpleTask completed normally")


class ProducerTask(Task):
    """Task that sends messages to an actor."""

    def __init__(self, sender: Sender[str], messages: List[str], delay: float = 0.05):
        self.sender = sender
        self.messages = messages
        self.delay = delay
        self.sent_count = 0

    def run(self, logger, halt) -> None:
        logger.info("ProducerTask starting")

        for msg in self.messages:
            if halt.is_set():
                break
            self.sender.send(msg)
            self.sent_count += 1
            logger.info("Sent message: %s", msg)
            time.sleep(self.delay)


def test_system_startup_shutdown():
    """Test basic system startup and shutdown."""
    # Create system
    sys = new_system()

    # Verify system is running
    assert sys.thread_count() == 1  # Just the root actor

    # Stop system gracefully
    sys.stop(immediate=False)

    # Wait for shutdown
    exceptions = sys.wait()
    assert len(exceptions) == 0  # No exceptions should occur


def test_system_immediate_shutdown():
    """Test immediate system shutdown."""
    sys = new_system()

    # Stop system immediately
    sys.stop(immediate=True)

    # Wait for shutdown
    exceptions = sys.wait()
    assert len(exceptions) == 0


def test_spawn_actor_basic():
    """Test basic actor spawning and lifecycle."""
    sys = new_system()

    # Create test actor
    actor = SimpleActor()

    # Spawn actor
    sys.spawn_actor("test-actor", actor)

    # Verify actor was spawned
    assert sys.thread_count() == 2  # Root + test actor

    # Give actor time to start
    time.sleep(0.05)
    assert actor.started

    # Stop system
    sys.stop(immediate=False)
    exceptions = sys.wait()

    # Verify actor stopped
    assert actor.stopped
    assert len(exceptions) == 0


def test_message_passing():
    """Test sending messages to actors."""
    sys = new_system()

    actor = SimpleActor()
    sender = sys.spawn_actor("test-actor", actor)

    # Send messages
    messages = ["hello", "world", "test"]
    for msg in messages:
        sender.send(msg)

    # Give time for messages to be processed
    time.sleep(0.1)

    # Verify messages were received
    assert actor.messages == messages

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_multiple_actors():
    """Test spawning multiple actors."""
    sys = new_system()

    # Spawn multiple actors
    actors = [SimpleActor() for _ in range(3)]
    senders = []

    for i, actor in enumerate(actors):
        sender = sys.spawn_actor(f"actor-{i}", actor)
        senders.append(sender)

    assert sys.thread_count() == 4  # Root + 3 actors

    # Send messages to each actor
    for i, sender in enumerate(senders):
        sender.send(f"message-{i}")

    time.sleep(0.05)

    # Verify each actor received its message
    for i, actor in enumerate(actors):
        assert actor.messages == [f"message-{i}"]

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_spawn_task_basic():
    """Test basic task spawning."""
    sys = new_system()

    # Create and spawn task
    task = SimpleTask(duration=0.1)
    sys.spawn_task("test-task", task)

    assert sys.thread_count() == 2  # Root + task

    # Wait for task to complete (task duration is 0.1, so wait a bit longer)
    time.sleep(0.15)

    assert task.completed
    assert not task.was_halted

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_task_halt():
    """Test halting a task before completion."""
    sys = new_system()

    # Create long-running task
    task = SimpleTask(duration=1.0)  # 1 second
    sender = sys.spawn_task("test-task", task)

    # Give task time to start, then stop it
    time.sleep(0.05)
    sender.stop(immediate=True)

    # Wait a bit more
    time.sleep(0.05)

    # Task should have been halted
    assert task.was_halted
    assert not task.completed

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_producer_consumer():
    """Test producer task sending messages to consumer actor."""
    sys = new_system()

    # Create consumer actor
    consumer = SimpleActor()
    consumer_sender = sys.spawn_actor("consumer", consumer)

    # Create producer task
    messages = ["msg1", "msg2", "msg3"]
    producer = ProducerTask(consumer_sender, messages)
    sys.spawn_task("producer", producer)

    # Wait for messages to be sent and processed
    time.sleep(0.2)

    # Verify all messages were sent and received
    assert producer.sent_count == len(messages)
    assert consumer.messages == messages

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_actor_stop_immediate():
    """Test stopping an actor immediately."""
    sys = new_system()

    actor = SimpleActor()
    sender = sys.spawn_actor("test-actor", actor)

    # Give actor time to start
    time.sleep(0.05)
    assert actor.started

    # Stop actor immediately
    sender.stop(immediate=True)

    # Give time for shutdown
    time.sleep(0.05)

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)

    assert actor.stopped


def test_actor_stop_graceful():
    """Test stopping an actor gracefully."""
    sys = new_system()

    actor = SimpleActor()
    sender = sys.spawn_actor("test-actor", actor)

    time.sleep(0.05)

    # Stop actor gracefully
    sender.stop(immediate=False)

    time.sleep(0.05)

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)

    assert actor.stopped


def test_concurrent_message_sending():
    """Test concurrent message sending to the same actor."""
    sys = new_system()

    actor = CountingActor()
    sender = sys.spawn_actor("counting-actor", actor)

    # Send messages from multiple threads
    def send_numbers(start: int, count: int):
        for i in range(start, start + count):
            sender.send(i)

    threads = []
    num_threads = 3
    numbers_per_thread = 10

    for t in range(num_threads):
        thread = threading.Thread(
            target=send_numbers, args=(t * numbers_per_thread, numbers_per_thread)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Give time for message processing
    time.sleep(0.1)

    # Verify all messages were processed
    expected_count = num_threads * numbers_per_thread
    expected_total = sum(range(expected_count))

    assert actor.count == expected_count
    assert actor.total == expected_total

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_sender_destination():
    """Test getting sender destination ID."""
    sys = new_system()

    actor = SimpleActor()
    sender = sys.spawn_actor("test-actor", actor)

    # Verify sender has a valid destination ID
    dest_id = sender.dest()
    assert isinstance(dest_id, int)  # UniqId is NewType(int)
    assert dest_id >= 0

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)


def test_empty_system_shutdown():
    """Test shutting down system with no spawned actors or tasks."""
    sys = new_system()

    # Immediately stop without spawning anything
    sys.stop(immediate=False)
    exceptions = sys.wait()

    assert len(exceptions) == 0
    assert sys.thread_count() == 0  # Should be 0 after shutdown


def test_multiple_message_types():
    """Test actor handling different message content."""
    sys = new_system()

    actor = SimpleActor()
    sender = sys.spawn_actor("test-actor", actor)

    # Send various string messages
    messages = ["", "single", "multiple words", "special!@#$%^&*()"]
    for msg in messages:
        sender.send(msg)

    time.sleep(0.05)

    assert actor.messages == messages

    sys.stop(immediate=False)
    sys.wait(timeout=1.0)
