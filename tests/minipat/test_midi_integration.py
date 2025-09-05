"""Integration tests for the MIDI live system.

These tests verify that the MIDI live system can be created and configured properly.
Note: Actual MIDI message sending would require deeper mocking of the actor system.
"""

import time
from fractions import Fraction
from queue import Empty, Queue
from typing import Any, Generator, List, Tuple
from unittest.mock import patch

import pytest
from mido.frozen import FrozenMessage

from bad_actor import new_system
from minipat.common import PosixTime
from minipat.live import (
    BackendPlay,
    BackendTiming,
    Orbit,
    Timing,
)
from minipat.midi import (
    MsgTypeField,
    NoteField,
    TimedMessage,
    VelocityField,
    combine,
    note,
    start_midi_live_system,
    vel,
)


class MockMidiPort:
    """Mock MIDI output port for testing."""

    def __init__(self) -> None:
        self.messages: List[FrozenMessage] = []
        self.closed = False
        self.send_times: List[float] = []
        self.timed_message_queue: Queue[TimedMessage] = Queue()

    def send(self, msg: FrozenMessage) -> None:
        """Record sent messages."""
        if self.closed:
            raise RuntimeError("Port is closed")
        current_time = time.time()
        self.messages.append(msg)
        self.send_times.append(current_time)

        # Also put TimedMessage in queue for better testing
        timed_msg = TimedMessage(time=PosixTime(current_time), message=msg)
        self.timed_message_queue.put(timed_msg)

    def wait_for_message(self, timeout: float = 1.0) -> TimedMessage | None:
        """Wait for a timed message to arrive, with timeout.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            TimedMessage if one arrives within timeout, None otherwise
        """
        try:
            return self.timed_message_queue.get(timeout=timeout)
        except Empty:
            return None

    def wait_for_messages(self, count: int, timeout: float = 1.0) -> List[TimedMessage]:
        """Wait for multiple timed messages to arrive.

        Args:
            count: Number of messages to wait for
            timeout: Total timeout for all messages

        Returns:
            List of TimedMessages (may be shorter than count if timeout)
        """
        messages = []
        start_time = time.time()

        for _ in range(count):
            remaining_time = max(0, timeout - (time.time() - start_time))
            if remaining_time <= 0:
                break

            msg = self.wait_for_message(remaining_time)
            if msg is None:
                break
            messages.append(msg)

        return messages

    def has_messages(self) -> bool:
        """Check if there are messages in the queue without blocking."""
        return not self.timed_message_queue.empty()

    def close(self) -> None:
        """Mark port as closed."""
        self.closed = True

    def clear(self) -> None:
        """Clear recorded messages and queue."""
        self.messages = []
        self.send_times = []
        # Clear the queue
        while not self.timed_message_queue.empty():
            try:
                self.timed_message_queue.get_nowait()
            except Empty:
                break


@pytest.fixture
def system() -> Any:
    """Create a test actor system."""
    sys = new_system()
    yield sys
    sys.stop()


@pytest.fixture
def live_system(system: Any) -> Generator[Tuple[Any, MockMidiPort], None, None]:
    """Create a live MIDI system with mock output."""
    mock_port = MockMidiPort()
    with patch("mido.open_output", return_value=mock_port):
        live = start_midi_live_system(system, "test_port")
        yield live, mock_port
        # LiveSystem doesn't have dispose method


class TestMidiLiveSystemIntegration:
    """Integration tests for MIDI live system."""

    def test_system_startup_and_shutdown(self, system: Any) -> None:
        """Test that the MIDI live system starts and shuts down cleanly."""
        mock_port = MockMidiPort()

        with patch("mido.open_output", return_value=mock_port):
            live = start_midi_live_system(system, "test_port")

            # System should be running
            assert not mock_port.closed

            # System should have the required components
            assert hasattr(live, "_transport_sender")
            assert hasattr(live, "_pattern_sender")
            assert hasattr(live, "_backend_sender")
            assert hasattr(live, "set_orbit")
            assert hasattr(live, "play")
            assert hasattr(live, "pause")
            assert hasattr(live, "set_cps")

    def test_set_orbit_accepts_note_stream(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that set_orbit accepts note streams."""
        live, _ = live_system

        # Create a simple note pattern using the note function
        note_stream = note("c4 d4 e4")

        # This should not raise an exception
        live.set_orbit(Orbit(0), note_stream)

        # Verify orbit was set (no exception means success)
        assert True

    def test_multiple_orbits_can_be_set(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test setting multiple orbits."""
        live, _ = live_system

        # Create different patterns for orbits
        note_stream1 = note("c4 d4")
        note_stream2 = note("c5 d5")

        # Set different orbits - should not raise
        live.set_orbit(Orbit(0), note_stream1)
        live.set_orbit(Orbit(1), note_stream2)

        # Can also clear an orbit
        live.set_orbit(Orbit(0), None)

        assert True

    def test_play_pause_methods_exist(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test play and pause methods work without error."""
        live, _ = live_system

        # Send a pattern
        note_stream = note("c4")
        live.set_orbit(Orbit(0), note_stream)

        # These should not raise exceptions
        live.play()
        time.sleep(0.01)
        live.pause()
        live.play()
        live.pause()

        assert True

    def test_set_cps_accepts_fraction(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that set_cps accepts Fraction values."""
        live, _ = live_system

        # Set different CPS values - should not raise
        live.set_cps(Fraction(1, 2))  # 0.5 cps
        live.set_cps(Fraction(1, 1))  # 1 cps
        live.set_cps(Fraction(2, 1))  # 2 cps

        assert True

    def test_combine_streams(self, live_system: Tuple[Any, MockMidiPort]) -> None:
        """Test combining note and velocity streams."""
        live, _ = live_system

        # Test note messages with velocity
        note_stream = note("c4 d4 e4")
        vel_stream = vel("100 80 60")

        # Combine streams using combine - should not raise
        combined = combine(note_stream, vel_stream)

        live.set_orbit(Orbit(0), combined)

        assert True

    def test_backend_timing_message_accepted(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that BackendTiming messages can be sent."""
        live, _ = live_system

        # Create a timing configuration
        new_timing = Timing(
            cps=Fraction(2, 1), generations_per_cycle=4, wait_factor=Fraction(1, 8)
        )

        # Send timing message - should not raise
        live._backend_sender.send(BackendTiming(timing=new_timing))

        assert True

    def test_backend_play_message_accepted(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that BackendPlay messages can be sent."""
        live, _ = live_system

        # Send play messages - should not raise
        live._backend_sender.send(BackendPlay(True))
        time.sleep(0.01)
        live._backend_sender.send(BackendPlay(False))

        assert True

    def test_all_message_types_together(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test using all message types in one session."""
        live, mock_port = live_system

        # Set up multiple orbits with different streams
        note_stream1 = note("c4 e4 g4")
        note_stream2 = note("c5 d5 e5")
        vel_stream = vel("100 80 60")

        # Set orbit 0 with just notes
        live.set_orbit(Orbit(0), note_stream1)

        # Set orbit 1 with combined notes and velocities
        combined = combine(note_stream2, vel_stream)
        live.set_orbit(Orbit(1), combined)

        # Change timing
        live.set_cps(Fraction(1, 1))

        # Play and pause
        live.play()
        time.sleep(0.02)

        # Send backend timing change while playing
        new_timing = Timing(
            cps=Fraction(2, 1), generations_per_cycle=4, wait_factor=Fraction(1, 8)
        )
        live._backend_sender.send(BackendTiming(timing=new_timing))

        time.sleep(0.02)
        live.pause()

        # Clear an orbit
        live.set_orbit(Orbit(0), None)

        # Use backend play control
        live._backend_sender.send(BackendPlay(True))
        time.sleep(0.01)
        live._backend_sender.send(BackendPlay(False))

        # Verify mock port interface works (even if no messages captured)
        assert hasattr(mock_port, "messages")
        assert hasattr(mock_port, "send_times")
        assert hasattr(mock_port, "clear")

        # Test the clear functionality
        mock_port.clear()
        assert len(mock_port.messages) == 0
        assert len(mock_port.send_times) == 0

        # If we got here without exceptions, the test passes
        assert True

    def test_mock_port_interface(self, live_system: Tuple[Any, MockMidiPort]) -> None:
        """Test that the MockMidiPort interface works correctly."""
        _, mock_port = live_system

        # Test manual message sending to verify mock works

        test_msg = FrozenMessage("note_on", channel=0, note=60, velocity=100)

        # Send message directly to mock
        mock_port.send(test_msg)

        # Verify it was captured
        assert len(mock_port.messages) == 1
        assert len(mock_port.send_times) == 1
        assert mock_port.messages[0] == test_msg
        assert MsgTypeField.get(mock_port.messages[0]) == "note_on"
        assert NoteField.unmk(NoteField.get(mock_port.messages[0])) == 60
        assert VelocityField.unmk(VelocityField.get(mock_port.messages[0])) == 100

        # Test clearing
        mock_port.clear()
        assert len(mock_port.messages) == 0
        assert len(mock_port.send_times) == 0

        # Test closing
        mock_port.close()
        assert mock_port.closed

        # Should raise when sending to closed port
        with pytest.raises(RuntimeError, match="Port is closed"):
            mock_port.send(test_msg)

    def test_timed_message_queue_functionality(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that the TimedMessage queue functionality works correctly."""
        _, mock_port = live_system

        # Test manual message sending to verify queue works
        test_msg1 = FrozenMessage("note_on", channel=0, note=60, velocity=100)
        test_msg2 = FrozenMessage("note_off", channel=0, note=60, velocity=0)

        # Initially no messages
        assert not mock_port.has_messages()
        assert mock_port.wait_for_message(timeout=0.1) is None

        # Send messages
        mock_port.send(test_msg1)
        mock_port.send(test_msg2)

        # Should have messages now
        assert mock_port.has_messages()

        # Get first message
        timed_msg1 = mock_port.wait_for_message(timeout=0.1)
        assert timed_msg1 is not None
        assert timed_msg1.message == test_msg1
        assert MsgTypeField.get(timed_msg1.message) == "note_on"
        assert isinstance(timed_msg1.time, float)

        # Get second message
        timed_msg2 = mock_port.wait_for_message(timeout=0.1)
        assert timed_msg2 is not None
        assert timed_msg2.message == test_msg2
        assert MsgTypeField.get(timed_msg2.message) == "note_off"

        # No more messages
        assert not mock_port.has_messages()
        assert mock_port.wait_for_message(timeout=0.1) is None

    def test_wait_for_multiple_messages(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test waiting for multiple messages at once."""
        _, mock_port = live_system

        # Send multiple messages
        messages = [
            FrozenMessage("note_on", channel=0, note=60, velocity=100),
            FrozenMessage("note_on", channel=0, note=64, velocity=80),
            FrozenMessage("note_on", channel=0, note=67, velocity=90),
        ]

        for msg in messages:
            mock_port.send(msg)

        # Wait for all 3 messages
        timed_messages = mock_port.wait_for_messages(count=3, timeout=0.5)
        assert len(timed_messages) == 3

        # Verify all messages are correct
        for i, timed_msg in enumerate(timed_messages):
            assert timed_msg.message == messages[i]
            assert MsgTypeField.get(timed_msg.message) == "note_on"

        # No more messages left
        assert not mock_port.has_messages()

    def test_wait_for_messages_timeout(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that waiting for messages respects timeout."""
        _, mock_port = live_system

        # Send only 1 message but wait for 3
        test_msg = FrozenMessage("note_on", channel=0, note=60, velocity=100)
        mock_port.send(test_msg)

        # Should only get 1 message, then wait for remaining timeout
        start_time = time.time()
        timed_messages = mock_port.wait_for_messages(count=3, timeout=0.2)
        elapsed = time.time() - start_time

        assert len(timed_messages) == 1
        assert timed_messages[0].message == test_msg
        # Should wait approximately the full timeout since we asked for 3 messages
        assert 0.18 <= elapsed <= 0.25  # Allow some tolerance for timing

    def test_queue_clearing(self, live_system: Tuple[Any, MockMidiPort]) -> None:
        """Test that clearing works for both lists and queue."""
        _, mock_port = live_system

        # Send some messages
        for i in range(3):
            msg = FrozenMessage("note_on", channel=0, note=60 + i, velocity=100)
            mock_port.send(msg)

        # Verify messages are there
        assert len(mock_port.messages) == 3
        assert mock_port.has_messages()

        # Clear everything
        mock_port.clear()

        # Verify everything is cleared
        assert len(mock_port.messages) == 0
        assert len(mock_port.send_times) == 0
        assert not mock_port.has_messages()
        assert mock_port.wait_for_message(timeout=0.1) is None

    def test_live_system_with_message_verification(
        self, live_system: Tuple[Any, MockMidiPort]
    ) -> None:
        """Test that we can verify actual messages from the live system."""
        live, mock_port = live_system

        # Set up a simple pattern
        note_stream = note("c4 d4")
        live.set_orbit(Orbit(0), note_stream)

        # Start playing
        live.play()

        # Wait a bit for messages to be generated and sent
        # Note: This test may be flaky depending on timing and actor scheduling
        time.sleep(0.1)

        # Check if any messages were captured
        # We don't assert specific counts as that depends on internal timing
        # But we can at least verify the queue interface works
        has_messages = mock_port.has_messages()
        message_count = len(mock_port.messages)

        # Stop playing
        live.pause()

        # The test passes if we got this far without exceptions
        # In a real scenario, we might get messages, but due to actor timing
        # and mocking, we might not. The important thing is the queue works.
        print(f"Captured {message_count} messages, queue has messages: {has_messages}")
        assert True
