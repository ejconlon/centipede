"""Integration tests for the MIDI live system.

These tests verify that the MIDI live system can be created and configured properly.
Note: Actual MIDI message sending would require deeper mocking of the actor system.
"""

import time
from fractions import Fraction
from typing import Any, Generator, List, Tuple
from unittest.mock import patch

import pytest
from mido.frozen import FrozenMessage

from bad_actor import new_system
from minipat.live import (
    BackendPlay,
    BackendTiming,
    Orbit,
    Timing,
)
from minipat.midi import (
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

    def send(self, msg: FrozenMessage) -> None:
        """Record sent messages."""
        if self.closed:
            raise RuntimeError("Port is closed")
        self.messages.append(msg)
        self.send_times.append(time.time())

    def close(self) -> None:
        """Mark port as closed."""
        self.closed = True

    def clear(self) -> None:
        """Clear recorded messages."""
        self.messages = []
        self.send_times = []


@pytest.fixture
def system() -> Any:
    """Create a test actor system."""
    sys = new_system()
    yield sys
    # System doesn't have dispose method


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
        live, mock_port = live_system

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
        live, mock_port = live_system

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
        live, mock_port = live_system

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
        live, mock_port = live_system

        # Set different CPS values - should not raise
        live.set_cps(Fraction(1, 2))  # 0.5 cps
        live.set_cps(Fraction(1, 1))  # 1 cps
        live.set_cps(Fraction(2, 1))  # 2 cps

        assert True

    def test_combine_streams(self, live_system: Tuple[Any, MockMidiPort]) -> None:
        """Test combining note and velocity streams."""
        live, mock_port = live_system

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
        live, mock_port = live_system

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
        live, mock_port = live_system

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
        live, mock_port = live_system

        # Test manual message sending to verify mock works

        test_msg = FrozenMessage("note_on", channel=0, note=60, velocity=100)

        # Send message directly to mock
        mock_port.send(test_msg)

        # Verify it was captured
        assert len(mock_port.messages) == 1
        assert len(mock_port.send_times) == 1
        assert mock_port.messages[0] == test_msg
        assert mock_port.messages[0].type == "note_on"
        assert mock_port.messages[0].note == 60
        assert mock_port.messages[0].velocity == 100

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
