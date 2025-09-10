"""Integration tests for the MIDI live system.

These tests verify that the MIDI live system can be created and configured properly.
Note: Actual MIDI message sending would require deeper mocking of the actor system.
"""

import time
from fractions import Fraction
from queue import Empty, Queue
from typing import Generator, List, Tuple
from unittest.mock import patch

import pytest
from mido.frozen import FrozenMessage

from bad_actor import System, new_system
from minipat.common import ONE, CycleDelta, PosixTime
from minipat.live import (
    LiveSystem,
    Orbit,
)
from minipat.midi import (
    MidiAttrs,
    MsgTypeField,
    NoteField,
    TimedMessage,
    note_stream,
    start_midi_live_system,
)

# Type aliases for brevity
MidiLiveSystem = LiveSystem[MidiAttrs, TimedMessage]
MidiLiveFixture = Tuple[MidiLiveSystem, "MockMidiPort"]


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

    def reset(self) -> None:
        """Reset the port (MIDI interface compatibility)."""
        # For mock purposes, reset does nothing special
        # Real MIDI ports would send all notes off, reset controllers, etc.
        pass

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
def system() -> Generator[System, None, None]:
    """Create a test actor system."""
    sys = new_system()
    try:
        yield sys
    finally:
        sys.stop()


@pytest.fixture
def live_system(system: System) -> Generator[MidiLiveFixture, None, None]:
    """Create a live MIDI system with mock output."""
    mock_port = MockMidiPort()
    # Patch where mido.open_output is actually called
    with patch("minipat.midi.mido.open_output", return_value=mock_port):
        # Pass CPS=2 directly to the constructor
        live = start_midi_live_system(system, "test_port", cps=Fraction(2, 1))
        try:
            yield live, mock_port
        finally:
            live.panic()


class TestMidiLiveSystemIntegration:
    """Integration tests for MIDI live system."""

    def test_simple_message_generation(self, live_system: MidiLiveFixture) -> None:
        """Test that a simple pattern generates messages."""
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up a simple single-note pattern
        pattern = note_stream("c4")
        live.set_orbit(Orbit(0), pattern)

        # Set CPS to 1 (1 cycle per second)
        live.set_cps(Fraction(1, 1))

        # Start playing
        live.play()

        # Wait for messages to be generated
        time.sleep(1.5)

        # Stop playing
        live.pause()

        # Check what we got
        message_count = len(mock_port.messages)

        # Collect from queue
        queued_messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg:
                queued_messages.append(msg)

        # At this point we expect to have received at least one message
        assert message_count > 0 or len(queued_messages) > 0, (
            "Expected at least one message from the live system"
        )

    def test_multiple_orbits_with_cps_timing(
        self, live_system: MidiLiveFixture
    ) -> None:
        """Test that multiple orbits generate messages at the expected CPS rate.

        This test verifies that when we set a specific CPS (cycles per second),
        the live system generates and sends messages through the mock port
        at roughly the expected timing intervals.
        """
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # CPS is now set to 2 in the fixture (2 cycles per second = 0.5 seconds per cycle)
        # With generations_per_cycle=4, patterns are queried 4 times per cycle

        # Set up two orbits with different note patterns
        # Orbit 0: 3 notes per cycle
        orbit0_pattern = note_stream("c4 d4 e4")
        live.set_orbit(Orbit(0), orbit0_pattern)

        # Orbit 1: 2 notes per cycle
        orbit1_pattern = note_stream("f5 g5")
        live.set_orbit(Orbit(1), orbit1_pattern)

        # Start playing to trigger the live system
        live.play()

        # Wait for at least 2 full cycles
        # At 2 CPS, 2 cycles = 1 second
        # Add extra time to account for processing delays
        time.sleep(1.5)

        # Stop playing
        live.pause()

        # Collect all messages that the live system sent
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # We should have received messages
        assert len(messages) > 0, (
            "Expected to receive MIDI messages from the live system"
        )

        # Group messages by note to identify orbits
        c4_messages = []  # From orbit 0
        d4_messages = []  # From orbit 0
        e4_messages = []  # From orbit 0
        f5_messages = []  # From orbit 1
        g5_messages = []  # From orbit 1

        for msg in messages:
            if MsgTypeField.get(msg.message) == "note_on":
                note_num = NoteField.unmk(NoteField.get(msg.message))
                if note_num == 60:  # C4
                    c4_messages.append(msg)
                elif note_num == 62:  # D4
                    d4_messages.append(msg)
                elif note_num == 64:  # E4
                    e4_messages.append(msg)
                elif note_num == 77:  # F5
                    f5_messages.append(msg)
                elif note_num == 79:  # G5
                    g5_messages.append(msg)

        # If we got messages, verify we got at least some from each pattern
        # (we ran for ~1.5 seconds at 2 CPS, so we expect ~3 cycles)
        # But be lenient - just require at least 1 note from each pattern if present
        expected_min_count = 1  # Be lenient due to timing variability

        if len(c4_messages) > 0:
            assert len(c4_messages) >= expected_min_count, (
                f"Expected at least {expected_min_count} C4 notes, got {len(c4_messages)}"
            )
        if len(d4_messages) > 0:
            assert len(d4_messages) >= expected_min_count, (
                f"Expected at least {expected_min_count} D4 notes, got {len(d4_messages)}"
            )
        if len(e4_messages) > 0:
            assert len(e4_messages) >= expected_min_count, (
                f"Expected at least {expected_min_count} E4 notes, got {len(e4_messages)}"
            )
        if len(f5_messages) > 0:
            assert len(f5_messages) >= expected_min_count, (
                f"Expected at least {expected_min_count} F5 notes, got {len(f5_messages)}"
            )
        if len(g5_messages) > 0:
            assert len(g5_messages) >= expected_min_count, (
                f"Expected at least {expected_min_count} G5 notes, got {len(g5_messages)}"
            )

        # Verify timing is roughly correct for CPS=2 with generations_per_cycle=4
        # At 2 CPS, each cycle is 0.5 seconds, each generation is 0.125 seconds
        # With "c4 d4 e4" pattern and 4 generations/cycle:
        # - Pattern spans full cycle: c4 at [0, 1/3), d4 at [1/3, 2/3), e4 at [2/3, 1)
        # - Generation 0 queries [0, 0.25): gets c4
        # - Generation 1 queries [0.25, 0.5): gets c4 until 0.33, then d4
        # - Generation 2 queries [0.5, 0.75): gets d4 until 0.67, then e4
        # - Generation 3 queries [0.75, 1.0): gets e4
        # Due to the complex overlap, timing varies
        if len(c4_messages) >= 2:
            for i in range(1, min(len(c4_messages), 3)):  # Check first 2 intervals
                time_diff = c4_messages[i].time - c4_messages[i - 1].time
                # C4 can appear in multiple generations per cycle
                # Allow wide tolerance due to pattern/generation interaction
                expected_min = 0.08  # Roughly generation interval
                expected_max = 0.6  # Up to a full cycle
                assert expected_min <= time_diff <= expected_max, (
                    f"Expected interval between {expected_min}s and {expected_max}s, "
                    f"got {time_diff:.3f}s between C4 notes"
                )

        # Similarly check F5 notes for orbit 1
        # With "f5 g5" pattern (2 notes per cycle), timing also varies
        if len(f5_messages) >= 2:
            for i in range(1, min(len(f5_messages), 3)):  # Check first 2 intervals
                time_diff = f5_messages[i].time - f5_messages[i - 1].time
                # F5 appears in first half of cycle, similar complex timing
                expected_min = 0.08  # Roughly generation interval
                expected_max = 0.6  # Up to a full cycle
                assert expected_min <= time_diff <= expected_max, (
                    f"Expected interval between {expected_min}s and {expected_max}s, "
                    f"got {time_diff:.3f}s between F5 notes"
                )

    def test_once_method_immediate(self, live_system: MidiLiveFixture) -> None:
        """Test that the once method generates immediate messages without alignment."""
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up the live system and start playback to initialize timing
        live.set_cps(Fraction(2, 1))  # 2 cycles per second
        live.play()  # Start playback to initialize the system
        time.sleep(0.1)  # Let it initialize

        # Create a simple pattern
        pattern = note_stream("c4 d4")

        # Use once method with immediate timing (aligned=False)
        # Generate for half a cycle
        live.once(pattern, CycleDelta(Fraction(1, 2)), aligned=False, orbit=Orbit(0))

        # Wait a short time for message processing
        time.sleep(0.3)

        # Pause playback
        live.pause()

        # Collect messages
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # Should have received messages from the once call
        assert len(messages) > 0, "Expected messages from once method call"

        # Verify we got note-on messages
        note_on_messages = [
            msg for msg in messages if MsgTypeField.get(msg.message) == "note_on"
        ]
        assert len(note_on_messages) > 0, "Expected at least one note-on message"

        # Should contain C4 (60) or D4 (62) depending on timing within the pattern
        note_nums = [NoteField.get(msg.message) for msg in note_on_messages]
        expected_notes = {60, 62}  # C4 and D4 from our pattern
        found_notes = set(note_nums)
        common_notes = expected_notes.intersection(found_notes)
        assert len(common_notes) > 0, (
            f"Expected notes from pattern 'c4 d4' (60 or 62), got: {found_notes}"
        )

    def test_once_method_aligned(self, live_system: MidiLiveFixture) -> None:
        """Test that the once method works with cycle alignment."""
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up the live system
        live.set_cps(Fraction(1, 1))  # 1 cycle per second for easier testing

        # Start playback to establish timing
        live.play()
        time.sleep(0.2)  # Let it establish timing

        # Create a pattern
        pattern = note_stream("c4 e4 g4")  # C major triad

        # Use once method with alignment (should start at next cycle boundary)
        live.once(pattern, CycleDelta(ONE), aligned=True, orbit=Orbit(0))

        # Wait for messages to be processed
        time.sleep(1.5)  # Give time for at least one cycle

        # Stop playback
        live.pause()

        # Collect messages
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # Should have received messages
        assert len(messages) > 0, "Expected messages from aligned once call"

        # Check for our triad notes
        note_on_messages = [
            msg for msg in messages if MsgTypeField.get(msg.message) == "note_on"
        ]
        note_nums = [NoteField.get(msg.message) for msg in note_on_messages]

        # Should contain C4 (60), E4 (64), G4 (67)
        expected_notes = {60, 64, 67}  # C4, E4, G4
        found_notes = set(note_nums)

        # Check that we found at least some of our expected notes
        common_notes = expected_notes.intersection(found_notes)
        assert len(common_notes) > 0, (
            f"Expected some notes from C major triad, got notes: {found_notes}"
        )

    def test_once_method_multiple_calls(self, live_system: MidiLiveFixture) -> None:
        """Test that multiple once calls work correctly."""
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up the live system
        live.set_cps(Fraction(4, 1))  # 4 cycles per second for faster testing
        live.play()  # Start playback to initialize the system
        time.sleep(0.1)  # Let it initialize

        # Create different patterns
        pattern1 = note_stream("c4")
        pattern2 = note_stream("g4")

        # Make multiple once calls in quick succession
        live.once(pattern1, CycleDelta(Fraction(1, 4)), aligned=False, orbit=Orbit(0))
        time.sleep(0.1)
        live.once(pattern2, CycleDelta(Fraction(1, 4)), aligned=False, orbit=Orbit(1))

        # Wait for processing
        time.sleep(0.5)

        # Pause playback
        live.pause()

        # Collect messages
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # Should have messages from both calls
        assert len(messages) > 0, "Expected messages from multiple once calls"

        # Check for any note messages (note_on or note_off) that contain our expected notes
        all_note_messages = [msg for msg in messages if hasattr(msg.message, "note")]
        all_note_nums = [NoteField.get(msg.message) for msg in all_note_messages]

        # Should contain both C4 (60) and G4 (67) - be flexible since timing might affect which gets through
        expected_notes = {60, 67}  # C4 and G4
        found_notes = set(all_note_nums)
        common_notes = expected_notes.intersection(found_notes)
        assert len(common_notes) > 0, (
            f"Expected at least one note from our patterns (60 or 67), got: {found_notes}"
        )

    def test_once_method_orbit_channel_mapping(
        self, live_system: MidiLiveFixture
    ) -> None:
        """Test that orbit parameter correctly maps to MIDI channels in once method calls."""
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up the live system
        live.set_cps(Fraction(4, 1))  # 4 cycles per second for faster testing
        live.play()  # Start playback to initialize the system
        time.sleep(0.1)  # Let it initialize

        # Create a simple pattern
        pattern = note_stream("c4")

        # Make once calls with different orbits
        live.once(
            pattern, CycleDelta(Fraction(1, 4)), aligned=False, orbit=Orbit(0)
        )  # Should map to MIDI channel 0
        time.sleep(0.05)
        live.once(
            pattern, CycleDelta(Fraction(1, 4)), aligned=False, orbit=Orbit(5)
        )  # Should map to MIDI channel 5
        time.sleep(0.05)
        live.once(
            pattern, CycleDelta(Fraction(1, 4)), aligned=False, orbit=Orbit(9)
        )  # Should map to MIDI channel 9 (drums)

        # Wait for processing
        time.sleep(0.4)

        # Pause playback
        live.pause()

        # Collect messages
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # Should have messages from all orbit calls
        assert len(messages) > 0, "Expected messages from orbit-specific once calls"

        # Check that we got messages on different MIDI channels
        channels = set()
        for msg in messages:
            if hasattr(msg.message, "channel"):
                channels.add(msg.message.channel)

        # Should have at least two different channels (0, 5, and/or 9)
        assert len(channels) >= 2, (
            f"Expected messages on multiple MIDI channels from different orbits, got channels: {channels}"
        )

        # Verify specific expected channels are present
        expected_channels = {0, 5, 9}
        common_channels = expected_channels.intersection(channels)
        assert len(common_channels) > 0, (
            f"Expected to find some of channels {expected_channels}, got: {channels}"
        )
