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
from minipat.common import PosixTime
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
    with patch("minipat.midi.mido.open_output", return_value=mock_port), \
         patch("minipat.midi.mido.get_output_names", return_value=["test_port"]):
        # Pass CPS=2 directly to the constructor
        live = start_midi_live_system(system, "test_port", cps=Fraction(2, 1))
        try:
            yield live, mock_port
        finally:
            live.panic()


class TestMidiLiveSystemIntegration:
    """Integration tests for MIDI live system."""

    def test_pattern_timing_with_cps(self, live_system: MidiLiveFixture) -> None:
        """Test that a repeating pattern generates notes with correct timing for the CPS.

        For CPS=2 (2 cycles per second):
        - Each cycle takes 0.5 seconds
        - A 4-note pattern (c4 d4 e4 f4) has each note lasting 0.125 seconds
        - Each note_on should be followed by note_off after 0.125 seconds
        - The next note_on starts immediately after the previous note_off
        - For 2 cycles, we expect 8 note_on/note_off pairs
        """
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Set up a 4-note pattern
        pattern = note_stream("c4 d4 e4 f4")
        live.set_orbit(Orbit(0), pattern)

        # CPS is already set to 2 in the fixture

        # Start playing
        live.play()

        # Wait for 2 full cycles (at CPS=2, this is 1 second)
        # Add a bit extra to ensure we capture all messages
        time.sleep(1.2)

        # Stop playing
        live.pause()

        # Collect all messages
        messages = []
        while mock_port.has_messages():
            msg = mock_port.wait_for_message(timeout=0.1)
            if msg is not None:
                messages.append(msg)

        # Separate note_on and note_off messages
        note_on_messages = []
        note_off_messages = []

        for msg in messages:
            msg_type = MsgTypeField.get(msg.message)
            if msg_type == "note_on":
                note_on_messages.append(msg)
            elif msg_type == "note_off":
                note_off_messages.append(msg)

        # We should have at least 4 note_on messages (1 complete cycle)
        assert len(note_on_messages) >= 4, (
            f"Expected at least 4 note_on messages for 1+ cycles, got {len(note_on_messages)}"
        )

        # We should have corresponding note_off messages
        assert len(note_off_messages) >= 4, (
            f"Expected at least 4 note_off messages for 1+ cycles, got {len(note_off_messages)}"
        )

        # Check the note sequence - should be c4, d4, e4, f4 repeated
        expected_notes = [60, 62, 64, 65]  # c4, d4, e4, f4 in MIDI note numbers

        # Extract the actual note sequence from all note_on messages we received
        actual_notes = []
        for msg in note_on_messages:
            note_num = NoteField.unmk(NoteField.get(msg.message))
            actual_notes.append(note_num)

        # Verify the note sequence matches expected pattern (allow for partial cycles)
        num_complete_cycles = len(actual_notes) // 4
        expected_sequence = expected_notes * num_complete_cycles
        # Add any partial cycle notes
        remaining_notes = len(actual_notes) % 4
        if remaining_notes > 0:
            expected_sequence.extend(expected_notes[:remaining_notes])

        assert actual_notes == expected_sequence, (
            f"Expected note sequence {expected_sequence}, got {actual_notes}"
        )

        # Check timing between consecutive note_on messages
        # Each should be approximately 0.125 seconds apart (125ms)
        expected_interval = 0.125  # 0.5 seconds per cycle / 4 notes
        tolerance = 0.030  # Allow 30ms tolerance for timing variations (system jitter)

        for i in range(1, len(note_on_messages)):
            time_diff = note_on_messages[i].time - note_on_messages[i - 1].time
            assert abs(time_diff - expected_interval) <= tolerance, (
                f"Note {i}: Expected interval ~{expected_interval}s, "
                f"got {time_diff:.3f}s (jitter must be <30ms)"
            )

        # Check that each note_on has a corresponding note_off at the right time
        # Group messages by note to match on/off pairs
        note_events: dict[int, list[tuple[str, float]]] = {}
        for msg in messages:  # Look at all messages
            msg_type = MsgTypeField.get(msg.message)
            if msg_type in ["note_on", "note_off"]:
                note_num = NoteField.unmk(NoteField.get(msg.message))
                if note_num not in note_events:
                    note_events[note_num] = []
                note_events[note_num].append((msg_type, msg.time))

        # For each note that actually appeared, check on/off pairing and duration
        for note_num in set(actual_notes):
            if note_num in note_events:
                events = note_events[note_num]
                # Should have alternating on/off events
                for j in range(0, len(events) - 1, 2):
                    if j + 1 < len(events):
                        on_type, on_time = events[j]
                        off_type, off_time = events[j + 1]

                        assert on_type == "note_on", f"Expected note_on at index {j}"
                        assert off_type == "note_off", (
                            f"Expected note_off at index {j + 1}"
                        )

                        # Duration between on and off should be ~0.125 seconds
                        # Note: Allow more tolerance for note durations as they may be affected
                        # by system timing and concurrent processing
                        duration = off_time - on_time
                        duration_tolerance = 0.075  # 75ms tolerance for note durations
                        assert (
                            abs(duration - expected_interval) <= duration_tolerance
                        ), (
                            f"Note {note_num}: Expected duration ~{expected_interval}s, "
                            f"got {duration:.3f}s (duration tolerance: 75ms)"
                        )

        # Verify total time span for the number of notes we actually received
        if len(note_on_messages) >= 2:
            total_time = note_on_messages[-1].time - note_on_messages[0].time
            num_intervals = len(note_on_messages) - 1
            expected_total = num_intervals * expected_interval
            assert abs(total_time - expected_total) <= tolerance * num_intervals, (
                f"Expected total time for {num_intervals} intervals ~{expected_total}s, "
                f"got {total_time:.3f}s"
            )
