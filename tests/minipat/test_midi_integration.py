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
from minipat.combinators import note_stream
from minipat.live import (
    LiveSystem,
    Orbit,
)
from minipat.messages import MidiAttrs, MsgTypeField, NoteField, TimedMessage
from minipat.midi import start_midi_live_system
from minipat.time import PosixTime, mk_cps

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

    def get_all_messages(self) -> List[TimedMessage]:
        """Get all messages currently in the queue without blocking."""
        messages = []
        while not self.timed_message_queue.empty():
            try:
                msg = self.timed_message_queue.get_nowait()
                messages.append(msg)
            except Empty:
                break
        return messages


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
    with (
        patch("minipat.midi.mido.open_output", return_value=mock_port),
        patch("minipat.midi.mido.get_output_names", return_value=["test_port"]),
    ):
        # Pass CPS=2 directly to the constructor
        live = start_midi_live_system(system, "test_port", cps=mk_cps(Fraction(2, 1)))
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

        # Calculate expected duration for 2 cycles based on current CPS
        current_cps = live.get_cps()
        current_gpc = live.get_gpc()
        cycles_to_wait = 2
        expected_duration = cycles_to_wait / current_cps
        max_wait_time = expected_duration * 1.5  # Add 50% buffer for safety

        # Calculate the duration of a single generation
        generation_duration = 1.0 / (float(current_cps) * current_gpc)

        # Poll current cycle time instead of sleeping fixed duration
        start_time = time.time()
        sleep_increment = 0.05  # Check every 50ms for more responsive polling

        while time.time() - start_time < max_wait_time:
            current_cycle = live.get_cycle()
            if current_cycle > cycles_to_wait:
                # Wait for one generation to ensure all pending MIDI messages are processed
                time.sleep(generation_duration)
                break
            time.sleep(sleep_increment)

        # Stop playing
        live.pause()

        # Collect all messages
        messages = mock_port.get_all_messages()

        # Expected pattern: c4, d4, e4, f4 in MIDI note numbers
        expected_notes = [48, 50, 52, 53]

        # First, verify we have the correct sequence of messages IN ORDER
        # Each note should produce: note_on followed by note_off
        # Pattern should be: on(c4), off(c4), on(d4), off(d4), on(e4), off(e4), on(f4), off(f4), repeat...

        # Build the actual sequence of (event_type, note_number) tuples
        actual_sequence = []
        for msg in messages:
            msg_type = MsgTypeField.get(msg.message)
            if msg_type in ["note_on", "note_off"]:
                note_num = NoteField.unmk(NoteField.get(msg.message))
                actual_sequence.append((msg_type, note_num))

        # We should have at least 2 complete cycles (16 messages: 8 on + 8 off)
        assert len(actual_sequence) >= 16, (
            f"Expected at least 16 messages for 2+ cycles, got {len(actual_sequence)}"
        )

        # Verify the pattern is correct: each note should have on followed by off, in order
        expected_pattern = []
        for _ in range(2):  # At least 2 cycles
            for note in expected_notes:
                expected_pattern.append(("note_on", note))
                expected_pattern.append(("note_off", note))

        # Check that we have at least the expected pattern
        for i, expected in enumerate(expected_pattern):
            assert i < len(actual_sequence), (
                f"Missing message at index {i}: expected {expected}"
            )
            assert actual_sequence[i] == expected, (
                f"Wrong message at index {i}: expected {expected}, got {actual_sequence[i]}"
            )

        # Useful to turn off timing to debug logic
        validate_timing = True

        if validate_timing:
            # Now separate messages for timing analysis
            note_on_messages = []
            note_off_messages = []

            for msg in messages:
                msg_type = MsgTypeField.get(msg.message)
                if msg_type == "note_on":
                    note_on_messages.append(msg)
                elif msg_type == "note_off":
                    note_off_messages.append(msg)

            # Now check timing between consecutive note_on messages
            # Each should be approximately 0.125 seconds apart (125ms)
            expected_interval = 0.125  # 0.5 seconds per cycle / 4 notes
            tolerance = (
                0.075  # 75ms tolerance for timing variations (increased from 50ms)
            )
            duration_tolerance = (
                0.1  # 100ms tolerance for note durations (increased from 75ms)
            )

            for i in range(
                1, min(8, len(note_on_messages))
            ):  # Check first 8 note_ons (2 cycles)
                time_diff = note_on_messages[i].time - note_on_messages[i - 1].time
                assert abs(time_diff - expected_interval) <= tolerance, (
                    f"Note_on {i}: Expected interval ~{expected_interval}s, "
                    f"got {time_diff:.3f}s (tolerance: {tolerance * 1000:.0f}ms)"
                )

            # Check note durations (time between note_on and its corresponding note_off)
            # Since we've already verified the order is correct, we can pair them directly
            for i in range(
                min(8, len(note_on_messages))
            ):  # Check first 8 notes (2 cycles)
                if i < len(note_off_messages):
                    note_on_time = note_on_messages[i].time
                    # The corresponding note_off is at the same index due to our ordering validation
                    note_off_time = note_off_messages[i].time
                    duration = note_off_time - note_on_time

                    # Note durations can vary more due to system timing
                    assert abs(duration - expected_interval) <= duration_tolerance, (
                        f"Note {i} duration: Expected ~{expected_interval}s, "
                        f"got {duration:.3f}s (tolerance: {duration_tolerance * 1000:.0f}ms)"
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

    def test_sequential_pattern_cycling(self, live_system: MidiLiveFixture) -> None:
        """Test that sequential patterns cycle correctly across multiple cycles.

        Bug case: Pattern "c4 f4" over 2 cycles should produce:
        Cycle 1: C4 (first half), F4 (second half)
        Cycle 2: C4 (first half), F4 (second half)
        Expected sequence: C4, F4, C4, F4

        WRONG behavior would be: C4, C4, F4, F4 (all C4s first, then all F4s)
        This indicates the pattern is not cycling properly.
        """
        live, mock_port = live_system

        # Clear any existing messages
        mock_port.clear()

        # Use a simple 2-note sequential pattern
        pattern = note_stream("c4 f4")
        live.set_orbit(Orbit(0), pattern)

        # Start playing
        live.play()

        # Wait for exactly 2 complete cycles
        current_cps = live.get_cps()
        current_gpc = live.get_gpc()
        cycles_to_wait = 2
        expected_duration = cycles_to_wait / current_cps
        max_wait_time = expected_duration * 1.5

        generation_duration = 1.0 / (float(current_cps) * current_gpc)

        start_time = time.time()
        sleep_increment = 0.05

        while time.time() - start_time < max_wait_time:
            current_cycle = live.get_cycle()
            if current_cycle > cycles_to_wait:
                time.sleep(generation_duration)
                break
            time.sleep(sleep_increment)

        # Stop playing
        live.pause()

        # Collect all messages
        messages = mock_port.get_all_messages()

        # Extract only note_on events with their notes
        note_on_sequence = []
        for msg in messages:
            msg_type = MsgTypeField.get(msg.message)
            if msg_type == "note_on":
                note_num = NoteField.unmk(NoteField.get(msg.message))
                note_on_sequence.append(note_num)

        # We should have at least 4 note_on events for 2 cycles of "c4 f4"
        assert len(note_on_sequence) >= 4, (
            f"Expected at least 4 note_on events for 2 cycles, got {len(note_on_sequence)}"
        )

        # CRITICAL TEST: Verify the pattern cycles correctly
        # Expected: [48, 53, 48, 53, ...] (C4, F4, C4, F4, ...)
        # Wrong:    [48, 48, 53, 53, ...] (C4, C4, F4, F4, ...)

        # Check the first 4 notes to verify correct cycling
        expected_sequence = [48, 53, 48, 53]  # C4, F4, C4, F4
        actual_first_four = note_on_sequence[:4]

        assert actual_first_four == expected_sequence, (
            f"Pattern should cycle correctly! Expected {expected_sequence} "
            f"(C4, F4, C4, F4), but got {actual_first_four}. "
            f"If you see [48, 48, 53, 53] or similar, the pattern is NOT cycling properly."
        )

        # Additional check: verify alternating pattern continues
        # The pattern should continue alternating C4/F4, not group same notes together
        for i in range(0, min(len(note_on_sequence) - 1, 8), 2):
            if i + 1 < len(note_on_sequence):
                # Every pair should be C4, F4
                pair = note_on_sequence[i : i + 2]
                assert pair == [48, 53], (
                    f"Expected alternating C4(48), F4(53) at position {i}, got {pair}. "
                    f"Full sequence: {note_on_sequence}"
                )
