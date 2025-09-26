"""Diagnostic tools for analyzing MIDI timing variation and jitter.

This module provides utilities to measure and analyze timing accuracy
in the MIDI live system, helping identify sources of timing variation.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

from minipat.messages import TimedMessage


@dataclass
class TimingStats:
    """Statistics about timing accuracy and variation."""

    count: int
    mean_interval: float
    expected_interval: float
    std_dev: float
    min_interval: float
    max_interval: float
    jitter_range: float
    max_deviation: float
    accuracy_percentage: float

    @property
    def jitter_ms(self) -> float:
        """Jitter range in milliseconds."""
        return self.jitter_range * 1000

    @property
    def max_deviation_ms(self) -> float:
        """Maximum deviation in milliseconds."""
        return self.max_deviation * 1000

    @property
    def std_dev_ms(self) -> float:
        """Standard deviation in milliseconds."""
        return self.std_dev * 1000

    def __str__(self) -> str:
        return (
            f"TimingStats(count={self.count}, "
            f"accuracy={self.accuracy_percentage:.1f}%, "
            f"jitter={self.jitter_ms:.1f}ms, "
            f"max_dev={self.max_deviation_ms:.1f}ms, "
            f"std_dev={self.std_dev_ms:.1f}ms)"
        )


class TimingAnalyzer:
    """Analyzes timing accuracy of MIDI messages."""

    def __init__(self, expected_interval: float):
        """Initialize analyzer with expected interval between messages.

        Args:
            expected_interval: Expected time between messages in seconds
        """
        self.expected_interval = expected_interval
        self.messages: List[TimedMessage] = []

    def add_message(self, message: TimedMessage) -> None:
        """Add a timed message for analysis."""
        self.messages.append(message)

    def add_messages(self, messages: List[TimedMessage]) -> None:
        """Add multiple timed messages for analysis."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def analyze_intervals(self) -> Optional[TimingStats]:
        """Analyze timing intervals between consecutive messages.

        Returns:
            TimingStats if there are enough messages, None otherwise
        """
        if len(self.messages) < 2:
            return None

        # Sort messages by time to ensure correct order
        sorted_messages = sorted(self.messages, key=lambda m: m.time)

        # Calculate intervals between consecutive messages
        intervals = []
        for i in range(1, len(sorted_messages)):
            interval = sorted_messages[i].time - sorted_messages[i - 1].time
            intervals.append(interval)

        if not intervals:
            return None

        # Calculate statistics
        mean_interval = statistics.mean(intervals)
        std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
        min_interval = min(intervals)
        max_interval = max(intervals)
        jitter_range = max_interval - min_interval

        # Calculate deviations from expected interval
        deviations = [abs(interval - self.expected_interval) for interval in intervals]
        max_deviation = max(deviations)

        # Calculate accuracy percentage (how close to expected)
        accuracy_percentage = max(0, 100 * (1 - max_deviation / self.expected_interval))

        return TimingStats(
            count=len(intervals),
            mean_interval=mean_interval,
            expected_interval=self.expected_interval,
            std_dev=std_dev,
            min_interval=min_interval,
            max_interval=max_interval,
            jitter_range=jitter_range,
            max_deviation=max_deviation,
            accuracy_percentage=accuracy_percentage,
        )

    def print_detailed_analysis(self) -> None:
        """Print detailed timing analysis to console."""
        stats = self.analyze_intervals()
        if stats is None:
            print("Not enough messages for analysis")
            return

        print("\n=== MIDI Timing Analysis ===")
        print(f"Messages analyzed: {len(self.messages)}")
        print(f"Intervals calculated: {stats.count}")
        print(f"Expected interval: {self.expected_interval * 1000:.1f}ms")
        print(f"Mean interval: {stats.mean_interval * 1000:.1f}ms")
        print(f"Standard deviation: {stats.std_dev_ms:.1f}ms")
        print(f"Min interval: {stats.min_interval * 1000:.1f}ms")
        print(f"Max interval: {stats.max_interval * 1000:.1f}ms")
        print(f"Jitter range: {stats.jitter_ms:.1f}ms")
        print(f"Max deviation: {stats.max_deviation_ms:.1f}ms")
        print(f"Timing accuracy: {stats.accuracy_percentage:.1f}%")

        # Quality assessment
        if stats.jitter_ms < 5:
            quality = "Excellent (< 5ms jitter)"
        elif stats.jitter_ms < 10:
            quality = "Good (< 10ms jitter)"
        elif stats.jitter_ms < 20:
            quality = "Fair (< 20ms jitter)"
        else:
            quality = "Poor (≥ 20ms jitter)"

        print(f"Quality assessment: {quality}")

        # Print individual intervals for detailed inspection
        if len(self.messages) <= 20:  # Only for small datasets
            print("\nIndividual intervals:")
            sorted_messages = sorted(self.messages, key=lambda m: m.time)
            for i in range(1, len(sorted_messages)):
                interval = sorted_messages[i].time - sorted_messages[i - 1].time
                deviation = interval - self.expected_interval
                print(
                    f"  {i:2d}: {interval * 1000:6.1f}ms (dev: {deviation * 1000:+6.1f}ms)"
                )


class SystemTimingTest:
    """Test system timing capabilities and identify issues."""

    @staticmethod
    def test_sleep_precision(
        duration: float = 0.01, iterations: int = 100
    ) -> TimingStats:
        """Test the precision of time.sleep() calls.

        Args:
            duration: Target sleep duration in seconds
            iterations: Number of sleep tests to perform

        Returns:
            TimingStats for sleep precision
        """
        actual_durations = []

        for _ in range(iterations):
            start_time = time.time()
            time.sleep(duration)
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_durations.append(actual_duration)

        if not actual_durations:
            raise ValueError("No timing data collected")

        mean_duration = statistics.mean(actual_durations)
        std_dev = (
            statistics.stdev(actual_durations) if len(actual_durations) > 1 else 0.0
        )
        min_duration = min(actual_durations)
        max_duration = max(actual_durations)
        jitter_range = max_duration - min_duration
        max_deviation = max(abs(d - duration) for d in actual_durations)
        accuracy = max(0, 100 * (1 - max_deviation / duration))

        return TimingStats(
            count=len(actual_durations),
            mean_interval=mean_duration,
            expected_interval=duration,
            std_dev=std_dev,
            min_interval=min_duration,
            max_interval=max_duration,
            jitter_range=jitter_range,
            max_deviation=max_deviation,
            accuracy_percentage=accuracy,
        )

    @staticmethod
    def test_time_resolution() -> float:
        """Test the resolution of time.time() on this system.

        Returns:
            Time resolution in seconds (minimum detectable time difference)
        """
        # Measure the smallest time difference we can detect
        last_time = time.time()
        min_diff = float("inf")
        attempts = 10000

        for _ in range(attempts):
            current_time = time.time()
            if current_time != last_time:
                diff = current_time - last_time
                min_diff = min(min_diff, diff)
                last_time = current_time

        return min_diff if min_diff != float("inf") else 0.0

    @staticmethod
    def diagnose_system() -> None:
        """Run comprehensive system timing diagnostics."""
        print("=== System Timing Diagnostics ===")

        # Test time resolution
        resolution = SystemTimingTest.test_time_resolution()
        print(f"Time resolution: {resolution * 1000:.3f}ms")

        # Test sleep precision at various durations
        for sleep_ms in [1, 5, 10, 20, 50]:
            sleep_seconds = sleep_ms / 1000.0
            stats = SystemTimingTest.test_sleep_precision(sleep_seconds, 50)
            print(
                f"Sleep {sleep_ms:2d}ms: jitter={stats.jitter_ms:5.1f}ms, "
                f"accuracy={stats.accuracy_percentage:5.1f}%"
            )


def create_timing_analyzer_for_cps(
    cps: float, notes_per_cycle: int = 4
) -> TimingAnalyzer:
    """Create a TimingAnalyzer configured for specific CPS and pattern.

    Args:
        cps: Cycles per second
        notes_per_cycle: Number of notes per cycle

    Returns:
        Configured TimingAnalyzer
    """
    cycle_duration = 1.0 / cps
    note_interval = cycle_duration / notes_per_cycle
    return TimingAnalyzer(note_interval)
