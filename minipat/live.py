"""Live pattern playback system inspired by minipat-live/Core.hs.

This module provides real-time pattern playback capabilities using the centipede
actor system for concurrent state management and event generation.
"""

from __future__ import annotations

import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from logging import Logger
from threading import Event
from typing import List, NewType, Optional, override

from bad_actor import Actor, ActorEnv, Mutex, Sender, System, Task
from minipat.arc import CycleArc
from minipat.common import (
    ONE,
    ONE_HALF,
    ZERO,
    CycleDelta,
    CycleTime,
    PosixTime,
    frac_ceil,
)
from minipat.ev import EvHeap, ev_heap_empty
from minipat.stream import Stream
from spiny.map import PMap
from spiny.seq import PSeq

Orbit = NewType("Orbit", int)

# =============================================================================
# Timing Constants
# =============================================================================

DEFAULT_CPS = ONE_HALF
"""Default cycles per second (tempo)."""

DEFAULT_GENERATIONS_PER_CYCLE = 4
"""Default number of event generations to calculate per cycle."""

DEFAULT_WAIT_FACTOR = Fraction(1, 4)
"""Default factor for calculating sleep intervals - use 1/4 of generation interval for responsive polling."""


def calculate_sleep_interval(
    cps: Fraction,
    generations_per_cycle: int,
    wait_factor: Fraction = DEFAULT_WAIT_FACTOR,
) -> float:
    """Calculate appropriate sleep interval based on timing configuration.

    Uses the specified wait_factor of the generation interval to ensure responsive timing
    while avoiding excessive CPU usage.

    Args:
        cps: Current cycles per second (tempo).
        generations_per_cycle: Number of event generations per cycle.
        wait_factor: Factor for calculating sleep intervals - fraction of generation interval to use.

    Returns:
        Sleep interval in seconds, clamped to reasonable bounds.
    """
    # Calculate time per generation
    generation_interval = 1.0 / (float(cps) * generations_per_cycle)

    # Use wait_factor of generation interval for responsive polling
    sleep_interval = generation_interval * float(wait_factor)

    # Clamp to reasonable bounds (0.5ms to 50ms)
    return max(0.0005, min(0.05, sleep_interval))


@dataclass(frozen=True)
class Timing:
    """Configuration for timing calculations (frozen for immutability)."""

    cps: Fraction
    """Current cycles per second (tempo)."""

    generations_per_cycle: int
    """Number of event generations to calculate per cycle."""

    wait_factor: Fraction
    """Factor for calculating sleep intervals - fraction of generation interval to use for polling."""

    @classmethod
    def default(cls) -> "Timing":
        """Create a Timing instance with default values."""
        return cls(
            cps=DEFAULT_CPS,
            generations_per_cycle=DEFAULT_GENERATIONS_PER_CYCLE,
            wait_factor=DEFAULT_WAIT_FACTOR,
        )

    def get_sleep_interval(self) -> float:
        """Calculate appropriate sleep interval based on timing configuration.

        Uses the shared calculation function for consistency.

        Returns:
            Sleep interval in seconds, clamped to reasonable bounds.
        """
        return calculate_sleep_interval(
            self.cps, self.generations_per_cycle, self.wait_factor
        )


@dataclass(frozen=True)
class Instant:
    """Represents a specific moment in time with cycle and posix timing information."""

    cycle_time: CycleTime
    """The cycle time for this instant."""

    cps: Fraction
    """Cycles per second at this instant."""

    posix_start: PosixTime
    """Fixed posix time reference point."""


@dataclass(frozen=True)
class LiveEnv:
    """Environment configuration for live pattern playback.

    Defines global settings that control pattern playback behavior.
    """

    generations_per_cycle: int = DEFAULT_GENERATIONS_PER_CYCLE
    """Number of event generations to calculate per cycle."""

    wait_factor: Fraction = DEFAULT_WAIT_FACTOR
    """Factor for calculating sleep intervals - fraction of generation interval to use for polling."""


@dataclass
class OrbitState[T]:
    """State for a single orbit (audio channel/stream).

    Each orbit contains a stream and maintains its own playback state.
    """

    stream: Optional[Stream[T]] = None
    """Current stream playing on this orbit."""

    muted: bool = False
    """Whether this orbit is muted."""

    solo: bool = False
    """Whether this orbit is soloed."""


@dataclass
class TransportState:
    """Mutable state for transport control (timing, playback).

    Handles tempo, playback state, and timing information.
    """

    playing: bool = False
    """Whether pattern playback is currently active."""

    current_cycle: CycleTime = CycleTime(ZERO)
    """Current cycle position in the timeline."""

    cps: Fraction = ONE_HALF
    """Current cycles per second (tempo)."""

    playback_start: PosixTime = PosixTime(0.0)
    """Wall clock time when playback started (seconds since epoch)."""


# Use Mutex[TransportState] directly instead of custom wrapper


@dataclass
class PatternState[T]:
    """Mutable state for pattern management.

    Handles orbits, streams, and pattern-related state.
    """

    orbits: PMap[Orbit, OrbitState[T]] = field(default_factory=PMap.empty)
    """Map of orbit to orbit state."""


class Processor[T, U](metaclass=ABCMeta):
    """Abstract interface for processing pattern events.

    Processors transform pattern events from one type to another
    (e.g., to MIDI messages, OSC messages, audio events, etc.).
    """

    @abstractmethod
    def process(
        self, instant: Instant, orbit: Optional[Orbit], events: EvHeap[T]
    ) -> PSeq[U]:
        """Process events from a single orbit (or global context).

        Args:
            instant: Timing information for this generation.
            orbit: The orbit these events belong to, or None if not specified.
            events: Events to process.

        Returns:
            Processed events as a sequence.
        """
        raise NotImplementedError


class LogProcessor[T](Processor[T, str]):
    """Debug processor that converts events to log strings."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("log_processor")

    @override
    def process(
        self, instant: Instant, orbit: Optional[Orbit], events: EvHeap[T]
    ) -> PSeq[str]:
        """Process events into log strings."""
        log_messages = []

        for span, ev in events:
            # Calculate posix time for the arc using instant timing info
            arc_start_posix = instant.posix_start + (
                float(span.active.start) / float(instant.cps)
            )
            arc_end_posix = instant.posix_start + (
                float(span.active.end) / float(instant.cps)
            )

            log_msg = f"Orbit {orbit} Event @{span.active} [posix: {arc_start_posix:.3f}-{arc_end_posix:.3f}]: {ev}"
            log_messages.append(log_msg)

        return PSeq.mk(log_messages)


class TransportMessage[T](metaclass=ABCMeta):
    """Base class for messages sent to the transport actor."""

    pass


@dataclass(frozen=True)
class TransportSetCps[T](TransportMessage[T]):
    """Set the cycles per second (tempo)."""

    cps: Fraction


@dataclass(frozen=True)
class TransportPlay[T](TransportMessage[T]):
    """Set the playing state."""

    playing: bool


@dataclass(frozen=True)
class TransportSetCycle[T](TransportMessage[T]):
    """Set the current cycle position."""

    cycle: CycleTime


class PatternMessage[T](metaclass=ABCMeta):
    """Base class for messages sent to the pattern state actor."""

    pass


@dataclass(frozen=True)
class PatternGenerate[T](PatternMessage[T]):
    """Request to generate events for a specific instant."""

    instant: Instant


@dataclass(frozen=True)
class PatternSetOrbit[T](PatternMessage[T]):
    """Set the stream for a specific orbit."""

    orbit: Orbit
    stream: Optional[Stream[T]]


@dataclass(frozen=True)
class PatternMuteOrbit[T](PatternMessage[T]):
    """Mute or unmute an orbit."""

    orbit: Orbit
    muted: bool


@dataclass(frozen=True)
class PatternSoloOrbit[T](PatternMessage[T]):
    """Solo or unsolo an orbit."""

    orbit: Orbit
    soloed: bool


@dataclass(frozen=True)
class PatternClearOrbits[T](PatternMessage[T]):
    """Clear all patterns from orbits."""

    pass


@dataclass(frozen=True)
class PatternOnce[T](PatternMessage[T]):
    """Send pre-evaluated events immediately."""

    instant: Instant
    orbit: Optional[Orbit]
    events: EvHeap[T]


class BackendMessage[U](metaclass=ABCMeta):
    """Base class for messages sent to backend processors."""

    pass


@dataclass(frozen=True)
class BackendPlay[U](BackendMessage[U]):
    """If playing, the backend should resume processing new messages.
    Otherwise, it should reset, clear all state, and discard new messages.
    """

    playing: bool


@dataclass(frozen=True)
class BackendEvents[U](BackendMessage[U]):
    """Events to be processed by the backend."""

    events: PSeq[U]


@dataclass(frozen=True)
class BackendTiming[U](BackendMessage[U]):
    """Update timing configuration for the backend."""

    timing: Timing
    """Timing configuration."""


class TimerTask[T](Task):
    """Task that manages timing and coordinates pattern generation."""

    def __init__(
        self,
        pattern_sender: Sender[PatternMessage[T]],
        transport_sender: Sender[TransportMessage[T]],
        env: LiveEnv,
        transport_state_mutex: Mutex[TransportState],
    ):
        self._pattern_sender = pattern_sender
        self._transport_sender = transport_sender
        self._env = env
        self._transport_state_mutex = transport_state_mutex

    @override
    def run(self, logger: Logger, halt: Event) -> None:
        logger.debug("Timer task starting")

        frac_cycle_length = Fraction(1) / self._env.generations_per_cycle
        float_cycle_length = float(frac_cycle_length)

        # Debug: Log the actual generations_per_cycle value
        # with open("midi_debug.log", "a") as f:
        #     f.write(f"TimerTask: generations_per_cycle={self._env.generations_per_cycle} frac_cycle_length={frac_cycle_length} float_cycle_length={float_cycle_length}\n")

        while not halt.is_set():
            # Get current state and decide what to do
            with self._transport_state_mutex as state:
                playing = state.playing
                if playing:
                    interval = float_cycle_length / state.cps
                    instant = Instant(
                        cycle_time=state.current_cycle,
                        cps=state.cps,
                        posix_start=state.playback_start,
                    )
                    state.current_cycle = CycleTime(
                        state.current_cycle + frac_cycle_length
                    )
                else:
                    # When not playing, use current CPS to calculate pause interval
                    interval = calculate_sleep_interval(
                        state.cps,
                        self._env.generations_per_cycle,
                        self._env.wait_factor,
                    )
                    instant = None

            if instant is not None:
                # Send generation request
                self._pattern_sender.send(PatternGenerate(instant))

                # Wait for next interval or halt
                if halt.wait(timeout=interval):
                    break
            else:
                # Not playing, wait a bit before checking again
                if halt.wait(timeout=interval):
                    break

        logger.debug("Timer task stopping")


class TransportActor[T](Actor[TransportMessage[T]]):
    """Actor that handles timing control messages."""

    def __init__(self, transport_state_mutex: Mutex[TransportState]):
        self._transport_state_mutex = transport_state_mutex

    @override
    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Transport actor started")

    @override
    def on_message(self, env: ActorEnv, value: TransportMessage[T]) -> None:
        match value:
            case TransportSetCps(cps):
                self._set_cps(env.logger, cps)
            case TransportPlay(playing):
                self._set_playing(env.logger, playing)
            case TransportSetCycle(cycle):
                self._set_cycle(env.logger, cycle)

    def _set_cps(self, logger: Logger, cps: Fraction) -> None:
        """Set the cycles per second (tempo)."""
        with self._transport_state_mutex as state:
            state.cps = cps
        logger.debug("Set CPS to %s", cps)

    def _set_playing(self, logger: Logger, playing: bool) -> None:
        """Set the playing state."""
        with self._transport_state_mutex as state:
            state.playing = playing
            if playing:
                # Record start time when starting
                state.playback_start = PosixTime(time.time())
            else:
                # Reset start time when stopping
                state.playback_start = PosixTime(0.0)
        logger.debug("Set playing to %s", playing)

    def _set_cycle(self, logger: Logger, cycle: CycleTime) -> None:
        """Set the current cycle position."""
        with self._transport_state_mutex as state:
            state.current_cycle = cycle
        logger.debug("Set cycle to %s", cycle)


class PatternActor[T, U](Actor[PatternMessage[T]]):
    """Actor that manages pattern state and generates events.

    Receives generation requests with timing information and produces events.
    """

    def __init__(
        self,
        pattern_state: PatternState[T],
        processor: Processor[T, U],
        backend_sender: Sender[BackendMessage[U]],
        env: LiveEnv,
    ):
        self._pattern_state = pattern_state
        self._processor = processor
        self._backend_sender = backend_sender
        self._env = env

    @override
    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Pattern actor started")

    @override
    def on_message(self, env: ActorEnv, value: PatternMessage[T]) -> None:
        match value:
            case PatternGenerate(instant):
                self._generate_events(env.logger, instant)
            case PatternSetOrbit(orbit, stream):
                self._set_orbit(env.logger, orbit, stream)
            case PatternMuteOrbit(orbit, muted):
                self._mute_orbit(env.logger, orbit, muted)
            case PatternSoloOrbit(orbit, solo):
                self._solo_orbit(env.logger, orbit, solo)
            case PatternClearOrbits():
                self.clear_all_patterns(env.logger)
            case PatternOnce(instant, orbit, events):
                self._generate_once(env.logger, instant, orbit, events)

    def _generate_events(self, logger: Logger, instant: Instant) -> None:
        """Generate events for the given instant."""
        logger.debug("Generating events for cycle %s", instant.cycle_time)

        # Calculate the time arc for this generation
        cycle_start = instant.cycle_time
        cycle_length = Fraction(1) / self._env.generations_per_cycle
        cycle_end = CycleTime(cycle_start + cycle_length)
        arc = CycleArc(cycle_start, cycle_end)

        # Debug: Log generation details
        # with open("midi_debug.log", "a") as f:
        #     f.write(f"GenerateEvents: generations_per_cycle={self._env.generations_per_cycle} cycle_length={cycle_length} arc={arc.start}-{arc.end}\n")

        # Check if any orbits are soloed
        has_solo = any(
            orbit_state.solo for _, orbit_state in self._pattern_state.orbits
        )

        # Collect and process events from all active orbits
        all_processed_events: List[U] = []

        for orbit, orbit_state in self._pattern_state.orbits:
            if orbit_state.stream is None:
                continue

            # Skip muted orbits, unless they're soloed
            if orbit_state.muted and not orbit_state.solo:
                continue

            # If there are soloed orbits, only play soloed ones
            if has_solo and not orbit_state.solo:
                continue

            # Generate events using the orbit stream
            events = orbit_state.stream.unstream(arc)

            # Convert events to EvHeap
            if events:
                event_heap: EvHeap[T] = ev_heap_empty()
                for span, ev in events:
                    event_heap = event_heap.insert(span, ev)

                # Process events for this orbit
                processed_events = self._processor.process(instant, orbit, event_heap)
                all_processed_events.extend(processed_events)
                logger.debug("Generated %s events for orbit %s", len(events), orbit)

        # Send all processed events to backend if any exist
        if all_processed_events:
            combined_events: PSeq[U] = PSeq.mk(all_processed_events)
            self._backend_sender.send(BackendEvents(combined_events))

    def _set_orbit(
        self, logger: Logger, orbit: Orbit, stream: Optional[Stream[T]]
    ) -> None:
        """Set the stream for a specific orbit."""
        # Get existing orbit state or create new one
        existing_state = self._pattern_state.orbits.get(orbit, OrbitState())
        updated_state = OrbitState(
            stream=stream, muted=existing_state.muted, solo=existing_state.solo
        )
        self._pattern_state.orbits = self._pattern_state.orbits.put(
            orbit, updated_state
        )

        if stream is None:
            logger.debug("Cleared orbit %s", orbit)
        else:
            logger.debug("Set stream for orbit %s", orbit)

    def _mute_orbit(self, logger: Logger, orbit: Orbit, muted: bool) -> None:
        """Mute or unmute an orbit."""
        # Get existing orbit state or create new one
        existing_state = self._pattern_state.orbits.get(orbit, OrbitState())
        updated_state = OrbitState(
            stream=existing_state.stream, muted=muted, solo=existing_state.solo
        )
        self._pattern_state.orbits = self._pattern_state.orbits.put(
            orbit, updated_state
        )
        logger.debug("Set orbit %s muted to %s", orbit, muted)

    def _solo_orbit(self, logger: Logger, orbit: Orbit, solo: bool) -> None:
        """Solo or unsolo an orbit."""
        # Get existing orbit state or create new one
        existing_state = self._pattern_state.orbits.get(orbit, OrbitState())
        updated_state = OrbitState(
            stream=existing_state.stream, muted=existing_state.muted, solo=solo
        )
        self._pattern_state.orbits = self._pattern_state.orbits.put(
            orbit, updated_state
        )
        logger.debug("Set orbit %s solo to %s", orbit, solo)

    def _generate_once(
        self,
        logger: Logger,
        instant: Instant,
        orbit: Optional[Orbit],
        events: EvHeap[T],
    ) -> None:
        """Process pre-evaluated events and send them immediately."""
        logger.debug("Processing %d once events", len(events))

        if not events.empty():
            # Process events through the processor
            processed_events = self._processor.process(instant, orbit, events)

            # Send all processed events to the backend immediately
            if not processed_events.empty():
                backend_events: BackendEvents[U] = BackendEvents(processed_events)
                self._backend_sender.send(backend_events)
                logger.debug("Sent %d once events to backend", len(processed_events))

    def clear_all_patterns(self, logger: Logger) -> None:
        """Clear all streams from orbits."""
        logger.info("Clearing all patterns")

        # Clear all streams
        self._pattern_state.orbits = PMap.empty()


class LiveSystem[T, U]:
    """Main interface for the live pattern system.

    Provides high-level controls for pattern playback using the actor system.
    """

    def __init__(
        self,
        transport_sender: Sender[TransportMessage[T]],
        pattern_sender: Sender[PatternMessage[T]],
        backend_sender: Sender[BackendMessage[U]],
        env: LiveEnv,
        transport_state_mutex: Mutex[TransportState],
    ):
        """Initialize the live system.

        Args:
            transport_sender: Sender for transport control messages.
            pattern_sender: Sender for pattern messages.
            backend_sender: Sender for backend messages.
            env: Environment configuration.
            transport_state_mutex: Mutex for accessing transport state.
        """
        self._logger = logging.getLogger("minipat.live")
        self._transport_sender = transport_sender
        self._pattern_sender = pattern_sender
        self._backend_sender = backend_sender
        self._env = env
        self._transport_state_mutex = transport_state_mutex

    @staticmethod
    def start(
        system: System,
        processor: Processor[T, U],
        backend_sender: Sender[BackendMessage[U]],
        cps: Optional[Fraction] = None,
        env: Optional[LiveEnv] = None,
    ) -> LiveSystem[T, U]:
        """Create and start the live pattern system.

        Args:
            system: The actor system to use.
            processor: Processor for transforming pattern events.
            backend_sender: Sender for backend messages.
            cps: Starting CPS (optional)
            env: Environment configuration (optional).

        Returns:
            A started LiveSystem instance.
        """
        cps = cps if cps is not None else DEFAULT_CPS
        env = env if env is not None else LiveEnv()

        logger = logging.getLogger("minipat")
        logger.info("Starting live pattern system")

        # Create state objects for actors
        transport_state = TransportState(cps=cps)
        transport_state_mutex = Mutex(transport_state)
        pattern_state = PatternState[T]()

        # Create the pattern actor
        pattern_actor = PatternActor(pattern_state, processor, backend_sender, env)
        pattern_sender = system.spawn_actor("pattern", pattern_actor)

        # Create the transport actor for handling control messages
        transport_actor: TransportActor[T] = TransportActor(transport_state_mutex)
        transport_sender = system.spawn_actor("transport", transport_actor)

        # Create the live system with the senders
        live_system = LiveSystem(
            transport_sender, pattern_sender, backend_sender, env, transport_state_mutex
        )

        # Create and spawn the timer task for timing loop
        timer_task = TimerTask(
            pattern_sender,
            transport_sender,
            env,
            transport_state_mutex,
        )
        system.spawn_task("timer_loop", timer_task)

        logger.info("Live pattern system started")

        # Send initial timing configuration to backend
        timing = Timing(
            cps=transport_state.cps,
            generations_per_cycle=env.generations_per_cycle,
            wait_factor=env.wait_factor,
        )
        initial_timing: BackendTiming[U] = BackendTiming(timing=timing)
        backend_sender.send(initial_timing)
        logger.debug(
            "Sent initial timing configuration - CPS: %s, Gens/Cycle: %d, Wait Factor: %s",
            timing.cps,
            timing.generations_per_cycle,
            timing.wait_factor,
        )

        return live_system

    def set_orbit(self, orbit: Orbit, stream: Optional[Stream[T]]) -> None:
        """Set the stream for a specific orbit.

        Args:
            orbit: The orbit identifier.
            stream: The stream to set, or None to clear.
        """
        self._pattern_sender.send(PatternSetOrbit(orbit, stream))

    def clear_orbits(self) -> None:
        self._pattern_sender.send(PatternClearOrbits())

    def set_cps(self, cps: Fraction) -> None:
        """Set the cycles per second (tempo).

        Args:
            cps: The new tempo in cycles per second.
        """
        self._transport_sender.send(TransportSetCps(cps))
        # Also send timing update to backend
        timing = Timing(
            cps=cps,
            generations_per_cycle=self._env.generations_per_cycle,
            wait_factor=self._env.wait_factor,
        )
        timing_update: BackendTiming[U] = BackendTiming(timing=timing)
        self._backend_sender.send(timing_update)

    def set_cycle(self, cycle: CycleTime) -> None:
        """Set the current cycle position.

        Args:
            cycle: The cycle position to set.
        """
        self._transport_sender.send(TransportSetCycle(cycle))

    def play(self, playing: bool = True) -> None:
        """Start pattern playback."""
        self._transport_sender.send(TransportPlay(playing))
        self._backend_sender.send(BackendPlay(playing))

    def pause(self) -> None:
        """Stop pattern playback."""
        self.play(False)

    def playing(self) -> bool:
        """Check if the pattern system is currently playing.

        Returns:
            True if playback is active, False if paused or stopped.
        """
        with self._transport_state_mutex as state:
            return state.playing

    def mute(self, orbit: Orbit, muted: bool = True) -> None:
        """Mute or unmute an orbit.

        Args:
            orbit: The orbit identifier.
            muted: Whether to mute the orbit.
        """
        self._pattern_sender.send(PatternMuteOrbit(orbit, muted))

    def unmute(self, orbit: Orbit) -> None:
        """Unmute an orbit.

        Args:
            orbit: The orbit identifier.
        """
        self.mute(orbit, False)

    def solo(self, orbit: Orbit, soloed: bool = True) -> None:
        """Solo or unsolo an orbit.

        Args:
            orbit: The orbit identifier.
            solo: Whether to solo the orbit.
        """
        self._pattern_sender.send(PatternSoloOrbit(orbit, soloed))

    def unsolo(self, orbit: Orbit) -> None:
        """Unsolo an orbit.

        Args:
            orbit: The orbit identifier.
        """
        self.solo(orbit, False)

    def once(
        self,
        stream: Stream[T],
        length: Optional[CycleDelta] = None,
        aligned: Optional[bool] = None,
        orbit: Optional[Orbit] = None,
    ) -> None:
        """Generate and immediately send events for a stream over a specified duration.

        Args:
            stream: The stream to generate events from.
            length: Duration of the generation in cycle time (defaults to 1 cycle).
            aligned: If True, start at the next cycle boundary; if False, start immediately.
            orbit: Optional orbit to use for event processing.
        """
        length = length if length is not None else CycleDelta(ONE)
        aligned = aligned if aligned is not None else False

        # Eagerly evaluate the stream
        with self._transport_state_mutex as state:
            current_cycle = state.current_cycle
            current_cps = state.cps
            posix_start = state.playback_start

        # Calculate start time
        generation_cycle_length = Fraction(1) / self._env.generations_per_cycle
        minimum_future_time = current_cycle + generation_cycle_length

        if aligned:
            # Start at the next cycle boundary, but ensure it's at least one generation cycle ahead
            next_cycle_boundary = CycleTime(Fraction(frac_ceil(current_cycle)))
            if next_cycle_boundary < minimum_future_time:
                # Next cycle boundary is too close, use the one after that
                start_cycle = CycleTime(next_cycle_boundary + 1)
            else:
                start_cycle = next_cycle_boundary
        else:
            # Start at one generation cycle in the future
            start_cycle = CycleTime(minimum_future_time)

        # Calculate end time
        end_cycle = CycleTime(start_cycle + length)

        # Create arc and instant
        arc = CycleArc(start_cycle, end_cycle)
        instant = Instant(
            cycle_time=start_cycle, cps=current_cps, posix_start=posix_start
        )

        # Generate events using the stream (outside the mutex)
        events = stream.unstream(arc)
        event_heap: EvHeap[T] = ev_heap_empty()

        for span, ev in events:
            event_heap = event_heap.insert(span, ev)

        # Send pattern once message to the pattern actor
        once_message: PatternOnce[T] = PatternOnce(
            instant=instant,
            orbit=orbit,
            events=event_heap,
        )
        self._pattern_sender.send(once_message)

        self._logger.debug(
            "Sent once message: cycles %s-%s, aligned=%s, length=%s, events=%d",
            start_cycle,
            end_cycle,
            aligned,
            length,
            len(event_heap),
        )

    def panic(self) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        self.pause()
        self.clear_orbits()
