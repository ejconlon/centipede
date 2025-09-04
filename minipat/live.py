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
from minipat.arc import Arc
from minipat.common import ONE_HALF, ZERO, CycleTime, PosixTime
from minipat.ev import EvHeap
from minipat.stream import Stream
from spiny.map import PMap
from spiny.seq import PSeq

Orbit = NewType("Orbit", int)


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

    debug: bool = False
    """Enable debug logging for pattern playback."""

    generations_per_cycle: int = 4
    """Number of event generations to calculate per cycle."""

    pause_interval: float = 0.1
    """Interval in seconds to check halt when paused."""


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
    def process(self, instant: Instant, orbit: Orbit, events: EvHeap[T]) -> PSeq[U]:
        """Process events from a single orbit.

        Args:
            instant: Timing information for this generation.
            orbit: The orbit these events belong to.
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
    def process(self, instant: Instant, orbit: Orbit, events: EvHeap[T]) -> PSeq[str]:
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
                    interval = self._env.pause_interval
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

    def _generate_events(self, logger: Logger, instant: Instant) -> None:
        """Generate events for the given instant."""
        logger.debug("Generating events for cycle %s", instant.cycle_time)

        # Calculate the time arc for this generation
        cycle_start = instant.cycle_time
        cycle_length = Fraction(1) / self._env.generations_per_cycle
        cycle_end = CycleTime(cycle_start + cycle_length)
        arc = Arc(cycle_start, cycle_end)

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
                from minipat.ev import ev_heap_empty

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
    ):
        """Initialize the live system.

        Args:
            transport_sender: Sender for transport control messages.
            pattern_sender: Sender for pattern messages.
            backend_sender: Sender for backend messages.
        """
        self._logger = logging.getLogger("minipat.live")
        self._transport_sender = transport_sender
        self._pattern_sender = pattern_sender
        self._backend_sender = backend_sender

    @staticmethod
    def start(
        system: System,
        processor: Processor[T, U],
        backend_sender: Sender[BackendMessage[U]],
        env: LiveEnv = LiveEnv(),
    ) -> LiveSystem[T, U]:
        """Start the live pattern system.

        Args:
            system: The actor system to use.
            processor: Processor for transforming pattern events.
            backend_sender: Sender for backend messages.
            env: Environment configuration.

        Returns:
            A started LiveSystem instance.
        """
        logger = logging.getLogger("minipat")
        logger.info("Starting live pattern system")

        # Create state objects for actors
        transport_state = TransportState()
        transport_state_mutex = Mutex(transport_state)
        pattern_state = PatternState[T]()

        # Create the pattern actor
        pattern_actor = PatternActor(pattern_state, processor, backend_sender, env)
        pattern_sender = system.spawn_actor("pattern", pattern_actor)

        # Create the transport actor for handling control messages
        transport_actor: TransportActor[T] = TransportActor(transport_state_mutex)
        transport_sender = system.spawn_actor("transport", transport_actor)

        # Create the live system with the senders
        live_system = LiveSystem(transport_sender, pattern_sender, backend_sender)

        # Create and spawn the timer task for timing loop
        timer_task = TimerTask(
            pattern_sender,
            transport_sender,
            env,
            transport_state_mutex,
        )
        system.spawn_task("timer_loop", timer_task)

        logger.info("Live pattern system started")
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

    def panic(self) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        self.pause()
        self.clear_orbits()
