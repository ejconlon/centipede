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
from typing import NewType, Optional, override

from centipede.actor import Actor, ActorEnv, Mutex, Sender, System, Task
from minipat.arc import Arc
from minipat.common import ONE_HALF, ZERO, CycleTime, PosixTime
from minipat.ev import Ev
from minipat.stream import Stream
from spiny.heapmap import PHeapMap
from spiny.map import PMap

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


# class Processor[T, U](metaclass=ABCMeta):
#     @abstractmethod
#     def process_event(self, instant: Instant, orbit: Orbit, arc: Arc, val: T) -> PSeq[U]:
#         raise NotImplementedError
#
#     @abstractmethod
#     def process_play(self) -> U:
#         raise NotImplementedError
#
#     @abstractmethod
#     def process_pause(self) -> U:
#         raise NotImplementedError
#
#     @abstractmethod
#     def process_mute(self, orbit: Orbit) -> U:
#         raise NotImplementedError


class Backend[T](metaclass=ABCMeta):
    """Abstract interface for pattern event backends.

    Backends are responsible for processing and outputting pattern events
    (e.g., to audio systems, MIDI, OSC, etc.).
    """

    @abstractmethod
    def send_events(
        self,
        instant: Instant,
        orbit_events: PMap[Orbit, PHeapMap[Arc, Ev[T]]],
    ) -> None:
        """Send events with attributes to the backend.

        Args:
            instant: Timing information for this generation.
            orbit_events: Events grouped by orbit.
        """
        raise NotImplementedError


class LogBackend[T](Backend[T]):
    """Debug backend that logs events instead of playing them."""

    def __init__(self):
        self._logger = logging.getLogger("log_backend")

    @override
    def send_events(
        self,
        instant: Instant,
        orbit_events: PMap[Orbit, PHeapMap[Arc, Ev[T]]],
    ) -> None:
        total_events = sum(len(events) for events in orbit_events.values())
        self._logger.debug(
            "Received %d events across %d orbits at cycle %s (CPS: %s)",
            total_events,
            len(orbit_events),
            instant.cycle_time,
            instant.cps,
        )
        if self._logger.getEffectiveLevel() <= logging.DEBUG:
            for orbit, events in orbit_events:
                for arc, ev in events:
                    # Calculate posix time for the arc using instant timing info
                    arc_start_posix = instant.posix_start + (
                        float(arc.start) / float(instant.cps)
                    )
                    arc_end_posix = instant.posix_start + (
                        float(arc.end) / float(instant.cps)
                    )
                    self._logger.debug(
                        "Orbit %s Event @%s [posix: %.3f-%.3f]: %s",
                        orbit,
                        arc,
                        arc_start_posix,
                        arc_end_posix,
                        ev,
                    )


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


class BackendMessage[T](metaclass=ABCMeta):
    pass


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


class PatternActor[T](Actor[PatternMessage[T]]):
    """Actor that manages pattern state and generates events.

    Receives generation requests with timing information and produces events.
    """

    def __init__(
        self, pattern_state: PatternState[T], backend: Backend[T], env: LiveEnv
    ):
        self._pattern_state = pattern_state
        self._backend = backend
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

        # Collect events from all active orbits
        orbit_events = PMap[Orbit, PHeapMap[Arc, Ev[T]]].empty()

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

            # Add events to orbit events map
            if events:
                event_map = PHeapMap[Arc, Ev[T]].empty()
                for event_arc, ev in events:
                    event_map = event_map.insert(event_arc, ev)

                orbit_events = orbit_events.put(orbit, event_map)
                logger.debug("Generated %s events for orbit %s", len(events), orbit)

        # Send all events to backend if any exist
        if orbit_events:
            self._backend.send_events(instant, orbit_events)

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


class LiveSystem[T]:
    """Main interface for the live pattern system.

    Provides high-level controls for pattern playback using the actor system.
    """

    def __init__(
        self,
        transport_sender: Sender[TransportMessage[T]],
        pattern_sender: Sender[PatternMessage[T]],
    ):
        """Initialize the live system.

        Args:
            backend: Backend for processing pattern events.
            transport_sender: Sender for transport control messages.
            pattern_sender: Sender for pattern messages.
            env: Environment configuration.
        """
        self._logger = logging.getLogger("minipat.live")
        self._transport_sender = transport_sender
        self._pattern_sender = pattern_sender

    @staticmethod
    def start(
        system: System,
        backend: Backend[T],
        env: LiveEnv = LiveEnv(),
    ) -> LiveSystem[T]:
        """Start the live pattern system.

        Args:
            system: The actor system to use.
            backend: Backend for processing pattern events.
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
        pattern_actor = PatternActor(pattern_state, backend, env)
        pattern_sender = system.spawn_actor("pattern", pattern_actor)

        # Create the transport actor for handling control messages
        transport_actor: TransportActor[T] = TransportActor(transport_state_mutex)
        transport_sender = system.spawn_actor("transport", transport_actor)

        # Create the live system with the senders
        live_system = LiveSystem(transport_sender, pattern_sender)

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
