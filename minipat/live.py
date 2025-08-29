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
from typing import Dict, NewType, Optional, override

from centipede.actor import Actor, ActorEnv, Sender, Task
from minipat.arc import Arc
from minipat.common import ONE, ZERO, CycleTime, PosixTime
from minipat.ev import Ev
from minipat.pat import Pat
from minipat.stream import Stream, pat_stream
from spiny.heapmap import PHeapMap
from spiny.pmap import PMap

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
class LiveDomain[T]:
    """Mutable state for the live pattern system.

    Tracks the current playback state including orbits, tempo, and cycles.
    """

    playing: bool = False
    """Whether pattern playback is currently active."""

    current_cycle: CycleTime = CycleTime(ZERO)
    """Current cycle position in the timeline."""

    orbits: Dict[Orbit, OrbitState[T]] = field(default_factory=dict)
    """Map of orbit to orbit state."""

    cps: Fraction = ONE
    """Current cycles per second (tempo)."""

    generations_per_cycle: int = 4
    """Current number of generations per cycle."""

    playback_start_time: Optional[PosixTime] = None
    """Wall clock time when playback started (seconds since epoch)."""


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

    def __init__(self, logger: Logger):
        self._logger = logger

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


@dataclass
class LiveState[T]:
    """Complete state for the live pattern system.

    Contains all configuration, mutable state, and backend references.
    """

    logger: Logger
    backend: Backend[T]
    env: LiveEnv
    domain: LiveDomain[T] = field(default_factory=LiveDomain)


class TransportMessage[T](metaclass=ABCMeta):
    """Base class for messages sent to the transport actor."""

    pass


class PatternMessage[T](metaclass=ABCMeta):
    """Base class for messages sent to the pattern state actor."""

    pass


@dataclass(frozen=True)
class GenerateEvents[T](PatternMessage[T]):
    """Request to generate events for a specific instant."""

    instant: Instant


@dataclass(frozen=True)
class SetOrbit[T](PatternMessage[T]):
    """Set the stream for a specific orbit."""

    orbit: Orbit
    stream: Optional[Stream[T]]


@dataclass(frozen=True)
class MuteOrbit[T](PatternMessage[T]):
    """Mute or unmute an orbit."""

    orbit: Orbit
    muted: bool


@dataclass(frozen=True)
class SoloOrbit[T](PatternMessage[T]):
    """Solo or unsolo an orbit."""

    orbit: Orbit
    solo: bool


@dataclass(frozen=True)
class SetCps[T](TransportMessage[T]):
    """Set the cycles per second (tempo)."""

    cps: Fraction


@dataclass(frozen=True)
class SetPlaying[T](TransportMessage[T]):
    """Set the playing state."""

    playing: bool


@dataclass(frozen=True)
class Panic[T](TransportMessage[T]):
    """Emergency stop - clear all patterns and stop playbook."""

    pass


class TimerTask[T](Task):
    """Task that manages timing and coordinates pattern generation."""

    def __init__(
        self,
        pattern_sender: Sender[PatternMessage[T]],
        transport_sender: Sender[TransportMessage[T]],
        env: LiveEnv,
        domain: LiveDomain[T],
    ):
        self._pattern_sender = pattern_sender
        self._transport_sender = transport_sender
        self._env = env
        self._domain = domain

    @override
    def run(self, logger: Logger, halt: Event) -> None:
        logger.debug("Timer task starting")

        while not halt.is_set():
            if self._domain.playing and self._domain.playback_start_time is not None:
                # Calculate current interval based on CPS
                interval = 1.0 / (
                    float(self._domain.cps) * self._env.generations_per_cycle
                )

                # Create instant for this generation
                instant = Instant(
                    cycle_time=self._domain.current_cycle,
                    cps=self._domain.cps,
                    posix_start=self._domain.playback_start_time,
                )

                # Send generation request
                self._pattern_sender.send(GenerateEvents(instant))

                # Advance cycle position
                cycle_length = Fraction(1) / self._env.generations_per_cycle
                self._domain.current_cycle = CycleTime(
                    self._domain.current_cycle + cycle_length
                )

                # Wait for next interval or halt
                if halt.wait(timeout=interval):
                    break
            else:
                # Not playing, wait a bit before checking again
                if halt.wait(timeout=self._env.pause_interval):
                    break

        logger.debug("Timer task stopping")


class TransportActor[T](Actor[TransportMessage[T]]):
    """Actor that handles timing control messages."""

    def __init__(self, domain: LiveDomain[T]):
        self._domain = domain

    @override
    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Transport actor started")

    @override
    def on_message(self, env: ActorEnv, value: TransportMessage[T]) -> None:
        match value:
            case SetCps(cps):
                self._set_cps(env.logger, cps)
            case SetPlaying(playing):
                self._set_playing(env.logger, playing)
            case Panic():
                self._panic(env.logger)

    def _set_cps(self, logger: Logger, cps: Fraction) -> None:
        """Set the cycles per second (tempo)."""
        self._domain.cps = cps
        logger.debug("Set CPS to %s", cps)

    def _set_playing(self, logger: Logger, playing: bool) -> None:
        """Set the playing state."""
        self._domain.playing = playing
        logger.debug("Set playing to %s", playing)

        if playing:
            # Reset cycle position and record start time when starting
            self._domain.current_cycle = CycleTime(Fraction(0))
            self._domain.playback_start_time = PosixTime(time.time())
        else:
            # Clear start time when stopping
            self._domain.playback_start_time = None

    def _panic(self, logger: Logger) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        logger.info("PANIC: Stopping playback")

        # Stop playback
        self._domain.playing = False
        self._domain.playback_start_time = None
        self._domain.current_cycle = CycleTime(Fraction(0))


class PatternStateActor[T](Actor[PatternMessage[T]]):
    """Actor that manages pattern state and generates events.

    Receives generation requests with timing information and produces events.
    """

    def __init__(self, initial_state: LiveState[T]):
        self._state = initial_state

    @override
    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Pattern state actor started")

    @override
    def on_message(self, env: ActorEnv, value: PatternMessage[T]) -> None:
        match value:
            case GenerateEvents(instant):
                self._generate_events(env.logger, instant)
            case SetOrbit(orbit, stream):
                self._set_orbit(env.logger, orbit, stream)
            case MuteOrbit(orbit, muted):
                self._mute_orbit(env.logger, orbit, muted)
            case SoloOrbit(orbit, solo):
                self._solo_orbit(env.logger, orbit, solo)

    def _generate_events(self, logger: Logger, instant: Instant) -> None:
        """Generate events for the given instant."""
        logger.debug("Generating events for cycle %s", instant.cycle_time)

        # Calculate the time arc for this generation
        cycle_start = instant.cycle_time
        cycle_length = Fraction(1) / self._state.domain.generations_per_cycle
        cycle_end = CycleTime(cycle_start + cycle_length)
        arc = Arc(cycle_start, cycle_end)

        # Check if any orbits are soloed
        has_solo = any(orbit.solo for orbit in self._state.domain.orbits.values())

        # Collect events from all active orbits
        orbit_events = PMap[Orbit, PHeapMap[Arc, Ev[T]]].empty()

        for orbit, orbit_state in self._state.domain.orbits.items():
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
                    event_map = event_map.put(event_arc, ev)

                orbit_events = orbit_events.put(orbit, event_map)
                logger.debug("Generated %s events for orbit %s", len(events), orbit)

        # Send all events to backend if any exist
        if orbit_events:
            self._state.backend.send_events(instant, orbit_events)

    def _set_orbit(
        self, logger: Logger, orbit: Orbit, stream: Optional[Stream[T]]
    ) -> None:
        """Set the stream for a specific orbit."""
        if orbit not in self._state.domain.orbits:
            self._state.domain.orbits[orbit] = OrbitState()

        self._state.domain.orbits[orbit].stream = stream

        if stream is None:
            logger.debug("Cleared orbit %s", orbit)
        else:
            logger.debug("Set stream for orbit %s", orbit)

    def _mute_orbit(self, logger: Logger, orbit: Orbit, muted: bool) -> None:
        """Mute or unmute an orbit."""
        if orbit not in self._state.domain.orbits:
            self._state.domain.orbits[orbit] = OrbitState()

        self._state.domain.orbits[orbit].muted = muted
        logger.debug("Set orbit %s muted to %s", orbit, muted)

    def _solo_orbit(self, logger: Logger, orbit: Orbit, solo: bool) -> None:
        """Solo or unsolo an orbit."""
        if orbit not in self._state.domain.orbits:
            self._state.domain.orbits[orbit] = OrbitState()

        self._state.domain.orbits[orbit].solo = solo
        logger.debug("Set orbit %s solo to %s", orbit, solo)

    def clear_all_patterns(self, logger: Logger) -> None:
        """Clear all streams from orbits."""
        logger.info("Clearing all patterns")

        # Clear all streams
        for orbit_state in self._state.domain.orbits.values():
            orbit_state.stream = None
            orbit_state.muted = False
            orbit_state.solo = False


class LiveSystem[T]:
    """Main interface for the live pattern system.

    Provides high-level controls for pattern playback using the actor system.
    """

    def __init__(self, backend: Backend[T], env: LiveEnv = LiveEnv()):
        """Initialize the live system.

        Args:
            backend: Backend for processing pattern events.
            env: Environment configuration.
        """
        self._logger = logging.getLogger("minipat.live")
        self._state = LiveState(logger=self._logger, backend=backend, env=env)
        self._transport_sender: Optional[Sender[TransportMessage[T]]] = None
        self._pattern_sender: Optional[Sender[PatternMessage[T]]] = None

    def start(self, system) -> None:
        """Start the live pattern system.

        Args:
            system: The actor system to use.
        """
        self._logger.info("Starting live pattern system")

        # Create the pattern state actor
        pattern_actor = PatternStateActor(self._state)
        pattern_sender = system.spawn_actor("pattern_state", pattern_actor)
        self._pattern_sender = pattern_sender

        # Create the transport actor for handling control messages
        transport_actor = TransportActor(self._state.domain)
        transport_sender = system.spawn_actor("transport", transport_actor)
        self._transport_sender = transport_sender

        # Create and spawn the timer task for timing loop
        timer_task = TimerTask(
            pattern_sender, transport_sender, self._state.env, self._state.domain
        )
        system.spawn_task("timer_loop", timer_task)

        self._logger.info("Live pattern system started")

    def set_orbit(self, orbit: Orbit, stream: Optional[Stream[T]]) -> None:
        """Set the stream for a specific orbit.

        Args:
            orbit: The orbit identifier.
            stream: The stream to set, or None to clear.
        """
        if self._pattern_sender is not None:
            self._pattern_sender.send(SetOrbit(orbit, stream))

    def set_cps(self, cps: Fraction) -> None:
        """Set the cycles per second (tempo).

        Args:
            cps: The new tempo in cycles per second.
        """
        if self._transport_sender is not None:
            self._transport_sender.send(SetCps(cps))

    def start_playback(self) -> None:
        """Start pattern playback."""
        if self._transport_sender is not None:
            self._transport_sender.send(SetPlaying(True))

    def stop_playback(self) -> None:
        """Stop pattern playback."""
        if self._transport_sender is not None:
            self._transport_sender.send(SetPlaying(False))

    def mute_orbit(self, orbit: Orbit, muted: bool = True) -> None:
        """Mute or unmute an orbit.

        Args:
            orbit: The orbit identifier.
            muted: Whether to mute the orbit.
        """
        if self._pattern_sender is not None:
            self._pattern_sender.send(MuteOrbit(orbit, muted))

    def unmute_orbit(self, orbit: Orbit) -> None:
        """Unmute an orbit.

        Args:
            orbit: The orbit identifier.
        """
        self.mute_orbit(orbit, False)

    def solo_orbit(self, orbit: Orbit, solo: bool = True) -> None:
        """Solo or unsolo an orbit.

        Args:
            orbit: The orbit identifier.
            solo: Whether to solo the orbit.
        """
        if self._pattern_sender is not None:
            self._pattern_sender.send(SoloOrbit(orbit, solo))

    def unsolo_orbit(self, orbit: Orbit) -> None:
        """Unsolo an orbit.

        Args:
            orbit: The orbit identifier.
        """
        self.solo_orbit(orbit, False)

    def panic(self) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        if self._transport_sender is not None:
            self._transport_sender.send(Panic())
        if self._pattern_sender is not None:
            # Clear all patterns - we'll need a new message for this or handle it differently
            pass


def create_live_system[T](
    backend: Optional[Backend[T]] = None,
    env: Optional[LiveEnv] = None,
    debug: bool = False,
) -> LiveSystem[T]:
    """Create a new live pattern system.

    Args:
        backend: Backend for processing events. If None, uses LogBackend.
        env: Environment configuration. If None, uses default.
        debug: Whether to enable debug mode.

    Returns:
        A new LiveSystem instance.
    """
    if env is None:
        env = LiveEnv(debug=debug)

    if backend is None:
        logger = logging.getLogger("minipat.live.backend")
        backend = LogBackend(logger)

    return LiveSystem(backend, env)


# Example usage and testing
if __name__ == "__main__":
    import logging

    from centipede.actor import new_system
    from minipat.pat import Pat

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Create pattern system
    system = new_system("live_test")
    live: LiveSystem[str] = create_live_system(debug=True)
    live.start(system)

    try:
        # Create some simple patterns and convert to streams
        pattern1 = Pat.pure("kick")
        pattern2 = Pat.seq([Pat.pure("snare"), Pat.silence()])
        stream1 = pat_stream(pattern1)
        stream2 = pat_stream(pattern2)

        # Set streams on orbits
        live.set_orbit(Orbit(0), stream1)
        live.set_orbit(Orbit(1), stream2)

        # Set tempo and start playback
        live.set_cps(Fraction(2))  # 2 cycles per second
        live.start_playback()

        # Let it play for a few seconds
        time.sleep(3)

        # Test muting
        live.mute_orbit(Orbit(1))
        time.sleep(2)

        # Test soloing
        live.solo_orbit(Orbit(0))
        time.sleep(2)

        # Panic stop
        live.panic()
        time.sleep(1)

    finally:
        # Clean shutdown
        system.stop()
        system.wait()
