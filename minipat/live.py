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
from typing import Dict, Optional, override

from centipede.actor import Actor, ActorEnv, Sender, Task
from minipat.arc import Arc
from minipat.common import ONE, ZERO
from minipat.ev import Ev
from minipat.pat import Pat
from minipat.stream import pat_stream
from spiny.heapmap import PHeapMap


@dataclass(frozen=True)
class LiveEnv:
    """Environment configuration for live pattern playback.

    Defines global settings that control pattern playback behavior.
    """

    debug: bool = False
    """Enable debug logging for pattern playback."""

    cycles_per_second: Fraction = ONE
    """Tempo in cycles per second (CPS). Default is 1 CPS."""

    generations_per_cycle: int = 4
    """Number of event generations to calculate per cycle."""


@dataclass
class OrbitState[T]:
    """State for a single orbit (audio channel/stream).

    Each orbit contains a pattern and maintains its own playback state.
    """

    pattern: Optional[Pat[T]] = None
    """Current pattern playing on this orbit."""

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

    current_cycle: Fraction = ZERO
    """Current cycle position in the timeline."""

    orbits: Dict[int, OrbitState[T]] = field(default_factory=dict)
    """Map of orbit ID to orbit state."""

    cps: Fraction = ONE
    """Current cycles per second (tempo)."""

    generations_per_cycle: int = 4
    """Current number of generations per cycle."""


class Backend[T](metaclass=ABCMeta):
    """Abstract interface for pattern event backends.

    Backends are responsible for processing and outputting pattern events
    (e.g., to audio systems, MIDI, OSC, etc.).
    """

    @abstractmethod
    def send_events(self, orbit_id: int, events: PHeapMap[Arc, Ev[T]]) -> None:
        """Send events from an orbit to the backend.

        Args:
            orbit_id: The orbit identifier.
            events: The events to send.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending events to the output."""
        ...


class LogBackend[T](Backend[T]):
    """Debug backend that logs events instead of playing them."""

    def __init__(self, logger: Logger):
        self._logger = logger

    @override
    def send_events(self, orbit_id: int, events: PHeapMap[Arc, Ev[T]]) -> None:
        if not events:
            self._logger.debug(f"Orbit {orbit_id}: {len(events)} events")
            for arc, ev in events:
                self._logger.debug(f"  {arc}: {ev}")

    @override
    def flush(self) -> None:
        pass


@dataclass
class LiveState[T]:
    """Complete state for the live pattern system.

    Contains all configuration, mutable state, and backend references.
    """

    logger: Logger
    backend: Backend[T]
    env: LiveEnv
    domain: LiveDomain[T] = field(default_factory=LiveDomain)


class GeneratorTask[T](Task):
    """Task that generates pattern events at regular intervals."""

    def __init__(self, state_sender: Sender[GeneratorMessage[T]], interval: float):
        self._state_sender = state_sender
        self._interval = interval

    @override
    def run(self, logger: Logger, halt: Event) -> None:
        logger.debug("Generator task starting")

        while not halt.is_set():
            # Send generation request
            self._state_sender.send(GenerateEvents())

            # Wait for next interval or halt
            if halt.wait(timeout=self._interval):
                break

        logger.debug("Generator task stopping")


class GeneratorMessage[T](metaclass=ABCMeta):
    """Base class for messages sent to the pattern state actor."""

    pass


@dataclass(frozen=True)
class GenerateEvents[T](GeneratorMessage[T]):
    """Request to generate events for the next cycle."""

    pass


@dataclass(frozen=True)
class SetOrbit[T](GeneratorMessage[T]):
    """Set the pattern for a specific orbit."""

    orbit_id: int
    pattern: Optional[Pat[T]]


@dataclass(frozen=True)
class SetCps[T](GeneratorMessage[T]):
    """Set the cycles per second (tempo)."""

    cps: Fraction


@dataclass(frozen=True)
class SetPlaying[T](GeneratorMessage[T]):
    """Set the playing state."""

    playing: bool


@dataclass(frozen=True)
class MuteOrbit[T](GeneratorMessage[T]):
    """Mute or unmute an orbit."""

    orbit_id: int
    muted: bool


@dataclass(frozen=True)
class SoloOrbit[T](GeneratorMessage[T]):
    """Solo or unsolo an orbit."""

    orbit_id: int
    solo: bool


@dataclass(frozen=True)
class Panic[T](GeneratorMessage[T]):
    """Emergency stop - clear all patterns and stop playback."""

    pass


class PatternStateActor[T](Actor[GeneratorMessage[T]]):
    """Actor that manages live pattern state and generates events.

    This is the core of the live pattern system, handling state updates
    and coordinating event generation with the backend.
    """

    def __init__(self, initial_state: LiveState[T]):
        self._state = initial_state

    @override
    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Pattern state actor started")

    @override
    def on_message(self, env: ActorEnv, value: GeneratorMessage[T]) -> None:
        match value:
            case GenerateEvents():
                self._generate_events(env.logger)
            case SetOrbit(orbit_id, pattern):
                self._set_orbit(env.logger, orbit_id, pattern)
            case SetCps(cps):
                self._set_cps(env.logger, cps)
            case SetPlaying(playing):
                self._set_playing(env.logger, playing)
            case MuteOrbit(orbit_id, muted):
                self._mute_orbit(env.logger, orbit_id, muted)
            case SoloOrbit(orbit_id, solo):
                self._solo_orbit(env.logger, orbit_id, solo)
            case Panic():
                self._panic(env.logger)

    def _generate_events(self, logger: Logger) -> None:
        """Generate events for the current cycle."""
        if not self._state.domain.playing:
            return

        logger.debug(f"Generating events for cycle {self._state.domain.current_cycle}")

        # Calculate the time arc for this generation
        cycle_start = self._state.domain.current_cycle
        cycle_length = Fraction(1) / self._state.domain.generations_per_cycle
        cycle_end = cycle_start + cycle_length
        arc = Arc(cycle_start, cycle_end)

        # Check if any orbits are soloed
        has_solo = any(orbit.solo for orbit in self._state.domain.orbits.values())

        # Generate events for each active orbit
        for orbit_id, orbit in self._state.domain.orbits.items():
            if orbit.pattern is None:
                continue

            # Skip muted orbits, unless they're soloed
            if orbit.muted and not orbit.solo:
                continue

            # If there are soloed orbits, only play soloed ones
            if has_solo and not orbit.solo:
                continue

            # Generate events using pattern stream
            stream = pat_stream(orbit.pattern)
            events = stream.unstream(arc)

            # Send events to backend
            if not events:
                logger.debug(f"Sending {len(events)} events for orbit {orbit_id}")
                self._state.backend.send_events(orbit_id, events)

        # Flush backend
        self._state.backend.flush()

        # Advance cycle position
        self._state.domain.current_cycle = cycle_end

    def _set_orbit(
        self, logger: Logger, orbit_id: int, pattern: Optional[Pat[T]]
    ) -> None:
        """Set the pattern for a specific orbit."""
        if orbit_id not in self._state.domain.orbits:
            self._state.domain.orbits[orbit_id] = OrbitState()

        self._state.domain.orbits[orbit_id].pattern = pattern

        if pattern is None:
            logger.debug(f"Cleared orbit {orbit_id}")
        else:
            logger.debug(f"Set pattern for orbit {orbit_id}")

    def _set_cps(self, logger: Logger, cps: Fraction) -> None:
        """Set the cycles per second (tempo)."""
        self._state.domain.cps = cps
        logger.debug(f"Set CPS to {cps}")

    def _set_playing(self, logger: Logger, playing: bool) -> None:
        """Set the playing state."""
        self._state.domain.playing = playing
        logger.debug(f"Set playing to {playing}")

        if playing:
            # Reset cycle position when starting
            self._state.domain.current_cycle = Fraction(0)

    def _mute_orbit(self, logger: Logger, orbit_id: int, muted: bool) -> None:
        """Mute or unmute an orbit."""
        if orbit_id not in self._state.domain.orbits:
            self._state.domain.orbits[orbit_id] = OrbitState()

        self._state.domain.orbits[orbit_id].muted = muted
        logger.debug(f"Set orbit {orbit_id} muted to {muted}")

    def _solo_orbit(self, logger: Logger, orbit_id: int, solo: bool) -> None:
        """Solo or unsolo an orbit."""
        if orbit_id not in self._state.domain.orbits:
            self._state.domain.orbits[orbit_id] = OrbitState()

        self._state.domain.orbits[orbit_id].solo = solo
        logger.debug(f"Set orbit {orbit_id} solo to {solo}")

    def _panic(self, logger: Logger) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        logger.info("PANIC: Clearing all patterns and stopping playback")

        # Stop playback
        self._state.domain.playing = False

        # Clear all patterns
        for orbit in self._state.domain.orbits.values():
            orbit.pattern = None
            orbit.muted = False
            orbit.solo = False

        # Reset cycle position
        self._state.domain.current_cycle = Fraction(0)


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
        self._state_sender: Optional[Sender[GeneratorMessage[T]]] = None

        # Calculate generator interval from CPS and generations
        self._interval = float(
            1.0 / (env.cycles_per_second * env.generations_per_cycle)
        )

    def start(self, system) -> None:
        """Start the live pattern system.

        Args:
            system: The actor system to use.
        """
        self._logger.info("Starting live pattern system")

        # Create the pattern state actor
        state_actor = PatternStateActor(self._state)
        state_sender = system.spawn_actor("pattern_state", state_actor)
        self._state_sender = state_sender

        # Create and spawn the generator task
        generator_task = GeneratorTask(state_sender, self._interval)
        system.spawn_task("generator", generator_task)

        self._logger.info("Live pattern system started")

    def set_orbit(self, orbit_id: int, pattern: Optional[Pat[T]]) -> None:
        """Set the pattern for a specific orbit.

        Args:
            orbit_id: The orbit identifier.
            pattern: The pattern to set, or None to clear.
        """
        if self._state_sender is not None:
            self._state_sender.send(SetOrbit(orbit_id, pattern))

    def set_cps(self, cps: Fraction) -> None:
        """Set the cycles per second (tempo).

        Args:
            cps: The new tempo in cycles per second.
        """
        if self._state_sender is not None:
            self._state_sender.send(SetCps(cps))

    def start_playback(self) -> None:
        """Start pattern playback."""
        if self._state_sender is not None:
            self._state_sender.send(SetPlaying(True))

    def stop_playback(self) -> None:
        """Stop pattern playback."""
        if self._state_sender is not None:
            self._state_sender.send(SetPlaying(False))

    def mute_orbit(self, orbit_id: int, muted: bool = True) -> None:
        """Mute or unmute an orbit.

        Args:
            orbit_id: The orbit identifier.
            muted: Whether to mute the orbit.
        """
        if self._state_sender is not None:
            self._state_sender.send(MuteOrbit(orbit_id, muted))

    def unmute_orbit(self, orbit_id: int) -> None:
        """Unmute an orbit.

        Args:
            orbit_id: The orbit identifier.
        """
        self.mute_orbit(orbit_id, False)

    def solo_orbit(self, orbit_id: int, solo: bool = True) -> None:
        """Solo or unsolo an orbit.

        Args:
            orbit_id: The orbit identifier.
            solo: Whether to solo the orbit.
        """
        if self._state_sender is not None:
            self._state_sender.send(SoloOrbit(orbit_id, solo))

    def unsolo_orbit(self, orbit_id: int) -> None:
        """Unsolo an orbit.

        Args:
            orbit_id: The orbit identifier.
        """
        self.solo_orbit(orbit_id, False)

    def panic(self) -> None:
        """Emergency stop - clear all patterns and stop playback."""
        if self._state_sender is not None:
            self._state_sender.send(Panic())


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
        # Create some simple patterns
        pattern1 = Pat.pure("kick")
        pattern2 = Pat.seq([Pat.pure("snare"), Pat.silence()])

        # Set patterns on orbits
        live.set_orbit(0, pattern1)
        live.set_orbit(1, pattern2)

        # Set tempo and start playback
        live.set_cps(Fraction(2))  # 2 cycles per second
        live.start_playback()

        # Let it play for a few seconds
        time.sleep(3)

        # Test muting
        live.mute_orbit(1)
        time.sleep(2)

        # Test soloing
        live.solo_orbit(0)
        time.sleep(2)

        # Panic stop
        live.panic()
        time.sleep(1)

    finally:
        # Clean shutdown
        system.stop()
        system.wait()
