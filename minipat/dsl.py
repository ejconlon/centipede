from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, NoReturn, Optional

from bad_actor import System, new_system
from minipat.common import CycleDelta, Numeric, numeric_frac
from minipat.live import LiveSystem, Orbit
from minipat.midi import (
    MidiAttrs,
    TimedMessage,
    channel_stream,
    combine_all,
    control_stream,
    midinote_stream,
    note_stream,
    program_stream,
    start_midi_live_system,
    value_stream,
    vel_stream,
)
from minipat.pat import Pat, SpeedOp
from minipat.stream import MergeStrat, Stream
from spiny import PSeq


@dataclass(frozen=True, eq=False)
class Flow:
    stream: Stream[MidiAttrs]

    @staticmethod
    def silent() -> Flow:
        """Create a silent flow."""
        return Flow(Stream.silent())

    @staticmethod
    def pure(val: MidiAttrs) -> Flow:
        """Create a flow with a single value."""
        return Flow(Stream.pure(val))

    @staticmethod
    def seqs(*flows: Flow) -> Flow:
        """Create a sequential flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.seq(streams))

    @staticmethod
    def pars(*flows: Flow) -> Flow:
        """Create a parallel flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.par(streams))

    @staticmethod
    def rands(*flows: Flow) -> Flow:
        """Create a random choice flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.rand(streams))

    @staticmethod
    def alts(*flows: Flow) -> Flow:
        """Create an alternating flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.alt(streams))

    @staticmethod
    def polys(*flows: Flow) -> Flow:
        """Create a polymetric flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.poly(streams, None))

    @staticmethod
    def polysubs(subdiv: int, *flows: Flow) -> Flow:
        """Create a polymetric flow with subdivision."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.poly(streams, subdiv))

    @staticmethod
    def combines(*flows: Flow) -> Flow:
        """Combine all flows"""
        streams = [flow.stream for flow in flows]
        return Flow(combine_all(streams))

    @staticmethod
    def pat(pattern: Pat[MidiAttrs]) -> Flow:
        """Create a flow from a pattern."""
        return Flow(Stream.pat(pattern))

    def euc(self, hits: int, steps: int, rotation: int = 0) -> Flow:
        """Create a Euclidean rhythm flow."""
        return Flow(Stream.euc(self.stream, hits, steps, rotation))

    def _speed(self, operator: SpeedOp, factor: Fraction) -> Flow:
        return Flow(Stream.speed(self.stream, operator, factor))

    def fast(self, factor: Numeric) -> Flow:
        """Speed events up by a factor"""
        return self._speed(SpeedOp.Fast, numeric_frac(factor))

    def slow(self, factor: Numeric) -> Flow:
        """Slow events down by a factor"""
        return self._speed(SpeedOp.Slow, numeric_frac(factor))

    def stretch(self, count: Numeric) -> Flow:
        """Create a stretched flow."""
        return Flow(Stream.stretch(self.stream, numeric_frac(count)))

    def prob(self, chance: Numeric) -> Flow:
        """Create a probabilistic flow."""
        return Flow(Stream.prob(self.stream, numeric_frac(chance)))

    def repeat(self, count: Numeric) -> Flow:
        """Create a repeat flow."""
        return Flow(Stream.repeat(self.stream, numeric_frac(count)))

    def map(self, func: Callable[[MidiAttrs], MidiAttrs]) -> Flow:
        """Map a function over the flow values."""
        return Flow(self.stream.map(func))

    def filter(self, predicate: Callable[[MidiAttrs], bool]) -> Flow:
        """Filter events in a flow based on a predicate."""
        return Flow(self.stream.filter(predicate))

    def bind(self, merge_strat: MergeStrat, func: Callable[[MidiAttrs], Flow]) -> Flow:
        """Bind a flow with a merge strategy."""

        def stream_func(x: MidiAttrs) -> Stream[MidiAttrs]:
            return func(x).stream

        return Flow(self.stream.bind(merge_strat, stream_func))

    def apply(
        self,
        merge_strat: MergeStrat,
        func: Callable[[MidiAttrs, MidiAttrs], MidiAttrs],
        other: Flow,
    ) -> Flow:
        """Apply a function across two flows."""
        return Flow(self.stream.apply(merge_strat, func, other.stream))

    def shift(self, delta: CycleDelta) -> Flow:
        """Shift flow events in time by a delta."""
        return Flow(self.stream.shift(delta))

    def early(self, delta: CycleDelta) -> Flow:
        """Shift flow events earlier in time."""
        return self.shift(CycleDelta(-delta))

    def late(self, delta: CycleDelta) -> Flow:
        """Shift flow events later in time."""
        return self.shift(delta)

    def par(self, other: Flow) -> Flow:
        """Combine this flow with another in parallel."""
        return Flow.pars(self, other)

    def seq(self, other: Flow) -> Flow:
        """Combine this flow with another sequentially."""
        return Flow.seqs(self, other)

    def combine(self, other: Flow) -> Flow:
        """Combine this flow with another by merging attributes."""
        return Flow.combines(self, other)

    def __or__(self, other: Flow) -> Flow:
        """Operator overload for parallel combination."""
        return self.par(other)

    def __and__(self, other: Flow) -> Flow:
        """Operator overload for sequential combination."""
        return self.seq(other)

    def __rshift__(self, other: Flow) -> Flow:
        """Operator overload for combining flows."""
        return self.combine(other)

    def __lshift__(self, other: Flow) -> Flow:
        """Operator overload for combining flows (flipped)."""
        return other.combine(self)

    def __mul__(self, factor: Numeric) -> Flow:
        """Operator overload for fast (speed up by factor)."""
        return self.fast(factor)

    def __truediv__(self, factor: Numeric) -> Flow:
        """Operator overload for slow (slow down by factor)."""
        return self.slow(factor)

    def __xor__(self, other: Flow) -> Flow:
        """Operator overload for alternating flows."""
        return Flow.alts(self, other)

    def __pow__(self, count: Numeric) -> Flow:
        """Operator overload for repeating flow."""
        return self.repeat(count)


def note(pat_str: str) -> Flow:
    """Create a flow from note names.

    Alias for Flow.note() that creates a Flow from musical note names.

    Args:
        pat_str: Pattern string containing musical note names with octaves

    Returns:
        A Flow containing MIDI note attributes

    Examples:
        note("c4 d4 e4")         # C major scale fragment
        note("c4 ~ g4")          # C4, rest, G4
        note("[c4,e4,g4]")       # C major chord (simultaneous)
    """
    return Flow(note_stream(pat_str))


def midinote(pat_str: str) -> Flow:
    """Create a flow from numeric MIDI notes.

    Alias for Flow.midinote() that creates a Flow from numeric MIDI note values.

    Args:
        pat_str: Pattern string containing numeric MIDI note values (0-127)

    Returns:
        A Flow containing MIDI note attributes

    Examples:
        midinote("60 62 64")     # C4, D4, E4 (C major triad)
        midinote("36 ~ 42")      # Kick, rest, snare pattern
        midinote("[60,64,67]")   # C major chord (simultaneous)
    """
    return Flow(midinote_stream(pat_str))


def vel(pat_str: str) -> Flow:
    """Create a flow from velocity values.

    Alias for Flow.vel() that creates a Flow from MIDI velocity values.

    Args:
        pat_str: Pattern string containing MIDI velocity values (0-127)

    Returns:
        A Flow containing MIDI velocity attributes

    Examples:
        vel("64 80 100")         # Medium, loud, very loud
        vel("127 0 64")          # Loud, silent, medium
        vel("100*8")             # Repeat loud velocity 8 times
    """
    return Flow(vel_stream(pat_str))


def program(pat_str: str) -> Flow:
    """Create a flow from program values.

    Alias for Flow.program() that creates a Flow from MIDI program values.

    Args:
        pat_str: Pattern string containing MIDI program values (0-127)

    Returns:
        A Flow containing MIDI program attributes

    Examples:
        program("0 1 40")         # Piano, Bright Piano, Violin
        program("128 ~ 0")        # Invalid program, rest, Piano (will error on 128)
        program("1*4")            # Repeat Bright Piano 4 times
    """
    return Flow(program_stream(pat_str))


def control(pat_str: str) -> Flow:
    """Create a flow from control number values.

    Alias for Flow.control() that creates a Flow from MIDI control numbers.

    Args:
        pat_str: Pattern string containing MIDI control numbers (0-127)

    Returns:
        A Flow containing MIDI control number attributes

    Examples:
        control("1 7 10")         # Modulation, Volume, Pan
        control("64 ~ 1")         # Sustain, rest, Modulation
        control("7*8")            # Repeat Volume control 8 times
    """
    return Flow(control_stream(pat_str))


def value(pat_str: str) -> Flow:
    """Create a flow from control value values.

    Alias for Flow.value() that creates a Flow from MIDI control values.

    Args:
        pat_str: Pattern string containing MIDI control values (0-127)

    Returns:
        A Flow containing MIDI control value attributes

    Examples:
        value("0 64 127")         # Min, center, max values
        value("127 ~ 0")          # Max, rest, min
        value("64*8")             # Repeat center value 8 times
    """
    return Flow(value_stream(pat_str))


def channel(pat_str: str) -> Flow:
    """Create a flow from channel values.

    Creates a Flow from MIDI channel values. If channel is not specified
    in patterns, the orbit number will be used as the default channel.

    Args:
        pat_str: Pattern string containing MIDI channel values (0-15)

    Returns:
        A Flow containing MIDI channel attributes

    Examples:
        channel("0 1 9")          # Channels 1, 2, 10 (drums)
        channel("15 ~ 0")         # Channel 16, rest, Channel 1
        channel("9*4")            # Repeat Channel 10 (drums) 4 times
    """
    return Flow(channel_stream(pat_str))


@dataclass(frozen=True, eq=False)
class Nucleus:
    sys: System
    live: LiveSystem[MidiAttrs, TimedMessage]

    @staticmethod
    def boot(
        port_name: Optional[str] = None,
        sys_name: Optional[str] = None,
        init_bpm: Optional[int] = None,
        init_bpc: Optional[int] = None,
        log_path: Optional[str] = None,
        log_level: Optional[str] = None,
    ) -> Nucleus:
        log_path = log_path or os.environ.get("MINIPAT_LOG_PATH", "/tmp/minipat.log")
        assert log_path is not None
        log_level = log_level or os.environ.get("MINIPAT_LOG_LEVEL", "INFO")
        assert log_level is not None
        port_name = port_name or os.environ.get("MINIPAT_PORT", "minipat")
        assert port_name is not None
        sys_name = sys_name or os.environ.get("MINIPAT_SYS", "system")
        assert sys_name is not None
        init_bpm = init_bpm or int(os.environ.get("MINIPAT_BPM", "120"))
        assert init_bpm is not None
        init_bpc = init_bpc or int(os.environ.get("MINIPAT_BPC", "4"))
        assert init_bpc is not None
        init_cps = Fraction(init_bpm, init_bpc * 60)
        logging.basicConfig(
            filename=log_path, filemode="w", level=getattr(logging, log_level)
        )
        sys = new_system(sys_name)
        live = start_midi_live_system(sys, port_name, init_cps)
        return Nucleus(sys, live)

    def stop(self) -> int:
        self.live.panic()
        self.sys.stop()
        saved_excs = self.sys.wait()
        if len(saved_excs) > 0:
            sys.stderr.write(f"System exceptions ({len(saved_excs)})\n")
            for exc in saved_excs:
                sys.stderr.write(f"{exc}\n")
        return len(saved_excs)

    def exit(self) -> NoReturn:
        ret = self.stop()
        sys.exit(ret)

    def play(self) -> None:
        self.live.play()

    def pause(self) -> None:
        self.live.pause()

    def panic(self) -> None:
        self.live.panic()

    def clear(self) -> None:
        self.live.clear_orbits()

    def running(self) -> bool:
        """Check if the actor system is currently running.

        Returns:
            True if the system is running, False if it has stopped or is stopping.
        """
        return self.sys.running()

    def playing(self) -> bool:
        """Check if the pattern system is currently playing.

        Returns:
            True if playback is active, False if paused or stopped.
        """
        return self.live.playing()

    def set_cps(self, cps: Numeric) -> None:
        self.live.set_cps(numeric_frac(cps))

    def once(
        self,
        flow: Flow,
        length: Optional[CycleDelta] = None,
        aligned: Optional[bool] = None,
    ) -> None:
        self.live.once(flow.stream, length=length, aligned=aligned, orbit=None)

    def orbital(self, num: int) -> Orbital:
        return Orbital(self, Orbit(num))

    def __getitem__(self, num: int) -> Orbital:
        return self.orbital(num)

    def __setitem__(self, num: int, flow: Optional[Flow]) -> None:
        o = self.orbital(num)
        if flow is None:
            o.clear()
        else:
            o.every(flow)

    def __delitem__(self, num: int) -> None:
        return self.orbital(num).clear()

    def __or__(self, flow: Flow) -> None:
        self.once(flow)


@dataclass(frozen=True, eq=False)
class Orbital:
    nucleus: Nucleus
    num: Orbit

    def once(
        self,
        flow: Flow,
        length: Optional[CycleDelta] = None,
        aligned: Optional[bool] = None,
    ) -> None:
        self.nucleus.live.once(
            flow.stream, length=length, aligned=aligned, orbit=self.num
        )

    def every(self, flow: Flow) -> None:
        self.nucleus.live.set_orbit(self.num, flow.stream)

    def solo(self) -> None:
        self.nucleus.live.solo(self.num)

    def unsolo(self) -> None:
        self.nucleus.live.unsolo(self.num)

    def mute(self) -> None:
        self.nucleus.live.mute(self.num)

    def unmute(self) -> None:
        self.nucleus.live.unmute(self.num)

    def clear(self) -> None:
        self.nucleus.live.set_orbit(self.num, None)

    def __matmul__(self, flow: Flow) -> None:
        self.once(flow)

    def __or__(self, flow: Flow) -> None:
        self.once(flow)
