from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable

from minipat.common import CycleDelta
from minipat.midi import MidiAttrs, midinote, note, vel
from minipat.pat import Pat, SpeedOp
from minipat.stream import MergeStrat, Stream
from spiny import PSeq


@dataclass(frozen=True, eq=False)
class Flow:
    stream: Stream[MidiAttrs]

    # Factory methods (static)
    @staticmethod
    def silent() -> Flow:
        """Create a silent flow."""
        return Flow(Stream.silent())

    @staticmethod
    def pure(val: MidiAttrs) -> Flow:
        """Create a flow with a single value."""
        return Flow(Stream.pure(val))

    @staticmethod
    def seq(*flows: Flow) -> Flow:
        """Create a sequential flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.seq(streams))

    @staticmethod
    def par(*flows: Flow) -> Flow:
        """Create a parallel flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.par(streams))

    @staticmethod
    def rand(*flows: Flow) -> Flow:
        """Create a random choice flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.rand(streams))

    @staticmethod
    def alt(patterns: Iterable[Flow]) -> Flow:
        """Create an alternating flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.alt(streams))

    @staticmethod
    def poly(*flows: Flow) -> Flow:
        """Create a polymetric flow."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.poly(streams, None))

    @staticmethod
    def polysub(subdiv: int, *flows: Flow) -> Flow:
        """Create a polymetric flow with subdivision."""
        streams = PSeq.mk(flow.stream for flow in flows)
        return Flow(Stream.poly(streams, subdiv))

    def euc(self, hits: int, steps: int, rotation: int = 0) -> Flow:
        """Create a Euclidean rhythm flow."""
        return Flow(Stream.euc(self.stream, hits, steps, rotation))

    def _speed(self, operator: SpeedOp, factor: Fraction) -> Flow:
        return Flow(Stream.speed(self.stream, operator, factor))

    def fast(self, factor: Fraction) -> Flow:
        """Speed events up by a factor"""
        return self._speed(SpeedOp.Fast, factor)

    def slow(self, factor: Fraction) -> Flow:
        """Slow events down by a factor"""
        return self._speed(SpeedOp.Slow, factor)

    def stretch(self, count: int) -> Flow:
        """Create an elongated flow."""
        return Flow(Stream.stretch(self.stream, count))

    def prob(self, chance: Fraction) -> Flow:
        """Create a probabilistic flow."""
        return Flow(Stream.prob(self.stream, chance))

    def repeat(self, count: Fraction) -> Flow:
        """Create a repeat flow."""
        return Flow(Stream.repeat(self.stream, count))

    @staticmethod
    def pat(pattern: Pat[MidiAttrs]) -> Flow:
        """Create a flow from a pattern."""
        return Flow(Stream.pat(pattern))

    @staticmethod
    def note(pat_str: str) -> Flow:
        return Flow(note(pat_str))

    @staticmethod
    def midinote(pat_str: str) -> Flow:
        return Flow(midinote(pat_str))

    @staticmethod
    def vel(pat_str: str) -> Flow:
        return Flow(vel(pat_str))

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

    # def __or__(self, other: Flow) -> Flow:
    #     """Operator overload for parallel combination."""
    #     return Flow.par(PSeq.mk([self, other]))
    #
    # def __and__(self, other: Flow) -> Flow:
    #     """Operator overload for sequential combination."""
    #     return Flow.seq(PSeq.mk([self, other]))
