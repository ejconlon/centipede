from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Union

from minipat.common import CycleDelta
from minipat.midi import MidiAttrs, combine_all, midinote, note, vel
from minipat.pat import Pat, SpeedOp
from minipat.stream import MergeStrat, Stream
from spiny import PSeq

Numeric = Union[int, float, Fraction]


def numeric_frac(numeric: Numeric) -> Fraction:
    """Convert a numeric value to a Fraction."""
    if isinstance(numeric, Fraction):
        return numeric
    return Fraction(numeric)


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

    @staticmethod
    def note(pat_str: str) -> Flow:
        return Flow(note(pat_str))

    @staticmethod
    def midinote(pat_str: str) -> Flow:
        return Flow(midinote(pat_str))

    @staticmethod
    def vel(pat_str: str) -> Flow:
        return Flow(vel(pat_str))

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
