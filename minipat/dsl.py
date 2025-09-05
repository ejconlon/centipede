from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Optional

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
    def rand(*choices: Flow) -> Flow:
        """Create a random choice flow."""
        streams = PSeq.mk(choice.stream for choice in choices)
        return Flow(Stream.rand(streams))

    @staticmethod
    def euc(flow: Flow, hits: int, steps: int, rotation: int = 0) -> Flow:
        """Create a Euclidean rhythm flow."""
        return Flow(Stream.euc(flow.stream, hits, steps, rotation))

    @staticmethod
    def poly(patterns: Iterable[Flow], subdiv: Optional[int] = None) -> Flow:
        """Create a polymetric flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.poly(streams, subdiv))

    def _repetition(self, operator: SpeedOp, count: Fraction) -> Flow:
        """Create a repetition flow."""
        return Flow(Stream.speed(self.stream, operator, count))

    def fast(self, count: Fraction) -> Flow:
        return self._repetition(SpeedOp.Fast, count)

    def slow(self, count: Fraction) -> Flow:
        return self._repetition(SpeedOp.Slow, count)

    def stretch(self, count: int) -> Flow:
        """Create an elongated flow."""
        return Flow(Stream.stretch(self.stream, count))

    def prob(self, chance: Fraction) -> Flow:
        """Create a probabilistic flow."""
        return Flow(Stream.prob(self.stream, chance))

    @staticmethod
    def alt(patterns: Iterable[Flow]) -> Flow:
        """Create an alternating flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.alt(streams))

    def repeat(self, count: int) -> Flow:
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

    def fast_by(self, factor: Fraction) -> Flow:
        """Speed up flow events by a given factor."""
        return Flow(self.stream.fast_by(factor))

    def slow_by(self, factor: Fraction) -> Flow:
        """Slow down flow events by a given factor."""
        return Flow(self.stream.slow_by(factor))

    def early_by(self, delta: CycleDelta) -> Flow:
        """Shift flow events earlier in time."""
        return Flow(self.stream.early_by(delta))

    def late_by(self, delta: CycleDelta) -> Flow:
        """Shift flow events later in time."""
        return Flow(self.stream.late_by(delta))

    # def __or__(self, other: Flow) -> Flow:
    #     """Operator overload for parallel combination."""
    #     return Flow.par(PSeq.mk([self, other]))
    #
    # def __and__(self, other: Flow) -> Flow:
    #     """Operator overload for sequential combination."""
    #     return Flow.seq(PSeq.mk([self, other]))
