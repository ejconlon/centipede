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
    def silence() -> Flow:
        """Create a silent flow."""
        return Flow(Stream.silence())

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
    def choice(*choices: Flow) -> Flow:
        """Create a choice flow."""
        streams = PSeq.mk(choice.stream for choice in choices)
        return Flow(Stream.choice(streams))

    @staticmethod
    def euc(flow: Flow, hits: int, steps: int, rotation: int = 0) -> Flow:
        """Create a Euclidean rhythm flow."""
        return Flow(Stream.euclidean(flow.stream, hits, steps, rotation))

    @staticmethod
    def poly(*patterns: Flow) -> Flow:
        """Create a polymetric flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.polymetric(streams, None))

    @staticmethod
    def poly_sub(patterns: Iterable[Flow], subdiv: int) -> Flow:
        """Create a polymetric flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.polymetric(streams, subdiv))

    def _repetition(self, operator: SpeedOp, count: int) -> Flow:
        """Create a repetition flow."""
        return Flow(Stream.repetition(self.stream, operator, count))

    def fast(self, count: int) -> Flow:
        return self._repetition(SpeedOp.Fast, count)

    def slow(self, count: int) -> Flow:
        return self._repetition(SpeedOp.Slow, count)

    def stretch(self, count: int) -> Flow:
        """Create an elongated flow."""
        return Flow(Stream.elongation(self.stream, count))

    def degrade(self, prob: Fraction) -> Flow:
        """Create a probabilistic flow."""
        return Flow(Stream.probability(self.stream, prob))

    @staticmethod
    def alt(patterns: Iterable[Flow]) -> Flow:
        """Create an alternating flow."""
        streams = PSeq.mk(pattern.stream for pattern in patterns)
        return Flow(Stream.alternating(streams))

    def rep(self, count: int) -> Flow:
        """Create a replicate flow."""
        return Flow(Stream.replicate(self.stream, count))

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
