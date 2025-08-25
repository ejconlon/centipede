"""Pattern types and operations for the minipat pattern language."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Tuple

from centipede.common import PartialMatchException, ignore_arg
from spiny import Box, PSeq


class RepetitionOp(Enum):
    """Enumeration for repetition operators."""

    FAST = "*"  # Multiply/repeat
    """Fast repetition operator (*) - repeats pattern faster."""

    SLOW = "/"  # Divide/slow down
    """Slow repetition operator (/) - slows down pattern."""


# sealed
class PatF[T, R]:
    """Base class for pattern functors.

    This is a sealed class that serves as the base for all pattern types.
    """

    pass


@dataclass(frozen=True)
class Pat[T]:
    """Main pattern type that wraps a pattern functor.

    Args:
        unwrap: The underlying pattern functor
    """

    unwrap: PatF[T, Pat[T]]

    @staticmethod
    def silence() -> Pat[T]:
        """Create a silent pattern.

        Returns:
            A pattern representing silence
        """
        return _PAT_SILENCE

    @staticmethod
    def pure(val: T) -> Pat[T]:
        """Create a pattern with a single value.

        Args:
            val: The value to wrap

        Returns:
            A pattern containing the single value
        """
        return Pat(PatPure(val))

    @staticmethod
    def seq(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create a sequential pattern.

        Args:
            pats: The patterns to sequence

        Returns:
            A pattern that plays the given patterns in sequence
        """
        return Pat(PatSeq(PSeq.mk(pats)))

    @staticmethod
    def par(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create a parallel pattern.

        Args:
            pats: The patterns to play in parallel

        Returns:
            A pattern that plays the given patterns simultaneously
        """
        return Pat(PatPar(PSeq.mk(pats)))

    @staticmethod
    def choice(choices: Iterable[Pat[T]]) -> Pat[T]:
        """Create a choice pattern.

        Args:
            choices: The patterns to choose from

        Returns:
            A pattern that cycles through the given choices
        """
        return Pat(PatChoice(PSeq.mk(choices)))

    @staticmethod
    def euclidean(atom: Pat[T], hits: int, steps: int, rotation: int = 0) -> Pat[T]:
        """Create a Euclidean rhythm pattern.

        Args:
            atom: The pattern to distribute
            hits: Number of hits to distribute
            steps: Total number of steps
            rotation: Optional rotation offset

        Returns:
            A pattern with Euclidean rhythm distribution
        """
        return Pat(PatEuclidean(atom, hits, steps, rotation))

    @staticmethod
    def polymetric(patterns: Iterable[Pat[T]]) -> Pat[T]:
        """Create a polymetric pattern.

        Args:
            patterns: The patterns to play polymetrically

        Returns:
            A pattern that plays patterns simultaneously at different rates
        """
        return Pat(PatPolymetric(PSeq.mk(patterns)))

    @staticmethod
    def repetition(pattern: Pat[T], operator: RepetitionOp, count: int) -> Pat[T]:
        """Create a repetition pattern.

        Args:
            pattern: The pattern to repeat
            operator: The repetition operator (FAST or SLOW)
            count: The repetition count

        Returns:
            A pattern with the specified repetition
        """
        return Pat(PatRepetition(pattern, operator, count))

    @staticmethod
    def elongation(pattern: Pat[T], count: int) -> Pat[T]:
        """Create an elongated pattern.

        Args:
            pattern: The pattern to elongate
            count: The elongation count

        Returns:
            A pattern stretched by the given count
        """
        return Pat(PatElongation(pattern, count))

    @staticmethod
    def probability(pattern: Pat[T], prob: Fraction = Fraction(1, 2)) -> Pat[T]:
        """Create a probabilistic pattern.

        Args:
            pattern: The pattern to apply probability to
            prob: The probability (0 to 1 as a Fraction)

        Returns:
            A pattern that plays with the given probability
        """
        return Pat(PatProbability(pattern, prob))

    @staticmethod
    def select(pattern: Pat[T], selector: str) -> Pat[T]:
        """Create a sample selection pattern.

        Args:
            pattern: The pattern to select from
            selector: The selection identifier

        Returns:
            A pattern with sample selection applied
        """
        return Pat(PatSelect(pattern, selector))

    @staticmethod
    def alternating(patterns: Iterable[Pat[T]]) -> Pat[T]:
        """Create an alternating pattern.

        Args:
            patterns: The patterns to alternate between

        Returns:
            A pattern that alternates between the given patterns
        """
        return Pat(PatAlternating(PSeq.mk(patterns)))

    @staticmethod
    def replicate(pattern: Pat[T], count: int) -> Pat[T]:
        """Create a replicate pattern (!).

        Args:
            pattern: The pattern to replicate
            count: The number of times to replicate

        Returns:
            A replicated pattern
        """
        return Pat(PatReplicate(pattern, count))

    @staticmethod
    def ratio(pattern: Pat[T], numerator: int, denominator: int) -> Pat[T]:
        """Create a ratio pattern (%).

        Args:
            pattern: The pattern to apply ratio to
            numerator: The numerator of the ratio
            denominator: The denominator of the ratio

        Returns:
            A ratio pattern
        """
        return Pat(PatRatio(pattern, numerator, denominator))

    @staticmethod
    def polymetric_sub(patterns: Iterable[Pat[T]], subdivision: int) -> Pat[T]:
        """Create a polymetric pattern with subdivision ({}%).

        Args:
            patterns: The patterns to play polymetrically
            subdivision: The subdivision factor

        Returns:
            A polymetric subdivision pattern
        """
        return Pat(PatPolymetricSub(PSeq.mk(patterns), subdivision))

    def map[U](self, fn: Callable[[T], U]) -> Pat[U]:
        """Map a function over the pattern values.

        Args:
            fn: The function to apply to each value

        Returns:
            A new pattern with transformed values
        """
        return pat_map(fn)(self)

    def fold[Z](self, start: Z, fn: Callable[[Box[Z], T], None]) -> Z:
        """Fold over the pattern values.

        Args:
            start: The initial accumulator value
            fn: The folding function

        Returns:
            The final accumulated value
        """
        return pat_fold(fn)(start, self)

    def cata[Z](self, fn: Callable[[PatF[T, Z]], Z]) -> Z:
        """Apply a catamorphism to the pattern.

        Args:
            fn: The algebra function

        Returns:
            The result of the catamorphism
        """
        return pat_cata(fn)(self)

    def cata_state[S, Z](
        self, start: S, fn: Callable[[Box[S], PatF[T, Z]], Z]
    ) -> Tuple[S, Z]:
        """Apply a stateful catamorphism to the pattern.

        Args:
            start: The initial state
            fn: The algebra function with state

        Returns:
            A tuple of (final_state, result)
        """
        return pat_cata_state(fn)(start, self)


@dataclass(frozen=True)
class PatSilence(PatF[Any, Any]):
    """Pattern functor representing silence."""

    pass


_PAT_SILENCE = Pat(PatSilence())


@dataclass(frozen=True)
class PatPure[T](PatF[T, Any]):
    """Pattern functor containing a single value.

    Args:
        val: The contained value
    """

    val: T


@dataclass(frozen=True)
class PatSeq[T, R](PatF[T, R]):
    """Pattern functor for sequential composition.

    Args:
        children: The child patterns to play in sequence
    """

    children: PSeq[R]


@dataclass(frozen=True)
class PatPar[T, R](PatF[T, R]):
    """Pattern functor for parallel composition.

    Args:
        children: The child patterns to play in parallel
    """

    children: PSeq[R]


@dataclass(frozen=True)
class PatChoice[T, R](PatF[T, R]):
    """Pattern functor for choice between patterns.

    Args:
        choices: The patterns to choose from
    """

    choices: PSeq[R]


@dataclass(frozen=True)
class PatEuclidean[T, R](PatF[T, R]):
    """Pattern functor for Euclidean rhythms.

    Args:
        atom: The pattern to distribute
        hits: Number of hits to distribute
        steps: Total number of steps
        rotation: Rotation offset
    """

    atom: R
    hits: int
    steps: int
    rotation: int


@dataclass(frozen=True)
class PatPolymetric[T, R](PatF[T, R]):
    """Pattern functor for polymetric patterns.

    Args:
        patterns: The patterns to play polymetrically
    """

    patterns: PSeq[R]


@dataclass(frozen=True)
class PatRepetition[T, R](PatF[T, R]):
    """Pattern functor for repetition.

    Args:
        pattern: The pattern to repeat
        operator: The repetition operator
        count: The repetition count
    """

    pattern: R
    operator: RepetitionOp
    count: int


@dataclass(frozen=True)
class PatElongation[T, R](PatF[T, R]):
    """Pattern functor for elongation.

    Args:
        pattern: The pattern to elongate
        count: The elongation count
    """

    pattern: R
    count: int


@dataclass(frozen=True)
class PatProbability[T, R](PatF[T, R]):
    """Pattern functor for probabilistic patterns.

    Args:
        pattern: The pattern to apply probability to
        probability: The probability value (0 to 1 as a Fraction)
    """

    pattern: R
    probability: Fraction


@dataclass(frozen=True)
class PatSelect[T, R](PatF[T, R]):
    """Pattern functor for sample selection.

    Args:
        pattern: The pattern to select from
        selector: The selection identifier
    """

    pattern: R
    selector: str


@dataclass(frozen=True)
class PatAlternating[T, R](PatF[T, R]):
    """Pattern functor for alternating patterns.

    Args:
        patterns: The patterns to alternate between
    """

    patterns: PSeq[R]


@dataclass(frozen=True)
class PatReplicate[T, R](PatF[T, R]):
    """Pattern functor for replicate patterns (!).

    Args:
        pattern: The pattern to replicate
        count: The number of times to replicate
    """

    pattern: R
    count: int


@dataclass(frozen=True)
class PatRatio[T, R](PatF[T, R]):
    """Pattern functor for ratio patterns (%).

    Args:
        pattern: The pattern to apply ratio to
        numerator: The numerator of the ratio
        denominator: The denominator of the ratio
    """

    pattern: R
    numerator: int
    denominator: int


@dataclass(frozen=True)
class PatPolymetricSub[T, R](PatF[T, R]):
    """Pattern functor for polymetric patterns with subdivision ({}%).

    Args:
        patterns: The patterns to play polymetrically
        subdivision: The subdivision factor
    """

    patterns: PSeq[R]
    subdivision: int


def pat_cata_env[V, T, Z](fn: Callable[[V, PatF[T, Z]], Z]) -> Callable[[V, Pat[T]], Z]:
    """Create a catamorphism with environment.

    Args:
        fn: The algebra function that takes environment and pattern functor

    Returns:
        A function that applies the catamorphism with environment
    """

    def wrapper(env: V, pat: Pat[T]) -> Z:
        pf = pat.unwrap
        match pf:
            case PatSilence():
                return fn(env, pf)
            case PatPure(_):
                return fn(env, pf)
            case PatSeq(cs):
                czs = PSeq.mk(wrapper(env, c) for c in cs)
                return fn(env, PatSeq(czs))
            case PatPar(cs):
                czs = PSeq.mk(wrapper(env, c) for c in cs)
                return fn(env, PatPar(czs))
            case PatChoice(cs):
                czs = PSeq.mk(wrapper(env, c) for c in cs)
                return fn(env, PatChoice(czs))
            case PatEuclidean(atom, hits, steps, rotation):
                atom_z = wrapper(env, atom)
                return fn(env, PatEuclidean(atom_z, hits, steps, rotation))
            case PatPolymetric(ps):
                pzs = PSeq.mk(wrapper(env, p) for p in ps)
                return fn(env, PatPolymetric(pzs))
            case PatRepetition(p, op, count):
                pz = wrapper(env, p)
                return fn(env, PatRepetition(pz, op, count))
            case PatElongation(p, count):
                pz = wrapper(env, p)
                return fn(env, PatElongation(pz, count))
            case PatProbability(p, prob):
                pz = wrapper(env, p)
                return fn(env, PatProbability(pz, prob))
            case PatSelect(p, selector):
                pz = wrapper(env, p)
                return fn(env, PatSelect(pz, selector))
            case PatAlternating(ps):
                pzs = PSeq.mk(wrapper(env, p) for p in ps)
                return fn(env, PatAlternating(pzs))
            case PatReplicate(p, count):
                pz = wrapper(env, p)
                return fn(env, PatReplicate(pz, count))
            case PatRatio(p, num, denom):
                pz = wrapper(env, p)
                return fn(env, PatRatio(pz, num, denom))
            case PatPolymetricSub(ps, subdivision):
                pzs = PSeq.mk(wrapper(env, p) for p in ps)
                return fn(env, PatPolymetricSub(pzs, subdivision))
            case _:
                raise PartialMatchException(pf)

    return wrapper


def pat_cata_state[S, T, Z](
    fn: Callable[[Box[S], PatF[T, Z]], Z],
) -> Callable[[S, Pat[T]], Tuple[S, Z]]:
    """Create a stateful catamorphism.

    Args:
        fn: The algebra function that takes state and pattern functor

    Returns:
        A function that applies the stateful catamorphism
    """
    k: Callable[[Box[S], Pat[T]], Z] = pat_cata_env(fn)

    def unwrap(start: S, pat: Pat[T]) -> Tuple[S, Z]:
        box = Box(start)
        out = k(box, pat)
        return (box.value, out)

    return unwrap


def pat_cata[T, Z](fn: Callable[[PatF[T, Z]], Z]) -> Callable[[Pat[T]], Z]:
    """Create a catamorphism.

    Args:
        fn: The algebra function

    Returns:
        A function that applies the catamorphism
    """
    k: Callable[[None, Pat[T]], Z] = pat_cata_env(ignore_arg(fn))
    return partial(k, None)


def pat_map[T, U](fn: Callable[[T], U]) -> Callable[[Pat[T]], Pat[U]]:
    """Create a pattern mapping function.

    Args:
        fn: The function to map over pattern values

    Returns:
        A function that maps the given function over patterns
    """

    def elim(pf: PatF[T, Pat[U]]) -> Pat[U]:
        match pf:
            case PatSilence():
                return Pat.silence()
            case PatPure(val):
                return Pat(PatPure(fn(val)))
            case PatSeq(cs):
                return Pat(PatSeq(cs))
            case PatPar(cs):
                return Pat(PatPar(cs))
            case PatChoice(cs):
                return Pat(PatChoice(cs))
            case PatEuclidean(atom, hits, steps, rotation):
                return Pat(PatEuclidean(atom, hits, steps, rotation))
            case PatPolymetric(ps):
                return Pat(PatPolymetric(ps))
            case PatRepetition(p, op, count):
                return Pat(PatRepetition(p, op, count))
            case PatElongation(p, count):
                return Pat(PatElongation(p, count))
            case PatProbability(p, prob):
                return Pat(PatProbability(p, prob))
            case PatSelect(p, selector):
                return Pat(PatSelect(p, selector))
            case PatAlternating(ps):
                return Pat(PatAlternating(ps))
            case PatReplicate(p, count):
                return Pat(PatReplicate(p, count))
            case PatRatio(p, num, denom):
                return Pat(PatRatio(p, num, denom))
            case PatPolymetricSub(ps, subdivision):
                return Pat(PatPolymetricSub(ps, subdivision))
            case _:
                raise PartialMatchException(pf)

    return pat_cata(elim)


def pat_fold[T, Z](fn: Callable[[Box[Z], T], None]) -> Callable[[Z, Pat[T]], Z]:
    """Create a pattern folding function.

    Args:
        fn: The folding function

    Returns:
        A function that folds over patterns
    """

    def visit(box: Box[Z], pf: PatF[T, None]) -> None:
        match pf:
            case PatPure(val):
                fn(box, val)
            case _:
                pass

    k: Callable[[Box[Z], Pat[T]], None] = pat_cata_env(visit)

    def wrapper(start: Z, pat: Pat[T]) -> Z:
        box = Box(start)
        k(box, pat)
        return box.value

    return wrapper
