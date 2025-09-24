"""Pattern types and operations for the minipat pattern language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Iterable, Optional, cast, override

from minipat.common import PartialMatchException, ignore_arg
from spiny.arrow import ArrowM
from spiny.common import Box, Singleton
from spiny.seq import PSeq


class SpeedOp(Enum):
    """Enumeration for speed operators."""

    Fast = "*"
    """Fast speed operator (*) - repeats pattern faster."""

    Slow = "/"
    """Slow speed operator (/) - slows down pattern."""


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
    def silent() -> Pat[T]:
        """Create a silent pattern.

        Textual form: ~

        Returns:
            A pattern representing silence
        """
        return _PAT_SILENT

    @staticmethod
    def pure(val: T) -> Pat[T]:
        """Create a pattern with a single value.

        Textual form: value

        Args:
            val: The value to wrap

        Returns:
            A pattern containing the single value
        """
        return Pat(PatPure(val))

    @staticmethod
    def seq(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create a sequential pattern.

        Textual form: [p1 p2 p3 ...]

        Args:
            pats: The patterns to sequence

        Returns:
            A pattern that plays the given patterns in sequence
        """
        return Pat(PatSeq(PSeq.mk(pats)))

    @staticmethod
    def par(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create a parallel pattern.

        Textual form: {p1, p2, p3, ...}

        Args:
            pats: The patterns to play in parallel

        Returns:
            A pattern that plays the given patterns simultaneously
        """
        return Pat(PatPar(PSeq.mk(pats)))

    @staticmethod
    def rand(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create a random choice pattern.

        Textual form: <p1 | p2 | p3 | ...>

        Args:
            pats: The patterns to choose from

        Returns:
            A pattern that randomly chooses from the given choices
        """
        return Pat(PatRand(PSeq.mk(pats)))

    @staticmethod
    def euc(pat: Pat[T], hits: int, steps: int, rotation: int = 0) -> Pat[T]:
        """Create a Euclidean rhythm pattern.

        Textual form: pattern(hits,steps) or pattern(hits,steps,rotation)

        Args:
            pat: The pattern to distribute
            hits: Number of hits to distribute
            steps: Total number of steps
            rotation: Optional rotation offset

        Returns:
            A pattern with Euclidean rhythm distribution
        """
        return Pat(PatEuc(pat, hits, steps, rotation))

    @staticmethod
    def poly(pats: Iterable[Pat[T]], subdiv: Optional[int] = None) -> Pat[T]:
        """Create a polymetric pattern.

        Textual form: {p1, p2, p3, ...} or {p1, p2, p3, ...}%N

        Args:
            pats: The patterns to play polymetrically
            subdiv: Optional subdivision factor for {}%N patterns

        Returns:
            A polymetric pattern with or without subdivision
        """
        return Pat(PatPoly(PSeq.mk(pats), subdiv))

    @staticmethod
    def speed(pat: Pat[T], op: SpeedOp, factor: Fraction) -> Pat[T]:
        """Create a speed up/down pattern.

        Textual form: pattern*count or pattern/count

        Args:
            pat: The pattern to speed up/down
            op: The speed operator (Fast or Slow)
            factor: The factor by which we speed up/down

        Returns:
            A pattern with the specified speed
        """
        return Pat(PatSpeed(pat, op, factor))

    @staticmethod
    def stretch(pattern: Pat[T], count: Fraction) -> Pat[T]:
        """Create a stretched pattern.

        Textual form: pattern@count

        Args:
            pattern: The pattern to stretch
            count: The stretch count (can be fractional)

        Returns:
            A pattern stretched by the given count
        """
        return Pat(PatStretch(pattern, count))

    @staticmethod
    def prob(pat: Pat[T], chance: Fraction = Fraction(1, 2)) -> Pat[T]:
        """Create a probabilistic pattern.

        Textual form: pattern?

        Args:
            pat: The pattern to apply probability to
            chance: The probability (0 to 1 as a Fraction)

        Returns:
            A pattern that plays with the given probability
        """
        return Pat(PatProb(pat, chance))

    @staticmethod
    def alt(pats: Iterable[Pat[T]]) -> Pat[T]:
        """Create an alternating pattern.

        Textual form: <p1 p2 p3 ...>

        Args:
            pat: The patterns to alternate between

        Returns:
            A pattern that alternates between the given patterns
        """
        return Pat(PatAlt(PSeq.mk(pats)))

    @staticmethod
    def repeat(pat: Pat[T], count: Fraction) -> Pat[T]:
        """Create a finitely repeating pattern (!).

        Textual form: pattern!count

        Args:
            pat: The pattern to repeat
            count: The number of times to repeat (can be fractional)

        Returns:
            The repeated pattern
        """
        return Pat(PatRepeat(pat, count))

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
    ) -> tuple[S, Z]:
        """Apply a stateful catamorphism to the pattern.

        Args:
            start: The initial state
            fn: The algebra function with state

        Returns:
            A tuple of (final_state, result)
        """
        return pat_cata_state(fn)(start, self)


@dataclass(frozen=True)
class PatSilent(PatF[Any, Any]):
    """Pattern functor representing silence.

    Textual form: ~
    """

    pass


_PAT_SILENT = Pat(PatSilent())


@dataclass(frozen=True)
class PatPure[T](PatF[T, Any]):
    """Pattern functor containing a single value.

    Textual form: value

    Args:
        value: The contained value
    """

    value: T


@dataclass(frozen=True)
class PatSeq[T, R](PatF[T, R]):
    """Pattern functor for sequential composition.

    Textual form: [p1 p2 p3 ...]

    Args:
        pats: The child patterns to play in sequence
    """

    pats: PSeq[R]


@dataclass(frozen=True)
class PatPar[T, R](PatF[T, R]):
    """Pattern functor for parallel composition.

    Textual form: {p1, p2, p3, ...}

    Args:
        patterns: The child patterns to play in parallel
    """

    pats: PSeq[R]


@dataclass(frozen=True)
class PatRand[T, R](PatF[T, R]):
    """Pattern functor for random choice between patterns.

    Textual form: <p1 | p2 | p3 | ...>

    Args:
        pats: The patterns to choose from
    """

    pats: PSeq[R]


@dataclass(frozen=True)
class PatEuc[T, R](PatF[T, R]):
    """Pattern functor for Euclidean rhythms.

    Textual form: pattern(hits,steps) or pattern(hits,steps,rotation)

    Args:
        pat: The pattern to distribute
        hits: Number of hits to distribute
        steps: Total number of steps
        rotation: Rotation offset
    """

    pat: R
    hits: int
    steps: int
    rotation: int


@dataclass(frozen=True)
class PatPoly[T, R](PatF[T, R]):
    """Pattern functor for polymetric patterns.

    Textual form: {p1, p2, p3, ...} or {p1, p2, p3, ...}%N

    Args:
        pats: The patterns to play polymetrically
        subdivision: Optional subdivision factor for {}%N patterns
    """

    pats: PSeq[R]
    subdiv: Optional[int] = None


@dataclass(frozen=True)
class PatSpeed[T, R](PatF[T, R]):
    """Pattern functor for speed patterns.

    Textual form: pattern*count or pattern/count

    Args:
        pat: The pattern to speed up/down
        op: The speed operator
        factor: The speed factor
    """

    pat: R
    op: SpeedOp
    factor: Fraction


@dataclass(frozen=True)
class PatStretch[T, R](PatF[T, R]):
    """Pattern functor for stretch patterns.

    Textual form: pattern@count

    Args:
        pat: The pattern to stretch
        count: The stretch count
    """

    pat: R
    count: Fraction


@dataclass(frozen=True)
class PatProb[T, R](PatF[T, R]):
    """Pattern functor for probabilistic patterns.

    Textual form: pattern?

    Args:
        pat: The pattern to apply probability to
        chance: The probability value (0 to 1 as a Fraction)
    """

    pat: R
    chance: Fraction


@dataclass(frozen=True)
class PatAlt[T, R](PatF[T, R]):
    """Pattern functor for alternating patterns.

    Textual form: <p1 p2 p3 ...>

    Args:
        pats: The patterns to alternate between
    """

    pats: PSeq[R]


@dataclass(frozen=True)
class PatRepeat[T, R](PatF[T, R]):
    """Pattern functor for repeat patterns (!).

    Textual form: pattern!count

    Args:
        pat: The pattern to repeat
        count: The number of times to repeat (can be fractional)
    """

    pat: R
    count: Fraction


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
            case PatSilent():
                return fn(env, pf)
            case PatPure(_):
                return fn(env, pf)
            case PatSeq(pats):
                czs = PSeq.mk(wrapper(env, c) for c in pats)
                return fn(env, PatSeq(czs))
            case PatPar(pats):
                czs = PSeq.mk(wrapper(env, c) for c in pats)
                return fn(env, PatPar(czs))
            case PatRand(pats):
                czs = PSeq.mk(wrapper(env, c) for c in pats)
                return fn(env, PatRand(czs))
            case PatEuc(pat, hits, steps, rotation):
                pat_z = wrapper(env, pat)
                return fn(env, PatEuc(pat_z, hits, steps, rotation))
            case PatPoly(pats, None):
                pzs = PSeq.mk(wrapper(env, p) for p in pats)
                return fn(env, PatPoly(pzs, None))
            case PatSpeed(pat, op, factor):
                pz = wrapper(env, pat)
                return fn(env, PatSpeed(pz, op, factor))
            case PatStretch(pat, count):
                pz = wrapper(env, pat)
                return fn(env, PatStretch(pz, count))
            case PatProb(pat, prob):
                pz = wrapper(env, pat)
                return fn(env, PatProb(pz, prob))
            case PatAlt(pats):
                pzs = PSeq.mk(wrapper(env, p) for p in pats)
                return fn(env, PatAlt(pzs))
            case PatRepeat(pat, count):
                pz = wrapper(env, pat)
                return fn(env, PatRepeat(pz, count))
            case PatPoly(pats, subdiv):
                pzs = PSeq.mk(wrapper(env, p) for p in pats)
                return fn(env, PatPoly(pzs, subdiv))
            case _:
                raise PartialMatchException(pf)

    return wrapper


def pat_cata_state[S, T, Z](
    fn: Callable[[Box[S], PatF[T, Z]], Z],
) -> Callable[[S, Pat[T]], tuple[S, Z]]:
    """Create a stateful catamorphism.

    Args:
        fn: The algebra function that takes state and pattern functor

    Returns:
        A function that applies the stateful catamorphism
    """
    k: Callable[[Box[S], Pat[T]], Z] = pat_cata_env(fn)

    def unwrap(start: S, pat: Pat[T]) -> tuple[S, Z]:
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
            case PatSilent():
                return Pat.silent()
            case PatPure(val):
                return Pat(PatPure(fn(val)))
            case PatSeq(pats):
                return Pat(PatSeq(pats))
            case PatPar(pats):
                return Pat(PatPar(pats))
            case PatRand(pats):
                return Pat(PatRand(pats))
            case PatEuc(pat, hits, steps, rotation):
                return Pat(PatEuc(pat, hits, steps, rotation))
            case PatPoly(pats, None):
                return Pat(PatPoly(pats, None))
            case PatSpeed(pat, op, factor):
                return Pat(PatSpeed(pat, op, factor))
            case PatStretch(pat, count):
                return Pat(PatStretch(pat, count))
            case PatProb(pat, prob):
                return Pat(PatProb(pat, prob))
            case PatAlt(pats):
                return Pat(PatAlt(pats))
            case PatRepeat(pat, count):
                return Pat(PatRepeat(pat, count))
            case PatPoly(pats, subdiv):
                return Pat(PatPoly(pats, subdiv))
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

    def elim(box: Box[Z], pf: PatF[T, None]) -> None:
        match pf:
            case PatPure(val):
                fn(box, val)
            case _:
                pass

    k: Callable[[Box[Z], Pat[T]], None] = pat_cata_env(elim)

    def wrapper(start: Z, pat: Pat[T]) -> Z:
        box = Box(start)
        k(box, pat)
        return box.value

    return wrapper


def pat_bind[T, U](fn: Callable[[T], Pat[U]]) -> Callable[[Pat[T]], Pat[U]]:
    """Create a pattern binding function.

    Args:
        fn: The binding function

    Returns:
        A function that binds over patterns
    """

    def elim(pf: PatF[T, Pat[U]]) -> Pat[U]:
        match pf:
            case PatPure(value):
                return fn(value)
            case _:
                return Pat(cast(PatF[U, Pat[U]], pf))

    return pat_cata(elim)


class PatBinder[T, U](ArrowM[T, U, Pat[U]]):
    @classmethod
    @override
    def pure(cls) -> PatBinder[T, T]:
        """Create a pure binder that returns patterns unchanged."""
        return IdPatBinder()

    @override
    def unsafe_bind[FV](self, context: Pat[U], fn: Callable[[U], FV]) -> FV:
        fn1 = cast(Callable[[U], Pat[Any]], fn)
        pat1 = pat_bind(fn1)(context)
        return cast(FV, pat1)

    def bind[V](self, pat: Pat[U], fn: Callable[[U], Pat[V]]) -> Pat[V]:
        return self.unsafe_bind(pat, fn)


class IdPatBinder[T](PatBinder[T, T], Singleton):
    """Identity pattern binder that returns values as pure patterns."""

    @override
    def apply(self, value: T) -> Pat[T]:
        """Wrap a value in a pure pattern."""
        return Pat.pure(value)
