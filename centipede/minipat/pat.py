from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Tuple

from centipede.common import PartialMatchException, ignore_arg
from centipede.minipat.arc import Arc
from centipede.minipat.common import Delta, Factor
from centipede.spiny import Box, PSeq


class RepetitionOp(Enum):
    """Enumeration for repetition operators."""

    FAST = "*"  # Multiply/repeat
    SLOW = "/"  # Divide/slow down


# sealed
class PatF[T, R]:
    pass


@dataclass(frozen=True)
class Pat[T]:
    unwrap: PatF[T, Pat[T]]

    @staticmethod
    def silence() -> Pat[T]:
        return _PAT_SILENCE

    @staticmethod
    def pure(val: T) -> Pat[T]:
        return Pat(PatPure(val))

    @staticmethod
    def seq(pats: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatSeq(PSeq.mk(pats)))

    @staticmethod
    def par(pats: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatPar(PSeq.mk(pats)))

    @staticmethod
    def choice(choices: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatChoice(PSeq.mk(choices)))

    @staticmethod
    def euclidean(atom: Pat[T], hits: int, steps: int, rotation: int = 0) -> Pat[T]:
        return Pat(PatEuclidean(atom, hits, steps, rotation))

    @staticmethod
    def polymetric(patterns: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatPolymetric(PSeq.mk(patterns)))

    @staticmethod
    def repetition(pattern: Pat[T], operator: RepetitionOp, count: int) -> Pat[T]:
        return Pat(PatRepetition(pattern, operator, count))

    @staticmethod
    def elongation(pattern: Pat[T], count: int) -> Pat[T]:
        return Pat(PatElongation(pattern, count))

    @staticmethod
    def probability(pattern: Pat[T], prob: float = 0.5) -> Pat[T]:
        return Pat(PatProbability(pattern, prob))

    @staticmethod
    def select(pattern: Pat[T], selector: str) -> Pat[T]:
        return Pat(PatSelect(pattern, selector))

    @staticmethod
    def group(pattern: Pat[T]) -> Pat[T]:
        return Pat(PatGroup(pattern))

    @staticmethod
    def alternating(patterns: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatAlternating(PSeq.mk(patterns)))

    def mask(self, arc: Arc) -> Pat[T]:
        if arc.null():
            return Pat.silence()
        else:
            match self.unwrap:
                case PatMask(a, p):
                    return p.mask(a.intersect(arc))
                case _:
                    return Pat(PatMask(arc, self))

    def shift(self, delta: Delta) -> Pat[T]:
        if delta == 0:
            return self
        else:
            match self.unwrap:
                case PatShift(d, p):
                    return p.shift(d + delta)
                case _:
                    return Pat(PatShift(delta, self))

    def scale(self, factor: Factor) -> Pat[T]:
        if factor <= 0:
            return Pat.silence()
        elif factor == 1:
            return self
        else:
            match self.unwrap:
                case PatScale(f, p):
                    return p.scale(f * factor)
                case _:
                    return Pat(PatScale(factor, self))

    def clip(self, factor: Factor) -> Pat[T]:
        if factor <= 0:
            return Pat.silence()
        elif factor == 1:
            return self
        else:
            match self.unwrap:
                case PatClip(f, p):
                    return p.clip(f * factor)
                case _:
                    return Pat(PatClip(factor, self))

    def map[U](self, fn: Callable[[T], U]) -> Pat[U]:
        return pat_map(fn)(self)

    def fold[Z](self, start: Z, fn: Callable[[Box[Z], T], None]) -> Z:
        return pat_fold(fn)(start, self)

    def cata[Z](self, fn: Callable[[PatF[T, Z]], Z]) -> Z:
        return pat_cata(fn)(self)

    def cata_state[S, Z](
        self, start: S, fn: Callable[[Box[S], PatF[T, Z]], Z]
    ) -> Tuple[S, Z]:
        return pat_cata_state(fn)(start, self)


@dataclass(frozen=True)
class PatSilence(PatF[Any, Any]):
    pass


_PAT_SILENCE = Pat(PatSilence())


@dataclass(frozen=True)
class PatPure[T](PatF[T, Any]):
    val: T


@dataclass(frozen=True)
class PatMask[T, R](PatF[T, R]):
    arc: Arc
    child: R

    # @override
    # def active(self, bounds: Arc) -> Arc:
    #     sub_bounds = bounds.intersect(self.mask)
    #     if sub_bounds.null():
    #         return sub_bounds
    #     else:
    #         return self.sub_pat.active(sub_bounds)
    #
    # @override
    # def rep(self, bounds: Arc) -> Generator[Ev[T]]:
    #     sub_bounds = bounds.intersect(self.mask)
    #     if sub_bounds.null():
    #         return
    #     else:
    #         yield from self.sub_pat.rep(sub_bounds)


@dataclass(frozen=True)
class PatShift[T, R](PatF[T, R]):
    delta: Delta
    child: R

    # @override
    # def active(self, bounds: Arc) -> Arc:
    #     return self.sub_pat.active(bounds.shift(self.delta)).shift(-self.delta)
    #
    # @override
    # def rep(self, bounds: Arc) -> Generator[Ev[T]]:
    #     yield from map(
    #         lambda ev: ev.shift(-self.delta), self.sub_pat.rep(bounds.shift(self.delta))
    #     )


@dataclass(frozen=True)
class PatScale[T, R](PatF[T, R]):
    factor: Factor
    child: R


@dataclass(frozen=True)
class PatClip[T, R](PatF[T, R]):
    factor: Factor
    child: R


@dataclass(frozen=True)
class PatSeq[T, R](PatF[T, R]):
    children: PSeq[R]


@dataclass(frozen=True)
class PatPar[T, R](PatF[T, R]):
    children: PSeq[R]


@dataclass(frozen=True)
class PatChoice[T, R](PatF[T, R]):
    choices: PSeq[R]


@dataclass(frozen=True)
class PatEuclidean[T, R](PatF[T, R]):
    atom: R
    hits: int
    steps: int
    rotation: int


@dataclass(frozen=True)
class PatPolymetric[T, R](PatF[T, R]):
    patterns: PSeq[R]


@dataclass(frozen=True)
class PatRepetition[T, R](PatF[T, R]):
    pattern: R
    operator: RepetitionOp
    count: int


@dataclass(frozen=True)
class PatElongation[T, R](PatF[T, R]):
    pattern: R
    count: int


@dataclass(frozen=True)
class PatProbability[T, R](PatF[T, R]):
    pattern: R
    probability: float


@dataclass(frozen=True)
class PatSelect[T, R](PatF[T, R]):
    pattern: R
    selector: str


@dataclass(frozen=True)
class PatGroup[T, R](PatF[T, R]):
    pattern: R


@dataclass(frozen=True)
class PatAlternating[T, R](PatF[T, R]):
    patterns: PSeq[R]


def pat_cata_env[V, T, Z](fn: Callable[[V, PatF[T, Z]], Z]) -> Callable[[V, Pat[T]], Z]:
    def wrapper(env: V, pat: Pat[T]) -> Z:
        pf = pat.unwrap
        match pf:
            case PatSilence():
                return fn(env, pf)
            case PatPure(_):
                return fn(env, pf)
            case PatClip(f, c):
                cz = wrapper(env, c)
                return fn(env, PatClip(f, cz))
            case PatMask(a, c):
                cz = wrapper(env, c)
                return fn(env, PatMask(a, cz))
            case PatScale(f, c):
                cz = wrapper(env, c)
                return fn(env, PatScale(f, cz))
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
            case PatGroup(p):
                pz = wrapper(env, p)
                return fn(env, PatGroup(pz))
            case PatAlternating(ps):
                pzs = PSeq.mk(wrapper(env, p) for p in ps)
                return fn(env, PatAlternating(pzs))
            case _:
                raise PartialMatchException(pf)

    return wrapper


def pat_cata_state[S, T, Z](
    fn: Callable[[Box[S], PatF[T, Z]], Z],
) -> Callable[[S, Pat[T]], Tuple[S, Z]]:
    k: Callable[[Box[S], Pat[T]], Z] = pat_cata_env(fn)

    def unwrap(start: S, pat: Pat[T]) -> Tuple[S, Z]:
        box = Box(start)
        out = k(box, pat)
        return (box.value, out)

    return unwrap


def pat_cata[T, Z](fn: Callable[[PatF[T, Z]], Z]) -> Callable[[Pat[T]], Z]:
    k: Callable[[None, Pat[T]], Z] = pat_cata_env(ignore_arg(fn))
    return partial(k, None)


def pat_map[T, U](fn: Callable[[T], U]) -> Callable[[Pat[T]], Pat[U]]:
    def elim(pf: PatF[T, Pat[U]]) -> Pat[U]:
        match pf:
            case PatSilence():
                return Pat.silence()
            case PatPure(val):
                return Pat(PatPure(fn(val)))
            case PatClip(f, c):
                return Pat(PatClip(f, c))
            case PatMask(a, c):
                return Pat(PatMask(a, c))
            case PatScale(f, c):
                return Pat(PatScale(f, c))
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
            case PatGroup(p):
                return Pat(PatGroup(p))
            case PatAlternating(ps):
                return Pat(PatAlternating(ps))
            case _:
                raise PartialMatchException(pf)

    return pat_cata(elim)


def pat_fold[T, Z](fn: Callable[[Box[Z], T], None]) -> Callable[[Z, Pat[T]], Z]:
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
