from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from centipede.spiny.common import Impossible, Todo

type OuterNode[T] = Union[Tuple[T], Tuple[T, T], Tuple[T, T, T], Tuple[T, T, T, T]]
type InnerNode[T] = Union[Tuple[int, T, T], Tuple[int, T, T, T]]


# sealed
class Seq[T]:
    """A Hinze-Patterson finger tree as persistent catenable sequence"""

    @staticmethod
    def empty() -> Seq[T]:
        return _SEQ_EMPTY

    @staticmethod
    def singleton(value: T) -> Seq[T]:
        return SeqSingle(value)

    def null(self) -> bool:
        match self:
            case SeqEmpty():
                return True
            case _:
                return False

    def size(self) -> int:
        match self:
            case SeqEmpty():
                return 0
            case SeqSingle(_):
                return 1
            case SeqDeep(size, _, _, _):
                return size
            case _:
                raise Impossible

    def uncons(self) -> Optional[Tuple[T, Seq[T]]]:
        return _seq_uncons(self)

    def cons(self, value: T) -> Seq[T]:
        return _seq_cons(self, value)

    def unsnoc(self) -> Optional[Tuple[Seq[T], T]]:
        return _seq_unsnoc(self)

    def snoc(self, value: T) -> Seq[T]:
        return _seq_snoc(self, value)

    def concat(self, other: Seq[T]) -> Seq[T]:
        return _seq_concat(self, other)

    def to_list(self) -> List[T]:
        return _seq_to_list(self)


@dataclass(frozen=True)
class SeqEmpty[T](Seq[T]):
    pass


_SEQ_EMPTY: Seq[Any] = SeqEmpty()


@dataclass(frozen=True)
class SeqSingle[T](Seq[T]):
    _value: T


@dataclass(frozen=True)
class SeqDeep[T](Seq[T]):
    _size: int
    _front: OuterNode[T]
    _between: Seq[InnerNode[T]]
    _back: OuterNode[T]


def _seq_uncons[T](seq: Seq[T]) -> Optional[Tuple[T, Seq[T]]]:
    match seq:
        case SeqEmpty():
            return None
        case SeqSingle(value):
            return (value, Seq.empty())
        case SeqDeep(size, front, between, back):
            match front:
                case (a,):
                    if between.null():
                        match back:
                            case (b,):
                                return (a, SeqSingle(b))
                            case (b, c):
                                return (
                                    a,
                                    SeqDeep(size - 1, (b,), Seq.empty(), (c,)),
                                )
                            case (b, c, d):
                                return (
                                    a,
                                    SeqDeep(size - 1, (b, c), Seq.empty(), (d,)),
                                )
                            case (b, c, d, e):
                                return (
                                    a,
                                    SeqDeep(size - 1, (b, c, d), Seq.empty(), (e,)),
                                )
                            case _:
                                raise Impossible
                    else:
                        between_uncons = _seq_uncons(between)
                        if between_uncons is None:
                            match back:
                                case (b,):
                                    return (a, SeqSingle(b))
                                case (b, c):
                                    return (
                                        a,
                                        SeqDeep(size - 1, (b,), Seq.empty(), (c,)),
                                    )
                                case (b, c, d):
                                    return (
                                        a,
                                        SeqDeep(size - 1, (b, c), Seq.empty(), (d,)),
                                    )
                                case (b, c, d, e):
                                    return (
                                        a,
                                        SeqDeep(size - 1, (b, c, d), Seq.empty(), (e,)),
                                    )
                                case _:
                                    raise Impossible
                        else:
                            inner_head, between_tail = between_uncons
                            if len(inner_head) == 3:
                                _, b, c = inner_head
                                return (
                                    a,
                                    SeqDeep(size - 1, (b, c), between_tail, back),
                                )
                            elif len(inner_head) == 4:
                                _, b, c, d = inner_head
                                return (
                                    a,
                                    SeqDeep(size - 1, (b, c, d), between_tail, back),
                                )
                            else:
                                raise Impossible
                case (a, b):
                    return (a, SeqDeep(size - 1, (b,), between, back))
                case (a, b, c):
                    return (a, SeqDeep(size - 1, (b, c), between, back))
                case (a, b, c, d):
                    return (a, SeqDeep(size - 1, (b, c, d), between, back))
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_cons[T](seq: Seq[T], value: T) -> Seq[T]:
    match seq:
        case SeqEmpty():
            return SeqSingle(value)
        case SeqSingle(existing_value):
            return SeqDeep(2, (value,), Seq.empty(), (existing_value,))
        case SeqDeep(size, front, between, back):
            match front:
                case (a,):
                    return SeqDeep(size + 1, (value, a), between, back)
                case (a, b):
                    return SeqDeep(size + 1, (value, a, b), between, back)
                case (a, b, c):
                    return SeqDeep(size + 1, (value, a, b, c), between, back)
                case (a, b, c, d):
                    new_inner = (2, a, b)
                    new_between = _seq_cons(between, new_inner)
                    return SeqDeep(size + 1, (value, c, d), new_between, back)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_snoc[T](seq: Seq[T], value: T) -> Seq[T]:
    match seq:
        case SeqEmpty():
            return SeqSingle(value)
        case SeqSingle(existing_value):
            return SeqDeep(2, (existing_value,), Seq.empty(), (value,))
        case SeqDeep(size, front, between, back):
            match back:
                case (a,):
                    return SeqDeep(size + 1, front, between, (a, value))
                case (a, b):
                    return SeqDeep(size + 1, front, between, (a, b, value))
                case (a, b, c):
                    return SeqDeep(size + 1, front, between, (a, b, c, value))
                case (a, b, c, d):
                    new_inner = (2, c, d)
                    new_between = _seq_snoc(between, new_inner)
                    return SeqDeep(size + 1, front, new_between, (a, b, value))
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_unsnoc[T](seq: Seq[T]) -> Optional[Tuple[Seq[T], T]]:
    match seq:
        case SeqEmpty():
            return None
        case SeqSingle(value):
            return (Seq.empty(), value)
        case SeqDeep(size, front, between, back):
            match back:
                case (a,):
                    if between.null():
                        match front:
                            case (b,):
                                return (SeqSingle(b), a)
                            case (b, c):
                                return (
                                    SeqDeep(size - 1, (b,), Seq.empty(), (c,)),
                                    a,
                                )
                            case (b, c, d):
                                return (
                                    SeqDeep(size - 1, (b,), Seq.empty(), (c, d)),
                                    a,
                                )
                            case (b, c, d, e):
                                return (
                                    SeqDeep(size - 1, (b,), Seq.empty(), (c, d, e)),
                                    a,
                                )
                            case _:
                                raise Impossible
                    else:
                        between_unsnoc = _seq_unsnoc(between)
                        if between_unsnoc is None:
                            match front:
                                case (b,):
                                    return (SeqSingle(b), a)
                                case (b, c):
                                    return (
                                        SeqDeep(size - 1, (b,), Seq.empty(), (c,)),
                                        a,
                                    )
                                case (b, c, d):
                                    return (
                                        SeqDeep(size - 1, (b,), Seq.empty(), (c, d)),
                                        a,
                                    )
                                case (b, c, d, e):
                                    return (
                                        SeqDeep(size - 1, (b,), Seq.empty(), (c, d, e)),
                                        a,
                                    )
                                case _:
                                    raise Impossible
                        else:
                            between_init, inner_last = between_unsnoc
                            if len(inner_last) == 3:
                                _, b, c = inner_last
                                return (
                                    SeqDeep(size - 1, front, between_init, (b, c)),
                                    a,
                                )
                            elif len(inner_last) == 4:
                                _, b, c, d = inner_last
                                return (
                                    SeqDeep(size - 1, front, between_init, (b, c, d)),
                                    a,
                                )
                            else:
                                raise Impossible
                case (a, b):
                    return (SeqDeep(size - 1, front, between, (a,)), b)
                case (a, b, c):
                    return (SeqDeep(size - 1, front, between, (a, b)), c)
                case (a, b, c, d):
                    return (SeqDeep(size - 1, front, between, (a, b, c)), d)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_concat[T](seq: Seq[T], other: Seq[T]) -> Seq[T]:
    match seq:
        case SeqEmpty():
            return other
        case SeqSingle(value):
            match other:
                case SeqEmpty():
                    return seq
                case SeqSingle(other_value):
                    raise Todo
                case SeqDeep(_, _, _, _):
                    raise Todo
                case _:
                    raise Impossible
        case SeqDeep(_, _, _, _):
            match other:
                case SeqEmpty():
                    return seq
                case SeqSingle(other_value):
                    raise Todo
                case SeqDeep(_, _, _, _):
                    raise Todo
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_to_list[T](seq: Seq[T]) -> List[T]:
    acc: List[T] = []
    _seq_to_list_sub(acc, seq)
    return acc


def _seq_to_list_sub[T](acc: List[T], seq: Seq[T]) -> None:
    match seq:
        case SeqEmpty():
            pass
        case SeqSingle(value):
            acc.append(value)
        case SeqDeep(_, front, between, back):
            acc.extend(front)
            raise Todo
            acc.extend(back)
        case _:
            raise Impossible
