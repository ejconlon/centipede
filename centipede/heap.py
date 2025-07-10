from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, Union

# ===== Preamble =====


class Irrelevant(NotImplementedError):
    pass


class Impossible(Exception):
    pass


class Todo(Exception):
    pass


class Comparable(Protocol):
    def __lt__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __le__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __gt__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __ge__[S](self: S, other: S) -> bool:
        raise Irrelevant


# ===== Seq =====


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

    def uncons(self) -> Optional[Tuple[T, Seq[T]]]:
        match self:
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
                            between_uncons = between.uncons()
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
                                            SeqDeep(
                                                size - 1, (b, c), Seq.empty(), (d,)
                                            ),
                                        )
                                    case (b, c, d, e):
                                        return (
                                            a,
                                            SeqDeep(
                                                size - 1, (b, c, d), Seq.empty(), (e,)
                                            ),
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
                                        SeqDeep(
                                            size - 1, (b, c, d), between_tail, back
                                        ),
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

    def cons(self, value: T) -> Seq[T]:
        match self:
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
                        new_between = between.cons(new_inner)
                        return SeqDeep(size + 1, (value, c, d), new_between, back)
                    case _:
                        raise Impossible
            case _:
                raise Impossible

    def unsnoc(self) -> Optional[Tuple[Seq[T], T]]:
        match self:
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
                            between_unsnoc = between.unsnoc()
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
                                            SeqDeep(
                                                size - 1, (b,), Seq.empty(), (c, d)
                                            ),
                                            a,
                                        )
                                    case (b, c, d, e):
                                        return (
                                            SeqDeep(
                                                size - 1, (b,), Seq.empty(), (c, d, e)
                                            ),
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
                                        SeqDeep(
                                            size - 1, front, between_init, (b, c, d)
                                        ),
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

    def snoc(self, value: T) -> Seq[T]:
        match self:
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
                        new_between = between.snoc(new_inner)
                        return SeqDeep(size + 1, front, new_between, (a, b, value))
                    case _:
                        raise Impossible
            case _:
                raise Impossible

    def concat(self, other: Seq[T]):
        match self:
            case SeqEmpty():
                return other
            case SeqSingle(value):
                match other:
                    case SeqEmpty():
                        return self
                    case SeqSingle(other_value):
                        raise Todo
                    case SeqDeep(_, _, _, _):
                        raise Todo
                    case _:
                        raise Impossible
            case SeqDeep(_, _, _, _):
                match other:
                    case SeqEmpty():
                        return self
                    case SeqSingle(other_value):
                        raise Todo
                    case SeqDeep(_, _, _, _):
                        raise Todo
                    case _:
                        raise Impossible
            case _:
                raise Impossible


@dataclass(frozen=True)
class SeqEmpty[T](Seq[T]):
    pass


_SEQ_EMPTY: Seq[Any] = SeqEmpty()


@dataclass(frozen=True)
class SeqSingle[T](Seq[T]):
    value: T


@dataclass(frozen=True)
class SeqDeep[T](Seq[T]):
    size: int
    front: OuterNode[T]
    between: Seq[InnerNode[T]]
    back: OuterNode[T]


# ===== Heap =====


@dataclass(frozen=True, eq=False)
class HeapNode[K: Comparable, V]:
    key: K
    value: V
    rank: int
    rest: Heap[K, V]


@dataclass(frozen=True, eq=False)
class Heap[K: Comparable, V]:
    """A Brodal-Okasaki persistent min-heap"""

    _unwrap: Seq[HeapNode[K, V]]

    @staticmethod
    def empty() -> Heap[K, V]:
        return _HEAP_EMPTY

    @staticmethod
    def singleton(key: K, value: V) -> Heap[K, V]:
        return Heap(Seq.singleton(HeapNode(key, value, 0, Heap.empty())))

    def null(self) -> bool:
        return bool(self._unwrap)

    def find_min(self) -> Optional[HeapNode[K, V]]:
        return _heap_find_min(self)

    def insert(self, key: K, value: V) -> Heap[K, V]:
        cand = HeapNode(key, value, 0, Heap.empty())
        return _heap_insert(cand, self)

    def meld(self, other: Heap[K, V]) -> Heap[K, V]:
        return _heap_meld(self, other)

    def delete_min(self) -> Optional[Heap[K, V]]:
        return _heap_delete_min(self)


_HEAP_EMPTY: Heap[Any, Any] = Heap(Seq.empty())


def _heap_insert[K: Comparable, V](
    cand: HeapNode[K, V], heap: Heap[K, V]
) -> Heap[K, V]:
    match heap._unwrap.uncons():
        case None:
            return Heap(Seq.singleton(cand))
        case (head, tail):
            if cand.rank < head.rank:
                return Heap(heap._unwrap.cons(cand))
            else:
                new_node = _heap_link(cand, head)
                return _heap_insert(new_node, Heap(tail))
        case _:
            raise Impossible


def _heap_link[K: Comparable, V](
    first: HeapNode[K, V], second: HeapNode[K, V]
) -> HeapNode[K, V]:
    if first.key <= second.key:
        return HeapNode(
            first.key,
            first.value,
            first.rank + 1,
            Heap(first.rest._unwrap.cons(second)),
        )
    else:
        return HeapNode(
            second.key,
            second.value,
            second.rank + 1,
            Heap(second.rest._unwrap.cons(first)),
        )


def _heap_meld[K: Comparable, V](first: Heap[K, V], second: Heap[K, V]) -> Heap[K, V]:
    match first._unwrap.uncons():
        case None:
            return second
        case (first_head, first_tail):
            match second._unwrap.uncons():
                case None:
                    return first
                case (second_head, second_tail):
                    if first_head.rank < second_head.rank:
                        tail = _heap_meld(Heap(first_tail), second)
                        return Heap(tail._unwrap.cons(first_head))
                    elif second_head.rank < first_head.rank:
                        tail = _heap_meld(first, Heap(second_tail))
                        return Heap(tail._unwrap.cons(second_head))
                    else:
                        head = _heap_link(first_head, second_head)
                        tail = _heap_meld(Heap(first_tail), Heap(second_tail))
                        return _heap_insert(head, tail)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _heap_find_min[K: Comparable, V](heap: Heap[K, V]) -> Optional[HeapNode[K, V]]:
    match heap._unwrap.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return head
            else:
                cand = _heap_find_min(Heap(tail))
                if cand is None or head.key <= cand.key:
                    rest = _heap_meld(head.rest, Heap(tail))
                    return HeapNode(head.key, head.value, 0, rest)
                else:
                    rest = _heap_meld(Heap(tail), cand.rest)
                    return HeapNode(cand.key, cand.value, 0, rest)
        case _:
            raise Impossible


def _heap_delete_min[K: Comparable, V](heap: Heap[K, V]) -> Optional[Heap[K, V]]:
    match heap._unwrap.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return Heap.empty()
            else:
                cand = _heap_find_min(Heap(tail))
                if cand is None or head.key <= cand.key:
                    return _heap_meld(head.rest, Heap(tail))
                else:
                    return _heap_meld(Heap(tail), cand.rest)
        case _:
            raise Impossible
