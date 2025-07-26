from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

from centipede.spiny.common import Impossible

__all__ = ["Seq"]


type OuterNode[T] = Union[Tuple[T], Tuple[T, T], Tuple[T, T, T], Tuple[T, T, T, T]]
type InnerNode[T] = Union[Tuple[T, T], Tuple[T, T, T]]


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
        return _seq_cons(value, self)

    def unsnoc(self) -> Optional[Tuple[Seq[T], T]]:
        return _seq_unsnoc(self)

    def snoc(self, value: T) -> Seq[T]:
        return _seq_snoc(self, value)

    def concat(self, other: Seq[T]) -> Seq[T]:
        return _seq_concat(self, other)

    def lookup(self, ix: int) -> Optional[T]:
        return _seq_lookup(self, ix)

    def update(self, ix: int, value: T) -> Optional[Seq[T]]:
        return _seq_update(self, ix, value)

    def to_iter(self) -> Generator[T]:
        return _seq_to_iter(self)

    def to_list(self) -> List[T]:
        return list(self.to_iter())


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
                            return (
                                a,
                                SeqDeep(size - 1, inner_head, between_tail, back),
                            )
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


def _seq_cons[T](value: T, seq: Seq[T]) -> Seq[T]:
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
                    new_inner = (a, b)
                    new_between = _seq_cons(new_inner, between)
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
                    new_inner = (c, d)
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
                            return (
                                SeqDeep(size - 1, front, between_init, inner_last),
                                a,
                            )
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
                    return SeqDeep(2, (value,), Seq.empty(), (other_value,))
                case SeqDeep(_, _, _, _):
                    return _seq_cons(value, other)
                case _:
                    raise Impossible
        case SeqDeep(_, _, _, _):
            match other:
                case SeqEmpty():
                    return seq
                case SeqSingle(other_value):
                    return _seq_snoc(seq, other_value)
                case SeqDeep(_, _, _, _):
                    return _seq_concat_deep(seq, other)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_concat_deep[T](left: Seq[T], right: Seq[T]) -> Seq[T]:
    match (left, right):
        case (
            SeqDeep(left_size, left_front, left_between, left_back),
            SeqDeep(right_size, right_front, right_between, right_back),
        ):
            middle_nodes = _nodes_from_touching_ends(left_back, right_front)
            new_between = _seq_concat_middle(left_between, middle_nodes, right_between)
            return SeqDeep(left_size + right_size, left_front, new_between, right_back)
        case _:
            raise Impossible


def _nodes_from_touching_ends[T](
    left_back: OuterNode[T], right_front: OuterNode[T]
) -> List[InnerNode[T]]:
    combined = list(left_back)
    combined.extend(right_front)
    nodes: List[InnerNode[T]] = []

    i = 0
    while i < len(combined):
        remaining = len(combined) - i
        if remaining == 1:
            if nodes and len(nodes[-1]) == 3:
                prev_node = nodes.pop()
                a, b = prev_node  # type: ignore
                nodes.append((a, b, combined[i]))
            else:
                raise Impossible
            i += 1
        elif remaining == 2:
            nodes.append((combined[i], combined[i + 1]))
            i += 2
        elif remaining == 3:
            nodes.append((combined[i], combined[i + 1], combined[i + 2]))
            i += 3
        elif remaining == 4:
            nodes.append((combined[i], combined[i + 1]))
            nodes.append((combined[i + 2], combined[i + 3]))
            i += 4
        else:
            nodes.append((combined[i], combined[i + 1], combined[i + 2]))
            i += 3

    return nodes


def _seq_concat_middle[T](
    left_between: Seq[InnerNode[T]],
    middle_nodes: List[InnerNode[T]],
    right_between: Seq[InnerNode[T]],
) -> Seq[InnerNode[T]]:
    result = left_between
    for node in middle_nodes:
        result = _seq_snoc(result, node)
    return _seq_concat(result, right_between)


def _seq_lookup[T](seq: Seq[T], ix: int) -> Optional[T]:
    if ix < 0:
        return None
    match seq:
        case SeqEmpty():
            return None
        case SeqSingle(value):
            return value if ix == 0 else None
        case SeqDeep(size, front, between, back):
            if ix >= size:
                return None
            front_size = len(front)
            if ix < front_size:
                return front[ix]
            ix -= front_size
            back_size = len(back)
            between_total_size = size - front_size - back_size
            if ix < between_total_size:
                return _seq_lookup_between(between, ix)
            ix -= between_total_size
            if ix < back_size:
                return back[ix]
            return None
        case _:
            raise Impossible


def _seq_lookup_between[T](between: Seq[InnerNode[T]], ix: int) -> Optional[T]:
    current_offset = 0
    for inner_node in _seq_to_iter(between):
        node_size = len(inner_node)
        if ix < current_offset + node_size:
            node_ix = ix - current_offset
            if node_ix < len(inner_node):
                return inner_node[node_ix]
        current_offset += node_size
    return None


def _seq_update[T](_seq: Seq[T], _ix: int, _value: T) -> Optional[Seq[T]]:
    raise Exception("TODO")


def _seq_to_iter[T](seq: Seq[T]) -> Generator[T]:
    match seq:
        case SeqEmpty():
            pass
        case SeqSingle(value):
            yield value
        case SeqDeep(_, front, between, back):
            yield from front
            for inner_node in _seq_to_iter(between):
                yield from inner_node
            yield from back
        case _:
            raise Impossible
